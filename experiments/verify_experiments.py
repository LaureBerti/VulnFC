#!/usr/bin/env python3
"""Static verification of the experiment harness.

Every failure this project hit in August 2026 was silent: a stage reported "done" while doing
nothing, or wrote rows with empty columns, or hung forever. None of them raised. This script
checks, without running anything, the specific contracts whose violation produced those failures.

  [A] CLI contract      — every flag a runner passes exists in the target's argparse, and every
                          literal value satisfies that argument's `choices`.
                          (C7 passed --model_name bert; choices are bert-base-uncased/roberta-base.)
  [B] Checkpoints       — every checkpoint directory a runner references exists on disk.
                          (C7 pointed at checkpoints/bert_tabfact, which never existed.)
  [C] Resume guards     — every column a guard keys on exists in the target's output schema.
                          (Guards keyed on seed= against CSVs that had no seed column, so no
                          condition was ever skipped and half the C9 grid was never reached.)
  [D] Schema honoured   — every field declared in RESULTS_FIELDS is actually written.
                          (C1 declared ir/n_target_correct/... and wrote none of them.)
  [E] Network timeouts  — every OpenAI client sets an explicit timeout.
                          (A stalled socket blocked one condition for 13 hours.)
  [F] Silent failure    — runners must check exit status rather than piping into a filter that
                          can swallow the error. (grep -E "Error" missed argparse's "error:".)
  [G] Synthetic data    — the loader must fail closed rather than substituting placeholder rows.

Exit status is 1 if any BLOCKING check fails, so this can gate a run.

    python experiments/verify_experiments.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP, SCR = ROOT / "experiments", ROOT / "scripts"

BLOCK: list[str] = []
WARN: list[str] = []


def parse_py(path: Path) -> dict:
    """Pull argparse spec, output schema, written keys and client kwargs out of a script."""
    tree = ast.parse(path.read_text())
    info = {"args": {}, "fields": {}, "written": {}, "clients": []}

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("--")):
            flag = node.args[0].value
            choices = None
            for kw in node.keywords:
                if kw.arg == "choices":
                    try:
                        choices = [c.value for c in kw.value.elts]
                    except AttributeError:
                        choices = None
            info["args"][flag] = choices

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id.endswith("_FIELDS") and isinstance(node.value, ast.List):
                try:
                    info["fields"][t.id] = [e.value for e in node.value.elts]
                except AttributeError:
                    pass

        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "append_csv" and len(node.args) >= 3):
            schema = node.args[2].id if isinstance(node.args[2], ast.Name) else None
            keys = None
            if isinstance(node.args[1], ast.Dict):
                keys = [k.value for k in node.args[1].keys if isinstance(k, ast.Constant)]
            elif isinstance(node.args[1], ast.Name):
                target = node.args[1].id
                for n2 in ast.walk(tree):
                    if (isinstance(n2, ast.Assign) and len(n2.targets) == 1
                            and isinstance(n2.targets[0], ast.Name)
                            and n2.targets[0].id == target and isinstance(n2.value, ast.Dict)):
                        keys = [k.value for k in n2.value.keys if isinstance(k, ast.Constant)]
            if schema and keys is not None:
                info["written"].setdefault(schema, set()).update(keys)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "OpenAI":
            info["clients"].append({kw.arg for kw in node.keywords if kw.arg})

    return info


INVOKE = re.compile(r"experiments/(\w+\.py)((?:\s*\\?\s*(?:--[\w-]+|\"[^\"]*\"|'[^']*'|\$\w+|[^\s\\])+)*)")


def invocations(text: str):
    """Yield (script, {flag: [literal values]}) for each experiments/*.py call in a shell script."""
    for m in INVOKE.finditer(text):
        script, blob = m.group(1), m.group(2).replace("\\\n", " ")
        blob = blob.split("\n\n")[0]
        toks = blob.split()
        flags, cur = {}, None
        for t in toks:
            if t.startswith("--"):
                cur = t
                flags.setdefault(cur, [])
            elif cur is not None:
                flags[cur].append(t.strip("\"'"))
        yield script, flags


def expand(vals, text):
    """Resolve a shell literal to the concrete values a loop can produce, else None."""
    out = []
    for v in vals:
        m = re.fullmatch(r"\$(\w+)|\$\{(\w+)\}", v)
        if m:
            var = m.group(1) or m.group(2)
            loop = re.search(rf"for\s+{var}\s+in\s+([^;\n]+)", text)
            if loop:
                items = loop.group(1).split()
                if all(not i.startswith("$") for i in items):
                    out.extend(items); continue
            return None
        if "$" in v:
            return None
        out.append(v)
    return out


def main() -> int:
    pys = {p.name: parse_py(p) for p in sorted(EXP.glob("eval_*.py")) if "_pre" not in p.name}
    shs = {p.name: p.read_text() for p in sorted(SCR.glob("*.sh"))
           if not re.search(r"_pre(seed|timeout)_", p.name)}

    print("=" * 82)
    print("EXPERIMENT HARNESS VERIFICATION")
    print("=" * 82)
    print(f"eval scripts: {len(pys)}   runner scripts: {len(shs)}\n")

    print("[A] CLI contract (flag exists, value satisfies choices)")
    print("[B] checkpoint directories referenced by runners")
    seen_pairs = set()
    for sh, text in shs.items():
        for script, flags in invocations(text):
            if script not in pys:
                continue
            spec = pys[script]["args"]
            for flag, vals in flags.items():
                if flag not in spec:
                    BLOCK.append(f"[A] {sh} passes {flag} to {script}, which has no such argument")
                    continue
                choices = spec[flag]
                conc = expand(vals, text)
                if choices and conc:
                    for v in conc:
                        if v not in choices:
                            BLOCK.append(f"[A] {sh}: {script} {flag}={v!r} not in choices {choices}")
            for flag, vals in flags.items():
                if "checkpoint" in flag or "ckpt" in flag:
                    for v in expand(vals, text) or []:
                        if v.startswith("checkpoints/") and (ROOT / v) not in seen_pairs:
                            seen_pairs.add(ROOT / v)
                            if not (ROOT / v).is_dir():
                                BLOCK.append(f"[B] {sh}: checkpoint {v} does not exist")
    for sh, text in shs.items():
        for m in re.finditer(r'checkpoints/\$\{?(\w+)\}?_(\w+)', text):
            var, suffix = m.group(1), m.group(2)
            loop = re.search(rf"for\s+{var}\s+in\s+([^;\n]+)", text)
            if not loop:
                continue
            for item in loop.group(1).split():
                if item.startswith("$"):
                    continue
                d = ROOT / "checkpoints" / f"{item}_{suffix}"
                if not d.is_dir():
                    BLOCK.append(f"[B] {sh}: interpolated checkpoint {d.relative_to(ROOT)} does not exist")
    print(f"    -> {sum(1 for b in BLOCK if b.startswith('[A]'))} CLI, "
          f"{sum(1 for b in BLOCK if b.startswith('[B]'))} checkpoint problems\n")

    print("[C] resume guards key on columns that exist in the target schema")
    csv_to_script = {"c1_v2": "eval_llm_victim.py",
                     "c9_v2": "eval_attack_comparison.py"}
    for sh, text in shs.items():
        for m in re.finditer(r'(cond_done|done_key|cell_done)[ \t]+(\S+)((?:[^\n\\]|\\\n)*)', text):
            fn, csvpath, rest = m.groups()
            tag = next((k for k in csv_to_script if k in csvpath), None)
            if not tag:
                continue
            spec = pys.get(csv_to_script[tag], {}).get("fields", {}).get("RESULTS_FIELDS", [])
            if not spec:
                continue
            for col in re.findall(r'"(\w+)=', rest):
                if col not in spec:
                    BLOCK.append(f"[C] {sh}: guard keys on {col!r} but {csv_to_script[tag]} "
                                 f"never writes that column (schema: {', '.join(spec[:6])}...)")
            if fn == "done_key":
                WARN.append(f"[C] {sh} still uses done_key(); the count-based cond_done() is the "
                            f"one that also covers legacy rows written before `seed` existed")
    print(f"    -> {sum(1 for b in BLOCK if b.startswith('[C]'))} guard problems\n")

    print("[D] every declared output field is actually written")
    for name, info in pys.items():
        for schema, declared in info["fields"].items():
            written = info["written"].get(schema)
            if written is None:
                continue
            missing = [f for f in declared if f not in written]
            if missing:
                BLOCK.append(f"[D] {name}: {schema} declares {missing} but the row never sets them "
                             f"-> those columns come out blank in every row")
    print(f"    -> {sum(1 for b in BLOCK if b.startswith('[D]'))} schema problems\n")

    print("[E] OpenAI clients set an explicit timeout")
    for p in sorted(EXP.glob("*.py")):
        if "_pre" in p.name:
            continue
        try:
            info = parse_py(p)
        except SyntaxError:
            continue
        for kwargs in info["clients"]:
            if "timeout" not in kwargs:
                BLOCK.append(f"[E] {p.name}: OpenAI(...) built without timeout= — a stalled "
                             f"socket blocks forever and the retry loop never fires")
    print(f"    -> {sum(1 for b in BLOCK if b.startswith('[E]'))} clients without timeout\n")

    print("[F] runners surface failures instead of filtering them away")
    for sh, text in shs.items():
        for m in re.finditer(r'\|\s*grep\s+(-\w+\s+)*"?([^"|\n]*)"?\s*\|?', text):
            opts = m.group(1) or ""
            pat = m.group(2)
            if "Error" in pat and "i" not in opts:
                WARN.append(f"[F] {sh}: filters on \"Error\" case-sensitively; argparse and many "
                            f"libraries print lowercase \"error:\", so failures vanish")
        for m in re.finditer(r'^\s*(?:\$N\s+)?"?\$PY"?\s+experiments/(\w+\.py)', text, re.M):
            seg = text[m.start(): m.start() + 700]
            if "| grep" in seg and not re.search(r'\bif\s+!?\s*(nice|\$N|"?\$PY)', seg):
                WARN.append(f"[F] {sh}: {m.group(1)} is piped into a filter with no exit-status "
                            f"check — a stage can report success while every condition failed")
    print(f"    -> {sum(1 for w in WARN if w.startswith('[F]'))} silent-failure risks\n")

    print("[H] API failures are not imputed with the true label")
    for p in sorted(EXP.glob("*.py")):
        if "_pre" in p.name:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"is not None else\s+(rec|item|r)\[[\"']label[\"']\]", line) or \
               re.search(r"pred\w*\s*=\s*(gt_label|item\[[\"']label[\"']\])", line):
                BLOCK.append(f"[H] {p.name}:{i} imputes the ground-truth label when a prediction "
                             f"is unavailable — an API failure then scores as a correct "
                             f"prediction and as a non-flip, biasing VS downward")
    print(f"    -> {sum(1 for b in BLOCK if b.startswith('[H]'))} imputation problems\n")

    print("[I] LLM generation with temperature>0 passes a seed to the API")
    ag = EXP / "attack_generators.py"
    if ag.exists():
        src = ag.read_text()
        for m in re.finditer(r"temperature\s*=\s*([0-9.]+)", src):
            temp = float(m.group(1))
            if temp <= 0:
                continue
            start = src.rfind("completions.create(", 0, m.start())
            end = src.find(")", m.end())
            window = src[start:end] if start != -1 else ""
            if '"seed"' not in window and "seed=" not in window:
                line = src[:m.start()].count("\n") + 1
                BLOCK.append(f"[I] attack_generators.py:{line} generates at temperature={temp} "
                             f"with no seed — --seed does not control the LLM, so re-running the "
                             f"same command produces different attacks and different VS")
    print(f"    -> {sum(1 for b in BLOCK if b.startswith('[I]'))} unseeded generators\n")

    print("[G] data loader fails closed on missing real data")
    dl = EXP / "data_loaders.py"
    if dl.exists():
        t = dl.read_text()
        if re.search(r"def load_dataset_split\([^)]*synthetic:\s*bool\s*=\s*True", t, re.S):
            BLOCK.append("[G] data_loaders.py: load_dataset_split defaults synthetic=True")
        elif re.search(r"synthetic:\s*bool\s*=\s*False", t):
            print("    -> synthetic=False default, explicit opt-in required  OK")
        if "VULNFC_ALLOW_SYNTHETIC" not in t:
            WARN.append("[G] data_loaders.py: no explicit env opt-in guarding synthetic data")
    print()

    print("=" * 82)
    if BLOCK:
        print(f"BLOCKING ({len(BLOCK)})")
        for b in BLOCK:
            print(f"  x {b}")
    else:
        print("BLOCKING: none")
    if WARN:
        print(f"\nWARNING ({len(WARN)})")
        for w in dict.fromkeys(WARN):
            print(f"  ! {w}")
    print("=" * 82)
    return 1 if BLOCK else 0


if __name__ == "__main__":
    sys.exit(main())
