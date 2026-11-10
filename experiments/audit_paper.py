#!/usr/bin/env python3
"""
audit_paper.py — mechanical consistency audit of the submitted LaTeX paper.

Checks that need no judgement, so they can be re-run after every edit:

  A. Metric arithmetic — for any table row exposing Precision/Recall/F1, verify
     F1 == 2PR/(P+R), the identity a reader can check by hand.
  B. Table -> CSV provenance — every numeric table cell should be traceable to a
     results CSV. Report cells that appear in NO csv.
  C. references.bib hygiene — duplicate keys, duplicate titles, venue strings sitting
     in the title field, entries with neither DOI nor URL.
  D. Cross-reference integrity — \\ref/\\cite targets that do not resolve.

Read-only: never modifies the paper.

Usage:
    python audit_paper.py --tex paper-short.tex
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tex", default="paper-short.tex")
    p.add_argument("--bib", default="references.bib")
    p.add_argument("--csv_root", default="outputs")
    p.add_argument("--tol", type=float, default=0.006,
                   help="Absolute tolerance for F1 arithmetic (rounding at 3dp)")
    return p.parse_args()



def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def check_metric_arithmetic(tex: str, tol: float) -> list:
    """Find tabular blocks whose header names Precision/Recall/F1 and verify each row."""
    issues = []
    for m in re.finditer(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", tex, re.S):
        block = m.group(1)
        line_off = tex[:m.start()].count("\n") + 1

        P_PAT, R_PAT, F_PAT = r"prec(ision|\.)?$", r"rec(all|\.)?$", r"f1|f-score"
        header = None
        for ln in block.split("\\\\"):
            plain = re.sub(r"\\[a-zA-Z]+|\{|\}|\$", "", ln)
            cs = [c.strip() for c in plain.split("&")]
            if any(re.match(P_PAT, c, re.I) for c in cs) and \
               any(re.match(R_PAT, c, re.I) for c in cs):
                header = ln
                break
        if not header:
            continue

        cols = [c.strip() for c in re.sub(r"\\[a-zA-Z]+|\{|\}|\$", "", header).split("&")]
        def idx(pat):
            for i, c in enumerate(cols):
                if re.match(pat, c, re.I):
                    return i
            return None
        i_p, i_r, i_f = idx(P_PAT), idx(R_PAT), idx(F_PAT)
        if None in (i_p, i_r, i_f):
            continue

        for ln in block.split("\\\\"):
            if ln is header or re.search(r"prec", ln, re.I):
                continue
            cells = [c.strip() for c in ln.split("&")]
            if len(cells) <= max(i_p, i_r, i_f):
                continue
            def num(c):
                mm = re.search(r"(\d+\.\d+)", c)
                return float(mm.group(1)) if mm else None
            P, R, F = num(cells[i_p]), num(cells[i_r]), num(cells[i_f])
            if None in (P, R, F):
                continue
            exp = f1(P, R)
            if abs(exp - F) > tol:
                label = re.sub(r"\\[a-zA-Z]+|[{}\\]", "", cells[min(1, len(cells) - 1)]).strip()
                alt = f1(P, 1.0)
                hint = ""
                if abs(alt - F) <= tol:
                    hint = f"  (consistent with Recall=1.000, not {R} -> recall column corrupted)"
                issues.append(
                    f"line ~{line_off}: row '{label}': P={P} R={R} F1={F} "
                    f"but 2PR/(P+R)={exp:.4f}{hint}"
                )
    return issues



def collect_csv_numbers(csv_root: Path) -> set:
    vals = set()
    for f in csv_root.rglob("*.csv"):
        try:
            for line in f.read_text(errors="ignore").splitlines()[1:]:
                for tok in line.split(","):
                    tok = tok.strip()
                    if re.fullmatch(r"\d+\.\d+", tok):
                        vals.add(round(float(tok), 3))
        except Exception:
            continue
    return vals


def check_provenance(tex: str, csv_vals: set) -> list:
    """Report table decimals that appear in no results CSV."""
    orphans = defaultdict(list)
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S):
        block = m.group(1)
        lbl = re.search(r"\\label\{(tab:[^}]+)\}", block)
        name = lbl.group(1) if lbl else f"table@{tex[:m.start()].count(chr(10))+1}"
        body = re.sub(r"\\caption\{.*?\n", "", block, flags=re.S)
        for tok in re.findall(r"\b(\d\.\d{3})\b", body):
            v = round(float(tok), 3)
            if v in (0.0, 1.0):
                continue
            if v not in csv_vals:
                orphans[name].append(tok)
    out = []
    for name, toks in orphans.items():
        uniq = sorted(set(toks))
        out.append(f"{name}: {len(uniq)} value(s) not found in any results CSV: "
                   f"{', '.join(uniq[:12])}{' …' if len(uniq) > 12 else ''}")
    return out



VENUE_TITLE = re.compile(
    r"^\s*(IPSN|KDD|ICDM|WWW|SIGMOD|VLDB|NIPS|NeurIPS|ICML|ACL|EMNLP|CIKM|AAAI|IJCAI|"
    r"Proceedings of the)\b", re.I)


def check_bib(bib_text: str) -> dict:
    ents = re.findall(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", bib_text, re.S)
    keys = defaultdict(int)
    titles = defaultdict(list)
    no_id, venue_titles = [], []
    for typ, key, body in ents:
        key = key.strip()
        keys[key] += 1
        t = re.search(r"(?:^|[,{\s])title\s*=\s*[{\"](.+?)[}\"]\s*,?\s*\n",
                      body, re.S | re.M)
        title = re.sub(r"[{}\\]|\s+", " ", t.group(1)).strip() if t else ""
        if title:
            titles[title.lower()].append(key)
            if VENUE_TITLE.match(title):
                venue_titles.append((key, title[:70]))
        if "doi" not in body.lower() and "url" not in body.lower():
            no_id.append(key)
    return {
        "total": len(ents),
        "dup_keys": {k: v for k, v in keys.items() if v > 1},
        "dup_titles": {t: ks for t, ks in titles.items() if len(ks) > 1},
        "venue_titles": venue_titles,
        "no_doi_url": no_id,
    }



def check_refs(tex: str, bib_text: str) -> dict:
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set()
    for m in re.finditer(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", tex):
        refs.add(m.group(1))
    bibkeys = {k.strip() for k in re.findall(r"@\w+\s*\{\s*([^,]+),", bib_text)}
    cites = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        cites.update(c.strip() for c in m.group(1).split(","))
    return {
        "dangling_refs":  sorted(refs - labels),
        "unused_labels":  sorted(labels - refs),
        "missing_cites":  sorted(cites - bibkeys),
        "uncited_bib":    sorted(bibkeys - cites),
    }


def main():
    args = parse_args()
    tex_path, bib_path = ROOT / args.tex, ROOT / args.bib
    tex = tex_path.read_text(errors="ignore")
    bib = bib_path.read_text(errors="ignore")

    def expand(text, depth=0):
        if depth > 3:
            return text
        def sub(m):
            p = ROOT / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
            return expand(p.read_text(errors="ignore"), depth + 1) if p.exists() else m.group(0)
        return re.sub(r"\\input\{([^}]+)\}", sub, text)

    tex = expand(tex)

    print("=" * 78)
    print(f"AUDIT  {args.tex}")
    print("=" * 78)

    print("\n[A] METRIC ARITHMETIC  (F1 == 2PR/(P+R))")
    issues = check_metric_arithmetic(tex, args.tol)
    if issues:
        for i in issues:
            print(f"  ✗ {i}")
        print(f"  -> {len(issues)} impossible row(s)")
    else:
        print("  ✓ all P/R/F1 rows consistent")

    print("\n[B] TABLE -> CSV PROVENANCE")
    csv_vals = collect_csv_numbers(ROOT / args.csv_root)
    print(f"  ({len(csv_vals)} distinct 3dp values across results CSVs)")
    for line in check_provenance(tex, csv_vals):
        print(f"  ✗ {line}")

    print("\n[C] references.bib")
    b = check_bib(bib)
    print(f"  entries: {b['total']}")
    if b["dup_keys"]:
        print(f"  ✗ duplicate keys: {b['dup_keys']}")
    if b["dup_titles"]:
        print(f"  ✗ {len(b['dup_titles'])} duplicated title(s):")
        for t, ks in list(b["dup_titles"].items())[:8]:
            print(f"      {ks} -> {t[:64]}")
    if b["venue_titles"]:
        print(f"  ✗ {len(b['venue_titles'])} entry/entries with a VENUE in the title field:")
        for k, t in b["venue_titles"]:
            print(f"      {k:22} title = {t!r}")
    print(f"  ⚠ {len(b['no_doi_url'])} entries with neither DOI nor URL")

    print("\n[D] CROSS-REFERENCES")
    r = check_refs(tex, bib)
    for k in ("dangling_refs", "missing_cites"):
        if r[k]:
            print(f"  ✗ {k}: {r[k][:12]}")
    if r["uncited_bib"]:
        print(f"  ⚠ {len(r['uncited_bib'])} bib entries never cited")
    if not r["dangling_refs"] and not r["missing_cites"]:
        print("  ✓ all \\ref and \\cite targets resolve")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
