#!/usr/bin/env python3
"""Fail if an attack definition exists in more than one place.

`attack_generators.py` holds the canonical `_INJECTION_SUFFIXES` -- the five prompt-injection
templates the paper reports -- and the word substituted into `{target}`. Every evaluator must
import them. A private copy in a caller is a reportable defect even when it is never exercised:
a user passing `--attack_type prompt_injection` would then get an attack the paper does not
describe, and the paper's description of its own instrument would be wrong.

This check scans every module under experiments/ and exits non-zero if a second definition
appears anywhere.

The check is textual on purpose: importing every evaluator to inspect it would pull in torch and
the model stack, which is slow and can fail for reasons unrelated to this question.

Usage:
    python experiments/check_single_source_attacks.py          # exits 1 on any duplicate
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

CANONICAL = {
    "_INJECTION_SUFFIXES": "attack_generators.py",
    "_INJECTION_LABEL_WORDS": "attack_generators.py",
}

DEF_RE = {n: re.compile(rf"^{re.escape(n)}\s*(?::[^=]+)?=", re.M) for n in CANONICAL}


def main() -> int:
    problems = []
    for name, owner in CANONICAL.items():
        definers = sorted(
            p.name for p in EXP.glob("*.py")
            if DEF_RE[name].search(p.read_text(encoding="utf-8", errors="replace"))
        )
        live = [d for d in definers if not re.search(r"_prev_\d|_backup|_old", d)]
        frozen = [d for d in definers if d not in live]

        print(f"{name}")
        print(f"  canonical : {owner}")
        print(f"  defined in: {', '.join(live) if live else '(nowhere!)'}")
        if frozen:
            print(f"  ignored   : {', '.join(frozen)}  (frozen snapshots)")

        if owner not in live:
            problems.append(f"{name}: canonical definition missing from {owner}")
        extra = [d for d in live if d != owner]
        if extra:
            problems.append(
                f"{name}: defined in {', '.join(extra)} as well as {owner} — "
                f"a second copy will silently diverge from the set the paper reports")
        print()

    if problems:
        print("=" * 76)
        print(f"FAIL — {len(problems)} duplicate attack definition(s):")
        for p in problems:
            print(f"  {p}")
        print("\nImport from attack_generators instead. If you genuinely need a variant, give it\n"
              "its own name there and select it explicitly.")
        return 1

    print("=" * 76)
    print(f"OK — each of {len(CANONICAL)} attack definition(s) has exactly one live definition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
