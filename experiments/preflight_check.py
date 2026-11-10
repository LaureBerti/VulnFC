#!/usr/bin/env python3
"""
preflight_check.py — Environment validation before any experiment run.

Usage:
    python3 preflight_check.py                    # base checks only
    python3 preflight_check.py rag_attack         # base + C4 deps
    python3 preflight_check.py whitebox           # base + C7 deps

Exit codes:
    0  — all checks pass
    1  — one or more checks failed (prints clear diagnostics)

Required base env: vulnfc12 (Python 3.12, torch 2.2.2, transformers 4.57.x)
Root cause of recurrent failures: transformers 5.x dropped PyTorch 2.2 support.
Fix applied 2026-04-25: pinned transformers==4.57.6 in vulnfc12.
"""

import sys
import importlib
from typing import List, Tuple

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "base"

BASE_CHECKS: List[Tuple] = [
    ("torch",            (2, 2),  "pip install torch  (or: conda install pytorch -c conda-forge)"),
    ("transformers",     (4, 57), "pip install 'transformers==4.57.6'  (4.57.x is last pre-5.x, torch-2.2-compatible)"),
    ("numpy",            (1, 24), "pip install 'numpy<2'"),
    ("sklearn",          None,    "pip install scikit-learn"),
    ("tqdm",             None,    "pip install tqdm"),
    ("datasets",         None,    "pip install datasets"),
]

PROFILE_CHECKS = {
    "rag_attack": [
        ("openai",               None, "pip install openai>=1.0"),
        ("faiss",                None, "pip install faiss-cpu"),
        ("sentence_transformers",None, "pip install sentence-transformers"),
    ],
    "whitebox": [
    ],
    "llm_victim": [
        ("openai",               None, "pip install openai>=1.0"),
    ],
    "multilingual": [
        ("datasets",             None, "pip install datasets"),
    ],
}

CRITICAL_IMPORTS = {
    "base":       [("transformers", "AutoModelForSequenceClassification"),
                   ("transformers", "AutoTokenizer")],
    "whitebox":   [("transformers", "AutoModelForSequenceClassification"),
                   ("transformers", "AutoTokenizer")],
    "rag_attack": [("transformers", "AutoModelForSequenceClassification"),
                   ("sentence_transformers", "SentenceTransformer")],
}


def _ver(mod) -> Tuple[int, ...]:
    v = getattr(mod, "__version__", "0.0.0")
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())


def _check_torch_transformers_compat():
    """Verify transformers does NOT disable PyTorch (happens when version check fails)."""
    try:
        import transformers
        from transformers import AutoModelForSequenceClassification
        from transformers.utils import is_torch_available
        if not is_torch_available():
            return False, "transformers reports PyTorch unavailable — likely version mismatch"
        return True, ""
    except Exception as e:
        return False, str(e)


def run_checks() -> bool:
    all_ok = True
    checks = BASE_CHECKS + PROFILE_CHECKS.get(PROFILE, [])
    critical = CRITICAL_IMPORTS.get(PROFILE, CRITICAL_IMPORTS.get("base", []))

    print(f"\n[preflight] Profile: {PROFILE}")
    print(f"[preflight] Python: {sys.version.split()[0]}  ({sys.executable})")
    print("─" * 60)

    for (modname, min_ver, fix_hint) in checks:
        try:
            mod = importlib.import_module(modname)
            ver = _ver(mod)
            ver_str = getattr(mod, "__version__", "?")
            if min_ver and ver < min_ver:
                req_str = ".".join(str(x) for x in min_ver)
                print(f"  ❌  {modname:<28} {ver_str}  (need >= {req_str})")
                print(f"       Fix: {fix_hint}")
                all_ok = False
            else:
                print(f"  ✅  {modname:<28} {ver_str}")
        except ImportError:
            print(f"  ❌  {modname:<28} NOT FOUND")
            print(f"       Fix: {fix_hint}")
            all_ok = False

    ok, msg = _check_torch_transformers_compat()
    if not ok:
        print(f"\n  ❌  torch/transformers compatibility FAILED")
        print(f"       {msg}")
        print(f"       Fix: pip install 'transformers==4.57.6'  (then re-run)")
        all_ok = False
    else:
        print(f"  ✅  {'torch/transformers compat':<28}")

    for (modname, classname) in critical:
        try:
            mod = importlib.import_module(modname)
            getattr(mod, classname)
            print(f"  ✅  {modname}.{classname}")
        except (ImportError, AttributeError) as e:
            print(f"  ❌  {modname}.{classname}  — {e}")
            all_ok = False

    print("─" * 60)
    if all_ok:
        print("[preflight] All checks passed — environment ready.\n")
    else:
        print("[preflight] *** ENVIRONMENT ERRORS ABOVE — fix before running experiment ***\n")
    return all_ok


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
