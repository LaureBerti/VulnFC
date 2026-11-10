#!/usr/bin/env python3
"""Label semantics of each attack, and the two metrics that follow from them.

An attack either preserves the truth value of the claim it rewrites or changes it, and the two
cases need different metrics because they ask different questions:

  * A LABEL-PRESERVING attack asks *can the victim be made to get a claim wrong without changing
    what the claim says?*  The gold label of the rewritten claim is the gold label of the original,
    so the rewritten claim is scored against it. The rate is the Vulnerability Score (VS).

  * A LABEL-CHANGING attack asks *can the victim catch a claim that has been falsified?*  The gold
    label of the rewritten claim is the opposite of the original's, so scoring it against the
    original's label inverts the measurement: a victim that correctly judges the falsified claim
    registers as compromised. The rate is the Detection Failure Rate (DFR), scored against the
    rewritten claim's own label.

The two are rates over the same population -- targets the victim judged correctly before the attack
-- so they are directly comparable, and they are never pooled into one column. This is the same
discipline Table II applies to VS and IR: a rate is only interpretable alongside the population it
is a rate over.

Every attack must be declared here. `metric_for()` raises on an unknown attack rather than guessing,
and `score()` refuses to compute a metric that does not match the attack's declared semantics, so a
caller cannot silently score a label-changing rewrite against the original label.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence


LABEL_PRESERVING: frozenset[str] = frozenset({
    "semantic_paraphrase",
    "prompt_injection",
    "fgsm",
    "pgd",
})

LABEL_CHANGING: frozenset[str] = frozenset({
    "llm_generated",
    "semantic_negation",
    "pos_negation",
    "content_perturbation",
    "sybil",
})

VS = "vs"
DFR = "dfr"


class UndeclaredAttack(KeyError):
    """Raised for an attack whose label semantics are not declared in this module."""


class MetricMismatch(ValueError):
    """Raised when a caller asks for a metric that does not match an attack's semantics."""


def metric_for(attack_type: str) -> str:
    """Return the metric name an attack must be scored with.

    Raises UndeclaredAttack for anything not listed above. A new attack is added by declaring its
    semantics here; until then it cannot be scored, which is the intended behaviour.
    """
    if attack_type in LABEL_PRESERVING:
        return VS
    if attack_type in LABEL_CHANGING:
        return DFR
    raise UndeclaredAttack(
        f"attack type {attack_type!r} has no declared label semantics. Add it to "
        f"LABEL_PRESERVING or LABEL_CHANGING in experiments/attack_semantics.py."
    )


def preserves_label(attack_type: str) -> bool:
    """True when the rewritten claim keeps the original claim's gold label."""
    return metric_for(attack_type) == VS


def gold_label_after(attack_type: str, original_label: int, n_classes: int = 2) -> int:
    """The gold label of the rewritten claim.

    For a label-preserving attack this is the original label. For a label-changing attack on a
    binary task it is the complement. Multi-class inversion is not defined, so it is refused rather
    than approximated.
    """
    if preserves_label(attack_type):
        return original_label
    if n_classes != 2:
        raise MetricMismatch(
            f"{attack_type!r} changes the label, which is only defined for a binary task; "
            f"got n_classes={n_classes}."
        )
    return 1 - original_label


def score(attack_type: str,
          pred_clean: Sequence[Optional[int]],
          pred_adv: Sequence[Optional[int]],
          gold: Sequence[int],
          n_classes: int = 2) -> dict:
    """Score one condition, returning the metric its attack's semantics requires.

    Both predictions must be aligned with `gold`, and a prediction of None means the victim returned
    nothing parseable. A target enters the denominator only when both of its predictions are
    available and the clean one was correct; the counts needed to audit that denominator are
    returned alongside the rate rather than folded into it.

    Returns a dict carrying the metric name, its value, the denominator, and the availability
    counts split by pass -- the split matters because an adversarial-pass failure can be an outcome
    of the attack while a clean-pass failure cannot.
    """
    if not (len(pred_clean) == len(pred_adv) == len(gold)):
        raise ValueError("pred_clean, pred_adv and gold must be the same length")

    metric = metric_for(attack_type)

    n_unavail_clean = sum(1 for p in pred_clean if p is None)
    n_unavail_adv = sum(1 for i, p in enumerate(pred_adv)
                        if p is None and pred_clean[i] is not None)

    hits = 0
    scored_gold: list[int] = []
    for pc, pa, g in zip(pred_clean, pred_adv, gold):
        if pc is None or pa is None:
            continue
        if pc != g:
            continue
        scored_gold.append(g)
        gold_adv = gold_label_after(attack_type, g, n_classes)
        if pa != gold_adv:
            hits += 1

    denom = len(scored_gold)
    return {
        "metric": metric,
        "value": round(hits / denom, 4) if denom else None,
        "n_denominator": denom,
        "n_hits": hits,
        "baseline": majority_baseline(attack_type, scored_gold, n_classes),
        "n_unavail_clean": n_unavail_clean,
        "n_unavail_adv": n_unavail_adv,
        "label_semantics": "preserving" if metric == VS else "changing",
    }


def majority_baseline(attack_type: str,
                      gold: Iterable[int],
                      n_classes: int = 2) -> Optional[float]:
    """The score a constant majority-class predictor would obtain on this condition.

    A constant predictor cannot be flipped, so its VS is 0; on a label-changing attack it answers
    the complement correctly whenever the original label was the minority class, so its DFR is
    neither 0 nor meaningless. Reporting this alongside DFR keeps a degenerate victim from reading
    as a strong detector, and complements the pre-attack learnability gate rather than replacing it.
    """
    g = list(gold)
    if not g:
        return None
    if preserves_label(attack_type):
        return 0.0
    if n_classes != 2:
        return None
    majority = max(set(g), key=g.count)
    missed = sum(1 for x in g if gold_label_after(attack_type, x, n_classes) != majority)
    return round(missed / len(g), 4)
