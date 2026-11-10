#!/usr/bin/env python3
"""Bidirectional-entailment check for label-preserving attacks.

A label-preserving attack is only label-preserving if something checks. Section IV-C defines the
semantic paraphrase by sem(c', c) ~ 1, measured by Natural Language Inference; this module is that
measurement, applied as a hard gate at generation time so a rewrite that changes the claim's
meaning never reaches the victim.

Two sentences are treated as equivalent when each entails the other. Bidirectional entailment is
the operative test rather than a single direction, because one-way entailment admits a strictly
weaker or stronger restatement -- "cases rose in March" entails "cases rose", which is not the same
claim and need not share its truth value.

EMBEDDING SIMILARITY IS NOT AN ACCEPTABLE SUBSTITUTE and this module does not offer it as a
fallback. Sentence embeddings place a sentence and its negation very close together: on the
cached all-MiniLM-L6-v2, "the vaccine is safe" and "the vaccine is not safe" score around 0.9
cosine. A similarity threshold therefore passes precisely the polarity flip this gate exists to
catch. When no NLI model is available the gate raises rather than degrading, because a guard that
silently stops guarding leaves the measurement in the state it was meant to prevent.

Setup (one-time, needs network; see README "Equivalence guard"):

    python -c "from transformers import pipeline; \\
               pipeline('text-classification', model='cross-encoder/nli-deberta-v3-small')"
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

DEFAULT_NLI_MODEL = os.environ.get("VULNFC_NLI_MODEL",
                                   "cross-encoder/nli-deberta-v3-small")

DEFAULT_ENTAILMENT_MIN = 0.70


class GuardUnavailable(RuntimeError):
    """Raised when the NLI model cannot be loaded, so equivalence cannot be established."""


@dataclass
class EquivalenceVerdict:
    equivalent: bool
    p_forward: float
    p_backward: float
    threshold: float

    def as_record(self) -> dict:
        return {"equivalent": self.equivalent,
                "p_entail_forward": round(self.p_forward, 4),
                "p_entail_backward": round(self.p_backward, 4),
                "entailment_min": self.threshold}


class EquivalenceGuard:
    """Admits a rewrite only when it entails the original and the original entails it."""

    def __init__(self,
                 model_name: str = DEFAULT_NLI_MODEL,
                 threshold: float = DEFAULT_ENTAILMENT_MIN,
                 device: Optional[str] = None):
        self.model_name = model_name
        self.threshold = threshold
        try:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=model_name,
                                  top_k=None, device=device)
        except Exception as exc:
            raise GuardUnavailable(
                f"could not load NLI model {model_name!r}: {exc}. Label-preserving attacks cannot "
                f"be generated without it. See the README section 'Equivalence guard' for the "
                f"one-time download, or set VULNFC_NLI_MODEL to a locally available NLI model."
            ) from exc

    def _p_entail(self, premise: str, hypothesis: str) -> float:
        scores = self._pipe({"text": premise, "text_pair": hypothesis})
        if isinstance(scores, list) and scores and isinstance(scores[0], list):
            scores = scores[0]
        for s in scores:
            if str(s["label"]).lower().startswith("entail"):
                return float(s["score"])
        return 0.0

    def check(self, original: str, rewritten: str) -> EquivalenceVerdict:
        fwd = self._p_entail(original, rewritten)
        bwd = self._p_entail(rewritten, original)
        return EquivalenceVerdict(
            equivalent=(fwd >= self.threshold and bwd >= self.threshold),
            p_forward=fwd, p_backward=bwd, threshold=self.threshold)

    def filter(self, original: str, candidates: Sequence[str]) -> tuple[Optional[str], list[dict]]:
        """Return the first candidate that passes, with the verdict record for every candidate.

        Every candidate's verdict is returned whether it passed or not, so the rejection rate of
        the generator is a reportable quantity rather than an invisible one.
        """
        records = []
        chosen = None
        for cand in candidates:
            v = self.check(original, cand)
            rec = v.as_record()
            rec["candidate"] = cand
            records.append(rec)
            if v.equivalent and chosen is None:
                chosen = cand
        return chosen, records
