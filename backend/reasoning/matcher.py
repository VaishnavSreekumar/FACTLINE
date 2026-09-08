"""Deterministic Candidate Fact Matcher."""

import re
from typing import List, Optional, Set, Tuple
from backend.models.normalization import NormalizedFact
from backend.models.relationship import CandidatePair


class CandidateMatcher:
    """Finds plausible candidate fact pairs for comparability evaluation using deterministic signals.
    
    Adheres to high-precision candidate generation to avoid noisy pairings while discovering
    legitimate metric variants (e.g., 'Revenue' vs 'Revenue from operations').
    """

    # Stopwords and generic temporal/operational predicates filtered from core metric nouns
    METRIC_STOPWORDS = {
        "from", "of", "in", "to", "for", "the", "and", "a", "an", "at", "by", "on", "as",
        "since", "during", "over", "per", "all", "total", "net", "gross", "delivered", "reported",
        "based", "approx", "approximately", "inception", "overall", "full", "annual",
    }

    QUANTITY_MODIFIERS = {"count", "number", "num", "no", "qty", "quantity"}

    CURRENCY_UNITS = {"inr", "usd", "eur", "gbp"}
    PERCENT_UNITS = {"percent"}

    @classmethod
    def _singularize(cls, token: str) -> str:
        """Deterministic basic plural stripping for metric noun alignment."""
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        elif token.endswith("es") and len(token) > 4 and token[-3] in "sxyzchsh":
            return token[:-2]
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]
        return token

    @classmethod
    def _extract_metric_tokens(cls, metric: str) -> Set[str]:
        """Extracts meaningful, singularized normalized metric tokens."""
        if not metric:
            return set()
        tokens = re.findall(r"[a-z0-9]+", metric.lower())
        meaningful = {cls._singularize(t) for t in tokens if len(t) >= 3 and t not in cls.METRIC_STOPWORDS}
        return meaningful

    @classmethod
    def _are_unit_families_compatible(cls, fact_a: NormalizedFact, fact_b: NormalizedFact) -> bool:
        """Verifies deterministic compatibility between unit families (if units are present)."""
        u_a = (fact_a.normalized_value.canonical_unit or "").strip().lower()
        u_b = (fact_b.normalized_value.canonical_unit or "").strip().lower()

        if not u_a or not u_b or u_a == "count" or u_b == "count":
            return True

        # Currency vs Percent
        is_curr_a = u_a in cls.CURRENCY_UNITS or fact_a.normalized_value.currency is not None
        is_curr_b = u_b in cls.CURRENCY_UNITS or fact_b.normalized_value.currency is not None

        is_pct_a = u_a in cls.PERCENT_UNITS
        is_pct_b = u_b in cls.PERCENT_UNITS

        if is_curr_a != is_curr_b and (is_curr_a or is_curr_b) and (is_pct_a or is_pct_b):
            return False

        if (is_curr_a and is_pct_b) or (is_pct_a and is_curr_b):
            return False

        return True

    @classmethod
    def match_pair(
        cls,
        fact_a: NormalizedFact,
        fact_b: NormalizedFact,
    ) -> Optional[CandidatePair]:
        """Evaluates whether two facts form a plausible candidate pair for comparability testing.

        Args:
            fact_a: First normalized fact.
            fact_b: Second normalized fact.

        Returns:
            CandidatePair if deterministic entity, metric, and unit signals align; None otherwise.
        """
        # Exclude self-pairing
        if fact_a.fact.fact_id == fact_b.fact.fact_id:
            return None

        # 1. Entity Alignment (Requires strong entity agreement)
        entity_a = (fact_a.canonical_entity or fact_a.fact.entity or "").strip().lower()
        entity_b = (fact_b.canonical_entity or fact_b.fact.entity or "").strip().lower()

        if not entity_a or not entity_b or entity_a != entity_b:
            return None

        entity_match = True

        # 2. Unit Family Pre-check
        if not cls._are_unit_families_compatible(fact_a, fact_b):
            return None

        # 3. Metric Signals / Overlap
        metric_a = (fact_a.canonical_metric or fact_a.fact.metric or "").strip().lower()
        metric_b = (fact_b.canonical_metric or fact_b.fact.metric or "").strip().lower()

        if not metric_a or not metric_b:
            return None

        reasons: List[str] = ["ENTITY_MATCH"]
        score = 0.5
        metric_match = False

        if metric_a == metric_b:
            metric_match = True
            score = 1.0
            reasons.append("EXACT_METRIC_MATCH")
        else:
            tokens_a = cls._extract_metric_tokens(metric_a)
            tokens_b = cls._extract_metric_tokens(metric_b)

            if not tokens_a or not tokens_b:
                return None

            if tokens_a == tokens_b:
                metric_match = True
                score = 0.95
                reasons.append("CANONICAL_METRIC_TOKEN_MATCH")
            elif tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a):
                # Subset containment: e.g. 'Revenue' subset of 'Revenue from operations'
                common = tokens_a.intersection(tokens_b)
                metric_match = True
                score = 0.85
                reasons.append(f"METRIC_SUBSET_MATCH: {', '.join(sorted(common))}")
            else:
                # Check for quantity modifier synonyms: e.g. 'Employee count' vs 'Number of employees'
                core_a = tokens_a - cls.QUANTITY_MODIFIERS
                core_b = tokens_b - cls.QUANTITY_MODIFIERS

                if core_a and core_b and core_a == core_b:
                    metric_match = True
                    score = 0.80
                    reasons.append(f"METRIC_CORE_NOUN_MATCH: {', '.join(sorted(core_a))}")
                elif core_a and core_b and (core_a.issubset(core_b) or core_b.issubset(core_a)):
                    common_core = core_a.intersection(core_b)
                    if common_core:
                        metric_match = True
                        score = 0.75
                        reasons.append(f"METRIC_CORE_SUBSET_MATCH: {', '.join(sorted(common_core))}")
                else:
                    # High-precision Jaccard token overlap threshold (>= 0.50)
                    common = tokens_a.intersection(tokens_b)
                    union = tokens_a.union(tokens_b)
                    jaccard = len(common) / len(union) if union else 0.0

                    if jaccard >= 0.50:
                        metric_match = True
                        score = round(0.5 + 0.5 * jaccard, 2)
                        reasons.append(f"HIGH_PRECISION_METRIC_OVERLAP (Jaccard {jaccard:.2f}): {', '.join(sorted(common))}")
                    else:
                        return None

        return CandidatePair(
            fact_a_id=fact_a.fact.fact_id,
            fact_b_id=fact_b.fact.fact_id,
            entity_match=entity_match,
            metric_match=metric_match,
            candidate_score=score,
            reasons=reasons,
        )

    def find_candidates(
        self,
        facts: List[NormalizedFact],
    ) -> List[Tuple[NormalizedFact, NormalizedFact]]:
        """Retrieves distinct candidate fact tuples from a list of facts.

        Args:
            facts: List of NormalizedFact objects.

        Returns:
            List of (NormalizedFact, NormalizedFact) pairs that pass candidate matching.
        """
        candidates: List[Tuple[NormalizedFact, NormalizedFact]] = []
        n = len(facts)
        for i in range(n):
            for j in range(i + 1, n):
                match = self.match_pair(facts[i], facts[j])
                if match is not None:
                    candidates.append((facts[i], facts[j]))
        return candidates

    def find_candidate_pairs(
        self,
        facts: List[NormalizedFact],
    ) -> List[CandidatePair]:
        """Retrieves structured CandidatePair records from a list of facts."""
        candidate_pairs: List[CandidatePair] = []
        n = len(facts)
        for i in range(n):
            for j in range(i + 1, n):
                match = self.match_pair(facts[i], facts[j])
                if match is not None:
                    candidate_pairs.append(match)
        return candidate_pairs
