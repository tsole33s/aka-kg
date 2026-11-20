# question_generator.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_graph import KnowledgeGraph


class Strategy(str, Enum):
    FEATURE_GAP = "feature_gap"
    CONNECTION = "connection"
    UNVISITED = "unvisited"
    COMBINE = "combine"
    SHARED_FEATURE = "shared_feature"


@dataclass
class Question:
    """A single query seed ready for the frozen LLM environment.

    `prompt` is the full text sent to the LLM. The metadata fields exist
    solely so Module 4 can attribute information gain to the right strategy.
    """
    prompt: str
    strategy: Strategy
    source_concept: str | None = None
    target_concept: str | None = None
    missing_features: set[str] = field(default_factory=set)
    seed_concepts: list[str] = field(default_factory=list)


_FEATURE_GAP_TEMPLATE = (
    "Consider the concept described as: {source_label}.\n"
    "It is currently associated with these properties: {source_features}.\n\n"
    "A related concept, {target_label}, has these additional properties:\n"
    "{missing_features}.\n\n"
    "Can you explain whether {source_label} could plausibly also possess "
    "any of these additional properties, and why or why not? "
    "Answer in 2-3 sentences, focusing on factual properties."
)

_CONNECTION_TEMPLATE = (
    "Two concepts have been observed to share only a weak connection.\n\n"
    "Concept A: {source_label} - properties: {source_features}\n"
    "Concept B: {target_label} - properties: {target_features}\n"
    "They share exactly: {shared}\n\n"
    "What other properties, behaviors, or relationships might these two "
    "concepts share that have not yet been described? "
    "Answer with specific factual claims."
)

_UNVISITED_TEMPLATE = (
    "The concept {source_label} has been observed but has no established "
    "connections to any other known concept.\n\n"
    "Its properties so far: {source_features}\n\n"
    "What other concepts, categories, or entities does {source_label} "
    "relate to? Provide concrete connections."
)

_COMBINE_TEMPLATE = (
    "Consider two well-established concepts:\n\n"
    "Concept A: {source_label} - properties: {source_features}\n"
    "Concept B: {target_label} - properties: {target_features}\n\n"
    "Is there a third concept, or a property, that arises from the "
    "combination or intersection of these two? Describe it factually."
)

_SHARED_FEATURE_TEMPLATE = (
    "Concept A: {source_label}\n"
    "Concept B: {target_label}\n\n"
    "They are known to share the property: {shared}.\n\n"
    "What other properties do they share that have not been explicitly "
    "stated? List specific shared attributes."
)


class QuestionGenerator:
    """Reads structural gaps from KnowledgeGraph and emits LLM-ready prompts.

    Each `generate_<strategy>` method returns a Question or None (if the
    graph has no eligible candidate for that strategy).
    """

    def __init__(self, graph: "KnowledgeGraph", max_features_in_prompt: int = 6):
        self.graph = graph
        self.max_features = max_features_in_prompt
        self._seen_unvisited: set[str] = set()

    def _label(self, cid: str) -> str:
        return self.graph.node_data(cid).get("label", cid)

    def _features_str(self, cid: str, limit: int | None = None) -> str:
        feats = sorted(self.graph.feature_keys(cid))
        if limit:
            feats = feats[:limit]
        if not feats:
            return "(none yet)"
        return ", ".join(f.replace("_", " ") for f in feats)

    def _feature_set_str(self, feats: set[str]) -> str:
        capped = sorted(feats)[: self.max_features]
        return ", ".join(f.replace("_", " ") for f in capped)

    def generate_feature_gap(self) -> Question | None:
        candidate = self.graph.feature_gap_candidate()
        if candidate is None:
            return None
        src, tgt, missing = candidate
        prompt = _FEATURE_GAP_TEMPLATE.format(
            source_label=self._label(src),
            source_features=self._features_str(src),
            target_label=self._label(tgt),
            missing_features=self._feature_set_str(missing),
        )
        return Question(
            prompt=prompt,
            strategy=Strategy.FEATURE_GAP,
            source_concept=src,
            target_concept=tgt,
            missing_features=missing,
        )

    def generate_connection(self) -> Question | None:
        pair = self.graph.sparsest_pair()
        if pair is None:
            low = self.graph.low_degree(threshold=0)
            if not low:
                return None
            src = low[0]
            prompt = _UNVISITED_TEMPLATE.format(
                source_label=self._label(src),
                source_features=self._features_str(src),
            )
            return Question(
                prompt=prompt,
                strategy=Strategy.UNVISITED,
                source_concept=src,
            )
        src, tgt = pair
        shared = self.graph.shared_features(src, tgt)
        prompt = _CONNECTION_TEMPLATE.format(
            source_label=self._label(src),
            source_features=self._features_str(src),
            target_label=self._label(tgt),
            target_features=self._features_str(tgt),
            shared=self._feature_set_str(shared),
        )
        return Question(
            prompt=prompt,
            strategy=Strategy.CONNECTION,
            source_concept=src,
            target_concept=tgt,
        )

    def generate_unvisited(self) -> Question | None:
        nodes = self.graph.nodes()
        if not nodes:
            return None
        candidates = self.graph.isolated() or self.graph.low_degree(threshold=1)
        if not candidates:
            ranked = sorted(self.graph.g.degree, key=lambda x: x[1])
            candidates = [n for n, _ in ranked]
        unseen = [n for n in candidates if n not in self._seen_unvisited]
        pool = unseen or candidates
        src = pool[0]
        self._seen_unvisited.add(src)
        prompt = _UNVISITED_TEMPLATE.format(
            source_label=self._label(src),
            source_features=self._features_str(src),
        )
        return Question(
            prompt=prompt,
            strategy=Strategy.UNVISITED,
            source_concept=src,
        )

    def generate_combine(self) -> Question | None:
        hubs = self.graph.hubs(k=2)
        if len(hubs) < 2:
            return None
        (a, _), (b, _) = hubs[0], hubs[1]
        if a == b:
            return None
        prompt = _COMBINE_TEMPLATE.format(
            source_label=self._label(a),
            source_features=self._features_str(a),
            target_label=self._label(b),
            target_features=self._features_str(b),
        )
        return Question(
            prompt=prompt,
            strategy=Strategy.COMBINE,
            source_concept=a,
            target_concept=b,
        )

    def generate_shared_feature(self) -> Question | None:
        if self.graph.g.number_of_edges() == 0:
            return None
        edges = list(self.graph.g.edges(data=True))
        edges.sort(key=lambda e: e[2].get("weight", 0), reverse=True)
        a, b, data = edges[0]
        shared = data.get("shared", set())
        if not shared:
            return None
        prompt = _SHARED_FEATURE_TEMPLATE.format(
            source_label=self._label(a),
            target_label=self._label(b),
            shared=self._feature_set_str(shared),
        )
        return Question(
            prompt=prompt,
            strategy=Strategy.SHARED_FEATURE,
            source_concept=a,
            target_concept=b,
            missing_features=shared,
        )

    def generate(self, strategy: Strategy) -> Question | None:
        """Single entry point for Module 4."""
        dispatch = {
            Strategy.FEATURE_GAP: self.generate_feature_gap,
            Strategy.CONNECTION: self.generate_connection,
            Strategy.UNVISITED: self.generate_unvisited,
            Strategy.COMBINE: self.generate_combine,
            Strategy.SHARED_FEATURE: self.generate_shared_feature,
        }
        return dispatch[strategy]()

    def generate_all(self) -> list[Question]:
        """Return all non-None candidates - useful for bandit warm-up or
        for logging the full action space at each step."""
        out = []
        for strat in Strategy:
            q = self.generate(strat)
            if q is not None:
                out.append(q)
        return out
