"""Smoke tests for the aka package.

These tests exercise the full loop end-to-end using a mock LLM, so they run
offline in CI with no API keys. They assert behavior the library *must* have
to be considered working, not implementation details of individual modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import aka
from aka import (
    AKAState,
    ConceptLayer,
    KnowledgeGraph,
    LLMEnvironment,
    QuestionGenerator,
    Strategy,
    StrategySelector,
    restore_graph,
    restore_layer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED_CORPUS = [
    "Fish have gills and live in water.",
    "Fish can swim and have gills for breathing.",
    "Birds have feathers and can fly.",
    "Birds use wings to fly and have feathers.",
    "Sharks have gills and are aquatic predators.",
    "Penguins have feathers and can swim in water.",
    "Penguins are birds that cannot fly.",
]


@pytest.fixture
def mock_env(tmp_path: Path) -> LLMEnvironment:
    """A deterministic LLM environment that never touches the network."""
    env = LLMEnvironment(model="mock", cache_dir=tmp_path / "cache")
    env.set_completion_fn(
        lambda prompt: (
            f"The concept described here shares properties with other known "
            f"entities. Specifically, it likely has attributes related to: "
            f"{prompt[:60].lower()}."
        )
    )
    return env


@pytest.fixture
def seeded_graph() -> tuple[ConceptLayer, KnowledgeGraph]:
    """A ConceptLayer + KnowledgeGraph built from the seed corpus."""
    layer = ConceptLayer(promotion_threshold=2)
    kg = KnowledgeGraph()
    for text in SEED_CORPUS:
        layer.observe(text)
        kg.update_from_snapshot(layer.snapshot())
    return layer, kg


# ---------------------------------------------------------------------------
# 1. Package surface
# ---------------------------------------------------------------------------

def test_package_imports():
    """The public API is importable from the top-level package."""
    assert aka.__version__ == "0.1.0"
    for name in aka.__all__:
        assert hasattr(aka, name), f"missing public symbol: {name}"


def test_py_typed_marker_exists():
    """The py.typed marker ships with the package (for type checkers)."""
    import aka as pkg
    pkg_dir = Path(pkg.__file__).parent
    assert (pkg_dir / "py.typed").exists(), "py.typed marker missing from package"


# ---------------------------------------------------------------------------
# 2. Module 1: ConceptLayer
# ---------------------------------------------------------------------------

def test_concept_layer_extracts_features():
    layer = ConceptLayer()
    report = layer.observe("Fish have gills and live in water.")
    assert report["step"] == 1
    assert len(report["new_features"]) > 0


def test_concept_layer_promotes_on_repeat():
    layer = ConceptLayer(promotion_threshold=2)
    r1 = layer.observe("Fish have gills.")
    assert r1["promoted_concepts"] == []  # threshold not met yet
    layer.observe("Fish have gills.")
    layer.observe("Fish have gills.")
    assert len(layer.concepts) > 0, "no concepts promoted after repeated observations"


def test_concept_embeddings_are_unit_norm():
    import numpy as np
    layer = ConceptLayer(promotion_threshold=1)
    layer.observe("Fish have gills and live in water.")
    layer.observe("Birds have feathers and can fly.")
    for concept in layer.concepts.values():
        norm = float(np.linalg.norm(concept.embedding))
        assert abs(norm - 1.0) < 1e-5, f"concept {concept.id} embedding not normalized"


# ---------------------------------------------------------------------------
# 3. Module 2: KnowledgeGraph
# ---------------------------------------------------------------------------

def test_graph_builds_nodes_and_edges(seeded_graph):
    _, kg = seeded_graph
    stats = kg.stats()
    assert stats["nodes"] > 0
    assert stats["edges"] > 0, "no edges created despite shared features in seed corpus"


def test_graph_edges_are_jaccard_weighted(seeded_graph):
    _, kg = seeded_graph
    for a, b, data in kg.g.edges(data=True):
        w = data["weight"]
        assert 0.0 < w <= 1.0, f"edge ({a},{b}) weight {w} outside (0,1]"


def test_graph_update_is_idempotent(seeded_graph):
    layer, kg = seeded_graph
    before = kg.stats()
    delta = kg.update_from_snapshot(layer.snapshot())
    assert delta.empty(), "re-syncing same snapshot produced a non-empty delta"
    after = kg.stats()
    assert before == after


def test_graph_information_gain_is_non_negative(seeded_graph):
    layer, kg = seeded_graph
    layer.observe("Whales have gills and live in water.")
    delta = kg.update_from_snapshot(layer.snapshot())
    assert delta.information_gain() >= 0.0


# ---------------------------------------------------------------------------
# 4. Module 3: QuestionGenerator
# ---------------------------------------------------------------------------

def test_question_generator_produces_valid_questions(seeded_graph):
    _, kg = seeded_graph
    qg = QuestionGenerator(kg)
    q = qg.generate(Strategy.FEATURE_GAP)
    if q is not None:
        assert isinstance(q.prompt, str)
        assert len(q.prompt) > 20
        assert q.strategy == Strategy.FEATURE_GAP


def test_question_generator_handles_empty_graph():
    kg = KnowledgeGraph()
    qg = QuestionGenerator(kg)
    # Every strategy should return None on an empty graph
    for strategy in Strategy:
        assert qg.generate(strategy) is None


# ---------------------------------------------------------------------------
# 5. Module 4: StrategySelector
# ---------------------------------------------------------------------------

def test_selector_returns_none_without_candidates():
    selector = StrategySelector(strategies=list(Strategy))
    assert selector.select({s: None for s in Strategy}) is None


def test_selector_warmup_visits_each_eligible_arm():
    selector = StrategySelector(
        strategies=list(Strategy),
        min_pulls_before_exploit=1,
    )
    candidates = {s: object() for s in Strategy}  # all eligible
    seen = set()
    for _ in range(len(Strategy) * 2):
        s = selector.select(candidates)
        seen.add(s)
    assert seen == set(Strategy), "warm-up did not visit every eligible arm"


def test_selector_discounts_old_rewards():
    selector = StrategySelector(strategies=list(Strategy), discount=0.5)
    s0 = list(Strategy)[0]
    for _ in range(20):
        selector.update(s0, information_gain=5.0)
    alpha_high = selector.arms[s0].alpha
    # After many more updates, alpha should not grow without bound
    for _ in range(20):
        selector.update(s0, information_gain=5.0)
    assert selector.arms[s0].alpha < alpha_high * 2


# ---------------------------------------------------------------------------
# 6. Module 5a: LLMEnvironment
# ---------------------------------------------------------------------------

def test_llm_environment_caches(tmp_path: Path):
    calls = {"n": 0}

    def counting_fn(prompt: str) -> str:
        calls["n"] += 1
        return f"answer to {prompt!r}"

    env = LLMEnvironment(model="mock", cache_dir=tmp_path)
    env.set_completion_fn(counting_fn)

    r1 = env.query("hello")
    r2 = env.query("hello")
    assert r1 == r2
    assert calls["n"] == 1, "cache miss — same prompt triggered a second LLM call"


def test_llm_environment_retries_on_failure(tmp_path: Path):
    attempts = {"n": 0}

    def flaky_fn(prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    env = LLMEnvironment(model="mock", cache_dir=tmp_path, retry_base_wait=0.01)
    env.set_completion_fn(flaky_fn)
    assert env.query("hi") == "ok"
    assert attempts["n"] == 3


# ---------------------------------------------------------------------------
# 7. Module 5b: Persistence
# ---------------------------------------------------------------------------

def test_state_roundtrip(seeded_graph, tmp_path: Path):
    layer, kg = seeded_graph
    selector = StrategySelector(strategies=list(Strategy))
    selector.update(list(Strategy)[0], information_gain=5.0)

    state = AKAState(tmp_path / "state.json")
    state.save(layer, kg, selector, step=42)

    saved = state.load()
    assert saved is not None
    assert saved["step"] == 42

    layer2 = restore_layer(saved["concept_layer"])
    kg2 = restore_graph(saved["knowledge_graph"])
    selector2 = AKAState.restore_selector(saved["selector"], list(Strategy))

    assert len(layer2.concepts) == len(layer.concepts)
    assert kg2.g.number_of_nodes() == kg.g.number_of_nodes()
    assert kg2.g.number_of_edges() == kg.g.number_of_edges()
    assert selector2.step == selector.step


# ---------------------------------------------------------------------------
# 8. Full loop
# ---------------------------------------------------------------------------

def test_full_loop_runs_end_to_end(mock_env: LLMEnvironment):
    """The complete pipeline: seed → select → query → observe → reward."""
    layer = ConceptLayer(promotion_threshold=2)
    kg = KnowledgeGraph()
    for text in SEED_CORPUS:
        layer.observe(text)
        kg.update_from_snapshot(layer.snapshot())

    qg = QuestionGenerator(kg)
    selector = StrategySelector(strategies=list(Strategy))

    initial_nodes = kg.g.number_of_nodes()
    total_gain = 0.0

    for _ in range(10):
        candidates = {s: qg.generate(s) for s in Strategy}
        strategy = selector.select(candidates)
        if strategy is None:
            break
        answer = mock_env.query(candidates[strategy].prompt)
        layer.observe(answer)
        delta = kg.update_from_snapshot(layer.snapshot())
        selector.update(strategy, delta.information_gain())
        total_gain += delta.information_gain()

    # Loose assertions: the loop must run and make progress
    assert kg.g.number_of_nodes() >= initial_nodes
    assert total_gain >= 0.0
    # Selector must have logged at least one pull (unless no candidates at all)
    if total_gain > 0:
        assert sum(a.pulls for a in selector.arms.values()) > 0


def test_full_loop_with_empty_corpus(mock_env: LLMEnvironment):
    """An empty seed corpus must not crash — the loop should no-op cleanly."""
    layer = ConceptLayer()
    kg = KnowledgeGraph()
    qg = QuestionGenerator(kg)
    selector = StrategySelector(strategies=list(Strategy))

    candidates = {s: qg.generate(s) for s in Strategy}
    assert selector.select(candidates) is None
