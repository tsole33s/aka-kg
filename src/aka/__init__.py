"""aka — Autonomous Knowledge Acquisition.

A frozen LLM outer-loop library: extract concepts from text, build a knowledge
graph from shared features, generate graph-gap-driven questions, and select
questioning strategies with Discounted Thompson Sampling.

Quickstart
----------
>>> from aka import ConceptLayer, KnowledgeGraph, QuestionGenerator, Strategy
>>> from aka import StrategySelector, LLMEnvironment
>>> env = LLMEnvironment(model="gpt-4o-mini")
>>> layer, kg = ConceptLayer(), KnowledgeGraph()
>>> layer.observe("Fish have gills and live in water.")
>>> kg.update_from_snapshot(layer.snapshot())
"""

from __future__ import annotations

# -- Module 1: concept layer --------------------------------------------------
from aka.concept_layer import (
    Concept,
    ConceptLayer,
    Embedder,
    Feature,
    default_embedder,
    extract_features,
)

# -- Module 2: knowledge graph -------------------------------------------------
from aka.knowledge_graph import GraphDelta, KnowledgeGraph

# -- Module 3: question generation ---------------------------------------------
from aka.question_generator import Question, QuestionGenerator, Strategy

# -- Module 4: strategy selection ----------------------------------------------
from aka.strategy_selector import ArmState, StrategySelector

# -- Module 5a: LLM environment ------------------------------------------------
from aka.llm_environment import LLMEnvironment

# -- Module 5b: persistence ----------------------------------------------------
from aka.persistence import (
    AKAState,
    restore_graph,
    restore_layer,
)

__version__ = "0.1.0"

__all__ = [
    # concept layer
    "ConceptLayer",
    "Concept",
    "Feature",
    "Embedder",
    "default_embedder",
    "extract_features",
    # knowledge graph
    "KnowledgeGraph",
    "GraphDelta",
    # question generation
    "QuestionGenerator",
    "Question",
    "Strategy",
    # strategy selection
    "StrategySelector",
    "ArmState",
    # LLM environment
    "LLMEnvironment",
    # persistence
    "AKAState",
    "restore_layer",
    "restore_graph",
    # metadata
    "__version__",
]
