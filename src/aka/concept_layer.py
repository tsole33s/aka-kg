# concept_layer.py
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Feature:
    """A normalized semantic unit extracted from text."""
    key: str                      # canonical short label, e.g. "has_gills"
    embedding: np.ndarray         # L2-normalized vector
    count: int = 1                # how many times observed
    examples: list[str] = field(default_factory=list)  # source snippets


@dataclass
class Concept:
    """A promoted, stable knowledge node."""
    id: str
    label: str
    embedding: np.ndarray
    feature_keys: set[str] = field(default_factory=set)
    created_at: int = 0           # step index when born
    observations: int = 1


# ---------------------------------------------------------------------------
# Embedding backend (pluggable)
# ---------------------------------------------------------------------------

class Embedder:
    """Wraps any sentence-embedding function. Keeps the layer model-agnostic."""

    def __init__(self, fn: Callable[[str], np.ndarray], dim: int):
        self._fn = fn
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        v = np.asarray(self._fn(text), dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"Embedder returned {v.shape}, expected ({self.dim},)")
        # L2 normalize - critical to prevent magnitude drift (mode collapse guard #1)
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v


def default_embedder() -> Embedder:
    """Tiny hash-based fallback so the module runs without external deps.
    Swap for sentence-transformers / OpenAI embeddings in production."""
    dim = 256

    def _hash_embed(text: str) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        return vec

    return Embedder(_hash_embed, dim)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "it", "its", "they", "them", "their", "we", "you", "i", "he", "she",
    "do", "does", "did", "have", "has", "had", "can", "could", "will",
    "would", "should", "may", "might", "must", "not", "no", "yes",
}

_CUE_PATTERNS = [
    re.compile(r"\b(?P<subj>[\w\- ]+?)\s+(?:has|have|possesses|exhibits)\s+(?P<obj>[\w\- ]+)", re.I),
    re.compile(r"\b(?P<subj>[\w\- ]+?)\s+(?:can|could|is able to)\s+(?P<obj>[\w\- ]+)", re.I),
    re.compile(r"\b(?P<subj>[\w\- ]+?)\s+(?:is|are)\s+(?:a|an|the)?\s*(?P<obj>[\w\- ]+)", re.I),
    re.compile(r"\b(?P<subj>[\w\- ]+?)\s+(?:uses|produces|generates|creates)\s+(?P<obj>[\w\- ]+)", re.I),
]


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2]


def extract_features(text: str, max_features: int = 12) -> list[str]:
    """Return canonical short feature keys, e.g. ['has_gills', 'aquatic', 'respires']."""
    features: list[str] = []

    # 1. Cue-based relational features
    for pat in _CUE_PATTERNS:
        for m in pat.finditer(text):
            obj_tokens = _tokenize(m.group("obj"))
            if obj_tokens:
                features.append("has_" + obj_tokens[0])
                features.extend(obj_tokens)

    # 2. Content tokens as standalone features
    features.extend(_tokenize(text))

    # 3. Deduplicate, preserve order, cap length
    seen, out = set(), []
    for f in features:
        if f not in seen:
            seen.add(f)
            out.append(f)
        if len(out) >= max_features:
            break
    return out


# ---------------------------------------------------------------------------
# Concept formation layer
# ---------------------------------------------------------------------------

class ConceptLayer:
    """Stateful layer that turns text into Features and promotes them to Concepts.

    Parameters
    ----------
    embedder : Embedder
    novelty_threshold : float
        Cosine *distance* above which an observation spawns a NEW concept
        instead of merging into an existing one. 0.35 ~= "fairly different".
    promotion_threshold : int
        How many times a feature must be seen before it becomes a Concept.
    max_concepts : int
        Hard cap; when hit, we force-merge instead of spawning.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        novelty_threshold: float = 0.35,
        promotion_threshold: int = 3,
        max_concepts: int = 512,
    ):
        self.embedder = embedder or default_embedder()
        self.novelty_threshold = novelty_threshold
        self.promotion_threshold = promotion_threshold
        self.max_concepts = max_concepts

        self.features: dict[str, Feature] = {}
        self.concepts: dict[str, Concept] = {}
        self._cooccurring: dict[str, set[str]] = {}
        self._step = 0

    def observe(self, text: str) -> dict:
        """Run one observation through the full pipeline."""
        self._step += 1
        report = {
            "step": self._step,
            "new_features": [],
            "promoted_concepts": [],
            "matched_concepts": [],
            "noisy_features": [],
        }

        # Stage 1: extract
        feature_keys = extract_features(text)
        valid_keys: list[str] = []

        # Stage 2: embed + novelty check per feature
        for key in feature_keys:
            emb = self.embedder(key.replace("_", " "))

            if np.linalg.norm(emb) < 0.5:
                report["noisy_features"].append(key)
                continue

            valid_keys.append(key)

            if key in self.features:
                f = self.features[key]
                f.count += 1
                f.embedding = self._blend(f.embedding, emb, weight=1.0 / f.count)
                if len(f.examples) < 3:
                    f.examples.append(text[:120])
            else:
                self.features[key] = Feature(key=key, embedding=emb, examples=[text[:120]])
                report["new_features"].append(key)

        valid_set = set(valid_keys)
        for key in valid_keys:
            cooc = self._cooccurring.setdefault(key, set())
            cooc.update(valid_set - {key})

        # Stage 3: promotion - features seen enough times become concepts
        for key, feat in self.features.items():
            if feat.count < self.promotion_threshold:
                continue
            existing = self._find_concept(feat.embedding)
            if existing is None:
                if len(self.concepts) < self.max_concepts:
                    c = self._spawn_concept(feat)
                    report["promoted_concepts"].append(c.id)
            else:
                self._merge_into(existing, feat)
                if existing.id not in report["matched_concepts"]:
                    report["matched_concepts"].append(existing.id)

        return report

    def snapshot(self) -> dict:
        """Serializable state for the knowledge graph module."""
        return {
            "step": self._step,
            "features": {
                k: {"key": v.key, "count": v.count, "embedding": v.embedding.tolist()}
                for k, v in self.features.items()
            },
            "concepts": {
                k: {
                    "id": v.id,
                    "label": v.label,
                    "embedding": v.embedding.tolist(),
                    "feature_keys": sorted(v.feature_keys),
                    "created_at": v.created_at,
                    "observations": v.observations,
                }
                for k, v in self.concepts.items()
            },
        }

    @staticmethod
    def _blend(old: np.ndarray, new: np.ndarray, weight: float) -> np.ndarray:
        """Running average, re-normalized - keeps embeddings on the unit sphere."""
        merged = (1 - weight) * old + weight * new
        n = np.linalg.norm(merged)
        return merged / n if n > 1e-8 else old

    def _find_concept(self, emb: np.ndarray) -> Concept | None:
        if not self.concepts:
            return None
        best, best_sim = None, -1.0
        for c in self.concepts.values():
            sim = float(np.dot(c.embedding, emb))
            if sim > best_sim:
                best, best_sim = c, sim
        if best is not None and (1.0 - best_sim) < self.novelty_threshold:
            return best
        return None

    def _spawn_concept(self, feat: Feature) -> Concept:
        cid = f"c{len(self.concepts) + 1:04d}"
        cooc = self._cooccurring.get(feat.key, set())
        c = Concept(
            id=cid,
            label=feat.key,
            embedding=feat.embedding.copy(),
            feature_keys={feat.key} | cooc,
            created_at=self._step,
        )
        self.concepts[cid] = c
        return c

    def _merge_into(self, concept: Concept, feat: Feature) -> None:
        cooc = self._cooccurring.get(feat.key, set())
        concept.feature_keys.update({feat.key} | cooc)
        concept.observations += 1
        concept.embedding = self._blend(
            concept.embedding, feat.embedding, weight=1.0 / concept.observations
        )
