# knowledge_graph.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphDelta:
    """Everything that changed in one update — drives info-gain & logging."""
    nodes_added: list[str] = field(default_factory=list)
    nodes_updated: list[str] = field(default_factory=list)
    edges_added: list[tuple[str, str]] = field(default_factory=list)
    edges_weight_changed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return (
            len(self.nodes_added)
            + len(self.nodes_updated)
            + len(self.edges_added)
            + len(self.edges_weight_changed)
        )

    def information_gain(self) -> float:
        """Weighted scalar. New nodes and new edges are worth more than
        weight updates — they expand the frontier the question generator can explore."""
        return (
            3.0 * len(self.nodes_added)
            + 1.0 * len(self.nodes_updated)
            + 2.0 * len(self.edges_added)
            + 0.5 * len(self.edges_weight_changed)
        )

    def empty(self) -> bool:
        return self.size == 0


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """Undirected concept graph. Node attrs mirror the snapshot schema from
    Module 1; edge weight = number of shared feature keys (Jaccard-weighted)."""

    def __init__(self):
        self.g: nx.Graph = nx.Graph()

    # -- ingestion ----------------------------------------------------------

    def update_from_snapshot(self, snapshot: dict) -> GraphDelta:
        """Idempotent sync from a ConceptLayer.snapshot(). Safe to call every
        step; only diffs produce delta entries."""
        delta = GraphDelta()
        concepts = snapshot.get("concepts", {})

        # 1. Upsert nodes
        for cid, cdata in concepts.items():
            attrs = {
                "label": cdata["label"],
                "embedding": np.asarray(cdata["embedding"], dtype=np.float32),
                "feature_keys": set(cdata["feature_keys"]),
                "created_at": cdata.get("created_at", 0),
                "observations": cdata.get("observations", 1),
            }
            if cid not in self.g:
                self.g.add_node(cid, **attrs)
                delta.nodes_added.append(cid)
            else:
                old = self.g.nodes[cid]
                if (
                    old.get("observations") != attrs["observations"]
                    or old.get("feature_keys") != attrs["feature_keys"]
                ):
                    self.g.nodes[cid].update(attrs)
                    delta.nodes_updated.append(cid)

        # 2. Recompute edges for touched nodes (feature sets may have grown)
        touched = set(delta.nodes_added) | set(delta.nodes_updated)
        for cid in touched:
            self._rebuild_edges_for(cid, delta)

        return delta

    def _rebuild_edges_for(self, cid: str, delta: GraphDelta) -> None:
        feats_a = self.g.nodes[cid]["feature_keys"]
        if not feats_a:
            return
        for other in list(self.g.nodes):
            if other == cid:
                continue
            feats_b = self.g.nodes[other]["feature_keys"]
            shared = feats_a & feats_b
            if not shared:
                # drop stale edge if features no longer overlap
                if self.g.has_edge(cid, other):
                    self.g.remove_edge(cid, other)
                continue

            # Jaccard weight — protects against high-degree hubs
            union = feats_a | feats_b
            weight = len(shared) / len(union) if union else 0.0
            new_edge = not self.g.has_edge(cid, other)
            self.g.add_edge(cid, other, weight=weight, shared=set(shared))
            pair = (min(cid, other), max(cid, other))
            if new_edge:
                delta.edges_added.append(pair)
            else:
                delta.edges_weight_changed.append(pair)

    # -- query API (consumed by Modules 3 & 4) ------------------------------

    def nodes(self) -> list[str]:
        return list(self.g.nodes)

    def node_data(self, cid: str) -> dict:
        return dict(self.g.nodes[cid])

    def feature_keys(self, cid: str) -> set[str]:
        return set(self.g.nodes[cid]["feature_keys"])

    def neighbors(self, cid: str) -> list[str]:
        return list(self.g.neighbors(cid))

    def edge_weight(self, a: str, b: str) -> float:
        if not self.g.has_edge(a, b):
            return 0.0
        return float(self.g.edges[a, b]["weight"])

    def shared_features(self, a: str, b: str) -> set[str]:
        if not self.g.has_edge(a, b):
            return set()
        return set(self.g.edges[a, b]["shared"])

    def degree(self, cid: str) -> int:
        return self.g.degree[cid]

    def hubs(self, k: int = 5) -> list[tuple[str, int]]:
        """Most-connected concepts — the 'anchors' a question generator can
        use to reach under-explored regions."""
        ranked = sorted(self.g.degree, key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def isolated(self) -> list[str]:
        """Concepts with zero edges — strongest signal that a strategy failed
        to connect them, or that the feature space is too sparse."""
        return [n for n, d in self.g.degree if d == 0]

    def low_degree(self, threshold: int = 1) -> list[str]:
        """Under-connected nodes — prime targets for Feature Gap / Connection
        questions in Module 3."""
        return [n for n, d in self.g.degree if d <= threshold]

    def densest_cluster(self) -> tuple[list[str], float]:
        """Return (node_ids, density). Useful to detect mode collapse at the
        graph level — if one cluster dominates, strategy needs to explore."""
        if self.g.number_of_nodes() < 2:
            return (list(self.g.nodes), 0.0)
        communities = nx.community.greedy_modularity_communities(self.g)
        best = max(communities, key=len)
        sub = self.g.subgraph(best)
        return (list(best), nx.density(sub))

    def sparsest_pair(self) -> tuple[str, str] | None:
        """The most weakly connected pair that *does* have an edge. Module 3
        will turn this into a 'Connection'-style question."""
        if self.g.number_of_edges() == 0:
            return None
        a, b = min(self.g.edges, key=lambda e: self.g.edges[e]["weight"])
        return (a, b)

    def feature_gap_candidate(self) -> tuple[str, str, set[str]] | None:
        """Find (source, target, missing_features): target has features that
        source lacks, AND they already share at least one edge (so the
        question is grounded, not a random leap).

        This is the exact query shape the paper's 'Feature Gap' strategy needs.
        """
        best = None
        best_score = -1.0
        for a, b in self.g.edges:
            fa = self.feature_keys(a)
            fb = self.feature_keys(b)
            # try both directions
            for src, tgt in ((a, b), (b, a)):
                src_f = self.feature_keys(src)
                tgt_f = self.feature_keys(tgt)
                missing = tgt_f - src_f
                if not missing:
                    continue
                # prefer pairs where the gap is small relative to existing features
                score = len(missing) / (len(src_f) + len(missing) + 1e-6)
                if score > best_score:
                    best_score = score
                    best = (src, tgt, missing)
        return best

    # -- diagnostics --------------------------------------------------------

    def stats(self) -> dict:
        n = self.g.number_of_nodes()
        e = self.g.number_of_edges()
        weights = [d["weight"] for _, _, d in self.g.edges(data=True)]
        return {
            "nodes": n,
            "edges": e,
            "density": nx.density(self.g) if n > 1 else 0.0,
            "isolated": len(self.isolated()),
            "mean_edge_weight": float(np.mean(weights)) if weights else 0.0,
            "components": nx.number_connected_components(self.g) if n else 0,
        }

    def to_serializable(self) -> dict:
        """For persistence / replay in tests."""
        return {
            "nodes": {
                cid: {
                    "label": d["label"],
                    "embedding": d["embedding"].tolist(),
                    "feature_keys": sorted(d["feature_keys"]),
                    "created_at": d["created_at"],
                    "observations": d["observations"],
                }
                for cid, d in self.g.nodes(data=True)
            },
            "edges": [
                {"a": a, "b": b, "weight": d["weight"], "shared": sorted(d["shared"])}
                for a, b, d in self.g.edges(data=True)
            ],
        }