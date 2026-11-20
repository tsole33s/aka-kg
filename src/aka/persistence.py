# persistence.py
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .concept_layer import ConceptLayer
    from .knowledge_graph import KnowledgeGraph
    from .strategy_selector import StrategySelector


class AKAState:
    """Atomic checkpoint of the full AKA loop."""

    def __init__(self, path: Path | str = ".aka_state.json"):
        self.path = Path(path)

    def save(
        self,
        layer: "ConceptLayer",
        kg: "KnowledgeGraph",
        selector: "StrategySelector",
        step: int,
    ) -> None:
        payload = {
            "step": step,
            "concept_layer": layer.snapshot(),
            "knowledge_graph": kg.to_serializable(),
            "selector": self._selector_to_dict(selector),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)  # atomic

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    # -- selector serialization --------------------------------------------

    @staticmethod
    def _selector_to_dict(selector: "StrategySelector") -> dict:
        return {
            "step": selector.step,
            "discount": selector.discount,
            "reward_scale": selector.reward_scale,
            "arms": {
                (s.value if hasattr(s, "value") else str(s)): {
                    "alpha": a.alpha,
                    "beta": a.beta,
                    "pulls": a.pulls,
                    "total_reward": a.total_reward,
                    "last_reward": a.last_reward,
                }
                for s, a in selector.arms.items()
            },
        }

    @staticmethod
    def restore_selector(data: dict, strategies: list) -> "StrategySelector":
        from .strategy_selector import StrategySelector
        sel = StrategySelector(
            strategies=strategies,
            discount=data["discount"],
            reward_scale=data["reward_scale"],
        )
        sel.step = data["step"]
        for s in strategies:
            key = s.value if hasattr(s, "value") else str(s)
            arm_data = data["arms"].get(key)
            if arm_data:
                arm = sel.arms[s]
                arm.alpha = arm_data["alpha"]
                arm.beta = arm_data["beta"]
                arm.pulls = arm_data["pulls"]
                arm.total_reward = arm_data["total_reward"]
                arm.last_reward = arm_data["last_reward"]
        return sel


def restore_layer(snapshot: dict) -> "ConceptLayer":
    """Rebuild a ConceptLayer from its snapshot."""
    import numpy as np
    from .concept_layer import ConceptLayer, Feature, Concept

    layer = ConceptLayer()
    layer._step = snapshot["step"]

    for key, fdata in snapshot["features"].items():
        layer.features[key] = Feature(
            key=fdata["key"],
            embedding=np.asarray(fdata["embedding"], dtype=np.float32),
            count=fdata["count"],
        )

    for cid, cdata in snapshot["concepts"].items():
        layer.concepts[cid] = Concept(
            id=cdata["id"],
            label=cdata["label"],
            embedding=np.asarray(cdata["embedding"], dtype=np.float32),
            feature_keys=set(cdata["feature_keys"]),
            created_at=cdata["created_at"],
            observations=cdata["observations"],
        )
    return layer


def restore_graph(data: dict) -> "KnowledgeGraph":
    """Rebuild a KnowledgeGraph from to_serializable() output."""
    import numpy as np
    import networkx as nx
    from .knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    for cid, ndata in data["nodes"].items():
        kg.g.add_node(
            cid,
            label=ndata["label"],
            embedding=np.asarray(ndata["embedding"], dtype=np.float32),
            feature_keys=set(ndata["feature_keys"]),
            created_at=ndata["created_at"],
            observations=ndata["observations"],
        )
    for edge in data["edges"]:
        kg.g.add_edge(
            edge["a"], edge["b"],
            weight=edge["weight"],
            shared=set(edge["shared"]),
        )
    return kg