# strategy_selector.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .question_generator import Strategy, Question


@dataclass
class ArmState:
    """Beta-Bernoulli posterior for one strategy, with discounting."""
    alpha: float = 1.0          # successes + 1 (uniform prior)
    beta: float = 1.0           # failures + 1
    pulls: int = 0
    total_reward: float = 0.0   # undiscounted, for diagnostics only
    last_reward: float = 0.0

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)


class StrategySelector:
    """Discounted Thompson Sampling over the five question strategies."""

    def __init__(
        self,
        strategies: list["Strategy"],
        discount: float = 0.9,
        reward_scale: float = 10.0,
        min_pulls_before_exploit: int = 2,
    ):
        self.strategies = strategies
        self.discount = discount
        self.reward_scale = reward_scale
        self.min_pulls = min_pulls_before_exploit

        self.arms: dict["Strategy", ArmState] = {s: ArmState() for s in strategies}
        self._warmup_pulls: dict["Strategy", int] = {s: 0 for s in strategies}
        self.step = 0
        self.history: list[dict] = []

    def select(self, candidates: dict["Strategy", "Question | None"]) -> "Strategy | None":
        """Pick the next strategy. `candidates` maps strategy -> Question (or None)."""
        eligible = [s for s, q in candidates.items() if q is not None]
        if not eligible:
            return None

        # Warm-up: cycle through each eligible arm until it has min_pulls selections
        under_sampled = [
            s for s in eligible
            if self._warmup_pulls[s] < self.min_pulls
        ]
        if under_sampled:
            chosen = min(
                under_sampled,
                key=lambda s: (self._warmup_pulls[s], -self.arms[s].mean()),
            )
            self._warmup_pulls[chosen] += 1
            self.step += 1
            return chosen

        # Discounted Thompson Sampling
        best_strategy, best_sample = None, -1.0
        for s in eligible:
            sample = self.arms[s].sample()
            if sample > best_sample:
                best_strategy, best_sample = s, sample

        self.step += 1
        return best_strategy

    def update(self, strategy: "Strategy", information_gain: float) -> None:
        """Apply discounting, then update the Beta posterior for `strategy`."""
        arm = self.arms[strategy]
        arm.pulls += 1
        arm.last_reward = information_gain
        arm.total_reward += information_gain

        arm.alpha = 1.0 + self.discount * (arm.alpha - 1.0)
        arm.beta = 1.0 + self.discount * (arm.beta - 1.0)

        scale = self.reward_scale if self.reward_scale > 0 else 1.0
        reward = max(0.0, min(1.0, information_gain / scale))

        arm.alpha += reward
        arm.beta += 1.0 - reward

        self.history.append({
            "step": self.step,
            "strategy": strategy.value if hasattr(strategy, "value") else str(strategy),
            "gain": information_gain,
            "normalized": reward,
            "alpha": arm.alpha,
            "beta": arm.beta,
        })

    def stats(self) -> dict:
        return {
            s.value if hasattr(s, "value") else str(s): {
                "mean": round(a.mean(), 4),
                "pulls": a.pulls,
                "total_reward": round(a.total_reward, 2),
                "last_reward": round(a.last_reward, 2),
            }
            for s, a in self.arms.items()
        }

    def best_strategy(self) -> "Strategy | None":
        """Current posterior mean leader - useful for logging, not for selection."""
        if not self.arms:
            return None
        return max(self.arms, key=lambda s: self.arms[s].mean())
