# aka_loop.py (updated)
from pathlib import Path
from .concept_layer import ConceptLayer
from .knowledge_graph import KnowledgeGraph
from .question_generator import QuestionGenerator, Strategy
from .strategy_selector import StrategySelector
from .llm_environment import LLMEnvironment
from .persistence import AKAState, restore_layer, restore_graph


def run_aka(
    seed_corpus: list[str],
    env: LLMEnvironment,
    steps: int = 100,
    checkpoint_path: Path | str = ".aka_state.json",
    resume: bool = False,
):
    state = AKAState(checkpoint_path)

    if resume and (saved := state.load()):
        layer = restore_layer(saved["concept_layer"])
        kg = restore_graph(saved["knowledge_graph"])
        selector = AKAState.restore_selector(saved["selector"], list(Strategy))
        start_step = saved["step"]
        print(f"Resumed from step {start_step}")
    else:
        layer = ConceptLayer(promotion_threshold=3)
        kg = KnowledgeGraph()
        selector = StrategySelector(strategies=list(Strategy))
        start_step = 0
        for text in seed_corpus:
            layer.observe(text)
            kg.update_from_snapshot(layer.snapshot())

    qg = QuestionGenerator(kg)

    for step in range(start_step, steps):
        candidates = {s: qg.generate(s) for s in Strategy}
        strategy = selector.select(candidates)
        if strategy is None:
            print(f"Step {step}: no candidates, stopping")
            break

        question = candidates[strategy]
        answer = env.query(question.prompt)

        layer.observe(answer)
        delta = kg.update_from_snapshot(layer.snapshot())
        selector.update(strategy, delta.information_gain())

        # Checkpoint every 10 steps
        if (step + 1) % 10 == 0:
            state.save(layer, kg, selector, step + 1)

    state.save(layer, kg, selector, steps)
    return layer, kg, selector