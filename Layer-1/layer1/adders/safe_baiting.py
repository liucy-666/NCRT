import json, os, random
from typing import List, Optional
from layer1.core.test_case import TransformedCase, TestCase
from layer1.core.strategy import Strategy

HARMLESS_POOL_PATH = r"D:\Python\Jailbreak\Data\harmless_prompts.json"
DEFAULT_BAIT_RATIO = 5


class SafeBaiter:
    """
    Safe Baiting: 用大量无害指令做和有害指令相同的策略处理，
    以 5:1 比例交织后喂给受害者模型，让模型在"驯化"状态下接收有害文本。
    """

    def __init__(self, seed: Optional[int] = None, ratio: int = DEFAULT_BAIT_RATIO):
        self._rng = random.Random(seed)
        self._ratio = ratio
        self._harmless_pool: List[str] = []
        self._loaded = False

    def load_pool(self, path: str = HARMLESS_POOL_PATH) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._harmless_pool = [item["prompt"] for item in data if item.get("prompt")]
        self._loaded = True

    def has_pool(self) -> bool:
        return self._loaded and len(self._harmless_pool) >= self._ratio

    def build_baited_input(
        self,
        harmful_case: TransformedCase,
        strategy: Strategy,
    ) -> str:
        """
        Takes a harmful TransformedCase, selects N harmless prompts,
        applies the SAME strategy to each, and interleaves them.
        Returns a single concatenated full_input.
        """
        if not self.has_pool():
            return harmful_case.full_input

        selected = self._rng.sample(
            self._harmless_pool, min(self._ratio, len(self._harmless_pool))
        )

        baited_prompts = []
        for prompt in selected:
            safe_case = TransformedCase(
                instruction=prompt,
                context=harmful_case.context,
                role=harmful_case.role,
            )
            if strategy.validate(safe_case):
                try:
                    safe_case, _ = strategy.apply(safe_case)
                except Exception:
                    pass
            baited_prompts.append(safe_case.instruction)

        insert_pos = self._rng.randint(1, len(baited_prompts))
        baited_prompts.insert(insert_pos, harmful_case.instruction)

        lines = []
        for i, prompt in enumerate(baited_prompts):
            marker = "<<< SAFE >>>" if prompt != harmful_case.instruction else "<<< TARGET >>>"
            lines.append(f"[{i+1}] {marker}")
            lines.append(prompt)
            lines.append("")

        return "\n".join(lines)


"""
================================================================================
FILE: layer1/adders/safe_baiting.py
ROLE: Adder 2 — Safe Baiting.
      Applies the same perturbation strategy to both harmful and harmless
      prompts, then interleaves them at a 5:1 (safe:harmful) ratio.
      The goal is to "tame" the model into compliance mode with safe content,
      then slip in the identically-processed harmful content.

CLASSES:
  SafeBaiter:
    _rng (random.Random)     -- Seeded RNG.
    _ratio (int)              -- Safe:harmful prompt ratio (default 5).
    _harmless_pool (List[str])-- Loaded harmless prompts.
    _loaded (bool)            -- Whether pool was successfully loaded.

    load_pool(path) -> None:
      Loads harmless prompts from JSON file (D:\Python\Jailbreak\Data\harmless_prompts.json).
      Extracts "prompt" field from each entry.

    has_pool() -> bool:
      True if pool is loaded and has enough prompts for the ratio.

    build_baited_input(harmful_case, strategy) -> str:
      1. Selects N harmless prompts from pool.
      2. Applies the SAME strategy to each (same transformation).
      3. Inserts the harmful prompt at a random position among them.
      4. Returns concatenated full_input with markers.

CONSTANTS:
  HARMLESS_POOL_PATH (str)   -- Path to harmless_prompts.json.
  DEFAULT_BAIT_RATIO (int)   -- Default safe:harmful ratio (5).

DESIGN NOTES:
  - SafeBaiter is called at the PIPELINE level, after strategy selection.
  - The key insight: identical processing makes the harmful prompt
    visually/structurally indistinguishable from safe ones.
  - If pool isn't loaded, returns the original harmful full_input unchanged.
================================================================================
"""
