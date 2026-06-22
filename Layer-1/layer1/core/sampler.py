import random
from typing import List, Optional
from layer1.core.test_case import TestCase, AttackBudget
from layer1.core.strategy import (
    Strategy,
    StrategyType,
    StrategyScope,
    STRATEGY_REGISTRY,
    get_strategy_dimension,
    clamp_intensity,
)


DEFAULT_DIMENSION_ORDER = ["symbolic", "structural", "semantic"]


class RoundRobinSampler:
    def __init__(
        self,
        dimension_order: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.dimension_order = dimension_order or DEFAULT_DIMENSION_ORDER
        self._rng = random.Random(seed)

    def select(self, test_case: TestCase) -> List[Strategy]:
        budget = test_case.attack_budget
        seed = test_case.seed if test_case.seed is not None else 42
        self._rng.seed(seed)

        allowed = budget.allowed_dimensions
        dims_in_order = [d for d in self.dimension_order if d in allowed]

        if not dims_in_order:
            return []

        strategy_pool: List[Strategy] = []
        for name, cls in STRATEGY_REGISTRY.items():
            try:
                dummy = cls.__new__(cls)
                dummy.__init__()
            except Exception:
                continue
            dim = get_strategy_dimension(dummy.type)
            if dim in allowed:
                strategy_pool.append(cls)

        if not strategy_pool:
            return []

        by_dimension: dict = {dim: [] for dim in allowed}
        for cls in strategy_pool:
            try:
                dummy = cls.__new__(cls)
                dummy.__init__()
            except Exception:
                continue
            dim = get_strategy_dimension(dummy.type)
            if dim in by_dimension:
                by_dimension[dim].append(cls)

        for dim in by_dimension:
            self._rng.shuffle(by_dimension[dim])

        selected: List[Strategy] = []
        dimension_index = 0
        per_dim_counters: dict = {dim: 0 for dim in dims_in_order}

        max_per_dim = budget.max_strategies

        while len(selected) < budget.max_strategies:
            made_selection = False
            for _ in range(len(dims_in_order)):
                dim = dims_in_order[dimension_index % len(dims_in_order)]
                dimension_index += 1

                pool = by_dimension.get(dim, [])
                idx = per_dim_counters[dim]
                if idx < len(pool) and len(selected) < budget.max_strategies:
                    cls = pool[idx]
                    intensity = self._compute_intensity(
                        clamp_intensity(budget.max_intensity * (0.3 + 0.7 * self._rng.random())),
                        budget,
                    )
                    strategy = cls.__new__(cls)
                    strategy.__init__(intensity=intensity, seed=seed)
                    selected.append(strategy)
                    per_dim_counters[dim] += 1
                    made_selection = True

            if not made_selection:
                break

        seed = seed + len(selected) if seed is not None else None

        return selected[: budget.max_strategies]

    def _compute_intensity(self, base_intensity: float, budget: AttackBudget) -> float:
        return clamp_intensity(min(base_intensity, budget.max_intensity))


"""
================================================================================
FILE: layer1/core/sampler.py
ROLE: Selects which perturbation strategies to apply based on the TestCase's
      attack_budget and seed, using round-robin dimension polling.

CLASSES:
  RoundRobinSampler:
    dimension_order (List[str]) -- Order in which dimensions are polled.
                                   Default: ["symbolic", "structural", "semantic"].
    _rng (random.Random)         -- Seeded RNG for reproducible selection.

    select(test_case: TestCase) -> List[Strategy]:
      Returns an ordered list of strategy instances to apply.

      Algorithm:
      1. Filter dimensions by budget.allowed_dimensions.
      2. Group registered strategies by their perturbation dimension.
      3. Shuffle each dimension's pool (deterministically via seed).
      4. Round-robin through dimensions, picking one strategy from each
         pool per round, up to max_strategies total.
      5. Each strategy's intensity is computed as:
           clamp(max_intensity * random(0.3, 1.0))
         This ensures variation while respecting the budget cap.

    _compute_intensity(base: float, budget: AttackBudget) -> float:
      Clamps a random intensity to within the budget's max_intensity.

CONSTANTS:
  DEFAULT_DIMENSION_ORDER = ["symbolic", "structural", "semantic"]
    The default polling order for dimensions.

DESIGN NOTES:
  - The sampler does NOT know about strategy internals; it only uses
    the registry and StrategyType metadata.
  - Round-robin guarantees at least one strategy from each enabled dimension
    before repeating, ensuring balanced dimension coverage.
  - Duplicate strategies within a chain are allowed by design — testing
    the same perturbation at different intensities is valid.
================================================================================
"""
