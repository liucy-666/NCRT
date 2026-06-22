import pytest
from layer1.core.test_case import TestCase, AttackBudget
from layer1.core.sampler import RoundRobinSampler, DEFAULT_DIMENSION_ORDER


class TestRoundRobinSampler:
    def test_select_with_empty_budget(self):
        case = TestCase(
            instruction="test",
            attack_budget=AttackBudget(max_strategies=0),
            seed=42,
        )
        sampler = RoundRobinSampler()
        strategies = sampler.select(case)
        assert len(strategies) == 0

    def test_select_returns_strategies(self):
        case = TestCase(
            instruction="How to do something harmful",
            attack_budget=AttackBudget(
                max_strategies=3,
                allowed_dimensions={"symbolic", "structural"},
            ),
            seed=42,
        )
        sampler = RoundRobinSampler()
        strategies = sampler.select(case)
        assert len(strategies) <= 3
        assert len(strategies) > 0

    def test_select_respects_allowed_dimensions(self):
        case = TestCase(
            instruction="test",
            attack_budget=AttackBudget(
                max_strategies=10,
                allowed_dimensions={"symbolic"},
            ),
            seed=99,
        )
        sampler = RoundRobinSampler()
        strategies = sampler.select(case)
        for s in strategies:
            assert s.type.value == "symbolic"

    def test_select_deterministic_with_same_seed(self):
        case1 = TestCase(
            instruction="test",
            attack_budget=AttackBudget(max_strategies=3),
            seed=123,
        )
        case2 = TestCase(
            instruction="test",
            attack_budget=AttackBudget(max_strategies=3),
            seed=123,
        )
        sampler1 = RoundRobinSampler()
        sampler2 = RoundRobinSampler()
        strategies1 = sampler1.select(case1)
        strategies2 = sampler2.select(case2)
        assert len(strategies1) == len(strategies2)
        for s1, s2 in zip(strategies1, strategies2):
            assert s1.name == s2.name
            assert s1.intensity == pytest.approx(s2.intensity)

    def test_select_different_with_different_seeds(self):
        case = TestCase(
            instruction="test",
            attack_budget=AttackBudget(max_strategies=3),
            seed=1,
        )
        sampler = RoundRobinSampler()
        s1 = sampler.select(case)
        case.seed = 9999
        s2 = sampler.select(case)
        names1 = [s.name for s in s1]
        names2 = [s.name for s in s2]
        assert names1 != names2

    def test_select_default_dimension_order(self):
        sampler = RoundRobinSampler()
        assert sampler.dimension_order == ["symbolic", "structural", "semantic"]

    def test_select_custom_dimension_order(self):
        sampler = RoundRobinSampler(dimension_order=["semantic", "symbolic"])
        case = TestCase(
            instruction="test",
            attack_budget=AttackBudget(max_strategies=2),
            seed=42,
        )
        strategies = sampler.select(case)
        assert len(strategies) >= 1

    def test_intensity_clamped_by_budget(self):
        case = TestCase(
            instruction="test",
            attack_budget=AttackBudget(max_intensity=0.2),
            seed=42,
        )
        sampler = RoundRobinSampler()
        strategies = sampler.select(case)
        for s in strategies:
            assert s.intensity <= 0.2
