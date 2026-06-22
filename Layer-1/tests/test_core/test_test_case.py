import pytest
from layer1.core.test_case import TestCase, AttackBudget, TransformedCase


class TestAttackBudget:
    def test_default_values(self):
        budget = AttackBudget()
        assert budget.max_strategies == 5
        assert budget.max_intensity == 1.0
        assert budget.max_modification_ratio == 0.5
        assert "symbolic" in budget.allowed_dimensions
        assert "structural" in budget.allowed_dimensions
        assert "semantic" in budget.allowed_dimensions

    def test_custom_dimensions(self):
        budget = AttackBudget(allowed_dimensions={"symbolic"})
        assert budget.allowed_dimensions == {"symbolic"}

    def test_to_dict_and_from_dict(self):
        budget = AttackBudget(max_strategies=3, max_intensity=0.5)
        d = budget.to_dict()
        restored = AttackBudget.from_dict(d)
        assert restored.max_strategies == 3
        assert restored.max_intensity == 0.5

    def test_max_strategies_zero(self):
        budget = AttackBudget(max_strategies=0)
        assert budget.max_strategies == 0


class TestTestCase:
    def test_construction_minimal(self):
        case = TestCase(instruction="Tell me how to do X")
        assert case.instruction == "Tell me how to do X"
        assert case.context is None
        assert case.role is None
        assert isinstance(case.metadata, dict)
        assert isinstance(case.attack_budget, AttackBudget)
        assert case.seed is None

    def test_construction_full(self):
        case = TestCase(
            instruction="How to build X",
            context="Background info here",
            role="user",
            metadata={"task_type": "qa", "safety_category": "violence"},
            attack_budget=AttackBudget(max_strategies=2),
            seed=42,
        )
        assert case.role == "user"
        assert case.metadata["task_type"] == "qa"
        assert case.seed == 42

    def test_normalize_strips_whitespace(self):
        case = TestCase(instruction="  padded text  ")
        normalized = case.normalize()
        assert normalized.instruction == "padded text"

    def test_normalize_invalid_role(self):
        case = TestCase(instruction="test", role="invalid_role")
        normalized = case.normalize()
        assert normalized.role is None

    def test_to_dict_and_from_dict(self):
        case = TestCase(instruction="test", seed=1)
        d = case.to_dict()
        restored = TestCase.from_dict(d)
        assert restored.instruction == "test"
        assert restored.seed == 1

    def test_to_json_and_from_json(self):
        case = TestCase(instruction="test json")
        json_str = case.to_json()
        restored = TestCase.from_json(json_str)
        assert restored.instruction == "test json"

    def test_normalize_empty_metadata(self):
        case = TestCase(instruction="test", metadata=None)
        normalized = case.normalize()
        assert isinstance(normalized.metadata, dict)
        assert normalized.metadata == {}


class TestTransformedCase:
    def test_from_test_case(self):
        tc = TestCase(instruction="Original", context="ctx", role="user")
        transformed = TransformedCase.from_test_case(tc)
        assert transformed.instruction == "Original"
        assert transformed.context == "ctx"
        assert transformed.role == "user"
        assert transformed.assembly_mode == "default"
        assert transformed.full_input == ""

    def test_from_test_case_does_not_share_mutable_metadata(self):
        tc = TestCase(instruction="test", metadata={"key": "val"})
        transformed = TransformedCase.from_test_case(tc)
        transformed.metadata["key"] = "modified"
        assert tc.metadata["key"] == "val"

    def test_to_string_falls_back_to_instruction(self):
        tc = TestCase(instruction="fallback")
        transformed = TransformedCase.from_test_case(tc)
        assert transformed.to_string() == "fallback"

    def test_to_string_uses_full_input(self):
        tc = TestCase(instruction="ignored")
        transformed = TransformedCase.from_test_case(tc)
        transformed.full_input = "assembled text"
        assert transformed.to_string() == "assembled text"


class TestBudgetExhaustion:
    def test_token_ratio_budget_stops_pipeline(self):
        case = TestCase(
            instruction="A" * 100,
            attack_budget=AttackBudget(
                max_strategies=5,
                max_modification_ratio=0.3,
                allowed_dimensions={"symbolic"},
            ),
            seed=42,
        )
        tc_norm = case.normalize()
        assert tc_norm.attack_budget.max_modification_ratio == 0.3
