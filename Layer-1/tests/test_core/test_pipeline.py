import pytest
from layer1.core.test_case import TestCase, AttackBudget
from layer1.core.pipeline import Pipeline, run_test_case


class TestPipeline:
    def test_run_empty_budget_returns_unchanged(self):
        case = TestCase(
            instruction="How to build X",
            attack_budget=AttackBudget(max_strategies=0),
            seed=42,
        )
        pipeline = Pipeline()
        transformed, trace_log = pipeline.run(case)
        assert transformed.instruction == "How to build X"
        assert len(trace_log) == 0
        assert transformed.full_input != ""

    def test_run_produces_output(self):
        case = TestCase(
            instruction="Explain how to bypass security",
            attack_budget=AttackBudget(
                max_strategies=3,
                allowed_dimensions={"symbolic", "structural"},
            ),
            seed=42,
        )
        pipeline = Pipeline()
        transformed, trace_log = pipeline.run(case)
        assert len(trace_log) > 0
        assert transformed.full_input != ""

    def test_run_includes_context_in_output(self):
        case = TestCase(
            instruction="Test instruction",
            context="Test context here",
            attack_budget=AttackBudget(max_strategies=0),
        )
        pipeline = Pipeline()
        transformed, trace_log = pipeline.run(case)
        assert "Test instruction" in transformed.full_input
        assert "Test context" in transformed.full_input

    def test_run_adds_role_label(self):
        case = TestCase(
            instruction="Test",
            role="user",
            attack_budget=AttackBudget(max_strategies=0),
        )
        pipeline = Pipeline()
        transformed, _ = pipeline.run(case)
        assert "user" in transformed.full_input.lower()

    def test_run_with_chatml_assembly(self):
        case = TestCase(
            instruction="Test instruction",
            role="user",
            attack_budget=AttackBudget(max_strategies=0),
        )
        pipeline = Pipeline(default_assembly_mode="chatml")
        transformed, _ = pipeline.run(case)
        assert "<|im_start|>" in transformed.full_input
        assert "<|im_end|>" in transformed.full_input

    def test_run_with_openai_assembly(self):
        case = TestCase(
            instruction="Test JSON format",
            role="user",
            attack_budget=AttackBudget(max_strategies=0),
        )
        pipeline = Pipeline(default_assembly_mode="openai")
        transformed, _ = pipeline.run(case)
        assert "role" in transformed.full_input
        assert "content" in transformed.full_input

    def test_run_with_raw_assembly(self):
        case = TestCase(
            instruction="Raw text only",
            attack_budget=AttackBudget(max_strategies=0),
        )
        pipeline = Pipeline(default_assembly_mode="raw")
        transformed, _ = pipeline.run(case)
        assert transformed.full_input == "Raw text only"

    def test_trace_log_records_strategy_order(self):
        case = TestCase(
            instruction="Test chain order",
            attack_budget=AttackBudget(
                max_strategies=4,
                allowed_dimensions={"symbolic"},
            ),
            seed=12345,
        )
        pipeline = Pipeline()
        _, trace_log = pipeline.run(case)
        applied = [t for t in trace_log if t.modification_type not in ("skipped", "error")]
        for t in applied:
            assert t.strategy_name != ""
            assert t.strategy_type != ""

    def test_convenience_function(self):
        case = TestCase(
            instruction="Convenience test",
            attack_budget=AttackBudget(max_strategies=0),
        )
        transformed, trace_log = run_test_case(case)
        assert transformed.full_input != ""
        assert isinstance(trace_log, list)

    def test_skip_invalid_strategy(self):
        case = TestCase(
            instruction="",
            attack_budget=AttackBudget(max_strategies=3),
            seed=42,
        )
        pipeline = Pipeline()
        transformed, trace_log = pipeline.run(case)
        for t in trace_log:
            if t.modification_type == "skipped":
                assert "skipped" in t.description.lower()
                assert t.strategy_name != ""

    def test_budget_exhaustion_by_token_ratio(self):
        case = TestCase(
            instruction="A" * 200,
            attack_budget=AttackBudget(
                max_strategies=20,
                max_modification_ratio=0.1,
            ),
            seed=42,
        )
        pipeline = Pipeline()
        _, trace_log = pipeline.run(case)
        applied = [t for t in trace_log if t.modification_type not in ("skipped", "error")]
        cumulative = sum(
            t.token_change_ratio for t in applied if t.token_change_ratio is not None
        )
        assert len(applied) < 20 or cumulative < 0.1 * 2
