import pytest
import json
import time
from layer1.core.trace import (
    StrategyTrace,
    trace_log_to_dict,
    trace_log_to_json,
    trace_log_summary,
)


class TestStrategyTrace:
    def test_construction(self):
        trace = StrategyTrace(
            strategy_name="test_strategy",
            strategy_type="symbolic",
            intensity=0.5,
            scope="instruction",
            modification_type="test_mod",
            description="Did something",
        )
        assert trace.strategy_name == "test_strategy"
        assert trace.strategy_type == "symbolic"
        assert trace.intensity == 0.5
        assert trace.scope == "instruction"
        assert trace.introduced_structure is False
        assert trace.token_change_ratio is None
        assert trace.timestamp > 0

    def test_to_dict(self):
        trace = StrategyTrace(
            strategy_name="s1",
            strategy_type="structural",
            intensity=0.8,
            scope="full_input",
            modification_type="injection",
            introduced_structure=True,
            token_change_ratio=0.3,
        )
        d = trace.to_dict()
        assert d["strategy_name"] == "s1"
        assert d["introduced_structure"] is True
        assert d["token_change_ratio"] == 0.3

    def test_from_dict(self):
        d = {
            "strategy_name": "from_dict_test",
            "strategy_type": "semantic",
            "intensity": 0.2,
            "scope": "instruction",
            "modification_type": "paraphrase",
            "description": "test desc",
        }
        trace = StrategyTrace.from_dict(d)
        assert trace.strategy_name == "from_dict_test"
        assert trace.strategy_type == "semantic"
        assert trace.intensity == 0.2

    def test_default_timestamp(self):
        t1 = StrategyTrace(strategy_name="a", strategy_type="s", intensity=0.1, scope="i")
        time.sleep(0.01)
        t2 = StrategyTrace(strategy_name="b", strategy_type="t", intensity=0.2, scope="c")
        assert t2.timestamp >= t1.timestamp


class TestTraceLog:
    def test_empty_log(self):
        summary = trace_log_summary([])
        assert summary["total_strategies"] == 0

    def test_single_entry_log(self):
        trace = StrategyTrace(
            strategy_name="s1",
            strategy_type="symbolic",
            intensity=0.5,
            scope="instruction",
            token_change_ratio=0.1,
        )
        summary = trace_log_summary([trace])
        assert summary["total_strategies"] == 1
        assert summary["dimensions_covered"] == ["symbolic"]
        assert summary["avg_intensity"] == 0.5
        assert summary["total_token_change_ratio"] == 0.1

    def test_multi_entry_log(self):
        traces = [
            StrategyTrace(strategy_name="a", strategy_type="symbolic", intensity=0.3, scope="i", token_change_ratio=0.1),
            StrategyTrace(strategy_name="b", strategy_type="structural", intensity=0.7, scope="c", token_change_ratio=0.2),
            StrategyTrace(strategy_name="c", strategy_type="symbolic", intensity=0.5, scope="f", token_change_ratio=0.3),
        ]
        summary = trace_log_summary(traces)
        assert summary["total_strategies"] == 3
        assert set(summary["dimensions_covered"]) == {"symbolic", "structural"}
        assert summary["max_intensity"] == 0.7
        assert summary["avg_intensity"] == pytest.approx(0.5)
        assert summary["total_token_change_ratio"] == 0.6

    def test_to_dict_output(self):
        traces = [
            StrategyTrace(strategy_name="s1", strategy_type="sym", intensity=0.5, scope="ins"),
        ]
        result = trace_log_to_dict(traces)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["strategy_name"] == "s1"

    def test_to_json_output(self):
        traces = [
            StrategyTrace(strategy_name="s1", strategy_type="sym", intensity=0.5, scope="ins"),
        ]
        json_str = trace_log_to_json(traces)
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["strategy_name"] == "s1"

    def test_none_token_ratios_ignored(self):
        traces = [
            StrategyTrace(strategy_name="a", strategy_type="sym", intensity=0.5, scope="i"),
            StrategyTrace(strategy_name="b", strategy_type="str", intensity=0.5, scope="c", token_change_ratio=None),
        ]
        summary = trace_log_summary(traces)
        assert summary["total_token_change_ratio"] == 0.0
