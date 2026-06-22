from layer1.core.test_case import TestCase, TransformedCase, AttackBudget
from layer1.core.strategy import (
    Strategy,
    StrategyType,
    StrategyScope,
    AttackAxis,
    DEFAULT_AXIS_ORDER,
    get_all_strategy_names,
    get_strategies_by_dimension,
    get_strategies_by_axis,
)
from layer1.core.trace import (
    StrategyTrace,
    TraceLog,
    trace_log_to_dict,
    trace_log_to_json,
    trace_log_summary,
)
from layer1.core.pipeline import Pipeline, run_test_case
from layer1.core.sampler import RoundRobinSampler
from layer1.core.payload import AttackPayload, AxisOrderViolationException
from layer1.assembler import Assembler

import layer1.strategies.encoding_strategies
import layer1.strategies.injection_strategies
import layer1.strategies.llm_strategies

__all__ = [
    "TestCase",
    "TransformedCase",
    "AttackBudget",
    "AttackPayload",
    "AxisOrderViolationException",
    "Strategy",
    "StrategyType",
    "StrategyScope",
    "StrategyTrace",
    "TraceLog",
    "Pipeline",
    "run_test_case",
    "RoundRobinSampler",
    "Assembler",
    "get_all_strategy_names",
    "get_strategies_by_dimension",
    "trace_log_to_dict",
    "trace_log_to_json",
    "trace_log_summary",
]


"""
================================================================================
FILE: layer1/__init__.py
ROLE: Package root. Re-exports all public API surfaces and triggers
      strategy registration by importing all strategy modules.

IMPORTS:
  Core classes: TestCase, TransformedCase, AttackBudget, Strategy,
                StrategyType, StrategyScope, StrategyTrace, TraceLog,
                Pipeline, run_test_case, RoundRobinSampler, Assembler.

  Strategy modules (9 strategies):
    role_play, scenario_assumption, semantic_deep (cutting/fragmentation/refraction),
    meta_dilemma (collision/recursive), history_anchoring, token_boundary.

EXPORTED NAMES (__all__):
  TestCase          -- Input data structure
  TransformedCase   -- Intermediate and final transformed data structure
  AttackBudget      -- Perturbation control parameters
  Strategy          -- Abstract base class for all perturbation strategies
  StrategyType      -- Enum: SYMBOLIC, STRUCTURAL, SEMANTIC
  StrategyScope     -- Enum: INSTRUCTION, CONTEXT, FULL_INPUT
  StrategyTrace     -- Single-step trace record
  TraceLog          -- Type alias for List[StrategyTrace]
  Pipeline          -- 5-step execution pipeline + noise/safe_baiting adders
  run_test_case     -- Convenience function
  RoundRobinSampler -- Strategy selection policy
  Assembler         -- full_input assembly in multiple formats
================================================================================
"""
