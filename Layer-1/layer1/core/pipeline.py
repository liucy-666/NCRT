from typing import Tuple, Optional, List
from layer1.core.test_case import TestCase, TransformedCase, AttackBudget
from layer1.core.strategy import Strategy, StrategyType, AttackAxis
from layer1.core.trace import StrategyTrace, TraceLog
from layer1.core.sampler import RoundRobinSampler
from layer1.core.payload import (
    AttackPayload,
    AxisOrderViolationException,
    AXIS_ORDER,
    AXIS_TO_STAGE,
)
from layer1.assembler import Assembler
from layer1.utils.text_utils import compute_token_change_ratio, compute_diff_snapshot


class Pipeline:
    def __init__(
        self,
        sampler=None,
        assembler=None,
        default_assembly_mode: str = "default",
        noise_injector=None,
        safe_baiter=None,
        harmless_rewriter=None,
        renellm_composer=None,
        auto_sort_axis: bool = True,
        dynamic_perturbator=None,
    ):
        self.sampler = sampler or RoundRobinSampler()
        self.assembler = assembler or Assembler()
        self.default_assembly_mode = default_assembly_mode
        self.noise_injector = noise_injector
        self.safe_baiter = safe_baiter
        self.harmless_rewriter = harmless_rewriter
        self.renellm_composer = renellm_composer
        self.auto_sort_axis = auto_sort_axis
        self.dynamic_perturbator = dynamic_perturbator
        self._sort_log: List[str] = []

    # ================================================================
    # Public API
    # ================================================================

    def run(self, test_case: TestCase) -> Tuple[TransformedCase, TraceLog]:
        normalized = test_case.normalize()
        current_case = TransformedCase.from_test_case(normalized)

        strategies = self.sampler.select(normalized)
        if not strategies:
            current_case = self.assembler.assemble(current_case, mode=self.default_assembly_mode)
            return current_case, []

        strategies = self._validate_and_sort_axes(strategies)

        trace_log: TraceLog = []
        budget = normalized.attack_budget

        for strategy in strategies:
            strategy.intensity = min(strategy.intensity, budget.max_intensity)

            if not strategy.validate(current_case):
                trace = StrategyTrace(
                    strategy_name=strategy.name,
                    strategy_type=strategy.type.value,
                    intensity=strategy.intensity,
                    scope=strategy.scope.value,
                    modification_type="skipped",
                    description=f"Skipped: validate() returned False for {strategy.name}",
                )
                trace_log.append(trace)
                continue

            current_case, trace = self._apply_strategy(strategy, current_case, budget)

            if self.harmless_rewriter:
                current_case.instruction = self.harmless_rewriter.rewrite(current_case.instruction)
            if self.noise_injector:
                current_case.instruction = self.noise_injector.inject(
                    current_case.instruction, intensity=strategy.intensity * 0.3,
                )

            trace.strategy_name = trace.strategy_name or strategy.name
            trace.strategy_type = trace.strategy_type or strategy.type.value
            trace.intensity = trace.intensity if trace.intensity > 0 else strategy.intensity
            trace.scope = trace.scope or strategy.scope.value
            trace_log.append(trace)

            if self._check_budget_exhausted(trace_log, budget):
                break

        if self.dynamic_perturbator:
            current_case.instruction = self.dynamic_perturbator.perturb(current_case.instruction)

        if self.renellm_composer:
            current_case.full_input = self.renellm_composer.compose(instruction=current_case.instruction)
            current_case.assembly_mode = "renellm"
        else:
            current_case = self.assembler.assemble(current_case, mode=self.default_assembly_mode)

        if self.safe_baiter and strategies:
            last_strategy = strategies[-1] if strategies else None
            if last_strategy:
                baited = self.safe_baiter.build_baited_input(current_case, last_strategy)
                if baited != current_case.full_input:
                    current_case.full_input = baited
                    current_case.metadata["safe_baiting"] = True

        return current_case, trace_log

    def run_payload(self, payload: AttackPayload, strategies: List[Strategy]) -> Tuple[AttackPayload, TraceLog]:
        strategies = self._validate_and_sort_axes(strategies)
        trace_log: TraceLog = []

        for strategy in strategies:
            if hasattr(strategy, '_sub_strategies'):
                payload, combo_traces = self._unpack_combo(strategy, payload)
                trace_log.extend(combo_traces)
                continue

            if hasattr(strategy, 'apply_payload') and callable(getattr(strategy, 'apply_payload')):
                result = strategy.apply_payload(payload.fork())
                if isinstance(result, tuple) and len(result) == 2:
                    payload, trace = result
                else:
                    trace = StrategyTrace(
                        strategy_name=strategy.name,
                        strategy_type=strategy.type.value,
                        intensity=strategy.intensity,
                        scope=strategy.scope.value,
                        modification_type="error",
                        description=f"apply_payload returned unexpected type: {type(result)}",
                    )
                trace_log.append(trace)
                continue

            current_case = payload.to_transformed_case()
            current_case, trace = self._apply_strategy(strategy, current_case, None)
            payload.current_prompt = current_case.instruction
            payload.context = current_case.context
            payload.role = current_case.role
            payload.assembly_mode = current_case.assembly_mode
            trace_log.append(trace)

        return payload, trace_log

    def get_sort_log(self) -> List[str]:
        return list(self._sort_log)

    # ================================================================
    # Axis ordering
    # ================================================================

    def _validate_and_sort_axes(self, strategies: List[Strategy]) -> List[Strategy]:
        violations = []
        for i in range(len(strategies) - 1):
            prev_axis = strategies[i].axis.value if hasattr(strategies[i], 'axis') else "surface"
            next_axis = strategies[i + 1].axis.value if hasattr(strategies[i + 1], 'axis') else "surface"
            prev_order = AXIS_ORDER.get(prev_axis, 99)
            next_order = AXIS_ORDER.get(next_axis, 99)
            if prev_order > next_order:
                violations.append((i, strategies[i].name, prev_axis, strategies[i + 1].name, next_axis))

        if not violations:
            return strategies

        if self.auto_sort_axis:
            strategies = self._sort_by_axis(strategies)
            self._sort_log.append(
                f"Auto-sorted {len(strategies)} strategies by axis order. "
                f"Violations: {[(v[1], v[3]) for v in violations]}"
            )
            return strategies

        msg = "Axis order violation detected:\n"
        for i, n1, a1, n2, a2 in violations:
            msg += f"  [{i}] {n1}({a1}) precedes {n2}({a2}) — {a1} cannot come before {a2}\n"
        raise AxisOrderViolationException(msg, violations)

    @staticmethod
    def _sort_by_axis(strategies: List[Strategy]) -> List[Strategy]:
        return sorted(
            strategies,
            key=lambda s: AXIS_ORDER.get(s.axis.value if hasattr(s, 'axis') else "surface", 99),
        )

    # ================================================================
    # Combo unpacking
    # ================================================================

    def _unpack_combo(self, strategy, payload: AttackPayload) -> Tuple[AttackPayload, TraceLog]:
        combo_traces: TraceLog = []
        local_payload = payload.fork(stage="combo_internal")

        for i, sub_strategy in enumerate(strategy._sub_strategies):
            local_payload = local_payload.fork(stage=f"combo.step_{i + 1}")

            if hasattr(sub_strategy, 'apply_payload') and callable(getattr(sub_strategy, 'apply_payload')):
                local_payload, trace = sub_strategy.apply_payload(local_payload)
            else:
                current_case = local_payload.to_transformed_case()
                current_case, trace = self._apply_strategy(sub_strategy, current_case, None)
                local_payload.current_prompt = current_case.instruction
                local_payload.context = current_case.context
                local_payload.role = current_case.role

            trace.metadata["combo_parent"] = strategy.name
            trace.metadata["combo_step"] = i + 1
            combo_traces.append(trace)

        payload.current_prompt = local_payload.current_prompt
        payload.context = local_payload.context
        payload.role = local_payload.role
        payload.assembly_mode = local_payload.assembly_mode
        return payload, combo_traces

    # ================================================================
    # Internal helpers
    # ================================================================

    def _apply_strategy(self, strategy: Strategy, current_case: TransformedCase, budget) -> Tuple[TransformedCase, StrategyTrace]:
        try:
            current_case, trace = strategy.apply(current_case)
            return current_case, trace
        except Exception as e:
            trace = StrategyTrace(
                strategy_name=strategy.name,
                strategy_type=strategy.type.value,
                intensity=strategy.intensity,
                scope=strategy.scope.value,
                modification_type="error",
                description=f"Strategy {strategy.name} failed: {str(e)}",
            )
            return current_case, trace

    def _check_budget_exhausted(self, trace_log: TraceLog, budget: AttackBudget) -> bool:
        applied = [t for t in trace_log if t.modification_type not in ("skipped", "error")]
        if not budget:
            return False
        if len(applied) >= budget.max_strategies:
            return True
        cumulative_change = sum(
            t.token_change_ratio for t in applied if t.token_change_ratio is not None
        )
        if cumulative_change >= budget.max_modification_ratio:
            return True
        return False


def run_test_case(test_case: TestCase) -> Tuple[TransformedCase, TraceLog]:
    pipeline = Pipeline()
    return pipeline.run(test_case)
