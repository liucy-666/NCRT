import random as _random
from typing import List, Optional, Tuple
from layer1.core.test_case import TestCase, TransformedCase
from layer1.core.strategy import (
    Strategy,
    StrategyType,
    STRATEGY_REGISTRY,
    STRATEGY_META,
    get_strategy_dimension,
    get_strategy_meta,
)
from layer2.core.types import (
    ExperienceRecord,
    TextFeatures,
    PerturbationVector,
    ResponseState,
    OutcomeType,
)
from layer2.core.embeddings import EmbeddingClient
from layer2.core.experience_base import ExperienceBase
from layer2.core.rule_filter import RuleFilter
from layer2.core.statistical_scorer import StatisticalScorer
from layer2.core.rewards import StubRewardFunction
from layer2.config import Layer2Config


class PolicySampler:
    """
    Layer 2 主入口: 策略选择器。
    执行 C→B→A 三级决策链:
      C 层 (PRIMARY):   基于原始有害文本的 embedding 检索历史经验
      B 层 (FALLBACK):  无相似经验时降级为全局统计评分
      A 层 (ALWAYS):    规则硬约束过滤最终候选

    故障熔断: 每条指令有最大轮数和连续失败上限，防止死循环。
    """

    def __init__(self, config: Optional[Layer2Config] = None, reward=None):
        self.config = config or Layer2Config()
        self._experience_base = ExperienceBase(self.config)
        self._embedding_client = EmbeddingClient(self.config)
        self._rule_filter = RuleFilter(self.config)
        self._scorer = StatisticalScorer(self.config)
        self._reward = reward if reward is not None else StubRewardFunction(self.config)
        self._current_round: int = 0
        self._consecutive_failures: int = 0
        self._instruction_rounds: int = 0
        self._current_instruction: str = ""
        self._attack_terminated: bool = False
        self._termination_reason: str = ""
        self._tried_strategies: set = set()
        self._last_outcome: str = ""
        self._last_allowed_axes: set = {"search", "representation", "surface"}

    @property
    def experience_base(self) -> ExperienceBase:
        return self._experience_base

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def instruction_rounds(self) -> int:
        return self._instruction_rounds

    @property
    def attack_terminated(self) -> bool:
        return self._attack_terminated

    @property
    def termination_reason(self) -> str:
        return self._termination_reason

    def select(self, test_case: TestCase) -> List[Strategy]:
        """
        为给定的 TestCase 选择一组策略。
        三级决策流程: C → B → A

        如果当前攻击已终止或已超轮数，返回空列表。
        调用方应检查 attack_terminated 属性。
        """
        if self._attack_terminated:
            return []

        if not self._should_continue_for_instruction(test_case):
            return []

        self._instruction_rounds += 1

        # Step 0: 构建候选策略池
        allowed_axes = getattr(test_case.attack_budget, "allowed_axes", {"search", "representation", "surface"})
        self._last_allowed_axes = allowed_axes
        candidates = self._build_candidate_pool(allowed_axes)

        if not candidates:
            return []

        budget = test_case.attack_budget.max_strategies

        # Step 1: C 层 — 案例检索
        c_selected, c_source = self._c_layer_select(
            test_case, candidates, budget
        )

        if c_selected:
            selected = c_selected
            decision_source = f"c_layer({c_source})"
        else:
            # Step 2: B 层 — 全局统计降级
            b_selected = self._b_layer_select(
                test_case, candidates, budget
            )
            selected = b_selected
            decision_source = "b_layer"

        # Step 3: A 层 — 规则硬约束
        filtered = self._rule_filter.filter(
            selected, test_case
        )

        # 如果 A 层过滤后为空，回退到 B 层的全量 top-k
        if not filtered and selected:
            all_candidates = self._build_candidate_pool(allowed_axes)
            b_fallback = self._b_layer_select(
                test_case, all_candidates, budget
            )
            filtered = self._rule_filter.filter(b_fallback, test_case)

        # 记录本次选择（延迟到收到反馈时再写 record_applied_round）
        self._last_selected = filtered
        self._last_decision_source = decision_source

        return filtered

    def update(
        self,
        test_case: TestCase,
        applied_strategies: List[Strategy],
        response_state: ResponseState,
    ) -> None:
        """
        接收一轮反馈, 更新经验库、统计和策略覆盖追踪。

        applied_strategies 应传入本轮实际应用的策略（非 sampler 选出的全部）。
        可从 trace_log 中提取： [t for t in trace_log if t.modification_type not in ("skipped", "error")]
        """
        self._current_round += 1

        # 评估
        score = self._reward.evaluate(response_state)
        outcome = self._reward.outcome(score)
        self._last_outcome = outcome

        # 从 strategy_trace 中提取实际应用的策略（优先于 applied_strategies 参数）
        actual_names: list = []
        if response_state.strategy_trace:
            actual_names = [
                t.get("strategy_name", "")
                for t in response_state.strategy_trace
                if t.get("modification_type", "") not in ("skipped", "error")
            ]
        if not actual_names and applied_strategies:
            actual_names = [s.name for s in applied_strategies]

        # 追踪本轮实际应用的策略
        for name in actual_names:
            self._tried_strategies.add(name)

        # 提取扰动向量
        pv = response_state.perturbation_vector

        # 提取文本特征
        text = test_case.instruction
        features = TextFeatures(
            text_length=len(text),
            word_count=len(text.split()),
            safety_category=test_case.metadata.get("safety_category", ""),
        )

        # 提取 embedding
        embedding = self._embedding_client.embed(text)

        # 构建经验记录（仅记录实际应用的策略）
        strategy_dims = list(set(
            s.type.value for s in applied_strategies
            if s.name in actual_names
        )) if applied_strategies else []

        record = ExperienceRecord(
            original_text=text,
            text_embedding=embedding,
            text_features=features,
            strategy_combination=actual_names,
            strategy_dimensions=strategy_dims,
            outcome=outcome,
            response_label=response_state.response_label,
            success_score=score,
            perturbation_vector=pv,
        )

        self._experience_base.add(record)

        # 更新 B 层统计
        for name in actual_names:
            self._scorer.update_stats(name, outcome, score)

        # 更新 A 层状态（仅记录实际应用的策略）
        applied_actual = [
            s for s in applied_strategies if s.name in actual_names
        ] if applied_strategies else []
        all_refused = (
            all(
                response_state.response_label == "refused"
                for _ in strategy_dims
            ) or (outcome == "failure")
        ) if strategy_dims else (outcome == "failure")
        self._rule_filter.record_applied_round(
            applied_actual,
            test_case.instruction,
            all_refused,
        )

        # 全局连续失败追踪
        if outcome == "failure":
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        if self._consecutive_failures >= self.config.reset_after_consecutive_failures:
            self._rule_filter.reset()
            self._consecutive_failures = 0

        # 探索率退火
        if self.config.decay_exploration:
            self.config.exploration_rate *= self.config.decay_gamma

    def reset_instruction_context(self, new_instruction: str = "") -> None:
        """
        切换到新指令时调用，重置指令级计数器。
        由调用方（攻击循环控制器）在新一轮开始时调用。
        """
        self._instruction_rounds = 0
        self._current_instruction = new_instruction
        self._attack_terminated = False
        self._termination_reason = ""
        self._tried_strategies.clear()
        self._last_outcome = ""
        self._rule_filter.full_reset()

    def should_continue_attack(self) -> bool:
        """
        检查当前攻击是否可以继续。

        Terminates when:
          - max_rounds_per_instruction reached (绝对上限)
          - strategy_exhaustion_ratio reached (已试遍大部分策略)
        """
        if self._attack_terminated:
            return False

        allowed = self._last_allowed_axes or {"search", "representation", "surface"}
        total_available = self._count_strategies_in_axes(allowed)
        tried_ratio = len(self._tried_strategies) / max(1, total_available)

        if tried_ratio >= self.config.strategy_exhaustion_ratio:
            self._attack_terminated = True
            self._termination_reason = (
                f"strategy_exhaustion({len(self._tried_strategies)}/{total_available} "
                f"= {tried_ratio:.1%} >= {self.config.strategy_exhaustion_ratio:.0%})"
            )
            return False

        if self._instruction_rounds >= self.config.max_rounds_per_instruction:
            self._attack_terminated = True
            self._termination_reason = (
                f"max_rounds_per_instruction_exceeded("
                f"{self._instruction_rounds} >= "
                f"{self.config.max_rounds_per_instruction})"
            )
            return False

        return True

    def _should_continue_for_instruction(self, test_case: TestCase) -> bool:
        instruction_text = test_case.instruction

        if self._current_instruction != instruction_text:
            self.reset_instruction_context(instruction_text)
            return True

        if self._attack_terminated:
            return False

        if self._instruction_rounds >= self.config.max_rounds_per_instruction:
            self._attack_terminated = True
            self._termination_reason = (
                f"max_rounds_per_instruction_exceeded"
            )
            return False

        return True

    def _c_layer_select(
        self,
        test_case: TestCase,
        candidates: List[Strategy],
        budget: int,
    ) -> Tuple[List[Strategy], str]:
        """
        C 层: 基于原始有害文本 embedding 检索最相似的历史经验,
        复用成功经验的策略组合。
        """
        if self._experience_base.total_records < self.config.min_experiences_for_c:
            return [], ""

        results = self._experience_base.search_with_preference(
            test_case.instruction,
            prefer_outcome="success",
        )

        if not results:
            return [], ""

        best_record, best_similarity = results[0]

        strategy_names = best_record.strategy_combination

        selected: List[Strategy] = []
        for name in strategy_names:
            if name in STRATEGY_REGISTRY and len(selected) < budget:
                cls = STRATEGY_REGISTRY[name]
                try:
                    s = cls.__new__(cls)
                    s.__init__(
                        intensity=min(0.5, test_case.attack_budget.max_intensity),
                        seed=test_case.seed,
                    )
                    selected.append(s)
                except Exception:
                    continue

        source = f"sim={best_similarity:.3f},record_round={best_record.round_index},outcome={best_record.outcome}"
        return selected, source

    def _b_layer_select(
        self,
        test_case: TestCase,
        candidates: List[Strategy],
        budget: int,
    ) -> List[Strategy]:
        """
        B 层: 全局统计评分降级.
        按维度取 top-1 直到达到 budget 上限, 策略分数 × 权重后排序.
        维度优先级: representation > surface > search (结构化/编码策略优先于LLM搜索策略).
        """
        safety_category = test_case.metadata.get("safety_category", "")
        dims = list(getattr(test_case.attack_budget, "allowed_axes", {"search", "representation", "surface"}))

        AXIS_PRIORITY = {"representation": 0, "surface": 1, "search": 2}
        dims = sorted(dims, key=lambda d: AXIS_PRIORITY.get(d, 99))

        self._scorer.set_current_round(self._current_round)

        selected = self._scorer.get_top_strategies_by_dimension(
            candidates=candidates,
            experience_base=self._experience_base,
            dimensions=dims,
            k_per_dim=1,
            safety_category=safety_category,
        )

        # ── Weight boost: 按 selection_weight 重排 ──
        def _weighted_score(s: Strategy) -> float:
            meta = get_strategy_meta(s.name)
            w = meta.get("selection_weight", 1.0)
            # 基础分 = 1.0，权重加成
            return w

        selected = sorted(selected, key=_weighted_score, reverse=True)
        return selected[:budget]

    def _build_candidate_pool(self, allowed_axes: set) -> List[Strategy]:
        """
        构建候选策略池，尊重 selection_weight 和 deprecated 标记。

        - deprecated 策略仅在探索轮次有概率被选中（exploration_rate）
        - 非 deprecated 策略按 selection_weight 决定保留概率
        - weight=1.0 的策略始终保留
        """
        pool: List[Strategy] = []
        for name, cls in STRATEGY_REGISTRY.items():
            try:
                dummy = cls.__new__(cls)
                dummy.__init__()
                axis_val = getattr(dummy, "axis", None)
                if not axis_val or axis_val.value not in allowed_axes:
                    continue

                meta = get_strategy_meta(name)
                weight = meta.get("selection_weight", 1.0)
                deprecated = meta.get("deprecated", False)

                # ── Deprecated 策略: 仅在探索轮次中保留 ──
                if deprecated:
                    if _random.random() < self.config.exploration_rate * weight:
                        pass  # 探索性保留
                    else:
                        continue  # 跳过

                # ── 非 deprecated 策略: 按权重概率保留 ──
                if not deprecated and weight < 1.0:
                    if _random.random() > weight:
                        continue

                instance = cls.__new__(cls)
                instance.__init__(intensity=0.5)
                pool.append(instance)
            except Exception:
                continue
        return pool

    def _count_strategies_in_axes(self, allowed_axes: set) -> int:
        count = 0
        for _name, cls in STRATEGY_REGISTRY.items():
            try:
                dummy = cls.__new__(cls)
                dummy.__init__()
                axis_val = getattr(dummy, "axis", None)
                if axis_val and axis_val.value in allowed_axes:
                    count += 1
            except Exception:
                continue
        return count

    def get_policy_report(self) -> dict:
        total_available = self._count_strategies_in_axes(
            self._last_allowed_axes
        )
        return {
            "current_round": self._current_round,
            "consecutive_failures": self._consecutive_failures,
            "instruction": {
                "text": self._current_instruction[:100],
                "rounds": self._instruction_rounds,
                "tried_strategies": sorted(self._tried_strategies),
                "tried_count": len(self._tried_strategies),
                "total_available": total_available,
                "exhaustion_ratio": len(self._tried_strategies) / max(1, total_available),
                "last_outcome": self._last_outcome,
                "attack_terminated": self._attack_terminated,
                "termination_reason": self._termination_reason,
            },
            "experience_base": self._experience_base.to_report(),
            "strategy_stats": {
                name: {
                    "total_uses": s.total_uses,
                    "success_rate": (
                        s.success_count / s.total_uses if s.total_uses > 0 else 0.0
                    ),
                    "avg_score": s.avg_success_score,
                    "consecutive_failures": s.consecutive_failures,
                }
                for name, s in self._scorer.get_all_stats().items()
            },
            "rule_filter_state": {
                "consecutive_counts": dict(self._rule_filter._consecutive_count),
            },
        }


"""
================================================================================
FILE: layer2/policy_sampler.py
ROLE: Layer 2 主编排器 — C→B→A 三级决策链入口.

CLASSES:
  PolicySampler:
    config (Layer2Config)                        -- 所有可调参数.
    _experience_base (ExperienceBase)            -- C 层: 经验库.
    _embedding_client (EmbeddingClient)          -- C 层: embedding 提取.
    _rule_filter (RuleFilter)                    -- A 层: 规则过滤.
    _scorer (StatisticalScorer)                  -- B 层: 统计评分.
    _reward (StubRewardFunction)                 -- stub 奖励函数.
    _current_round (int)                         -- 全局累计轮次.
    _consecutive_failures (int)                  -- 全局连续失败计数.
    _instruction_rounds (int)                    -- 当前指令攻击轮数.
    _current_instruction (str)                   -- 当前攻击的指令文本.
    _attack_terminated (bool)                    -- 当前攻击是否已终止.
    _termination_reason (str)                    -- 终止原因描述.
    _tried_strategies (set)                      -- 当前指令已尝试的策略名集合.
    _last_outcome (str)                          -- 上一轮结果 (success/partial/failure).
    _last_allowed_dimensions (set)               -- 上一个 test_case 的 allowed_dimensions.
    _last_selected (List[Strategy])              -- 本轮最终选择的策略.
    _last_decision_source (str)                  -- 本轮决策来源标记.

  PROPERTIES:
    experience_base -> ExperienceBase            -- 只读属性.
    current_round -> int
    instruction_rounds -> int
    attack_terminated -> bool
    termination_reason -> str

  PUBLIC METHODS:
    select(test_case: TestCase) -> List[Strategy]:
      主决策流程 (C→B→A). 若攻击已终止返回空列表.

    update(test_case, applied_strategies, response_state) -> None:
      一轮反馈处理: 评估→记录经验→追踪策略覆盖→更新统计.

    reset_instruction_context(new_instruction: str) -> None:
      切换到新指令时调用, 重置指令级计数器与已尝试策略集合.

    should_continue_attack() -> bool:
      检查当前攻击是否可继续 (防死循环).

    get_policy_report() -> dict:
      导出完整状态报告.

  故障熔断条件:
    1. max_rounds_per_instruction (默认10): 单指令绝对最大攻击轮数.
    2. strategy_exhaustion_ratio (默认80%): 已尝试策略占可用策略的比例,
        超过此阈值后终止, 保证有 10 种策略时不会在 3 轮就放弃.

  调用方的标准循环:
    sampler.reset_instruction_context(instruction)
    while sampler.should_continue_attack():
        transformed, trace = pipeline.run(case)
        response = model(transformed.full_input)
        sampler.update(case, sampler._last_selected, response_state)

DESIGN NOTES:
  - 这是 Layer 2 的唯一入口类。Layer 1 的 Pipeline 只需要将 RoundRobinSampler
    替换为 PolicySampler 实例。
  - select() 和 update() 是独立的两个调用: select 给 Layer 1 提供策略,
    update 在收到反馈后更新内部状态。
  - C 层是 PRIMARY, B 层是 FALLBACK, A 层是 ALWAYS — 优先级和执行顺序不同。
  - 策略的 intensity 由 test_case.attack_budget 控制, Layer 2 不介入。
================================================================================
"""
