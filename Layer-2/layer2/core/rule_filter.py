from typing import List, Set, Optional
from layer2.core.types import ExperienceRecord
from layer1.core.strategy import Strategy
from layer1.core.test_case import TestCase
from layer2.config import Layer2Config


class RuleFilter:
    """
    A 层: 硬约束过滤器。
    在所有决策（C 层或 B 层）之后执行，从候选策略集中移除违反规则的策略。
    """

    def __init__(self, config: Optional[Layer2Config] = None):
        self.config = config or Layer2Config()
        self._consecutive_count: dict = {}
        self._failed_on_instruction: dict = {}
        self._last_dimensions_used: Set[str] = set()
        self._last_all_refused: bool = False

    def filter(
        self,
        candidates: List[Strategy],
        test_case: TestCase,
        recent_history: Optional[List[ExperienceRecord]] = None,
    ) -> List[Strategy]:
        if not candidates:
            return []

        allowed: List[Strategy] = list(candidates)
        removed: List[str] = []

        # Rule 1: 上下文存在性 — context 为空时移除 scope=context 的策略
        if test_case.context is None:
            before = len(allowed)
            allowed = [
                s for s in allowed
                if s.scope.value != "context"
            ]
            if len(allowed) < before:
                removed.append(f"context_scope_removed_{before - len(allowed)}")

        # Rule 2: 连续重复上限 — 同一策略连续使用超过阈值时移除
        before = len(allowed)
        allowed = [
            s for s in allowed
            if self._get_consecutive_count(s.name) < self.config.max_consecutive_same_strategy
        ]
        if len(allowed) < before:
            removed.append(f"consecutive_limit_{before - len(allowed)}")

        # Rule 3: 同指令同策略失败禁止
        if self._failed_on_instruction:
            before = len(allowed)
            failed_set: set = set()
            for names in self._failed_on_instruction.values():
                failed_set.update(names)
            allowed = [s for s in allowed if s.name not in failed_set]
            if len(allowed) < before:
                removed.append(f"failed_on_instruction_{before - len(allowed)}")

        # Rule 4: 全维度失败回退 — 清空连续重复计数（允许同策略再试），但不撤销 Rule 3 禁令
        if self._last_all_refused and self.config.all_dimensions_failed_reset:
            self._last_all_refused = False
            self._consecutive_count.clear()
            removed.append("all_dimensions_failed_reset_temp")

        # Rule 5: 最少维度多样性 — 如果预算>=2 且只剩1个维度, 强制保留
        if (
            len(allowed) >= 2
            and candidates
            and test_case.attack_budget.max_strategies >= 2
        ):
            dims_present = set()
            for s in candidates:
                dims_present.add(s.type.value)
            if len(dims_present) >= 2:
                allowed_dims = set(s.type.value for s in allowed)
                if len(allowed_dims) < 2 and len(allowed) < len(candidates):
                    for s in candidates:
                        if s.type.value not in allowed_dims and s not in allowed:
                            allowed.append(s)
                            allowed_dims.add(s.type.value)
                            if len(allowed_dims) >= 2:
                                break

        return allowed

    def record_applied_round(
        self,
        applied_strategies: List[Strategy],
        instruction: str,
        all_refused: bool,
    ) -> None:
        strategy_names = [s.name for s in applied_strategies]

        for name in strategy_names:
            self._consecutive_count[name] = self._consecutive_count.get(name, 0) + 1

        for name in list(self._consecutive_count.keys()):
            if name not in strategy_names:
                self._consecutive_count[name] = 0

        self._last_dimensions_used = set(
            s.type.value for s in applied_strategies
        )

        if all_refused and applied_strategies:
            for s in applied_strategies:
                self._failed_on_instruction[instruction] = self._failed_on_instruction.get(instruction, set())
                self._failed_on_instruction[instruction].add(s.name)

        self._last_all_refused = all_refused

    def _get_consecutive_count(self, strategy_name: str) -> int:
        return self._consecutive_count.get(strategy_name, 0)

    def reset(self) -> None:
        """仅清空短期状态（连续计数、维度记录、all_refused flag）."""
        self._consecutive_count.clear()
        self._last_dimensions_used.clear()
        self._last_all_refused = False

    def full_reset(self) -> None:
        """清空所有状态，包括指令级失败记录."""
        self.reset()
        self._failed_on_instruction.clear()


"""
================================================================================
FILE: layer2/core/rule_filter.py
ROLE: A 层硬约束过滤器 — 最后一道安全门。

CLASSES:
  RuleFilter:
    config (Layer2Config)                    -- 阈值配置.
    _consecutive_count (dict)                -- 策略 → 连续使用次数.
    _failed_on_instruction (dict)            -- instruction → Set[strategy_name].
    _last_dimensions_used (Set[str])         -- 上一轮使用的维度.
    _last_all_refused (bool)                 -- 上一轮是否全维度失败.

    filter(candidates, test_case, recent_history) -> List[Strategy]:
      对候选策略集执行 5 条硬规则过滤:

      Rule 1 — 上下文存在性: context=None 时移除 scope=context 的策略.
      Rule 2 — 连续重复上限: 同一策略连续使用 >= max_consecutive_same_strategy 轮则移除.
      Rule 3 — 同指令失败禁止: 上轮同 instruction 且结果为 refused 的策略本轮移除.
      Rule 4 — 全维度失败回退: 上轮所有维度都 refused 时清空所有限制, 恢复全部候选.
      Rule 5 — 最少维度多样性: 如果预算允许且候选只剩1个维度, 从被过滤的策略中补回.

    record_applied_round(applied_strategies, instruction, all_refused) -> None:
      记录本轮最终应用的策略, 更新内部状态 (连续计数/失败记录/维度记录).

    _get_consecutive_count(strategy_name) -> int:
      查询策略的连续使用次数.

    reset() -> None:
      清空所有内部状态 (用于连续失败重置).

DESIGN NOTES:
  - A 层是确定性、零学习的, 所有规则都是布尔判断.
  - 执行顺序从宽到严: 先移除明确违规的, 再检查剩余的.
  - Rule 4 (全维度失败回退) 是逃生舱口, 防止系统陷入"所有策略都被禁止"的死锁.
  - Rule 5 确保采样覆盖至少 2 个扰动维度, 防止决策退化到单一维度.
================================================================================
"""
