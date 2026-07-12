"""
NCRT v3 — BasePlanner: 统一接口

所有 Planner 继承此类，实现 plan_turn() 方法。
Scheduler 每轮调用 plan_turn()，Planner 返回 TurnPlan。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from core.types import AttackResult, PlannerConfig
from core.generator import Generator
from core.judge import Judge
from core.memory import ConversationState, ExperienceMemory


@dataclass
class TurnPlan:
    """Planner 单轮策略输出."""
    prompt: str
    strategy: str = ""
    internal_calls: int = 1
    expected_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePlanner:
    """Planner 统一基类."""

    name: str = "base"

    def __init__(self, config: Optional[PlannerConfig] = None,
                 generator: Optional[Generator] = None,
                 judge: Optional[Judge] = None,
                 memory: Optional[ExperienceMemory] = None):
        self.config = config or PlannerConfig()
        self.generator = generator or Generator()
        self.judge = judge or Judge()
        self.memory = memory or ExperienceMemory()

    def plan_turn(self, goal: str, state: ConversationState,
                  round_num: int, internal_budget: int = 2) -> TurnPlan:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement plan_turn()"
        )

    # ── AttackState 辅助 ──
    #
    # 设计原则: AttackState 是"内容参考"，不是"策略指令"。
    #
    #   ✅ 影响 Prompt 内容: 聊什么话题、避开什么措辞、在哪个 persona 基础上构建
    #   ❌ 不改变策略逻辑: PAIR 仍然是 refine 循环, TAP 仍然是树搜索,
    #      Crescendo 仍然是渐进升级, 各自的算法骨架完全不变
    #
    #   AttackState 告诉 Planner "Victim 现在处于什么状态、哪些路走过了"，
    #   Planner 用自己的策略算法决定"基于这些信息，我下一步该怎么做"。

    @staticmethod
    def _get_attack_state(state: ConversationState) -> dict:
        """从 ConversationState.metadata 中提取 AttackState."""
        return state.metadata.get("attack_state", {})

    @staticmethod
    def _attack_state_hint(state: ConversationState) -> str:
        """读取 Scheduler 预先生成的纯客观态势摘要."""
        ctx = state.metadata.get("planner_context", "")
        return ctx if ctx else ""

    @staticmethod
    def _build_system(base_prompt: str, state: ConversationState) -> str:
        """构建完整 System Prompt: 策略定义 + 情报简报.

        情报简报是黑板上的纯客观事实——Planner 可以选用或忽略。
        """
        briefing = state.metadata.get("planner_context", "")
        if not briefing:
            return base_prompt
        return (
            base_prompt
            + "\n\n── INTELLIGENCE BRIEFING (objective facts, use at your discretion) ──\n"
            + briefing
            + "\n── END BRIEFING ──"
        )
