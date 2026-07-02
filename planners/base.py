"""
NCRT v3 — BasePlanner: 统一接口

所有 Planner 继承此类，实现 attack() 方法。
"""

from abc import ABC, abstractmethod
from typing import Optional
from core.types import AttackResult, PlannerConfig
from core.generator import Generator
from core.judge import Judge
from core.memory import ConversationState, ExperienceMemory


class BasePlanner(ABC):
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

    @abstractmethod
    def attack(self, goal: str) -> AttackResult:
        """执行攻击，返回统一 AttackResult."""
        ...

    def generate_prompt(self, goal: str, state: ConversationState,
                        round_num: int) -> str:
        """
        生成单条攻击提示词。各 Planner 覆写此方法，注入自身的策略逻辑。

        供 AttackScheduler 调用 —— 调度器维护共享的 ConversationState，
        每个 Planner 基于当前状态贡献一条 prompt。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_prompt()"
        )

    def _create_result(self, goal: str, success: bool,
                       state: ConversationState = None,
                       final_prompt: str = "",
                       final_response: str = "",
                       **meta) -> AttackResult:
        from core.types import Outcome
        return AttackResult(
            success=success,
            outcome=Outcome.SUCCESS if success else Outcome.FAILURE,
            planner=self.name,
            goal=goal,
            turns=state.history if state else [],
            best_score=state.best_score if state else 0.0,
            total_rounds=state.current_round if state else 0,
            final_prompt=final_prompt,
            final_response=final_response,
            metadata=meta,
        )
