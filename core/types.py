"""
NCRT v3 — 统一数据类型

AttackResult: 所有 Planner 的统一返回格式
ConversationTurn: 单轮对话记录
PlannerConfig: Planner 配置
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class Outcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    round_num: int
    role: str                    # "attacker" | "victim"
    content: str                 # prompt 或 response
    score: Optional[float] = None       # Judge 评分 (仅 attacker turn)
    judge_reason: Optional[str] = None  # Judge 评语
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    """统一攻击结果"""
    success: bool
    outcome: Outcome = Outcome.FAILURE
    planner: str = ""                    # 使用的 Planner 名称
    goal: str = ""                       # 攻击目标
    turns: List[ConversationTurn] = field(default_factory=list)
    best_score: float = 0.0
    total_rounds: int = 0
    final_prompt: str = ""
    final_response: str = ""
    trajectory: List[Dict] = field(default_factory=list)  # 完整轨迹
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Planner.step() 的统一返回格式"""
    prompt: str
    response: str
    score: float
    reason: str
    status: str = "CONTINUE"         # "CONTINUE" | "HANDOFF"
    summary: str = ""                # 交接摘要 (仅 HANDOFF 时有意义)
    recommendation: List[str] = field(default_factory=list)  # 推荐的下一个 Planner
    is_internal: bool = False        # 内部步骤，不调用 victim，前端不应显示为攻击回合


@dataclass
class PlannerConfig:
    """Planner 通用配置"""
    max_rounds: int = 20
    success_threshold: float = 0.7
    failure_threshold: float = 0.3
    # Generator 参数
    attack_temperature: float = 0.8
    attack_max_tokens: int = 4096
    # Memory 参数
    max_history_turns: int = 2
