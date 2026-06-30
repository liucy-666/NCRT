"""
NCRT v3 — Planner 模块

四种 Planner，统一接口，可直接互换:
  CrescendoPlanner — 渐进式多轮越狱
  PAIRPlanner      — 迭代对抗攻击
  TAPPlanner       — 树搜索攻击
  SEMAPlanner      — 多 Agent 协同攻击
"""

from planners.base import BasePlanner
from planners.crescendo import CrescendoPlanner
from planners.pair import PAIRPlanner
from planners.tap import TAPPlanner
from planners.sema import SEMAPlanner
from planners.controller import BudgetController

PLANNERS = {
    "crescendo": CrescendoPlanner,
    "pair": PAIRPlanner,
    "tap": TAPPlanner,
    "sema": SEMAPlanner,
    "budget": BudgetController,
}


def get_planner(name: str, **kwargs) -> BasePlanner:
    """根据名称创建 Planner."""
    if name not in PLANNERS:
        raise ValueError(f"Unknown planner: {name}. Available: {list(PLANNERS.keys())}")
    return PLANNERS[name](**kwargs)
