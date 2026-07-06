"""
NCRT v3 — Planner 模块

六种 Planner，统一接口，可直接互换:
  CrescendoPlanner  — 渐进式多轮越狱
  PAIRPlanner       — 迭代对抗攻击
  TAPPlanner        — 树搜索攻击
  SEMAPlanner       — 单智能体反思攻击
  ICRTPlanner       — 认知分解攻击 (ICML 2025)
  Safe2HarmPlanner  — 语义同构攻击
"""

from planners.base import BasePlanner, TurnPlan
from planners.crescendo import CrescendoPlanner
from planners.pair import PAIRPlanner
from planners.tap import TAPPlanner
from planners.sema import SEMAPlanner
from planners.icrt import ICRTPlanner
from planners.safe2harm import Safe2HarmPlanner

PLANNERS = {
    "crescendo": CrescendoPlanner,
    "pair": PAIRPlanner,
    "tap": TAPPlanner,
    "sema": SEMAPlanner,
    "icrt": ICRTPlanner,
    "safe2harm": Safe2HarmPlanner,
    "graph": None,  # 特殊: AttackScheduler, 在 run.py 中处理
}


def get_planner(name: str, **kwargs) -> BasePlanner:
    """根据名称创建 Planner."""
    if name not in PLANNERS:
        raise ValueError(f"Unknown planner: {name}. Available: {list(PLANNERS.keys())}")
    return PLANNERS[name](**kwargs)
