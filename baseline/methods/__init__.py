"""
Baseline Methods — 独立实现 (可插拔 Planner)

每个方法直接调用 LLM API，自行管理攻击循环。
通过 step() 接口供 Scheduler 逐轮调用。
"""

from baseline.methods.pair import PAIRBaseline
from baseline.methods.tap import TAPBaseline
from baseline.methods.crescendo import CrescendoBaseline
from baseline.methods.safe2harm import Safe2HarmBaseline

METHODS = {
    "pair": PAIRBaseline,
    "tap": TAPBaseline,
    "crescendo": CrescendoBaseline,
    "safe2harm": Safe2HarmBaseline,
}

__all__ = ["METHODS"]
