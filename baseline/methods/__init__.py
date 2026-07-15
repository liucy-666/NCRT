"""
Baseline Methods — 独立实现，不经过 Graph Scheduler

每个方法直接调用 LLM API，自行管理攻击循环。
与 planners/ 目录的同名文件是独立的两套实现:
  - planners/*.py: 供 Graph Scheduler 调用
  - baseline/methods/*.py: 供 baseline 对比模式直接调用
"""

from baseline.methods.pair import PAIRBaseline
from baseline.methods.tap import TAPBaseline
from baseline.methods.crescendo import CrescendoBaseline
from baseline.methods.safe2harm import Safe2HarmBaseline
from baseline.methods.sema import SEMABaseline

METHODS = {
    "pair": PAIRBaseline,
    "tap": TAPBaseline,
    "crescendo": CrescendoBaseline,
    "safe2harm": Safe2HarmBaseline,
    "sema": SEMABaseline,
}

__all__ = ["METHODS"]
