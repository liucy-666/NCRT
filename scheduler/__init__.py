"""
StrategyManager — HANDOFF 驱动调度系统

Planner 状态机决定终止, Scheduler 基于推荐/已用/全局分做选择。
"""

from scheduler.scheduler import StrategyManager, SchedulerConfig
