import math
from typing import List, Optional, Dict, Tuple
from layer2.core.types import ExperienceRecord, StrategyStats
from layer2.core.experience_base import ExperienceBase
from layer1.core.strategy import Strategy, get_strategies_by_dimension, STRATEGY_REGISTRY
from layer2.config import Layer2Config


class StatisticalScorer:
    """
    B 层: 全局统计评分器。
    在 C 层无匹配经验时作为降级决策，对所有可用策略打分取 top-k。
    """

    def __init__(self, config: Optional[Layer2Config] = None):
        self.config = config or Layer2Config()
        self._strategy_stats: Dict[str, StrategyStats] = {}
        self._current_round: int = 0

    def score_strategies(
        self,
        candidates: List[Strategy],
        experience_base: ExperienceBase,
        safety_category: str = "",
    ) -> List[Tuple[Strategy, float]]:
        if not candidates:
            return []

        scored: List[Tuple[Strategy, float]] = []
        for strategy in candidates:
            score = self._compute_score(strategy, experience_base, safety_category)
            scored.append((strategy, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def top_k(
        self,
        candidates: List[Strategy],
        experience_base: ExperienceBase,
        k: int,
        safety_category: str = "",
    ) -> List[Strategy]:
        scored = self.score_strategies(candidates, experience_base, safety_category)
        return [s for s, _ in scored[:k]]

    def _compute_score(
        self,
        strategy: Strategy,
        experience_base: ExperienceBase,
        safety_category: str,
    ) -> float:
        # 1. success_rate
        success, failure, partial = experience_base.get_strategy_stats_by_category(
            safety_category, strategy.name
        )
        total = success + failure + partial
        if total >= self.config.min_samples_for_stats:
            success_rate = (success + 0.5 * partial) / total if total > 0 else 0.5
        else:
            success_rate = 0.5

        # 2. recency_bonus
        stats = self._strategy_stats.get(strategy.name)
        if stats and stats.last_success_round >= 0:
            rounds_since = experience_base.total_rounds - stats.last_success_round
            recency_bonus = 1.0 / (rounds_since + 1)
        else:
            recency_bonus = 0.0

        # 3. overuse_penalty
        if stats and stats.total_uses > 0:
            usage_ratio = stats.total_uses / max(1, experience_base.total_rounds)
            overuse_penalty = math.log(1 + usage_ratio)
        else:
            overuse_penalty = 0.0

        # 4. diversity_bonus
        dim = getattr(strategy, "axis", None)
        dim_key = dim.value if dim else strategy.type.value
        dim_uses = experience_base.get_dimension_uses(dim_key, recent_window=10)
        diversity_bonus = 1.0 / (dim_uses + 1)

        score = (
            self.config.alpha_success * success_rate
            + self.config.beta_recency * recency_bonus
            - self.config.gamma_overuse * overuse_penalty
            + self.config.delta_diversity * diversity_bonus
        )

        return score

    def update_stats(
        self,
        strategy_name: str,
        outcome: str,
        success_score: float,
    ) -> None:
        if strategy_name not in self._strategy_stats:
            self._strategy_stats[strategy_name] = StrategyStats(
                strategy_name=strategy_name,
            )
        stats = self._strategy_stats[strategy_name]
        stats.total_uses += 1
        if outcome == "success":
            stats.success_count += 1
            stats.consecutive_failures = 0
            stats.last_success_round = self._current_round
        elif outcome == "failure":
            stats.failure_count += 1
            stats.consecutive_failures += 1
        else:
            stats.partial_count += 1
            stats.consecutive_failures = 0

        n = stats.total_uses
        stats.avg_success_score = (
            stats.avg_success_score * (n - 1) + success_score
        ) / n

    def set_current_round(self, round_num: int) -> None:
        self._current_round = round_num

    def get_all_stats(self) -> Dict[str, StrategyStats]:
        return dict(self._strategy_stats)

    def get_top_strategies_by_dimension(
        self,
        candidates: List[Strategy],
        experience_base: ExperienceBase,
        dimensions: List[str],
        k_per_dim: int = 1,
        safety_category: str = "",
    ) -> List[Strategy]:
        result: List[Strategy] = []
        seen_names: set = set()

        for dim in dimensions:
            dim_strategies = [
                s for s in candidates
                if getattr(s, "axis", None) and s.axis.value == dim
            ]
            if not dim_strategies:
                continue
            scored = self.score_strategies(
                dim_strategies, experience_base, safety_category
            )
            for strategy, _ in scored[:k_per_dim]:
                if strategy.name not in seen_names:
                    result.append(strategy)
                    seen_names.add(strategy.name)

        return result


"""
================================================================================
FILE: layer2/core/statistical_scorer.py
ROLE: B 层全局统计评分器 — C 层降级决策。

CLASSES:
  StatisticalScorer:
    config (Layer2Config)               -- 评分权重参数.
    _strategy_stats (Dict[str,StrategyStats]) -- 每个策略的累计统计.

    score_strategies(candidates, experience_base, safety_category) -> List[(Strategy, float)]:
      对候选策略全量打分并排序。四项评分:

      1. success_rate = (success + 0.5*partial) / total
         per-category 统计, 样本不足时默认 0.5.
      2. recency_bonus = 1 / (rounds_since_last_success + 1)
         最近成功的策略获得加分。
      3. overuse_penalty = log(1 + usage_ratio)
         被频繁使用（全局）的策略受到惩罚。
      4. diversity_bonus = 1 / (recent_dim_uses + 1)
         近期使用少的维度获得加分。

    top_k(candidates, experience_base, k, safety_category) -> List[Strategy]:
      返回 top-k 策略.

    _compute_score(strategy, experience_base, safety_category) -> float:
      单独计算一个策略的得分.

    update_stats(strategy_name, outcome, success_score) -> None:
      更新 per-strategy 统计 (累计使用次数/成功数/失败数/平均得分).

    set_current_round(round_num) -> None:
      同步当前轮次号 (用于 recency_bonus 计算).

    get_all_stats() -> Dict[str, StrategyStats]:
      导出所有策略统计.

    get_top_strategies_by_dimension(candidates, experience_base, dimensions,
                                     k_per_dim, safety_category) -> List[Strategy]:
      按维度取 top-k, 用于 policy_sampler 的按维度选择模式.

DESIGN NOTES:
  - B 层永远可用（不需要 embedding），是 C 层的降级路径.
  - 评分函数的四项权重可通过 config 调整（alpha/beta/gamma/delta).
  - overuse_penalty 使用对数函数，确保惩罚边际递减——第一次多用惩罚重，继续多用惩罚增量小.
  - recency_bonus 使用倒数衰减，确保"最近有效"的信号随轮次自然衰减.
================================================================================
"""
