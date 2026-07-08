"""
NCRT v3 — Selector: 阶段分层 Thompson Sampling 策略选择器

核心设计:
  - Key = (planner, goal_tier, phase) → Beta(α, β)
  - goal_tier 复用调度器已有的难度分级 (normal / hard / extreme)
  - Phase: early(1-3) / mid(4-8) / late(9+)
  - Reward: 本轮是否提升了 best_score

用法:
    selector = ThompsonSelector()
    chosen = selector.select(roster, goal_tier, round_num=1)
    selector.reward_round(planner, goal_tier, round_num, score, best_before)
    should, reason, alt = selector.should_switch(current, goal_tier, round_num,
                                                  score, best_score, roster)
"""

import random
from typing import List, Dict, Tuple


# ─── Phase 定义 ───

def get_phase(round_num: int) -> str:
    if round_num <= 3:
        return "early"
    elif round_num <= 8:
        return "mid"
    else:
        return "late"


ALL_PHASES = ["early", "mid", "late"]


# ─── Phase-Stratified Thompson Bandit ───

class PhaseStratifiedBandit:
    """
    阶段分层 Beta 分布。

    Key = "planner|goal_tier|phase" → (alpha, beta)
      alpha = 本轮提升 best_score 的次数 + 1
      beta  = 本轮未提升的次数 + 1
    """

    def __init__(self):
        self._params: Dict[str, Tuple[float, float]] = {}

    def _key(self, planner: str, goal_tier: str, phase: str) -> str:
        return f"{planner}|{goal_tier}|{phase}"

    def sample(self, planner: str, goal_tier: str, phase: str) -> float:
        alpha, beta = self._params.get(
            self._key(planner, goal_tier, phase), (1.0, 1.0))
        return random.betavariate(alpha, beta)

    def update(self, planner: str, goal_tier: str, phase: str,
               improved: bool):
        key = self._key(planner, goal_tier, phase)
        alpha, beta = self._params.get(key, (1.0, 1.0))
        if improved:
            alpha += 1.0
        else:
            beta += 1.0
        self._params[key] = (alpha, beta)

    def get_stats(self, planner: str, goal_tier: str, phase: str) -> Dict:
        key = self._key(planner, goal_tier, phase)
        alpha, beta = self._params.get(key, (1.0, 1.0))
        trials = alpha + beta - 2
        return {
            "improvements": int(alpha - 1),
            "stalls": int(beta - 1),
            "trials": int(trials),
            "improve_rate": (alpha - 1) / max(1, trials),
        }

    def all_stats(self) -> Dict:
        result = {}
        for key, (alpha, beta) in self._params.items():
            parts = key.split("|")
            if len(parts) == 3:
                p, gt, ph = parts
                trials = alpha + beta - 2
                result.setdefault(gt, {}).setdefault(ph, {})[p] = {
                    "improvements": int(alpha - 1),
                    "stalls": int(beta - 1),
                    "trials": int(trials),
                    "improve_rate": (alpha - 1) / max(1, trials),
                }
        return result


# ─── Thompson Selector ───

class ThompsonSelector:
    """
    阶段分层 Thompson Sampling 选择器。

    适配 NCRT 调度器，复用 goal_tier (normal/hard/extreme) 作为分类维度。
    """

    def __init__(self):
        self.bandit = PhaseStratifiedBandit()

    # ── 选择 ──

    def select(self, planners: List[str], goal_tier: str,
               round_num: int = 1, top_k: int = 1) -> List[str]:
        phase = get_phase(round_num)
        total = self._total_trials(goal_tier, phase)

        if total < len(planners) * 2:
            # 冷启动: 随机排列探索
            shuffled = list(planners)
            random.shuffle(shuffled)
            return shuffled[:top_k]

        # Thompson Sampling
        samples = [(p, self.bandit.sample(p, goal_tier, phase))
                   for p in planners]
        samples.sort(key=lambda x: -x[1])
        return [p for p, _ in samples[:top_k]]

    # ── 逐轮 reward ──

    def reward_round(self, planner: str, goal_tier: str, round_num: int,
                     score: float, best_before: float):
        phase = get_phase(round_num)
        improved = score > best_before
        self.bandit.update(planner, goal_tier, phase, improved)

    # ── Episode 级 reward ──

    def reward_episode(self, planner: str, goal_tier: str,
                       scores: List[Tuple[int, float]]):
        """scores: [(round_num, score), ...]"""
        best_so_far = 0.0
        for rnd, score in scores:
            phase = get_phase(rnd)
            improved = score > best_so_far
            if improved:
                best_so_far = score
            self.bandit.update(planner, goal_tier, phase, improved)

    # ── 切换判断 ──

    def should_switch(self, current_planner: str, goal_tier: str,
                      round_num: int, score: float,
                      best_score: float, roster: List[str],
                      consecutive_improvements: int = 0
                      ) -> Tuple[bool, str, str]:
        """
        纯统计推断: 基于历史数据判断当前策略是否劣于替代策略。

        注意: 调用方需要自行检查 per-strategy warmup + 最低步数。
        此方法仅做纯粹的 TS 统计推断，不做任何强制判断（如 round≥6 强制切）。
        """
        phase = get_phase(round_num)

        # 高分继续
        if score >= 0.7:
            return False, "", current_planner

        # 太早不切
        if round_num < 3:
            return False, "", current_planner

        # 稳步前进保护
        if consecutive_improvements >= 2:
            return False, "", current_planner

        # 需要足够数据
        self_stats = self.bandit.get_stats(current_planner, goal_tier, phase)
        if self_stats["trials"] < 2:
            return False, "", current_planner

        own_rate = self_stats["improve_rate"]

        # 自身推进率还行就不切
        if own_rate >= 0.3:
            return False, "", current_planner

        # 找最佳替代
        best_alt = current_planner
        best_alt_rate = 0.0
        for alt in roster:
            if alt == current_planner:
                continue
            alt_stats = self.bandit.get_stats(alt, goal_tier, phase)
            if alt_stats["trials"] >= 2:
                if alt_stats["improve_rate"] > best_alt_rate:
                    best_alt_rate = alt_stats["improve_rate"]
                    best_alt = alt

        # 切换条件: 自身推进率差 + 替代明显更好 + 当前轮确认停滞
        stalled = score <= best_score
        if own_rate < 0.3 and best_alt_rate > 0.4 and stalled:
            return True, (
                f"Phase [{phase}]: {current_planner} "
                f"improve_rate={own_rate:.0%} "
                f"vs {best_alt} improve_rate={best_alt_rate:.0%}"
            ), best_alt

        return False, "", current_planner

    # ── 统计 ──

    def get_statistics(self) -> Dict:
        return self.bandit.all_stats()

    def print_stats(self):
        stats = self.bandit.all_stats()
        if not stats:
            print("  [TS] No data collected yet.")
            return
        print(f"\n  {'='*80}")
        print(f"  PHASE-STRATIFIED THOMPSON SAMPLING STATISTICS")
        print(f"  {'='*80}")
        for gt in sorted(stats):
            print(f"\n  [{gt.upper()}]")
            for phase in ALL_PHASES:
                if phase not in stats[gt]:
                    continue
                rng = {"early": "1-3", "mid": "4-8", "late": "9+"}[phase]
                print(f"\n    --- {phase.upper()} (rounds {rng}) ---")
                print(f"    {'Planner':<15s} {'Impr':>5s} {'Stall':>6s} "
                      f"{'Total':>6s} {'Rate':>7s}")
                print(f"    {'-'*15} {'-'*5} {'-'*6} {'-'*6} {'-'*7}")
                for p in stats[gt][phase]:
                    s = stats[gt][phase][p]
                    print(f"    {p:<15s} {s['improvements']:>4d}  "
                          f"{s['stalls']:>5d}  {s['trials']:>5d}  "
                          f"{s['improve_rate']:>6.1%}")
        print(f"\n  {'='*80}")

    def _total_trials(self, goal_tier: str, phase: str) -> int:
        total = 0
        for p in ["crescendo", "pair", "tap", "sema", "icrt", "safe2harm"]:
            s = self.bandit.get_stats(p, goal_tier, phase)
            total += s["trials"]
        return total
