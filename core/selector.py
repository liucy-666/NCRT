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

    def _key(self, planner: str, goal_tier: str, phase: str,
             stage: str = "") -> str:
        base = f"{planner}|{goal_tier}|{phase}"
        return f"{base}|{stage}" if stage else base

    # ── 跨攻击经验持久化 ──

    def save(self, path: str) -> int:
        """保存 Beta 参数到 JSON 文件。返回保存的 key 数量。"""
        import json, os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {k: [round(a, 3), round(b, 3)] for k, (a, b) in self._params.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return len(data)

    def load(self, path: str) -> int:
        """从 JSON 文件加载 Beta 参数。返回加载的 key 数量。"""
        import json, os
        if not os.path.exists(path):
            return 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key, (alpha, beta) in data.items():
            self._params[key] = (float(alpha), float(beta))
        return len(data)

    def sample(self, planner: str, goal_tier: str, phase: str,
               stage: str = "") -> float:
        alpha, beta = self._params.get(
            self._key(planner, goal_tier, phase, stage), (1.0, 1.0))
        return random.betavariate(alpha, beta)

    def update_continuous(self, planner: str, goal_tier: str, phase: str,
                          reward: float, stage: str = ""):
        """连续奖励: reward ∈ [0, 1].

        α += reward      (越接近 1 → 越成功)
        β += (1-reward)  (越接近 0 → 越失败)
        """
        key = self._key(planner, goal_tier, phase, stage)
        alpha, beta = self._params.get(key, (1.0, 1.0))
        alpha += max(0.0, min(1.0, reward))
        beta += max(0.0, min(1.0, 1.0 - reward))
        self._params[key] = (alpha, beta)

    def decay_all(self, rate: float = 0.95):
        """全局时间衰减: 所有参数向 (1,1) 回归.

        α' = 1 + (α-1) * rate
        β' = 1 + (β-1) * rate
        旧经验逐渐淡出，最近 ~20 条 goal 主导决策.
        """
        if rate >= 1.0:
            return
        for key in list(self._params.keys()):
            alpha, beta = self._params[key]
            alpha = 1.0 + (alpha - 1.0) * rate
            beta = 1.0 + (beta - 1.0) * rate
            if alpha <= 1.01 and beta <= 1.01:
                del self._params[key]  # 衰减到接近无信息 → 删除
            else:
                self._params[key] = (alpha, beta)

    def get_stats(self, planner: str, goal_tier: str, phase: str,
                  stage: str = "") -> Dict:
        key = self._key(planner, goal_tier, phase, stage)
        alpha, beta = self._params.get(key, (1.0, 1.0))
        trials = alpha + beta - 2
        return {
            "alpha": round(alpha, 2),
            "beta": round(beta, 2),
            "trials": int(trials),
            "mean": alpha / max(1, alpha + beta),
        }

    def all_stats(self) -> Dict:
        result = {}
        for key, (alpha, beta) in self._params.items():
            parts = key.split("|")
            if len(parts) >= 3:
                p, gt, ph = parts[0], parts[1], parts[2]
                posture = parts[3] if len(parts) >= 4 else "-"
                trials = alpha + beta - 2
                stat = {
                    "alpha": round(alpha, 2),
                    "beta": round(beta, 2),
                    "trials": int(trials),
                    "mean": alpha / max(1, alpha + beta),
                }
                result.setdefault(gt, {}).setdefault(ph, {}).setdefault(p, {})[posture] = stat
        return result


# ─── Thompson Selector ───

class ThompsonSelector:
    """
    阶段分层 Thompson Sampling 选择器。

    适配 NCRT 调度器，复用 goal_tier (normal/hard/extreme) 作为分类维度。
    """

    def __init__(self):
        self.bandit = PhaseStratifiedBandit()

    # ── 持久化 ──

    def save(self, path: str) -> int:
        """保存 TS 经验到文件。"""
        return self.bandit.save(path)

    def load(self, path: str) -> int:
        """从文件加载 TS 经验。"""
        return self.bandit.load(path)

    # ── 选择 ──

    def select(self, planners: List[str], goal_tier: str,
               round_num: int = 1, top_k: int = 1,
               stage: str = "") -> List[str]:
        phase = get_phase(round_num)
        total = self._total_trials(planners, goal_tier, phase, stage)

        if total < len(planners) * 2:
            # 数据不足: 6 个 Planner 等权重随机探索
            shuffled = list(planners)
            random.shuffle(shuffled)
            return shuffled[:top_k]

        # Thompson Sampling (state-aware: stage 加入 key)
        samples = [(p, self.bandit.sample(p, goal_tier, phase, stage))
                   for p in planners]
        samples.sort(key=lambda x: -x[1])
        return [p for p, _ in samples[:top_k]]

    # ── 从黑板学习 (唯一活跃的奖励入口) ──

    def reward_from_blackboard(self, planner_calls: dict,
                                max_progress: float, goal_tier: str,
                                posture: str, decay: float = 0.95):
        """从战术黑板读取的信号更新 TS 参数.

        Args:
          planner_calls: {planner_name: [(round_num, score), ...]}
          max_progress:  AttackState.max_progress (最终突破分数, 0-1)
          goal_tier:     难度分级 (normal/hard/extreme)
          posture:       最终受害者姿态
          decay:         全局时间衰减率 (0.95 → 最近 ~20 条主导)

        设计:
          1. 先全局衰减 → 旧经验淡出
          2. 对攻击链上每个 Planner, 奖励 = max_progress × 位置权重
             - 越靠近链尾 (接近突破) → 权重越高 (0.5-1.0)
             - 连续奖励 α += reward, β += (1-reward)
        """
        # 1. 全局时间衰减
        if decay < 1.0:
            self.bandit.decay_all(decay)

        if not planner_calls:
            return

        # 2. 攻击链排序 (按轮数)
        chain = sorted(planner_calls.items(),
                       key=lambda kv: max(r for r, _ in kv[1]) if kv[1] else 0)
        n = len(chain)
        if n == 0:
            return

        for i, (planner, rounds) in enumerate(chain):
            if not rounds:
                continue
            # 位置权重: 链尾 1.0, 链首 0.5, 线性插值
            position_weight = 0.5 + 0.5 * (i + 1) / n
            reward = max_progress * position_weight

            # 对每一轮, 写入对应 phase 和 posture
            for rnd, score in rounds:
                phase = get_phase(rnd)
                self.bandit.update_continuous(planner, goal_tier, phase,
                                              reward, posture)

            print(f"  [TS learn] {planner}: pos_w={position_weight:.2f} "
                  f"reward={reward:.3f} (max_p={max_progress:.2f})", flush=True)

    # ── 统计 ──

    def get_statistics(self) -> Dict:
        return self.bandit.all_stats()

    def print_stats(self):
        stats = self.bandit.all_stats()
        if not stats:
            print("  [TS] No data collected yet.")
            return
        print(f"\n  {'='*80}")
        print(f"  PHASE-STRATIFIED THOMPSON SAMPLING (posture-aware)")
        print(f"  {'='*80}")
        for gt in sorted(stats):
            print(f"\n  [{gt.upper()}]")
            for phase in ALL_PHASES:
                if phase not in stats[gt]:
                    continue
                rng = {"early": "1-3", "mid": "4-8", "late": "9+"}[phase]
                print(f"\n    --- {phase.upper()} (rounds {rng}) ---")
                print(f"    {'Planner':<12s} {'Posture':<16s} {'Alpha':>6s} {'Beta':>6s} {'Mean':>6s}")
                print(f"    {'-'*12} {'-'*16} {'-'*6} {'-'*6} {'-'*6}")
                for p in sorted(stats[gt][phase]):
                    for posture, s in sorted(stats[gt][phase][p].items()):
                        if s["trials"] > 0:
                            print(f"    {p:<12s} {posture:<16s} "
                                  f"{s['alpha']:>6.2f} {s['beta']:>6.2f} {s['mean']:>6.3f}")
        print(f"\n  {'='*80}")

    def _total_trials(self, planners: list, goal_tier: str, phase: str, stage: str = "") -> int:
        total = 0
        for p in planners:
            s = self.bandit.get_stats(p, goal_tier, phase, stage)
            total += s["trials"]
        return total
