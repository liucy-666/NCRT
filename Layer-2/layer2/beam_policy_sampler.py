"""
BeamPolicySampler v2 — Layer 2 Beam Search with origin control, crossover pool,
roulette expansion, uncertainty-aware scoring, and strategy families.

核心改进 (v1 → v2):
  1. Origin 系统 — EXPERIENCE / STATISTICAL / CROSSOVER 三分天下, max_per_origin 防垄断
  2. CROSSOVER_POOL — 核心语义 × 编码器 系统性交叉, 独立维护
  3. Roulette 扩展 — 按分数抽样父本, 避免 top-2 局部最优
  4. 评分 v2 — 新增 novelty(试次反比) + uncertainty(方差) 鼓励探索未知
  5. StrategyFamily — 将策略归类为原语族, 搜索移至原语组合层级

用法:
  from layer2.beam_policy_sampler import BeamPolicySampler
  sampler = BeamPolicySampler(beam_width=5)
"""

import random as _random
import math
from enum import Enum
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field

from layer1.core.test_case import TestCase
from layer1.core.strategy import (
    Strategy, StrategyType, AttackAxis,
    STRATEGY_REGISTRY, get_strategy_meta,
)
from layer2.policy_sampler import PolicySampler
from layer2.config import Layer2Config
from layer2.core.types import ResponseState, PerturbationVector


# ═══════════════════════════════════════════════════════════
# Origin — 候选来源标记
# ═══════════════════════════════════════════════════════════

class Origin(str, Enum):
    EXPERIENCE = "experience"    # C 层 embedding 检索复用
    STATISTICAL = "statistical"  # B 层统计评分生成
    CROSSOVER = "crossover"      # 核心原语交叉池


# ═══════════════════════════════════════════════════════════
# StrategyFamily — 攻击原语族
# ═══════════════════════════════════════════════════════════

class StrategyFamily(str, Enum):
    """策略按攻击原语归类, 而非按 type/axis 机械分类."""
    ENCODING = "encoding"        # 表面编码: reverse_text, caesar, rot13, base64, atbash...
    SEMANTIC = "semantic"        # 语义重构: dual_model_hijack, deep_inception, pair_enhanced...
    GENERATION = "generation"    # 内容生成: gptfuzzer_style, ica_enhanced, code_chameleon...
    CONSTRAINT = "constraint"    # 格式约束: academic_framing, role_play, persuasion, prefix_hijack...
    UNKNOWN = "unknown"


# ── 策略→原语族映射 ──
STRATEGY_FAMILY_MAP: Dict[str, StrategyFamily] = {
    # Encoding family
    "reverse_text": StrategyFamily.ENCODING,
    "caesar": StrategyFamily.ENCODING,
    "rot13": StrategyFamily.ENCODING,
    "base64": StrategyFamily.ENCODING,
    "base64_raw": StrategyFamily.ENCODING,
    "atbash": StrategyFamily.ENCODING,
    "morse": StrategyFamily.ENCODING,
    "ascii_encode": StrategyFamily.ENCODING,
    "odd_even": StrategyFamily.ENCODING,
    "binary_tree": StrategyFamily.ENCODING,
    "leetspeak": StrategyFamily.ENCODING,
    "disemvowel": StrategyFamily.ENCODING,
    "misspell": StrategyFamily.ENCODING,
    "zerowidth": StrategyFamily.ENCODING,
    "length_encode": StrategyFamily.ENCODING,
    "multilingual": StrategyFamily.ENCODING,
    "payload_split": StrategyFamily.ENCODING,

    # Semantic family
    "dual_model_hijack": StrategyFamily.SEMANTIC,
    "deep_inception_enhanced": StrategyFamily.SEMANTIC,
    "pair_enhanced": StrategyFamily.SEMANTIC,
    "tap_style": StrategyFamily.SEMANTIC,
    "renellm_enhanced": StrategyFamily.SEMANTIC,

    # Generation family
    "gptfuzzer_style": StrategyFamily.GENERATION,
    "ica_enhanced": StrategyFamily.GENERATION,
    "code_chameleon_enhanced": StrategyFamily.GENERATION,

    # Constraint family
    "academic_framing": StrategyFamily.CONSTRAINT,
    "role_play": StrategyFamily.CONSTRAINT,
    "persuasion": StrategyFamily.CONSTRAINT,
    "prefix_hijack": StrategyFamily.CONSTRAINT,
    "refusal_suppression": StrategyFamily.CONSTRAINT,
    "jailbreak_skeleton": StrategyFamily.CONSTRAINT,
    "style_constraint": StrategyFamily.CONSTRAINT,
}


def get_strategy_family(name: str) -> StrategyFamily:
    """获取策略的原语族."""
    return STRATEGY_FAMILY_MAP.get(name, StrategyFamily.UNKNOWN)


def get_family_vector(strategy_names: List[str]) -> Set[StrategyFamily]:
    """获取策略链的原语族向量."""
    return {get_strategy_family(n) for n in strategy_names} - {StrategyFamily.UNKNOWN}


# ═══════════════════════════════════════════════════════════
# CROSSOVER_POOL — 核心原语交叉
# ═══════════════════════════════════════════════════════════

# 实验数据驱动: SEMANTIC 核心 × ENCODING 核心
CROSSOVER_SEMANTIC = ["dual_model_hijack", "deep_inception_enhanced", "pair_enhanced"]
CROSSOVER_ENCODING = ["reverse_text", "caesar", "binary_tree", "atbash", "rot13", "base64"]
CROSSOVER_CONSTRAINT = ["academic_framing", "persuasion", "refusal_suppression"]


def generate_crossover_chains(
    budget: int,
    include_constraint: bool = False,
) -> List[List[str]]:
    """生成核心原语交叉链列表."""
    chains: List[List[str]] = []
    # sem × enc
    for sem in CROSSOVER_SEMANTIC:
        for enc in CROSSOVER_ENCODING:
            if budget >= 2:
                chains.append([sem, enc])
            if include_constraint and budget >= 3:
                for con in CROSSOVER_CONSTRAINT:
                    chains.append([sem, enc, con])
    # sem × enc × enc (3-chain for budget=3)
    if budget >= 3:
        for sem in CROSSOVER_SEMANTIC[:2]:  # top 2 only
            for e1 in CROSSOVER_ENCODING[:3]:
                for e2 in CROSSOVER_ENCODING[3:]:
                    chains.append([sem, e1, e2])
    return chains


# ═══════════════════════════════════════════════════════════
# BeamCandidate v2
# ═══════════════════════════════════════════════════════════

@dataclass
class BeamCandidate:
    """Beam 候选策略链 v2."""
    strategies: List[Strategy]
    strategy_names: Tuple[str, ...]
    family_vector: Set[str] = field(default_factory=set)   # 原语族向量
    score: float = 0.0
    tried: bool = False
    origin: Origin = Origin.STATISTICAL                    # ← 来源标记
    source_detail: str = ""                                 # 来源详情
    parent_score: float = 0.0                               # 实际执行得分
    generation: int = 0
    trial_count: int = 0                                    # 历史被尝试次数 (用于 novelty)

    def __hash__(self):
        return hash(self.strategy_names)

    def __eq__(self, other):
        if not isinstance(other, BeamCandidate):
            return False
        return self.strategy_names == other.strategy_names


# ═══════════════════════════════════════════════════════════
# BeamPolicySampler v2
# ═══════════════════════════════════════════════════════════

class BeamPolicySampler(PolicySampler):
    """
    Beam Search 策略选择器 v2.

    相比 v1 的五项关键改进:
      - Origin 三分 + max_per_origin 防垄断
      - CROSSOVER_POOL 系统交叉
      - Roulette 扩展替代 top-2
      - scoring: novelty + uncertainty
      - StrategyFamily 原语族搜索
    """

    def __init__(
        self,
        config: Optional[Layer2Config] = None,
        reward=None,
        beam_width: int = 5,
        expansion_factor: int = 3,
        max_generations: int = 3,
    ):
        super().__init__(config, reward)
        self.beam_width = max(2, beam_width)
        self.expansion_factor = max(2, expansion_factor)
        self.max_generations = max(1, max_generations)

        # Origin 配额
        self._max_per_origin = max(1, beam_width // 2)

        # Beam state
        self._beam: List[BeamCandidate] = []
        self._tried_chains: Set[Tuple[str, ...]] = set()
        self._generation: int = 0
        self._all_candidates: Dict[Tuple[str, ...], BeamCandidate] = {}
        self._global_trial_counts: Dict[str, int] = {}  # 策略名 → 全局被选次数

    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════

    def select(self, test_case: TestCase) -> List[Strategy]:
        if self._attack_terminated:
            return []
        if not self._should_continue_for_instruction(test_case):
            return []
        self._instruction_rounds += 1

        if not self._beam or self._generation == 0:
            self._initialize_beam(test_case)
            self._generation = 1

        candidate = self._pick_best_untried()
        if candidate is not None:
            self._tried_chains.add(candidate.strategy_names)
            candidate.tried = True
            candidate.trial_count += 1
            for name in candidate.strategy_names:
                self._global_trial_counts[name] = self._global_trial_counts.get(name, 0) + 1
            self._last_selected = candidate.strategies
            return candidate.strategies

        # 扩展
        if self._generation < self.max_generations:
            self._expand_beam(test_case)
            self._generation += 1
            candidate = self._pick_best_untried()
            if candidate is not None:
                self._tried_chains.add(candidate.strategy_names)
                candidate.tried = True
                candidate.trial_count += 1
                for name in candidate.strategy_names:
                    self._global_trial_counts[name] = self._global_trial_counts.get(name, 0) + 1
                self._last_selected = candidate.strategies
                return candidate.strategies

        # 兜底
        allowed_axes = getattr(test_case.attack_budget, "allowed_axes", {"search", "representation", "surface"})
        pool = self._build_candidate_pool(allowed_axes)
        fallback = self._b_layer_select(test_case, pool, test_case.attack_budget.max_strategies)
        self._last_selected = fallback
        return fallback

    def reset_instruction_context(self, new_instruction: str = "") -> None:
        super().reset_instruction_context(new_instruction)
        self._beam = []
        self._tried_chains = set()
        self._generation = 0
        self._all_candidates = {}

    def update(self, test_case, applied_strategies, response_state):
        """覆盖 update, 将实际执行得分回写到 beam 候选."""
        super().update(test_case, applied_strategies, response_state)
        # 回写实际得分
        score = self._reward.evaluate(response_state)
        for c in self._beam:
            if c.tried and c.strategy_names in self._tried_chains:
                c.parent_score = max(c.parent_score, score)

    # ═══════════════════════════════════════════════════════
    # Beam 初始化 v2
    # ═══════════════════════════════════════════════════════

    def _initialize_beam(self, test_case: TestCase) -> None:
        """
        v2 初始化: 三分天下 (EXPERIENCE / STATISTICAL / CROSSOVER)
        max_per_origin = beam_width // 2 保证多样性.
        """
        budget = test_case.attack_budget.max_strategies
        allowed_axes = getattr(test_case.attack_budget, "allowed_axes", {"search", "representation", "surface"})
        self._last_allowed_axes = allowed_axes

        all_raw: Dict[Origin, List[BeamCandidate]] = {
            Origin.EXPERIENCE: [],
            Origin.STATISTICAL: [],
            Origin.CROSSOVER: [],
        }

        # ── Source EXPERIENCE: C 层检索, 但限制数量 ──
        exp_limit = min(self._max_per_origin, self.beam_width)
        if self._experience_base.total_records >= self.config.min_experiences_for_c:
            results = self._experience_base.search_with_preference(
                test_case.instruction, prefer_outcome="success"
            )
            for record, sim in results[:exp_limit]:
                chain = self._build_chain_from_names(record.strategy_combination, budget, test_case.seed)
                if chain:
                    names = tuple(sorted(s.name for s in chain))
                    candidate = BeamCandidate(
                        strategies=chain,
                        strategy_names=names,
                        family_vector=get_family_vector(list(names)),
                        score=self._compute_beam_score_v2(chain, test_case, origin=Origin.EXPERIENCE),
                        origin=Origin.EXPERIENCE,
                        source_detail=f"c_layer(sim={sim:.3f})",
                        generation=0,
                    )
                    all_raw[Origin.EXPERIENCE].append(candidate)

        # ── Source CROSSOVER: 核心原语交叉 (优先级最高) ──
        cross_limit = min(self._max_per_origin, self.beam_width)
        cross_chains = generate_crossover_chains(budget, include_constraint=False)
        _random.shuffle(cross_chains)
        for chain_names in cross_chains:
            if len(all_raw[Origin.CROSSOVER]) >= cross_limit:
                break
            chain = self._build_chain_from_names(chain_names, budget, test_case.seed)
            if chain:
                names = tuple(sorted(s.name for s in chain))
                candidate = BeamCandidate(
                    strategies=chain,
                    strategy_names=names,
                    family_vector=get_family_vector(list(names)),
                    score=0,  # scored below
                    origin=Origin.CROSSOVER,
                    source_detail=f"crossover({'×'.join(chain_names)})",
                    generation=0,
                )
                candidate.score = self._compute_beam_score_v2(chain, test_case, origin=Origin.CROSSOVER)
                all_raw[Origin.CROSSOVER].append(candidate)

        # ── Source STATISTICAL: B 层多样化生成 ──
        stat_limit = self.beam_width * self.expansion_factor
        base_pool = self._build_candidate_pool(allowed_axes)
        if base_pool:
            dims = list(allowed_axes)
            AXIS_PRIORITY = {"representation": 0, "surface": 1, "search": 2}
            dims = sorted(dims, key=lambda d: AXIS_PRIORITY.get(d, 99))
            self._scorer.set_current_round(self._current_round)
            for _ in range(stat_limit):
                chain: List[Strategy] = []
                names: List[str] = []
                for dim in dims:
                    dim_strats = [s for s in base_pool if getattr(s, "axis", None) and s.axis.value == dim]
                    if not dim_strats:
                        continue
                    s = self._weighted_choice(dim_strats)
                    if s and s.name not in names:
                        chain.append(s)
                        names.append(s.name)
                    if len(chain) >= budget:
                        break
                if chain:
                    names_t = tuple(sorted(names))
                    candidate = BeamCandidate(
                        strategies=chain,
                        strategy_names=names_t,
                        family_vector=get_family_vector(list(names_t)),
                        score=0,
                        origin=Origin.STATISTICAL,
                        source_detail="b_layer_diverse",
                        generation=0,
                    )
                    candidate.score = self._compute_beam_score_v2(chain, test_case, origin=Origin.STATISTICAL)
                    all_raw[Origin.STATISTICAL].append(candidate)

        # ── Origin-aware dedup & prune ──
        self._dedup_and_prune_v2(all_raw)

    # ═══════════════════════════════════════════════════════
    # Beam 扩展 v2 — Roulette 抽样 + 原语族变异
    # ═══════════════════════════════════════════════════════

    def _expand_beam(self, test_case: TestCase) -> None:
        """
        v2 扩展: roulette 抽样 3 个父本, 按原语族变异.
        """
        budget = test_case.attack_budget.max_strategies
        tried = [c for c in self._beam if c.tried]
        if not tried:
            return

        # ── Roulette 抽样: 按 parent_score (实际执行分) 加权 ──
        expand_k = min(3, len(tried))
        scores = [max(0.01, c.parent_score) for c in tried]
        total = sum(scores)
        parents: List[BeamCandidate] = []
        tried_copy = list(tried)
        for _ in range(expand_k):
            r = _random.uniform(0, total)
            cum = 0.0
            for i, c in enumerate(tried_copy):
                cum += scores[i]
                if r <= cum:
                    parents.append(c)
                    break

        new_candidates: List[BeamCandidate] = []

        for parent in parents:
            parent_names = set(parent.strategy_names)
            parent_families = parent.family_vector

            # ── 变异 1: 替换同族策略 (encoding → encoding) ──
            if StrategyFamily.ENCODING.value in parent_families:
                enc_in_parent = [
                    n for n in parent_names
                    if get_strategy_family(n) == StrategyFamily.ENCODING
                ]
                alt_enc = [
                    n for n in STRATEGY_REGISTRY
                    if get_strategy_family(n) == StrategyFamily.ENCODING
                    and n not in parent_names
                    and not get_strategy_meta(n).get("deprecated", False)
                    and get_strategy_meta(n).get("selection_weight", 1.0) >= 0.3
                ]
                if enc_in_parent and alt_enc:
                    for alt in _random.sample(alt_enc, min(2, len(alt_enc))):
                        new_names = [n for n in parent.strategy_names if n not in enc_in_parent] + [alt]
                        chain = self._build_chain_from_names(new_names, budget, test_case.seed)
                        if chain:
                            new_candidates.append(BeamCandidate(
                                strategies=chain,
                                strategy_names=tuple(sorted(new_names)),
                                family_vector=get_family_vector(new_names),
                                origin=Origin.CROSSOVER,
                                source_detail=f"expand_swap_enc({alt})",
                                generation=self._generation,
                                parent_score=parent.parent_score,
                            ))

            # ── 变异 2: 替换语义策略 ──
            if StrategyFamily.SEMANTIC.value in parent_families:
                sem_in_parent = [
                    n for n in parent_names
                    if get_strategy_family(n) == StrategyFamily.SEMANTIC
                ]
                alt_sem = [
                    n for n in STRATEGY_REGISTRY
                    if get_strategy_family(n) == StrategyFamily.SEMANTIC
                    and n not in parent_names
                    and not get_strategy_meta(n).get("deprecated", False)
                    and get_strategy_meta(n).get("selection_weight", 1.0) >= 0.5
                ]
                if sem_in_parent and alt_sem:
                    for alt in _random.sample(alt_sem, min(2, len(alt_sem))):
                        new_names = [n for n in parent.strategy_names if n not in sem_in_parent] + [alt]
                        chain = self._build_chain_from_names(new_names, budget, test_case.seed)
                        if chain:
                            new_candidates.append(BeamCandidate(
                                strategies=chain,
                                strategy_names=tuple(sorted(new_names)),
                                family_vector=get_family_vector(new_names),
                                origin=Origin.CROSSOVER,
                                source_detail=f"expand_swap_sem({alt})",
                                generation=self._generation,
                                parent_score=parent.parent_score,
                            ))

            # ── 变异 3: 追加新族 (如果 budget 允许) ──
            if len(parent_names) < budget:
                missing_families = {StrategyFamily.CONSTRAINT, StrategyFamily.GENERATION} - {
                    StrategyFamily(f) for f in parent_families
                }
                for mf in missing_families:
                    alt_in_family = [
                        n for n in STRATEGY_REGISTRY
                        if get_strategy_family(n) == mf
                        and n not in parent_names
                        and not get_strategy_meta(n).get("deprecated", False)
                        and get_strategy_meta(n).get("selection_weight", 1.0) >= 0.5
                    ]
                    if alt_in_family:
                        alt = _random.choice(alt_in_family)
                        new_names = list(parent.strategy_names) + [alt]
                        chain = self._build_chain_from_names(new_names, budget, test_case.seed)
                        if chain:
                            new_candidates.append(BeamCandidate(
                                strategies=chain,
                                strategy_names=tuple(sorted(new_names)),
                                family_vector=get_family_vector(new_names),
                                origin=Origin.CROSSOVER,
                                source_detail=f"expand_add({mf.value}:{alt})",
                                generation=self._generation,
                                parent_score=parent.parent_score,
                            ))
                            break  # 只加一个

        # Score & merge
        for c in new_candidates:
            c.score = self._compute_beam_score_v2(c.strategies, test_case, origin=c.origin)

        # Origin-aware merge
        raw_by_origin: Dict[Origin, List[BeamCandidate]] = {
            Origin.EXPERIENCE: [],
            Origin.STATISTICAL: [],
            Origin.CROSSOVER: [c for c in new_candidates],
        }
        self._dedup_and_prune_v2(raw_by_origin)

    # ═══════════════════════════════════════════════════════
    # Scoring v2 — 新增 novelty + uncertainty
    # ═══════════════════════════════════════════════════════

    def _compute_beam_score_v2(
        self,
        strategies: List[Strategy],
        test_case: TestCase,
        origin: Origin = Origin.STATISTICAL,
    ) -> float:
        """
        v2 评分: success_rate + diversity + novelty + uncertainty + origin_bonus.

        success_rate  (0.30): 历史成功率
        diversity     (0.20): 原语族多样性
        novelty       (0.25): 未充分探索的候选加分 (trial_count 反比)
        uncertainty   (0.15): 统计方差 → 鼓励探索不确定候选
        origin_bonus  (0.10): 来源加分 (CROSSOVER > STATISTICAL > EXPERIENCE)
        """
        names = [s.name for s in strategies]

        # 1. Success rate
        hist_scores = []
        for name in names:
            stats = self._scorer.get_all_stats().get(name)
            if stats and stats.total_uses > 0:
                sr = stats.success_count / stats.total_uses
                hist_scores.append(sr)
        success_rate = sum(hist_scores) / len(hist_scores) if hist_scores else 0.5

        # 2. Diversity: 原语族级多样性
        my_families = get_family_vector(names)
        seen_families: Set[str] = set()
        for c in self._beam:
            seen_families |= c.family_vector
        novel_families = len(my_families - seen_families)
        diversity = min(1.0, novel_families * 0.5 + 0.2 * len(my_families))

        # 3. Novelty: 试次反比 — 被尝试越少, novelty 越高
        max_trials = max(self._global_trial_counts.values()) if self._global_trial_counts else 1
        name_trials = [self._global_trial_counts.get(n, 0) for n in names]
        avg_trials = sum(name_trials) / max(1, len(name_trials))
        novelty = 1.0 - min(1.0, avg_trials / max(1, max_trials + 1))

        # 4. Uncertainty: 统计方差 → 高分给不确定的候选
        uncertainties = []
        for name in names:
            stats = self._scorer.get_all_stats().get(name)
            if stats and stats.total_uses >= 3:
                p = stats.success_count / stats.total_uses
                var = p * (1 - p)  # Bernoulli variance
                uncertainties.append(var)
            elif stats and stats.total_uses > 0:
                uncertainties.append(0.25)  # 样本不足 → 高不确定性
            else:
                uncertainties.append(0.5)   # 完全未知 → 最高不确定性
        uncertainty = sum(uncertainties) / len(uncertainties)

        # 5. Origin bonus
        origin_bonus_map = {
            Origin.CROSSOVER: 0.9,
            Origin.STATISTICAL: 0.6,
            Origin.EXPERIENCE: 0.4,
        }
        origin_bonus = origin_bonus_map.get(origin, 0.5)

        # 6. Weight bonus
        weight_bonus = 1.0
        for name in names:
            meta = get_strategy_meta(name)
            weight_bonus *= meta.get("selection_weight", 1.0)
        weight_bonus = weight_bonus ** (1.0 / max(1, len(names)))

        score = (
            0.30 * success_rate
            + 0.20 * diversity
            + 0.25 * novelty
            + 0.15 * uncertainty
            + 0.10 * origin_bonus
        )
        score *= (0.5 + 0.5 * weight_bonus)

        return min(1.0, max(0.0, score))

    # ═══════════════════════════════════════════════════════
    # Dedup & Prune v2 — origin-aware
    # ═══════════════════════════════════════════════════════

    def _dedup_and_prune_v2(
        self,
        raw_by_origin: Dict[Origin, List[BeamCandidate]],
    ) -> None:
        """
        v2: 先按 origin 限额裁剪, 再合并全局排序.
        max_per_origin = beam_width // 2
        """
        origin_limited: List[BeamCandidate] = []

        for origin, candidates in raw_by_origin.items():
            # Dedup
            unique: List[BeamCandidate] = []
            for c in candidates:
                if c.strategy_names in self._all_candidates:
                    continue
                if c.strategy_names in self._tried_chains:
                    continue
                self._all_candidates[c.strategy_names] = c
                unique.append(c)

            # Sort by score, limit per origin
            unique.sort(key=lambda c: -c.score)
            limited = unique[:self._max_per_origin]
            origin_limited.extend(limited)

        # ── Merge with existing beam ──
        for c in origin_limited:
            if c not in self._beam:
                self._beam.append(c)

        # ── Global sort & prune ──
        self._beam.sort(key=lambda c: -c.score)
        self._beam = self._beam[:self.beam_width]

    # ═══════════════════════════════════════════════════════
    # Helpers (unchanged from v1 + family-aware)
    # ═══════════════════════════════════════════════════════

    def _pick_best_untried(self) -> Optional[BeamCandidate]:
        untried = [c for c in self._beam if not c.tried]
        if not untried:
            return None
        untried.sort(key=lambda c: (-c.score, c.generation))
        return untried[0]

    def _build_chain_from_names(self, names: List[str], budget: int, seed: int) -> List[Strategy]:
        chain: List[Strategy] = []
        seen: Set[str] = set()
        for i, name in enumerate(names):
            if name not in STRATEGY_REGISTRY:
                continue
            if name in seen:
                continue
            if get_strategy_meta(name).get("deprecated", False):
                continue
            cls = STRATEGY_REGISTRY[name]
            try:
                s = cls.__new__(cls)
                s.__init__(intensity=0.7, seed=seed + i)
                chain.append(s)
                seen.add(name)
            except Exception:
                continue
            if len(chain) >= budget:
                break
        return chain if chain else []

    def _get_strategy_axis(self, name: str) -> str:
        if name not in STRATEGY_REGISTRY:
            return "unknown"
        try:
            dummy = STRATEGY_REGISTRY[name].__new__(STRATEGY_REGISTRY[name])
            dummy.__init__()
            ax = getattr(dummy, "axis", None)
            return ax.value if ax else "unknown"
        except Exception:
            return "unknown"

    def _weighted_choice(self, strategies: List[Strategy]) -> Optional[Strategy]:
        if not strategies:
            return None
        weights = []
        for s in strategies:
            meta = get_strategy_meta(s.name)
            weights.append(meta.get("selection_weight", 1.0))
        total = sum(weights)
        if total == 0:
            return _random.choice(strategies)
        r = _random.uniform(0, total)
        cum = 0.0
        for s, w in zip(strategies, weights):
            cum += w
            if r <= cum:
                return s
        return strategies[-1]

    def get_beam_report(self) -> dict:
        return {
            "beam_width": self.beam_width,
            "max_per_origin": self._max_per_origin,
            "current_generation": self._generation,
            "beam_size": len(self._beam),
            "tried_chains": len(self._tried_chains),
            "candidates": [
                {
                    "chain": list(c.strategy_names),
                    "families": sorted(c.family_vector),
                    "score": round(c.score, 4),
                    "tried": c.tried,
                    "origin": c.origin.value,
                    "source_detail": c.source_detail,
                    "generation": c.generation,
                    "trial_count": c.trial_count,
                }
                for c in sorted(self._beam, key=lambda c: -c.score)
            ],
        }

    def get_policy_report(self) -> dict:
        report = super().get_policy_report()
        report["beam"] = self.get_beam_report()
        return report
