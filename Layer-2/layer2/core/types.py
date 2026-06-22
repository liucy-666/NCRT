from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
import time


OutcomeType = Literal["success", "partial", "failure"]
ResponsePatternType = Literal["compliant", "partially_compliant", "refused", "safe_refusal", "unknown"]


@dataclass
class TextFeatures:
    text_length: int
    word_count: int
    safety_category: str = ""


@dataclass
class PerturbationVector:
    symbolic_intensity: float = 0.0
    structural_intensity: float = 0.0
    semantic_intensity: float = 0.0
    total_strategies: int = 0
    introduced_structures: int = 0
    token_change_ratio: float = 0.0

    def to_list(self) -> List[float]:
        return [
            self.symbolic_intensity,
            self.structural_intensity,
            self.semantic_intensity,
            float(self.total_strategies),
            float(self.introduced_structures),
            self.token_change_ratio,
        ]


@dataclass
class ResponseState:
    response_text: str
    response_label: ResponsePatternType = "unknown"
    success_score: float = 0.0
    perturbation_vector: PerturbationVector = field(default_factory=PerturbationVector)
    strategy_trace: Optional[List[Dict[str, Any]]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperienceRecord:
    original_text: str
    text_embedding: Optional[List[float]] = None
    text_features: TextFeatures = field(default_factory=lambda: TextFeatures(text_length=0, word_count=0))
    strategy_combination: List[str] = field(default_factory=list)
    strategy_dimensions: List[str] = field(default_factory=list)
    outcome: OutcomeType = "failure"
    response_label: ResponsePatternType = "unknown"
    success_score: float = 0.0
    perturbation_vector: PerturbationVector = field(default_factory=PerturbationVector)
    timestamp: float = field(default_factory=time.time)
    round_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text[:200],
            "strategy_combination": self.strategy_combination,
            "strategy_dimensions": self.strategy_dimensions,
            "outcome": self.outcome,
            "response_label": self.response_label,
            "success_score": self.success_score,
            "perturbation_vector": {
                "symbolic": self.perturbation_vector.symbolic_intensity,
                "structural": self.perturbation_vector.structural_intensity,
                "semantic": self.perturbation_vector.semantic_intensity,
                "total_strategies": self.perturbation_vector.total_strategies,
                "token_change_ratio": self.perturbation_vector.token_change_ratio,
            },
            "round_index": self.round_index,
            "timestamp": self.timestamp,
        }


@dataclass
class StrategyStats:
    strategy_name: str
    total_uses: int = 0
    success_count: int = 0
    failure_count: int = 0
    partial_count: int = 0
    last_used_round: int = -1
    last_success_round: int = -1
    consecutive_failures: int = 0
    avg_success_score: float = 0.0
    per_category: Dict[str, Dict[str, int]] = field(default_factory=dict)


"""
================================================================================
FILE: layer2/core/types.py
ROLE: Core data structures shared across Layer-2 components.

TYPE ALIASES:
  OutcomeType          = Literal["success", "partial", "failure"]
  ResponsePatternType  = Literal["compliant", "partially_compliant",
                                 "refused", "safe_refusal", "unknown"]

DATA CLASSES:

  TextFeatures:
    text_length (int)      -- 原始有害文本字符数.
    word_count (int)        -- 原始有害文本词数.
    safety_category (str)   -- 安全类别标签 (from metadata).

  PerturbationVector:
    symbolic_intensity (float)   -- 符号维扰动总强度.
    structural_intensity (float) -- 结构维扰动总强度.
    semantic_intensity (float)   -- 语义维扰动总强度.
    total_strategies (int)       -- 应用策略数.
    introduced_structures (int)  -- 引入新结构的策略数.
    token_change_ratio (float)   -- 累计 token 变化比例.
    to_list() -> List[float]     -- 转为特征向量 (6维).

  ResponseState:
    受害者模型返回的结构化状态.
    response_text (str)                     -- 模型输出原文.
    response_label (ResponsePatternType)    -- 评估分类.
    success_score (float)                   -- 0.0-1.0.
    perturbation_vector (PerturbationVector)-- 扰动向量.
    strategy_trace (Optional[List[Dict]])   -- Layer-1 策略轨迹.
    timestamp (float)                       -- 时间戳.

  ExperienceRecord:
    C 层检索的核心经验记录.
    original_text (str)                   -- 原始有害指令文本.
    text_embedding (Optional[List[float]])-- nomic-embed-text 编码.
    text_features (TextFeatures)          -- 文本统计特征.
    strategy_combination (List[str])      -- 生效的策略组合.
    strategy_dimensions (List[str])       -- 涉及的扰动维度.
    outcome (OutcomeType)                 -- success / partial / failure.
    response_label (ResponsePatternType)  -- 受害者 response 标签.
    success_score (float)                 -- 连续评分.
    perturbation_vector (PerturbationVector)-- 扰动向量.
    timestamp (float)                     -- 记录时间.
    round_index (int)                     -- 第几轮.
    to_dict() -> Dict                     -- 序列化.

  StrategyStats:
    每个策略的全局/分类型统计.
    strategy_name (str)             -- 策略名.
    total_uses (int)                -- 总使用次数.
    success_count (int)             -- 成功次数.
    failure_count (int)             -- 失败次数.
    partial_count (int)             -- 部分成功次数.
    last_used_round (int)           -- 最后使用轮次.
    last_success_round (int)        -- 最后成功轮次.
    consecutive_failures (int)      -- 连续失败次数.
    avg_success_score (float)       -- 平均得分.
    per_category (Dict[str,Dict[str,int]]) -- 按 safety_category 细分的 (success/failure/partial) 计数.

DESIGN NOTES:
  - ExperienceRecord 的 text_embedding 可能为 None (TF-IDF 降级时).
  - PerturbationVector 从 Layer-1 的 TraceLog 聚合而成.
  - ResponseState 由 policy_sampler.update() 接收, 在 Layer-3 接入前用 stub 填充.
================================================================================
"""
