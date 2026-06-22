from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import time


OutcomeLabelType = Literal["compliant", "partially_compliant", "refused", "error"]


@dataclass
class JudgmentResult:
    complied: bool
    compliance_score: float
    label: OutcomeLabelType
    breached_categories: List[str] = field(default_factory=list)
    explanation: str = ""
    raw_judge_response: str = ""
    judge_model: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complied": self.complied,
            "compliance_score": self.compliance_score,
            "label": self.label,
            "breached_categories": self.breached_categories,
            "explanation": self.explanation,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgmentResult":
        return cls(
            complied=d.get("complied", False),
            compliance_score=d.get("compliance_score", 0.0),
            label=d.get("label", "refused"),
            breached_categories=d.get("breached_categories", []),
            explanation=d.get("explanation", ""),
            raw_judge_response=d.get("raw_judge_response", ""),
            judge_model=d.get("judge_model", ""),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class AttackConfigSummary:
    original_instruction: str
    transformed_prompt: str
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    total_strategies: int = 0
    dimensions_used: List[str] = field(default_factory=list)
    token_change_ratio: float = 0.0
    assembly_mode: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_instruction": self.original_instruction,
            "transformed_prompt": self.transformed_prompt[:500],
            "strategies": self.strategies,
            "total_strategies": self.total_strategies,
            "dimensions_used": self.dimensions_used,
            "token_change_ratio": self.token_change_ratio,
            "assembly_mode": self.assembly_mode,
        }


@dataclass
class OutcomeClassification:
    jailbreak_successful: bool
    compliance_score: float
    compliance_label: OutcomeLabelType
    breached_categories: List[str] = field(default_factory=list)
    judge_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jailbreak_successful": self.jailbreak_successful,
            "compliance_score": self.compliance_score,
            "compliance_label": self.compliance_label,
            "breached_categories": self.breached_categories,
            "judge_explanation": self.judge_explanation,
        }


@dataclass
class AuditReport:
    attack_config: AttackConfigSummary
    outcome: OutcomeClassification
    victim_response: str = ""
    trace_log: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_config": self.attack_config.to_dict(),
            "outcome": self.outcome.to_dict(),
            "victim_response": self.victim_response[:1000],
            "trace_log": self.trace_log,
            "metadata": self.metadata,
            "generated_at": self.generated_at,
        }


"""
================================================================================
FILE: layer3/core/types.py
ROLE: Core data structures for Layer 3.

TYPE ALIASES:
  OutcomeLabelType = Literal["compliant", "partially_compliant", "refused", "error"]

DATA CLASSES:

  JudgmentResult:
    complied (bool)                  -- Whether the victim model complied.
    compliance_score (float)         -- 0.0 (fully refused) to 1.0 (fully complied).
    label (OutcomeLabelType)         -- compliant / partially_compliant / refused / error.
    breached_categories (List[str])  -- Safety categories breached.
    explanation (str)                -- Judge's natural language explanation.
    raw_judge_response (str)         -- Raw JSON response from Judge LLM.
    judge_model (str)                -- Which judge model was used.
    timestamp (float)                -- Unix timestamp.

  AttackConfigSummary:
    original_instruction (str)       -- Original harmful instruction text.
    transformed_prompt (str)         -- Perturbed prompt sent to victim model.
    strategies (List[Dict])          -- List of strategy details (name/type/intensity).
    total_strategies (int)           -- Number of strategies applied.
    dimensions_used (List[str])      -- Perturbation dimensions covered.
    token_change_ratio (float)       -- Cumulative token change ratio.
    assembly_mode (str)              -- Prompt assembly mode.

  OutcomeClassification:
    jailbreak_successful (bool)      -- Whether jailbreak was successful.
    compliance_score (float)         -- Numeric compliance score.
    compliance_label (str)           -- compliant / partially_compliant / refused.
    breached_categories (List[str])  -- Which safety categories were breached.
    judge_explanation (str)          -- Judge's explanation text.

  AuditReport:
    attack_config (AttackConfigSummary) -- Attack configuration details.
    outcome (OutcomeClassification)     -- Outcome classification.
    victim_response (str)               -- Victim model's response text.
    trace_log (List[Dict])              -- Layer 1 strategy trace.
    metadata (Dict)                     -- Additional metadata.
    generated_at (float)                -- Report generation timestamp.
================================================================================
"""
