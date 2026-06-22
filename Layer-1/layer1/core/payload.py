from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal

StageType = Literal["semantic", "structural", "surface", "assembly", "combo_internal"]


class AxisOrderViolationException(Exception):
    def __init__(self, message: str, violations: List[tuple] = None):
        super().__init__(message)
        self.violations = violations or []


AXIS_ORDER: Dict[str, int] = {
    "search": 0,
    "representation": 1,
    "structural": 2,
    "surface": 3,
}

AXIS_TO_STAGE: Dict[str, str] = {
    "search": "semantic",
    "representation": "semantic",
    "structural": "structural",
    "surface": "surface",
}


@dataclass
class AttackPayload:
    original_intent: str
    current_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage: StageType = "semantic"
    _frozen: bool = False
    assembly_mode: str = "default"
    context: Optional[str] = None
    role: Optional[str] = None

    def fork(self, stage: StageType = None) -> "AttackPayload":
        override_stage = stage if stage is not None else self.stage
        return AttackPayload(
            original_intent=self.original_intent,
            current_prompt=deepcopy(self.current_prompt),
            metadata=deepcopy(self.metadata),
            stage=override_stage,
            assembly_mode=self.assembly_mode,
            context=deepcopy(self.context) if self.context else None,
            role=self.role,
        )

    @classmethod
    def from_transformed_case(cls, case) -> "AttackPayload":
        from layer1.core.test_case import TransformedCase
        if isinstance(case, TransformedCase):
            return cls(
                original_intent=case.metadata.get("original_instruction", case.instruction),
                current_prompt=case.instruction,
                metadata=deepcopy(case.metadata),
                stage="semantic",
                assembly_mode=case.assembly_mode or "default",
                context=case.context,
                role=case.role,
            )
        return cls(original_intent=str(case), current_prompt=str(case))

    def to_transformed_case(self):
        from layer1.core.test_case import TransformedCase
        return TransformedCase(
            instruction=self.current_prompt,
            context=self.context,
            role=self.role,
            metadata=deepcopy(self.metadata),
            assembly_mode=self.assembly_mode,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_intent": self.original_intent,
            "current_prompt": self.current_prompt,
            "metadata": deepcopy(self.metadata),
            "stage": self.stage,
            "assembly_mode": self.assembly_mode,
            "context": self.context,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttackPayload":
        return cls(
            original_intent=d.get("original_intent", ""),
            current_prompt=d.get("current_prompt", ""),
            metadata=deepcopy(d.get("metadata", {})),
            stage=d.get("stage", "semantic"),
            assembly_mode=d.get("assembly_mode", "default"),
            context=d.get("context"),
            role=d.get("role"),
        )
