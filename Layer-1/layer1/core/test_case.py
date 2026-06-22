from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, Any, Set
import json


RoleType = Literal["user", "system", "external", "retrieved"]
AssemblyModeType = Literal["default", "chatml", "openai", "raw"]


@dataclass
class AttackBudget:
    max_strategies: int = 5
    max_intensity: float = 1.0
    max_modification_ratio: float = 0.5
    allowed_dimensions: Set[str] = field(default_factory=lambda: {"symbolic", "structural", "semantic"})
    allowed_axes: Set[str] = field(default_factory=lambda: {"search", "representation", "surface"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_strategies": self.max_strategies,
            "max_intensity": self.max_intensity,
            "max_modification_ratio": self.max_modification_ratio,
            "allowed_dimensions": list(self.allowed_dimensions),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttackBudget":
        return cls(
            max_strategies=d.get("max_strategies", 5),
            max_intensity=d.get("max_intensity", 1.0),
            max_modification_ratio=d.get("max_modification_ratio", 0.5),
            allowed_dimensions=set(d.get("allowed_dimensions", {"symbolic", "structural", "semantic"})),
        )


@dataclass
class TestCase:
    instruction: str
    context: Optional[str] = None
    role: Optional[RoleType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    attack_budget: AttackBudget = field(default_factory=AttackBudget)
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "context": self.context,
            "role": self.role,
            "metadata": self.metadata,
            "attack_budget": self.attack_budget.to_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestCase":
        return cls(
            instruction=d["instruction"],
            context=d.get("context"),
            role=d.get("role"),
            metadata=d.get("metadata", {}),
            attack_budget=AttackBudget.from_dict(d.get("attack_budget", {})),
            seed=d.get("seed"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TestCase":
        return cls.from_dict(json.loads(json_str))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def normalize(self) -> "TestCase":
        return TestCase(
            instruction=self.instruction.strip(),
            context=self.context.strip() if self.context else None,
            role=self.role if self.role in ("user", "system", "external", "retrieved") else None,
            metadata=self.metadata if isinstance(self.metadata, dict) else {},
            attack_budget=self.attack_budget,
            seed=self.seed,
        )


@dataclass
class TransformedCase:
    instruction: str
    context: Optional[str] = None
    role: Optional[RoleType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    full_input: str = ""
    assembly_mode: AssemblyModeType = "default"

    @classmethod
    def from_test_case(cls, tc: TestCase) -> "TransformedCase":
        return cls(
            instruction=tc.instruction,
            context=tc.context,
            role=tc.role,
            metadata=dict(tc.metadata),
            full_input="",
            assembly_mode="default",
        )

    def to_string(self) -> str:
        return self.full_input if self.full_input else self.instruction


"""
================================================================================
FILE: layer1/core/test_case.py
ROLE: Defines the three fundamental data structures of Layer 1.

VARIABLES AND TYPES:
  RoleType            = Literal["user", "system", "external", "retrieved"]
                        -- Valid semantic roles for input classification.
                           Used only as a label in Layer 1, no execution semantics.

  AssemblyModeType    = Literal["default", "chatml", "openai", "raw"]
                        -- How full_input is assembled from instruction + context.
                           'default': plain text with labels
                           'chatml' : <|im_start|> delimited format
                           'openai' : JSON [{"role", "content"}] format
                           'raw'    : bare instruction string

DATA CLASSES:

  AttackBudget:
    max_strategies (int)        -- Maximum number of strategies to apply in sequence.
    max_intensity (float)       -- Upper bound for per-strategy intensity [0.0, 1.0].
    max_modification_ratio(float)-- Upper bound for total token-level modification ratio.
    allowed_dimensions (set)    -- Which perturbation dimensions are enabled.
                                   Subset of {"symbolic", "structural", "semantic"}.

  TestCase:
    instruction (str)           -- The (harmful) task instruction text to perturb.
    context (Optional[str])     -- Auxiliary context; may be empty or hold injection payloads.
    role (Optional[RoleType])   -- Semantic role tag (user/system/external/retrieved).
    metadata (dict)             -- Arbitrary tags: task_type, safety_category, domain, etc.
    attack_budget (AttackBudget)-- Controls what strategies and how many are applied.
    seed (Optional[int])        -- RNG seed for reproducible perturbation.

  TransformedCase:
    instruction (str)           -- Instruction text after perturbation (may differ from original).
    context (Optional[str])     -- Context text after perturbation.
    role (Optional[RoleType])   -- Role tag (may be changed by structural strategies).
    metadata (dict)             -- Copied from source TestCase; strategies may append entries.
    full_input (str)            -- The complete assembled prompt string, ready for model consumption.
    assembly_mode (AssemblyModeType) -- How full_input was assembled.

DESIGN NOTES:
  - TestCase is the external interface (input contract).
  - TransformedCase is the internal chain carrier (intermediate states).
  - AttackBudget acts as a "control knob" — users set it, Pipeline enforces it.
  - normalize() fills defaults and strips whitespace; call this at Pipeline start.
================================================================================
"""
