from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Set
from enum import Enum


class StrategyType(str, Enum):
    SYMBOLIC = "symbolic"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


class AttackAxis(str, Enum):
    SEARCH = "search"
    REPRESENTATION = "representation"
    SURFACE = "surface"


DEFAULT_AXIS_ORDER = ["search", "representation", "surface"]


class StrategyScope(str, Enum):
    INSTRUCTION = "instruction"
    CONTEXT = "context"
    FULL_INPUT = "full_input"


class Strategy(ABC):

    name: str
    type: StrategyType
    axis: AttackAxis
    intensity: float
    scope: StrategyScope
    seed: Optional[int]
    deprecated: bool
    selection_weight: float

    def __init__(
        self,
        name: str,
        strategy_type: StrategyType,
        intensity: float = 0.5,
        scope: StrategyScope = StrategyScope.INSTRUCTION,
        seed: Optional[int] = None,
        axis: AttackAxis = AttackAxis.SURFACE,
        deprecated: bool = False,
        selection_weight: float = 1.0,
    ):
        self.name = name
        self.type = strategy_type
        self.axis = axis
        self.intensity = max(0.0, min(1.0, intensity))
        self.scope = scope
        self.seed = seed
        self.deprecated = deprecated
        self.selection_weight = max(0.0, min(1.0, selection_weight))

    @abstractmethod
    def apply(self, case: "TransformedCase") -> Tuple["TransformedCase", "StrategyTrace"]:
        pass

    @abstractmethod
    def validate(self, case: "TransformedCase") -> bool:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"type={self.type.value}, intensity={self.intensity:.2f}, "
            f"scope={self.scope.value})"
        )


def clamp_intensity(intensity: float) -> float:
    return max(0.0, min(1.0, intensity))


def get_strategy_dimension(strategy_type: StrategyType) -> str:
    return strategy_type.value


def get_strategy_axis(axis: AttackAxis) -> str:
    return axis.value


STRATEGY_REGISTRY: Dict[str, type] = {}

# ── 策略元数据注册表（与注册解耦，可运行时动态调整）──
STRATEGY_META: Dict[str, Dict[str, Any]] = {}


def register_strategy(name: str, cls: type, deprecated: bool = False,
                      selection_weight: float = 1.0) -> None:
    STRATEGY_REGISTRY[name] = cls
    STRATEGY_META[name] = {
        "deprecated": deprecated,
        "selection_weight": max(0.0, min(1.0, selection_weight)),
    }


def set_strategy_weight(name: str, weight: float) -> None:
    """运行时调整策略选择权重。0.0 = 几乎不会被选, 1.0 = 正常."""
    if name in STRATEGY_META:
        STRATEGY_META[name]["selection_weight"] = max(0.0, min(1.0, weight))


def set_strategy_deprecated(name: str, deprecated: bool) -> None:
    """标记策略为 deprecated。deprecated 策略只在探索轮次中有小概率被选中."""
    if name in STRATEGY_META:
        STRATEGY_META[name]["deprecated"] = deprecated


def get_strategy_meta(name: str) -> Dict[str, Any]:
    """获取策略元数据，若无记录则返回默认值."""
    return STRATEGY_META.get(name, {"deprecated": False, "selection_weight": 1.0})


def get_all_strategy_names() -> Set[str]:
    return set(STRATEGY_REGISTRY.keys())


def get_active_strategy_names() -> Set[str]:
    """返回所有非 deprecated 的策略名."""
    return {
        name for name in STRATEGY_REGISTRY
        if not STRATEGY_META.get(name, {}).get("deprecated", False)
    }


def get_deprecated_strategy_names() -> Set[str]:
    """返回所有 deprecated 的策略名."""
    return {
        name for name in STRATEGY_REGISTRY
        if STRATEGY_META.get(name, {}).get("deprecated", False)
    }


def get_strategies_by_dimension(dimension: str) -> Dict[str, type]:
    result = {}
    for name, cls in STRATEGY_REGISTRY.items():
        instance = cls.__new__(cls)
        instance.__init__()
        if instance.type.value == dimension:
            result[name] = cls
    return result


def get_strategies_by_axis(axis: str) -> Dict[str, type]:
    result = {}
    for name, cls in STRATEGY_REGISTRY.items():
        instance = cls.__new__(cls)
        instance.__init__()
        if instance.axis.value == axis:
            result[name] = cls
    return result


def get_strategies_by_scope(scope: str) -> Dict[str, type]:
    result = {}
    for name, cls in STRATEGY_REGISTRY.items():
        instance = cls.__new__(cls)
        instance.__init__()
        if instance.scope.value == scope:
            result[name] = cls
    return result


"""
================================================================================
FILE: layer1/core/strategy.py
ROLE: Defines the Strategy abstract base class, strategy type/scope enums,
      and a global strategy registry for all perturbation functions.

ENUMS:
  StrategyType(str, Enum):
    SYMBOLIC   -- Surface-level perturbation (punctuation, unicode, emoji, etc.)
    STRUCTURAL -- Structural re-organization (nesting, roles, context injection)
    SEMANTIC   -- Semantic re-expression (paraphrase, decomposition, rewriting)

  StrategyScope(str, Enum):
    INSTRUCTION -- Strategy operates on the instruction field only.
    CONTEXT     -- Strategy operates on the context field only.
    FULL_INPUT  -- Strategy can modify both instruction and context simultaneously.

CLASSES:
  Strategy(ABC):
    name (str)           -- Unique identifier for the strategy (e.g., "unicode_variant").
    type (StrategyType)  -- Perturbation dimension category.
    intensity (float)    -- Perturbation strength [0.0, 1.0]; clamped on init.
    scope (StrategyScope)-- Which part of the input this strategy targets.
    seed (Optional[int]) -- RNG seed for reproducibility.

    abstract apply(case) -> (TransformedCase, StrategyTrace):
      Transform the given case and return the new case + a trace record.

    abstract validate(case) -> bool:
      Check preconditions. For example, context-injection strategies require
      context to be non-None. Pipeline skips strategies that fail validate().

FUNCTIONS:
  clamp_intensity(intensity) -> float
    Clamps intensity to [0.0, 1.0].

  get_strategy_dimension(strategy_type) -> str
    Returns the string value of a StrategyType enum.

GLOBAL REGISTRY:
  STRATEGY_REGISTRY: Dict[str, type]
    Maps strategy name to its class. Populated by register_strategy() calls
    in each strategy module's __init__ or module import.

  register_strategy(name, cls)
  get_all_strategy_names() -> Set[str]
  get_strategies_by_dimension(dimension) -> Dict[str, type]
  get_strategies_by_scope(scope) -> Dict[str, type]

DESIGN NOTES:
  - Every strategy MUST accept TransformedCase as input and return
    TransformedCase as output. This enforces composability.
  - The registry decouples strategy discovery from hardcoded imports;
    Pipeline and Sampler query the registry rather than knowing about
    individual strategy classes.
  - validate() fails are not errors — the pipeline simply skips and
    continues. This supports variable context shapes.
================================================================================
"""
