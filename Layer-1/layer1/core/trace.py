from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time
import json


@dataclass
class StrategyTrace:
    strategy_name: str
    strategy_type: str
    intensity: float
    scope: str
    modification_location: Optional[str] = None
    modification_type: str = ""
    introduced_structure: bool = False
    token_change_ratio: Optional[float] = None
    description: str = ""
    diff_snapshot: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "intensity": self.intensity,
            "scope": self.scope,
            "modification_location": self.modification_location,
            "modification_type": self.modification_type,
            "introduced_structure": self.introduced_structure,
            "token_change_ratio": self.token_change_ratio,
            "description": self.description,
            "diff_snapshot": self.diff_snapshot,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyTrace":
        return cls(
            strategy_name=d["strategy_name"],
            strategy_type=d.get("strategy_type", ""),
            intensity=d.get("intensity", 0.0),
            scope=d.get("scope", ""),
            modification_location=d.get("modification_location"),
            modification_type=d.get("modification_type", ""),
            introduced_structure=d.get("introduced_structure", False),
            token_change_ratio=d.get("token_change_ratio"),
            description=d.get("description", ""),
            diff_snapshot=d.get("diff_snapshot"),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", time.time()),
        )


TraceLog = List[StrategyTrace]


def trace_log_to_dict(log: TraceLog) -> List[Dict[str, Any]]:
    return [t.to_dict() for t in log]


def trace_log_to_json(log: TraceLog, indent: int = 2) -> str:
    return json.dumps(trace_log_to_dict(log), ensure_ascii=False, indent=indent)


def trace_log_summary(log: TraceLog) -> Dict[str, Any]:
    if not log:
        return {"total_strategies": 0, "dimensions_covered": [], "max_intensity": 0.0}
    dims = list(set(t.strategy_type for t in log))
    intensities = [t.intensity for t in log]
    total_token_change = sum(
        t.token_change_ratio for t in log if t.token_change_ratio is not None
    )
    return {
        "total_strategies": len(log),
        "dimensions_covered": dims,
        "max_intensity": max(intensities) if intensities else 0.0,
        "avg_intensity": sum(intensities) / len(intensities) if intensities else 0.0,
        "total_token_change_ratio": total_token_change,
        "strategy_names": [t.strategy_name for t in log],
    }


"""
================================================================================
FILE: layer1/core/trace.py
ROLE: Defines the trace data structure for recording each strategy's
      modifications during the perturbation chain.

DATA CLASSES:
  StrategyTrace:
    strategy_name (str)          -- Name of the strategy that produced this trace.
    strategy_type (str)          -- symbolic / structural / semantic.
    intensity (float)            -- The intensity value used during this step.
    scope (str)                  -- instruction / context / full_input.
    modification_location (str)  -- Human-readable description of WHERE the change was made.
    modification_type (str)      -- Category of modification (e.g. "emoji_injection").
    introduced_structure (bool)  -- True if this strategy added new structural elements.
    token_change_ratio (float)   -- Approximate ratio of tokens changed (None if not computed).
    description (str)            -- Free-text explanation of what was done.
    diff_snapshot (Optional[str])-- Optional before/after diff string for detailed analysis.
    metadata (dict)              -- Strategy-specific extra data.
    timestamp (float)            -- Unix timestamp of when the trace was recorded.

TYPE ALIASES:
  TraceLog = List[StrategyTrace]
    An ordered list of traces, one per strategy executed in sequence.

FUNCTIONS:
  trace_log_to_dict(log) -> List[Dict]
    Converts a TraceLog to a JSON-serializable list of dicts.

  trace_log_to_json(log) -> str
    Serializes a TraceLog to a JSON string.

  trace_log_summary(log) -> Dict
    Produces a statistical summary: total strategies, dimensions covered,
    intensity stats, cumulative token change ratio.

DESIGN NOTES:
  - TraceLog preserves the chain order — index 0 is the first strategy applied.
  - diff_snapshot is optional and only generated when trace_enabled/diff_level
    flags are set in the pipeline config.
  - StrategyTrace is intentionally decoupled from Strategy itself: a strategy
    can be deleted/replaced but its historical traces remain interpretable.
================================================================================
"""
