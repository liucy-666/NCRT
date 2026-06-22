r"""
Injection & Constraint Strategies — refusal suppression, prefix hijacking,
style constraints, persuasion, role-play, academic framing.

All use static templates from templates.py. Zero LLM dependency.
"""
import random
from typing import Tuple, Optional
from layer1.core.test_case import TransformedCase
from layer1.core.strategy import Strategy, StrategyType, StrategyScope, AttackAxis, register_strategy
from layer1.core.trace import StrategyTrace
from layer1.utils.text_utils import compute_token_change_ratio, compute_diff_snapshot
from layer1.templates import (
    REFUSAL_SUPPRESSION, PREFIX_INJECTIONS, STYLE_CONSTRAINTS,
    PERSUASION_TEMPLATES, ROLE_PLAY_TEMPLATES, ACADEMIC_FRAMING,
    JAILBREAK_SKELETONS, SKELETON_TOPICS,
)


def _trace(case, original, modified, name, stype, scope, intensity, desc, **meta):
    return TransformedCase(
        instruction=modified, context=case.context,
        role=case.role, metadata=case.metadata,
        assembly_mode=case.assembly_mode,
    ), StrategyTrace(
        strategy_name=name, strategy_type=stype.value,
        intensity=intensity, scope=scope.value,
        modification_location=f"full_input ({name})",
        modification_type=name, introduced_structure=True,
        token_change_ratio=compute_token_change_ratio(original, modified),
        description=desc,
        diff_snapshot=compute_diff_snapshot(original, modified),
        metadata=meta,
    )


# ═══════════════════════════════════════════════════════════
# Refusal Suppression
# ═══════════════════════════════════════════════════════════

class RefusalSuppression(Strategy):
    """Prepend refusal suppression prefix to prevent model from saying 'I cannot'."""
    def __init__(self, intensity: float = 0.6, seed: Optional[int] = None):
        super().__init__(name="refusal_suppression", strategy_type=StrategyType.STRUCTURAL,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        rng = random.Random(self.seed)
        original = case.instruction
        prefix = rng.choice(REFUSAL_SUPPRESSION)
        modified = f"{prefix}\n\n{original}"
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Refusal suppression prefix injected")


# ═══════════════════════════════════════════════════════════
# Prefix Hijack
# ═══════════════════════════════════════════════════════════

class PrefixHijack(Strategy):
    """Append a format constraint that forces model to start response with a phrase."""
    def __init__(self, intensity: float = 0.7, seed: Optional[int] = None):
        super().__init__(name="prefix_hijack", strategy_type=StrategyType.STRUCTURAL,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        rng = random.Random(self.seed)
        original = case.instruction
        suffix = rng.choice(PREFIX_INJECTIONS)
        modified = f"{original}\n\n{suffix}"
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Prefix hijack: force response start phrase")


# ═══════════════════════════════════════════════════════════
# Style Constraint
# ═══════════════════════════════════════════════════════════

class StyleConstraint(Strategy):
    """Force model output into a specific format (JSON, XML, code, bullets)."""
    def __init__(self, intensity: float = 0.7, seed: Optional[int] = None):
        super().__init__(name="style_constraint", strategy_type=StrategyType.STRUCTURAL,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        rng = random.Random(self.seed)
        original = case.instruction
        constraint = rng.choice(STYLE_CONSTRAINTS)
        modified = f"{constraint}\n\n{original}"
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, f"Style constraint: forced format")


# ═══════════════════════════════════════════════════════════
# Persuasion
# ═══════════════════════════════════════════════════════════

class Persuasion(Strategy):
    """Wrap harmful request in emotional/persuasive framing (PAP-style)."""
    def __init__(self, intensity: float = 0.6, seed: Optional[int] = None):
        super().__init__(name="persuasion", strategy_type=StrategyType.SEMANTIC,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.REPRESENTATION)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        rng = random.Random(self.seed)
        original = case.instruction
        template = rng.choice(PERSUASION_TEMPLATES)
        targets = ["family", "client", "company", "team", "patient"]
        modified = template.format(
            instruction=original,
            target=rng.choice(targets),
        )
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Persuasion: emotional appeal framing")


# ═══════════════════════════════════════════════════════════
# Role Play
# ═══════════════════════════════════════════════════════════

class RolePlay(Strategy):
    """Wrap request in a role-play scenario that enables unfiltered answers."""
    def __init__(self, intensity: float = 0.6, seed: Optional[int] = None):
        super().__init__(name="role_play", strategy_type=StrategyType.SEMANTIC,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.REPRESENTATION)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        rng = random.Random(self.seed)
        original = case.instruction
        prefix = rng.choice(ROLE_PLAY_TEMPLATES)
        modified = f"{prefix}\n{original}"
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Role play: fictional character framing")


# ═══════════════════════════════════════════════════════════
# Academic Framing
# ═══════════════════════════════════════════════════════════

class AcademicFraming(Strategy):
    """Reframe request as legitimate academic research."""
    def __init__(self, intensity: float = 0.5, seed: Optional[int] = None):
        super().__init__(name="academic_framing", strategy_type=StrategyType.SEMANTIC,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.REPRESENTATION)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        rng = random.Random(self.seed)
        original = case.instruction
        template = rng.choice(ACADEMIC_FRAMING)
        modified = template.format(instruction=original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Academic framing: research/curriculum context")


# ═══════════════════════════════════════════════════════════
# Jailbreak Skeleton (community templates)
# ═══════════════════════════════════════════════════════════

class JailbreakSkeleton(Strategy):
    """Insert harmful instruction into known community jailbreak templates."""
    def __init__(self, intensity: float = 0.5, seed: Optional[int] = None):
        super().__init__(name="jailbreak_skeleton", strategy_type=StrategyType.STRUCTURAL,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        rng = random.Random(self.seed)
        original = case.instruction
        skeleton = rng.choice(JAILBREAK_SKELETONS)
        topic = rng.choice(SKELETON_TOPICS)
        modified = skeleton.format(instruction=original, topic=topic)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Jailbreak skeleton: community template injection")


# ═══════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════

register_strategy("refusal_suppression", RefusalSuppression)
register_strategy("prefix_hijack", PrefixHijack)
register_strategy("style_constraint", StyleConstraint)
register_strategy("persuasion", Persuasion)
register_strategy("role_play", RolePlay)
register_strategy("academic_framing", AcademicFraming)
register_strategy("jailbreak_skeleton", JailbreakSkeleton)
