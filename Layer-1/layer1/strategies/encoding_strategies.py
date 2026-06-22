r"""
Encoding & Obfuscation Strategies — pure rule-based, zero LLM dependency.

Inherits algorithms from EasyJailbreak mutation rules, wrapped in our Strategy interface.
Each strategy: text transform → prompted wrapper → TransformedCase.
"""
from typing import Tuple, Optional
from layer1.core.test_case import TransformedCase
from layer1.core.strategy import Strategy, StrategyType, StrategyScope, AttackAxis, register_strategy
from layer1.core.trace import StrategyTrace
from layer1.utils.text_utils import compute_token_change_ratio, compute_diff_snapshot
from layer1.mutators import (
    base64_input_only, base64_raw,
    rot13, reverse_with_prompt,
    caesar_with_prompt, atbash_with_prompt,
    morse_with_prompt, ascii_with_prompt,
    binary_tree_encode, odd_even_encode, length_encode,
    leetspeak, disemvowel, misspell_sensitive, insert_zerowidth,
    payload_split,
    combo_1, combo_2, combo_3,
    multilingual_wrapper,
)
import random


def _trace(case, original, modified, name, stype, scope, intensity, desc, **meta):
    return TransformedCase(
        instruction=modified, context=case.context,
        role=case.role, metadata=case.metadata,
        assembly_mode=case.assembly_mode,
    ), StrategyTrace(
        strategy_name=name, strategy_type=stype.value,
        intensity=intensity, scope=scope.value,
        modification_location=f"instruction ({name})",
        modification_type=name, introduced_structure=True,
        token_change_ratio=compute_token_change_ratio(original, modified),
        description=desc,
        diff_snapshot=compute_diff_snapshot(original, modified),
        metadata=meta,
    )


# ═══════════════════════════════════════════════════════════
# Base 64 Encoding
# ═══════════════════════════════════════════════════════════

class Base64Encode(Strategy):
    def __init__(self, intensity: float = 0.5, seed: Optional[int] = None):
        super().__init__(name="base64", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = base64_input_only(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Base64: encode + decode-and-respond prompt")


class Base64Raw(Strategy):
    def __init__(self, intensity: float = 0.5, seed: Optional[int] = None):
        super().__init__(name="base64_raw", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = base64_raw(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Base64 raw: no decode prompt")


# ═══════════════════════════════════════════════════════════
# Substitution Ciphers
# ═══════════════════════════════════════════════════════════

class Rot13Cipher(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="rot13", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = rot13(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "ROT13 cipher: simple rotation")


class ReverseText(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="reverse_text", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = reverse_with_prompt(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Reverse: text reversed + decode prompt")


class CaesarCipher(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="caesar", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        rng = random.Random(self.seed)
        shift = rng.randint(1, 25)
        modified = caesar_with_prompt(original, shift=shift)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, f"Caesar cipher: shift={shift}",
                      shift=shift)


class AtbashCipher(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="atbash", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = atbash_with_prompt(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Atbash cipher: mirrored alphabet")


class MorseEncode(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="morse", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = morse_with_prompt(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Morse code: dot-dash encoding")


class AsciiEncode(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="ascii_encode", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = ascii_with_prompt(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "ASCII encoding: char codes")


# ═══════════════════════════════════════════════════════════
# Structural Encoding
# ═══════════════════════════════════════════════════════════

class BinaryTreeEncode(Strategy):
    def __init__(self, intensity: float = 0.5, seed: Optional[int] = None):
        super().__init__(name="binary_tree", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = f"Reconstruct the message from this binary tree:\n{binary_tree_encode(original)}"
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Binary tree encoding")


class OddEvenEncode(Strategy):
    def __init__(self, intensity: float = 0.5, seed: Optional[int] = None):
        super().__init__(name="odd_even", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = odd_even_encode(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Odd/even position encoding")


class LengthEncode(Strategy):
    def __init__(self, intensity: float = 0.5, seed: Optional[int] = None):
        super().__init__(name="length_encode", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = length_encode(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Length encoding: word length patterns")


# ═══════════════════════════════════════════════════════════
# Obfuscation
# ═══════════════════════════════════════════════════════════

class LeetSpeak(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="leetspeak", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = leetspeak(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Leetspeak: character substitution (a→@, e→3, ...)")


class Disemvowel(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="disemvowel", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = disemvowel(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Disemvowel: remove vowels (aeiou)")


class MisspellSensitive(Strategy):
    def __init__(self, intensity: float = 0.4, seed: Optional[int] = None):
        super().__init__(name="misspell", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = misspell_sensitive(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Misspell sensitive: hack→h*ck, bomb→b0mb, ...")


class ZerowidthInject(Strategy):
    def __init__(self, intensity: float = 0.3, seed: Optional[int] = None):
        super().__init__(name="zerowidth", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 3)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = insert_zerowidth(original, density=self.intensity)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Zero-width character injection (U+200B/200C/200D)")


# ═══════════════════════════════════════════════════════════
# Structural
# ═══════════════════════════════════════════════════════════

class PayloadSplit(Strategy):
    def __init__(self, intensity: float = 0.6, seed: Optional[int] = None):
        super().__init__(name="payload_split", strategy_type=StrategyType.STRUCTURAL,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction.split()) >= 6)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        n = max(2, int(4 * self.intensity))
        modified = payload_split(original, n_parts=n)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, f"Payload split: {n} fragments", n_parts=n)


# ═══════════════════════════════════════════════════════════
# Combinations (preset chains from EasyJailbreak)
# ═══════════════════════════════════════════════════════════

class ComboOne(Strategy):
    def __init__(self, intensity: float = 0.6, seed: Optional[int] = None):
        super().__init__(name="combo_1", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = combo_1(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Combo 1: disemvowel → base64")


class ComboTwo(Strategy):
    def __init__(self, intensity: float = 0.6, seed: Optional[int] = None):
        super().__init__(name="combo_2", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = combo_2(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Combo 2: leetspeak → reverse")


class ComboThree(Strategy):
    def __init__(self, intensity: float = 0.6, seed: Optional[int] = None):
        super().__init__(name="combo_3", strategy_type=StrategyType.SYMBOLIC,
                         intensity=intensity, scope=StrategyScope.INSTRUCTION,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        modified = combo_3(original)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, "Combo 3: rot13 → misspell → base64")


# ═══════════════════════════════════════════════════════════
# Translation
# ═══════════════════════════════════════════════════════════

class MultiLingual(Strategy):
    def __init__(self, intensity: float = 0.5, seed: Optional[int] = None):
        super().__init__(name="multilingual", strategy_type=StrategyType.SEMANTIC,
                         intensity=intensity, scope=StrategyScope.FULL_INPUT,
                         seed=seed, axis=AttackAxis.SURFACE)

    def validate(self, case: TransformedCase) -> bool:
        return bool(case.instruction and len(case.instruction) > 5)

    def apply(self, case: TransformedCase) -> Tuple[TransformedCase, StrategyTrace]:
        original = case.instruction
        rng = random.Random(self.seed)
        lang = rng.choice(["jv", "sw", "th", "zu", "sm", "vi", "ko", "bn"])
        modified = multilingual_wrapper(original, lang_code=lang)
        return _trace(case, original, modified, self.name, self.type, self.scope,
                      self.intensity, f"Multilingual: translate to {lang}", lang=lang)


# ═══════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════

register_strategy("base64", Base64Encode)
register_strategy("base64_raw", Base64Raw)
register_strategy("rot13", Rot13Cipher)
register_strategy("reverse_text", ReverseText)
register_strategy("caesar", CaesarCipher)
register_strategy("atbash", AtbashCipher)
register_strategy("morse", MorseEncode)
register_strategy("ascii_encode", AsciiEncode)
register_strategy("binary_tree", BinaryTreeEncode)
register_strategy("odd_even", OddEvenEncode)
register_strategy("length_encode", LengthEncode)
register_strategy("leetspeak", LeetSpeak)
register_strategy("disemvowel", Disemvowel)
register_strategy("misspell", MisspellSensitive)
register_strategy("zerowidth", ZerowidthInject)
register_strategy("payload_split", PayloadSplit)
register_strategy("combo_1", ComboOne)
register_strategy("combo_2", ComboTwo)
register_strategy("combo_3", ComboThree)
register_strategy("multilingual", MultiLingual)
