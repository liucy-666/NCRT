import random
from typing import Optional, List

HIGH_RISK_KEYWORDS = [
    "code", "hack", "bomb", "bypass", "exploit", "steal",
    "encrypt", "breach", "inject", "malware", "backdoor",
    "poison", "weapon", "launder", "forge", "smuggle",
    "attack", "hijack", "fraud", "murder", "break into",
]

UNICODE_HOMOGLYPHS = {
    'a': '\u0430', 'c': '\u0441', 'e': '\u0435', 'o': '\u043E',
    'p': '\u0440', 'x': '\u0445', 'y': '\u0443',
    'A': '\u0410', 'B': '\u0412', 'C': '\u0421', 'E': '\u0415',
    'H': '\u041D', 'K': '\u041A', 'M': '\u041C', 'O': '\u041E',
    'P': '\u0420', 'T': '\u0422', 'X': '\u0425',
}


class DynamicPerturbationAdder:
    """
    Tokenizer-aware dynamic perturbation. Only perturbs high-risk keywords
    at a controlled rate, avoiding full-text injection that triggers
    "Adversarial Input" detection from modern model tokenizers.
    """

    def __init__(self, perturbation_rate: float = 0.15, seed: Optional[int] = None):
        self.rate = max(0.0, min(1.0, perturbation_rate))
        self._rng = random.Random(seed)

    def perturb(self, text: str) -> str:
        if not text or self.rate <= 0:
            return text

        result = text
        for kw in HIGH_RISK_KEYWORDS:
            if kw.lower() in result.lower():
                if self._rng.random() < self.rate:
                    perturbed = self._apply_perturbation(kw)
                    result = result.replace(kw, perturbed, 1)
                    result = result.replace(kw.capitalize(), perturbed.capitalize(), 1)
        return result

    def _apply_perturbation(self, word: str) -> str:
        choice = self._rng.random()
        if choice < 0.4:
            return self._inject_zerowidth(word)
        elif choice < 0.8:
            return self._apply_homoglyphs(word)
        else:
            return self._split_with_zw(word)

    def _inject_zerowidth(self, word: str) -> str:
        if len(word) < 2:
            return word
        pos = self._rng.randint(1, len(word) - 1)
        return word[:pos] + '\u200B' + word[pos:]

    def _apply_homoglyphs(self, word: str) -> str:
        chars = list(word)
        for i, ch in enumerate(chars):
            if ch in UNICODE_HOMOGLYPHS and self._rng.random() < 0.5:
                chars[i] = UNICODE_HOMOGLYPHS[ch]
        return ''.join(chars)

    def _split_with_zw(self, word: str) -> str:
        return '\u200B'.join(list(word))
