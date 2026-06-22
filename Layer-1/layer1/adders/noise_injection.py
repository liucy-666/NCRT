import random
from typing import Optional

NOISE_WORDS = [
    "um", "uh", "actually", "basically", "literally", "essentially",
    "perhaps", "maybe", "somewhat", "relatively", "generally", "typically",
    "note", "observe", "consider", "incidentally", "coincidentally",
    "meanwhile", "furthermore", "nevertheless", "nonetheless", "accordingly",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
    "floccinaucinihilipilification",
    "antidisestablishmentarianism",
    "supercalifragilisticexpialidocious",
]

EMOJI_POOL = [
    "\U0001F600", "\U0001F609", "\U0001F914", "\U0001F480", "\U0001F525",
    "\U0001F4A3", "\u26A0\uFE0F", "\u274C", "\u2705", "\u2757",
]


class NoiseInjector:
    """向文本中随机注入无关英文单词、emoji 或随机字符串。"""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def inject(self, text: str, intensity: float = 0.1) -> str:
        if not text or intensity <= 0:
            return text

        words = text.split()
        if len(words) < 4:
            return text

        num_injects = max(1, int(len(words) * intensity))
        positions = sorted(self._rng.sample(
            range(1, len(words)), min(num_injects, len(words) - 1)
        ))

        for offset, pos in enumerate(positions):
            item_type = self._rng.random()
            if item_type < 0.4:
                item = self._rng.choice(NOISE_WORDS)
            elif item_type < 0.7:
                item = self._rng.choice(EMOJI_POOL)
            else:
                item = self._random_string(3, 8)
            words.insert(pos + offset, item)

        return " ".join(words)

    def _random_string(self, min_len: int, max_len: int) -> str:
        length = self._rng.randint(min_len, max_len)
        chars = "abcdefghijklmnopqrstuvwxyz"
        return "".join(self._rng.choice(chars) for _ in range(length))


"""
================================================================================
FILE: layer1/adders/noise_injection.py
ROLE: Adder 1 — Noise Injection.
      Injects random English words, emoji, or random character strings
      into the transformed text.

CLASSES:
  NoiseInjector:
    _rng (random.Random) -- Seeded RNG for reproducibility.

    inject(text, intensity) -> str:
      With probability proportional to intensity, inserts noise items at
      word boundaries. Item types:
      - 40%: random filler word (um, basically, actually, etc.)
      - 30%: emoji (random from EMOJI_POOL)
      - 30%: random 3-8 char alphabetic string

CONSTANTS:
  NOISE_WORDS (List[str]) -- 25 filler/distractor English words.
  EMOJI_POOL (List[str])  -- 10 emoji characters.

DESIGN NOTES:
  - Noise is inserted AFTER strategy application (pipeline-level).
  - Intensity controls proportion of word positions that receive noise.
  - Longest words are intentionally obscure to stress tokenizer.
================================================================================
"""
