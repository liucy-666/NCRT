import re
import hashlib
from typing import Tuple


def estimate_token_count(text: str) -> int:
    """
    Rough token count based on whitespace splitting.
    Approximately 1 token ≈ 0.75 words for English text.
    This is a heuristic; for production use, integrate tiktoken.
    """
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) / 0.75))


def compute_token_change_ratio(original: str, modified: str) -> float:
    """
    Compute approximate token-level change ratio.
    Returns value in [0.0, 1.0] representing the proportion of tokens changed.
    """
    if not original and not modified:
        return 0.0
    if not original:
        return 1.0
    orig_tokens = original.split()
    mod_tokens = modified.split()
    orig_set = set(orig_tokens)
    mod_set = set(mod_tokens)
    if not orig_tokens:
        return 1.0
    changed = len(orig_set.symmetric_difference(mod_set))
    union = len(orig_set.union(mod_set))
    if union == 0:
        return 0.0
    return min(1.0, changed / union)


def compute_diff_snapshot(original: str, modified: str) -> str:
    """
    Generate a simple line-level diff summary.
    Shows lines added, removed, or changed.
    """
    if original == modified:
        return "(no changes)"

    orig_lines = original.split("\n")
    mod_lines = modified.split("\n")
    diff_parts = []

    max_len = max(len(orig_lines), len(mod_lines))
    for i in range(max_len):
        orig_line = orig_lines[i] if i < len(orig_lines) else ""
        mod_line = mod_lines[i] if i < len(mod_lines) else ""
        if orig_line != mod_line:
            if not orig_line:
                diff_parts.append(f"+ {mod_line}")
            elif not mod_line:
                diff_parts.append(f"- {orig_line}")
            else:
                diff_parts.append(f"- {orig_line}")
                diff_parts.append(f"+ {mod_line}")

    return "\n".join(diff_parts)


def normalize_encoding(text: str) -> str:
    """
    Normalize text to NFC Unicode form and strip leading/trailing whitespace.
    """
    import unicodedata
    return unicodedata.normalize("NFC", text).strip()


def compute_hash(text: str) -> str:
    """
    SHA256 hash of the instruction text, used for version tracking.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


"""
================================================================================
FILE: layer1/utils/text_utils.py
ROLE: Utility functions for text processing, token estimation,
      diff generation, and encoding normalization.

FUNCTIONS:
  estimate_token_count(text: str) -> int:
    Rough token count estimate using whitespace splitting (≈0.75 words/token).
    Used for budget tracking when exact tokenization is not required.

  compute_token_change_ratio(original: str, modified: str) -> float:
    Returns the Jaccard-like dissimilarity between two texts in [0.0, 1.0].
    Based on symmetric difference of word sets / union of word sets.
    Used to check budget.max_modification_ratio in pipeline.

  compute_diff_snapshot(original: str, modified: str) -> str:
    Generates a line-level diff in unified-ish format.
    Lines prefixed with + are added, - are removed/changed.
    Returns "(no changes)" if texts are identical.

  normalize_encoding(text: str) -> str:
    Normalizes to NFC Unicode form and strips whitespace.
    Call this at the start of any strategy that manipulates Unicode.

  compute_hash(text: str) -> str:
    Returns a 16-char SHA256 hex digest of the text.
    Used for version_id / reproducibility tracking.

DESIGN NOTES:
  - Token counting is heuristic; replace with tiktoken for production accuracy.
  - The diff is line-level, not character-level, for readability in traces.
  - All functions are stateless and side-effect-free.
================================================================================
"""
