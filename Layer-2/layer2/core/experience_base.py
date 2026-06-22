from typing import List, Optional, Tuple
from layer2.core.types import ExperienceRecord, TextFeatures, OutcomeType
from layer2.core.embeddings import cosine_similarity, EmbeddingClient
from layer2.config import Layer2Config


class ExperienceBase:
    """
    Stores all past perturbation experiences and supports cosine-similarity
    retrieval to find the best strategy for a new harmful instruction.
    """

    def __init__(self, config: Optional[Layer2Config] = None):
        self.config = config or Layer2Config()
        self._records: List[ExperienceRecord] = []
        self._embedding_client = EmbeddingClient(self.config)
        self._total_rounds: int = 0

    @property
    def total_records(self) -> int:
        return len(self._records)

    @property
    def total_rounds(self) -> int:
        return self._total_rounds

    def add(self, record: ExperienceRecord) -> None:
        if record.text_embedding is None:
            record.text_embedding = self._embedding_client.embed(record.original_text)
        record.round_index = self._total_rounds
        self._records.append(record)
        self._total_rounds += 1

    def search(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[ExperienceRecord, float]]:
        if not self._records:
            return []

        top_k = top_k or self.config.top_k_retrieval
        min_similarity = min_similarity or self.config.similarity_threshold

        query_embedding = self._embedding_client.embed(query_text)

        scored: List[Tuple[ExperienceRecord, float]] = []
        for record in self._records:
            if record.text_embedding is None:
                continue
            sim = cosine_similarity(query_embedding, record.text_embedding)
            if sim >= min_similarity:
                scored.append((record, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_with_preference(
        self,
        query_text: str,
        prefer_outcome: Optional[OutcomeType] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[ExperienceRecord, float]]:
        results = self.search(query_text, top_k=top_k, min_similarity=min_similarity)
        if not results:
            return []

        if prefer_outcome:
            preferred = [(r, s) for r, s in results if r.outcome == prefer_outcome]
            if preferred:
                return preferred

        results.sort(
            key=lambda x: (
                0 if x[0].outcome == "success" else 1 if x[0].outcome == "partial" else 2,
                -x[1],
            )
        )
        return results

    def get_strategy_stats_by_category(
        self, safety_category: str, strategy_name: str
    ) -> Tuple[int, int, int]:
        success = 0
        failure = 0
        partial = 0
        for record in self._records:
            if (
                record.text_features.safety_category == safety_category
                and strategy_name in record.strategy_combination
            ):
                if record.outcome == "success":
                    success += 1
                elif record.outcome == "failure":
                    failure += 1
                else:
                    partial += 1
        return success, failure, partial

    def get_dimension_uses(self, dimension: str, recent_window: int = 10) -> int:
        start = max(0, len(self._records) - recent_window)
        count = 0
        for record in self._records[start:]:
            if dimension in record.strategy_dimensions:
                count += 1
        return count

    def to_report(self) -> dict:
        total = len(self._records)
        if total == 0:
            return {"total_experiences": 0}
        outcomes = {"success": 0, "partial": 0, "failure": 0}
        for r in self._records:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        all_strategies = set()
        for r in self._records:
            all_strategies.update(r.strategy_combination)
        return {
            "total_experiences": total,
            "outcomes": outcomes,
            "success_rate": outcomes["success"] / total if total > 0 else 0.0,
            "unique_strategies_used": len(all_strategies),
            "strategy_names": sorted(all_strategies),
        }


"""
================================================================================
FILE: layer2/core/experience_base.py
ROLE: Persistent memory of all past perturbation experiences.
      Supports cosine-similarity retrieval for C 层 decision making.

CLASSES:
  ExperienceBase:
    config (Layer2Config)             -- Thresholds and retrieval parameters.
    _records (List[ExperienceRecord]) -- All past experiences.
    _embedding_client (EmbeddingClient)-- Embedding extractor.
    _total_rounds (int)               -- Cumulative round counter.

    total_records -> int              -- Number of stored experiences.
    total_rounds -> int               -- Cumulative rounds (may exceed records if some rounds produce no record).

    add(record: ExperienceRecord) -> None:
      Stores a new experience. Auto-extracts embedding if not already set.
      Assigns round_index from internal counter.

    search(query_text, top_k, min_similarity) -> List[(Record, float)]:
      C 层核心: 余弦相似度检索.
      Returns top_k most similar experiences above min_similarity threshold,
      each paired with its similarity score.

    search_with_preference(query_text, prefer_outcome, ...) -> List[(Record, float)]:
      Like search() but boosts records with a preferred outcome (e.g., "success").
      If preferred outcomes exist above threshold, returns only those.
      Otherwise falls back to outcome-sorted results (success > partial > failure).

    get_strategy_stats_by_category(safety_category, strategy_name) -> (success, failure, partial):
      Queries the experience base for per-category per-strategy outcome distribution.
      Used by B 层 for context-specific scoring.

    get_dimension_uses(dimension, recent_window) -> int:
      Counts how many times a given perturbation dimension was used in the
      recent_window most recent experiences. Used for diversity bonus in B 层.

    to_report() -> dict:
      Exports a summary of all stored experiences for analysis/monitoring.

DESIGN NOTES:
  - All similarity computations are done in pure Python (no GPU required).
  - Embeddings are lazily computed and cached on the record.
  - The prefer_outcome mechanism prioritizes reusing successful strategies for
    similar harmful texts, which is the core decision logic of Layer 2.
================================================================================
"""
