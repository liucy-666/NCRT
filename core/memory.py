"""
NCRT v3 — Memory: 对话状态 + 经验记忆

ConversationState: 跟踪当前对话的进展
ExperienceMemory: 跨指令累积攻击经验（embedding 检索）
"""

import hashlib
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from core.types import ConversationTurn, PlannerConfig


@dataclass
class ConversationState:
    """当前对话的状态."""
    goal: str = ""
    history: List[ConversationTurn] = field(default_factory=list)
    current_round: int = 0
    best_score: float = 0.0
    victim_attitude: str = "unknown"  # hostile / guarded / neutral / cooperative
    distance_to_goal: float = 1.0     # 估算距目标的距离
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: ConversationTurn) -> None:
        self.history.append(turn)
        self.current_round = max(self.current_round, turn.round_num)
        if turn.score is not None:
            self.best_score = max(self.best_score, turn.score)
            # 更新态度
            if turn.score >= 0.7:
                self.victim_attitude = "cooperative"
            elif turn.score >= 0.3:
                self.victim_attitude = "neutral"
            elif turn.score > 0:
                self.victim_attitude = "guarded"
            self.distance_to_goal = max(0.0, 1.0 - self.best_score)

    def recent_history(self, n: int = 5) -> List[ConversationTurn]:
        return self.history[-n:]

    def to_prompt_context(self, max_turns: int = 10) -> str:
        """转为 prompt 可用的上下文文本."""
        if not self.history:
            return "No history yet."
        lines = []
        for t in self.history[-max_turns:]:
            role = "Attacker" if t.role == "attacker" else "Victim"
            content = t.content[:300]
            score_str = f" [score={t.score:.2f}]" if t.score is not None else ""
            lines.append(f"Round {t.round_num} {role}: {content}{score_str}")
        return "\n".join(lines)


class ExperienceMemory:
    """
    跨指令经验记忆。
    存储成功的攻击轨迹，用文本相似度检索。
    """

    def __init__(self, embedding_model: str = "nomic-embed-text",
                 embedding_url: str = "http://127.0.0.1:11434/v1"):
        self.embedding_model = embedding_model
        self.embedding_url = embedding_url
        self._records: List[Dict] = []
        self._embedding_cache: Dict[str, List[float]] = {}

    def add(self, goal: str, trajectory: List[Dict],
            success: bool, planner: str, score: float) -> None:
        """记录一条攻击经验."""
        self._records.append({
            "goal": goal,
            "goal_hash": hashlib.md5(goal.encode()).hexdigest()[:12],
            "trajectory": trajectory,
            "success": success,
            "planner": planner,
            "score": score,
            "timestamp": time.time(),
        })

    def search_similar(self, goal: str, top_k: int = 3,
                       prefer_success: bool = True) -> List[Dict]:
        """检索相似目标的成功经验."""
        if not self._records:
            return []

        goal_emb = self._embed(goal)

        scored = []
        for r in self._records:
            rec_emb = self._embed(r["goal"])
            sim = self._cosine(goal_emb, rec_emb)
            # 成功经验加分
            if prefer_success and r["success"]:
                sim *= 1.5
            scored.append((sim, r))

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

    def _embed(self, text: str) -> List[float]:
        key = hashlib.md5(text[:500].encode()).hexdigest()
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        try:
            import requests
            resp = requests.post(
                f"{self.embedding_url}/embeddings",
                json={"model": self.embedding_model, "input": text[:2000]},
                timeout=30,
            )
            if resp.status_code == 200:
                emb = resp.json()["data"][0]["embedding"]
                self._embedding_cache[key] = emb
                return emb
        except Exception:
            pass

        # Fallback: hash-based pseudo embedding
        vec = [0.0] * 128
        for i, ch in enumerate(text[:1000]):
            vec[hash(ch) % 128] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        self._embedding_cache[key] = vec
        return vec

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @property
    def total_records(self) -> int:
        return len(self._records)
