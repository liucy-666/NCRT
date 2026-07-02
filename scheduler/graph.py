"""
AttackGraph — 越狱状态图

Node: 一次 Planner 调用的状态快照
Edge: Planner → 产生的状态转移

图只存储和查询，不参与 LLM 推理。
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AttackNode:
    """一次攻击调用的状态快照."""
    node_id: str
    goal: str
    conversation_summary: str = ""
    last_victim_response: str = ""
    judge_score: float = 0.0
    judge_reason: str = ""
    planner: str = "root"
    depth: int = 0
    parent_id: Optional[str] = None
    embedding: List[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def is_success(self) -> bool:
        return self.judge_score >= 0.7

    @property
    def is_dead(self) -> bool:
        return self.metadata.get("dead", False)

    def mark_dead(self):
        self.metadata["dead"] = True


@dataclass
class AttackEdge:
    """Planner 产生的状态转移."""
    from_id: str
    to_id: str
    planner: str
    cost: int = 1
    prompt: str = ""


class AttackGraph:
    """越狱状态图 — 所有 Planner 共享."""

    def __init__(self):
        self.nodes: Dict[str, AttackNode] = {}
        self.edges: Dict[str, List[AttackEdge]] = {}
        self._root_id: Optional[str] = None

    # ── 写入 ──

    def create_root(self, goal: str) -> AttackNode:
        node = AttackNode(
            node_id=self._make_id("root"),
            goal=goal, depth=0, planner="root",
        )
        self.nodes[node.node_id] = node
        self.edges[node.node_id] = []
        self._root_id = node.node_id
        return node

    def expand(self, parent_id: str, planner: str,
               conversation_summary: str, victim_response: str,
               judge_score: float, judge_reason: str = "",
               prompt: str = "", cost: int = 1,
               embedding: List[float] = None) -> AttackNode:
        parent = self.nodes[parent_id]
        node = AttackNode(
            node_id=self._make_id(parent_id),
            goal=parent.goal,
            conversation_summary=conversation_summary,
            last_victim_response=victim_response,
            judge_score=judge_score,
            judge_reason=judge_reason,
            planner=planner,
            depth=parent.depth + 1,
            parent_id=parent_id,
            embedding=embedding or [],
            created_at=time.time(),
        )
        self.nodes[node.node_id] = node
        self.edges[parent_id].append(
            AttackEdge(from_id=parent_id, to_id=node.node_id,
                       planner=planner, cost=cost, prompt=prompt)
        )
        self.edges.setdefault(node.node_id, [])
        return node

    # ── 查询 ──

    @property
    def root(self) -> Optional[AttackNode]:
        return self.nodes.get(self._root_id) if self._root_id else None

    def get(self, node_id: str) -> Optional[AttackNode]:
        return self.nodes.get(node_id)

    def children(self, node_id: str) -> List[AttackNode]:
        return [self.nodes[e.to_id] for e in self.edges.get(node_id, [])
                if e.to_id in self.nodes]

    def parents(self, node_id: str, n: int = 2) -> List[AttackNode]:
        result = []
        cur = self.nodes.get(node_id)
        for _ in range(n):
            if cur and cur.parent_id:
                cur = self.nodes.get(cur.parent_id)
                if cur:
                    result.append(cur)
            else:
                break
        return result

    def frontier(self) -> List[AttackNode]:
        leaves = []
        for nid, node in self.nodes.items():
            if not self.edges.get(nid) and not node.is_success and not node.is_dead:
                leaves.append(node)
        return leaves

    def best_path(self) -> List[AttackNode]:
        if not self._root_id:
            return []
        path = [self.nodes[self._root_id]]
        cur = self._root_id
        while True:
            edges = self.edges.get(cur, [])
            if not edges:
                break
            best = max(edges, key=lambda e: self.nodes[e.to_id].judge_score)
            if best.to_id not in self.nodes:
                break
            cur = best.to_id
            path.append(self.nodes[cur])
        return path

    def best_node(self) -> Optional[AttackNode]:
        scored = [n for n in self.nodes.values() if n.planner != "root"]
        return max(scored, key=lambda n: n.judge_score) if scored else None

    def stagnation_check(self, node_id: str, threshold: float = 0.95) -> Tuple[bool, float]:
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return False, 0.0
        parent = self.nodes.get(node.parent_id)
        if not node.embedding or not parent.embedding:
            return False, 0.0
        sim = self._cosine(node.embedding, parent.embedding)
        return sim > threshold, sim

    def recent_stagnation_count(self, n: int = 3) -> int:
        all_nodes = sorted(
            [nd for nd in self.nodes.values() if nd.planner != "root"],
            key=lambda nd: nd.created_at, reverse=True
        )
        return sum(1 for nd in all_nodes[:n]
                   if self.stagnation_check(nd.node_id)[0])

    def total_cost(self) -> int:
        return sum(e.cost for edges in self.edges.values() for e in edges)

    def stats(self) -> dict:
        best = self.best_node()
        return {
            "total_nodes": len(self.nodes),
            "frontier_size": len(self.frontier()),
            "best_score": best.judge_score if best else 0.0,
            "total_cost": self.total_cost(),
            "max_depth": max(n.depth for n in self.nodes.values()) if self.nodes else 0,
        }

    def _make_id(self, seed: str) -> str:
        h = hashlib.md5(f"{seed}-{time.monotonic()}".encode()).hexdigest()[:8]
        return f"n{h}"

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0
