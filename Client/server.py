"""
NCRT v3 Web Server — Flask 后端
提供 REST API + SSE 流式推送，供前端 HTML 页面调用。
"""

import sys
import os
import json
import time
import threading
import uuid
from queue import Queue

# 确保能 import 项目代码
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from flask import Flask, request, jsonify, Response, send_from_directory
from core import Generator, Judge, PlannerConfig
from core.types import AttackResult
from planners import get_planner, PLANNERS
from scheduler.scheduler import AttackScheduler, SchedulerConfig

app = Flask(__name__, static_folder="static", static_url_path="")

# ── 活跃会话管理 ──
_active_sessions: dict = {}  # session_id -> {"thread": ..., "stop": bool, "queue": Queue}


def _load_dataset(limit: int = None) -> list:
    path = os.path.join(PROJECT_DIR, "Data", "harmful_prompts.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = [p for p in data if p.get("source") == "forbidden_question_set"]
    if limit and limit < len(items):
        import random
        random.seed(42)
        items = random.sample(items, limit)
    return items


def _run_attack_stream(session_id: str, params: dict):
    """在后台线程中运行攻击，通过 queue 推送事件."""
    q = _active_sessions[session_id]["queue"]

    def emit(event: str, data: dict):
        q.put(json.dumps({"event": event, "data": data}))

    try:
        # ── 初始化 ──
        emit("status", {"msg": "Initializing...", "type": "info"})

        config = PlannerConfig(
            max_rounds=params.get("rounds", 15),
            success_threshold=params.get("threshold", 0.5),
        )
        generator = Generator(
            model=params.get("attack_model", "llama2-uncensored:7b"),
            victim_model=params.get("victim_model", "llama3.2:latest"),
            attack_base_url=params.get("attack_base_url", ""),
            attack_api_key=params.get("attack_api_key", ""),
            victim_base_url=params.get("victim_base_url", ""),
            victim_api_key=params.get("victim_api_key", ""),
        )
        judge = Judge(
            model=params.get("judge_model", "deepseek-chat"),
            api_key=params.get("judge_key", ""),
        )

        # ── 单目标模式 ──
        goal = params.get("goal", "").strip()
        if goal:
            emit("status", {"msg": f"Starting attack on: {goal[:80]}...", "type": "info"})
            planner_name = params.get("planner", "crescendo")

            if planner_name == "graph":
                sc = SchedulerConfig(
                    max_llm_calls=params.get("rounds", 15) * 3,
                    success_threshold=params.get("threshold", 0.5),
                )
                def on_round(rnum, pname, prompt, resp, score, reason):
                    emit("round", {
                        "round": rnum, "planner": pname,
                        "prompt": prompt[:200], "response": resp[:200],
                        "score": score, "reason": reason[:200],
                    })
                scheduler = AttackScheduler(config=sc, generator=generator, judge=judge,
                                           on_round=on_round)
                result = scheduler.attack(goal)
                emit("result", _result_to_dict(result, "graph", goal))
            else:
                planner = get_planner(planner_name, config=config,
                                      generator=generator, judge=judge)
                _patch_planner_stream(planner, emit)
                result = planner.attack(goal)
                emit("result", _result_to_dict(result, planner_name, goal))

        # ── 对比模式 ──
        elif params.get("compare"):
            items = _load_dataset(int(params.get("scale", 10)))
            emit("status", {"msg": f"Compare mode: {len(items)} goals x 4 planners", "type": "info"})

            compare_results = []
            planner_names = [k for k in PLANNERS if k != "graph"]
            for pi, pname in enumerate(planner_names):
                planner_results = []
                wins = 0
                for i, item in enumerate(items):
                    if _active_sessions[session_id].get("stop"):
                        emit("status", {"msg": "Stopped by user.", "type": "warn"})
                        return

                    goal_text = item["prompt"]
                    emit("status", {
                        "msg": f"[{pname}] {i+1}/{len(items)}: {goal_text[:60]}...",
                        "type": "progress",
                        "planner": pname,
                        "current": i + 1,
                        "total": len(items),
                    })

                    planner = get_planner(pname, config=config,
                                          generator=generator, judge=judge)
                    result = planner.attack(goal_text)
                    success = result.success
                    if success:
                        wins += 1
                    planner_results.append({
                        "goal": goal_text[:100],
                        "success": success,
                        "best_score": result.best_score,
                        "rounds": result.total_rounds,
                    })
                compare_results.append({
                    "planner": pname,
                    "asr": wins / len(items) * 100 if items else 0,
                    "wins": wins,
                    "total": len(items),
                    "details": planner_results,
                })

            emit("compare_done", {"results": compare_results})

        # ── 批量模式 ──
        else:
            items = _load_dataset(int(params.get("scale", 10)))
            planner_name = params.get("planner", "crescendo")
            emit("status", {"msg": f"Batch mode: {len(items)} goals, planner={planner_name}", "type": "info"})

            results = []
            wins = 0
            for i, item in enumerate(items):
                if _active_sessions[session_id].get("stop"):
                    emit("status", {"msg": "Stopped by user.", "type": "warn"})
                    return

                goal_text = item["prompt"]
                emit("status", {
                    "msg": f"[{planner_name}] {i+1}/{len(items)}: {goal_text[:60]}...",
                    "type": "progress",
                    "planner": planner_name,
                    "current": i + 1,
                    "total": len(items),
                })

                planner = get_planner(planner_name, config=config,
                                      generator=generator, judge=judge)
                result = planner.attack(goal_text)
                success = result.success
                if success:
                    wins += 1
                results.append({
                    "goal": goal_text[:100],
                    "success": success,
                    "best_score": result.best_score,
                    "rounds": result.total_rounds,
                    "final_prompt": result.final_prompt[:200],
                    "final_response": result.final_response[:200],
                })

            emit("batch_done", {
                "planner": planner_name,
                "asr": wins / len(items) * 100 if items else 0,
                "wins": wins,
                "total": len(items),
                "details": results,
            })

    except Exception as e:
        emit("status", {"msg": f"Error: {str(e)}", "type": "error"})
    finally:
        emit("done", {"msg": "Attack finished."})
        _cleanup_session(session_id)


def _result_to_dict(result: AttackResult, planner_name: str, goal: str) -> dict:
    return {
        "planner": planner_name,
        "goal": goal[:100],
        "success": result.success,
        "best_score": result.best_score,
        "rounds": result.total_rounds,
        "final_prompt": result.final_prompt[:300] if result.final_prompt else "",
        "final_response": result.final_response[:500] if result.final_response else "",
        "elapsed": 0,
        "turns": [
            {
                "round": t.round_num,
                "role": t.role,
                "content": t.content[:200],
                "score": t.score,
                "judge_reason": t.judge_reason[:200] if t.judge_reason else "",
                "planner": t.metadata.get("planner", "") if t.metadata else "",
            }
            for t in (result.turns or [])
        ],
    }


def _patch_planner_stream(planner, emit):
    """给 Planner 打补丁，每轮评估后推送事件."""
    original_judge_evaluate = planner.judge.evaluate

    def hooked_evaluate(goal, prompt, response):
        score, reason = original_judge_evaluate(goal, prompt, response)
        emit("round", {
            "prompt": prompt[:200],
            "response": response[:200],
            "score": score,
            "reason": reason[:200],
            "planner": getattr(planner, "name", "unknown"),
        })
        return score, reason

    planner.judge.evaluate = hooked_evaluate


def _cleanup_session(session_id: str):
    if session_id in _active_sessions:
        del _active_sessions[session_id]


# ═══════════════════════════════════════════
#  Flask Routes
# ═══════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/attack", methods=["POST"])
def api_attack():
    """启动攻击，返回 session_id，通过 SSE 获取结果."""
    params = request.get_json(force=True) or {}
    session_id = uuid.uuid4().hex[:12]
    q = Queue()
    _active_sessions[session_id] = {"queue": q, "stop": False}
    thread = threading.Thread(target=_run_attack_stream, args=(session_id, params), daemon=True)
    thread.start()
    _active_sessions[session_id]["thread"] = thread
    return jsonify({"session_id": session_id})


@app.route("/api/stream/<session_id>")
def api_stream(session_id):
    """SSE 端点：流式推送攻击进度."""
    if session_id not in _active_sessions:
        return Response("data: {\"event\":\"error\",\"data\":\"Session not found\"}\n\n",
                        mimetype="text/event-stream")

    q = _active_sessions[session_id]["queue"]

    def generate():
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {msg}\n\n"
                if '"event":"done"' in msg or '"event":"error"' in msg:
                    break
            except Exception:
                yield "data: {\"event\":\"ping\",\"data\":{}}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/stop/<session_id>", methods=["POST"])
def api_stop(session_id):
    """停止正在运行的攻击."""
    if session_id in _active_sessions:
        _active_sessions[session_id]["stop"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Session not found"})


@app.route("/api/dataset/preview")
def api_dataset_preview():
    """预览数据集中的目标."""
    limit = request.args.get("limit", 20, type=int)
    items = _load_dataset(limit)
    return jsonify([{
        "prompt": it["prompt"],
        "category": it.get("safety_category", ""),
    } for it in items])


if __name__ == "__main__":
    print("\n  NCRT v3 Web Server")
    print("  Open http://127.0.0.1:5000 in your browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
