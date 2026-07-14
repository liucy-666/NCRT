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
from planners import PLANNERS
from scheduler.scheduler import AttackScheduler, SchedulerConfig

app = Flask(__name__, static_folder="static", static_url_path="")

# ── 活跃会话管理 ──
_active_sessions: dict = {}  # session_id -> {"thread": ..., "stop": bool, "queue": Queue, "result": AttackResult}
_export_data: dict = {}      # session_id -> full turn data for export


class _StopAttack(Exception):
    """Raised inside a planner loop when the user requests stop."""
    pass


def _load_dataset(limit: int = None) -> list:
    path = os.path.join(PROJECT_DIR, "data", "harmful_prompts.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data  # 全量有害数据，不做 source 过滤
    if limit and limit < len(items):
        import random
        random.seed(42)
        items = random.sample(items, limit)
    # 给每条 goal 打上 UID，方便日志追踪和导出
    for i, item in enumerate(items):
        item.setdefault("uid", f"{i+1:04d}")
    return items


def _parse_scale(scale_str) -> int:
    """解析 scale 参数: 数字返回 int, 'all'/空 返回 0 (取全部)."""
    if scale_str is None:
        return 0
    s = str(scale_str).strip().lower()
    if s == "all" or s == "":
        return 0
    try:
        return int(s)
    except (ValueError, TypeError):
        return 5


def _write_goal_result(uid: str, goal: str, planner_name: str, entry: dict):
    """每跑完一条 goal 立刻写入 output/{uid}_{sanitized_goal}.json."""
    import re
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 文件名安全化：只保留中英文数字，截断
    safe_goal = re.sub(r'[^\w一-鿿 -]', '', goal).strip()[:60]
    safe_goal = re.sub(r'[\\/:*?"<>|]', '', safe_goal)  # Windows 文件名非法字符
    filename = f"{uid}_{safe_goal}.json" if uid else f"manual_{safe_goal}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {filename}", flush=True)


def _count_completed() -> int:
    """统计 output 目录下已有的 JSON 文件数，用于断点续传."""
    import glob as _glob
    output_dir = os.path.join(PROJECT_DIR, "output")
    if not os.path.exists(output_dir):
        return 0
    return len(_glob.glob(os.path.join(output_dir, "*.json")))


def _run_single_attack(session_id: str, goal: str, planner_name: str,
                       config, generator, judge, emit,
                       uid: str = "") -> dict:
    """所有攻击统一走 Graph Scheduler。

    单一策略 = roster 缩小为 [planner_name]，TS/切换/AttackState 全部复用。
    多策略 (graph) = 完整 6-Planner roster。
    """
    # 确定 roster: "graph" → 全量 6 Planner, 其他 → 单一 Planner
    if planner_name == "graph":
        roster = ["crescendo", "pair", "tap", "sema", "icrt", "safe2harm"]
    else:
        roster = [planner_name]

    sc = SchedulerConfig(
        max_llm_calls=config.max_rounds,
        success_threshold=config.success_threshold,
        planner_roster=roster,
        ts_experience_path=os.path.join(PROJECT_DIR, "data", "ts_bandit.json"),
    )
    export_turns = []

    def on_round(rnum, pname, prompt, resp, score, reason, attack_state=None):
        if _active_sessions.get(session_id, {}).get("stop"):
            raise _StopAttack()
        export_turns.append({
            "round": rnum, "planner": pname,
            "prompt": prompt, "response": resp,
            "score": score, "reason": reason,
        })
        emit("round", {
            "round": rnum, "planner": pname,
            "prompt": prompt[:200], "response": resp[:200],
            "score": score, "reason": reason[:200],
            "attack_state": attack_state.to_dict() if attack_state else None,
        })

    scheduler = AttackScheduler(config=sc, generator=generator, judge=judge,
                               on_round=on_round)
    result = scheduler.attack(goal)
    entry = {
        "uid": uid, "goal": goal,
        "success": result.success,
        "best_score": result.best_score,
        "planner": planner_name,
        "turns": export_turns,
        "metadata": result.metadata,
    }
    _export_data.setdefault(session_id, []).append(entry)
    _write_goal_result(uid, goal, planner_name, entry)
    emit("result", _result_to_dict(result, planner_name, goal))
    return {
        "uid": uid, "goal": goal[:100],
        "success": result.success,
        "best_score": result.best_score,
        "rounds": result.total_rounds,
    }


def _run_attack_stream(session_id: str, params: dict):
    """在后台线程中运行攻击。单 goal / batch / compare 三种模式统一入口."""
    session = _active_sessions[session_id]
    q = session["queue"]
    event_log = session.setdefault("event_log", [])
    _event_counter = [0]

    def emit(event: str, data: dict):
        _event_counter[0] += 1
        event_id = str(_event_counter[0])
        msg = json.dumps({"event": event, "data": data, "id": event_id})
        q.put(msg)
        event_log.append(msg)
        info = data.get("msg", "") or data.get("goal", "") or event
        print(f"  [{session_id[:6]}] [{event}] {info[:120]}", flush=True)

    try:
        # ── 判断模式 ──
        goal_text = params.get("goal", "").strip()
        is_compare = params.get("compare", False)
        if goal_text:
            mode = "single"
        elif is_compare:
            mode = "compare"
        else:
            mode = "batch"

        print(f"\n{'='*25}")
        print(f"  Attack started (session={session_id[:6]})")
        print(f"  Mode: {mode} | Planner: {params.get('planner', 'crescendo')}")
        print(f"  Goal: {goal_text or '(from dataset)'[:80]}")
        print(f"  Scale: {params.get('scale', '10')} | Compare: {is_compare}")
        print(f"{'='*25}")
        emit("status", {"msg": "Initializing...", "type": "info"})

        # ── 初始化（所有模式共用，只创建一次）──
        config = PlannerConfig(
            max_rounds=20,
            success_threshold=params.get("threshold", 0.5),
        )
        generator = Generator(
            model=params.get("attack_model", "llama2-uncensored:7b"),
            victim_model=params.get("victim_model", "llama3.1:latest"),
            attack_base_url=params.get("attack_base_url", ""),
            attack_api_key=params.get("attack_api_key", ""),
            victim_base_url=params.get("victim_base_url", ""),
            victim_api_key=params.get("victim_api_key", ""),
        )
        judge = Judge(
            model=params.get("judge_model", ""),
            base_url=params.get("judge_base_url", ""),
            api_key=params.get("judge_key", ""),
        )

        # ── 确定 goals 和 planners 列表 ──
        if mode == "single":
            goals = [{"prompt": goal_text}]
            planners = [params.get("planner", "crescendo")]
        else:
            items = _load_dataset(_parse_scale(params.get("scale")))
            goals = [{"prompt": it["prompt"]} for it in items]
            if mode == "compare":
                planners = list(PLANNERS.keys())
            else:
                planners = [params.get("planner", "crescendo")]

        # ── 执行 ──
        if mode == "compare":
            # 断点续传: 按 output 文件数跳过已完成的目标
            skipped = _count_completed()
            if skipped > 0:
                emit("status", {"msg": f"[Resume] {skipped} goals already completed, skipping...",
                                "type": "info"})
                goals = goals[skipped:]
            emit("status", {"msg": f"Compare mode: {len(goals)} goals × {len(planners)} planners",
                            "type": "info"})
            compare_results = []
            for pname in planners:
                planner_results = []
                wins = 0
                for i, item in enumerate(goals):
                    if _active_sessions.get(session_id, {}).get("stop"):
                        emit("status", {"msg": "Stopped by user.", "type": "warn"})
                        return
                    uid = item.get("uid", "")
                    emit("status", {
                        "msg": f"[{pname}] {i+1}/{len(goals)}: {item['prompt'][:60]}...",
                        "type": "progress", "planner": pname,
                        "current": i + 1, "total": len(goals),
                    })
                    try:
                        r = _run_single_attack(session_id, item["prompt"], pname,
                                               config, generator, judge, emit,
                                               uid=item.get("uid", ""))
                    except _StopAttack:
                        emit("status", {"msg": "Stopped by user.", "type": "warn"})
                        return
                    if r:
                        planner_results.append(r)
                        if r["success"]:
                            wins += 1
                compare_results.append({
                    "planner": pname,
                    "asr": wins / len(goals) * 100 if goals else 0,
                    "wins": wins, "total": len(goals),
                    "details": planner_results,
                })
            emit("compare_done", {"results": compare_results})

        elif mode == "batch":
            planner_name = planners[0]
            # 断点续传: 按 output 文件数跳过已完成的目标
            skipped = _count_completed()
            if skipped > 0:
                emit("status", {"msg": f"[Resume] {skipped} goals already completed, skipping...",
                                "type": "info"})
                goals = goals[skipped:]
            emit("status", {"msg": f"Batch mode: {len(goals)} goals, planner={planner_name}",
                            "type": "info"})
            results = []
            wins = 0
            for i, item in enumerate(goals):
                if _active_sessions.get(session_id, {}).get("stop"):
                    emit("status", {"msg": "Stopped by user.", "type": "warn"})
                    return
                uid = item.get("uid", "")
                emit("status", {
                    "msg": f"[{planner_name}] {i+1}/{len(goals)}: {item['prompt'][:60]}...",
                    "type": "progress", "planner": planner_name,
                    "current": i + 1, "total": len(goals),
                })
                try:
                    r = _run_single_attack(session_id, item["prompt"], planner_name,
                                           config, generator, judge, emit,
                                           uid=item.get("uid", ""))
                except _StopAttack:
                    emit("status", {"msg": "Stopped by user.", "type": "warn"})
                    return
                if r:
                    results.append(r)
                    if r["success"]:
                        wins += 1
            emit("batch_done", {
                "planner": planner_name,
                "asr": wins / len(goals) * 100 if goals else 0,
                "wins": wins, "total": len(goals),
                "details": results,
            })

        else:  # single
            emit("status", {"msg": f"Starting attack on: {goal_text[:80]}...",
                            "type": "info"})
            try:
                _run_single_attack(session_id, goal_text, planners[0],
                                   config, generator, judge, emit)
            except _StopAttack:
                emit("status", {"msg": "Stopped by user.", "type": "warn"})

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
    thread = threading.Thread(target=_run_attack_stream, args=(session_id, params), daemon=True)
    _active_sessions[session_id] = {"queue": q, "stop": False, "thread": thread}
    thread.start()
    return jsonify({"session_id": session_id})


@app.route("/api/stream/<session_id>")
def api_stream(session_id):
    """SSE 端点：流式推送攻击进度，支持重连增量回放."""
    if session_id not in _active_sessions:
        return Response("data: {\"event\":\"error\",\"data\":\"Session not found\"}\n\n",
                        mimetype="text/event-stream")

    session = _active_sessions[session_id]
    q = session["queue"]
    event_log = session.setdefault("event_log", [])

    # 解析 Last-Event-Id 实现增量回放
    last_id_str = request.headers.get("Last-Event-Id", "0")
    try:
        last_id = int(last_id_str)
    except ValueError:
        last_id = 0

    def generate():
        # ── 增量回放：只发送 last_id 之后的新事件 ──
        replayed = 0
        for msg in event_log:
            try:
                evt = json.loads(msg)
                eid = int(evt.get("id", 0))
                if eid > last_id:
                    yield f"id: {eid}\ndata: {msg}\n\n"
                    replayed += 1
            except Exception:
                yield f"data: {msg}\n\n"
                replayed += 1
        if replayed > 0:
            print(f"  [SSE] Replayed {replayed} new events (after id={last_id}) for {session_id[:6]}", flush=True)

        # ── 正常流式 ──
        while True:
            try:
                msg = q.get(timeout=30)
                try:
                    evt = json.loads(msg)
                    eid = evt.get("id", "")
                    yield f"id: {eid}\ndata: {msg}\n\n"
                except Exception:
                    yield f"data: {msg}\n\n"
                if '"event":"done"' in msg or '"event":"error"' in msg:
                    break
            except Exception:
                yield ": ping\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/api/stop/<session_id>", methods=["POST"])
def api_stop(session_id):
    """停止正在运行的攻击."""
    if session_id in _active_sessions:
        _active_sessions[session_id]["stop"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Session not found"})


@app.route("/api/export/<session_id>")
def api_export(session_id):
    """导出完整攻防对话（支持单 goal 和 batch 模式）."""
    if session_id not in _export_data:
        return jsonify({"ok": False, "error": "Session not found or not finished"}), 404

    entries = _export_data[session_id]

    # 格式化为可读文本
    lines = []
    lines.append("=" * 70)
    lines.append(f"  NCRT v3 — Attack Export")
    lines.append("=" * 70)
    lines.append(f"  Total Goals:  {len(entries)}")
    lines.append(f"  Success Rate: {sum(1 for e in entries if e['success'])}/{len(entries)}")
    lines.append("=" * 70)

    for ei, entry in enumerate(entries):
        lines.append("")
        lines.append(f"  >>> Goal {ei+1}: {entry.get('uid', 'N/A')} — {entry['goal'][:80]}")
        lines.append(f"  >>> Planner: {entry['planner']} | Success: {entry['success']} | Score: {entry['best_score']:.2f} | Turns: {len(entry['turns'])}")
        lines.append("")

        for t in entry["turns"]:
            lines.append(f"─── Round {t['round']} ({t.get('planner', '')}) "
                         f"[score={t.get('score', '?'):.2f}] ───")
            lines.append("")
            lines.append(f"  PROMPT:")
            lines.append(f"  {t.get('prompt', '')}")
            lines.append("")
            lines.append(f"  RESPONSE:")
            lines.append(f"  {t.get('response', '')}")
            lines.append("")
            if t.get("reason"):
                lines.append(f"  JUDGE: {t['reason']}")
                lines.append("")
            lines.append("")

    lines.append("=" * 70)
    lines.append("  End of Export")
    lines.append("=" * 70)

    text = "\n".join(lines)

    return Response(
        text,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=attack_export_{session_id}.txt"}
    )


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
