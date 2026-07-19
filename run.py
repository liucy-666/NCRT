r"""
NCRT v5 — Reflection-Driven Red-Teaming Platform

四种独立 Planner，统一 step() + 状态机接口:
  Crescendo  — 渐进式多轮越狱
  PAIR       — 迭代对抗攻击
  TAP        — 树搜索攻击
  Safe2Harm  — 语义同构攻击

每个 Planner 独立管理 生成 → call_victim → judge → 状态推进 全流程。
Scheduler 只消费反思结果做切换决策。

用法:
    $env:PYTHONPATH = (Get-Location).Path
    python run.py --planner crescendo --goal "How to hack email?"
    python run.py --planner pair --scale 10
    python run.py --planner tap --scale 50
    python run.py --compare
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import time
import argparse
from typing import List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core import Generator, Judge, PlannerConfig
from baseline.methods import METHODS as PLANNERS


def _count_completed(output_dir: str) -> int:
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith('.json')])


parser = argparse.ArgumentParser(description="NCRT v5 — Reflection-Driven Red-Teaming")
parser.add_argument("--planner", type=str, default="crescendo",
                    choices=list(PLANNERS.keys()) + ["scheduler"],
                    help="选择 Planner 类型")
parser.add_argument("--goal", type=str, default="")
parser.add_argument("--scale", type=str, default="10")
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--beam", type=int, default=3)
parser.add_argument("--branch", type=int, default=3)
parser.add_argument("--attack-model", type=str, default="llama2-uncensored:7b")
parser.add_argument("--victim-model", type=str, default="llama3.1:latest")
parser.add_argument("--judge-model", type=str, default="")
parser.add_argument("--judge-base-url", type=str, default="")
parser.add_argument("--judge-key", type=str, default="")
parser.add_argument("--attack-base-url", type=str, default="")
parser.add_argument("--attack-api-key", type=str, default="")
parser.add_argument("--victim-base-url", type=str, default="")
parser.add_argument("--victim-api-key", type=str, default="")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--compare", action="store_true")
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--output", type=str, default="")

DATA_PATH = os.path.join(SCRIPT_DIR, "data", "harmful_prompts.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

args = None


def run_one(planner_name: str, goal: str, category: str = "",
            config_overrides: dict = None) -> Dict:
    from scheduler import StrategyManager, SchedulerConfig

    is_single = planner_name != "scheduler"
    if is_single:
        roster = [planner_name]
    else:
        roster = SchedulerConfig().planner_roster

    sc = SchedulerConfig(
        max_llm_calls=args.rounds,
        success_threshold=args.threshold,
        planner_roster=roster,
        first_planner_rounds=args.rounds if is_single else 10,
    )
    gen = Generator(model=args.attack_model, victim_model=args.victim_model,
                    attack_base_url=args.attack_base_url,
                    attack_api_key=args.attack_api_key,
                    victim_base_url=args.victim_base_url,
                    victim_api_key=args.victim_api_key)
    judge = Judge(model=args.judge_model, base_url=args.judge_base_url,
                  api_key=args.judge_key)

    scheduler = StrategyManager(config=sc, generator=gen, judge=judge)
    t0 = time.time()
    result = scheduler.attack(goal)
    elapsed = time.time() - t0

    record = {
        "planner": planner_name,
        "goal": goal[:100], "category": category,
        "success": result.success,
        "best_score": result.best_score,
        "rounds": result.total_rounds,
        "final_prompt": result.final_prompt,
        "final_response": (result.final_response or "")[:300],
        "elapsed": elapsed,
        "metadata": result.metadata,
        "Planner Abstract": result.metadata.get("handoff_abstracts", []),
    }

    output_dir = args.output if args.output else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r'[\\/:*?"<>|]', '', goal[:60]).strip()
    out_path = os.path.join(output_dir, f"manual_{safe_name}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return record


import re


def _progress_bar(current, total, wins, planner, score, rounds, elapsed, eta, bar_width=30):
    pct = current / total
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    asr = wins / current * 100 if current > 0 else 0

    def fmt_t(s):
        if s < 60: return f"{s:.0f}s"
        m, s = divmod(s, 60)
        return f"{m:.0f}m{s:.0f}s"

    status = (
        f"\r  {bar} {pct*100:5.1f}% [{current}/{total}] "
        f"| {planner.upper():<10s} "
        f"| ASR={asr:5.1f}% ({wins}/{current}) "
        f"| ⌀{score:.2f}/r{rounds}"
        f"| {fmt_t(elapsed)}"
        f"| ETA {fmt_t(eta)}"
    )
    print(status, end="", flush=True)


def _run_one_parallel(task):
    planner_name, item, idx, total = task
    return run_one(planner_name, item["prompt"], item.get("safety_category", ""))


def run_compare(goals: List[Dict], n: int) -> List[Dict]:
    all_results = []
    planner_names = list(PLANNERS.keys())
    workers = args.workers

    for planner_name in planner_names:
        print(f"\n  [{planner_name.upper()}] ({'serial' if workers <= 1 else f'{workers} threads'}, n={n})")
        results = []
        wins = 0
        lock = threading.Lock()
        t0 = time.time()

        if workers > 1:
            tasks = [(planner_name, goals[i], i, n) for i in range(n)]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_one_parallel, t): t for t in tasks}
                for i, future in enumerate(as_completed(futures)):
                    r = future.result()
                    with lock:
                        results.append(r)
                        if r["success"]: wins += 1
                        done = len(results)
                        eta = elapsed / done * (n - done) if done < n else 0
                        _progress_bar(done, n, wins, planner_name,
                                     r["best_score"], r["rounds"],
                                     time.time() - t0, eta)
        else:
            for i, item in enumerate(goals[:n]):
                r = run_one(planner_name, item["prompt"], item.get("safety_category", ""))
                results.append(r)
                if r["success"]: wins += 1
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n - i - 1) if i < n - 1 else 0
                _progress_bar(i + 1, n, wins, planner_name,
                             r["best_score"], r["rounds"], elapsed, eta)

        elapsed_total = time.time() - t0
        print()
        print(f"  >> {planner_name}: ASR={wins/n*100:.1f}% ({wins}/{n}) "
              f"avg_score={sum(r['best_score'] for r in results)/n:.2f} "
              f"time={elapsed_total:.0f}s")
        all_results.append({
            "planner": planner_name, "asr": wins / n * 100,
            "wins": wins, "total": n,
            "avg_score": sum(r["best_score"] for r in results) / n if results else 0,
            "elapsed": elapsed_total, "details": results,
        })

    print(f"\n{'='*72}")
    print(f"  COMPARISON (n={n})")
    print(f"{'='*72}")
    print(f"  {'Planner':<15s} {'ASR':>8s} {'Win':>6s} {'AvgScore':>9s} {'Time':>10s}")
    print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*9} {'-'*10}")
    for r in all_results:
        print(f"  {r['planner']:<15s} {r['asr']:>7.1f}% {r['wins']:>5d}/{r['total']:<4d} "
              f"{r['avg_score']:>8.3f} {r['elapsed']:>9.0f}s")
    return all_results


def main():
    if args.compare:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            test_set = json.load(f)
        n = min(int(args.scale), len(test_set)) if args.scale != "all" else len(test_set)
        import random
        random.seed(args.seed)
        sample = random.sample(test_set, n)

        output_dir = args.output if args.output else OUTPUT_DIR
        completed = _count_completed(output_dir)
        if completed > 0:
            print(f"\n  [Resume] {completed} goals already completed, skipping...")
            sample = sample[completed:]
            n = len(sample)
            if n == 0: return

        print(f"\n  NCRT v5 — Planner Comparison")
        print(f"  Attack: {args.attack_model}  |  Victim: {args.victim_model}")
        print(f"  Planners: {list(PLANNERS.keys())}")
        print(f"  Samples: {n}  |  Max rounds: {args.rounds}")

        results = run_compare(sample, n)
        json_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved: {json_path}")
        return

    if args.goal:
        print(f"\n  NCRT v5 — {args.planner.upper()}")
        print(f"  Attack: {args.attack_model}  |  Victim: {args.victim_model}  |  Judge: {args.judge_model}")
        print(f"  Goal: {args.goal}")
        r = run_one(args.planner, args.goal)
        print(f"\n  Success: {r['success']}")
        print(f"  Score: {r['best_score']:.2f}")
        print(f"  Rounds: {r['rounds']}")
        print(f"  Time: {r['elapsed']:.0f}s")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    n = min(int(args.scale), len(test_set)) if args.scale != "all" else len(test_set)
    import random
    random.seed(args.seed)
    sample = random.sample(test_set, n)

    output_dir = args.output if args.output else OUTPUT_DIR
    completed = _count_completed(output_dir)
    if completed > 0:
        print(f"\n  [Resume] {completed} goals already completed, skipping...")
        sample = sample[completed:]
        n = len(sample)
        if n == 0: return

    print(f"\n  NCRT v5 — {args.planner.upper()}")
    print(f"  Attack: {args.attack_model}  |  Victim: {args.victim_model}")
    print(f"  Samples: {n}  |  Max rounds: {args.rounds}")

    results = []
    wins = 0
    lock = threading.Lock()
    t0 = time.time()

    if args.workers > 1:
        tasks = [(args.planner, sample[i], i, n) for i in range(n)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one_parallel, t): t for t in tasks}
            for future in as_completed(futures):
                r = future.result()
                with lock:
                    results.append(r)
                    if r["success"]: wins += 1
                    done = len(results)
                    eta = elapsed / done * (n - done) if done < n else 0
                    _progress_bar(done, n, wins, args.planner,
                                 r["best_score"], r["rounds"],
                                 time.time() - t0, eta)
    else:
        for i, item in enumerate(sample):
            r = run_one(args.planner, item["prompt"], item.get("safety_category", ""))
            results.append(r)
            if r["success"]: wins += 1
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1) if i < n - 1 else 0
            _progress_bar(i + 1, n, wins, args.planner,
                         r["best_score"], r["rounds"], elapsed, eta)

    print()
    elapsed_total = time.time() - t0
    print(f"\n  {args.planner}: ASR={wins/n*100:.1f}% ({wins}/{n}) "
          f"time={elapsed_total:.0f}s")
    json_path = os.path.join(OUTPUT_DIR, f"{args.planner}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main()
