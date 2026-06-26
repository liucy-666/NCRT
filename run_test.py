r"""
Jailbreak Benchmark Runner — 带进度条的完整 benchmark 脚本

用法:
    .\.jailbreak\Scripts\Activate.ps1
    $env:PYTHONPATH = "D:\Python\Jailbreak\Layer-1;D:\Python\Jailbreak\Layer-2;D:\Python\Jailbreak\Layer-3"

    python run_test.py                        # 默认配置运行
    python run_test.py --scale 50             # 只跑 50 条
    python run_test.py --rounds 5             # 每条指令最多 5 轮
    python run_test.py --strategies 2         # 每轮 2 个策略
"""
import sys
import json
import os
import time
import random
import argparse
from typing import List, Optional, Dict, Any
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

# ── Path resolution ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [
    os.path.join(SCRIPT_DIR, "Layer-1"),
    os.path.join(SCRIPT_DIR, "Layer-2"),
    os.path.join(SCRIPT_DIR, "Layer-3"),
]:
    sys.path.insert(0, p)

# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Jailbreak Benchmark Runner")

parser.add_argument("--attack-model", type=str, default="ollama,llama2-uncensored:7b",
                    help="攻击模型 (默认: ollama,llama2-uncensored:7b)")
parser.add_argument("--victim-model", type=str, default="ollama,llama3.2:latest",
                    help="受害者模型 (默认: ollama,llama3.2:latest)")
parser.add_argument("--judge-model", type=str,
                    default="https://api.deepseek.com/v1,sk-b329f34033aa4852a2c16751134dbe26,deepseek-chat",
                    help="Judge 模型 (默认: DeepSeek API)")
parser.add_argument("--scale", type=str, default="all",
                    help="测试数量: 数字 或 'all' (默认: all)")
parser.add_argument("--rounds", type=int, default=10,
                    help="每条指令最大攻击轮数 (默认: 10)")
parser.add_argument("--strategies", type=int, default=3,
                    help="每轮最大策略数 (默认: 3)")
parser.add_argument("--strong", action="store_true",
                    help="强制使用高强度的 LLM 策略组合")
parser.add_argument("--noise", action="store_true",
                    help="启用噪声注入")
parser.add_argument("--rewrite", action="store_true",
                    help="启用无害改写")
parser.add_argument("--dynamic", action="store_true",
                    help="启用动态扰动")
parser.add_argument("--beam", type=int, default=0,
                    help="Beam Search 宽度 (0=关闭, 3-10=推荐, 默认: 0)")

args = parser.parse_args()

# ═══════════════════════════════════════════════════════════
# Imports (after path setup)
# ═══════════════════════════════════════════════════════════

from layer1 import TestCase, AttackBudget, Pipeline
from layer1.adders import NoiseInjector, HarmlessRewriter
from layer1.core.payload import AttackPayload
from layer1.core.attack_state import AttackState, RoundRecord
from layer1.core.strategy import STRATEGY_REGISTRY
from layer1.strategy_weights import apply_experiment_weights
from layer1.utils.llm_client import LLMClient, ModelEndpoint, parse_model_arg
from layer2 import PolicySampler, BeamPolicySampler, ResponseState, PerturbationVector
from layer3 import JudgeRewardFunction, Auditor, Layer3Config, ReportGenerator, ReportData

# ── 应用实验数据驱动的策略权重 ──
_weight_report = apply_experiment_weights()
print(f"[权重] active={_weight_report['active']}, deprecated={_weight_report['deprecated']}")
print(f"[权重] deprecated: {_weight_report['deprecated_list']}")

# ═══════════════════════════════════════════════════════════
# Model setup
# ═══════════════════════════════════════════════════════════

attack_ep = parse_model_arg(args.attack_model)
victim_ep = parse_model_arg(args.victim_model)
judge_ep   = parse_model_arg(args.judge_model)

LLMClient.configure_attack(attack_ep)

config = Layer3Config()
config.victim_model    = victim_ep.model
config.victim_api_base = victim_ep.base_url
config.victim_api_key  = victim_ep.api_key
config.judge_model     = judge_ep.model
config.judge_api_base  = judge_ep.base_url
if judge_ep.api_key:
    config.judge_api_key = judge_ep.api_key

MAX_ROUNDS_PER_INSTRUCTION = args.rounds
MAX_STRATEGIES             = args.strategies
SEED_BASE                  = 42
DATA_PATH   = os.path.join(SCRIPT_DIR, "Data", "harmful_prompts.json")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "Output")
STRONG_STRATEGY_NAMES = ["ica_enhanced", "renellm_enhanced", "deep_inception_enhanced"]
SEARCH_STRATEGY_NAMES = {"pair_enhanced", "tap_style", "gptfuzzer_style"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Victim model client
# ═══════════════════════════════════════════════════════════

_victim_client: Optional[LLMClient] = None

def call_victim(prompt: str) -> str:
    global _victim_client
    if _victim_client is None:
        ep = ModelEndpoint(
            model=config.victim_model,
            base_url=config.victim_api_base,
            api_key=config.victim_api_key or "ollama",
            backend="ollama" if "11434" in config.victim_api_base else "api",
        )
        _victim_client = LLMClient(endpoint=ep, timeout=config.victim_timeout)
    try:
        return _victim_client.chat(prompt, temperature=0.7, max_tokens=1024)
    except Exception as e:
        return f"[VICTIM ERROR: {e}]"


# ═══════════════════════════════════════════════════════════
# ForcedStrategySampler (--strong mode)
# ═══════════════════════════════════════════════════════════

class ForcedStrategySampler:
    def __init__(self, real_sampler, forced_names: List[str], budget: int):
        self._real = real_sampler
        self._forced_names = forced_names
        self._budget = budget
        self._last_selected: list = []

    def _build_forced_strategies(self, seed: int) -> list:
        strategies = []
        for name in self._forced_names:
            if len(strategies) >= self._budget:
                break
            if name in STRATEGY_REGISTRY:
                cls = STRATEGY_REGISTRY[name]
                try:
                    s = cls.__new__(cls)
                    s.__init__(intensity=0.7, seed=seed + len(strategies))
                    strategies.append(s)
                except Exception:
                    continue
        return strategies

    def select(self, test_case) -> list:
        self._real._instruction_rounds += 1
        strategies = self._build_forced_strategies(test_case.seed)
        self._last_selected = strategies
        self._real._last_selected = strategies
        return strategies

    def update(self, test_case, applied_strategies, response_state):
        self._real.update(test_case, applied_strategies, response_state)

    def reset_instruction_context(self, instruction: str = ""):
        self._real.reset_instruction_context(instruction)

    def should_continue_attack(self) -> bool:
        return self._real.should_continue_attack()

    @property
    def _last_decision_source(self):
        return "forced(strong)"


def refine_with_search_strategies(case, attack_state, applied_strategies: list) -> str:
    """Multi-turn: 用 SEARCH 类策略基于上一轮反馈优化攻击."""
    for s in applied_strategies:
        sname = getattr(s, "name", "")
        if sname not in SEARCH_STRATEGY_NAMES:
            continue
        if not hasattr(s, "refine_with_feedback"):
            continue
        prev = attack_state.round_history[-1].payload if attack_state.round_history else ""
        ctx = attack_state.context_window.to_prompt_context()
        result = s.refine_with_feedback(
            original_intent=attack_state.instruction,
            previous_payload=prev or case.instruction,
            victim_response=attack_state.round_history[-1].victim_response[:800] if attack_state.round_history else "",
            context_window=ctx,
        )
        if result and len(result) > 20:
            return result
    return ""


# ═══════════════════════════════════════════════════════════
# Per-prompt attack runner
# ═══════════════════════════════════════════════════════════

def run_attack(instruction, safety_category, index) -> Dict[str, Any]:
    judge = JudgeRewardFunction(config)
    if args.beam > 0:
        policy_sampler = BeamPolicySampler(
            reward=judge,
            beam_width=args.beam,
            expansion_factor=3,
            max_generations=3,
        )
    else:
        policy_sampler = PolicySampler(reward=judge)
    policy_sampler.config.max_rounds_per_instruction = MAX_ROUNDS_PER_INSTRUCTION
    policy_sampler.config.strategy_exhaustion_ratio = 0.8

    pipeline_kw = dict(sampler=policy_sampler, default_assembly_mode="raw",
                       auto_sort_axis=True)
    if args.noise:
        pipeline_kw["noise_injector"] = NoiseInjector(seed=SEED_BASE + index)
    if args.rewrite:
        pipeline_kw["harmless_rewriter"] = HarmlessRewriter(seed=SEED_BASE + index)
    if args.dynamic:
        from layer1.adders.dynamic_perturbation import DynamicPerturbationAdder
        pipeline_kw["dynamic_perturbator"] = DynamicPerturbationAdder(
            perturbation_rate=0.15, seed=SEED_BASE + index)

    pipeline = Pipeline(**pipeline_kw)
    auditor = Auditor(config)

    if args.strong:
        sampler = ForcedStrategySampler(policy_sampler, STRONG_STRATEGY_NAMES, MAX_STRATEGIES)
        pipeline.sampler = sampler
    else:
        sampler = policy_sampler

    case = TestCase(
        instruction=instruction,
        metadata={"task_type": "qa", "safety_category": safety_category},
        attack_budget=AttackBudget(
            max_strategies=MAX_STRATEGIES, max_modification_ratio=3.0,
            allowed_dimensions={"symbolic", "structural", "semantic"},
            allowed_axes={"search", "representation", "surface"},
        ),
        seed=SEED_BASE + index,
    )

    attack_state = AttackState(instruction=instruction, safety_category=safety_category)
    sampler.reset_instruction_context(case.instruction)
    rounds = 0
    final_outcome = "none"
    rounds_detail: List[Dict[str, Any]] = []

    while sampler.should_continue_attack() and attack_state.should_continue(
        MAX_ROUNDS_PER_INSTRUCTION, terminate_on_success=False
    ):
        transformed, trace_log = pipeline.run(case)
        applied = [t.strategy_name for t in trace_log if t.modification_type not in ("skipped", "error")]
        if not applied:
            break

        strategy_summary = ", ".join(
            f"{t.strategy_name}({t.strategy_type})" for t in trace_log
            if t.modification_type not in ("skipped", "error")
        )

        victim_response = call_victim(transformed.full_input)

        pv = PerturbationVector()
        for t in trace_log:
            if t.modification_type in ("skipped", "error"):
                continue
            if t.strategy_type == "symbolic":
                pv.symbolic_intensity += t.intensity
            elif t.strategy_type == "structural":
                pv.structural_intensity += t.intensity
            elif t.strategy_type == "semantic":
                pv.semantic_intensity += t.intensity
            if t.introduced_structure:
                pv.introduced_structures += 1
        pv.total_strategies = len(applied)
        pv.token_change_ratio = sum(
            t.token_change_ratio for t in trace_log
            if t.token_change_ratio is not None and t.modification_type not in ("skipped", "error")
        )

        judge.set_context(instruction, transformed.full_input, strategy_summary)
        score = judge.evaluate(ResponseState(
            response_text=victim_response, response_label="unknown", success_score=0.0,
            perturbation_vector=pv, strategy_trace=[t.to_dict() for t in trace_log],
        ))
        outcome = judge.outcome(score)

        rounds += 1

        attack_state.record_round(RoundRecord(
            round_num=rounds, payload=transformed.full_input,
            trace_log=[t.to_dict() for t in trace_log], victim_response=victim_response,
            judge_result=judge.last_judgment, score=score, outcome=outcome,
            strategies_applied=applied,
        ))

        rounds_detail.append({
            "round_num": rounds,
            "strategies_applied": list(applied),
            "strategy_summary": strategy_summary,
            "score": score,
            "outcome": outcome,
            "judge_label": judge.last_judgment.label if judge.last_judgment else "unknown",
            "victim_response": victim_response[:500],
        })

        applied_strategies = (
            [s for s in sampler._real._last_selected if hasattr(s, "name") and s.name in applied]
            if hasattr(sampler, "_real")
            else [s for s in sampler._last_selected if hasattr(s, "name") and s.name in applied]
        )
        sampler.update(case, applied_strategies, ResponseState(
            response_text=victim_response, response_label=judge.last_judgment.label,
            success_score=score, perturbation_vector=pv,
            strategy_trace=[t.to_dict() for t in trace_log],
        ))

        if judge.should_generate_audit(score):
            report = auditor.generate(
                original_instruction=instruction, transformed_prompt=transformed.full_input,
                victim_response=victim_response,
                trace_log=[t.to_dict() for t in trace_log],
                judgment=judge.last_judgment, assembly_mode=transformed.assembly_mode,
            )
            fname = f"audit_{index + 1:04d}_r{rounds}.json"
            auditor.save(report, fname)

        final_outcome = outcome
        if outcome == "success":
            attack_state.best_score = max(attack_state.best_score, score)
            break

        if outcome == "failure" and rounds < MAX_ROUNDS_PER_INSTRUCTION:
            last_selected = (
                sampler._real._last_selected if hasattr(sampler, "_real")
                else sampler._last_selected
            )
            refined = refine_with_search_strategies(case, attack_state, last_selected)
            if refined:
                case.instruction = refined
                case.metadata["refined_round"] = rounds
                case.metadata["original_instruction"] = instruction

    return {
        "instruction": instruction, "safety_category": safety_category,
        "rounds": rounds, "final_outcome": final_outcome,
        "judge_score": judge.last_judgment.compliance_score if judge.last_judgment else 0.0,
        "judge_label": judge.last_judgment.label if judge.last_judgment else "unknown",
        "attack_state": attack_state.to_dict(),
        "rounds_detail": rounds_detail,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    test_set = [p for p in all_prompts if p.get("source") == "forbidden_question_set"]

    if args.scale.lower() == "all":
        scale_count = len(test_set)
    else:
        try:
            scale_count = int(args.scale)
        except ValueError:
            print(f"[WARN] Invalid --scale '{args.scale}', using all ({len(test_set)})")
            scale_count = len(test_set)
        scale_count = min(scale_count, len(test_set))

    random.seed(SEED_BASE)
    test_sample = random.sample(test_set, scale_count)

    # ── 打印配置 ──
    print()
    print("=" * 72)
    print("  🔴 JAILBREAK BENCHMARK")
    print("=" * 72)
    print(f"  攻击模型 : {attack_ep.resolve().model} ({attack_ep.backend})")
    print(f"  受害者   : {victim_ep.resolve().model} ({victim_ep.backend})")
    print(f"  Judge    : {judge_ep.resolve().model} ({judge_ep.backend})")
    print(f"  数据池   : {len(test_set)} 条 (forbidden_question_set)")
    print(f"  测试数量 : {scale_count} 条")
    print(f"  轮数上限 : {MAX_ROUNDS_PER_INSTRUCTION} 轮/指令")
    print(f"  策略上限 : {MAX_STRATEGIES} 个/轮")
    print(f"  Strong   : {'ON' if args.strong else 'OFF'}")
    print(f"  Beam     : {args.beam} (width={args.beam})" if args.beam > 0 else "  Beam     : OFF (round-robin)")
    print(f"  Noise    : {'ON' if args.noise else 'OFF'}")
    print(f"  Rewrite  : {'ON' if args.rewrite else 'OFF'}")
    print(f"  Dynamic  : {'ON' if args.dynamic else 'OFF'}")
    print("=" * 72)
    print()

    results = []
    t_start = time.time()

    # ── 统计追踪 ──
    cumulative_success = 0
    curve_labels: List[str] = []
    curve_data: List[float] = []
    combo_stats: Dict[frozenset, Dict[str, Any]] = defaultdict(
        lambda: {"success": 0, "partial": 0, "failure": 0, "scores": []}
    )
    strategy_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "success": 0, "partial": 0, "failure": 0, "scores": []}
    )
    prompt_details: List[Dict[str, Any]] = []
    total_judge_calls = 0
    total_audits = 0

    # ── 主循环：外层进度条 ──
    pbar_outer = tqdm(
        enumerate(test_sample),
        total=scale_count,
        desc="📋 Prompts",
        unit="p",
        ncols=100,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for i, item in pbar_outer:
        instruction = item["prompt"]
        category = item.get("safety_category", "")
        pbar_outer.set_postfix_str(f"✅{cumulative_success} | {instruction[:30]}...")

        result = run_attack(instruction, category, i)
        results.append(result)

        # ── 更新学习曲线 ──
        if result["final_outcome"] == "success":
            cumulative_success += 1
        curve_labels.append(str(i + 1))
        curve_data.append(round(cumulative_success / (i + 1) * 100, 1))

        # ── 更新组合 & 策略统计 ──
        for rd in result["attack_state"].get("round_history", []):
            combo_key = frozenset(sorted(rd.get("strategies_applied", [])))
            if combo_key:
                outcome = rd.get("outcome", "failure")
                combo_stats[combo_key][outcome] += 1
                combo_stats[combo_key]["scores"].append(rd.get("score", 0.0))
            for sname in rd.get("strategies_applied", []):
                strategy_stats[sname]["total"] += 1
                strategy_stats[sname][rd.get("outcome", "failure")] += 1
                strategy_stats[sname]["scores"].append(rd.get("score", 0.0))

        # ── 记录详情 ──
        prompt_details.append({
            "index": i + 1,
            "instruction": result["instruction"],
            "final_outcome": result["final_outcome"],
            "rounds": result["rounds"],
            "judge_score": result["judge_score"],
            "judge_label": result["judge_label"],
            "rounds_detail": result.get("rounds_detail", []),
        })

        # ── 实时更新进度条后缀 ──
        cur_rate = cumulative_success / (i + 1) * 100
        pbar_outer.set_postfix_str(
            f"✅{cumulative_success} 💀{i+1-cumulative_success} 📈{cur_rate:.1f}%"
        )

    pbar_outer.close()
    t_elapsed = time.time() - t_start

    # ═══════════════════════════════════════════════════════
    # Console Summary
    # ═══════════════════════════════════════════════════════

    successes = sum(1 for r in results if r["final_outcome"] == "success")
    partials  = sum(1 for r in results if r["final_outcome"] == "partial")
    failures  = sum(1 for r in results if r["final_outcome"] == "failure")

    print()
    print("=" * 72)
    print("  📊 RESULTS SUMMARY")
    print("=" * 72)
    for r in results:
        icon = "✅" if r["final_outcome"] == "success" else "⚠️" if r["final_outcome"] == "partial" else "❌"
        print(f"  {icon} {r['final_outcome']:<10} score={r['judge_score']:.2f}  "
              f"rounds={r['rounds']}  {r['instruction'][:50]}")

    print(f"\n  ✅ Success : {successes} ({successes/scale_count*100:.1f}%)")
    print(f"  ⚠️  Partial : {partials} ({partials/scale_count*100:.1f}%)")
    print(f"  ❌ Failure : {failures} ({failures/scale_count*100:.1f}%)")
    print(f"  🕐 Time    : {t_elapsed:.1f}s ({t_elapsed/scale_count:.1f}s/prompt)")
    print(f"  🔢 Judge calls: {total_judge_calls}")

    # ═══════════════════════════════════════════════════════
    # Strategy Combination Analysis
    # ═══════════════════════════════════════════════════════

    print()
    print("=" * 72)
    print("  🔗 STRATEGY COMBINATION ANALYSIS (Top 20)")
    print("=" * 72)

    combo_list: List[Dict[str, Any]] = []
    best_combo = None
    best_combo_rate = 0.0

    for combo_key, stats in combo_stats.items():
        t = stats["success"] + stats["partial"] + stats["failure"]
        success_rate = stats["success"] / t if t > 0 else 0.0
        avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
        combo_name = " + ".join(sorted(combo_key))
        combo_list.append({
            "combination": sorted(combo_key),
            "combo_name": combo_name,
            "total": t,
            "success": stats["success"],
            "partial": stats["partial"],
            "failure": stats["failure"],
            "success_rate": success_rate,
            "avg_score": avg_score,
        })
        if success_rate > best_combo_rate:
            best_combo_rate = success_rate
            best_combo = sorted(combo_key)

    combo_list.sort(key=lambda x: x["success_rate"], reverse=True)

    if combo_list:
        print(f"  {'Combination':<45} {'Uses':<6} {'S':<5} {'Rate':<8} {'Avg':<6}")
        print(f"  {'-'*45} {'-'*6} {'-'*5} {'-'*8} {'-'*6}")
        for c in combo_list[:20]:
            print(f"  {c['combo_name']:<45} {c['total']:<6} {c['success']:<5} "
                  f"{c['success_rate']:<8.1%} {c['avg_score']:<6.2f}")

    print(f"\n  🏆 Best: {' + '.join(best_combo) if best_combo else 'N/A'} "
          f"(rate={best_combo_rate:.1%})")

    # ═══════════════════════════════════════════════════════
    # Individual Strategy Analysis
    # ═══════════════════════════════════════════════════════

    print()
    print("=" * 72)
    print("  🎯 INDIVIDUAL STRATEGY EFFECTIVENESS")
    print("=" * 72)

    strategy_list: List[Dict[str, Any]] = []
    for name, stats in strategy_stats.items():
        t = stats["total"]
        stats["success_rate"] = stats["success"] / t if t > 0 else 0.0
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
        stats["name"] = name
        strategy_list.append(stats)

    if strategy_list:
        print(f"  {'Strategy':<32} {'Uses':<6} {'S':<5} {'P':<5} {'F':<5} {'Rate':<8} {'Avg':<6}")
        print(f"  {'-'*32} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*6}")
        for s in sorted(strategy_list, key=lambda x: x["success_rate"], reverse=True):
            print(f"  {s['name']:<32} {s['total']:<6} {s['success']:<5} {s['partial']:<5} "
                  f"{s['failure']:<5} {s['success_rate']:<8.1%} {s['avg_score']:<6.2f}")

    # ═══════════════════════════════════════════════════════
    # Verdict
    # ═══════════════════════════════════════════════════════

    print()
    print("=" * 72)
    print("  ⚖️  VERDICT")
    print("=" * 72)
    if best_combo_rate == 0.0:
        print("  ❌ 防御模型太强 / 攻击模型太弱 — 未找到有效策略组合。")
        print("  建议: 使用更强的攻击模型、--strong 模式、或增加 --rounds/--strategies。")
    else:
        print(f"  🏆 最优化策略组合: {' + '.join(best_combo)} ({best_combo_rate:.1%})")

    # ═══════════════════════════════════════════════════════
    # HTML Report
    # ═══════════════════════════════════════════════════════

    report_data = ReportData(
        meta={
            "attack_model": attack_ep.resolve().model,
            "attack_backend": attack_ep.backend,
            "victim_model": victim_ep.resolve().model,
            "victim_backend": victim_ep.backend,
            "judge_model": judge_ep.resolve().model,
            "judge_backend": judge_ep.backend,
            "scale": scale_count,
            "total_pool": len(test_set),
            "max_rounds": MAX_ROUNDS_PER_INSTRUCTION,
            "max_strategies": MAX_STRATEGIES,
            "strong_mode": args.strong,
            "noise_mode": args.noise,
            "rewrite_mode": args.rewrite,
            "elapsed_seconds": round(t_elapsed, 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        summary={
            "total_prompts": scale_count,
            "success": successes,
            "partial": partials,
            "failure": failures,
            "success_rate": successes / scale_count if scale_count > 0 else 0.0,
            "judge_calls": total_judge_calls,
            "audit_reports": total_audits,
        },
        curve_labels=curve_labels,
        curve_data=curve_data,
        combo_stats=combo_list,
        strategy_stats=sorted(strategy_list, key=lambda x: x["success_rate"], reverse=True),
        prompt_details=prompt_details,
        best_combo=best_combo,
        best_combo_rate=best_combo_rate,
        verdict=(
            "Defense model too strong — no optimal strategy found"
            if best_combo_rate == 0.0
            else f"Optimal strategy combination: {' + '.join(best_combo)} ({best_combo_rate:.1%})"
        ),
    )

    report_gen = ReportGenerator()
    html_path = os.path.join(OUTPUT_DIR, "report.html")
    report_gen.generate(report_data, html_path)
    print(f"\n  📄 HTML Report: {html_path}")

    # ═══════════════════════════════════════════════════════
    # JSON Summary
    # ═══════════════════════════════════════════════════════

    summary = {
        "meta": report_data.meta,
        "summary": report_data.summary,
        "learning_curve": {"labels": curve_labels, "data": curve_data},
        "best_combination": {"combination": best_combo, "success_rate": best_combo_rate},
        "combination_breakdown": {
            c["combo_name"]: {
                "total": c["total"], "success": c["success"],
                "partial": c["partial"], "failure": c["failure"],
                "success_rate": c["success_rate"], "avg_score": c["avg_score"],
            }
            for c in combo_list
        },
        "strategy_breakdown": {
            s["name"]: {
                "total": s["total"], "success": s["success"],
                "partial": s["partial"], "failure": s["failure"],
                "success_rate": s["success_rate"], "avg_score": s["avg_score"],
            }
            for s in strategy_list
        },
        "verdict": report_data.verdict,
        "results": results,
    }

    json_path = os.path.join(OUTPUT_DIR, "full_run_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  📊 JSON Summary: {json_path}")

    print()
    print("=" * 72)
    print("  ✅ Benchmark 完成!")
    print("=" * 72)


if __name__ == "__main__":
    main()
