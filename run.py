r"""
Jailbreak Platform v2 — Full System Runner
Layers 1 + 2 + 3 + Victim Model + HTML Report Generator

Usage:
    . D:\Python\Jailbreak\.jailbreak\Scripts\Activate.ps1
    $env:PYTHONPATH = "D:\Python\Jailbreak\Layer-1;D:\Python\Jailbreak\Layer-2;D:\Python\Jailbreak\Layer-3"

    python run.py --scale 1
    python run.py --scale 5
    python run.py --attack-model ollama,llama2-uncensored:7b --scale 10
    python run.py --judge-model https://api.deepseek.com/v1,sk-xxx,deepseek-chat
    python run.py --strong --scale all

Model format:
  Ollama:  ollama,<modelname>
  API:     <url>,<apikey>,<modelname>
"""
import sys
import json
import os
import time
import random
import argparse
from typing import List, Optional, Dict, Any
from collections import defaultdict

# ── Path resolution (relative to this script) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [
    os.path.join(SCRIPT_DIR, "Layer-1"),
    os.path.join(SCRIPT_DIR, "Layer-2"),
    os.path.join(SCRIPT_DIR, "Layer-3"),
]:
    sys.path.insert(0, p)

parser = argparse.ArgumentParser(
    description="Jailbreak Platform Runner v2",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Model format:
  Ollama:  ollama,<modelname>
  API:     <url>,<apikey>,<modelname>
""",
)

parser.add_argument("--attack-model", type=str, default="ollama,llama2-uncensored:7b",
                    help="Attack model (default: ollama,llama2-uncensored:7b)")
parser.add_argument("--victim-model", type=str, default="ollama,llama3.2:latest",
                    help="Victim model (default: ollama,llama3.2:latest)")
parser.add_argument("--judge-model", type=str, default="ollama,llama3.2:latest",
                    help="Judge model (default: ollama,llama3.2:latest)")
parser.add_argument("--scale", type=str, default="5",
                    help="Sample count without replacement, number or 'all' (default: 5)")
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--strategies", type=int, default=3)
parser.add_argument("--strong", action="store_true")
parser.add_argument("--combo", action="store_true")
parser.add_argument("--noise", action="store_true")
parser.add_argument("--rewrite", action="store_true")
parser.add_argument("--renellm", action="store_true")
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--no-auto-sort", action="store_true")

args = parser.parse_args()

from layer1 import TestCase, AttackBudget, Pipeline
from layer1.adders import NoiseInjector, HarmlessRewriter, ReNeLLMComposer
from layer1.core.payload import AttackPayload, AxisOrderViolationException
from layer1.core.attack_state import AttackState, RoundRecord
from layer1.core.strategy import STRATEGY_REGISTRY
from layer1.strategy_weights import apply_experiment_weights
from layer1.utils.llm_client import LLMClient, ModelEndpoint, parse_model_arg
from layer2 import PolicySampler, ResponseState, PerturbationVector
from layer3 import JudgeRewardFunction, Auditor, Layer3Config, ReportGenerator, ReportData

# ── 应用实验数据驱动的策略权重 ──
_weight_report = apply_experiment_weights()
print(f"[权重] active={_weight_report['active']}, deprecated={_weight_report['deprecated']}")
print(f"[权重] deprecated: {_weight_report['deprecated_list']}")

attack_ep = parse_model_arg(args.attack_model)
victim_ep = parse_model_arg(args.victim_model)
judge_ep   = parse_model_arg(args.judge_model)

LLMClient.configure_attack(attack_ep)

config = Layer3Config()
config.victim_model    = victim_ep.model
config.victim_api_base = victim_ep.base_url
config.victim_api_key  = victim_ep.api_key
config.judge_model    = judge_ep.model
config.judge_api_base = judge_ep.base_url
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


def _safe_print(text: str) -> None:
    try:
        print(text[:300])
    except UnicodeEncodeError:
        print(text[:300].encode("ascii", "replace").decode("ascii"))


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
    global _judge_calls, _audit_count

    judge = JudgeRewardFunction(config)
    policy_sampler = PolicySampler(reward=judge)
    policy_sampler.config.max_rounds_per_instruction = MAX_ROUNDS_PER_INSTRUCTION
    policy_sampler.config.strategy_exhaustion_ratio = 0.8

    pipeline_kw = dict(sampler=policy_sampler, default_assembly_mode="raw",
                       auto_sort_axis=not args.no_auto_sort)
    if args.noise:
        pipeline_kw["noise_injector"] = NoiseInjector(seed=SEED_BASE + index)
    if args.rewrite:
        pipeline_kw["harmless_rewriter"] = HarmlessRewriter(seed=SEED_BASE + index)
    if args.renellm:
        pipeline_kw["renellm_composer"] = ReNeLLMComposer(seed=SEED_BASE + index)
    if args.dynamic:
        from layer1.adders.dynamic_perturbation import DynamicPerturbationAdder
        pipeline_kw["dynamic_perturbator"] = DynamicPerturbationAdder(perturbation_rate=0.15, seed=SEED_BASE + index)

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
        if args.strong:
            strategy_summary += " [STRONG]"

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
        _judge_calls += 1
        attack_state.total_judge_calls = _judge_calls
        outcome = judge.outcome(score)

        rounds += 1
        print(f"  R{rounds}: [{judge.last_judgment.label}] {score:.2f} | {strategy_summary}")
        _safe_print(f"    perturbed: {transformed.full_input}")
        print(f"    victim: {victim_response[:120]}...")

        attack_state.record_round(RoundRecord(
            round_num=rounds, payload=transformed.full_input,
            trace_log=[t.to_dict() for t in trace_log], victim_response=victim_response,
            judge_result=judge.last_judgment, score=score, outcome=outcome,
            strategies_applied=applied,
        ))

        # ── Record round detail (for HTML report) ──
        rounds_detail.append({
            "round_num": rounds,
            "strategies_applied": list(applied),
            "score": score,
            "outcome": outcome,
            "victim_response": victim_response,
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
            _audit_count += 1
            report = auditor.generate(
                original_instruction=instruction, transformed_prompt=transformed.full_input,
                victim_response=victim_response,
                trace_log=[t.to_dict() for t in trace_log],
                judgment=judge.last_judgment, assembly_mode=transformed.assembly_mode,
            )
            fname = f"audit_{index + 1:02d}_r{rounds}.json"
            auditor.save(report, fname)
            print(f"    *** AUDIT SAVED: {fname} ***")

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
                print("    [MULTI-TURN] refined instruction via SEARCH feedback")

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

_judge_calls = 0
_audit_count = 0

with open(DATA_PATH, "r", encoding="utf-8") as f:
    all_prompts = json.load(f)

test_set = [p for p in all_prompts if p.get("source") == "forbidden_question_set"]

if args.scale.lower() == "all":
    scale_count = len(test_set)
else:
    try:
        scale_count = int(args.scale)
    except ValueError:
        print(f"[WARN] Invalid --scale '{args.scale}', using 5")
        scale_count = 5
    scale_count = min(scale_count, len(test_set))

random.seed(SEED_BASE)
test_sample = random.sample(test_set, scale_count)

print("=" * 70)
print("  JAILBREAK PLATFORM v2")
print(f"  Attack: {attack_ep.resolve().model} ({attack_ep.backend})")
print(f"  Victim: {victim_ep.resolve().model} ({victim_ep.backend})")
print(f"  Judge:  {judge_ep.resolve().model} ({judge_ep.backend})")
print(f"  Pool: {len(test_set)} total, sampled {scale_count} (random without replacement)")
print(f"  Rounds/instruction: {MAX_ROUNDS_PER_INSTRUCTION}  Strategies/round: {MAX_STRATEGIES}")
print(f"  Strong: {'ON' if args.strong else 'OFF'}  Noise: {'ON' if args.noise else 'OFF'}  "
      f"Rewrite: {'ON' if args.rewrite else 'OFF'}")
print("=" * 70)

results = []
t_start = time.time()

# ── Learning curve tracking ──
cumulative_success = 0
curve_labels: List[str] = []
curve_data: List[float] = []

# ── Strategy combination tracking ──
# Key: frozenset of strategy names → {success, partial, failure, scores}
combo_stats: Dict[frozenset, Dict[str, Any]] = defaultdict(
    lambda: {"success": 0, "partial": 0, "failure": 0, "scores": []}
)
# ── Individual strategy tracking ──
strategy_stats: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {"total": 0, "success": 0, "partial": 0, "failure": 0, "scores": []}
)
# ── Prompt details (for HTML report) ──
prompt_details: List[Dict[str, Any]] = []

for i, item in enumerate(test_sample):
    print(f"\n[{i+1}/{scale_count}] {item['prompt'][:80]}")
    print(f"  Category: {item['safety_category']}")
    result = run_attack(item["prompt"], item["safety_category"], i)
    results.append(result)

    # ── Update learning curve ──
    if result["final_outcome"] == "success":
        cumulative_success += 1
    curve_labels.append(str(i + 1))
    curve_data.append(round(cumulative_success / (i + 1) * 100, 1))

    # ── Update combo & strategy stats ──
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

    # ── Collect prompt detail ──
    prompt_details.append({
        "index": i + 1,
        "instruction": result["instruction"],
        "final_outcome": result["final_outcome"],
        "rounds": result["rounds"],
        "judge_score": result["judge_score"],
        "judge_label": result["judge_label"],
        "rounds_detail": result.get("rounds_detail", []),
    })

t_elapsed = time.time() - t_start

# ── Console summary ──
print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
for i, r in enumerate(results):
    print(f"  {i+1:<3} {r['final_outcome']:<10} {r['judge_score']:<7.2f} {r['rounds']:<7} {r['instruction'][:50]}")

successes = sum(1 for r in results if r["final_outcome"] == "success")
partials  = sum(1 for r in results if r["final_outcome"] == "partial")
failures  = sum(1 for r in results if r["final_outcome"] == "failure")

print(f"\n  Success: {successes}  Partial: {partials}  Failure: {failures}")
print(f"  Judge calls: {_judge_calls}  Audits: {_audit_count}  Time: {t_elapsed:.1f}s")

# ── Console: Strategy Combination Analysis ──
print("\n" + "=" * 70)
print("  STRATEGY COMBINATION ANALYSIS")
print("=" * 70)

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
    print(f"  {'Combination':<45} {'Uses':<6} {'Success':<8} {'Rate':<8} {'AvgScore':<8}")
    print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for c in combo_list[:20]:
        print(f"  {c['combo_name']:<45} {c['total']:<6} {c['success']:<8} "
              f"{c['success_rate']:<8.1%} {c['avg_score']:<8.2f}")

print(f"\n  Best combination: {' + '.join(best_combo) if best_combo else 'N/A'} "
      f"(rate: {best_combo_rate:.1%})")

# ── Console: Individual Strategy Analysis ──
print("\n" + "=" * 70)
print("  INDIVIDUAL STRATEGY EFFECTIVENESS")
print("=" * 70)

strategy_list: List[Dict[str, Any]] = []
for name, stats in strategy_stats.items():
    t = stats["total"]
    stats["success_rate"] = stats["success"] / t if t > 0 else 0.0
    stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
    stats["name"] = name
    strategy_list.append(stats)

if strategy_list:
    print(f"  {'Strategy':<30} {'Uses':<6} {'Success':<8} {'Rate':<8} {'AvgScore':<8}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for s in sorted(strategy_list, key=lambda x: x["success_rate"], reverse=True):
        print(f"  {s['name']:<30} {s['total']:<6} {s['success']:<8} "
              f"{s['success_rate']:<8.1%} {s['avg_score']:<8.2f}")

# ── Verdict ──
print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)
if best_combo_rate == 0.0:
    print("  Defense model too strong / attack model too weak — no optimal strategy found.")
    print("  Suggestions: use stronger attack model, --strong mode, or increase --rounds/--strategies.")
else:
    print(f"  Optimal strategy combination: {' + '.join(best_combo)} ({best_combo_rate:.1%})")

# ═══════════════════════════════════════════════════════════
# HTML Report Generation
# ═══════════════════════════════════════════════════════════

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
        "renellm_mode": args.renellm,
        "elapsed_seconds": round(t_elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    summary={
        "total_prompts": scale_count,
        "success": successes,
        "partial": partials,
        "failure": failures,
        "success_rate": successes / scale_count if scale_count > 0 else 0.0,
        "judge_calls": _judge_calls,
        "audit_reports": _audit_count,
    },
    curve_labels=curve_labels,
    curve_data=curve_data,
    combo_stats=combo_list,
    strategy_stats=sorted(strategy_list, key=lambda x: x["success_rate"], reverse=True),
    prompt_details=prompt_details,
    best_combo=best_combo,
    best_combo_rate=best_combo_rate,
    verdict=(
        "Defense model too strong, attack model too weak — no optimal strategy found"
        if best_combo_rate == 0.0
        else f"Optimal strategy combination: {' + '.join(best_combo)} ({best_combo_rate:.1%})"
    ),
)

report_gen = ReportGenerator()
html_path = os.path.join(OUTPUT_DIR, "report.html")
report_gen.generate(report_data, html_path)
print(f"\n📄 HTML Report: {html_path}")

# ── JSON summary (for programmatic consumption) ──
summary = {
    "meta": report_data.meta,
    "summary": report_data.summary,
    "learning_curve": {
        "labels": curve_labels,
        "data": curve_data,
    },
    "best_combination": {
        "combination": best_combo,
        "success_rate": best_combo_rate,
    },
    "combination_breakdown": {
        c["combo_name"]: {
            "total": c["total"],
            "success": c["success"],
            "partial": c["partial"],
            "failure": c["failure"],
            "success_rate": c["success_rate"],
            "avg_score": c["avg_score"],
        }
        for c in combo_list
    },
    "strategy_breakdown": {
        s["name"]: {
            "total": s["total"],
            "success": s["success"],
            "partial": s["partial"],
            "failure": s["failure"],
            "success_rate": s["success_rate"],
            "avg_score": s["avg_score"],
        }
        for s in strategy_list
    },
    "verdict": report_data.verdict,
    "results": results,
}

json_path = os.path.join(OUTPUT_DIR, "full_run_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"📊 JSON Summary: {json_path}")
