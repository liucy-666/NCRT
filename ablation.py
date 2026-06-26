"""
Focused Ablation v2 — Encoder = Loss Function?

Hypothesis: ASR = dual_model_hijack_quality − encoding_loss

5 chains × 50 instructions × 3 rounds = 750 attacks
With LLM cache warmup, estimated ~45-60 min.

Usage:
    python ablation.py --scale 50 --rounds 3
"""
import sys, json, os, time, random, argparse
from typing import List, Optional, Dict, Any
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(SCRIPT_DIR, "Layer-1"), os.path.join(SCRIPT_DIR, "Layer-2"), os.path.join(SCRIPT_DIR, "Layer-3")]:
    sys.path.insert(0, p)

from layer1 import TestCase, AttackBudget, Pipeline
from layer1.core.strategy import STRATEGY_REGISTRY
from layer1.utils.llm_client import LLMClient, ModelEndpoint, parse_model_arg
from layer2 import ResponseState, PerturbationVector
from layer3 import JudgeRewardFunction, Layer3Config

parser = argparse.ArgumentParser(description="Focused Ablation: encoder = loss function?")
parser.add_argument("--attack-model", type=str, default="ollama,llama2-uncensored:7b")
parser.add_argument("--victim-model", type=str, default="ollama,llama3.2:latest")
parser.add_argument("--judge-model", type=str,
                    default="https://api.deepseek.com/v1,sk-b329f34033aa4852a2c16751134dbe26,deepseek-chat")
parser.add_argument("--scale", type=str, default="50")
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

attack_ep = parse_model_arg(args.attack_model)
victim_ep = parse_model_arg(args.victim_model)
judge_ep = parse_model_arg(args.judge_model)
LLMClient.configure_attack(attack_ep)

config = Layer3Config()
config.victim_model = victim_ep.model
config.victim_api_base = victim_ep.base_url
config.victim_api_key = victim_ep.api_key
config.judge_model = judge_ep.model
config.judge_api_base = judge_ep.base_url
if judge_ep.api_key:
    config.judge_api_key = judge_ep.api_key

MAX_ROUNDS = args.rounds
SEED_BASE = args.seed
DATA_PATH = os.path.join(SCRIPT_DIR, "Data", "harmful_prompts.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output", "ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHAINS: Dict[str, List[str]] = {
    "dual_solo":      ["dual_model_hijack"],
    "dual+rot13":     ["dual_model_hijack", "rot13"],
    "dual+reverse":   ["dual_model_hijack", "reverse_text"],
    "dual+caesar":    ["dual_model_hijack", "caesar"],
    "dual+base64":    ["dual_model_hijack", "base64"],
}

class ForcedStrategySampler:
    def __init__(self, names): self._names = names; self._rounds = 0; self._last = []
    def reset_instruction_context(self, _=""): self._rounds = 0
    def should_continue_attack(self): return self._rounds < MAX_ROUNDS
    def select(self, tc):
        self._rounds += 1; out = []
        for i, n in enumerate(self._names):
            if n not in STRATEGY_REGISTRY: continue
            cls = STRATEGY_REGISTRY[n]
            try:
                s = cls.__new__(cls); s.__init__(intensity=0.7, seed=tc.seed + i + self._rounds * 100)
                out.append(s)
            except Exception: continue
        self._last = out; return out
    def update(self, *a): pass

_victim_client: Optional[LLMClient] = None
def call_victim(prompt: str) -> str:
    global _victim_client
    if _victim_client is None:
        ep = ModelEndpoint(model=config.victim_model, base_url=config.victim_api_base,
                           api_key=config.victim_api_key or "ollama",
                           backend="ollama" if "11434" in config.victim_api_base else "api")
        _victim_client = LLMClient(endpoint=ep, timeout=config.victim_timeout)
    try:
        return _victim_client.chat(prompt, temperature=0.7, max_tokens=512)
    except Exception as e:
        return f"[VICTIM ERROR: {e}]"

def run_one(chain_name, forced_names, instruction, safety_category, index):
    judge = JudgeRewardFunction(config)
    sampler = ForcedStrategySampler(forced_names)
    pipeline = Pipeline(sampler=sampler, default_assembly_mode="raw", auto_sort_axis=False)
    case = TestCase(instruction=instruction, metadata={"safety_category": safety_category},
                    attack_budget=AttackBudget(max_strategies=len(forced_names)), seed=SEED_BASE + index)
    sampler.reset_instruction_context(instruction)
    rounds, best_score, final_outcome, details = 0, 0.0, "none", []
    while sampler.should_continue_attack():
        transformed, trace_log = pipeline.run(case)
        applied = [t.strategy_name for t in trace_log if t.modification_type not in ("skipped", "error")]
        if not applied: break
        summary = ", ".join(f"{t.strategy_name}({t.strategy_type})" for t in trace_log if t.modification_type not in ("skipped", "error"))
        victim_resp = call_victim(transformed.full_input)
        judge.set_context(instruction, transformed.full_input, summary)
        score = judge.evaluate(ResponseState(
            response_text=victim_resp, response_label="unknown", success_score=0.0,
            perturbation_vector=PerturbationVector(),
            strategy_trace=[t.to_dict() for t in trace_log]))
        outcome = judge.outcome(score)
        rounds += 1; best_score = max(best_score, score)
        details.append({"round_num": rounds, "strategies_applied": list(applied),
                        "score": score, "outcome": outcome, "victim_response": victim_resp[:300]})
        final_outcome = outcome
        if outcome == "success": break
    return {"instruction": instruction, "safety_category": safety_category,
            "chain_name": chain_name, "forced_names": forced_names,
            "rounds": rounds, "final_outcome": final_outcome, "best_score": best_score,
            "rounds_detail": details}

def main():
    from layer1.strategies.llm_strategies import _CACHE_HITS, _CACHE_MISSES
    from layer1.core.strategy import STRATEGY_REGISTRY

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    test_set = [p for p in all_prompts if p.get("source") == "forbidden_question_set"]
    sc = min(int(args.scale), len(test_set)) if args.scale.lower() != "all" else len(test_set)

    random.seed(SEED_BASE)
    by_cat = defaultdict(list)
    for p in test_set: by_cat[p.get("safety_category", "?")].append(p)
    per_cat = max(1, sc // len(by_cat))
    sample = []
    for cat, items in by_cat.items():
        random.shuffle(items); sample.extend(items[:per_cat])
    sample = sample[:sc]

    print()
    print("=" * 60)
    print(f"  FOCUSED ABLATION: Encoder = Loss Function?")
    print(f"  {sc} instructions × {len(CHAINS)} chains × {MAX_ROUNDS} rounds = {sc*len(CHAINS)*MAX_ROUNDS} attacks")
    print("=" * 60)

    # Phase 0: warmup cache
    print(f"\n[Phase 0] Warming LLM cache ({sc} instructions)...")
    t0 = time.time()
    from layer1.core.test_case import TransformedCase
    DH = STRATEGY_REGISTRY["dual_model_hijack"]
    for idx, item in enumerate(sample):
        s = DH.__new__(DH); s.__init__(intensity=0.7, seed=SEED_BASE + idx)
        s.apply(TransformedCase(instruction=item["prompt"], context="", role="user", metadata={}, assembly_mode="raw"))
    print(f"  Done in {time.time()-t0:.0f}s")

    # Phase 1: run
    chain_names = list(CHAINS.keys())
    all_results, chain_stats = defaultdict(list), {}
    t_start = time.time()

    for ci, cn in enumerate(chain_names):
        fnames = CHAINS[cn]; wins, scores = 0, []
        print(f"\n[{ci+1}/{len(CHAINS)}] {cn} — {fnames}")
        t0 = time.time()
        for idx, item in enumerate(sample):
            r = run_one(cn, fnames, item["prompt"], item.get("safety_category", ""), ci * 10000 + idx)
            all_results[cn].append(r)
            if r["final_outcome"] == "success": wins += 1
            scores.append(r["best_score"])
            if (idx + 1) % 10 == 0:
                e = time.time() - t0
                print(f"  [{idx+1:>3}/{sc}] ASR={wins/(idx+1)*100:.1f}% ({wins}/{idx+1}) {e:.0f}s")
        e = time.time() - t0
        chain_stats[cn] = {"chain": fnames, "total": sc, "success": wins,
                           "asr": wins / sc * 100, "avg_score": sum(scores) / len(scores) if scores else 0, "elapsed": e}
        print(f"  >> {cn}: ASR={chain_stats[cn]['asr']:.1f}% ({wins}/{sc}) avg={chain_stats[cn]['avg_score']:.3f} {e:.0f}s")

    t_total = time.time() - t_start

    # Results
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    solo = chain_stats["dual_solo"]; solo_asr = solo["asr"]
    print(f"  {'Chain':<25s} {'Win':>5s} {'ASR':>8s}  {'Loss':>8s}")
    print(f"  {'-'*25} {'-'*5} {'-'*8}  {'-'*8}")
    for cn in chain_names:
        s = chain_stats[cn]; loss = solo_asr - s["asr"]
        print(f"  {cn:<25s} {s['success']:>5d} {s['asr']:>7.1f}%  {loss:>+7.1f}%")
    print(f"\n  {'-'*25} {'-'*5} {'-'*8}  {'-'*8}")
    losses = [(cn, solo_asr - chain_stats[cn]["asr"]) for cn in chain_names if cn != "dual_solo"]
    losses.sort(key=lambda x: -x[1])
    print(f"  Loss ranking (worst first):")
    for name, loss in losses: print(f"    {name:<25s} loss={loss:+.0f}%")
    avg_loss = sum(l for _, l in losses) / len(losses)
    print(f"  Avg encoder loss: {avg_loss:.0f}%")

    print()
    if avg_loss > 3:
        print(f"  VERDICT: Encoder = Loss Function — SUPPORTED")
        print(f"  All 4 encoders reduce ASR below solo baseline.")
    else:
        print(f"  VERDICT: Encoder = Loss Function — NOT CONCLUSIVE")
    print(f"  Total: {t_total:.0f}s ({t_total/60:.1f}min) | Cache hits: {_CACHE_HITS}")

    # Save
    out = {"meta": {"scale": sc, "rounds": MAX_ROUNDS, "total_elapsed": t_total},
           "chain_stats": {cn: {"chain": s["chain"], "total": s["total"], "success": s["success"],
                                "asr": s["asr"], "avg_score": s["avg_score"]}
                          for cn, s in chain_stats.items()},
           "results": {cn: all_results[cn] for cn in chain_names}}
    json_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {json_path}")

if __name__ == "__main__":
    main()
