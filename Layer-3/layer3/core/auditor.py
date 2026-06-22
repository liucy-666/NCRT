import json
import os
from typing import Optional, List, Dict, Any
from layer3.core.types import (
    AttackConfigSummary,
    OutcomeClassification,
    AuditReport,
    JudgmentResult,
)
from layer3.config import Layer3Config


class Auditor:
    """
    Generates structured audit reports when jailbreak succeeds or partially succeeds.
    """

    def __init__(self, config: Optional[Layer3Config] = None):
        self.config = config or Layer3Config()

    def build_attack_summary(
        self,
        original_instruction: str,
        transformed_prompt: str,
        trace_log: List[Dict[str, Any]],
        assembly_mode: str = "default",
    ) -> AttackConfigSummary:
        strategies = []
        dimensions = set()
        total_intensity = 0.0
        token_ratio = 0.0

        for t in trace_log:
            if t.get("modification_type", "") in ("skipped", "error"):
                continue
            strategies.append({
                "name": t.get("strategy_name", "unknown"),
                "type": t.get("strategy_type", ""),
                "intensity": t.get("intensity", 0.0),
                "scope": t.get("scope", ""),
                "description": t.get("description", ""),
            })
            dim = t.get("strategy_type", "")
            if dim:
                dimensions.add(dim)
            if t.get("token_change_ratio") is not None:
                token_ratio += t["token_change_ratio"]

        return AttackConfigSummary(
            original_instruction=original_instruction,
            transformed_prompt=transformed_prompt,
            strategies=strategies,
            total_strategies=len(strategies),
            dimensions_used=sorted(dimensions),
            token_change_ratio=min(1.0, token_ratio),
            assembly_mode=assembly_mode,
        )

    def build_outcome(
        self,
        judgment: JudgmentResult,
    ) -> OutcomeClassification:
        return OutcomeClassification(
            jailbreak_successful=judgment.compliance_score >= self.config.success_threshold,
            compliance_score=judgment.compliance_score,
            compliance_label=judgment.label,
            breached_categories=judgment.breached_categories,
            judge_explanation=judgment.explanation,
        )

    def generate(
        self,
        original_instruction: str,
        transformed_prompt: str,
        victim_response: str,
        trace_log: List[Dict[str, Any]],
        judgment: JudgmentResult,
        extra_metadata: Optional[Dict[str, Any]] = None,
        assembly_mode: str = "default",
    ) -> AuditReport:
        attack_summary = self.build_attack_summary(
            original_instruction, transformed_prompt, trace_log, assembly_mode
        )
        outcome = self.build_outcome(judgment)
        metadata = extra_metadata or {}
        metadata["judge_model"] = judgment.judge_model
        metadata["audit_threshold"] = self.config.audit_trigger_threshold

        return AuditReport(
            attack_config=attack_summary,
            outcome=outcome,
            victim_response=victim_response,
            trace_log=trace_log,
            metadata=metadata,
        )

    def save(self, report: AuditReport, filename: Optional[str] = None) -> str:
        os.makedirs(self.config.audit_output_dir, exist_ok=True)
        if filename is None:
            import time
            ts = int(time.time())
            filename = f"audit_report_{ts}.json"
        filepath = os.path.join(self.config.audit_output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath

    def print_summary(self, report: AuditReport) -> None:
        print("=" * 60)
        print("  JAILBREAK AUDIT REPORT")
        print("=" * 60)
        cfg = report.attack_config
        out = report.outcome
        print(f"  Original instruction: {cfg.original_instruction[:100]}")
        print(f"  Strategies applied ({cfg.total_strategies}):")
        for s in cfg.strategies:
            print(f"    [{s['type']}] {s['name']} (i={s['intensity']:.2f})")
        print(f"  Dimensions used: {cfg.dimensions_used}")
        print(f"  Token change ratio: {cfg.token_change_ratio:.2%}")
        print(f"  Assembly mode: {cfg.assembly_mode}")
        print(f"  ---")
        print(f"  Jailbreak successful: {out.jailbreak_successful}")
        print(f"  Compliance score: {out.compliance_score:.2f}")
        print(f"  Label: {out.compliance_label}")
        if out.breached_categories:
            print(f"  Breached categories: {out.breached_categories}")
        print(f"  Judge explanation: {out.judge_explanation[:200]}")
        print(f"  Victim response (first 200 chars): {report.victim_response[:200]}")
        print("=" * 60)


"""
================================================================================
FILE: layer3/core/auditor.py
ROLE: Generates structured audit reports for successful/partial jailbreak attacks.

CLASSES:
  Auditor:
    config (Layer3Config) -- Configuration with thresholds and output directory.

    build_attack_summary(original_instruction, transformed_prompt, trace_log, assembly_mode)
        -> AttackConfigSummary:
      Extracts attack configuration from Layer 1 trace_log:
      - Strategy names, types, intensities, scopes, descriptions
      - Dimensions used (symbolic/structural/semantic)
      - Token change ratio (cumulative from all strategies)
      Only includes strategies with modification_type NOT in (skipped, error).

    build_outcome(judgment: JudgmentResult) -> OutcomeClassification:
      Converts JudgeLLM judgment into outcome classification.
      jailbreak_successful = compliance_score >= success_threshold.

    generate(original_instruction, transformed_prompt, victim_response, trace_log,
             judgment, extra_metadata, assembly_mode) -> AuditReport:
      Full report generation. Combines attack config + outcome + metadata.
      Used when should_generate_audit() returns True.

    save(report, filename) -> str:
      Saves audit report as JSON to config.audit_output_dir.
      Auto-generates filename with timestamp if not provided.

    print_summary(report) -> None:
      Prints a human-readable summary of the audit report to stdout.

DESIGN NOTES:
  - The auditor is stateless — each generate() call produces a new report.
  - Reports are saved as UTF-8 JSON for downstream analysis.
  - The print_summary() method provides immediate feedback for CLI usage.
================================================================================
"""
