from typing import Dict, Any
from layer1.core.test_case import TransformedCase, AssemblyModeType


class Assembler:

    def assemble(
        self, case: TransformedCase, mode: AssemblyModeType = "default"
    ) -> TransformedCase:
        if mode == "default":
            case.full_input = self._default(case)
        elif mode == "chatml":
            case.full_input = self._chatml(case)
        elif mode == "openai":
            case.full_input = self._openai(case)
        elif mode == "raw":
            case.full_input = self._raw(case)
        else:
            case.full_input = self._default(case)
        case.assembly_mode = mode
        return case

    def _default(self, case: TransformedCase) -> str:
        parts = []
        if case.role:
            parts.append(f"[Role: {case.role}]")
        parts.append(f"Instruction: {case.instruction}")
        if case.context:
            parts.append(f"Context: {case.context}")
        return "\n\n".join(parts)

    def _chatml(self, case: TransformedCase) -> str:
        role = case.role if case.role else "user"
        lines = [f"<|im_start|>{role}"]
        if case.context:
            lines.append(f"Context: {case.context}")
        lines.append(case.instruction)
        lines.append("<|im_end|>")
        lines.append("<|im_start|>assistant")
        return "\n".join(lines)

    def _openai(self, case: TransformedCase) -> str:
        import json
        role = case.role if case.role else "user"
        content = case.instruction
        if case.context:
            content = f"Context: {case.context}\n\n{case.instruction}"
        message = {"role": role, "content": content}
        return json.dumps([message], ensure_ascii=False)

    def _raw(self, case: TransformedCase) -> str:
        return case.instruction

    @staticmethod
    def to_dict(case: TransformedCase) -> Dict[str, Any]:
        return {
            "instruction": case.instruction,
            "context": case.context,
            "role": case.role,
            "metadata": case.metadata,
            "full_input": case.full_input,
            "assembly_mode": case.assembly_mode,
        }


"""
================================================================================
FILE: layer1/assembler.py
ROLE: Assembles the final full_input string from the perturbed instruction
      and context fields, supporting multiple output formats.

CLASSES:
  Assembler:
    assemble(case, mode) -> TransformedCase:
      Sets case.full_input based on the assembly mode.

    _default(case) -> str:
      Format: "[Role: {role}]\n\nInstruction: {instruction}\n\nContext: {context}"

    _chatml(case) -> str:
      Format: "<|im_start|>{role}\n{instruction}<|im_end|>\n<|im_start|>assistant"
      With optional context prefix.

    _openai(case) -> str:
      Format: JSON array [{"role": "...", "content": "..."}]

    _raw(case) -> str:
      Format: bare instruction string only.

    to_dict(case) -> Dict:
      Static method that converts TransformedCase to a serializable dict.
      Used by downstream consumers (Layer 2, Layer 3).

DESIGN NOTES:
  - The assembler is intentionally separate from strategies — strategies
    modify instruction/context, the assembler only packages them.
  - Assembly happens at the END of the pipeline (Step 5), not during
    chain execution. Intermediate TransformedCases carry fragmented fields.
  - Adding new assembly modes requires adding a new _method() and a new
    branch in assemble().
================================================================================
"""
