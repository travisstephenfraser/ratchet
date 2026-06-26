"""Structured prompt assembly: separate policy (constraints) / instructions (the
searched candidate) / data, so the model — and you — can tell them apart. Optional
reasoning field gives the model room to think before answering."""


def assemble(policy, instructions, data, output_contract, reasoning=False):
    parts = []
    if policy.strip():
        parts.append(f"<policy>\n{policy.strip()}\n</policy>")
    parts.append(f"<instructions>\n{instructions.strip()}\n</instructions>")
    parts.append(f"<data>\n{data.strip()}\n</data>")
    if reasoning:
        parts.append("<reasoning>\n(think step by step here before answering)\n</reasoning>")
    parts.append(output_contract.strip())
    return "\n\n".join(parts)
