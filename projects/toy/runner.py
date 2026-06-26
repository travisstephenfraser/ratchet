"""Synthetic grader. Calls assemble() so the e2e exercises structured prompt assembly,
then computes a deterministic grade. Leniency comes from the candidate OR the active
policy (constraints) — so the run path covers both the search and the feedback channel."""
from ratchet.prompt import assemble


class Runner:
    def run(self, candidate, item, policy=""):
        # exercise prompt assembly on the real path (output discarded by the synthetic stub)
        assemble(policy=policy, instructions=candidate, data=str(item),
                 output_contract="Return the integer grade.")
        lenient = "lenient" in candidate.lower() or "lenient" in policy.lower()
        true = item["true"]
        return true if lenient else max(0, true - (2 if item["messy"] else 0))
