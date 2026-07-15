"""Implement run(): apply ONE candidate to ONE item, return a prediction.
- Prepend `policy` (active constraints) to the prompt via assemble().
- If one response can't be parsed, raise ratchet.adapter.Unparseable so it counts as a
  visible miss. Let transport, timeout, and harness exceptions propagate and halt.
- Do arithmetic (sums, clamps) HERE; the model emits judgments only."""
from ratchet.prompt import assemble


class Runner:
    def run(self, candidate, item, policy=""):
        # prompt = assemble(policy=policy, instructions=candidate, data=item["text"],
        #                   output_contract="Return ONLY JSON {...}", reasoning=True)
        # response = call_your_model(prompt)            # the only network call
        # return parse_and_clamp(response)              # raises Unparseable if malformed
        raise NotImplementedError("implement Runner.run for your project")
