"""Implement run(): apply ONE candidate to ONE item, return a prediction.
- Prepend `policy` (active constraints) to the prompt via assemble().
- FAIL LOUD: if you can't parse the model response, RAISE — never return a silent 0.
- Do arithmetic (sums, clamps) HERE; the model emits judgments only."""
from ratchet.prompt import assemble


class Runner:
    def run(self, candidate, item, policy=""):
        # prompt = assemble(policy=policy, instructions=candidate, data=item["text"],
        #                   output_contract="Return ONLY JSON {...}", reasoning=True)
        # response = call_your_model(prompt)            # the only network call
        # return parse_and_clamp(response)              # raises on malformed output
        raise NotImplementedError("implement Runner.run for your project")
