from ratchet.prompt import assemble


def test_assemble_separates_concerns():
    p = assemble(policy="be fair", instructions="grade Q1", data="WORK", output_contract="JSON")
    assert "<policy>\nbe fair\n</policy>" in p
    assert "<instructions>\ngrade Q1\n</instructions>" in p
    assert "<data>\nWORK\n</data>" in p
    assert p.index("<policy>") < p.index("<instructions>") < p.index("<data>")
    assert "JSON" in p and "<reasoning>" not in p


def test_reasoning_optional():
    assert "<reasoning>" in assemble("", "x", "y", "z", reasoning=True)


def test_empty_policy_omits_tag():
    assert "<policy>" not in assemble("", "x", "y", "z")
