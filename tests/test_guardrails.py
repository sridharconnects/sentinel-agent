"""Guardrails: input cap, wall-clock timeout, output blocklist."""
from app.agents import MAX_INPUT_CHARS, stream_briefing


def test_input_length_cap():
    events = list(stream_briefing("x" * (MAX_INPUT_CHARS + 1)))
    assert events[0]["type"] == "error"
    assert events[0]["data"]["reason"] == "input_too_long"
    assert events[0]["data"]["got_chars"] == MAX_INPUT_CHARS + 1


def test_wall_clock_timeout_aborts_run():
    events = list(stream_briefing(
        "What is agentic AI?", config={"timeout_seconds": 0}))
    assert events[0]["type"] == "error"
    assert events[0]["data"]["reason"] == "timeout"
    # No briefing may complete after a timeout.
    assert not [e for e in events if e["type"] == "done"]


def test_blocklist_redacts_final_output():
    events = list(stream_briefing(
        "What is agentic AI?", config={"blocklist": ["briefing"]}))
    guard = [e for e in events if e["type"] == "guardrail"]
    assert guard and guard[0]["data"]["action"] == "blocklist_redaction"
    done = [e for e in events if e["type"] == "done"][0]
    assert "[REDACTED]" in done["data"]["final_answer"]
    assert "briefing" not in done["data"]["final_answer"].lower()


def test_clean_output_passes_blocklist_untouched():
    events = list(stream_briefing("What is agentic AI?"))
    assert not [e for e in events if e["type"] == "guardrail"]
    done = [e for e in events if e["type"] == "done"][0]
    assert "[REDACTED]" not in done["data"]["final_answer"]
