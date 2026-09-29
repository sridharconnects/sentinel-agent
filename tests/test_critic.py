"""Critic revision loop: termination and pass-through."""
from app.agents import stream_briefing
from app.llm import MockLLM, _split_tag


class FailCriticLLM(MockLLM):
    def generate(self, prompt: str) -> str:
        tag, _ = _split_tag(prompt)
        if tag == "CRITIC":
            return "VERDICT: FAIL\nREASON: stubbed failure for loop-termination test"
        return super().generate(prompt)


class PassCriticLLM(MockLLM):
    def generate(self, prompt: str) -> str:
        tag, _ = _split_tag(prompt)
        if tag == "CRITIC":
            return "VERDICT: PASS"
        return super().generate(prompt)


def _analyst_starts(events):
    return [e for e in events
            if e["type"] == "node_start" and e["node"] == "analyst"]


def test_critic_loop_terminates_at_max_rounds():
    events = list(stream_briefing(
        "Will AI agents replace dashboards?",
        llm=FailCriticLLM(),
        config={"max_critic_rounds": 3},
    ))
    # 1 initial analyst run + 3 revision rounds, then it must stop.
    assert len(_analyst_starts(events)) == 4
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["data"]["revision_count"] == 3
    assert done[0]["data"]["critic_passed"] is False
    assert done[0]["data"]["final_answer"]


def test_critic_pass_stops_after_one_analyst_run():
    events = list(stream_briefing("test", llm=PassCriticLLM()))
    assert len(_analyst_starts(events)) == 1
    done = [e for e in events if e["type"] == "done"][0]
    assert done["data"]["critic_passed"] is True


def test_mock_end_to_end_critic_passes_on_cited_draft():
    # The real MockLLM analyst emits cited bullets, so the real MockLLM
    # critic should PASS without any stubbing.
    events = list(stream_briefing("What is retrieval-augmented generation?"))
    done = [e for e in events if e["type"] == "done"][0]
    assert done["data"]["critic_passed"] is True
    assert "[1]" in done["data"]["final_answer"]
