"""
Tests for Agent conversation, tool calling, and structured outputs.
"""
import pytest
from app.agent.engine import LocalAgentRunner
from app.schemas.agent import AgentChatRequest
from app.agent.memory import ContextMemoryManager


def test_agent_chat_search_intent():
    runner = LocalAgentRunner()
    req = AgentChatRequest(
        message="Search for Python backend jobs with remote work",
        session_id="test_session_1"
    )
    resp = runner.chat(req)

    assert resp.reply != ""
    assert len(resp.tool_calls) > 0
    assert resp.tool_calls[0].tool_name == "search_jobs"


def test_agent_memory_constraint_extraction():
    runner = LocalAgentRunner()
    req = AgentChatRequest(
        message="I am looking for jobs, but remember that I have no crypto tolerance",
        candidate_id="cand_01_alex",
        session_id="test_session_mem"
    )
    resp = runner.chat(req)

    # Agent should have identified the constraint and recorded memory or tool call
    tool_names = [tc.tool_name for tc in resp.tool_calls]
    assert "remember_candidate_fact" in tool_names or len(resp.active_memories) > 0
