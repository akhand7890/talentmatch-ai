"""
Conversational Agent API endpoints.
Provides tool-augmented reasoning, context memory retrieval, and structured matching reports.
"""
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.agent import AgentChatRequest, AgentChatResponse
from app.agent.engine import get_agent_runner
from app.agent.memory import get_memory_manager

router = APIRouter(prefix="/agent", tags=["AI Agent Co-Pilot"])


@router.post("/chat", response_model=AgentChatResponse)
def chat_with_agent(request: AgentChatRequest):
    """
    Primary agent interaction endpoint:
    - Maintains conversational memory (short-term & long-term candidate facts)
    - Dynamically decides when to call tools (search_jobs, match_candidate, remember_fact)
    - Outputs structured recommendations and gap analysis
    """
    agent = get_agent_runner()
    response = agent.chat(request)
    return response


@router.get("/sessions/{session_id}")
def get_session_history(session_id: str):
    """Inspects the short-term conversation context for a session."""
    mem_mgr = get_memory_manager()
    return {"session_id": session_id, "messages": mem_mgr.get_history(session_id)}


@router.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Clears short-term conversational context."""
    mem_mgr = get_memory_manager()
    mem_mgr.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
