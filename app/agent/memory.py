"""
Memory manager for conversational context and persistent candidate preferences.
Maintains short-term chat window per session and long-term candidate facts from database.
"""
from typing import Dict, List, Optional, Any
from app.db.session import SessionLocal
from app.db.models import Candidate, CandidateMemory
from app.core.config import settings
from app.core.logger import logger


class ContextMemoryManager:
    """
    Manages both short-term (in-memory multi-turn session)
    and long-term (SQL database backed) context for the AI Agent.
    """

    def __init__(self):
        # session_id -> list of {"role": "user"|"assistant"|"system"|"tool", "content": str, ...}
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self._sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str, **kwargs) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self._sessions[session_id].append(msg)

        # Trim short-term memory to configured max turns
        max_turns = settings.MAX_CHAT_TURNS_MEMORY * 2  # user + assistant pairs
        if len(self._sessions[session_id]) > max_turns:
            self._sessions[session_id] = self._sessions[session_id][-max_turns:]

    def load_candidate_context(self, candidate_id: str) -> Dict[str, Any]:
        """Loads long-term candidate profile and recorded memories from database."""
        db = SessionLocal()
        try:
            candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
            if not candidate:
                return {"candidate": None, "memories": []}

            cand_dict = candidate.to_dict()
            memories = [m.to_dict() for m in candidate.memories]
            return {
                "candidate": cand_dict,
                "memories": memories,
                "summary": candidate.to_profile_summary()
            }
        finally:
            db.close()

    def build_agent_context(
        self,
        session_id: str,
        candidate_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Assembles full prompt messages list:
        1. Base System Prompt
        2. Injected Candidate Profile & Long-term Memories
        3. Short-term Session Conversation History
        """
        from app.agent.prompts import AGENT_SYSTEM_PROMPT

        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

        if candidate_id:
            ctx = self.load_candidate_context(candidate_id)
            if ctx.get("candidate"):
                memories_text = "\n".join([f"- [{m['category'].upper()}] {m['content']}" for m in ctx.get("memories", [])])
                context_injection = (
                    f"\n\n### ACTIVE CANDIDATE CONTEXT:\n"
                    f"{ctx['summary']}\n\n"
                    f"### REMEMBERED CONSTRAINTS & PREFERENCES:\n"
                    f"{memories_text if memories_text else 'No prior memories recorded.'}\n"
                    f"Always adhere strictly to these constraints when recommending positions."
                )
                messages[0]["content"] += context_injection

        # Append short-term session turns
        messages.extend(self.get_history(session_id))
        return messages

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]


_memory_manager_instance: Optional[ContextMemoryManager] = None


def get_memory_manager() -> ContextMemoryManager:
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = ContextMemoryManager()
    return _memory_manager_instance
