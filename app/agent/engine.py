"""
Core Agent Engine executing Tool-calling ReAct loops and Structured Output generation.
Supports OpenAI / Groq / Gemini, with an intelligent offline Deterministic Agent fallback.
"""
import json
import re
from typing import List, Dict, Any, Optional
from app.agent.tools import AGENT_TOOLS_SCHEMA, ToolExecutor
from app.agent.memory import get_memory_manager, ContextMemoryManager
from app.schemas.agent import AgentChatRequest, AgentChatResponse, ToolCallRecord, StructuredMatchReport
from app.core.config import settings
from app.core.logger import logger


class BaseAgentRunner:
    def chat(self, request: AgentChatRequest) -> AgentChatResponse:
        raise NotImplementedError


class LocalAgentRunner(BaseAgentRunner):
    """
    Intelligent zero-cost local agent runner.
    Employs intent analysis, constraint extraction, automatic tool execution,
    context-aware reasoning, and structured output formatting.
    """

    def __init__(self, memory_mgr: Optional[ContextMemoryManager] = None):
        self.memory_mgr = memory_mgr or get_memory_manager()

    def chat(self, request: AgentChatRequest) -> AgentChatResponse:
        session_id = request.session_id or "default_session"
        candidate_id = request.candidate_id
        user_msg = request.message.strip()

        # Add user message to conversation memory
        self.memory_mgr.add_message(session_id, "user", user_msg)

        tool_calls: List[ToolCallRecord] = []
        structured_report: Optional[StructuredMatchReport] = None
        active_memories: List[str] = []

        # Load existing candidate context
        cand_ctx = self.memory_mgr.load_candidate_context(candidate_id) if candidate_id else {}
        for m in cand_ctx.get("memories", []):
            active_memories.append(f"[{m['category']}] {m['content']}")

        # Step 1: Detect and extract new constraints / preferences to remember
        msg_lower = user_msg.lower()
        if candidate_id and settings.AUTO_EXTRACT_CANDIDATE_MEMORY:
            self._extract_and_save_memories(user_msg, candidate_id, tool_calls, active_memories)

        # Step 2: Determine tool action based on intent
        reply_text = ""

        # Intent A: Personalized Matching Request
        if any(w in msg_lower for w in ["match", "recommend", "best job", "suitable for me", "roles for me", "profile"]) and candidate_id:
            logger.info("Executing match_candidate_profile tool...")
            tool_res = ToolExecutor.execute_tool(
                "match_candidate_profile",
                {"candidate_id": candidate_id, "custom_query": user_msg, "limit": 4}
            )
            tool_calls.append(
                ToolCallRecord(
                    tool_name="match_candidate_profile",
                    arguments={"candidate_id": candidate_id, "custom_query": user_msg},
                    output=tool_res["result"],
                    duration_ms=tool_res["duration_ms"]
                )
            )

            if "top_matches" in tool_res["result"]:
                structured_report = StructuredMatchReport(**tool_res["result"])
                top_items = structured_report.top_matches
                if top_items:
                    lines = [
                        f"### 🎯 Personalized Job Recommendations for {structured_report.candidate_name}:",
                        "",
                        f"**Analysis Summary**: {structured_report.executive_summary}",
                        ""
                    ]
                    for idx, m in enumerate(top_items, 1):
                        lines.append(f"#### {idx}. {m.title} @ {m.company} (Fit Score: **{m.fit_score}%**)")
                        lines.append(f"- **Location & Mode:** {m.location} ({m.remote_type})")
                        lines.append(f"- **Compensation Alignment:** {m.salary_fit}")
                        lines.append(f"- **Matching Skills:** {', '.join(m.matching_skills) if m.matching_skills else 'Baseline fit'}")
                        if m.missing_skills:
                            lines.append(f"- **Gaps to Bridge:** {', '.join(m.missing_skills)}")
                        lines.append(f"- **Why Recommended:** {m.why_recommended}")
                        lines.append("")

                    if structured_report.recommended_upskilling:
                        lines.append(f"**Recommended Upskilling:** {', '.join(structured_report.recommended_upskilling)}")

                    reply_text = "\n".join(lines)
                else:
                    reply_text = f"I evaluated your profile against active openings, but couldn't find strong matches matching your current constraints. Would you like to adjust your filters?"

        # Intent B: Job Search Request
        elif any(w in msg_lower for w in ["search", "find", "jobs", "openings", "looking for", "hire", "python", "developer", "engineer"]):
            # Extract filters from text
            remote_type = None
            if "remote" in msg_lower:
                remote_type = "remote"
            elif "hybrid" in msg_lower:
                remote_type = "hybrid"
            elif "onsite" in msg_lower or "on-site" in msg_lower:
                remote_type = "on-site"

            # Salary extraction
            min_salary = None
            salary_match = re.search(r"\$?(\d{2,3})[kK]", user_msg)
            if salary_match:
                min_salary = int(salary_match.group(1)) * 1000
            else:
                salary_match_num = re.search(r"\$?(\d{5,6})", user_msg)
                if salary_match_num:
                    min_salary = int(salary_match_num.group(1))

            tool_res = ToolExecutor.execute_tool(
                "search_jobs",
                {
                    "query": user_msg,
                    "remote_type": remote_type,
                    "min_salary": min_salary,
                    "limit": 5
                }
            )
            tool_calls.append(
                ToolCallRecord(
                    tool_name="search_jobs",
                    arguments={"query": user_msg, "remote_type": remote_type, "min_salary": min_salary},
                    output=tool_res["result"],
                    duration_ms=tool_res["duration_ms"]
                )
            )

            jobs = tool_res["result"].get("jobs", [])
            if jobs:
                lines = [f"Found **{len(jobs)}** matching positions in the live jobs index:\n"]
                for j in jobs:
                    lines.append(
                        f"- **{j['title']}** at **{j['company']}** ({j['location']}, {j['remote_type']})\n"
                        f"  - **Compensation:** {j['salary_range']}\n"
                        f"  - **Required Skills:** {', '.join(j['required_skills'])}\n"
                        f"  - **Relevance Score:** {round(j['match_score'] * 100, 1)}%\n"
                    )
                reply_text = "\n".join(lines)
            else:
                reply_text = "I searched our live job index with your parameters, but didn't find active positions matching all constraints. Try broadening the search criteria."

        # Intent C: General Conversation / Profile Inquiry
        else:
            if candidate_id and cand_ctx.get("candidate"):
                cand = cand_ctx["candidate"]
                reply_text = (
                    f"Hello {cand['name']}! I'm your TalentMatch AI co-pilot. I have your profile loaded "
                    f"({cand['headline']}, {cand['years_of_experience']}y experience). "
                    f"I can match you to positions, search the changing jobs index with specific salary/remote criteria, "
                    f"or track your career constraints. What role would you like to explore today?"
                )
            else:
                reply_text = (
                    "Hello! I am TalentMatch AI. I can search through our live semantic jobs index, "
                    "match candidates to opportunities using 2-stage retrieval and ranking, "
                    "and remember preferences and constraints. How can I help you today?"
                )

        # Store assistant response to memory
        self.memory_mgr.add_message(session_id, "assistant", reply_text)

        return AgentChatResponse(
            reply=reply_text,
            session_id=session_id,
            candidate_id=candidate_id,
            tool_calls=tool_calls,
            structured_report=structured_report,
            active_memories=active_memories
        )

    def _extract_and_save_memories(
        self,
        msg: str,
        candidate_id: str,
        tool_calls: List[ToolCallRecord],
        active_memories: List[str]
    ) -> None:
        """Extracts and persists constraints (e.g. no crypto, minimum salary) into long-term SQL memory."""
        msg_lower = msg.lower()

        # Check rejection / constraints
        for keyword in ["no crypto", "no blockchain", "avoid web3", "no gambling", "no defense", "no relocation", "fully remote only"]:
            if keyword in msg_lower:
                res = ToolExecutor.execute_tool(
                    "remember_candidate_fact",
                    {
                        "candidate_id": candidate_id,
                        "category": "constraint",
                        "content": f"Candidate constraint: {keyword}"
                    }
                )
                tool_calls.append(
                    ToolCallRecord(
                        tool_name="remember_candidate_fact",
                        arguments={"candidate_id": candidate_id, "category": "constraint", "content": keyword},
                        output=res["result"],
                        duration_ms=res["duration_ms"]
                    )
                )
                active_memories.append(f"[constraint] Candidate constraint: {keyword}")


class OpenAIAgentRunner(BaseAgentRunner):
    """
    OpenAI Function/Tool Calling Agent Runner.
    Used when OPENAI_API_KEY is supplied in configuration.
    """

    def __init__(self, memory_mgr: Optional[ContextMemoryManager] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.memory_mgr = memory_mgr or get_memory_manager()

    def chat(self, request: AgentChatRequest) -> AgentChatResponse:
        session_id = request.session_id or "default_session"
        candidate_id = request.candidate_id
        user_msg = request.message.strip()

        # Add user message
        self.memory_mgr.add_message(session_id, "user", user_msg)
        messages = self.memory_mgr.build_agent_context(session_id, candidate_id)

        tool_calls_logged: List[ToolCallRecord] = []
        structured_report: Optional[StructuredMatchReport] = None

        max_iterations = 4
        current_iter = 0

        while current_iter < max_iterations:
            current_iter += 1
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                tools=AGENT_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                # Add tool calls to messages
                messages.append(msg)
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)

                    exec_res = ToolExecutor.execute_tool(fn_name, fn_args)
                    tool_calls_logged.append(
                        ToolCallRecord(
                            tool_name=fn_name,
                            arguments=fn_args,
                            output=exec_res["result"],
                            duration_ms=exec_res["duration_ms"]
                        )
                    )

                    if fn_name == "match_candidate_profile" and "top_matches" in exec_res["result"]:
                        structured_report = StructuredMatchReport(**exec_res["result"])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": json.dumps(exec_res["result"])
                    })
            else:
                # Finished reasoning
                reply_text = msg.content or ""
                self.memory_mgr.add_message(session_id, "assistant", reply_text)

                # Fetch active memories
                active_mems = []
                if candidate_id:
                    ctx = self.memory_mgr.load_candidate_context(candidate_id)
                    active_mems = [f"[{m['category']}] {m['content']}" for m in ctx.get("memories", [])]

                return AgentChatResponse(
                    reply=reply_text,
                    session_id=session_id,
                    candidate_id=candidate_id,
                    tool_calls=tool_calls_logged,
                    structured_report=structured_report,
                    active_memories=active_mems
                )

        return AgentChatResponse(
            reply="I processed your request using the tools, but reached the maximum reasoning steps.",
            session_id=session_id,
            candidate_id=candidate_id,
            tool_calls=tool_calls_logged,
            structured_report=structured_report
        )


def get_agent_runner() -> BaseAgentRunner:
    if settings.OPENAI_API_KEY and settings.LLM_PROVIDER == "openai":
        try:
            logger.info("Using OpenAI Agent Runner...")
            return OpenAIAgentRunner()
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI runner: {e}. Falling back to LocalAgentRunner.")
    return LocalAgentRunner()
