from __future__ import annotations

"""
Mock backend implementation that simulates the behaviors from mock_collaboration_demo.py.
"""

import asyncio
import random
import time
from typing import Dict, List, Any, AsyncGenerator, Optional
from .base import LLMBackend, StreamChunk
from ..utils.api_tracer import trace_api_call


class MockBackend(LLMBackend):
    """Mock backend that follows simple, predictable rules."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or "mock-api-key"
        # Track what the last tool call was for each agent
        self.last_tool_calls = {}
    
    async def stream_with_tools(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                              has_called_update_summary: bool = False, has_voted: bool = False,
                              pending_notifications: int = 0, agent_ids: List[str] = None, 
                              provider_tools: Optional[List[str]] = None, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using simple mock logic.
        If there are pending notifications, it will return a tool call request of `check_updates`.

        Else:
        If `update_summary` is never called, it will return a tool call request of `update_summary`.

        If the previous tool call is `check_updates`, decide whether to call `update_summary` (a random choice).
        
        If the previous tool call is `check_updates` or `update_summary`, decide whether to call `vote` (a random choice).
        The voting target doesn't matter.
        
        If the previous tool call is `check_updates`, call `no_update`.

        After a tool call request is generated, the function should return.
        """
        
        # Determine call type for tracing
        call_type = self._determine_call_type(messages, has_called_update_summary, has_voted, pending_notifications)
        
        # Filter provider tools to supported ones
        filtered_provider_tools = self.filter_provider_tools(provider_tools)
        
        # Prepare request data for tracing
        request_data = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "provider_tools": filtered_provider_tools,
            "has_called_update_summary": has_called_update_summary,
            "has_voted": has_voted,
            "pending_notifications": pending_notifications,
            "agent_ids": agent_ids
        }
        
        # Trace the API call
        trace_api_call(
            provider="Mock",
            model=model,
            call_type=call_type,
            request_data=request_data,
            agent_id=kwargs.get("agent_id"),
            session_id=kwargs.get("session_id")
        )
        
        # Extract agent ID from system message or use a simple counter approach
        agent_key = "default_agent"
        if messages and len(messages) > 0:
            # Look for agent ID in system message
            system_msg = messages[0].get("content", "") if messages[0].get("role") == "system" else ""
            if "You are " in system_msg:
                # Extract agent ID from "You are {agent_id} in a multi-agent..."
                parts = system_msg.split("You are ")
                if len(parts) > 1:
                    agent_part = parts[1].split(" ")[0]
                    agent_key = agent_part
        
        # Fallback: use a simple approach based on has_called_update_summary state
        if agent_key == "default_agent":
            agent_key = f"agent_{hash(str(has_called_update_summary))}"
        
        # Get the last tool call for this agent key
        last_tool = self.last_tool_calls.get(agent_key, None)
        
        tool_call = None
        response_text = ""
        
        # Check if this is final answer presentation (no collaboration tools needed)
        if not tools or len(tools) == 0 or (len(tools) == 1 and tools[0]["function"]["name"] == "get_session_info"):
            # This is final answer presentation - extract the original task from messages
            original_task = "the given task"
            if messages:
                # Look for the original task in the messages (usually in system context)
                for msg in messages:
                    if msg.get("role") == "system" and "Original task:" in msg.get("content", ""):
                        # Extract text after "Original task:"
                        content = msg["content"]
                        if "Original task:" in content:
                            task_part = content.split("Original task:")[1].split("\n")[0].strip()
                            if task_part:
                                original_task = task_part
                        break
                    elif msg.get("role") == "user":
                        content = msg["content"]
                        if "Original task:" in content:
                            # Extract the task after "Original task:"
                            task_part = content.split("Original task:")[1].strip()
                            # Remove any additional instructions after the task
                            if "\n" in task_part:
                                task_part = task_part.split("\n")[0].strip()
                            if task_part:
                                original_task = task_part
                            break
            
            # Provide a clean final answer focused on the actual task
            response_text = f"""Answer to "{original_task}":

Through multi-agent collaboration, we analyzed this from multiple perspectives and reached consensus on the optimal approach.

Final collaborative answer: [Specific solution would be provided here based on the actual task analysis]"""
            # No tool calls needed for final presentation
            tool_call = None
        
        # Collaboration rules (only if not final presentation)
        elif pending_notifications > 0:
            tool_call = {
                "id": "call_check_updates",
                "name": "check_updates", 
                "arguments": {}
            }
            response_text = "Let me check the notifications."
            self.last_tool_calls[agent_key] = "check_updates"
        
        # Rule 2: If update_summary was never called, call it
        elif not has_called_update_summary:
            content = f"Initial analysis on the task (timestamp: {time.time():.0f})"
            tool_call = {
                "id": "call_update_summary",
                "name": "update_summary",
                "arguments": {"new_content": content}
            }
            response_text = f"Let me analyze this task. {content}"
            self.last_tool_calls[agent_key] = "update_summary"
        
        # Rule 3: If previous tool was check_updates, decide on update_summary
        elif last_tool == "check_updates":
            if random.choice([True, False]):  # Random choice to call update_summary
                content = f"Updated analysis after notifications (timestamp: {time.time():.0f})"
                tool_call = {
                    "id": "call_update_after_check",
                    "name": "update_summary",
                    "arguments": {"new_content": content}
                }
                response_text = f"After checking notifications, I'll update my analysis. {content}"
                self.last_tool_calls[agent_key] = "update_summary"

        # Rule 4: If previous tool was check_updates or update_summary, decide on vote
        elif last_tool in ["check_updates", "update_summary"]:
            if random.choice([True, False]):  # Random choice to vote
                # Vote for a random agent from the provided list
                vote_target = random.choice(agent_ids) if agent_ids else "self"
                tool_call = {
                    "id": "call_vote",
                    "name": "vote",
                    "arguments": {"agent_id": vote_target}
                }
                response_text = f"I'll vote for {vote_target}."
                self.last_tool_calls[agent_key] = "vote"

        # Rule 5: Final fallback - if check_updates was called but no other action, call vote or no_update
        if last_tool == "check_updates" and tool_call is None:
            # if not voted yet, vote
            if not has_voted:
                vote_target = random.choice(agent_ids) if agent_ids else "self"
                tool_call = {
                    "id": "call_vote_after_check",
                    "name": "vote",
                    "arguments": {"agent_id": vote_target}
                }
                response_text = f"I'll vote for {vote_target}."
                self.last_tool_calls[agent_key] = "vote"
            else:
                # if voted already:
                tool_call = {
                    "id": "call_no_update",
                    "name": "no_update",
                    "arguments": {}
                }
                response_text = "I've completed my work. No further updates needed."
                self.last_tool_calls[agent_key] = "no_update"
                        
        # Stream the response text
        if response_text:
            words = response_text.split()
            for i in range(0, len(words), 5):  # Stream in smaller chunks
                chunk = " ".join(words[i:i+5]) + " "
                yield StreamChunk(type="content", content=chunk)
                await asyncio.sleep(0.01)  # Small delay for realism
        
        # Then yield the single tool call
        if tool_call:
            yield StreamChunk(
                type="tool_calls", 
                content="",
                tool_calls=[tool_call]  # Always exactly one tool call
            )
            return
        
        # Signal completion
        yield StreamChunk(type="done")
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "Mock"
    
    def get_supported_builtin_tools(self) -> List[str]:
        """Get list of builtin tools supported by Mock."""
        return ["web_search", "code_interpreter"]  # Mock supports all for testing
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        return int(len(text.split()) * 1.3)  # Rough token estimate
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for token usage (mock pricing)."""
        # Mock pricing: $0.001 per 1000 tokens
        return (input_tokens + output_tokens) * 0.001 / 1000
    
    def _determine_call_type(self, messages: List[Dict[str, Any]], has_called_update_summary: bool, 
                           has_voted: bool, pending_notifications: int) -> str:
        """Determine the type of API call for tracing purposes."""
        if not messages:
            return "initial"
        
        last_message = messages[-1].get("content", "")
        
        if "You were selected as the representative" in last_message:
            return "final"
        elif "System instruction:" in last_message or pending_notifications > 0:
            return "restart"
        elif any("tool_calls" in msg for msg in messages if isinstance(msg, dict)):
            return "tool_response"
        else:
            return "initial"