"""Simplified MassAgent interface designed for LLM-based agents."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncGenerator
from enum import Enum
from pathlib import Path

from .agent_config import AgentConfig
from .backends.factory import create_backend


@dataclass
class WorkingSummary:
    """Agent's working summary that gets shared with other agents."""
    content: str
    version: int = 0
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    
    def copy(self) -> 'WorkingSummary':
        """Create a copy of this working summary."""
        return WorkingSummary(
            content=self.content,
            version=self.version,
            timestamp=self.timestamp
        )


class AgentStatus(Enum):
    """Agent status in the session."""
    INITIALIZING = "initializing"
    WORKING = "working"
    STOPPED = "stopped"
    VOTING = "voting"
    PRESENTING = "presenting"


@dataclass
class SessionInfo:
    """Session information available to agents."""
    session_id: str
    task: str
    start_time: float
    current_time: float
    agent_statuses: Dict[str, str]
    total_agents: int
    stopped_agents: int
    votes_cast: int
    time_elapsed: float
    estimated_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task": self.task,
            "time_elapsed": self.time_elapsed,
            "agent_statuses": self.agent_statuses,
            "total_agents": self.total_agents,
            "stopped_agents": self.stopped_agents,
            "votes_cast": self.votes_cast,
            "estimated_cost": self.estimated_cost
        }


@dataclass 
class NotificationUpdate:
    """Notification about another agent's update."""
    agent_id: str
    summary: str
    timestamp: float
    version: int


class MassAgent:
    """Simplified agent interface designed for LLM-based agents.
    
    Provides 5 LLM-callable tools and unified streaming interface.
    """
    
    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        
        # Create backend for LLM interactions
        self.backend = create_backend(config.backend_params.get("model"), config.backend_params.get("api_key"))
        
        # Working state
        self.working_summary = WorkingSummary("")
        self.status = AgentStatus.INITIALIZING
        self.has_voted = False
        self.vote_target: Optional[str] = None
        self.has_declared_no_update = False
        
        # Tool usage tracking for validation and restart logic
        self.has_called_update_summary = False
        self.work_session_tool_calls: List[str] = []  # Tools called in current work session
        self.all_tool_calls: List[str] = []  # All tools called across entire session
        
        # Notification queue (queued by default until check_updates called)
        self.pending_notifications: List[NotificationUpdate] = []
        self.processed_notification_ids: set = set()
        
        # Session reference (set by orchestrator)
        self.session_info: Optional[SessionInfo] = None
        self.orchestrator = None  # Reference to orchestrator for tool calls
        
        # Summary file management
        self.summaries_dir = Path("summaries")
        self.summaries_dir.mkdir(exist_ok=True)
        
        # Cost tracking (delegated to backend)
        # Note: Token tracking is now handled by the backend
        
    # === LLM-Callable Tools (provided by system) ===
    
    async def update_summary(self, new_content: str) -> str:
        """Update working summary, triggers notifications to others."""
        if self.status in [AgentStatus.STOPPED, AgentStatus.VOTING, AgentStatus.PRESENTING]:
            return f"Error: Cannot update summary in status {self.status.value}"
        
        # Track tool usage
        self.has_called_update_summary = True
        self.work_session_tool_calls.append("update_summary")
        self.all_tool_calls.append("update_summary")
        
        # Update working summary only if content actually changed
        old_content = self.working_summary.content
        if new_content.strip() != old_content.strip():
            # Content actually changed - update and notify
            self.working_summary = WorkingSummary(
                content=new_content,
                version=self.working_summary.version + 1
            )
            
            # Save to file
            summary_file = await self._save_summary_to_file(self.working_summary)
            
            # Notify orchestrator to queue notifications to other agents
            if self.orchestrator:
                await self.orchestrator.notify_summary_update(self.agent_id, self.working_summary)
            
            return f"Summary updated to version {self.working_summary.version}. Saved to {summary_file}"
        else:
            # No actual change - don't increment version or send notifications
            return f"No change to summary content (current version {self.working_summary.version})"
    
    async def check_updates(self) -> Dict[str, Any]:
        """Get dict of updated summaries from other agents."""
        if not self.pending_notifications:
            return {"updates": [], "message": "No new updates"}
        
        # Process all pending notifications
        updates = []
        for notification in self.pending_notifications:
            if notification.agent_id not in self.processed_notification_ids:
                updates.append({
                    "agent_id": notification.agent_id,
                    "summary": notification.summary,
                    "timestamp": notification.timestamp,
                    "version": notification.version
                })
                self.processed_notification_ids.add(f"{notification.agent_id}_{notification.version}")
        
        # Clear processed notifications
        self.pending_notifications = []
        
        return {
            "updates": updates,
            "message": f"Retrieved {len(updates)} updates"
        }
    
    async def no_update(self) -> str:
        """Signal that no further updates are needed."""
        # Validate that update_summary has been called before
        if not self.has_called_update_summary:
            return "Error: Must call update_summary() before declaring no_update"
        
        # Validate that vote has been called before
        if not self.has_voted:
            return "Error: Must call vote() before declaring no_update"
        
        # Validate agent hasn't already declared no_update
        if self.has_declared_no_update:
            return "Error: Already declared no_update"
        
        # Validate notifications have been processed
        if self.pending_notifications:
            return f"Error: Must process {len(self.pending_notifications)} pending notifications before declaring no_update"
        
        # Track tool usage and set status
        self.has_declared_no_update = True
        self.work_session_tool_calls.append("no_update")
        self.all_tool_calls.append("no_update")
        self.status = AgentStatus.STOPPED
        
        return "Declared no further updates needed"
    
    async def vote(self, agent_id: str) -> str:
        """Vote for representative (can be called multiple times to change vote)."""
        # Validate that update_summary has been called before
        if not self.has_called_update_summary:
            return "Error: Must call update_summary() before voting"
        
        # Cannot vote after declaring no_update (that signals finality)
        if self.has_declared_no_update:
            return "Error: Cannot vote after declaring no_update"
        
        # Validate notifications have been processed
        if self.pending_notifications:
            return f"Error: Must process {len(self.pending_notifications)} pending notifications before voting"
        
        # Validate agent_id exists in session
        if self.session_info and agent_id not in self.session_info.agent_statuses:
            # Additional check: if orchestrator is available, check if agent exists there
            if self.orchestrator and agent_id in self.orchestrator.agents:
                # Agent exists in orchestrator but not in session_info - this is a race condition
                # Log it and allow the vote to proceed
                print(f"Warning: Agent {agent_id} found in orchestrator but not in session_info - proceeding with vote")
            else:
                # Agent doesn't exist - try to find a valid fallback target
                if self.orchestrator:
                    other_agents = [aid for aid in self.orchestrator.agents.keys() if aid != self.agent_id]
                    if other_agents:
                        fallback_target = other_agents[0]
                        print(f"Warning: Agent {agent_id} not found, voting for {fallback_target} instead")
                        agent_id = fallback_target  # Use fallback target
                    else:
                        return f"Error: Agent {agent_id} not found in session"
                else:
                    return f"Error: Agent {agent_id} not found in session"
        
        # Track tool usage and record vote
        if "vote" not in self.work_session_tool_calls:
            self.work_session_tool_calls.append("vote")
        if "vote" not in self.all_tool_calls:
            self.all_tool_calls.append("vote")
        
        # Allow changing vote
        old_vote = self.vote_target
        self.has_voted = True
        self.vote_target = agent_id
        
        # Notify orchestrator
        if self.orchestrator:
            await self.orchestrator.register_vote(self.agent_id, agent_id)
        
        if old_vote and old_vote != agent_id:
            return f"Changed vote from {old_vote} to {agent_id}"
        else:
            return f"Voted for {agent_id}"
    
    async def get_session_info(self, include_summaries: bool = False) -> Dict[str, Any]:
        """Get session context including agent statuses, time, costs."""
        if not self.session_info:
            return {"error": "Session info not available"}
        
        info = self.session_info.to_dict()
        
        if include_summaries and self.orchestrator:
            info["agent_summaries"] = await self.orchestrator.get_all_summaries()
        
        # Add this agent's specific info
        info["your_status"] = self.status.value
        info["your_summary_version"] = self.working_summary.version
        info["pending_notifications"] = len(self.pending_notifications)
        info["has_voted"] = self.has_voted
        info["vote_target"] = self.vote_target
        
        return info
    
    # === Core Methods ===
    
    async def process_message(self, message: str) -> AsyncGenerator[str, None]:
        """Common async streaming method for all LLM interactions with tool calling.
        
        This properly handles the complete tool calling flow:
        1. Call LLM API with streaming to get response (may include tool calls)
        2. If LLM requests tools, execute them and send results back
        3. Continue until LLM provides final response without tool calls
        """
        # Build messages for LLM
        system_context = self._build_system_context()
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": message}
        ]
        
        # Get available tools for this agent's current status
        tools = self._get_tool_schema()
        
        # Add user-provided tools (with collision detection)
        tools.extend(self._get_user_tools())
        
        try:
            # Tool calling loop - continue until LLM stops requesting tools
            while True:
                # Call backend with streaming
                response_content = ""
                # Get list of agent IDs from orchestrator for voting
                agent_ids = list(self.orchestrator.agents.keys()) if self.orchestrator else [self.agent_id]
                
                # Get provider tools
                provider_tools = self._get_provider_tools()
                
                async for chunk in self.backend.stream_with_tools(
                    model=self.config.backend_params.get("model"),
                    messages=messages,
                    tools=tools,
                    max_tokens=self.config.backend_params.get("max_tokens"),
                    temperature=self.config.backend_params.get("temperature"),
                    has_called_update_summary=self.has_called_update_summary,
                    has_voted=self.has_voted,
                    pending_notifications=len(self.pending_notifications),
                    agent_ids=agent_ids,
                    provider_tools=provider_tools,
                    agent_id=self.agent_id,
                    session_id=getattr(self.orchestrator, 'session_id', None) if self.orchestrator else None
                ):
                    if chunk.type == "content":
                        # Stream text content from LLM
                        response_content += chunk.content
                        yield chunk.content
                    elif chunk.type == "tool_calls":
                        # LLM requested tool calls
                        tool_calls = chunk.tool_calls
                        
                        # Add assistant message to conversation
                        await self._add_assistant_message_to_conversation(messages, chunk)
                        
                        # Execute tool calls in parallel and process results as they complete
                        if len(tool_calls) == 1:
                            # Single tool call - no need for parallelization
                            tool_result = await self._execute_tool_call({
                                "name": tool_calls[0]["name"],
                                "arguments": tool_calls[0]["arguments"]
                            })
                            
                            # Add tool result to conversation
                            await self._add_tool_result_to_conversation(messages, tool_calls[0], tool_result)
                            
                            # Yield tool execution feedback with file link if available
                            feedback = f"\n[Tool: {tool_result['function']}] {tool_result['result']}"
                            if tool_result['function'] == 'update_summary' and 'Saved to' in tool_result['result']:
                                feedback += f"\n📄 Click to view: {tool_result['result'].split('Saved to ')[-1]}"
                            yield feedback + "\n"
                        else:
                            # Multiple tool calls - execute in parallel
                            tasks = {
                                asyncio.create_task(self._execute_tool_call({
                                    "name": tool_call["name"],
                                    "arguments": tool_call["arguments"]
                                })): tool_call
                                for tool_call in tool_calls
                            }
                            
                            # Process results as they complete
                            for completed_task in asyncio.as_completed(tasks):
                                tool_result = await completed_task
                                tool_call = tasks[completed_task]
                                
                                # Add tool result to conversation
                                await self._add_tool_result_to_conversation(messages, tool_call, tool_result)
                                
                                # Yield tool execution feedback with file link if available
                                feedback = f"\n[Tool: {tool_result['function']}] {tool_result['result']}"
                                if tool_result['function'] == 'update_summary' and 'Saved to' in tool_result['result']:
                                    feedback += f"\n📄 Click to view: {tool_result['result'].split('Saved to ')[-1]}"
                                yield feedback + "\n"
                        
                        # Continue loop to get LLM's response to tool results
                        break
                    elif chunk.type == "done":
                        # LLM is done - no more tool calls
                        # Update token usage tracking
                        self.backend.update_token_usage(messages, response_content, self.config.backend_params.get("model"))
                        return
                    elif chunk.type == "error":
                        yield f"\n[Error: {chunk.error}]\n"
                        return
            
        except Exception as e:
            error_msg = f"[{self.agent_id}] Backend Error: {e}"
            yield error_msg
            # Fall back to basic response if backend fails
            basic_response = f"I'm working on: {message[:100]}{'...' if len(message) > 100 else ''}"
            yield basic_response
    
    async def work_on_task(self, task: str, restart_instruction: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Work on task - handles both initial work and restarts."""
        # Reset work session state
        self.status = AgentStatus.WORKING
        self.work_session_tool_calls = []
        
        # Allow re-declaring after restart (but preserve vote and no_update state)
        if restart_instruction:
            # Only reset no_update flag - vote should persist until explicitly changed
            self.has_declared_no_update = False
            # Note: has_voted and vote_target are preserved across restarts
        
        # Build appropriate instruction
        if restart_instruction:
            work_instruction = self._build_restart_instruction(task, restart_instruction)
        else:
            work_instruction = self._build_work_instruction(task)
        
        async for chunk in self.process_message(work_instruction):
            yield chunk
    
    async def present_final_answer(self, task: str) -> AsyncGenerator[str, None]:
        """Present final answer as representative (uses process_message)."""
        self.status = AgentStatus.PRESENTING
        
        # Build final answer instruction from config
        final_instruction = self._build_final_answer_instruction(task)
        
        async for chunk in self.process_message(final_instruction):
            yield chunk
    
    # === LLM Integration Methods ===
    
    async def _add_assistant_message_to_conversation(self, messages, chunk):
        """Add assistant message to conversation based on backend type."""
        backend_name = self.backend.get_provider_name()
        
        if backend_name == "OpenAI":
            # OpenAI format
            messages.append({
                "role": "assistant", 
                "content": chunk.content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": str(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"]
                        }
                    }
                    for tc in chunk.tool_calls
                ]
            })
        elif backend_name == "Anthropic":
            # Anthropic format - assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": chunk.content or ""
            })
        else:
            # Mock format
            messages.append({
                "role": "assistant",
                "content": chunk.content or ""
            })
    
    async def _add_tool_result_to_conversation(self, messages, tool_call, tool_result):
        """Add tool result to conversation based on backend type."""
        backend_name = self.backend.get_provider_name()
        
        if backend_name == "OpenAI":
            # OpenAI format
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(tool_result["result"])
            })
        elif backend_name == "Anthropic":
            # Anthropic format
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": str(tool_result["result"])
                }]
            })
        else:
            # Mock format - simple user message
            messages.append({
                "role": "user",
                "content": f"Tool {tool_result['function']} result: {tool_result['result']}"
            })
    
    def _get_tool_schema(self):
        """Get tool schema for LLM function calling."""
        if self.status == AgentStatus.PRESENTING:
            # Limited tools for final answer
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "get_session_info",
                        "description": "Get session status and final state information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "include_summaries": {
                                    "type": "boolean",
                                    "description": "Whether to include agent summaries"
                                }
                            }
                        }
                    }
                }
            ]
        else:
            # Full tools for collaboration
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "update_summary",
                        "description": "Update your working summary with new insights",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "new_content": {
                                    "type": "string",
                                    "description": "The updated summary content"
                                }
                            },
                            "required": ["new_content"]
                        }
                    }
                },
                {
                    "type": "function", 
                    "function": {
                        "name": "check_updates",
                        "description": "Check for updates from other agents",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vote",
                        "description": "Vote for the best representative agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "agent_id": {
                                    "type": "string",
                                    "description": "ID of the agent to vote for"
                                }
                            },
                            "required": ["agent_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "no_update",
                        "description": "Signal that no further updates are needed",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_session_info",
                        "description": "Get session status, time elapsed, and agent progress",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "include_summaries": {
                                    "type": "boolean",
                                    "description": "Whether to include agent summaries"
                                }
                            }
                        }
                    }
                }
            ]
    
    def _get_provider_tools(self) -> List[str]:
        """Get provider tools based on configuration."""
        if self.config.provider_tools is None:
            # None means use all supported tools
            return self.backend.get_supported_builtin_tools()
        elif self.config.provider_tools == []:
            # Empty list means no provider tools
            return []
        else:
            # Use specified tools (filtered by what's actually supported)
            supported = self.backend.get_supported_builtin_tools()
            return [tool for tool in self.config.provider_tools if tool in supported]
    
    def _get_user_tools(self) -> List[Dict[str, Any]]:
        """Get user-provided tools with collision detection."""
        if not self.config.user_tools:
            return []
        
        # Get reserved MASS framework tool names
        framework_tools = self._get_tool_schema()
        framework_tool_names = {tool["function"]["name"] for tool in framework_tools}
        
        # Check for collisions and filter out conflicting tools
        safe_user_tools = []
        for tool in self.config.user_tools:
            tool_name = tool["function"]["name"]
            if tool_name in framework_tool_names:
                print(f"Warning: User tool '{tool_name}' conflicts with MASS framework tool. Skipping.")
                continue
            safe_user_tools.append(tool)
        
        return safe_user_tools
    
    async def _execute_tool_call(self, tool_call):
        """Execute a tool call from the LLM response."""
        function_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        try:
            if function_name == "update_summary":
                result = await self.update_summary(arguments.get("new_content", ""))
                return {"function": function_name, "result": result}
                
            elif function_name == "check_updates":
                result = await self.check_updates()
                return {"function": function_name, "result": f"Found {len(result.get('updates', []))} updates"}
                
            elif function_name == "vote":
                result = await self.vote(arguments.get("agent_id"))
                return {"function": function_name, "result": result}
                
            elif function_name == "no_update":
                result = await self.no_update()
                return {"function": function_name, "result": result}
                
            elif function_name == "get_session_info":
                result = await self.get_session_info(arguments.get("include_summaries", False))
                return {"function": function_name, "result": f"Session info with {len(result)} fields"}
                
            else:
                return {"function": function_name, "result": f"Unknown function: {function_name}"}
                
        except Exception as e:
            return {"function": function_name, "result": f"ERROR: {e}"}
    
    
    # Token tracking has been moved to backend implementations
    
    # === Internal Methods ===
    
    def _build_system_context(self) -> str:
        """Build system context with available tools using centralized templates."""
        from .message_templates import get_templates
        templates = self.config.message_templates or get_templates()
        
        if self.status == AgentStatus.PRESENTING:
            # Final answer phase - limited tools
            parts = [
                templates.system_agent_intro(self.agent_id, "presenting"),
                "",
                templates.system_tool_descriptions("final"),
                "",
                templates.system_final_answer_guidance(),
                "",
                f"Current working summary: {self.working_summary.content}",
                f"Status: {self.status.value}"
            ]
        else:
            # Collaboration phase - full tools
            parts = [
                templates.system_agent_intro(self.agent_id, "collaboration"),
                "",
                templates.system_tool_descriptions("collaboration"),
                "",
                templates.system_tool_rules(),
                "",
                f"Current working summary: {self.working_summary.content}",
                f"Status: {self.status.value}",
                f"Pending notifications: {len(self.pending_notifications)}"
            ]
        
        return "\n".join(parts)
    
    def _build_work_instruction(self, task: str) -> str:
        """Build work instruction from centralized templates."""
        templates = self.config.message_templates
        if templates:
            return templates.build_initial_work_instruction(task)
        else:
            from .message_templates import build_initial_work_instruction
            return build_initial_work_instruction(task)
    
    def _build_restart_instruction(self, task: str, instruction: str) -> str:
        """Build restart instruction from centralized templates."""
        templates = self.config.message_templates
        has_notifications = len(self.pending_notifications) > 0
        if templates:
            return templates.build_restart_instruction(task, instruction, has_notifications)
        else:
            from .message_templates import build_restart_instruction
            return build_restart_instruction(task, instruction, has_notifications)
    
    def _build_final_answer_instruction(self, task: str) -> str:
        """Build final answer instruction from centralized templates."""
        templates = self.config.message_templates
        if templates:
            return templates.build_final_answer_instruction(task)
        else:
            from .message_templates import build_final_answer_instruction
            return build_final_answer_instruction(task)
    
    # === System Methods (called by orchestrator) ===
    
    async def queue_notification(self, notification: NotificationUpdate):
        """Queue notification from another agent."""
        self.pending_notifications.append(notification)
    
    def should_restart(self) -> tuple[bool, list[str]]:
        """Check if agent should be restarted and return all applicable reasons."""
        # Only restart stopped agents
        if self.status != AgentStatus.STOPPED:
            return False, []
        
        reasons = []
        
        # Check for unprocessed notifications
        if self.pending_notifications:
            reasons.append("notifications")
        
        # Check if hasn't voted (unless they declared no_update)
        if not self.has_voted and not self.has_declared_no_update:
            reasons.append("missing_vote")
        
        # Check if didn't call any required tools in their last work session
        required_tools = {"update_summary", "vote", "no_update"}
        if not any(tool in required_tools for tool in self.work_session_tool_calls):
            reasons.append("missing_tools")
        
        return len(reasons) > 0, reasons
    
    def get_summary(self) -> WorkingSummary:
        """Get current working summary."""
        return self.working_summary
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "summary_version": self.working_summary.version,
            "has_voted": self.has_voted,
            "vote_target": self.vote_target,
            "has_declared_no_update": self.has_declared_no_update,
            "has_called_update_summary": self.has_called_update_summary,
            "work_session_tool_calls": self.work_session_tool_calls.copy(),
            "all_tool_calls": self.all_tool_calls.copy(),
            "pending_notifications": len(self.pending_notifications),
            "input_tokens": self.backend.get_token_usage().input_tokens,
            "output_tokens": self.backend.get_token_usage().output_tokens,
            "estimated_cost": self.backend.get_token_usage().estimated_cost
        }
    
    async def _save_summary_to_file(self, summary: WorkingSummary) -> str:
        """Save summary to file and return the file path."""
        import json
        import datetime
        
        # Create filename with timestamp and version
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent_id}_summary_v{summary.version}_{timestamp}.json"
        filepath = self.summaries_dir / filename
        
        # Save summary data
        summary_data = {
            "agent_id": self.agent_id,
            "content": summary.content,
            "version": summary.version,
            "timestamp": summary.timestamp,
            "created_at": datetime.datetime.now().isoformat(),
            "session_id": getattr(self.orchestrator, 'session_id', None) if self.orchestrator else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        return str(filepath)
    
    def _get_summary_file_link(self, version: int) -> str:
        """Get the most recent summary file for a given version."""
        pattern = f"{self.agent_id}_summary_v{version}_*.json"
        matching_files = list(self.summaries_dir.glob(pattern))
        
        if matching_files:
            # Return the most recent file
            latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)
        
        return f"summaries/{self.agent_id}_summary_v{version}_*.json"