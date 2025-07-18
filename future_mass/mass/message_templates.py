"""
Centralized message templates for MASS agents.
All configurable message strings with automatic source tracing.
"""

import inspect
from typing import Dict, Any, Optional
from .utils.api_tracer import get_tracer


class MessageTemplates:
    """Centralized message templates with automatic tracing."""
    
    def __init__(self, **template_overrides):
        # Store template usage for tracing
        self._usage_traces = []
        
        # Store template overrides
        self._template_overrides = template_overrides
    
    # =============================================================================
    # SYSTEM MESSAGE TEMPLATES
    # =============================================================================
    
    def system_agent_intro(self, agent_id: str, mode: str = "collaboration") -> str:
        """Agent introduction for system message."""
        self._trace_template_usage("system_agent_intro", locals())
        
        # Check for override
        if "system_agent_intro" in self._template_overrides:
            override = self._template_overrides["system_agent_intro"]
            if callable(override):
                return override(agent_id, mode)
            else:
                return override.format(agent_id=agent_id, mode=mode)
        
        if mode == "collaboration":
            return f"You are {agent_id} in a multi-agent collaboration session."
        elif mode == "presenting":
            return f"You are {agent_id} presenting the final answer as the selected representative."
        else:
            return f"You are {agent_id}."
    
    def system_tool_descriptions(self, mode: str = "collaboration") -> str:
        """Tool descriptions for system message."""
        self._trace_template_usage("system_tool_descriptions", locals())
        
        if mode == "collaboration":
            return """Available tools:
- update_summary(new_content): Update your working summary when you have new insights
- check_updates(): See what other agents have discovered  
- vote(agent_id): Vote for the best representative (can be called multiple times to change vote)
- no_update(): Signal that no further updates are needed (requires update_summary AND vote first)
- get_session_info(): Check session status, time elapsed, and agent progress"""
        elif mode == "final":
            return """Available tools (final answer phase):
- get_session_info(): Check session status and final state"""
        else:
            return "Available tools: See function definitions below."
    
    def system_tool_rules(self) -> str:
        """Tool usage rules for system message."""
        self._trace_template_usage("system_tool_rules", locals())
        
        return """Tool usage rules:
- Must call update_summary() at least once before using vote() or no_update()
- Must call vote() at least once before using no_update()
- vote() can be called multiple times to change your vote
- Must use one of: update_summary(), vote(), or no_update() in each work session
- If there are pending notifications, must process them with check_updates() before vote() or no_update()"""
    
    def system_final_answer_guidance(self) -> str:
        """Final answer guidance for system message."""
        self._trace_template_usage("system_final_answer_guidance", locals())
        
        return """You should present a comprehensive final answer based on your working summary.
Do NOT call collaboration tools (update_summary, vote, etc.) in this phase."""
    
    # =============================================================================
    # USER MESSAGE TEMPLATES - INITIAL WORK
    # =============================================================================
    
    def initial_task_prefix(self, task: str) -> str:
        """Task prefix for initial work instruction."""
        self._trace_template_usage("initial_task_prefix", locals())
        return f"Task: {task}"
    
    def initial_tool_instructions(self) -> str:
        """Tool usage instructions for initial work."""
        self._trace_template_usage("initial_tool_instructions", locals())
        
        # Check for override
        if "initial_tool_instructions" in self._template_overrides:
            override = self._template_overrides["initial_tool_instructions"]
            if callable(override):
                return override()
            else:
                return str(override)
        
        return """Work systematically on the task:
1. Analyze the task and develop insights
2. Call update_summary() to share your progress  
3. Use check_updates() to see what others have discovered
4. Call vote() when ready to select a representative
5. Call no_update() when you're completely done

Follow the tool usage rules in the system message."""
    
    def initial_collaboration_guidance(self) -> str:
        """Collaboration guidance for initial work."""
        self._trace_template_usage("initial_collaboration_guidance", locals())
        
        # Check for override
        if "initial_collaboration_guidance" in self._template_overrides:
            override = self._template_overrides["initial_collaboration_guidance"]
            if callable(override):
                return override()
            else:
                return str(override)
        
        return """You're working with other intelligent agents on this task. 
Collaborate effectively by:
- Building on others' insights rather than just repeating them
- Offering unique perspectives and approaches
- Being open to changing your approach based on others' discoveries
- Contributing meaningfully to the collective intelligence
Remember: the goal is to achieve the best possible answer through collaboration."""
    
    def initial_work_prompt(self) -> str:
        """Final prompt for initial work."""
        self._trace_template_usage("initial_work_prompt", locals())
        return "Begin working on this task. Use update_summary() when you develop insights."
    
    # =============================================================================
    # USER MESSAGE TEMPLATES - RESTART
    # =============================================================================
    
    def restart_task_reference(self, task: str) -> str:
        """Task reference for restart instruction."""
        self._trace_template_usage("restart_task_reference", locals())
        return f"Original task: {task}"
    
    def restart_instruction_content(self, instruction: str) -> str:
        """Restart instruction content for user message."""
        self._trace_template_usage("restart_instruction_content", locals())
        return instruction
    
    def restart_notification_guidance(self) -> str:
        """Notification handling guidance for restart."""
        self._trace_template_usage("restart_notification_guidance", locals())
        
        return """Review your notifications and incorporate relevant insights.
Build on what other agents have discovered."""
    
    def restart_continuation_prompt(self) -> str:
        """Continuation prompt for restart."""
        self._trace_template_usage("restart_continuation_prompt", locals())
        return "Continue working."
    
    # =============================================================================
    # USER MESSAGE TEMPLATES - FINAL ANSWER
    # =============================================================================
    
    def final_selection_notice(self) -> str:
        """Representative selection notice."""
        self._trace_template_usage("final_selection_notice", locals())
        return "You were selected as the representative to present the final answer."
    
    def final_task_reference(self, task: str) -> str:
        """Task reference for final answer."""
        self._trace_template_usage("final_task_reference", locals())
        return f"Original task: {task}"
    
    def final_answer_requirements(self) -> str:
        """Final answer requirements."""
        self._trace_template_usage("final_answer_requirements", locals())
        
        return """Provide a comprehensive, well-structured response that:
- Directly addresses the original task
- Synthesizes the best insights from all agents' work
- Is clear, actionable, and complete
- Demonstrates the collaborative intelligence of the multi-agent system
Present your response as the definitive answer to the task."""
    
    def final_presentation_prompt(self) -> str:
        """Final presentation prompt."""
        self._trace_template_usage("final_presentation_prompt", locals())
        return "Present your comprehensive final answer now."
    
    # =============================================================================
    # TEMPLATE COMPOSITION METHODS
    # =============================================================================
    
    def build_initial_work_instruction(self, task: str) -> str:
        """Build complete initial work instruction."""
        self._trace_template_usage("build_initial_work_instruction", {"task": task})
        
        parts = [
            self.initial_task_prefix(task),
            "",
            self.initial_tool_instructions(),
            "",
            self.initial_collaboration_guidance(),
            "",
            self.initial_work_prompt()
        ]
        return "\n".join(parts)
    
    def build_restart_instruction(self, task: str, instruction: str, has_notifications: bool = False) -> str:
        """Build complete restart instruction."""
        self._trace_template_usage("build_restart_instruction", {"task": task, "instruction": instruction, "has_notifications": has_notifications})
        
        parts = [
            self.restart_task_reference(task),
            "",
            self.restart_instruction_content(instruction),
            ""
        ]
        
        # Only add notification guidance if there are actual notifications
        if has_notifications:
            parts.extend([
                self.restart_notification_guidance(),
                ""
            ])
        
        parts.append(self.restart_continuation_prompt())
        return "\n".join(parts)
    
    def build_final_answer_instruction(self, task: str) -> str:
        """Build complete final answer instruction."""
        self._trace_template_usage("build_final_answer_instruction", {"task": task})
        
        parts = [
            self.final_selection_notice(),
            "",
            self.final_task_reference(task),
            "",
            self.final_answer_requirements(),
            "",
            self.final_presentation_prompt()
        ]
        return "\n".join(parts)
    
    # =============================================================================
    # TRACING INFRASTRUCTURE
    # =============================================================================
    
    def _trace_template_usage(self, template_name: str, context: Dict[str, Any]) -> str:
        """Trace template usage for debugging."""
        # Get caller information
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name
        
        # Create trace record
        trace_record = {
            "template_name": template_name,
            "file": filename,
            "line": line_number,
            "function": function_name,
            "context": {k: v for k, v in context.items() if k != "self"},
            "caller_locals": dict(frame.f_locals)
        }
        
        self._usage_traces.append(trace_record)
        
        # Add to global tracer if available
        tracer = get_tracer()
        if tracer.enabled and tracer.traces:
            # Add source link for this template
            tracer.traces[-1].source_links.append({
                "file_path": f"mass/message_templates.py",
                "line_number": line_number,
                "function_name": template_name,
                "template_type": "message_template"
            })
        
        return template_name  # Return identifier for debugging
    
    def get_usage_traces(self) -> list:
        """Get all template usage traces."""
        return self._usage_traces
    
    def clear_traces(self) -> None:
        """Clear template usage traces."""
        self._usage_traces.clear()


# Global template instance
_templates = MessageTemplates()


def get_templates() -> MessageTemplates:
    """Get global message templates instance."""
    return _templates


# Convenience functions for common templates
def build_initial_work_instruction(task: str) -> str:
    """Build initial work instruction with tracing."""
    return get_templates().build_initial_work_instruction(task)


def build_restart_instruction(task: str, instruction: str, has_notifications: bool = False) -> str:
    """Build restart instruction with tracing."""
    return get_templates().build_restart_instruction(task, instruction, has_notifications)


def build_final_answer_instruction(task: str) -> str:
    """Build final answer instruction with tracing."""
    return get_templates().build_final_answer_instruction(task)