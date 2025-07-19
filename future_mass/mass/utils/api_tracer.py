"""
API call tracing utility for debugging and analysis.
Provides reusable tracing functionality with file links and structured output.
"""

import datetime
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union, Optional, List, Dict


@dataclass
class SourceLink:
    """Reference to source code location."""

    file_path: str
    line_number: Optional[int] = None
    function_name: Optional[str] = None

    def __str__(self) -> str:
        """Format as clickable link."""
        if self.line_number:
            return f"{self.file_path}:{self.line_number}"
        return self.file_path


@dataclass
class APITrace:
    """Single API call trace record."""

    timestamp: str
    provider: str
    model: str
    call_type: str  # "initial", "tool_response", "restart", "final"
    request_data: Dict[str, Any]
    response_summary: Optional[str] = None
    source_links: List[SourceLink] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.source_links is None:
            self.source_links = []


class APITracer:
    """Reusable API call tracer with file output and source linking."""

    def __init__(self, output_dir: str = "traces", enabled: bool = True):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.traces: List[APITrace] = []

        # Common source file paths for linking
        self.source_files = {
            "agent": "mass/agent.py",
            "openai_backend": "mass/backends/openai_backend.py",
            "grok_backend": "mass/backends/grok_backend.py",
            "mock_backend": "mass/backends/mock_backend.py",
            "agent_config": "mass/agent_config.py",
            "orchestrator": "mass/orchestrator.py",
        }

    def trace_api_call(
        self,
        provider: str,
        model: str,
        call_type: str,
        request_data: Dict[str, Any],
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        response_summary: Optional[str] = None,
    ) -> None:
        """Record an API call trace."""
        if not self.enabled:
            return

        # Create source links based on call context
        source_links = self._get_source_links(provider, call_type)

        trace = APITrace(
            timestamp=datetime.datetime.now().isoformat(),
            provider=provider,
            model=model,
            call_type=call_type,
            request_data=self._sanitize_request_data(request_data),
            response_summary=response_summary,
            source_links=source_links,
            agent_id=agent_id,
            session_id=session_id,
        )

        self.traces.append(trace)
        self._write_trace_file(trace)

    def _get_source_links(self, provider: str, call_type: str) -> List[SourceLink]:
        """Generate relevant source file links based on context."""
        links = []

        # Call type specific files with precise locations
        if call_type == "initial":
            links.append(
                SourceLink(
                    file_path=self.source_files["agent"],
                    line_number=220,  # _build_work_instruction method
                    function_name="_build_work_instruction",
                )
            )
            links.append(
                SourceLink(
                    file_path=self.source_files["agent_config"],
                    line_number=25,  # summary_update_instructions
                    function_name="summary_update_instructions",
                )
            )
            links.append(
                SourceLink(
                    file_path=self.source_files["agent_config"],
                    line_number=69,  # collaboration_instructions
                    function_name="collaboration_instructions",
                )
            )

        elif call_type == "restart":
            links.append(
                SourceLink(
                    file_path=self.source_files["agent"],
                    line_number=225,  # _build_restart_instruction method
                    function_name="_build_restart_instruction",
                )
            )
            links.append(
                SourceLink(
                    file_path=self.source_files["agent_config"],
                    line_number=42,  # notification_handling_instructions
                    function_name="notification_handling_instructions",
                )
            )
            links.append(
                SourceLink(
                    file_path=self.source_files["agent_config"],
                    line_number=79,  # reactivation_instructions
                    function_name="reactivation_instructions",
                )
            )

        elif call_type == "final":
            links.append(
                SourceLink(
                    file_path=self.source_files["agent"],
                    line_number=230,  # _build_final_answer_instruction method
                    function_name="_build_final_answer_instruction",
                )
            )
            links.append(
                SourceLink(
                    file_path=self.source_files["agent_config"],
                    line_number=59,  # final_answer_instructions
                    function_name="final_answer_instructions",
                )
            )

        elif call_type == "tool_response":
            links.append(
                SourceLink(
                    file_path=self.source_files["agent"],
                    line_number=360,  # _add_assistant_message_to_conversation
                    function_name="_add_assistant_message_to_conversation",
                )
            )
            links.append(
                SourceLink(
                    file_path=self.source_files["agent"],
                    line_number=380,  # _add_tool_result_to_conversation
                    function_name="_add_tool_result_to_conversation",
                )
            )

        # Always include main processing logic
        links.append(
            SourceLink(
                file_path=self.source_files["agent"],
                line_number=271,  # process_message method
                function_name="process_message",
            )
        )

        # Tool schema definition
        links.append(
            SourceLink(
                file_path=self.source_files["agent"],
                line_number=470,  # _get_tool_schema method
                function_name="_get_tool_schema",
            )
        )

        # Provider-specific backend files
        if provider.lower() == "openai":
            links.append(
                SourceLink(
                    file_path=self.source_files["openai_backend"],
                    line_number=29,  # stream_with_tools method
                    function_name="stream_with_tools",
                )
            )
        elif provider.lower() == "grok":
            links.append(
                SourceLink(
                    file_path=self.source_files["grok_backend"],
                    line_number=38,  # chat.completions.create
                    function_name="stream_with_tools",
                )
            )

        return links

    def _sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data and format for readability."""
        sanitized = data.copy()

        # Remove API keys
        if "api_key" in sanitized:
            sanitized["api_key"] = "[REDACTED]"

        # Keep full messages - don't truncate for debugging
        # User wants to see complete API inputs

        return sanitized

    def _write_trace_file(self, trace: APITrace) -> None:
        """Write individual trace to file."""
        timestamp_str = trace.timestamp.replace(":", "-").replace(".", "-")
        filename = f"trace_{trace.provider}_{trace.call_type}_{timestamp_str}.json"
        filepath = self.output_dir / filename

        # Convert to dict for JSON serialization
        trace_dict = asdict(trace)

        # Add formatted output for human readability
        trace_dict["formatted_summary"] = self._format_trace_summary(trace)

        with open(filepath, "w") as f:
            json.dump(trace_dict, f, indent=2)

    def _format_trace_summary(self, trace: APITrace) -> str:
        """Create human-readable trace summary."""
        summary = []
        summary.append(f"=== {trace.provider} API Call Trace ===")
        summary.append(f"Time: {trace.timestamp}")
        summary.append(f"Model: {trace.model}")
        summary.append(f"Type: {trace.call_type}")

        if trace.agent_id:
            summary.append(f"Agent: {trace.agent_id}")
        if trace.session_id:
            summary.append(f"Session: {trace.session_id}")

        summary.append("")
        summary.append("Source Files:")
        for link in trace.source_links:
            summary.append(f"  - {link}")

        summary.append("")
        summary.append("Request Data:")
        summary.append(json.dumps(trace.request_data, indent=2))

        if trace.response_summary:
            summary.append("")
            summary.append("Response Summary:")
            summary.append(trace.response_summary)

        return "\n".join(summary)

    def export_session_report(
        self, session_id: str, filename: Optional[str] = None
    ) -> str:
        """Export all traces for a session to a comprehensive report."""
        session_traces = [t for t in self.traces if t.session_id == session_id]

        if not filename:
            filename = f"session_report_{session_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(f"# API Trace Report - Session {session_id}\n\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")

            f.write("## Summary\n\n")
            f.write(f"Total API calls: {len(session_traces)}\n")
            f.write(
                f"Providers used: {', '.join(set(t.provider for t in session_traces))}\n"
            )
            f.write(
                f"Models used: {', '.join(set(t.model for t in session_traces))}\n\n"
            )

            f.write("## Call Sequence\n\n")
            for i, trace in enumerate(session_traces, 1):
                f.write(
                    f"### {i}. {trace.call_type.title()} Call - {trace.provider}\n\n"
                )
                f.write(f"**Time:** {trace.timestamp}\n")
                f.write(f"**Model:** {trace.model}\n")
                if trace.agent_id:
                    f.write(f"**Agent:** {trace.agent_id}\n")

                f.write(f"\n**Source Files:**\n")
                for link in trace.source_links:
                    f.write(f"- [{link.file_path}]({link})")
                    if link.function_name:
                        f.write(f" (`{link.function_name}`)")
                    f.write("\n")

                f.write(
                    f"\n**Request:**\n```json\n{json.dumps(trace.request_data, indent=2)}\n```\n\n"
                )

                if trace.response_summary:
                    f.write(f"**Response:** {trace.response_summary}\n\n")

        return str(filepath)

    def clear_traces(self) -> None:
        """Clear all stored traces."""
        self.traces.clear()


# Global tracer instance
_global_tracer: Optional[APITracer] = None


def get_tracer() -> APITracer:
    """Get or create global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        # Check environment variable for tracing
        enabled = os.getenv("MASS_API_TRACING", "true").lower() == "true"
        output_dir = os.getenv("MASS_TRACE_DIR", "traces")
        _global_tracer = APITracer(output_dir=output_dir, enabled=enabled)
    return _global_tracer


def trace_api_call(
    provider: str, model: str, call_type: str, request_data: Dict[str, Any], **kwargs
) -> None:
    """Convenience function for tracing API calls."""
    get_tracer().trace_api_call(provider, model, call_type, request_data, **kwargs)
