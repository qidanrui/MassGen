"""
Summary file management utilities for MASS agents.
Provides tools to manage, browse, and clean up summary files.
"""

import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Dict


@dataclass
class SummaryFile:
    """Represents a summary file with metadata."""

    agent_id: str
    version: int
    filepath: str
    timestamp: float
    created_at: str
    session_id: Optional[str] = None
    content: Optional[str] = None

    @classmethod
    def from_file(cls, filepath: Path) -> Optional["SummaryFile"]:
        """Create SummaryFile from a file path."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            return cls(
                agent_id=data["agent_id"],
                version=data["version"],
                filepath=str(filepath),
                timestamp=data["timestamp"],
                created_at=data["created_at"],
                session_id=data.get("session_id"),
                content=data.get("content"),
            )
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return None


class SummaryManager:
    """Manager for working summary files."""

    def __init__(self, summaries_dir: str = "summaries"):
        self.summaries_dir = Path(summaries_dir)
        self.summaries_dir.mkdir(exist_ok=True)

    def list_summaries(self, agent_id: Optional[str] = None) -> List[SummaryFile]:
        """List all summary files, optionally filtered by agent_id."""
        summaries = []

        pattern = f"{agent_id}_summary_*.json" if agent_id else "*_summary_*.json"

        for filepath in self.summaries_dir.glob(pattern):
            summary = SummaryFile.from_file(filepath)
            if summary:
                summaries.append(summary)

        # Sort by timestamp (most recent first)
        summaries.sort(key=lambda s: s.timestamp, reverse=True)
        return summaries

    def get_latest_summary(self, agent_id: str) -> Optional[SummaryFile]:
        """Get the latest summary for an agent."""
        summaries = self.list_summaries(agent_id)
        return summaries[0] if summaries else None

    def get_summary_by_version(self, agent_id: str, version: int) -> Optional[SummaryFile]:
        """Get a specific version of an agent's summary."""
        summaries = self.list_summaries(agent_id)
        for summary in summaries:
            if summary.version == version:
                return summary
        return None

    def cleanup_old_summaries(self, keep_latest: int = 5) -> Dict[str, int]:
        """Clean up old summary files, keeping only the latest N per agent."""
        cleanup_count = {}

        # Group summaries by agent_id
        agent_summaries = {}
        for summary in self.list_summaries():
            if summary.agent_id not in agent_summaries:
                agent_summaries[summary.agent_id] = []
            agent_summaries[summary.agent_id].append(summary)

        # Clean up old summaries for each agent
        for agent_id, summaries in agent_summaries.items():
            # Sort by version (highest first)
            summaries.sort(key=lambda s: s.version, reverse=True)

            # Keep only the latest N
            to_remove = summaries[keep_latest:]
            removed_count = 0

            for summary in to_remove:
                try:
                    os.remove(summary.filepath)
                    removed_count += 1
                except OSError:
                    pass

            cleanup_count[agent_id] = removed_count

        return cleanup_count

    def export_session_summaries(
        self, session_id: str, output_file: Optional[str] = None
    ) -> str:
        """Export all summaries for a session to a markdown file."""
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"session_{session_id}_{timestamp}.md"

        output_path = self.summaries_dir / output_file

        # Get all summaries for this session
        session_summaries = [
            s for s in self.list_summaries() if s.session_id == session_id
        ]

        # Group by agent
        agent_summaries = {}
        for summary in session_summaries:
            if summary.agent_id not in agent_summaries:
                agent_summaries[summary.agent_id] = []
            agent_summaries[summary.agent_id].append(summary)

        # Sort each agent's summaries by version
        for agent_id in agent_summaries:
            agent_summaries[agent_id].sort(key=lambda s: s.version)

        # Write markdown report
        with open(output_path, "w") as f:
            f.write(f"# Session Summary Report - {session_id}\n\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")

            f.write("## Agent Summaries\n\n")
            for agent_id, summaries in agent_summaries.items():
                f.write(f"### {agent_id}\n\n")

                for summary in summaries:
                    f.write(f"**Version {summary.version}** - {summary.created_at}\n")
                    f.write(f"Source: [{summary.filepath}]({summary.filepath})\n\n")
                    if summary.content:
                        f.write(f"```\n{summary.content}\n```\n\n")
                    f.write("---\n\n")

        return str(output_path)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get statistics about summary files."""
        summaries = self.list_summaries()

        if not summaries:
            return {"total_files": 0, "agents": 0, "sessions": 0}

        agent_counts = {}
        session_ids = set()

        for summary in summaries:
            # Count by agent
            if summary.agent_id not in agent_counts:
                agent_counts[summary.agent_id] = 0
            agent_counts[summary.agent_id] += 1

            # Track sessions
            if summary.session_id:
                session_ids.add(summary.session_id)

        return {
            "total_files": len(summaries),
            "agents": len(agent_counts),
            "sessions": len(session_ids),
            "agent_counts": agent_counts,
            "oldest_summary": min(summaries, key=lambda s: s.timestamp).created_at,
            "newest_summary": max(summaries, key=lambda s: s.timestamp).created_at,
        }


# Global manager instance
_summary_manager = SummaryManager()


def get_summary_manager() -> SummaryManager:
    """Get the global summary manager instance."""
    return _summary_manager
