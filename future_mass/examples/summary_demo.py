#!/usr/bin/env python3
"""
Demo of improved working summary functionality with file persistence and clickable links.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import mass
sys.path.insert(0, str(Path(__file__).parent.parent))

from mass.agent import MassAgent
from mass.agent_config import AgentConfig
from mass.summary_manager import get_summary_manager


async def demo_summary_improvements():
    """Demo the improved summary functionality."""
    print("📄 WORKING SUMMARY IMPROVEMENTS DEMO")
    print("=" * 50)

    # Create agent with mock backend
    config = AgentConfig(backend_params={"model": "mock-test"})
    agent = MassAgent("demo_agent", config)

    print(f"Agent: {agent.agent_id}")
    print(f"Summaries directory: {agent.summaries_dir}")
    print()

    # Test 1: Update summary and see file link
    print("1. 📝 Testing summary update with file persistence...")
    result = await agent.update_summary("Initial analysis of the task at hand")
    print(f"Result: {result}")
    print()

    # Test 2: Update summary again to see versioning
    print("2. 🔄 Testing summary versioning...")
    result = await agent.update_summary("Updated analysis with new insights")
    print(f"Result: {result}")
    print()

    # Test 3: Show summary files created
    print("3. 📂 Listing created summary files...")
    summary_manager = get_summary_manager()
    summaries = summary_manager.list_summaries(agent.agent_id)

    for summary in summaries:
        print(f"  - Version {summary.version}: {summary.filepath}")
        print(f"    Created: {summary.created_at}")
        print(f"    Content: {summary.content[:50]}...")
        print()

    # Test 4: Show summary statistics
    print("4. 📊 Summary statistics...")
    stats = summary_manager.get_summary_stats()
    print(f"  Total files: {stats['total_files']}")
    print(f"  Agents: {stats['agents']}")
    print(f"  Sessions: {stats['sessions']}")
    print()

    # Test 5: Simulate streamed output with file links
    print("5. 🔄 Simulating streamed output with file links...")
    print("Agent working on task...")

    # This would normally come from agent.work_on_task()
    async def simulate_streaming():
        yield "Let me analyze this task..."
        yield "\n[Tool: update_summary] Summary updated to version 3. Saved to summaries/demo_agent_summary_v3_20250717_233045.json"
        yield "\n📄 Click to view: summaries/demo_agent_summary_v3_20250717_233045.json"
        yield "\nContinuing work based on the updated summary..."

    async for chunk in simulate_streaming():
        print(chunk, end="")

    print("\n")
    print("=" * 50)
    print("✅ DEMO COMPLETED!")
    print("=" * 50)
    print("\nKey improvements:")
    print("• Working summaries are now saved to files")
    print("• File links are displayed in streamed output")
    print("• Summary versioning is preserved")
    print("• File management utilities are available")
    print("• Users can click links to view summary files")


if __name__ == "__main__":
    asyncio.run(demo_summary_improvements())
