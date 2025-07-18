#!/usr/bin/env python3
"""
Demo showing flexible tool configuration and collision detection.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import mass
sys.path.insert(0, str(Path(__file__).parent.parent))

from mass.agent import MassAgent
from mass.agent_config import AgentConfig


async def demonstrate_flexible_tools():
    """Demonstrate flexible tool configuration and collision detection."""
    print("🔧 FLEXIBLE TOOL CONFIGURATION DEMO")
    print("=" * 60)

    # Define some user tools (including one that conflicts)
    database_tool = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query the company database",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "SQL query"}},
                "required": ["query"],
            },
        },
    }

    # This will conflict with MASS framework tool
    conflicting_tool = {
        "type": "function",
        "function": {
            "name": "update_summary",  # Same name as MASS framework tool!
            "description": "User's version of update_summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Summary content"}
                },
                "required": ["content"],
            },
        },
    }

    custom_tool = {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    }

    print("🧪 TESTING DIFFERENT CONFIGURATIONS:")
    print("=" * 60)

    # Test cases for provider tools
    test_cases = [
        ("All Provider Tools", None),
        ("No Provider Tools", []),
        ("Only Web Search", ["web_search"]),
        ("Only Code Interpreter", ["code_interpreter"]),
        ("Invalid Tools", ["nonexistent_tool", "web_search"]),
        ("Web Search + Code", ["web_search", "code_interpreter"]),
    ]

    for name, provider_tools in test_cases:
        print(f"\n📋 {name} (provider_tools={provider_tools}):")

        config = AgentConfig(
            backend_params={"model": "gpt-4o-mini"},
            provider_tools=provider_tools,
            user_tools=[database_tool, custom_tool],
        )

        agent = MassAgent(f"test_{name.lower().replace(' ', '_')}", config)

        # Show what tools are actually available
        actual_provider_tools = agent._get_provider_tools()
        user_tools = agent._get_user_tools()

        print(f"  Requested: {provider_tools}")
        print(f"  Actual provider tools: {actual_provider_tools}")
        print(f"  User tools: {[t['function']['name'] for t in user_tools]}")

    print("\n" + "=" * 60)
    print("🚨 TESTING COLLISION DETECTION:")
    print("=" * 60)

    # Test collision detection
    print("\n📋 Agent with conflicting user tool:")
    config_with_conflict = AgentConfig(
        backend_params={"model": "gpt-4o-mini"},
        provider_tools=["web_search"],
        user_tools=[
            database_tool,
            conflicting_tool,
            custom_tool,
        ],  # conflicting_tool will be rejected
    )

    agent_with_conflict = MassAgent("conflict_test", config_with_conflict)

    # This should show the warning and filter out the conflicting tool
    user_tools = agent_with_conflict._get_user_tools()
    framework_tools = agent_with_conflict._get_tool_schema()

    print(
        f"  User tools (after filtering): {[t['function']['name'] for t in user_tools]}"
    )
    print(f"  Framework tools: {[t['function']['name'] for t in framework_tools]}")

    print("\n" + "=" * 60)
    print("✅ USAGE EXAMPLES:")
    print("=" * 60)

    print("1. Use all provider tools (default):")
    print(
        """
    config = AgentConfig(
        backend_params={"model": "gpt-4o-mini"},
        provider_tools=None  # None = all supported tools
    )
    """
    )

    print("2. Disable all provider tools:")
    print(
        """
    config = AgentConfig(
        backend_params={"model": "gpt-4o-mini"},
        provider_tools=[]  # Empty list = no provider tools
    )
    """
    )

    print("3. Use specific provider tools:")
    print(
        """
    config = AgentConfig(
        backend_params={"model": "gpt-4o-mini"},
        provider_tools=["web_search"]  # Only web search
    )
    """
    )

    print("4. Add user tools (with collision protection):")
    print(
        """
    config = AgentConfig(
        backend_params={"model": "gpt-4o-mini"},
        user_tools=[my_custom_tool]  # Framework prevents name conflicts
    )
    """
    )

    print("\n🎯 TOOL HIERARCHY:")
    print("1. 🔧 MASS Framework Tools (highest priority, reserved names)")
    print("2. 🌐 Provider Tools (configurable: None/[]/specific list)")
    print("3. 🛠️  User Tools (collision-protected)")


if __name__ == "__main__":
    asyncio.run(demonstrate_flexible_tools())
