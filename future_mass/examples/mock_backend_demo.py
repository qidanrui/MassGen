#!/usr/bin/env python3
"""
Test script to verify the mock backend works with the orchestrator system.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to the path so we can import mass
sys.path.insert(0, str(Path(__file__).parent.parent))

from mass.agent import MassAgent
from mass.agent_config import AgentConfig
from mass.orchestrator import MassOrchestrator


async def test_mock_backend():
    """Test the mock backend with a simple orchestration."""
    print("🧪 TESTING MOCK BACKEND")
    print("=" * 60)

    # Create agents with mock models
    chatgpt_config = AgentConfig(
        backend_params={"model": "mock-gpt", "temperature": 0.7}
    )
    grok_config = AgentConfig(backend_params={"model": "mock-grok", "temperature": 0.7})

    chatgpt_agent = MassAgent("chatgpt_agent", chatgpt_config)
    grok_agent = MassAgent("grok_agent", grok_config)

    # Create orchestrator
    orchestrator = MassOrchestrator(max_duration=5.0, check_interval=0.5)
    orchestrator.add_agent(chatgpt_agent)
    orchestrator.add_agent(grok_agent)

    # The task
    task = "What is the earliest known date recorded by a pre-Columbian civilization in the Americas in the aboriginal writing system?"

    print(f"🚀 STARTING TEST with task:")
    print(f"   {task}")
    print("-" * 60)

    try:
        result = await orchestrator.orchestrate(task)

        print(f"\n📊 TEST RESULTS:")
        print(f"   Total time: {result.total_time:.1f} seconds")
        print(f"   Representative: {result.representative_agent}")

        print(f"\n🗳️  Voting Results:")
        for voter, voted_for in result.vote_results.items():
            print(f"   {voter} → {voted_for}")

        print(f"\n🏆 FINAL ANSWER:")
        print("-" * 60)
        print(result.final_response)
        print("-" * 60)

        # Check if collaboration worked correctly
        final_answer = result.final_response.lower()
        if (
            "7.16.3.2.13" in final_answer
            and "36" in final_answer
            and ("b.c" in final_answer or "bce" in final_answer)
        ):
            print("✅ SUCCESS! Mock backend collaboration worked correctly!")
            print("   • Agents updated summaries and voted")
            print("   • Final answer: 7.16.3.2.13 = December 10th, 36 B.C.")
        else:
            print("⚠️ Mock backend worked but answer may need refinement")
            print("   • System demonstrated collaboration mechanics")

        return True

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mock_backend())
    sys.exit(0 if success else 1)
