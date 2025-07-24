#!/usr/bin/env python3
"""
MASS Example: Two Agent Coordination with Real API

This example demonstrates real multi-agent coordination with the OpenAI API.
Two agents with different perspectives work together on a simple question,
showing the complete MASS workflow with actual API calls.

Key Learning Points:
- Two agents provide different answers (Case 1)
- Agents evaluate and vote for the best answer (Case 2)
- Real coordination and consensus building
- Cost-effective with simple question and gpt-4o-mini

Note: Requires OPENAI_API_KEY in environment or .env file.
"""

import asyncio
import sys
import os
from pathlib import Path

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, try manual .env loading
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')

# Add the parent directory to the path to import mass modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from mass.backends.openai_backend import OpenAIBackend
from mass.chat_agent import create_simple_agent
from mass.orchestrator import create_orchestrator
from mass.frontend.coordination_ui import coordinate_with_terminal_ui


async def two_agent_real_api_example():
    """Demonstrate two agent coordination with real OpenAI API."""
    
    print("🎯 MASS Example: Two Agent Coordination with Real API")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set: export OPENAI_API_KEY='your-key-here'")
        return False
    
    print("✅ OpenAI API key found")
    
    print("\n🔍 What This Example Shows:")
    print("- Two agents with different perspectives")
    print("- Real multi-agent coordination and voting")
    print("- Complete MASS workflow: answers → evaluation → consensus")
    print("- Cost-effective using gpt-4o-mini model")
    
    # Step 1: Create real backends
    backend1 = OpenAIBackend()
    backend2 = OpenAIBackend()
    print("\n✅ Real OpenAI backends created (using gpt-4o-mini)")
    
    # Step 2: Create two agents with different roles
    agent1 = create_simple_agent(
        backend=backend1, 
        agent_id="mathematician",
        system_message="You are a mathematician who focuses on precise mathematical explanations."
    )
    agent2 = create_simple_agent(
        backend=backend2, 
        agent_id="teacher", 
        system_message="You are a teacher who focuses on simple, educational explanations."
    )
    
    agents = [("mathematician", agent1), ("teacher", agent2)]
    print(f"✅ Created {len(agents)} agents with different perspectives")
    print("  • mathematician: Focuses on mathematical precision")
    print("  • teacher: Focuses on educational clarity")
    
    # Step 3: Create orchestrator
    orchestrator = create_orchestrator(agents, orchestrator_id="coordination_hub")
    print("✅ MASS orchestrator ready for coordination")
    
    # Step 4: Simple question that can have different approaches
    question = "What is 8 ÷ 2?"
    print(f"\n💬 Question: {question}")
    print("🔄 Starting two-agent coordination...")
    print("👀 Watch for: Different answers → Evaluation → Voting → Consensus\n")
    
    try:
        # Use the new frontend coordination UI
        result = await coordinate_with_terminal_ui(orchestrator, question)
        
        print(f"\n✅ Success! Two agent coordination completed successfully!")
        print("🎉 Got coordinated response from agents!")
        
        # Get orchestrator status for analysis
        orchestrator_status = orchestrator.get_status()
        vote_results = orchestrator_status.get('vote_results', {})
        
        # Count tool usage from vote results
        tool_usage = {"new_answers": 0, "votes": 0}
        if vote_results.get('agents_with_answers', 0) > 0:
            tool_usage["new_answers"] = vote_results.get('agents_with_answers', 0)
        if vote_results.get('total_votes', 0) > 0:
            tool_usage["votes"] = vote_results.get('total_votes', 0)
        
        print(f"\n📊 MASS Workflow Analysis:")
        print(f"1. 🔧 New answers provided: {tool_usage['new_answers']} (Case 1 behavior)")
        print(f"2. 🗳️ Votes cast: {tool_usage['votes']} (Case 2 behavior)")
        print("3. 🎯 Orchestrator coordinated agents and selected best answer")
        print("4. 📡 All communication streamed in real-time from OpenAI API")
        
        if tool_usage["new_answers"] > 0 and tool_usage["votes"] > 0:
            print("\n💡 Perfect! Complete MASS coordination cycle:")
            print("   Agents provided different perspectives → Evaluated options → Reached consensus")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Two agent coordination failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Run the two agent real API example
    result = asyncio.run(two_agent_real_api_example())
    if result:
        print("\n🚀 Two agent coordination works perfectly!")
        print("💡 What you just saw:")
        print("  • Real multi-agent coordination with OpenAI API")
        print("  • Different agent perspectives creating diverse answers")
        print("  • Democratic voting process to select best response")
        print("  • Orchestrator managing the entire workflow seamlessly")
        print("\n💡 Next steps:")
        print("  • Try renewable_energy_coordination.py for 3+ agents")
        print("  • Try banking_security_coordination.py for specialized teams")
    else:
        print("\n⚠️ Please check the error above and try again.")
        print("💡 Make sure OPENAI_API_KEY is set in your environment or .env file.")
    
    print(f"\n📝 Two agent real API example completed!")