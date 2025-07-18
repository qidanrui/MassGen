#!/usr/bin/env python3
"""
Simple demo to generate API traces for all scenarios.
Run this, then check traces/ directory for the generated trace files.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import mass
sys.path.insert(0, str(Path(__file__).parent.parent))

from mass.agent import MassAgent, AgentStatus, NotificationUpdate
from mass.agent_config import AgentConfig
from mass.utils.api_tracer import get_tracer


async def generate_all_traces():
    """Generate API traces for all scenarios."""
    print("🔍 GENERATING API TRACES")
    print("=" * 50)
    
    tracer = get_tracer()
    tracer.clear_traces()
    
    # Create agent with cost-effective config
    config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",  # 97% cheaper than gpt-4o
            "temperature": 0.7,
            "max_tokens": 1000
        },
        provider_tools=["web_search", "code_interpreter"]
    )
    
    agent = MassAgent("demo_agent", config)
    
    print(f"Agent: {agent.agent_id}")
    print(f"Provider: {agent.backend.get_provider_name()}")
    print(f"Provider tools: {agent._get_provider_tools()}")
    
    # 1. Generate initial trace
    print("\n1. 📤 Generating INITIAL request trace...")
    try:
        async for chunk in agent.work_on_task("Analyze the benefits of multi-agent AI collaboration"):
            break  # Just need the trace
    except:
        pass
    
    # 2. Generate restart trace
    print("2. 🔄 Generating RESTART request trace...")
    agent.status = AgentStatus.STOPPED
    agent.has_called_update_summary = True
    agent.has_voted = True
    
    # Add notification
    notification = NotificationUpdate(
        agent_id="other_agent",
        summary="Important collaborative insight from another agent",
        timestamp=1234567890,
        version=2
    )
    await agent.queue_notification(notification)
    
    # Use the new multi-reason restart logic
    should_restart, reasons = agent.should_restart()
    if should_restart and reasons:
        # Build multi-reason instruction like the orchestrator does (concise, no overlap with system)
        instruction_parts = ["You were stopped and need to address the following:"]
        
        if "notifications" in reasons:
            instruction_parts.append(f"- Process {len(agent.pending_notifications)} unprocessed notification(s)")
        if "missing_vote" in reasons:
            instruction_parts.append("- Cast your vote for a representative")
        if "missing_tools" in reasons:
            instruction_parts.append("- Use required tools (see system rules)")
        
        instruction_parts.append("Continue working to complete these requirements.")
        multi_reason_instruction = "\n".join(instruction_parts)
        
        try:
            async for chunk in agent.work_on_task("Original AI collaboration task", multi_reason_instruction):
                break  # Just need the trace  
        except:
            pass
    else:
        # Fallback for single reason
        try:
            async for chunk in agent.work_on_task("Original AI collaboration task", "notifications"):
                break
        except:
            pass
    
    # 3. Generate final answer trace
    print("3. 🏆 Generating FINAL answer trace...")
    agent.status = AgentStatus.PRESENTING
    
    try:
        async for chunk in agent.present_final_answer("What are the benefits of AI collaboration?"):
            break
    except:
        pass
    
    # Show results
    traces_generated = len(tracer.traces)
    print(f"\n✅ Generated {traces_generated} API traces")
    
    if traces_generated > 0:
        call_types = [trace.call_type for trace in tracer.traces]
        print(f"Call types: {call_types}")
        
        traces_dir = Path("traces")
        if traces_dir.exists():
            trace_files = list(traces_dir.glob("*.json"))
            print(f"Trace files: {len(trace_files)}")
            for f in trace_files[-3:]:  # Show last 3
                print(f"  - {f.name}")
    
    return traces_generated > 0


if __name__ == "__main__":
    print("🔍 API Trace Generator")
    print("Generates traces for all API call scenarios.\n")
    
    success = asyncio.run(generate_all_traces())
    
    if success:
        print("\n" + "=" * 50)
        print("✅ TRACES GENERATED SUCCESSFULLY!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. ls traces/                            # List trace files")
        print("2. cat traces/trace_*.json               # View raw JSON")
        print("\nSee API_INPUTS.md for complete documentation.")
    else:
        print("\n❌ No traces generated. Check for errors above.")
    
    sys.exit(0 if success else 1)