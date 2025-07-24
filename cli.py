#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASS (Multi-Agent Scaling System) - Command Line Interface

This provides a clean command-line interface for the MASS system.

Usage examples:
    # Use YAML configuration file
    python cli.py "What is 2+2?" --config examples/production.yaml
    
    # Use model names directly (single or multiple agents)
    python cli.py "What is 2+2?" --models gpt-4.1 gemini-2.5-flash
    python cli.py "What is 2+2?" --models gpt-4.1  # Single agent mode
    
    # Interactive mode (no question provided)
    python cli.py --models gpt-4.1 grok-4
"""

import argparse
import sys
import os
from pathlib import Path

# Add mass package to path
sys.path.insert(0, str(Path(__file__).parent))

from mass import (
    run_mass_with_config, load_config_from_yaml, create_config_from_models, 
    ConfigurationError
)


def run_interactive_mode(config):
    """Run MASS in interactive mode, asking for questions repeatedly."""
    print("\n🤖 MASS Interactive Mode")
    print("="*60)
    
    # Display current configuration
    print("📋 Current Configuration:")
    print("-" * 30)
    
    # Show models/agents
    if hasattr(config, 'agents') and config.agents:
        print(f"🤖 Agents ({len(config.agents)}):")
        for i, agent in enumerate(config.agents, 1):
            model_name = getattr(agent.model_config, 'model', 'Unknown') if hasattr(agent, 'model_config') else 'Unknown'
            agent_type = getattr(agent, 'agent_type', 'Unknown')
            tools = getattr(agent.model_config, 'tools', []) if hasattr(agent, 'model_config') else []
            tools_str = ', '.join(tools) if tools else 'None'
            print(f"   {i}. {model_name} ({agent_type})")
            print(f"      Tools: {tools_str}")
    else:
        print("🤖 Single Agent Mode")
    
    # Show orchestrator settings
    if hasattr(config, 'orchestrator'):
        orch = config.orchestrator
        print(f"⚙️  Orchestrator:")
        print(f"   • Duration: {getattr(orch, 'max_duration', 'Default')}s")
        print(f"   • Consensus: {getattr(orch, 'consensus_threshold', 'Default')}")
        print(f"   • Max Rounds: {getattr(orch, 'max_debate_rounds', 'Default')}")
    
    # Show model parameters (from first agent as representative)
    if hasattr(config, 'agents') and config.agents and hasattr(config.agents[0], 'model_config'):
        model_config = config.agents[0].model_config
        print(f"🔧 Model Config:")
        temp = getattr(model_config, 'temperature', 'Default')
        timeout = getattr(model_config, 'inference_timeout', 'Default')
        max_rounds = getattr(model_config, 'max_rounds', 'Default')
        print(f"   • Temperature: {temp}")
        print(f"   • Timeout: {timeout}s")
        print(f"   • Max Rounds: {max_rounds}")
    
    # Show display settings
    if hasattr(config, 'streaming_display'):
        display = config.streaming_display
        display_status = "✅ Enabled" if getattr(display, 'display_enabled', True) else "❌ Disabled"
        logs_status = "✅ Enabled" if getattr(display, 'save_logs', True) else "❌ Disabled"
        print(f"📺 Display: {display_status}")
        print(f"📁 Logs: {logs_status}")
    
    print("-" * 30)
    print("💬 Type your questions below. Type 'quit', 'exit', or press Ctrl+C to stop.")
    print("="*60)
    
    chat_history = ""
    try:
        while True:
            try:
                question = input("\nUser: ").strip()
                chat_history += f"User: {question}\n"
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    print("Please enter a question or type 'quit' to exit.")
                    continue
                
                print("\n🔄 Processing your question...")
                
                # Run MASS
                result = run_mass_with_config(chat_history, config)
                
                response = result["answer"]
                chat_history += f"Assistant: {response}\n"
                
                # Display results
                print("\n" + "="*60)
                print(f"🎯 FINAL ANSWER (Agent {result['representative_agent_id']}):")
                print("="*60)
                print(response)
                print("\n" + "="*60)
                
                # Show different metadata based on single vs multi-agent mode
                if result.get("single_agent_mode", False):
                    print("🤖 Single Agent Mode")
                    print(f"✅ Model: {result.get('model_used', 'Unknown')}")
                    print(f"⏱️  Duration: {result['session_duration']:.1f}s")
                    if result.get("citations"):
                        print(f"📚 Citations: {len(result['citations'])}")
                    if result.get("code"):
                        print(f"💻 Code blocks: {len(result['code'])}")
                else:
                    print(f"✅ Consensus: {result['consensus_reached']}")
                    print(f"⏱️  Duration: {result['session_duration']:.1f}s")
                    print(f"🗳️  Votes: {result['summary']['final_vote_distribution']}")
                    print(f"🤖 Agents: {len(config.agents)}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                print("Please try again or type 'quit' to exit.")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


def main():
    """Clean CLI interface for MASS."""
    parser = argparse.ArgumentParser(
        description="MASS (Multi-Agent Scaling System) - Clean CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML configuration
  python cli.py "What is the capital of France?" --config examples/production.yaml
  
  # Use model names directly (single or multiple agents)
  python cli.py "What is 2+2?" --models gpt-4.1 gemini-2.5-flash
  python cli.py "What is 2+2?" --models gpt-4.1  # Single agent mode
  
  # Interactive mode (no question provided)
  python cli.py --models gpt-4.1 grok-4
  
  # Override parameters
  python cli.py "Question" --models gpt-4.1 gemini-2.5-flash --max-duration 1200 --consensus 0.8
        """
    )
    
    # Task input (now optional for interactive mode)
    parser.add_argument("question", nargs='?', help="Question to solve (optional - if not provided, enters interactive mode)")
    
    # Configuration options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", type=str,
                             help="Path to YAML configuration file")
    config_group.add_argument("--models", nargs="+",
                             help="Model names (e.g., gpt-4.1 gemini-2.5-flash)")
    
    # Configuration overrides
    parser.add_argument("--max-duration", type=int, default=None,
                       help="Max duration in seconds")
    parser.add_argument("--consensus", type=float, default=None,
                       help="Consensus threshold (0.0-1.0)")
    parser.add_argument("--max-debates", type=int, default=None,
                       help="Maximum debate rounds")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable streaming display")
    parser.add_argument("--no-logs", action="store_true",
                       help="Disable file logging")
    
    args = parser.parse_args()
    
    # DEBUGGING
    for file in ["function_calls.txt", "errors.txt", "openai_streaming.txt",
                 "gemini_streaming.txt", "grok_streaming.txt"]:
        if os.path.exists(file):
            os.remove(file)
            
    # Load configuration
    try:
        if args.config:
            config = load_config_from_yaml(args.config)
        else:  # args.models
            config = create_config_from_models(args.models)
        
        # Apply command-line overrides
        if args.max_duration is not None:
            config.orchestrator.max_duration = args.max_duration
        if args.consensus is not None:
            config.orchestrator.consensus_threshold = args.consensus
        if args.max_debates is not None:
            config.orchestrator.max_debate_rounds = args.max_debates
        if args.no_display:
            config.streaming_display.display_enabled = False
        if args.no_logs:
            config.streaming_display.save_logs = False
        
        # Validate final configuration
        config.validate()
        
        # Check if question was provided
        if args.question:
            # Single question mode
            result = run_mass_with_config(args.question, config)
            
            # Display results
            print("\n" + "="*60)
            print(f"🎯 FINAL ANSWER (Agent {result['representative_agent_id']}):")
            print("="*60)
            print(result["answer"])
            print("\n" + "="*60)
            
            # Show different metadata based on single vs multi-agent mode
            if result.get("single_agent_mode", False):
                print("🤖 Single Agent Mode")
                print(f"✅ Model: {result.get('model_used', 'Unknown')}")
                print(f"⏱️  Duration: {result['session_duration']:.1f}s")
                if result.get("citations"):
                    print(f"📚 Citations: {len(result['citations'])}")
                if result.get("code"):
                    print(f"💻 Code blocks: {len(result['code'])}")
            else:
                print(f"✅ Consensus: {result['consensus_reached']}")
                print(f"⏱️  Duration: {result['session_duration']:.1f}s")
                print(f"🗳️  Votes: {result['summary']['final_vote_distribution']}")
                print(f"🤖 Agents: {len(config.agents)}")
        else:
            # Interactive mode
            run_interactive_mode(config)
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 