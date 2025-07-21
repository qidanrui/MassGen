#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASS (Multi-Agent Scaling System) - Command Line Interface

This provides a clean command-line interface for the MASS system.

Usage examples:
    # Use YAML configuration file
    python cli.py "What is 2+2?" --config examples/production.yaml
    
    # Use model names directly (single or multiple agents)
    python cli.py "What is 2+2?" --models gpt-4o gemini-2.5-flash
    python cli.py "What is 2+2?" --models gpt-4o  # Single agent mode
"""

import argparse
import sys
from pathlib import Path

# Add mass package to path
sys.path.insert(0, str(Path(__file__).parent))

from mass import (
    run_mass_with_config, load_config_from_yaml, create_config_from_models, 
    ConfigurationError
)


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
  python cli.py "What is 2+2?" --models gpt-4o gemini-2.5-flash
  python cli.py "What is 2+2?" --models gpt-4o  # Single agent mode
  
  # Override parameters
  python cli.py "Question" --models gpt-4o gemini-2.5-flash --max-duration 1200 --consensus 0.8
        """
    )
    
    # Task input (positional argument must come before nargs='+' to avoid conflicts)
    parser.add_argument("question", help="Question to solve")
    
    # Configuration options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", type=str,
                             help="Path to YAML configuration file")
    config_group.add_argument("--models", nargs="+",
                             help="Model names (e.g., gpt-4o gemini-2.5-flash)")
    
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
        
        # Run MASS
        result = run_mass_with_config(args.question, config)
        
        # Display results
        print("\n" + "="*60)
        print("🎯 FINAL ANSWER:")
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
            print(f"🗳️  Votes: {result['voting_results']['distribution']}")
            print(f"🤖 Agents: {len(config.agents)}")
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 