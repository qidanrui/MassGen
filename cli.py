#!/usr/bin/env python3
"""
Command Line Interface for MASS (Multi-Agent Scaling System)

This provides a simple command-line interface to run MASS tasks.

Usage examples:
    python cli.py --question "What is 2+2?" --agents gpt-4o,gemini-2.5-flash
    python cli.py --question "test_case1.json" --agents gpt-4o,gemini-2.5-flash --output results.json
"""

import argparse
import json
import logging
import sys
from typing import List

from mass import MassSystem
from mass.backends.constants import get_available_models

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MASS (Multi-Agent Scaling System) CLI")
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="The question/task to solve (or path to .json/.txt file)"
    )
    
    parser.add_argument(
        "--agents", "-a",
        type=str,
        help="Comma-separated list of model names (e.g., 'gpt-4o,gemini-2.5-flash,grok-4')"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum collaboration rounds (default: 5)"
    )
    
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=1.0,
        help="Consensus threshold (0.0-1.0, default: 1.0 = unanimous)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model names and exit"
    )
    
    args = parser.parse_args()
    
    # Validate that question and agents are provided unless --list-models is used
    if not args.list_models:
        if not args.question:
            parser.error("--question/-q is required when not using --list-models")
        if not args.agents:
            parser.error("--agents/-a is required when not using --list-models")
    
    return args

def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Handle list models command
    if args.list_models:
        try:
            available_models = get_available_models()
            print("Available models:")
            for model in sorted(available_models):
                print(f"  - {model}")
        except Exception as e:
            print(f"Error getting available models: {e}")
            sys.exit(1)
        return
    
    logger.info("MASS CLI Starting")
    
    # Parse model names
    model_names = [model.strip() for model in args.agents.split(",")]
    
    # Validate model names
    try:
        available_models = get_available_models()
        for model_name in model_names:
            if model_name not in available_models:
                logger.error(f"Invalid model name: {model_name}")
                logger.error(f"Available models: {', '.join(available_models)}")
                logger.error("Use --list-models to see all available models")
                sys.exit(1)
    except Exception as e:
        logger.error(f"Error validating model names: {str(e)}")
        sys.exit(1)
    
    # Create configuration
    config = {
        "max_rounds": args.max_rounds,
        "consensus_threshold": args.consensus_threshold,
    }
    
    # Create and run MASS system
    try:
        mass_system = MassSystem(config=config)
        result = mass_system.run_mass_agents(args.question, model_names)
    except Exception as e:
        logger.error(f"Error running MASS system: {str(e)}")
        sys.exit(1)
    
    # Print results
    print("\n" + "="*60)
    print("MASS EXECUTION COMPLETED")
    print("="*60)
    
    if result["success"]:
        print(f"✅ Workflow completed successfully")
        print(f"🏆 Winning Agent: {result['agent_id']}")
        
        if result["answer"]:
            print(f"\n📝 FINAL ANSWER:")
            print(f"{result['answer']}")
        else:
            print(f"\n❌ No answer could be extracted")
            
        if result["summary"]:
            print(f"\n📄 FULL SUMMARY:")
            print(f"{result['summary']}")
        
    else:
        print(f"❌ Workflow failed: {result['error']}")
    
    # Save results if requested
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n📁 Results saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    # Exit with appropriate code
    exit_code = 0 if result["success"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 