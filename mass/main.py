#!/usr/bin/env python3
"""
MASS (Multi-Agent Scaling System) - Simple Main Interface

This module provides a clean, easy-to-use interface for running the MASS system.
Just specify your agents, model configs, and user question - the orchestrator handles everything!

Usage examples:
    # Simple usage with model names
    from mass import run_mass_agents
    result = run_mass_agents("What is 2+2?", ["gpt-4o", "gemini-2.5-flash"])
    print(result["answer"])
    
    # Advanced usage with custom configs  
    from mass import MassSystem
    system = MassSystem()
    result = system.run("Complex question here", agent_configs=[...])
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from .types import TaskInput, ModelConfig
from .orchestrator import MassOrchestrator  
from .agents import create_agent
from .streaming_display import create_streaming_display
from .utils import get_agent_type_from_model

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MassSystem:
    """
    Simple MASS system interface.
    
    Just specify agents and questions - the orchestrator handles all the complexity!
    """
    
    def __init__(self, 
                 max_duration: int = 600,
                 consensus_threshold: float = 1.0,
                 streaming_display: bool = True):
        """
        Initialize the MASS system.
        
        Args:
            max_duration: Maximum duration in seconds (default: 10 minutes)
            consensus_threshold: Consensus threshold (default: 1.0 = unanimous)  
            streaming_display: Whether to show real-time progress (default: True)
        """
        self.max_duration = max_duration
        self.consensus_threshold = consensus_threshold
        self.streaming_display = streaming_display
        
    def run(self, 
            question: str,
            models: Optional[List[str]] = None,
            agent_configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run MASS agents on a question and return the final answer.
        
        Args:
            question: The question to solve
            models: List of model names (e.g., ["gpt-4o", "gemini-2.5-flash"])  
            agent_configs: Advanced agent configurations (optional)
            
        Returns:
            Dict containing:
            - answer: The final answer string
            - consensus_reached: Whether consensus was reached
            - agent_summary: Summary of each agent's work
            - session_duration: How long it took
            - voting_results: Voting details
            
        Examples:
            # Simple usage
            result = system.run("What is 2+2?", ["gpt-4o", "claude-3"])
            
            # Advanced usage  
            configs = [
                {"type": "openai", "model_config": ModelConfig(model="gpt-4o", tools=["python_interpreter"])},
                {"type": "gemini", "model_config": ModelConfig(model="gemini-2.5-flash")}
            ]
            result = system.run("Complex question", agent_configs=configs)
        """
        
        # Create task input
        task = TaskInput(question=question)
        
        # Convert models to agent configs if provided
        if models and not agent_configs:
            agent_configs = self._convert_models_to_configs(models)
        elif not models and not agent_configs:
            raise ValueError("Must provide either 'models' list or 'agent_configs'")
        
        # Create streaming display
        streaming_orchestrator = create_streaming_display(
            display_type="terminal",
            display_enabled=self.streaming_display
        ) if self.streaming_display else None
        
        # Create orchestrator
        orchestrator = MassOrchestrator(
            max_duration=self.max_duration,
            consensus_threshold=self.consensus_threshold,
            streaming_orchestrator=streaming_orchestrator
        )
        
        # Register agents
        for i, config in enumerate(agent_configs):
            agent_id = i + 1
            agent_type = config["type"]
            model_config = config.get("model_config", ModelConfig())
            
            agent = create_agent(
                agent_type=agent_type,
                agent_id=agent_id, 
                orchestrator=orchestrator,
                model_config=model_config
            )
            
            orchestrator.register_agent(agent)
            
        logger.info(f"🚀 Starting MASS with {len(agent_configs)} agents")
        logger.info(f"   Question: {question}")
        logger.info(f"   Models: {[cfg.get('model_config', {}).model for cfg in agent_configs]}")
        
        # Start the task and get results
        try:
            result = orchestrator.start_task(task)
            logger.info("✅ MASS completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ MASS failed: {e}")
            raise
        finally:
            # Cleanup
            orchestrator.cleanup()
            
    def _convert_models_to_configs(self, models: List[str]) -> List[Dict[str, Any]]:
        """Convert model names to agent configurations."""
        configs = []
        for model in models:
            agent_type = get_agent_type_from_model(model)
            model_config = ModelConfig(
                model=model,
                tools=["python_interpreter", "calculator"]  # Default tools
            )
            configs.append({
                "type": agent_type,
                "model_config": model_config
            })
        return configs


def run_mass_agents(question: str, 
                   models: List[str],
                   max_duration: int = 600,
                   consensus_threshold: float = 1.0,
                   streaming_display: bool = True) -> Dict[str, Any]:
    """
    Simple function to run MASS agents on a question.
    
    This is the easiest way to use MASS - just provide a question and model names!
    
    Args:
        question: The question to solve
        models: List of model names (e.g., ["gpt-4o", "gemini-2.5-flash", "grok-4"])
        max_duration: Maximum duration in seconds (default: 10 minutes)
        consensus_threshold: Consensus threshold (default: 1.0 = unanimous)
        streaming_display: Whether to show real-time progress (default: True)
        
    Returns:
        Dict containing the answer and detailed results
        
    Examples:
        # Simple math question
        result = run_mass_agents("What is 15 * 23?", ["gpt-4o", "gemini-2.5-flash"])
        print(result["answer"])
        
        # Complex analysis
        result = run_mass_agents(
            "Analyze the pros and cons of renewable energy", 
            ["gpt-4o", "claude-3", "gemini-2.5-flash"]
        )
        print(result["answer"])
    """
    system = MassSystem(
        max_duration=max_duration,
        consensus_threshold=consensus_threshold, 
        streaming_display=streaming_display
    )
    
    return system.run(question=question, models=models)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MASS agents on a question")
    parser.add_argument("question", help="Question to solve")
    parser.add_argument("--models", nargs="+", required=True, 
                       help="Model names (e.g., gpt-4o gemini-2.5-flash)")
    parser.add_argument("--max-duration", type=int, default=600,
                       help="Max duration in seconds (default: 600)")
    parser.add_argument("--consensus", type=float, default=1.0,
                       help="Consensus threshold (default: 1.0)")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable streaming display")
    
    args = parser.parse_args()
    
    try:
        result = run_mass_agents(
            question=args.question,
            models=args.models,
            max_duration=args.max_duration,
            consensus_threshold=args.consensus,
            streaming_display=not args.no_display
        )
        
        print("\n" + "="*60)
        print("🎯 FINAL ANSWER:")
        print("="*60)
        print(result["answer"])
        print("\n" + "="*60)
        print(f"✅ Consensus: {result['consensus_reached']}")
        print(f"⏱️  Duration: {result['session_duration']:.1f}s")
        print(f"🗳️  Votes: {result['voting_results']['distribution']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 