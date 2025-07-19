#!/usr/bin/env python3
"""
MASS (Multi-Agent Scaling System) - Main Entry Point

This module provides the main interface for running the MASS system on given tasks.
It handles system initialization, agent setup, and workflow execution.

Usage examples:
    # Simple usage - just provide a question and models
    from mass import MassSystem
    system = MassSystem()
    result = system.run_mass_agents("What is 2+2?", ["gpt-4o", "gemini-2.5-flash"])
    print(result["answer"])
    print(result["summary"])

    # Command line usage
    python -m mass.main --task "What is 2+2?" --agents gpt-4o,gemini-2.5-flash
"""

import argparse
import json
import logging
import sys
import os
import signal
import atexit
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from .agent import TaskInput
from .orchestration import MassOrchestrationSystem
from .workflow import MassWorkflowManager
from .agents import create_agent
from .logging import initialize_logging, cleanup_logging
from agents.constants import get_agent_type_from_model, get_available_models

def create_agent_configs_from_models(model_names: List[str]) -> List[Dict[str, Any]]:
    """
    Convert a list of model names to agent configurations.
    
    Args:
        model_names: List of model names (e.g., ["gpt-4o", "gemini-2.5-flash", "grok-4"])
        
    Returns:
        List of agent configuration dictionaries with type and model specified
        
    Raises:
        ValueError: If any model name is invalid
    """
    agent_configs = []
    for model_name in model_names:
        try:
            agent_type = get_agent_type_from_model(model_name)
            agent_configs.append({
                "type": agent_type,
                "kwargs": {"model": model_name}
            })
        except ValueError as e:
            raise ValueError(f"Invalid model name '{model_name}': {str(e)}")
    
    return agent_configs

# Initialize basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global reference for cleanup
_global_mass_system = None

def _cleanup_on_exit():
    """Global cleanup function called on exit."""
    global _global_mass_system
    if _global_mass_system:
        try:
            _global_mass_system.cleanup_agents()
        except Exception as e:
            print(f"Warning: Failed to cleanup agents on exit: {e}")

def _signal_handler(signum, frame):
    """Handle termination signals to ensure cleanup."""
    print(f"\nReceived signal {signum}, cleaning up...")
    _cleanup_on_exit()
    os._exit(1)

# Register cleanup handlers
atexit.register(_cleanup_on_exit)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

class MassSystem:
    """
    Main MASS system orchestrator.
    
    This class provides a high-level interface for running the complete
    MASS workflow on given tasks with multiple agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MASS system.
        
        Args:
            config: Optional configuration dictionary
        """
        global _global_mass_system
        _global_mass_system = self  # Set global reference for cleanup
        
        self.config = config or {}
        
        # System components
        self.orchestration_system: Optional[MassOrchestrationSystem] = None
        self.workflow_manager: Optional[MassWorkflowManager] = None
        self.agents: List[Any] = []
        
        # Configuration parameters
        self.max_rounds = self.config.get("max_rounds", 5)
        self.consensus_threshold = self.config.get("consensus_threshold", 1.0)
        self.parallel_execution = self.config.get("parallel_execution", True)
        self.check_update_frequency = self.config.get("check_update_frequency", 3)
        self.save_logs = self.config.get("save_logs", True)  # Whether to save logs to files

    def initialize_system(self, agent_configs: List[Dict[str, Any]]):
        """
        Initialize the orchestration system and agents.
        
        Args:
            agent_configs: List of agent configuration dictionaries
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING MASS SYSTEM")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  - Max rounds: {self.max_rounds}")
        logger.info(f"  - Consensus threshold: {self.consensus_threshold}")
        logger.info(f"  - Parallel execution: {self.parallel_execution}")
        logger.info(f"  - Agent count: {len(agent_configs)}")
        
        # Initialize logging system
        logger.info("Initializing logging system...")
        self.log_manager = initialize_logging(non_blocking=True)
        logger.info(f"✓ Logging system initialized")
        
        # Create orchestration system
        logger.info("Creating orchestration system...")
        self.orchestration_system = MassOrchestrationSystem(
            max_rounds=self.max_rounds,
            consensus_threshold=self.consensus_threshold
        )
        
        # Create and register agents
        logger.info("Creating and registering agents...")
        self.agents = []
        for i, agent_config in enumerate(agent_configs):
            agent_type = agent_config.get("type", "openai")
            agent_kwargs = agent_config.get("kwargs", {})
            
            try:
                agent = create_agent(
                    agent_type=agent_type,
                    agent_id=i,
                    orchestration_system=self.orchestration_system,
                    **agent_kwargs
                )
                self.orchestration_system.register_agent(agent)
                self.agents.append(agent)
                model_name = agent_kwargs.get("model", "default")
                logger.info(f"✓ Registered agent {i}: {agent_type} ({model_name})")
                
            except Exception as e:
                logger.error(f"✗ Failed to create agent {i} of type {agent_type}: {str(e)}")
                raise e
        
        # Create workflow manager
        logger.info("Creating workflow manager...")
        self.workflow_manager = MassWorkflowManager(
            orchestration_system=self.orchestration_system,
            parallel_execution=self.parallel_execution,
            check_update_frequency=self.check_update_frequency,
            streaming_display=True,
            stream_callback=None,
            save_logs=self.save_logs
        )
        
        # Connect streaming orchestrator to orchestration system
        if self.workflow_manager.streaming_orchestrator:
            self.orchestration_system.streaming_orchestrator = self.workflow_manager.streaming_orchestrator
            
            # Set agent model names in the display
            for agent_id, agent in self.orchestration_system.agents.items():
                if hasattr(agent, 'model'):
                    model_name = agent.model
                    self.workflow_manager.streaming_orchestrator.set_agent_model(agent_id, model_name)
        
        logger.info("✓ MASS system initialization completed successfully")
        logger.info("=" * 60)
    
    def cleanup_agents(self):
        """Clean up all agent resources to prevent hanging processes."""
        if hasattr(self, 'agents') and self.agents:
            logger.info("Cleaning up agent resources...")
            for agent in self.agents:
                if hasattr(agent, 'cleanup'):
                    try:
                        agent.cleanup()
                        logger.debug(f"✓ Cleaned up Agent {agent.agent_id}")
                    except Exception as e:
                        logger.warning(f"Warning: Failed to cleanup Agent {agent.agent_id}: {e}")
            logger.info("✓ Agent cleanup completed")
    
    def _default_progress_callback(self, phase: str, current: int, total: int):
        """Default progress callback that logs progress."""
        logger.info(f"Phase {phase}: {current}/{total} completed")
    
    def run_task(self, task: TaskInput, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run the MASS workflow on a given task.
        
        Args:
            task: TaskInput containing the problem to solve
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing the complete workflow results
        """
        if not self.workflow_manager:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        logger.info("=" * 80)
        logger.info("STARTING MASS WORKFLOW EXECUTION")
        logger.info("=" * 80)
        logger.info(f"Task Details:")
        logger.info(f"  - Task ID: {task.task_id}")
        logger.info(f"  - Question: {task.question[:200]}{'...' if len(task.question) > 200 else ''}")
        logger.info("=" * 80)
        
        # Add default progress callback if none provided
        if progress_callback is None:
            progress_callback = self._default_progress_callback
        
        # Record workflow start time
        import time
        workflow_start = time.time()
        
        # Run the complete workflow
        logger.info("🚀 Launching workflow manager...")
        results = self.workflow_manager.run_complete_workflow(task, progress_callback)
        
        # Calculate total execution time
        workflow_duration = time.time() - workflow_start
        
        # Log results summary
        logger.info("=" * 80)
        logger.info("WORKFLOW EXECUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total workflow duration: {workflow_duration:.2f} seconds")
        
        if results["success"]:
            logger.info("✅ MASS workflow completed successfully")
            if results["final_solution"]:
                final_sol = results["final_solution"]
                logger.info(f"🏆 Final solution details:")
                logger.info(f"  - Winning agent: {final_sol['agent_id']}")
                logger.info(f"  - Answer: {final_sol.get('extracted_answer', 'No answer extracted')}")
        else:
            logger.error("❌ MASS workflow failed")
            error_msg = results.get('error', 'Unknown error')
            logger.error(f"Error details: {error_msg}")
        
        logger.info("=" * 80)
        return results
    
    def run_mass_agents(self, question: str, model_names: List[str]) -> Dict[str, Any]:
        """
        Simple interface to run MASS system with just a question and model names.
        
        Args:
            question: The question/task to solve
            model_names: List of model names to use (e.g., ["gpt-4o", "gemini-2.5-flash"])
            
        Returns:
            Dictionary containing:
            - success: bool indicating if workflow succeeded
            - answer: extracted answer from winning agent (or None)
            - summary: full summary from winning agent (or None)
            - agent_id: ID of the winning agent (or None)
            - error: error message if failed (or None)
        """
        try:
            
            if os.path.exists(question):
                if question.endswith(".json"):
                    # load it as a json file
                    with open(question, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    question = task_data.get("question", "")
                    context = task_data.get("context", {})
                    task_id = task_data.get("task_id", f"simple_task_{int(datetime.now().timestamp())}")
                elif question.endswith(".txt"):
                    # load it as a txt file
                    with open(question, 'r', encoding='utf-8') as f:
                        question = f.read()
                    context = {}
                    task_id = f"simple_task_{int(datetime.now().timestamp())}"
                else:
                    raise ValueError(f"Unsupported file type: {question}")
            else:
                # Create simple task input
                task = TaskInput(
                    question=question,
                    context={},
                    task_id=f"simple_task_{int(datetime.now().timestamp())}"
                )
            
            # Create agent configurations
            agent_configs = create_agent_configs_from_models(model_names)
            
            # Initialize system
            self.initialize_system(agent_configs)
            
            # Run the workflow
            results = self.run_task(task)
            
            # Extract simple results
            if results["success"] and results.get("final_solution"):
                final_solution = results["final_solution"]
                return {
                    "success": True,
                    "answer": final_solution.get("extracted_answer"),
                    "summary": final_solution.get("solution"),
                    "agent_id": final_solution.get("agent_id"),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "answer": None,
                    "summary": None,
                    "agent_id": None,
                    "error": results.get("error", "Unknown error occurred")
                }
                
        except Exception as e:
            logger.error(f"Error in run_mass_agents: {str(e)}")
            return {
                "success": False,
                "answer": None,
                "summary": None,
                "agent_id": None,
                "error": str(e)
            }
        finally:
            # Always cleanup
            try:
                self.cleanup_agents()
                if hasattr(self, 'log_manager') and self.log_manager:
                    cleanup_logging()
            except Exception as e:
                logger.warning(f"Warning: Failed to cleanup resources: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MASS (Multi-Agent Scaling System)")
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="The question/task to solve"
    )
    
    parser.add_argument(
        "--agents", "-a",
        type=str,
        required=True,
        help="Comma-separated list of model names (e.g., 'gpt-4o,gemini-2.5-flash,grok-4')"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to output file for results"
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
    
    return parser.parse_args()

def main():
    """Main entry point for command line usage."""
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("MASS System Starting")
    
    # Parse model names
    model_names = [model.strip() for model in args.agents.split(",")]
    
    # Validate model names
    try:
        available_models = get_available_models()
        for model_name in model_names:
            if model_name not in available_models:
                logger.error(f"Invalid model name: {model_name}")
                logger.error(f"Available models: {', '.join(available_models)}")
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
    mass_system = MassSystem(config=config)
    result = mass_system.run_mass_agents(args.question, model_names)
    
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