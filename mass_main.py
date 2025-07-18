#!/usr/bin/env python3
"""
MASS (Multi-Agent Scaling System) - Main Entry Point

This module provides the main interface for running the MASS system on given tasks.
It handles system initialization, agent setup, and workflow execution.

Usage examples:
    # Run with a JSON task file
    python mass_main.py --task-file test_case1.json --agents o4-mini,gemini-2.5-flash,grok-4

    # Run with custom configuration
    python mass_main.py --task-file test_case1.json --agents o4-mini,gemini-2.5-flash,grok-4 --config config.json

    # Run programmatically
    from mass_main import MassSystem
    system = MassSystem()
    result = system.run_task_from_file("hle/test_case.json", ["gpt-4o", "gemini-2.5-flash", "grok-4"])
"""

import argparse
import json
import logging
import sys
import os
import signal
import atexit
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from mass_agent import TaskInput
from mass_orchestration import MassOrchestrationSystem
from mass_workflow import MassWorkflowManager
from mass_agents import create_agent
from mass_logging import initialize_logging, cleanup_logging
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

def setup_enhanced_logging(verbose: bool = False):
    """
    Set up enhanced logging that writes to both console and timestamped log file.
    
    Args:
        verbose: If True, set logging level to DEBUG
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/{timestamp}.log'
    
    # Set logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters for console and file
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Enhanced logging initialized. Log file: {log_filename}")
    logger.info(f"Console log level: {log_level}")
    logger.info(f"File log level: DEBUG")
    
    return log_filename

# Initialize basic logging (will be enhanced in main() if needed)
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
        self.check_update_frequency = self.config.get("check_update_frequency", 3)  # Default 3 seconds as per README
        
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
        
        # Initialize comprehensive logging system
        logger.info("Initializing comprehensive logging system...")
        
        # Check for non-blocking mode (useful if logging system is causing hangs)
        non_blocking_mode = os.environ.get('MASS_NON_BLOCKING_LOGGING', 'false').lower() in ('true', '1', 'yes')
        if non_blocking_mode:
            logger.info("⚠️  Non-blocking logging mode enabled via MASS_NON_BLOCKING_LOGGING")
        
        self.log_manager = initialize_logging(non_blocking=non_blocking_mode)
        logger.info(f"✓ Logging system initialized with session ID: {self.log_manager.session_id}")
        
        # Create orchestration system
        logger.info("Creating orchestration system...")
        self.orchestration_system = MassOrchestrationSystem(
            max_rounds=self.max_rounds,
            consensus_threshold=self.consensus_threshold
        )
        logger.debug(f"Orchestration system created with consensus threshold {self.consensus_threshold}")
        
        # Create and register agents
        logger.info("Creating and registering agents...")
        self.agents = []
        for i, agent_config in enumerate(agent_configs):
            agent_type = agent_config.get("type", "openai")
            agent_kwargs = agent_config.get("kwargs", {})
            
            logger.debug(f"Creating agent {i} of type '{agent_type}' with kwargs: {agent_kwargs}")
            
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
                logger.debug(f"Agent creation error details:", exc_info=True)
                raise e
        
        # Create workflow manager
        logger.info("Creating workflow manager...")
        self.workflow_manager = MassWorkflowManager(
            orchestration_system=self.orchestration_system,
            parallel_execution=self.parallel_execution,
            check_update_frequency=self.check_update_frequency,
            streaming_display=True,  # Enable streaming display
            stream_callback=None  # Use default terminal display
        )
        logger.debug(f"Workflow manager created with parallel={self.parallel_execution}, check_update_frequency={self.check_update_frequency}")
        
        # Connect streaming orchestrator to orchestration system
        if self.workflow_manager.streaming_orchestrator:
            self.orchestration_system.streaming_orchestrator = self.workflow_manager.streaming_orchestrator
        
        logger.info("✓ MASS system initialization completed successfully")
        logger.info(f"✓ Total agents registered: {len(self.agents)}")
        logger.info("=" * 60)
    
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
        logger.info(f"  - Context keys: {list(task.context.keys()) if hasattr(task.context, 'keys') else 'N/A'}")
        logger.info(f"Execution Configuration:")
        logger.info(f"  - Number of agents: {len(self.agents)}")
        logger.info(f"  - Agent models: {[getattr(agent, 'model', 'unknown') for agent in self.agents]}")
        logger.info(f"  - Parallel execution: {self.parallel_execution}")
        logger.info("=" * 80)
        
        # Add default progress callback if none provided
        if progress_callback is None:
            progress_callback = self._enhanced_progress_callback
        
        # Record workflow start time
        import time
        workflow_start = time.time()
        
        # Run the complete workflow
        logger.info("🚀 Launching workflow manager...")
        results = self.workflow_manager.run_complete_workflow(task, progress_callback)
        
        # Calculate total execution time
        workflow_duration = time.time() - workflow_start
        
        # Log detailed results summary
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
                logger.info(f"  - Vote distribution: {final_sol.get('vote_distribution', 'N/A')}")
                logger.info(f"  - Solution length: {len(final_sol.get('solution', '')) if final_sol.get('solution') else 0} characters")
                logger.debug(f"  - Solution preview: {final_sol.get('solution', '')[:500]}{'...' if len(final_sol.get('solution', '')) > 500 else ''}")
        else:
            logger.error("❌ MASS workflow failed")
            error_msg = results.get('error', 'Unknown error')
            logger.error(f"Error details: {error_msg}")
            logger.debug(f"Full error context: {results}")
        
        # Log phase results summary
        if "phase_results" in results:
            logger.info("📊 Phase execution summary:")
            for phase, phase_results in results["phase_results"].items():
                if isinstance(phase_results, list):
                    success_count = sum(1 for r in phase_results if r.get('success', False))
                    logger.info(f"  - {phase}: {success_count}/{len(phase_results)} agents successful")
                    
        logger.info("=" * 80)
        
        # Cleanup agents and logging system
        try:
            # Clean up agent resources first
            self.cleanup_agents()
            
            # Then cleanup logging system (if not using run_task_from_file)
            if hasattr(self, 'log_manager') and self.log_manager:
                cleanup_logging()
        except Exception as e:
            logger.warning(f"Warning: Failed to cleanup resources: {e}")
        
        return results
    
    def run_task_from_file(self, task_file_path: str, model_names: List[str], 
                          output_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the MASS system on a task loaded from a JSON file.
        Now includes expected answer loading and evaluation.
        
        Args:
            task_file_path: Path to the JSON file containing the task
            model_names: List of model names to use (e.g., ["gpt-4o", "gemini-2.5-flash", "grok-4"])
            output_file_path: Optional path to save the results
            
        Returns:
            Dictionary containing the complete results, metrics, and evaluation
        """
        # Load task from file
        try:
            with open(task_file_path, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            # Extract task information including expected answer
            task_question = task_data.get("question", "")
            task_id = task_data.get("id", Path(task_file_path).stem)
            
            # Build context with all available information for evaluation
            task_context = {
                "source_file": task_file_path,
                "answer": task_data.get("answer"),  # Expected answer
                "answer_type": task_data.get("answer_type", "exactMatch"),
                "rationale": task_data.get("rationale"),
                "raw_subject": task_data.get("raw_subject"),
                "category": task_data.get("category"),
                "author_name": task_data.get("author_name"),
                **task_data.get("context", {})  # Include any existing context
            }
            
            # Create TaskInput with enhanced context
            task = TaskInput(
                question=task_question,
                context=task_context,
                task_id=task_id
            )
            
            print(f"\n📋 Loaded task from: {task_file_path}")
            print(f"🆔 Task ID: {task_id}")
            print(f"❓ Question: {task_question}")
            if task_context.get("answer"):
                print(f"✅ Expected Answer: {task_context['answer']}")
            if task_context.get("category"):
                print(f"📂 Category: {task_context['category']}")
            if task_context.get("raw_subject"):
                print(f"📚 Subject: {task_context['raw_subject']}")
            
        except Exception as e:
            logger.error(f"Failed to load task from {task_file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to load task: {str(e)}",
                "task_file": task_file_path
            }
        
        # Initialize system with agents
        try:
            # Convert model names to proper configurations
            agent_configs = create_agent_configs_from_models(model_names)
            self.initialize_system(agent_configs)
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to initialize system: {str(e)}",
                "task_file": task_file_path
            }
        
        # Run the workflow
        results = self.run_task(task)
        
        # Add task file information to results
        results["task_file"] = task_file_path
        results["task_data"] = task_data
        
        # Save results if requested
        if output_file_path:
            try:
                self.save_results(results, output_file_path)
                results["output_file"] = output_file_path
                print(f"📁 Results saved to: {output_file_path}")
            except Exception as e:
                logger.error(f"Failed to save results to {output_file_path}: {str(e)}")
                results["save_error"] = str(e)
        
        # Print evaluation summary
        if results["success"] and results.get("final_solution"):
            final_solution = results["final_solution"]
            print("\n" + "─" * 60)
            print("📊 EVALUATION SUMMARY")
            print("─" * 60)
            
            # Final Answer and Evaluation
            print("\n" + "=" * 40)
            print("🔥 FINAL ANSWER & EVALUATION")
            print("=" * 40)
            
            extracted_answer = final_solution.get('extracted_answer', 'None')
            expected_answer = final_solution.get('expected_answer', 'None')
            is_correct = final_solution.get('is_answer_correct')
            
            print(f"📝 EXTRACTED ANSWER: {extracted_answer}")
            print(f"✅ EXPECTED ANSWER: {expected_answer}")
            
            if is_correct is not None:
                if is_correct:
                    print(f"🎉 EVALUATION: ✅ CORRECT")
                    print(f"🏆 SUCCESS: System found the correct answer")
                else:
                    print(f"💥 EVALUATION: ❌ INCORRECT")
                    print(f"🚨 FAILURE: System did not find the correct answer")
            else:
                print("⚠️  EVALUATION: Not Available")
            
            print("=" * 40)
            print("=" * 40)
            
            # Additional performance metrics
            total_time = final_solution.get('total_runtime', 0.0)
            print(f"\n⏱️  Total Time: {total_time:.2f} seconds")
            if total_time == 0.0:
                print("   ⚠️  Warning: Total time is 0 - timing may not be calculated correctly")
            print("=" * 60)
        
        # Cleanup agents and logging system
        try:
            # Clean up agent resources first
            self.cleanup_agents()
            
            # Then cleanup logging system and export final report with timeout protection
            if hasattr(self, 'log_manager') and self.log_manager:
                print(f"📋 Exporting comprehensive session logs...")
                
                # Use timeout for the entire logging cleanup process
                import threading
                import time
                cleanup_result = {"completed": False, "report_file": ""}
                
                def logging_cleanup_worker():
                    try:
                        cleanup_result["report_file"] = self.log_manager.export_full_session_report()
                        cleanup_logging()
                        cleanup_result["completed"] = True
                    except Exception as e:
                        print(f"Warning: Logging cleanup error: {e}")
                        cleanup_result["completed"] = True  # Mark as completed even if failed
                
                # Run cleanup in daemon thread with timeout
                cleanup_thread = threading.Thread(target=logging_cleanup_worker, daemon=True)
                cleanup_thread.start()
                cleanup_thread.join(timeout=45.0)  # 45 second total timeout for logging cleanup
                
                if cleanup_result["completed"]:
                    if cleanup_result["report_file"]:
                        print(f"📄 Full session report: {cleanup_result['report_file']}")
                    else:
                        print(f"⚠️  Session report export was skipped or failed")
                else:
                    print(f"⏰ Logging cleanup timed out after 45 seconds - forcing exit")
            else:
                print(f"ℹ️  No log manager found, skipping logging cleanup")
                
        except Exception as e:
            logger.warning(f"Warning: Failed to cleanup resources: {e}")
            print(f"Warning: Cleanup error: {e}")
        
        return results
    
    def run_task_from_dict(self, task_dict: Dict[str, Any], model_names: List[str]) -> Dict[str, Any]:
        """
        Run MASS workflow from a task dictionary.
        
        Args:
            task_dict: Dictionary containing task information
            model_names: List of model names to use
            
        Returns:
            Dictionary containing the complete workflow results
        """
        # Create task input
        task = TaskInput(
            question=task_dict.get("question", ""),
            context=task_dict,
            task_id=task_dict.get("id")
        )
        
        # Create agent configurations
        agent_configs = create_agent_configs_from_models(model_names)
        
        # Initialize system
        self.initialize_system(agent_configs)
        
        # Run the task
        return self.run_task(task)
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Export results to a JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Results exported to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export results to {output_file}: {str(e)}")
            raise e
    
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
    
    def _load_task_from_file(self, file_path: str) -> TaskInput:
        """Load task from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                task_data = json.load(f)
            
            return TaskInput(
                question=task_data.get("question", ""),
                context=task_data,
                task_id=task_data.get("id")
            )
        except Exception as e:
            logger.error(f"Failed to load task from {file_path}: {str(e)}")
            raise e
    
    def _default_progress_callback(self, phase: str, current: int, total: int):
        """Default progress callback that logs progress."""
        logger.info(f"Phase {phase}: {current}/{total} completed")
        
    def _enhanced_progress_callback(self, phase: str, current: int, total: int):
        """Enhanced progress callback with detailed logging and emojis."""
        import time
        
        # Calculate progress percentage
        percentage = (current / total * 100) if total > 0 else 0
        
        # Progress bar visualization
        bar_length = 20
        filled_length = int(bar_length * current // total) if total > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Phase-specific emojis
        phase_emojis = {
            'initial': '🔵',
            'collaboration': '🟡', 
            'debate': '🟢',
            'presentation': '🔴',
            'processing': '⚙️',
            'voting': '🗳️',
            'finalization': '🎯'
        }
        emoji = phase_emojis.get(phase, '📊')
        
        # Enhanced logging with visual elements
        logger.info(f"{emoji} Phase: {phase.upper()}")
        logger.info(f"   Progress: [{bar}] {current}/{total} ({percentage:.1f}%)")
        logger.info(f"   Timestamp: {time.strftime('%H:%M:%S')}")
        
        # Additional details for specific phases
        if current == total and total > 0:
            logger.info(f"   ✅ Phase {phase} completed successfully!")
        elif current == 0:
            logger.info(f"   🚀 Starting phase {phase}...")



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MASS (Multi-Agent Scaling System)")
    
    parser.add_argument(
        "--task-file", "-t",
        type=str,
        required=True,
        help="Path to JSON file containing the task"
    )
    
    parser.add_argument(
        "--agents", "-a",
        type=str,
        required=True,
        help="Comma-separated list of model names (e.g., 'gpt-4o,gemini-2.5-flash,grok-4')"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to output file for results"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run agents in parallel (default: True)"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run agents sequentially instead of in parallel"
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
        "--check-update-frequency",
        type=float,
        default=3.0,
        help="How often agents check for updates in seconds (default: 3.0)"
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
    
    # Set up enhanced logging with timestamped log file
    log_filename = setup_enhanced_logging(verbose=args.verbose)
    logger.info(f"MASS System Starting - Log file: {log_filename}")
    logger.info(f"Command line arguments: {vars(args)}")
    
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Override config with command line arguments
        config.update({
            "max_rounds": args.max_rounds,
            "consensus_threshold": args.consensus_threshold,
            "parallel_execution": not args.sequential if args.sequential else args.parallel,
            "check_update_frequency": args.check_update_frequency
        })
        
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
        
        # Validate task file exists
        if not os.path.exists(args.task_file):
            logger.error(f"Task file not found: {args.task_file}")
            sys.exit(1)
        
        # Create and run MASS system
        mass_system = MassSystem(config=config)
        results = mass_system.run_task_from_file(args.task_file, model_names, args.output)
        
        # The detailed results are already printed by run_task_from_file
        # Print final summary
        print("\n" + "─"*60)
        print("MASS EXECUTION COMPLETED")
        print("─"*60)
        
        if results["success"]:
            print(f"✅ Workflow completed successfully in {results['total_workflow_time']:.2f} seconds")
            
            final_solution = results["final_solution"]
            if final_solution:
                print(f"🏆 Final solution from Agent {final_solution['agent_id']}")
                print(f"📊 Vote distribution: {final_solution['vote_distribution']}")
                
                # Final Answer Evaluation
                print("\n" + "─" * 50)
                print("🔥 FINAL ANSWER EVALUATION")
                print("─" * 50)
                
                extracted_answer = final_solution.get('extracted_answer')
                expected_answer = final_solution.get('expected_answer')
                is_correct = final_solution.get('is_answer_correct')
                
                if extracted_answer:
                    print(f"📝 EXTRACTED ANSWER: {extracted_answer}")
                else:
                    print("❌ NO ANSWER EXTRACTED!")
                    
                if expected_answer:
                    print(f"✅ EXPECTED ANSWER: {expected_answer}")
                    
                if is_correct is not None:
                    if is_correct:
                        print(f"🎉 RESULT: ✅ CORRECT")
                    else:
                        print(f"💥 RESULT: ❌ INCORRECT")
                else:
                    print(f"⚠️  RESULT: Cannot evaluate (no expected answer)")
                    
                print("─" * 50)
        else:
            print(f"❌ Workflow failed: {results.get('error', 'Unknown error')}")
            
            # Even for failed workflows, try to show any partial results
            final_solution = results.get("final_solution")
            if final_solution:
                print("\n⚠️  PARTIAL RESULTS FROM FAILED WORKFLOW:")
                extracted_answer = final_solution.get('extracted_answer')
                if extracted_answer:
                    print(f"📝 Partial Answer Found: {extracted_answer}")
                else:
                    print("❌ No answer could be extracted")
        
        # Results are already saved by run_task_from_file if args.output was provided
        
        # Exit with appropriate code
        exit_code = 0 if results["success"] else 1
        logger.info(f"Exiting with code {exit_code}")
        
        # Force exit to ensure process terminates even with lingering threads
        try:
            sys.exit(exit_code)
        finally:
            # Last resort: force process termination if sys.exit doesn't work
            os._exit(exit_code)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        try:
            sys.exit(1)
        finally:
            # Last resort: force process termination
            os._exit(1)

if __name__ == "__main__":
    main() 