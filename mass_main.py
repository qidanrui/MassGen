#!/usr/bin/env python3
"""
MASS (Multi-Agent Scaling System) - Main Entry Point

This module provides the main interface for running the MASS system on given tasks.
It handles system initialization, agent setup, and workflow execution.

Usage examples:
    # Run with a JSON task file
    python mass_main.py --task-file hle/test_case.json --agents openai,gemini,grok

    # Run with custom configuration
    python mass_main.py --task-file task.json --agents openai,openai,gemini --config config.json

    # Run programmatically
    from mass_main import MassSystem
    system = MassSystem()
    result = system.run_task_from_file("hle/test_case.json", ["openai", "gemini", "grok"])
"""

import argparse
import json
import logging
import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from mass_agent import TaskInput
from mass_coordination import MassCoordinationSystem
from mass_workflow import MassWorkflowManager
from mass_agents import create_agent

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
        self.config = config or {}
        
        # System components
        self.coordination_system: Optional[MassCoordinationSystem] = None
        self.workflow_manager: Optional[MassWorkflowManager] = None
        self.agents: List[Any] = []
        
        # Configuration parameters
        self.max_rounds = self.config.get("max_rounds", 5)
        self.consensus_threshold = self.config.get("consensus_threshold", 1.0)
        self.max_phase_duration = self.config.get("max_phase_duration", 300)
        self.parallel_execution = self.config.get("parallel_execution", True)
        self.check_update_frequency = self.config.get("check_update_frequency", 3)  # Default 3 seconds as per README
        
    def initialize_system(self, agent_configs: List[Dict[str, Any]]):
        """
        Initialize the coordination system and agents.
        
        Args:
            agent_configs: List of agent configuration dictionaries
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING MASS SYSTEM")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  - Max rounds: {self.max_rounds}")
        logger.info(f"  - Consensus threshold: {self.consensus_threshold}")
        logger.info(f"  - Max phase duration: {self.max_phase_duration} seconds")
        logger.info(f"  - Parallel execution: {self.parallel_execution}")
        logger.info(f"  - Agent count: {len(agent_configs)}")
        
        # Create coordination system
        logger.info("Creating coordination system...")
        self.coordination_system = MassCoordinationSystem(
            max_rounds=self.max_rounds,
            consensus_threshold=self.consensus_threshold
        )
        logger.debug(f"Coordination system created with consensus threshold {self.consensus_threshold}")
        
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
                    coordination_system=self.coordination_system,
                    **agent_kwargs
                )
                self.coordination_system.register_agent(agent)
                self.agents.append(agent)
                logger.info(f"✓ Registered agent {i}: {agent_type}")
                
            except Exception as e:
                logger.error(f"✗ Failed to create agent {i} of type {agent_type}: {str(e)}")
                logger.debug(f"Agent creation error details:", exc_info=True)
                raise e
        
        # Create workflow manager
        logger.info("Creating workflow manager...")
        self.workflow_manager = MassWorkflowManager(
            coordination_system=self.coordination_system,
            max_phase_duration=self.max_phase_duration,
            parallel_execution=self.parallel_execution,
            check_update_frequency=self.check_update_frequency
        )
        logger.debug(f"Workflow manager created with max_phase_duration={self.max_phase_duration}, parallel={self.parallel_execution}, check_update_frequency={self.check_update_frequency}")
        
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
        logger.info(f"  - Agent types: {[type(agent).__name__ for agent in self.agents]}")
        logger.info(f"  - Parallel execution: {self.parallel_execution}")
        logger.info(f"  - Max phase duration: {self.max_phase_duration}s")
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
                logger.info(f"  - Total rounds: {final_sol.get('total_rounds', 'N/A')}")
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
        
        return results
    
    def run_task_from_file(self, task_file_path: str, agent_types: List[str], 
                          output_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the MASS system on a task loaded from a JSON file.
        Now includes expected answer loading and evaluation.
        
        Args:
            task_file_path: Path to the JSON file containing the task
            agent_types: List of agent types to use (e.g., ["openai", "gemini", "grok"])
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
            # Convert agent types to proper configurations
            agent_configs = [{"type": agent_type} for agent_type in agent_types]
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
            print("\n" + "🔍"*60)
            print("📊 EVALUATION SUMMARY")
            print("🔍"*60)
            
            # 🚨 MOST IMPORTANT: FINAL ANSWER AND EVALUATION 🚨
            print("\n" + "🎯" * 40)
            print("🔥 FINAL ANSWER & EVALUATION - KEY RESULT!")
            print("🎯" * 40)
            
            extracted_answer = final_solution.get('extracted_answer', 'None')
            expected_answer = final_solution.get('expected_answer', 'None')
            is_correct = final_solution.get('is_answer_correct')
            
            print(f"📝 EXTRACTED ANSWER: {extracted_answer}")
            print(f"✅ EXPECTED ANSWER: {expected_answer}")
            
            if is_correct is not None:
                if is_correct:
                    print(f"🎉 EVALUATION: ✅ CORRECT!")
                    print(f"🏆 SUCCESS: The system found the correct answer!")
                else:
                    print(f"💥 EVALUATION: ❌ INCORRECT!")
                    print(f"🚨 FAILURE: The system did not find the correct answer!")
            else:
                print("⚠️  EVALUATION: Not Available")
            
            print("🎯" * 40)
            print("🔥 END KEY RESULT")
            print("🎯" * 40)
            
            # Additional performance metrics
            session_log = results.get("session_log", {})
            system_metrics = session_log.get("system_metrics", {}).get("performance", {})
            print(f"\n💰 Total Cost: ${system_metrics.get('total_cost_usd', 0.0):.4f}")
            print(f"⏱️  Total Time: {final_solution.get('total_runtime', 0.0):.2f} seconds")
            print(f"🔧 Total Tokens: {system_metrics.get('total_tokens_used', 0):,}")
            print("🔍"*60)
        
        return results
    
    def run_task_from_dict(self, task_dict: Dict[str, Any], agent_types: List[str]) -> Dict[str, Any]:
        """
        Run MASS workflow from a task dictionary.
        
        Args:
            task_dict: Dictionary containing task information
            agent_types: List of agent types to use
            
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
        agent_configs = [{"type": agent_type} for agent_type in agent_types]
        
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
            'consensus': '🟢',
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
        help="Comma-separated list of agent types (e.g., 'openai,gemini,grok')"
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
        "--max-phase-duration",
        type=int,
        default=300,
        help="Maximum duration per phase in seconds (default: 300)"
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
            "max_phase_duration": args.max_phase_duration,
            "parallel_execution": not args.sequential if args.sequential else args.parallel,
            "check_update_frequency": args.check_update_frequency
        })
        
        # Parse agent types
        agent_types = [agent.strip() for agent in args.agents.split(",")]
        
        # Validate task file exists
        if not os.path.exists(args.task_file):
            logger.error(f"Task file not found: {args.task_file}")
            sys.exit(1)
        
        # Create and run MASS system
        mass_system = MassSystem(config=config)
        results = mass_system.run_task_from_file(args.task_file, agent_types, args.output)
        
        # The detailed results are already printed by run_task_from_file
        # Print final summary
        print("\n" + "="*80)
        print("MASS EXECUTION COMPLETED")
        print("="*80)
        
        if results["success"]:
            print(f"✅ Workflow completed successfully in {results['total_workflow_time']:.2f} seconds")
            
            final_solution = results["final_solution"]
            if final_solution:
                print(f"🏆 Final solution from Agent {final_solution['agent_id']}")
                print(f"🔄 Total rounds: {final_solution['total_rounds']}")
                print(f"📊 Vote distribution: {final_solution['vote_distribution']}")
                
                # 🚨 MOST IMPORTANT: FINAL ANSWER EVALUATION 🚨
                print("\n" + "🎯" * 50)
                print("🔥 FINAL ANSWER EVALUATION - CRITICAL RESULT!")
                print("🎯" * 50)
                
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
                        print(f"🎉 RESULT: ✅ CORRECT! System succeeded!")
                    else:
                        print(f"💥 RESULT: ❌ INCORRECT! System failed!")
                else:
                    print(f"⚠️  RESULT: Cannot evaluate (no expected answer)")
                    
                print("🎯" * 50)
                print("🔥 END CRITICAL RESULT")
                print("🎯" * 50)
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
        sys.exit(0 if results["success"] else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 