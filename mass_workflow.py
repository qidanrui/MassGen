"""
MASS Workflow Manager

This module handles the orchestration of the four-phase workflow:
1. Initial Processing - Agents work independently
2. Collaboration & Refinement - Agents review each other's work
3. Consensus Building & Debate - Agents reach consensus through structured debate
4. Final Presentation - Representative presents the final solution
"""
import os
import asyncio
import threading
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from mass_agent import MassAgent, TaskInput, AgentResponse
from mass_orchestration import MassOrchestrationSystem
from mass_logging import get_log_manager
from mass_streaming_display import StreamingOrchestrator, create_streaming_display

# Set up logging
logger = logging.getLogger(__name__)

class MassWorkflowManager:
    """
    Orchestrates the complete MASS workflow across multiple agents.
    
    This manager handles the execution of all four phases, ensures proper
    orchestration between agents, and manages the overall workflow state.
    """
    
    def __init__(self, orchestration_system: MassOrchestrationSystem, 
                 parallel_execution: bool = True,
                 check_update_frequency: float = 3.0,  # 3 seconds default as per README
                 max_collaboration_rounds: int = 5,  # Maximum collaboration rounds
                 streaming_display: bool = True,  # Enable streaming display
                 stream_callback: Optional[Callable] = None,  # Custom stream callback
                 max_display_lines: int = 40,  # Maximum lines to display per agent
                 save_logs: bool = True):  # Whether to save logs to files
        """
        Initialize the workflow manager.
        
        Args:
            orchestration_system: The orchestration system managing agents
            parallel_execution: Whether to run agents in parallel
            check_update_frequency: How often agents check for updates in seconds (default 3)
            max_collaboration_rounds: Maximum number of collaboration rounds (default 5)
            streaming_display: Whether to enable streaming display (default True)
            stream_callback: Optional custom callback for streaming integration
            max_display_lines: Maximum lines to display per agent (default 40)
            save_logs: Whether to save agent outputs and system messages to log files (default True)
        """
        self.orchestration_system = orchestration_system
        self.parallel_execution = parallel_execution
        self.check_update_frequency = check_update_frequency
        self.max_collaboration_rounds = max_collaboration_rounds
        
        # Initialize streaming display
        self.streaming_orchestrator = create_streaming_display(
            display_type="terminal",
            display_enabled=streaming_display,
            stream_callback=stream_callback,
            max_lines=max_display_lines,
            save_logs=save_logs
        )
        
        # Workflow state
        self.current_phase = "initial"
        self.phase_results: Dict[str, List[Dict[str, Any]]] = {
            "initial": [],
            "collaboration": [],
            "debate": [],
            "presentation": []
        }
        
        # Callbacks for phase completion
        self.phase_callbacks: Dict[str, List[Callable]] = {
            "initial": [],
            "collaboration": [],
            "debate": [],
            "presentation": []
        }
    
    def add_phase_callback(self, phase: str, callback: Callable):
        """Add a callback to be executed when a phase completes."""
        if phase in self.phase_callbacks:
            self.phase_callbacks[phase].append(callback)
    
    def _update_orchestration_system_phase(self, new_phase: str):
        """Update the orchestration system's phase to match the workflow manager's phase."""
        old_phase = self.orchestration_system.system_state.phase
        self.orchestration_system.system_state.phase = new_phase
        logger.info(f"🔄 Phase transition: {old_phase} → {new_phase}")
        print(f"   🔄 System phase updated: {old_phase} → {new_phase}")
        
        # Update streaming display
        if self.streaming_orchestrator:
            self.streaming_orchestrator.update_system_phase(f"{old_phase} → {new_phase}")
    
    def run_complete_workflow(self, task: TaskInput, 
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the complete four-phase MASS workflow.
        
        Args:
            task: The task to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing comprehensive workflow results
        """
        workflow_start_time = time.time()
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE MASS WORKFLOW")
        logger.info("=" * 80)
        
        # Initialize orchestration system for this task
        self.orchestration_system.start_task(task)
        
        # Set expected answer if available in task context
        if task.context and "answer" in task.context:
            self.orchestration_system.set_expected_answer(task.context["answer"])
            logger.info(f"Expected answer set: {task.context['answer']}")
        
        # Phase 1: Initial Processing
        print("\n" + "🔵"*20)
        print("📝 PHASE 1: INITIAL PROCESSING")
        print("🔵"*20)
        print("🎯 Goal: Each agent works independently on the task")
        print("🛠️  Tools: live_search, code_execution")
        logger.info("Phase 1: Initial Processing")
        
        # Log phase transition
        log_manager = get_log_manager()
        if log_manager:
            log_manager.log_phase_transition("workflow_start", "initial")
        
        self.orchestration_system.start_phase_timing("initial")
        self.current_phase = "initial"
        self._update_orchestration_system_phase("initial")
        initial_results = self._run_phase_1_initial_processing(task, progress_callback)
        self.orchestration_system.end_phase_timing("initial")
        self.phase_results["initial"] = initial_results
        
        # Execute phase callbacks
        for callback in self.phase_callbacks["initial"]:
            callback(initial_results)
        
        successful_initial = len([r for r in initial_results if r['success']])
        print(f"✅ SYSTEM: Phase 1 Complete - {successful_initial}/{len(initial_results)} agents successful")
        
        # Phase 2: Collaboration & Refinement
        print("\n" + "🟡"*20)
        print("🤝 PHASE 2: COLLABORATION & REFINEMENT")
        print("🟡"*20)
        print("🎯 Goal: Agents review peer solutions and collaborate")
        print("🔄 Process: Multiple rounds until consensus or voting complete")
        logger.info("Phase 2: Collaboration & Refinement")
        
        # Log phase transition
        if log_manager:
            log_manager.log_phase_transition("initial", "collaboration")
        
        self.orchestration_system.start_phase_timing("collaboration")
        self.current_phase = "collaboration"
        self._update_orchestration_system_phase("collaboration")
        collaboration_results = self._run_phase_2_collaboration(task, progress_callback)
        self.orchestration_system.end_phase_timing("collaboration")
        self.phase_results["collaboration"] = collaboration_results
        
        # Execute phase callbacks
        for callback in self.phase_callbacks["collaboration"]:
            callback(collaboration_results)
        
        # Phase 3: Consensus Building & Debate
        print("─" * 60)
        print("🔥 PHASE 3: CONSENSUS BUILDING & DEBATE")
        print("─" * 60)
        print("🎯 Goal: Final consensus building and solution selection")
        logger.info("Phase 3: Consensus Building & Debate")
        
        # Log phase transition
        if log_manager:
            log_manager.log_phase_transition("collaboration", "debate")
        
        self.orchestration_system.start_phase_timing("debate")
        self.current_phase = "debate"
        self._update_orchestration_system_phase("debate")
        debate_results = self._run_phase_3_debate(task, progress_callback)
        self.orchestration_system.end_phase_timing("debate")
        self.phase_results["debate"] = debate_results
        
        # Execute phase callbacks
        for callback in self.phase_callbacks["debate"]:
            callback(debate_results)
        
        # Phase 4: Final Presentation (only if consensus reached)
        if self.orchestration_system.system_state.consensus_reached:
            print("─" * 60)
            print("🎉 PHASE 4: FINAL PRESENTATION")
            print("─" * 60)
            print("🎯 Goal: Representative presents final solution with full context")
            logger.info("Phase 4: Final Presentation")
            
            # Log phase transition
            if log_manager:
                log_manager.log_phase_transition("debate", "presentation")
            
            self.orchestration_system.start_phase_timing("presentation")
            self.current_phase = "presentation"
            self._update_orchestration_system_phase("presentation")
            presentation_results = self._run_phase_4_presentation(task, progress_callback)
            self.orchestration_system.end_phase_timing("presentation")
            self.phase_results["presentation"] = presentation_results
            
            # Execute phase callbacks
            for callback in self.phase_callbacks["presentation"]:
                callback(presentation_results)
        else:
            logger.error("Debate phase completed but consensus not reached")
            print("\n❌ WORKFLOW FAILED: Consensus not reached after debate phase")
            raise RuntimeError("Consensus not reached after debate phase")
        
        # Get final solution - let it fail if consensus wasn't reached
        final_solution = self.orchestration_system.get_final_solution()
        
        # Calculate final metrics
        total_workflow_time = time.time() - workflow_start_time
        
        # Build comprehensive results
        results = {
            "success": True,
            "total_workflow_time": total_workflow_time,
            "final_solution": final_solution,
            "phase_results": self.phase_results,
            "system_status": self.orchestration_system.get_system_status(),
            "session_log": self.orchestration_system.export_detailed_session_log()
        }
        
        # Log completion
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {total_workflow_time:.2f} seconds")
        logger.info(f"Winning solution: Agent {final_solution['agent_id']}")
        logger.info(f"Consensus method: {final_solution.get('consensus_method', 'natural')}")
        logger.info("=" * 80)
        
        return results
    
    def _run_phase_1_initial_processing(self, task: TaskInput, 
                                      progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Phase 1: All agents work independently on the initial task.
        
        Args:
            task: The task to process
            progress_callback: Optional progress callback
            
        Returns:
            List of results from all agents
        """
        agents = list(self.orchestration_system.agents.values())
        results = []
        
        if self.parallel_execution:
            # Run agents in parallel
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                future_to_agent = {
                    executor.submit(self._run_agent_phase, agent, task, "initial"): agent 
                    for agent in agents
                }
                
                completed_count = 0
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    result = future.result()  # Let exceptions propagate
                    results.append({
                        "agent_id": agent.agent_id,
                        "phase": "initial",
                        "result": result,
                        "success": True,
                        "error": None
                    })
                    completed_count += 1
                    print(f"✅ SYSTEM: Agent {agent.agent_id} completed initial processing ({completed_count}/{len(agents)})")
                    
                    if progress_callback:
                        progress_callback("initial", completed_count, len(agents))
        else:
            # Run agents sequentially
            for i, agent in enumerate(agents):
                result = self._run_agent_phase(agent, task, "initial")  # Let exceptions propagate
                results.append({
                    "agent_id": agent.agent_id,
                    "phase": "initial",
                    "result": result,
                    "success": True,
                    "error": None
                })
                
                if progress_callback:
                    progress_callback("initial", i + 1, len(agents))
        
        return results
    
    def _run_phase_2_collaboration(self, task: TaskInput, 
                                 progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Phase 2: Agents review each other's work and either improve or vote.
        
        This phase implements continuous update checking every check_update_frequency seconds
        and agent reactivation when new updates are available, as specified in README.
        
        The collaboration phase continues until either:
        1. All agents have voted (status == "voted"), OR
        2. Maximum collaboration rounds have been reached, OR
        3. Consensus is reached through voting
        """
        results = []
        phase_start_time = time.time()
        last_update_check = phase_start_time
        
        print(f"\n🔄 Collaboration Phase with continuous update checking (every {self.check_update_frequency}s)")
        
        # Add to streaming display
        if self.streaming_orchestrator:
            self.streaming_orchestrator.add_system_message("🔄 Starting Collaboration Phase")
        print("📋 Exit conditions: All agents voted OR max rounds reached OR consensus reached")
        print("-" * 70)
        logger.info(f"Collaboration phase with continuous update checking every {self.check_update_frequency} seconds")
        
        for round_num in range(self.max_collaboration_rounds):
            print(f"\n🔄 Collaboration Round {round_num + 1}/{self.max_collaboration_rounds}")
            print("-" * 50)
            logger.info(f"Collaboration round {round_num + 1}")
            
            # Check agent status at start of round
            all_agents = list(self.orchestration_system.agents.values())
            voted_agents = [
                agent for agent in all_agents
                if self.orchestration_system.agent_states[agent.agent_id].status == "voted"
            ]
            working_agents = [
                agent for agent in all_agents
                if self.orchestration_system.agent_states[agent.agent_id].status == "working"
            ]
            failed_agents = [
                agent for agent in all_agents
                if self.orchestration_system.agent_states[agent.agent_id].status == "failed"
            ]
            
            print(f"📊 SYSTEM: Round {round_num + 1} Status:")
            print(f"   👥 Working: {[a.agent_id for a in working_agents]} ({len(working_agents)} total)")
            print(f"   🗳️  Voted: {[a.agent_id for a in voted_agents]} ({len(voted_agents)} total)")
            print(f"   💥 Failed: {[a.agent_id for a in failed_agents]} ({len(failed_agents)} total)")
            
            # Exit condition 1: All agents have voted
            if len(voted_agents) == len(all_agents):
                logger.info("All agents have voted - ending collaboration phase")
                print(f"✅ SYSTEM: All {len(all_agents)} agents have voted")
                print(f"➡️  Moving to debate phase")
                break
            
            # Exit condition 2: All agents either voted, or failed
            failed_agents = [
                agent for agent in all_agents
                if self.orchestration_system.agent_states[agent.agent_id].status == "failed"
            ]
            if len(voted_agents) + len(failed_agents) == len(all_agents):
                logger.info("All agents have either voted or failed - ending collaboration phase")
                print(f"   ✅ EXIT CONDITION: All agents voted ({len(voted_agents)}) or failed ({len(failed_agents)})")
                print(f"   ➡️  Moving to debate phase")
                break
            
            # Exit condition 3: Consensus already reached
            if self.orchestration_system.system_state.consensus_reached:
                logger.info("Consensus already reached at start of round - ending collaboration phase")
                print(f"   ✅ EXIT CONDITION: Consensus already reached")
                print(f"   ➡️  Moving to debate phase")
                
                # Add to streaming display
                if self.streaming_orchestrator:
                    self.streaming_orchestrator.add_system_message("✅ Consensus already reached - moving to debate phase")
                break
            
            # Continuous update checking mechanism as per README
            round_start_time = time.time()
            agents_processed_this_round = set()
            round_timeout = 150  # Maximum time per round to prevent infinite loops
            
            while True:
                current_time = time.time()
                
                # Safety check: Prevent infinite loops
                if current_time - round_start_time > round_timeout:
                    logger.warning(f"Round {round_num + 1} timeout reached - forcing round completion")
                    print(f"   ⏰ Round timeout reached - completing round {round_num + 1}")
                    break
                
                # Check if we should do an update check (every check_update_frequency seconds)
                if current_time - last_update_check >= self.check_update_frequency:
                    print(f"   ⏰ Update check at {current_time - phase_start_time:.1f}s")
                    
                    # Check for agent reactivation due to new updates
                    reactivation_candidates = self.orchestration_system.check_for_reactivation()
                    if reactivation_candidates:
                        print(f"   🔄 Reactivating {len(reactivation_candidates)} agents due to new updates:")
                        for agent_id in reactivation_candidates:
                            self.orchestration_system.reactivate_agent(agent_id)
                            print(f"      - Agent {agent_id} reactivated")
                            # Remove from processed set so they can be processed again
                            agents_processed_this_round.discard(agent_id)
                    
                    last_update_check = current_time
                
                # Get agents that are still working and haven't been processed this round
                current_working_agents = [
                    agent for agent in self.orchestration_system.agents.values()
                    if (self.orchestration_system.agent_states[agent.agent_id].status == "working" 
                        and agent.agent_id not in agents_processed_this_round)
                ]
                
                # If no working agents left for this round, move to next round
                if not current_working_agents:
                    print(f"   ✅ No more working agents for round {round_num + 1}")
                    break
                
                # Process agents that need to work
                round_results = []
                
                if self.parallel_execution and len(current_working_agents) > 1:
                    # Run working agents in parallel
                    with ThreadPoolExecutor(max_workers=len(current_working_agents)) as executor:
                        future_to_agent = {
                            executor.submit(self._run_agent_phase, agent, task, "collaboration"): agent
                            for agent in current_working_agents
                        }
                        
                        # Agents will handle their own timeouts internally
                        for future in as_completed(future_to_agent):
                            agent = future_to_agent[future]
                            result = future.result()  # Let exceptions propagate
                            round_results.append({
                                "agent_id": agent.agent_id,
                                "phase": "collaboration",
                                "round": round_num + 1,
                                "result": result,
                                "success": True,
                                "error": None
                            })
                            agents_processed_this_round.add(agent.agent_id)
                else:
                    # Run agents sequentially
                    for agent in current_working_agents:
                        result = self._run_agent_phase(agent, task, "collaboration")  # Let exceptions propagate
                        round_results.append({
                            "agent_id": agent.agent_id,
                            "phase": "collaboration",
                            "round": round_num + 1,
                            "result": result,
                            "success": True,
                            "error": None
                        })
                        agents_processed_this_round.add(agent.agent_id)
                
                results.extend(round_results)
                
                # Check if consensus is reached after processing agents
                if self.orchestration_system.system_state.consensus_reached:
                    logger.info("Consensus reached during collaboration phase")
                    print(f"   ✅ EXIT CONDITION: Consensus reached during round processing")
                    print(f"   ➡️  Moving to debate phase")
                    
                    # Add to streaming display
                    if self.streaming_orchestrator:
                        self.streaming_orchestrator.add_system_message("✅ Consensus reached during round processing")
                    return results
                
                # Brief pause before next update check
                time.sleep(min(1, self.check_update_frequency / 2))
            
            # Check final agent status at end of round
            final_voted_agents = [
                agent for agent in self.orchestration_system.agents.values()
                if self.orchestration_system.agent_states[agent.agent_id].status == "voted"
            ]
            final_working_agents = [
                agent for agent in self.orchestration_system.agents.values()
                if self.orchestration_system.agent_states[agent.agent_id].status == "working"
            ]
            final_failed_agents = [
                agent for agent in self.orchestration_system.agents.values()
                if self.orchestration_system.agent_states[agent.agent_id].status == "failed"
            ]
            
            print(f"\n📊 End of Round {round_num + 1}:")
            print(f"   👥 Working agents: {[a.agent_id for a in final_working_agents]} ({len(final_working_agents)} total)")
            print(f"   🗳️  Voted agents: {[a.agent_id for a in final_voted_agents]} ({len(final_voted_agents)} total)")
            print(f"   💥 Failed agents: {[a.agent_id for a in final_failed_agents]} ({len(final_failed_agents)} total)")
            
            if progress_callback:
                progress_callback("collaboration", round_num + 1, self.max_collaboration_rounds)
            
            # Exit condition check at end of round
            if self.orchestration_system.system_state.consensus_reached:
                logger.info("Consensus reached at end of collaboration round")
                print(f"   ✅ EXIT CONDITION: Consensus reached at end of round {round_num + 1}")
                print(f"   ➡️  Moving to debate phase")
                
                # Add to streaming display
                if self.streaming_orchestrator:
                    self.streaming_orchestrator.add_system_message(f"✅ Consensus reached at end of round {round_num + 1}")
                break
            
            # Exit condition: All agents have voted or failed
            if len(final_voted_agents) + len(final_failed_agents) == len(all_agents):
                logger.info("All agents have either voted or failed at end of collaboration round")
                print(f"   ✅ EXIT CONDITION: All agents voted ({len(final_voted_agents)}) or failed ({len(final_failed_agents)})")
                print(f"   ➡️  Moving to debate phase")
                break
        
        # Check if we exited due to max rounds reached
        if round_num == self.max_collaboration_rounds - 1:
            final_voted = [
                agent for agent in self.orchestration_system.agents.values()
                if self.orchestration_system.agent_states[agent.agent_id].status == "voted"
            ]
            final_working = [
                agent for agent in self.orchestration_system.agents.values()
                if self.orchestration_system.agent_states[agent.agent_id].status == "working"
            ]
            final_failed = [
                agent for agent in self.orchestration_system.agents.values()
                if self.orchestration_system.agent_states[agent.agent_id].status == "failed"
            ]
            logger.info(f"Maximum collaboration rounds ({self.max_collaboration_rounds}) reached")
            print(f"\n🔄 EXIT CONDITION: Maximum collaboration rounds ({self.max_collaboration_rounds}) reached")
            print(f"   👥 Final working agents: {[a.agent_id for a in final_working]} ({len(final_working)} total)")
            print(f"   🗳️  Final voted agents: {[a.agent_id for a in final_voted]} ({len(final_voted)} total)")
            print(f"   💥 Final failed agents: {[a.agent_id for a in final_failed]} ({len(final_failed)} total)")
            print(f"   ➡️  Moving to debate phase")
        
        return results
    
    def _run_phase_3_debate(self, task: TaskInput, 
                             progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Phase 3: Consensus building & debate if not already reached.
        
        This phase handles consensus building through structured debate.
        """
        results = []
        
        # If consensus is already reached, just return
        if self.orchestration_system.system_state.consensus_reached:
            logger.info("Consensus already reached. Finalizing.")
            if progress_callback:
                progress_callback("debate", 1, 1)
            return results
        
        logger.info("Consensus not yet reached - running debate phase")
        print("🎯 Consensus not reached - running debate phase")
        
        # Add to streaming display
        if self.streaming_orchestrator:
            self.streaming_orchestrator.add_system_message("🎯 Starting Debate Phase")
        
        # Run any agents that might need to reconsider their votes
        working_agents = [
            agent for agent in self.orchestration_system.agents.values()
            if self.orchestration_system.agent_states[agent.agent_id].status == "working"
        ]
        
        if working_agents:
            logger.info(f"Running debate phase with {len(working_agents)} remaining agents")
            print(f"   👥 Processing {len(working_agents)} remaining working agents")
            
            for i, agent in enumerate(working_agents):
                result = self._run_agent_phase(agent, task, "debate")  # Let exceptions propagate
                results.append({
                    "agent_id": agent.agent_id,
                    "phase": "debate",
                    "result": result,
                    "success": True,
                    "error": None
                })
                
                if progress_callback:
                    progress_callback("debate", i + 1, len(working_agents))
                
                # Check if consensus reached after each agent
                if self.orchestration_system.system_state.consensus_reached:
                    logger.info("Consensus reached during debate phase")
                    print(f"   ✅ Consensus reached after processing Agent {agent.agent_id}")
                    
                    # Add to streaming display
                    if self.streaming_orchestrator:
                        self.streaming_orchestrator.add_system_message(f"✅ Consensus reached after processing Agent {agent.agent_id}")
                    break
        
        return results
    
    def _run_phase_4_presentation(self, task: TaskInput, 
                             progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Phase 4: Final presentation by the representative agent.
        
        The selected representative receives complete context and presents the final solution.
        """
        results = []
        
        # At this point, consensus is guaranteed to be reached (checked in main workflow)
        
        # Get the winning agent (representative) to present final solution
        final_solution = self.orchestration_system.get_final_solution()
        if not final_solution:
            logger.error("No final solution available despite consensus being reached")
            print("❌ ERROR: No final solution available despite consensus")
            raise RuntimeError("No final solution available despite consensus")
        
        representative_agent_id = final_solution['agent_id']
        representative_agent = self.orchestration_system.agents.get(representative_agent_id)
        
        if not representative_agent:
            logger.error(f"Representative agent {representative_agent_id} not found")
            print(f"❌ ERROR: Representative agent {representative_agent_id} not found")
            raise RuntimeError(f"Representative agent {representative_agent_id} not found")
        
        logger.info(f"Running presentation phase with representative Agent {representative_agent_id}")
        print(f"🎯 Running final presentation with representative Agent {representative_agent_id}")
        
        # Run the representative agent in presentation mode
        result = self._run_agent_phase(representative_agent, task, "presentation")
        results.append({
            "agent_id": representative_agent_id,
            "phase": "presentation",
            "result": result,
            "success": True,
            "error": None
        })
        
        if progress_callback:
            progress_callback("presentation", 1, 1)
        
        logger.info(f"Presentation phase completed by Agent {representative_agent_id}")
        print(f"✅ Final presentation completed by Agent {representative_agent_id}")
        
        return results
    
    def _run_agent_phase(self, agent: MassAgent, task: TaskInput, phase: str) -> AgentResponse:
        """
        Run a single agent for a specific phase with proper timing and orchestration.
        
        Args:
            agent: The agent to run
            task: The task to process
            phase: The workflow phase
            
        Returns:
            AgentResponse from the agent
        """
        print(f"\n🤖 Running Agent {agent.agent_id} ({phase} phase):")
        logger.debug(f"Starting agent {agent.agent_id} for {phase} phase")
        
        # Update streaming display - agent starting
        if self.streaming_orchestrator:
            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                agent.agent_id, f"{'=' * 40}\n🚀 Starting {phase} phase...\n{'=' * 40}\n"
            ))
            self.streaming_orchestrator.update_agent_status(agent.agent_id, f"working_{phase}")
        
        # Set execution start time if this is the initial phase
        if phase == "initial" and agent.state.execution_start_time is None:
            agent.state.execution_start_time = time.time()
        
        agent_start_time = time.time()
        
        # Process the task using the agent's new method with timeout
        # Set phase-specific timeouts
        phase_timeouts = {
            "initial": 150,      # 2 minutes for initial processing
            "collaboration": 150, # 2 minutes for collaboration
            "debate": 150,        # 2 minutes for debate
            "presentation": 150   # 2 minutes for presentation
        }
        timeout = phase_timeouts.get(phase, 150)
        
        print(f"⏰ Agent {agent.agent_id} timeout: {timeout}s for {phase} phase")
        logger.debug(f"Agent {agent.agent_id} processing with {timeout}s timeout")
        
        # Update streaming display - agent processing
        if self.streaming_orchestrator:
            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                agent.agent_id, f"🔄 Processing task (timeout: {timeout}s)...\n"
            ))
        
        # Create stream callback for real-time LLM response streaming
        # Track what we've already streamed to avoid duplicates
        streamed_content = {"total": ""}
        
        def llm_stream_callback(chunk: str):
            """Callback to stream LLM response chunks to the display."""
            if self.streaming_orchestrator and chunk:
                try:
                    # Check if this is a search query, function call, code, or reasoning message
                    if (chunk.startswith("[SEARCH]") or chunk.startswith("[FUNCTION]") or 
                        chunk.startswith("[CODE]") or chunk.startswith("[REASONING]") or 
                        chunk.startswith("[COMPLETE]") or chunk.startswith("[DONE]")):
                        # Display search queries and function calls in agent region with better formatting
                        if "[SEARCH]" in chunk:
                            # Clean up search query display and add to agent output
                            clean_chunk = chunk.replace("[SEARCH]", "").strip()
                            formatted_chunk = f"🔍 {clean_chunk}\n"
                            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                                agent.agent_id, formatted_chunk
                            ))
                        elif "[FUNCTION]" in chunk:
                            # Clean up function call display and add to agent output
                            clean_chunk = chunk.replace("[FUNCTION]", "").strip()
                            formatted_chunk = f"🔧 {clean_chunk}\n"
                            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                                agent.agent_id, formatted_chunk
                            ))
                        elif "[CODE]" in chunk:
                            # Clean up code execution display and add to agent output
                            clean_chunk = chunk.replace("[CODE]", "").strip()
                            formatted_chunk = f"💻 {clean_chunk}\n"
                            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                                agent.agent_id, formatted_chunk
                            ))
                        elif "[REASONING]" in chunk:
                            # Clean up reasoning display and add to agent output
                            clean_chunk = chunk.replace("[REASONING]", "").strip()
                            formatted_chunk = f"🧠 {clean_chunk}\n"
                            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                                agent.agent_id, formatted_chunk
                            ))
                        elif "[COMPLETE]" in chunk or "[DONE]" in chunk:
                            # Clean up completion messages and add to agent output with line break
                            clean_chunk = chunk.replace("[COMPLETE]", "").replace("[DONE]", "").strip()
                            formatted_chunk = f"\n✅ {clean_chunk}"
                            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                                agent.agent_id, formatted_chunk
                            ))
                        else:
                            # Default agent output display
                            formatted_chunk = f"ℹ️ {chunk}\n"
                            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                                agent.agent_id, formatted_chunk
                            ))
                    else:
                        # Regular content goes to agent output
                        asyncio.run(self.streaming_orchestrator.stream_agent_output(
                            agent.agent_id, chunk
                        ))
                except Exception as e:
                    print(f"LLM streaming error: {e}")
        
        # Process task with streaming if available
        response = agent.process_task(
            task, 
            phase, 
            timeout=timeout, 
            stream=bool(self.streaming_orchestrator), 
            stream_callback=llm_stream_callback if self.streaming_orchestrator else None
        )
        
        # Record execution time
        agent_execution_time = time.time() - agent_start_time
        
        # Check for timeout: empty response AND execution time close to timeout limit
        is_timeout = (
            (not response.text or len(response.text.strip()) == 0) and 
            agent_execution_time >= (timeout - 5)  # Within 5 seconds of timeout
        )
        
        if is_timeout:
            # Handle timeout case - show timeout message in streaming display
            if self.streaming_orchestrator:
                asyncio.run(self.streaming_orchestrator.stream_agent_output(
                    agent.agent_id, f"\n{'─' * 40}\n⏰ TIMEOUT after {timeout}s\n💥 Agent terminated due to timeout\n{'─' * 40}\n"
                ))
                self.streaming_orchestrator.finalize_agent_message(agent.agent_id)
                self.streaming_orchestrator.update_agent_status(agent.agent_id, "timeout")
                # Add system message for timeout
                self.streaming_orchestrator.add_system_message(f"⏰ Agent {agent.agent_id} TIMED OUT after {timeout}s")
            
            # Mark agent as failed due to timeout
            agent.mark_failed(f"Timeout after {timeout}s")
            print(f"⏰ SYSTEM: Agent {agent.agent_id} TIMED OUT after {timeout}s")
            logger.warning(f"Agent {agent.agent_id} timed out after {agent_execution_time:.2f}s")
            
            # Update orchestration system with execution metrics
            if phase == "initial":
                agent.state.execution_end_time = time.time()
                total_agent_time = agent.state.execution_time or agent_execution_time
                self.orchestration_system.record_agent_execution_time(agent.agent_id, total_agent_time)
            
            return response  # Return early for timeout case
        
        # Normal case - finalize streaming and handle orchestration
        if self.streaming_orchestrator:
            self.streaming_orchestrator.finalize_agent_message(agent.agent_id)
        
        # Update orchestration system with execution metrics
        if phase == "initial":
            agent.state.execution_end_time = time.time()
            total_agent_time = agent.state.execution_time or agent_execution_time
            self.orchestration_system.record_agent_execution_time(agent.agent_id, total_agent_time)
        
        # Handle post-phase orchestration
        self._handle_post_phase_orchestration(agent, response, phase)
        
        # Update streaming display - agent completed
        if self.streaming_orchestrator:
            asyncio.run(self.streaming_orchestrator.stream_agent_output(
                agent.agent_id, f"{'=' * 40}\n✅ Completed {phase} phase ({agent_execution_time:.2f}s)\n{'=' * 40}\n"
            ))
        
        print(f"✅ SYSTEM: Agent {agent.agent_id} completed {phase} phase ({agent_execution_time:.2f}s)")
        logger.debug(f"Agent {agent.agent_id} completed {phase} phase successfully in {agent_execution_time:.2f}s")
        
        return response
    
    def _handle_post_phase_orchestration(self, agent: MassAgent, response: AgentResponse, phase: str):
        """
        Handle orchestration functions after an agent completes a phase.
        
        Args:
            agent: The agent that completed the phase
            response: The agent's response
            phase: The phase that was completed
        """
        print(f"🔧 SYSTEM: Processing Agent {agent.agent_id} orchestration ({phase} phase):")
        
        # Show the actual response content for better visibility
        response_preview = response.text[:200] + "..." if len(response.text) > 200 else response.text
        print(f"   📝 Response preview: {response_preview}")
        
        # Don't stream the full response content again - it was already streamed live
        # The streaming already happened during the agent execution
        
        # Check if agent is already marked as failed (e.g., due to timeout)
        if self.orchestration_system.agent_states[agent.agent_id].status == "failed":
            print(f"⚠️  SYSTEM: Agent {agent.agent_id} already marked as failed - skipping orchestration")
            return  # Exit early for already failed agents
        
        # Extract summary report first
        summary_report = self._extract_summary_report(response.text)
        
        # Check if the summary is effectively empty and mark agent as failed if so
        if not summary_report or len(summary_report.strip()) == 0:
            agent.mark_failed("Empty summary report")
            print(f"⚠️  SYSTEM: Agent {agent.agent_id} FAILED - Empty summary report")
            logger.warning(f"Agent {agent.agent_id} failed due to empty summary report")
            return  # Exit early for failed agents
        
        # Check for extremely short summaries that might indicate failure
        # Allow short responses for simple questions, but catch obvious failures
        if len(summary_report.strip()) < 5:  # Less than 5 characters (very likely a failure)
            agent.mark_failed("Summary too short (likely model error)")
            print(f"⚠️  SYSTEM: Agent {agent.agent_id} FAILED - Summary too short ({len(summary_report.strip())} chars)")
            logger.warning(f"Agent {agent.agent_id} failed due to very short summary ({len(summary_report.strip())} chars)")
            return  # Exit early for failed agents
        
        if phase == "initial":
            # Phase 1: Only extract summary, NO answer extraction
            agent.update_summary(summary_report)  # No final_answer parameter
            print(f"ℹ️  SYSTEM: Updated summary for Agent {agent.agent_id} ({len(summary_report)} chars)")
            logger.debug(f"Updated summary for agent {agent.agent_id} after initial phase")
            
        elif phase == "collaboration":
            # Phase 2: Check for voting, extract summary if continuing, NO answer extraction
            vote_target = self._extract_vote_intention(agent.agent_id, response.text)
            # Make sure vote_target is an integer and is within the range of agent IDs
            if not isinstance(vote_target, int) or vote_target < 0 or vote_target >= len(self.orchestration_system.agents):
                print(f"   ⚠️  Invalid vote target: {vote_target} - must be an integer between 0 and {len(self.orchestration_system.agents)-1}")
                vote_target = None
                    
            if vote_target is not None:
                # Agent prefers another agent's solution
                agent.vote(vote_target, response.text)
                print(f"🗳️  Agent {agent.agent_id} VOTED for Agent {vote_target}")
                print(f"💭 Voting reason detected in response")
                logger.info(f"Agent {agent.agent_id} voted for agent {vote_target}")
                
                # Update streaming display for voting
                if self.streaming_orchestrator:
                    asyncio.run(self.streaming_orchestrator.stream_agent_output(
                        agent.agent_id, f"{'─' * 30}\n🗳️  VOTED for Agent {vote_target}\n{'─' * 30}\n"
                    ))
                    self.streaming_orchestrator.update_voting_status(agent.agent_id, vote_target)
                    self.streaming_orchestrator.update_agent_status(agent.agent_id, "voted")
                    # Send to streaming orchestrator for persistent display
                    self.streaming_orchestrator.add_system_message(f"🗳️  Agent {agent.agent_id} VOTED for Agent {vote_target}")
            else:
                # Agent updated their own solution - extract summary only, NO answer
                agent.update_summary(summary_report)  # No final_answer parameter
                print(f"📝 Agent {agent.agent_id} updated their solution")
                print(f"📊 Updated summary length: {len(summary_report)} characters")
                print(f"🔄 Continuing to work (no vote cast)")
                logger.debug(f"Updated summary for agent {agent.agent_id} after {phase} phase")
                
                # Update streaming display for summary update
                if self.streaming_orchestrator:
                    asyncio.run(self.streaming_orchestrator.stream_agent_output(
                        agent.agent_id, f"Updated solution ({len(summary_report)} chars)\n"
                    ))
                
        elif phase == "debate":
            # Phase 3: Check for voting, extract summary AND final answer if continuing
            vote_target = self._extract_vote_intention(agent.agent_id, response.text)
            # Make sure vote_target is an integer and is within the range of agent IDs
            if not isinstance(vote_target, int) or vote_target < 0 or vote_target >= len(self.orchestration_system.agents):
                print(f"   ⚠️  Invalid vote target: {vote_target} - must be an integer between 0 and {len(self.orchestration_system.agents)-1}")
                vote_target = None
                
            if vote_target is not None:
                # Agent prefers another agent's solution
                agent.vote(vote_target, response.text)
                print(f"🗳️  Agent {agent.agent_id} VOTED for Agent {vote_target}")
                print(f"💭 Voting reason detected in response")
                logger.info(f"Agent {agent.agent_id} voted for agent {vote_target}")
                
                # Update streaming display for voting
                if self.streaming_orchestrator:
                    asyncio.run(self.streaming_orchestrator.stream_agent_output(
                        agent.agent_id, f"{'─' * 30}\n🗳️  VOTED for Agent {vote_target}\n{'─' * 30}\n"
                    ))
                    self.streaming_orchestrator.update_voting_status(agent.agent_id, vote_target)
                    self.streaming_orchestrator.update_agent_status(agent.agent_id, "voted")
                    # Send to streaming orchestrator for persistent display
                    self.streaming_orchestrator.add_system_message(f"🗳️  Agent {agent.agent_id} VOTED for Agent {vote_target}")
            else:
                # Agent updated their own solution - extract summary AND final answer
                final_answer = self._extract_answer(response.text)  # Only in debate phase!
                agent.update_summary(summary_report, final_answer)
                print(f"   📝 Agent {agent.agent_id} updated their solution")
                print(f"   📊 Updated summary length: {len(summary_report)} characters")
                if final_answer:
                    print(f"   🎯 Extracted final answer length: {len(final_answer)} characters")
                print(f"   🔄 Continuing to work (no vote cast)")
                logger.debug(f"Updated summary for agent {agent.agent_id} after {phase} phase")
                
                # Update streaming display for summary update
                if self.streaming_orchestrator:
                    answer_info = f" + answer ({len(final_answer)} chars)" if final_answer else ""
                    asyncio.run(self.streaming_orchestrator.stream_agent_output(
                        agent.agent_id, f"Updated solution ({len(summary_report)} chars{answer_info})\n"
                    ))
                
        elif phase == "presentation":
            # Phase 4: Extract final summary report and answer (no voting in presentation phase)
            final_answer = self._extract_answer(response.text)
            agent.update_summary(summary_report, final_answer)
            print(f"   📝 Agent {agent.agent_id} provided final presentation")
            print(f"   📊 Final summary length: {len(summary_report)} characters")
            if final_answer:
                print(f"   🎯 Final answer length: {len(final_answer)} characters")
            else:
                print(f"   ⚠️  No final answer extracted from presentation")
            logger.info(f"Agent {agent.agent_id} completed final presentation with summary and answer")
            
            # Update streaming display for presentation
            if self.streaming_orchestrator:
                answer_info = f" + answer ({len(final_answer)} chars)" if final_answer else ""
                asyncio.run(self.streaming_orchestrator.stream_agent_output(
                    agent.agent_id, f"🎉 SYSTEM: Final presentation ({len(summary_report)} chars{answer_info})\n"
                ))
                self.streaming_orchestrator.update_agent_status(agent.agent_id, "completed_presentation")
    
    def _extract_vote_intention(self, agent_id: int, response_text: str) -> Optional[int]:
        """
        Analyze agent response to determine if they're voting for another agent.
        
        Args:
            agent_id: ID of the responding agent
            response_text: The agent's response text
            
        Returns:
            Agent ID to vote for, or None if agent is updating their own solution
        """
        import re
        import json
        
        print(f"   🔍 Analyzing vote intention for Agent {agent_id}")
        print(f"   📝 Response text length: {len(response_text)} characters")
        
        def extract_agent_id_from_value(vote_value: str) -> Optional[int]:
            """Extract numeric agent ID from various vote value formats."""
            vote_value = vote_value.strip()
            
            # Check for null/none values first
            if vote_value.lower() in ["none", "null", "nil", ""]:
                return None
            
            # Try direct integer conversion first
            try:
                return int(vote_value)
            except ValueError:
                pass
            
            # Extract number from "Agent X", "agent X", etc.
            agent_patterns = [
                r"[Aa]gent\s+(\d+)",          # "Agent 1", "agent 1"
                r"(\d+)",                     # Any number in the string
                r"[Aa]gent[\s_-]*(\d+)",      # "Agent_1", "agent-1", etc.
            ]
            
            for pattern in agent_patterns:
                match = re.search(pattern, vote_value)
                if match:
                    try:
                        return int(match.group(1))
                    except (ValueError, IndexError):
                        continue
            
            return None
        
        # Look for the voting format from mass_agent.py prompt: "### Decision\n{"voting": agent_id}"
        # More robust patterns that handle various formatting variations including spaces and text
        voting_patterns = [
            # Primary format: ### Voting Decision\n{"voting": "value"}
            r"###\s*Voting\s+Decision\s*[\n\r]+\s*\{\s*[\"']?voting[\"']?\s*:\s*[\"']?([^\"'}]+?)[\"']?\s*\}",
            # Alternative: ### Decision\n{"voting": "value"}  
            r"###\s*Decision\s*[\n\r]+\s*\{\s*[\"']?voting[\"']?\s*:\s*[\"']?([^\"'}]+?)[\"']?\s*\}",
            # With extra whitespace/content before JSON
            r"###\s*(?:Voting\s+)?Decision\s*[\n\r]+[^\{]*\{\s*[\"']?voting[\"']?\s*:\s*[\"']?([^\"'}]+?)[\"']?\s*\}",
            # JSON-like but potentially malformed
            r"###\s*(?:Voting\s+)?Decision\s*[\n\r]+.*?[\"']?voting[\"']?\s*:\s*[\"']?([^\"'}]+?)[\"']?",
            # More flexible pattern for any JSON-like voting structure
            r"\{\s*[\"']?voting[\"']?\s*:\s*[\"']?([^\"'}]+?)[\"']?\s*\}",
        ]
        
        has_voting_text = "voting" in response_text.lower() or "decision" in response_text.lower()
        
        for i, pattern in enumerate(voting_patterns):
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                vote_value = match.group(1).strip()
                
                target_agent_id = extract_agent_id_from_value(vote_value)
                
                if target_agent_id is None:
                    return None
                else:
                    return target_agent_id
        
        
        # Legacy format support: "### Voting\nAgent [id]"
        legacy_voting_pattern = r"###\s*Voting\s*[\n\r]+\s*[Aa]gent\s+(\d+)"
        match = re.search(legacy_voting_pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                target_agent_id = int(match.group(1))
                print(f"   🔍 Found legacy voting pattern: Agent {target_agent_id}")
                return target_agent_id
            except ValueError:
                pass
        
        # Fallback to heuristic patterns for backward compatibility
        old_patterns = [
            r"[Aa]gent (\d+) (?:has|provides|offers) (?:the )?(?:best|better|superior|correct) (?:solution|answer|response)",
            r"[Ii] (?:prefer|choose|select|support) [Aa]gent (\d+)",
            r"[Aa]gent (\d+)'s (?:solution|answer|response) is (?:better|best|superior|correct)",
        ]
        
        for pattern in old_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                try:
                    target_agent_id = int(match)
                    if target_agent_id != agent_id:  # Don't vote for self with old patterns
                        print(f"   🔍 Found heuristic voting pattern: Agent {target_agent_id}")
                        return target_agent_id
                except ValueError:
                    continue
        
        print(f"   ✅ No vote detected - Agent {agent_id} continues working")
        return None
    
    def _extract_summary_report(self, response_text: str) -> str:
        """
        Extract summary report from agent response using the structured format.
        
        Args:
            response_text: The agent's response text
            
        Returns:
            Extracted summary report or the full response if no structured format found
        """
        import re
        
        # Multiple patterns to handle various formatting variations
        summary_patterns = [
            # Primary README format: "### Summary report\n[content]" (stops at ## sections)
            r"###\s*Summary\s+report\s*[\n\r]+(.*?)(?=##[^#]|\Z)",
            # Legacy format: "### Summary Report\n[content]" (uppercase R)
            r"###\s*Summary\s+Report\s*[\n\r]+(.*?)(?=##[^#]|\Z)",
            # More permissive: any ## sections
            r"###\s*Summary\s+[Rr]eport\s*[\n\r]+(.*?)(?=###|\Z)",
            # Very permissive: just look for content after Summary
            r"###\s*Summary\s*[\n\r]+(.*?)(?=###|\Z)",
        ]
        
        for i, pattern in enumerate(summary_patterns):
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                summary = match.group(1).strip()
                if summary:
                    print(f"   📝 Extracted summary using pattern {i+1}: {len(summary)} chars")
                    return summary
        
        # Fallback: return the full response if no structured format found
        print(f"   ⚠️  No summary pattern found, using full response: {len(response_text)} chars")
        return response_text 
    
    def _extract_answer(self, response_text: str) -> str:
        """
        Extract final answer from agent response using the structured format.
        
        Args:
            response_text: The agent's response text
            
        Returns:
            Extracted final answer or empty string if no structured format found
        """
        import re
        
        # Multiple patterns to handle various answer formatting variations
        answer_patterns = [
            # Primary format from mass_agent.py presentation phase: "### Final answer\n[content]"
            r"###\s*Final\s+answer\s*[\n\r]+(.*?)(?=###|\Z)",
            # Alternative with capital A: "### Final Answer\n[content]"
            r"###\s*Final\s+Answer\s*[\n\r]+(.*?)(?=###|\Z)",
            # Legacy format: "## Final Answer\n[content]"
            r"##\s*Final\s+Answer\s*[\n\r]+(.*?)(?=##|\Z)",
            # Generic: "### Answer\n[content]"
            r"###\s*Answer\s*[\n\r]+(.*?)(?=###|\Z)",
            # More permissive: any format with "Answer" (case-insensitive)
            r"#+\s*[Ff]inal\s+[Aa]nswer\s*[\n\r]+(.*?)(?=#|\Z)",
            r"#+\s*[Aa]nswer\s*[\n\r]+(.*?)(?=#|\Z)",
        ]
        
        for i, pattern in enumerate(answer_patterns):
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if answer:
                    print(f"   🎯 Extracted final answer using pattern {i+1}: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
                    return answer
        
        # Return empty string if no structured answer format found
        print(f"   ⚠️  No final answer pattern found in response")
        return ""