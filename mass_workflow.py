"""
MASS Workflow Manager

This module handles the orchestration of the three-phase workflow:
1. Initial Processing - Agents work independently
2. Collaboration & Refinement - Agents review each other's work
3. Consensus & Finalization - Agents reach consensus on the best solution
"""

import asyncio
import threading
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from mass_agent import MassAgent, TaskInput, AgentResponse
from mass_coordination import MassCoordinationSystem

# Set up logging
logger = logging.getLogger(__name__)

class MassWorkflowManager:
    """
    Orchestrates the complete MASS workflow across multiple agents.
    
    This manager handles the execution of all three phases, ensures proper
    coordination between agents, and manages the overall workflow state.
    """
    
    def __init__(self, coordination_system: MassCoordinationSystem, 
                 max_phase_duration: int = 300,  # 5 minutes per phase
                 parallel_execution: bool = True,
                 check_update_frequency: float = 3.0):  # 3 seconds default as per README
        """
        Initialize the workflow manager.
        
        Args:
            coordination_system: The coordination system managing agents
            max_phase_duration: Maximum time allowed per phase in seconds
            parallel_execution: Whether to run agents in parallel
            check_update_frequency: How often agents check for updates in seconds (default 3)
        """
        self.coordination_system = coordination_system
        self.max_phase_duration = max_phase_duration
        self.parallel_execution = parallel_execution
        self.check_update_frequency = check_update_frequency
        
        # Workflow state
        self.current_phase = "initial"
        self.phase_results: Dict[str, List[Dict[str, Any]]] = {
            "initial": [],
            "collaboration": [],
            "consensus": []
        }
        
        # Callbacks for phase completion
        self.phase_callbacks: Dict[str, List[Callable]] = {
            "initial": [],
            "collaboration": [],
            "consensus": []
        }
    
    def add_phase_callback(self, phase: str, callback: Callable):
        """Add a callback to be executed when a phase completes."""
        if phase in self.phase_callbacks:
            self.phase_callbacks[phase].append(callback)
    
    def run_complete_workflow(self, task: TaskInput, 
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the complete three-phase MASS workflow.
        Guarantees returning a final solution regardless of agent failures or other issues.
        
        Args:
            task: The task to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing comprehensive workflow results with guaranteed final solution
        """
        workflow_start_time = time.time()
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE MASS WORKFLOW")
        logger.info("=" * 80)
        
        try:
            # Set expected answer if available in task context
            if task.context and "answer" in task.context:
                self.coordination_system.set_expected_answer(task.context["answer"])
                logger.info(f"Expected answer set: {task.context['answer']}")
            
            # Phase 1: Initial Processing
            print("\n" + "🔵"*20)
            print("📝 PHASE 1: INITIAL PROCESSING")
            print("🔵"*20)
            print("🎯 Goal: Each agent works independently on the task")
            print("🛠️  Tools: live_search, code_execution")
            logger.info("Phase 1: Initial Processing")
            
            self.coordination_system.start_phase_timing("initial")
            self.current_phase = "initial"
            initial_results = self._run_phase_1_initial_processing(task, progress_callback)
            self.coordination_system.end_phase_timing("initial")
            self.phase_results["initial"] = initial_results
            
            # Execute phase callbacks
            for callback in self.phase_callbacks["initial"]:
                callback(initial_results)
            
            successful_initial = len([r for r in initial_results if r['success']])
            print(f"\n✅ Phase 1 Complete: {successful_initial}/{len(initial_results)} agents successful")
            
            # Check if we have any working solutions - if not, force completion
            if successful_initial == 0:
                logger.warning("No agents succeeded in initial phase - forcing basic completion")
                print("⚠️  WARNING: No successful initial solutions - using emergency completion")
                return self._emergency_workflow_completion(task, workflow_start_time)
            
            # Phase 2: Collaboration & Refinement
            print("\n" + "🟡"*20)
            print("🤝 PHASE 2: COLLABORATION & REFINEMENT")
            print("🟡"*20)
            print("🎯 Goal: Agents review peer solutions and collaborate")
            print("🔄 Process: Multiple rounds until consensus or voting complete")
            logger.info("Phase 2: Collaboration & Refinement")
            
            self.coordination_system.start_phase_timing("collaboration")
            self.current_phase = "collaboration"
            collaboration_results = self._run_phase_2_collaboration(task, progress_callback)
            self.coordination_system.end_phase_timing("collaboration")
            self.phase_results["collaboration"] = collaboration_results
            
            # Execute phase callbacks
            for callback in self.phase_callbacks["collaboration"]:
                callback(collaboration_results)
            
            # Phase 3: Consensus & Finalization
            print("\n" + "🟢"*20)
            print("🎉 PHASE 3: CONSENSUS & FINALIZATION")
            print("🟢"*20)
            print("🎯 Goal: Final consensus building and solution selection")
            logger.info("Phase 3: Consensus & Finalization")
            
            self.coordination_system.start_phase_timing("consensus")
            self.current_phase = "consensus"
            consensus_results = self._run_phase_3_consensus(task, progress_callback)
            self.coordination_system.end_phase_timing("consensus")
            self.phase_results["consensus"] = consensus_results
            
            # Execute phase callbacks
            for callback in self.phase_callbacks["consensus"]:
                callback(consensus_results)
            
            # GUARANTEE: Get final solution (this will force consensus if needed)
            final_solution = self.coordination_system.get_final_solution()
            
            if final_solution is None:
                # This should never happen now, but extra safety
                logger.error("CRITICAL: get_final_solution returned None - using emergency fallback")
                return self._emergency_workflow_completion(task, workflow_start_time)
            
        except Exception as e:
            logger.error(f"Workflow failed with exception: {str(e)}")
            logger.error("Using emergency completion to ensure result is returned")
            print(f"❌ WORKFLOW ERROR: {str(e)}")
            print("🔧 Using emergency completion")
            return self._emergency_workflow_completion(task, workflow_start_time)
        
        # Calculate final metrics
        total_workflow_time = time.time() - workflow_start_time
        
        # Build comprehensive results
        results = {
            "success": True,
            "total_workflow_time": total_workflow_time,
            "final_solution": final_solution,
            "phase_results": self.phase_results,
            "system_status": self.coordination_system.get_system_status(),
            "session_log": self.coordination_system.export_detailed_session_log()
        }
        
        # Log completion
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {total_workflow_time:.2f} seconds")
        logger.info(f"Winning solution: Agent {final_solution['agent_id']}")
        logger.info(f"Consensus method: {final_solution.get('consensus_method', 'natural')}")
        logger.info("=" * 80)
        
        return results
    
    def _emergency_workflow_completion(self, task: TaskInput, workflow_start_time: float) -> Dict[str, Any]:
        """
        Emergency completion when normal workflow fails.
        Ensures we always return a result.
        """
        logger.warning("EMERGENCY WORKFLOW COMPLETION")
        print("🚨 EMERGENCY COMPLETION: Ensuring a result is returned")
        
        # Try to force a solution from coordination system
        try:
            self.coordination_system._force_consensus()
            final_solution = self.coordination_system.get_final_solution()
        except Exception as e:
            logger.error(f"Emergency consensus failed: {e}")
            # Create a basic solution manually
            final_solution = {
                "agent_id": 0,
                "solution": "Emergency fallback: Unable to complete normal workflow",
                "extracted_answer": "No answer available",
                "execution_time": 0,
                "total_rounds": 0,
                "vote_distribution": {},
                "all_agent_summaries": {},
                "is_correct": False,
                "expected_answer": task.context.get("answer") if task.context else None,
                "total_runtime": time.time() - workflow_start_time,
                "consensus_method": "emergency"
            }
        
        return {
            "success": False,  # Mark as unsuccessful due to emergency completion
            "total_workflow_time": time.time() - workflow_start_time,
            "final_solution": final_solution,
            "phase_results": self.phase_results,
            "system_status": self.coordination_system.get_system_status(),
            "session_log": self.coordination_system.export_detailed_session_log(),
            "emergency_completion": True,
            "error_message": "Workflow completed using emergency fallback"
        }
    
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
        agents = list(self.coordination_system.agents.values())
        results = []
        
        if self.parallel_execution:
            # Run agents in parallel
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                future_to_agent = {
                    executor.submit(self._run_agent_phase, agent, task, "initial"): agent 
                    for agent in agents
                }
                
                completed_count = 0
                for future in as_completed(future_to_agent, timeout=self.max_phase_duration):
                    agent = future_to_agent[future]
                    try:
                        result = future.result()
                        results.append({
                            "agent_id": agent.agent_id,
                            "phase": "initial",
                            "result": result,
                            "success": True,
                            "error": None
                        })
                        completed_count += 1
                        print(f"✅ Agent {agent.agent_id} completed initial processing ({completed_count}/{len(agents)})")
                        
                        if progress_callback:
                            progress_callback("initial", completed_count, len(agents))
                            
                    except Exception as e:
                        logger.error(f"Agent {agent.agent_id} failed in initial phase: {str(e)}")
                        results.append({
                            "agent_id": agent.agent_id,
                            "phase": "initial",
                            "result": None,
                            "success": False,
                            "error": str(e)
                        })
        else:
            # Run agents sequentially
            for i, agent in enumerate(agents):
                try:
                    result = self._run_agent_phase(agent, task, "initial")
                    results.append({
                        "agent_id": agent.agent_id,
                        "phase": "initial",
                        "result": result,
                        "success": True,
                        "error": None
                    })
                    
                    if progress_callback:
                        progress_callback("initial", i + 1, len(agents))
                        
                except Exception as e:
                    logger.error(f"Agent {agent.agent_id} failed in initial phase: {str(e)}")
                    results.append({
                        "agent_id": agent.agent_id,
                        "phase": "initial",
                        "result": None,
                        "success": False,
                        "error": str(e)
                    })
        
        return results
    
    def _run_phase_2_collaboration(self, task: TaskInput, 
                                 progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Phase 2: Agents review each other's work and either improve or vote.
        
        This phase implements continuous update checking every check_update_frequency seconds
        and agent reactivation when new updates are available, as specified in README.
        Includes safety mechanisms to ensure progress toward finalization.
        """
        results = []
        max_collaboration_rounds = 5
        phase_start_time = time.time()
        last_update_check = phase_start_time
        
        print(f"\n🔄 Collaboration Phase with continuous update checking (every {self.check_update_frequency}s)")
        print("-" * 70)
        logger.info(f"Collaboration phase with continuous update checking every {self.check_update_frequency} seconds")
        
        for round_num in range(max_collaboration_rounds):
            print(f"\n🔄 Collaboration Round {round_num + 1}/{max_collaboration_rounds}")
            print("-" * 50)
            logger.info(f"Collaboration round {round_num + 1}")
            
            # Continuous update checking mechanism as per README
            round_start_time = time.time()
            agents_processed_this_round = set()
            round_timeout = 60  # Maximum time per round to prevent infinite loops
            
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
                    reactivation_candidates = self.coordination_system.check_for_reactivation()
                    if reactivation_candidates:
                        print(f"   🔄 Reactivating {len(reactivation_candidates)} agents due to new updates:")
                        for agent_id in reactivation_candidates:
                            self.coordination_system.reactivate_agent(agent_id)
                            print(f"      - Agent {agent_id} reactivated")
                            # Remove from processed set so they can be processed again
                            agents_processed_this_round.discard(agent_id)
                    
                    last_update_check = current_time
                
                # Get agents that are still working and haven't been processed this round
                working_agents = [
                    agent for agent in self.coordination_system.agents.values()
                    if (self.coordination_system.agent_states[agent.agent_id].status == "working" 
                        and agent.agent_id not in agents_processed_this_round)
                ]
                
                # If no working agents left, move to next round
                if not working_agents:
                    break
                
                # Process agents that need to work
                round_results = []
                
                if self.parallel_execution and len(working_agents) > 1:
                    # Run working agents in parallel
                    with ThreadPoolExecutor(max_workers=len(working_agents)) as executor:
                        future_to_agent = {
                            executor.submit(self._run_agent_phase, agent, task, "collaboration"): agent
                            for agent in working_agents
                        }
                        
                        for future in as_completed(future_to_agent):
                            agent = future_to_agent[future]
                            try:
                                result = future.result()
                                round_results.append({
                                    "agent_id": agent.agent_id,
                                    "phase": "collaboration",
                                    "round": round_num + 1,
                                    "result": result,
                                    "success": True,
                                    "error": None
                                })
                                agents_processed_this_round.add(agent.agent_id)
                            except Exception as e:
                                logger.error(f"Agent {agent.agent_id} failed in collaboration round {round_num + 1}: {str(e)}")
                                round_results.append({
                                    "agent_id": agent.agent_id,
                                    "phase": "collaboration",
                                    "round": round_num + 1,
                                    "result": None,
                                    "success": False,
                                    "error": str(e)
                                })
                                agents_processed_this_round.add(agent.agent_id)
                
                else:
                    # Run agents sequentially
                    for agent in working_agents:
                        try:
                            result = self._run_agent_phase(agent, task, "collaboration")
                            round_results.append({
                                "agent_id": agent.agent_id,
                                "phase": "collaboration",
                                "round": round_num + 1,
                                "result": result,
                                "success": True,
                                "error": None
                            })
                            agents_processed_this_round.add(agent.agent_id)
                        except Exception as e:
                            logger.error(f"Agent {agent.agent_id} failed in collaboration round {round_num + 1}: {str(e)}")
                            round_results.append({
                                "agent_id": agent.agent_id,
                                "phase": "collaboration",
                                "round": round_num + 1,
                                "result": None,
                                "success": False,
                                "error": str(e)
                            })
                            agents_processed_this_round.add(agent.agent_id)
                
                results.extend(round_results)
                
                # Check if consensus is reached after processing agents
                if self.coordination_system.system_state.consensus_reached:
                    logger.info("Consensus reached during collaboration phase")
                    print(f"   ✅ Consensus reached! Ending collaboration phase.")
                    return results
                
                # Brief pause before next update check
                time.sleep(min(1, self.check_update_frequency / 2))
            
            # Show status at end of round
            voted_agents = [
                agent for agent in self.coordination_system.agents.values()
                if self.coordination_system.agent_states[agent.agent_id].status == "voted"
            ]
            working_agents = [
                agent for agent in self.coordination_system.agents.values()
                if self.coordination_system.agent_states[agent.agent_id].status == "working"
            ]
            
            print(f"\n📊 End of Round {round_num + 1}:")
            print(f"   👥 Working agents: {[a.agent_id for a in working_agents]}")
            print(f"   🗳️  Voted agents: {[a.agent_id for a in voted_agents]}")
            
            if progress_callback:
                progress_callback("collaboration", round_num + 1, max_collaboration_rounds)
            
            # Check if consensus is reached
            if self.coordination_system.system_state.consensus_reached:
                logger.info("Consensus reached during collaboration phase")
                break
            
            # If no working agents left, end collaboration
            if not working_agents:
                print(f"   ✅ All agents have voted. Ending collaboration phase.")
                break
            
            # Safety mechanism: If we're on the last round and still have working agents,
            # encourage them to vote by updating round counter for consensus logic
            if round_num == max_collaboration_rounds - 1 and working_agents:
                logger.warning(f"Final collaboration round - updating system round counter to trigger consensus")
                self.coordination_system.system_state.total_rounds = self.coordination_system.max_rounds
                print(f"   ⚠️  Final round: System will force consensus on remaining agents")
        
        # Final safety check: If we still don't have consensus, ensure we get one
        if not self.coordination_system.system_state.consensus_reached:
            logger.warning("Collaboration ended without consensus - will be handled in consensus phase")
            print(f"   ⚠️  Collaboration ended without consensus - proceeding to consensus phase")
        
        return results
    
    def _run_phase_3_consensus(self, task: TaskInput, 
                             progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Phase 3: Final consensus building if not already reached.
        
        This phase handles any remaining consensus building and finalization.
        Includes safety mechanisms to guarantee a final decision is reached.
        """
        results = []
        
        # If consensus is already reached, just return
        if self.coordination_system.system_state.consensus_reached:
            logger.info("Consensus already reached. Finalizing.")
            if progress_callback:
                progress_callback("consensus", 1, 1)
            return results
        
        logger.info("Consensus not yet reached - running final consensus phase")
        print("🎯 Consensus not reached - running final decision phase")
        
        # Run any agents that might need to reconsider their votes
        working_agents = [
            agent for agent in self.coordination_system.agents.values()
            if self.coordination_system.agent_states[agent.agent_id].status == "working"
        ]
        
        if working_agents:
            logger.info(f"Running final consensus phase with {len(working_agents)} remaining agents")
            print(f"   👥 Processing {len(working_agents)} remaining working agents")
            
            # Force final round to trigger consensus mechanisms
            self.coordination_system.system_state.total_rounds = self.coordination_system.max_rounds
            
            for i, agent in enumerate(working_agents):
                try:
                    result = self._run_agent_phase(agent, task, "consensus")
                    results.append({
                        "agent_id": agent.agent_id,
                        "phase": "consensus",
                        "result": result,
                        "success": True,
                        "error": None
                    })
                except Exception as e:
                    logger.error(f"Agent {agent.agent_id} failed in consensus phase: {str(e)}")
                    results.append({
                        "agent_id": agent.agent_id,
                        "phase": "consensus",
                        "result": None,
                        "success": False,
                        "error": str(e)
                    })
                
                if progress_callback:
                    progress_callback("consensus", i + 1, len(working_agents))
                
                # Check if consensus reached after each agent
                if self.coordination_system.system_state.consensus_reached:
                    logger.info("Consensus reached during consensus phase")
                    print(f"   ✅ Consensus reached after processing Agent {agent.agent_id}")
                    break
        
        # GUARANTEE: Force consensus if still not reached
        if not self.coordination_system.system_state.consensus_reached:
            logger.warning("Still no consensus after consensus phase - forcing final decision")
            print("   🔧 No consensus reached - forcing final decision")
            
            # Force consensus using existing coordination system logic
            try:
                self.coordination_system._force_consensus()
                if self.coordination_system.system_state.consensus_reached:
                    logger.info("Successfully forced consensus")
                    print("   ✅ Successfully forced consensus")
                else:
                    logger.error("Failed to force consensus - critical system error")
                    print("   ❌ Failed to force consensus - system error")
            except Exception as e:
                logger.error(f"Error forcing consensus: {e}")
                print(f"   ❌ Error forcing consensus: {e}")
        
        return results
    
    def _run_agent_phase(self, agent: MassAgent, task: TaskInput, phase: str) -> AgentResponse:
        """
        Run a single agent for a specific phase with proper timing and coordination.
        
        Args:
            agent: The agent to run
            task: The task to process
            phase: The workflow phase
            
        Returns:
            AgentResponse from the agent
        """
        print(f"\n🤖 Running Agent {agent.agent_id} ({phase} phase):")
        logger.debug(f"Starting agent {agent.agent_id} for {phase} phase")
        
        # Set execution start time if this is the initial phase
        if phase == "initial" and agent.state.execution_start_time is None:
            agent.state.execution_start_time = time.time()
        
        agent_start_time = time.time()
        
        try:
            # Process the task using the agent's new method
            response = agent.process_task(task, phase)
            
            # Record execution time
            agent_execution_time = time.time() - agent_start_time
            
            # Update coordination system with execution metrics
            if phase == "initial":
                agent.state.execution_end_time = time.time()
                total_agent_time = agent.state.execution_time or agent_execution_time
                self.coordination_system.record_agent_execution_time(agent.agent_id, total_agent_time)
            
            # Handle post-phase coordination
            self._handle_post_phase_coordination(agent, response, phase)
            
            print(f"   ✅ Agent {agent.agent_id} completed {phase} phase ({agent_execution_time:.2f}s)")
            logger.debug(f"Agent {agent.agent_id} completed {phase} phase successfully in {agent_execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            print(f"   ❌ Agent {agent.agent_id} failed in {phase} phase: {str(e)}")
            logger.error(f"Agent {agent.agent_id} failed in {phase} phase: {str(e)}")
            raise
    
    def _handle_post_phase_coordination(self, agent: MassAgent, response: AgentResponse, phase: str):
        """
        Handle coordination functions after an agent completes a phase.
        
        Args:
            agent: The agent that completed the phase
            response: The agent's response
            phase: The phase that was completed
        """
        print(f"\n🔧 Processing Agent {agent.agent_id} coordination ({phase} phase):")
        
        if phase == "initial":
            # Extract and update summary and answer with agent's initial response
            summary_report = self._extract_summary_report(response.text)
            final_answer = self._extract_answer(response.text)
            agent.update_summary(summary_report, final_answer)
            print(f"   📝 Updated summary for Agent {agent.agent_id}")
            print(f"   📊 Summary length: {len(summary_report)} characters")
            if final_answer:
                print(f"   🎯 Extracted answer length: {len(final_answer)} characters")
            logger.debug(f"Updated summary for agent {agent.agent_id} after initial phase")
            
        elif phase in ["collaboration", "consensus"]:
            # Check if agent indicates preference for another agent's solution
            vote_target = self._extract_vote_intention(agent.agent_id, response.text)
            
            if vote_target is not None:
                # Agent prefers another agent's solution
                agent.vote(vote_target)
                print(f"   🗳️  Agent {agent.agent_id} VOTED for Agent {vote_target}")
                print(f"   💭 Voting reason detected in response")
                logger.info(f"Agent {agent.agent_id} voted for agent {vote_target}")
            else:
                # Agent updated their own solution - extract summary report and answer
                summary_report = self._extract_summary_report(response.text)
                final_answer = self._extract_answer(response.text)
                agent.update_summary(summary_report, final_answer)
                print(f"   📝 Agent {agent.agent_id} updated their solution")
                print(f"   📊 Updated summary length: {len(summary_report)} characters")
                if final_answer:
                    print(f"   🎯 Updated answer length: {len(final_answer)} characters")
                print(f"   🔄 Continuing to work (no vote cast)")
                logger.debug(f"Updated summary for agent {agent.agent_id} after {phase} phase")
    
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
        
        # Look for the new structured voting format: "### Voting\nAgent [id]"
        voting_pattern = r"###\s*Voting\s*\n\s*Agent\s+(\d+)"
        
        match = re.search(voting_pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                target_agent_id = int(match.group(1))
                # Allow voting for self or others
                return target_agent_id
            except ValueError:
                pass
        
        # Fallback to old heuristic patterns for backward compatibility
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
                        return target_agent_id
                except ValueError:
                    continue
        
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
        
        # Look for the new structured summary format: "### Summary Report\n[content]"
        summary_pattern = r"###\s*Summary\s+Report\s*\n(.*?)(?=###|\Z)"
        
        match = re.search(summary_pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            summary = match.group(1).strip()
            if summary:
                return summary
        
        # Fallback: return the full response if no structured format found
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
        
        # Look for the structured answer format: "### Answer\n[content]"
        answer_pattern = r"###\s*Answer\s*\n(.*?)(?=###|\Z)"
        
        match = re.search(answer_pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer
        
        # Return empty string if no structured answer format found
        return ""