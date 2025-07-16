from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import time

@dataclass
class AgentState:
    """Represents the current state of an agent in the MASS system."""
    agent_id: int
    status: str = "working"  # "working" or "voted"
    working_summary: str = ""
    final_answer: str = ""  # The final answer extracted from agent response
    execution_start_time: Optional[float] = None
    execution_end_time: Optional[float] = None
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    vote_target: Optional[int] = None  # Which agent's solution this agent voted for
    seen_updates_timestamps: Dict[int, float] = field(default_factory=dict)  # agent_id -> last_seen_timestamp
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time if both start and end times are available."""
        if self.execution_start_time and self.execution_end_time:
            return self.execution_end_time - self.execution_start_time
        return None
    
    def add_update(self, summary: str, final_answer: str = "", timestamp: Optional[float] = None):
        """Add an update to the agent's history."""
        if timestamp is None:
            timestamp = time.time()
        
        self.update_history.append({
            "timestamp": timestamp,
            "summary": summary,
            "final_answer": final_answer,
            "status": self.status
        })
        self.working_summary = summary
        if final_answer:  # Only update if a final answer is provided
            self.final_answer = final_answer
    
    def mark_updates_seen(self, agent_updates: Dict[int, float]):
        """Mark updates from other agents as seen."""
        for agent_id, timestamp in agent_updates.items():
            if agent_id != self.agent_id:  # Don't track own updates
                self.seen_updates_timestamps[agent_id] = timestamp
    
    def has_unseen_updates(self, other_agent_updates: Dict[int, float]) -> bool:
        """Check if there are unseen updates from other agents."""
        for agent_id, latest_timestamp in other_agent_updates.items():
            if agent_id != self.agent_id:  # Don't check own updates
                last_seen = self.seen_updates_timestamps.get(agent_id, 0.0)
                if latest_timestamp > last_seen:
                    return True
        return False

@dataclass
class TaskInput:
    """Represents a task to be processed by the MASS system."""
    question: str
    context: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None

@dataclass
class AgentResponse:
    """Response from an agent's process_message function."""
    text: str
    code: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    function_calls: List[Dict[str, Any]] = field(default_factory=list)

class MassAgent(ABC):
    """
    Abstract base class for all agents in the MASS system.
    
    All agent implementations must inherit from this class and implement
    the required methods while following the standardized workflow.
    """
    
    def __init__(self, agent_id: int, coordination_system=None):
        """
        Initialize the agent with an ID and reference to the coordination system.
        
        Args:
            agent_id: Unique identifier for this agent
            coordination_system: Reference to the MassCoordinationSystem
        """
        self.agent_id = agent_id
        self.coordination_system = coordination_system
        self.state = AgentState(agent_id=agent_id)
    
    @abstractmethod
    def process_message(self, messages: List[Dict[str, str]], tools: List[str] = None, 
                       temperature: float = 0.5, **kwargs) -> AgentResponse:
        """
        Core LLM inference function for task processing.
        
        This method should handle the actual LLM interaction using the agent's
        specific backend (OpenAI, Gemini, Grok, etc.) and return a standardized response.
        
        Args:
            messages: List of messages in OpenAI format
            tools: List of tools available to the agent
            temperature: Sampling temperature for the LLM
            **kwargs: Additional parameters specific to the agent implementation
            
        Returns:
            AgentResponse containing the agent's response text, code, citations, etc.
        """
        pass
    
    def update_summary(self, summary_report: str, final_answer: str = ""):
        """
        Record working process and summary report for sharing with other agents.
        
        Args:
            summary_report: Comprehensive progress report including reasoning and findings
            final_answer: The final answer to the question (optional)
        """
        if self.coordination_system:
            self.coordination_system.update_agent_summary(self.agent_id, summary_report, final_answer)
        
        # Update local state
        self.state.add_update(summary_report, final_answer)
    
    def check_updates(self) -> Dict[str, Any]:
        """
        Check for updates from other agents working on the same task.
        Now tracks which updates have been seen to avoid processing duplicates.
        
        Returns:
            Dictionary containing updates from all agents, including their summaries and states
        """
        if self.coordination_system:
            # Get updates (only new ones by default)
            updates = self.coordination_system.get_all_updates(
                exclude_agent_id=self.agent_id, 
                check_new_only=True
            )
            
            # Mark the updates we've seen
            if updates.get("agents"):
                seen_updates = {}
                for agent_id, agent_info in updates["agents"].items():
                    if "latest_timestamp" in agent_info:
                        seen_updates[int(agent_id)] = agent_info["latest_timestamp"]
                
                if seen_updates:
                    self.coordination_system.mark_updates_seen_by_agent(self.agent_id, seen_updates)
            
            return updates
        return {}
    
    def check_all_updates(self) -> Dict[str, Any]:
        """
        Check for all updates from other agents (including previously seen ones).
        
        Returns:
            Dictionary containing all updates from all agents
        """
        if self.coordination_system:
            return self.coordination_system.get_all_updates(
                exclude_agent_id=self.agent_id, 
                check_new_only=False
            )
        return {}
    
    def vote(self, target_agent_id: int):
        """
        Vote for the best solution among all agents.
        
        Args:
            target_agent_id: ID of the agent whose solution this agent votes for
        """
        if self.coordination_system:
            self.coordination_system.cast_vote(self.agent_id, target_agent_id)
        
        # Update local state
        self.state.status = "voted"
        self.state.vote_target = target_agent_id
        self.state.execution_end_time = time.time()
    
    def get_workflow_instructions(self, phase: str) -> str:
        """
        Get phase-specific instructions for the agent workflow.
        
        Args:
            phase: Current workflow phase ("initial", "collaboration", "consensus")
            
        Returns:
            Detailed instructions for the specified phase
        """
        instructions = {
            "initial": f"""
You are Agent {self.agent_id} working with other agents to solve a task. 

Your goals:
1. Process the given task using your reasoning capabilities and available tools (search and code interpreter)
2. Provide a clear, and comprehensive summary report at the end of your response that will be shared with other agents

Response format:
```
### Summary Report
[Your summary report]

### Answer
[Your final answer to this question]
```

""",
            "collaboration": f"""
You are Agent {self.agent_id} working with other agents to solve a task.

You will be shown your own previous summary and other agents' working summaries below for reference. 

You should use your tools to verify the accuracy of these solutions and the final answers.

After reviewing all summaries, you have two options:

**Option 1: Update Your Solution**
If you want to continue working on finding a better solution, provide an updated summary report and conclude with:
```
### Summary Report
[Your updated summary report]

### Answer
[Your final answer to this question]
```

**Option 2: Vote for Best Solution**  
If you believe one of the existing solutions (including your own) is the correct answer, vote by concluding your response with:
```
### Voting
Agent [agent_id]
```

Guidelines:
- Remember: You are Agent {self.agent_id}
- Carefully review all solutions alongside the original task
- Check each step, statement, assumption, and conclusion of all provided solutions
- Look for accuracy, completeness, and quality when evaluating solutions
- When voting, specify the agent ID (including your own ID {self.agent_id} if you believe your solution is best)
"""
        }
        instructions["consensus"] = f"""
You are Agent {self.agent_id} working with other agents to solve a task.

You will be shown your own previous summary and other agents' working summaries below for reference.

This is the consensus phase. You should make a final decision on which solution is the best.

After reviewing all summaries, you have two options:

**Option 1: Update Your Solution**
If you want to provide a final improved solution, conclude with:
```
### Summary Report
[Your final summary report]

### Answer
[Your final answer to this question]
```

**Option 2: Vote for Best Solution**  
If you believe one of the existing solutions (including your own) is the correct answer, vote by concluding your response with:
```
### Voting
Agent [agent_id]
```

Guidelines:
- Remember: You are Agent {self.agent_id}
- This is your final opportunity to contribute to the solution
- Choose the most accurate and complete solution among all options
- When voting, specify the agent ID (including your own ID {self.agent_id} if you believe your solution is best)
"""
        
        return instructions.get(phase, "")
    
    def process_task(self, task: TaskInput, phase: str = "initial") -> AgentResponse:
        """
        Process a task in a specific workflow phase.
        Now leverages incremental update detection to avoid reprocessing seen updates.
        
        Args:
            task: The task to process
            phase: The workflow phase ("initial", "collaboration", "consensus")
            
        Returns:
            AgentResponse containing the agent's response
        """
        # Phase-specific coordination handled by workflow system
        if phase == "collaboration":
            # Get only new updates from other agents
            updates = self.check_updates()
            
            # Always include agent's own summary and ID context as required by README
            own_summary_context = f"\n\nYour ID: Agent {self.agent_id}"
            if self.state.working_summary:
                own_summary_context += f"\n\nYour previous summary:\n{self.state.working_summary}"
            
            if updates and updates.get("agents") and updates.get("has_new_updates"):
                # Add only new peer updates to the context for agent consideration
                peer_summaries = []
                new_updates_count = 0
                for agent_id, agent_info in updates["agents"].items():
                    if agent_info.get("working_summary") and agent_info.get("is_new_update", False):
                        peer_summaries.append(f"Agent {agent_id}: {agent_info['working_summary']}")
                        new_updates_count += 1
                
                if peer_summaries:
                    peer_context = f"\n\nNew peer agent solutions to consider ({new_updates_count} new updates):\n" + "\n".join(peer_summaries)
                    print(f"   📥 Agent {self.agent_id} received {new_updates_count} new updates from peers")
                else:
                    peer_context = ""
                    print(f"   📥 Agent {self.agent_id} received no new updates")
            else:
                peer_context = ""
                print(f"   📥 Agent {self.agent_id} received no new updates")
            
            # Combine own summary and peer context as required by README
            full_peer_context = own_summary_context + peer_context
            
        elif phase == "consensus":
            # In consensus phase, check all updates to make final decision
            updates = self.check_all_updates()
            
            # Include agent's own ID and summary for consensus phase too
            own_summary_context = f"\n\nYour ID: Agent {self.agent_id}"
            if self.state.working_summary:
                own_summary_context += f"\n\nYour previous summary:\n{self.state.working_summary}"
            
            if updates and updates.get("agents"):
                peer_summaries = []
                for agent_id, agent_info in updates["agents"].items():
                    if agent_info.get("working_summary"):
                        peer_summaries.append(f"Agent {agent_id}: {agent_info['working_summary']}")
                
                if peer_summaries:
                    peer_context = "\n\nAll peer agent solutions for final consideration:\n" + "\n".join(peer_summaries)
                else:
                    peer_context = ""
            else:
                peer_context = ""
            
            # Combine own summary and peer context
            full_peer_context = own_summary_context + peer_context
        else:
            # Initial phase - no peer context, but include agent ID
            full_peer_context = f"\n\nYour ID: Agent {self.agent_id}"
        
        # Get phase-specific instructions
        instructions = self.get_workflow_instructions(phase)
        
        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": task.question + full_peer_context}
        ]
        
        # Get available tools (only live_search and code_execution)
        tools = self._get_available_tools()
        
        # Process the message using the agent's specific implementation
        response = self.process_message(messages, tools=tools)
        
        return response
    
    def _get_available_tools(self) -> List[str]:
        """
        Get list of tools available to this agent.
        
        Note: Coordination tools (update_summary, check_updates, vote) are NOT 
        exposed as tools to agents. They are called programmatically by the 
        workflow system at appropriate times.
        """
        # Only live_search and code_execution are available as tools to agents
        return ["live_search", "code_execution"] 