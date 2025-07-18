"""
MASS Streaming Display System - Simple Working Version
"""

import os
import time
import threading
from typing import Dict, List, Optional, Callable

class MultiRegionDisplay:
    def __init__(self, display_enabled: bool = True):
        self.display_enabled = display_enabled
        self.agent_outputs: Dict[int, str] = {}
        self.system_messages: List[str] = []
        self.start_time = time.time()
        self._lock = threading.Lock()
        
    async def stream_output(self, agent_id: int, content: str):
        if not self.display_enabled:
            return
            
        with self._lock:
            if agent_id not in self.agent_outputs:
                self.agent_outputs[agent_id] = ""
            
            self.agent_outputs[agent_id] += content
            self._update_display()
    
    def finalize_message(self, agent_id: int):
        if not self.display_enabled:
            return
            
        with self._lock:
            if agent_id in self.agent_outputs:
                self.agent_outputs[agent_id] += "\n"
                self._update_display()
    
    def add_system_message(self, message: str):
        if not self.display_enabled:
            return
            
        with self._lock:
            self.system_messages.append(message)
            # Keep only the last 10 system messages to prevent overflow
            if len(self.system_messages) > 10:
                self.system_messages = self.system_messages[-10:]
            self._update_display()
    
    def _update_display(self):
        if not self.display_enabled:
            return
            
        os.system('clear' if os.name == 'posix' else 'cls')
        
        try:
            terminal_width = os.get_terminal_size().columns
        except:
            terminal_width = 120
            
        # Ensure minimum width and maximum practical width
        terminal_width = max(60, min(terminal_width, 200))
            
        agent_ids = sorted(self.agent_outputs.keys())
        if not agent_ids:
            return
        
        # Multi-column layout with better spacing
        num_agents = len(agent_ids)
        # Reserve space for separators and padding
        available_width = terminal_width - (num_agents - 1) * 3 - 4
        col_width = max(30, available_width // num_agents)
        
        # Split content into lines for each agent
        agent_lines = {}
        max_lines = 0
        for agent_id in agent_ids:
            lines = self.agent_outputs[agent_id].split('\n')
            agent_lines[agent_id] = lines
            max_lines = max(max_lines, len(lines))
        
        # Print header with better formatting
        header_sep = "─" * terminal_width
        print(f"\n{header_sep}")
        header = ""
        for i, agent_id in enumerate(agent_ids):
            agent_name = f"🤖 Agent {agent_id}"
            header += f"{agent_name:^{col_width}}"
            if i < num_agents - 1:
                header += " │ "
        print(f" {header} ")
        print(header_sep)
        
        # Print content rows with better line handling
        for line_idx in range(max_lines):
            row = ""
            for i, agent_id in enumerate(agent_ids):
                lines = agent_lines[agent_id]
                content = lines[line_idx] if line_idx < len(lines) else ""
                
                # Better content truncation
                if len(content) > col_width:
                    content = content[:col_width-1] + "…"
                
                row += f"{content:<{col_width}}"
                if i < num_agents - 1:
                    row += " │ "
            print(f" {row} ")
        
        # Print system messages with better formatting
        if self.system_messages:
            print(f"\n{header_sep}")
            print(f" 📋 SYSTEM MESSAGES{' ' * (terminal_width - 19)} ")
            print(header_sep)
            for message in self.system_messages:
                # Wrap long system messages
                if len(message) > terminal_width - 4:
                    words = message.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > terminal_width - 4:
                            print(f" {current_line.strip()} ")
                            current_line = word + " "
                        else:
                            current_line += word + " "
                    if current_line.strip():
                        print(f" {current_line.strip()} ")
                else:
                    print(f" {message} ")
        print(header_sep)

class StreamingOrchestrator:
    def __init__(self, display_enabled: bool = True, stream_callback: Optional[Callable] = None):
        self.display = MultiRegionDisplay(display_enabled)
        self.stream_callback = stream_callback
        self.active_agents: Dict[int, str] = {}
        
    async def stream_agent_output(self, agent_id: int, content: str):
        await self.display.stream_output(agent_id, content)
        if self.stream_callback:
            try:
                self.stream_callback(agent_id, content)
            except Exception:
                pass
    
    def finalize_agent_message(self, agent_id: int):
        self.display.finalize_message(agent_id)
    
    def update_agent_status(self, agent_id: int, status: str):
        old_status = self.active_agents.get(agent_id, "unknown")
        self.active_agents[agent_id] = status
        # Add system message for status changes
        if old_status != status:
            self.display.add_system_message(f"Agent {agent_id}: {old_status} → {status}")
    
    def update_system_phase(self, phase: str):
        self.display.add_system_message(f"Phase: {phase}")
    
    def update_voting_status(self, voter_id: int, target_id: int):
        self.display.add_system_message(f"🗳️  Agent {voter_id} VOTED for Agent {target_id}")
    
    def update_consensus_status(self, winning_agent_id: int, vote_distribution: Dict[int, int]):
        vote_summary = ", ".join([f"Agent {k}: {v}" for k, v in vote_distribution.items()])
        self.display.add_system_message(f"🎯 Consensus reached! Winner: Agent {winning_agent_id} ({vote_summary})")
    
    def add_system_message(self, message: str):
        """Add a custom system message to the display"""
        self.display.add_system_message(message)

def create_streaming_display(display_type: str = "terminal", 
                           display_enabled: bool = True,
                           stream_callback: Optional[Callable] = None):
    return StreamingOrchestrator(display_enabled, stream_callback)

def integrate_with_workflow_manager(workflow_manager, streaming_orchestrator):
    workflow_manager.streaming_orchestrator = streaming_orchestrator