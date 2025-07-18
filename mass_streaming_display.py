"""
MASS Streaming Display System - Simple Working Version
"""

import os
import time
import threading
import unicodedata
from typing import Dict, List, Optional, Callable

class MultiRegionDisplay:
    def __init__(self, display_enabled: bool = True, max_lines: int = 40):
        self.display_enabled = display_enabled
        self.max_lines = max_lines  # Maximum lines to display per agent
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
        
        # Calculate exact column widths with precise separator spacing
        num_agents = len(agent_ids)
        separator_width = 3  # " │ " is exactly 3 characters
        separators_total_width = (num_agents - 1) * separator_width
        padding_width = 2  # 1 space on each side
        
        # Calculate available width for content
        available_width = terminal_width - separators_total_width - padding_width
        col_width = max(30, available_width // num_agents)
        
        # Recalculate actual terminal width needed
        actual_width = (col_width * num_agents) + separators_total_width + padding_width
        
        # Split content into lines for each agent and limit to max_lines
        agent_lines = {}
        max_lines = 0
        for agent_id in agent_ids:
            lines = self.agent_outputs[agent_id].split('\n')
            # Keep only the last max_lines lines (tail behavior)
            if len(lines) > self.max_lines:
                lines = lines[-self.max_lines:]
            agent_lines[agent_id] = lines
            max_lines = max(max_lines, len(lines))
        
        # Function to calculate actual display width of text
        def get_display_width(text):
            """Calculate the actual display width of text accounting for emojis, unicode, and ANSI codes."""
            if not text:
                return 0
            
            # Strip ANSI escape sequences first
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_text = ansi_escape.sub('', text)
            
            # Try to use wcwidth library if available for more accurate width calculation
            try:
                import wcwidth
                total_width = 0
                for char in clean_text:
                    char_width = wcwidth.wcwidth(char)
                    if char_width is None:
                        # wcwidth returns None for control characters, treat as 0 width
                        char_width = 0
                    elif char_width < 0:
                        # Some terminals treat certain characters as 0 width
                        char_width = 0
                    total_width += char_width
                return total_width
            except ImportError:
                # Fallback to manual calculation if wcwidth not available
                pass
            
            # Manual fallback with specific emoji handling
            width = 0
            i = 0
            while i < len(clean_text):
                char = clean_text[i]
                char_code = ord(char)
                
                # Check for specific problematic emojis first
                if char == '🗳':  # Ballot box (often 1 width)
                    # Check if followed by variation selector
                    if i + 1 < len(clean_text) and ord(clean_text[i + 1]) == 0xFE0F:
                        width += 1  # 🗳️ often renders as 1 width despite being an emoji
                        i += 2  # Skip both characters
                        continue
                    else:
                        width += 1
                        i += 1
                        continue
                
                # Zero-width characters
                if char_code in [0xFE0F, 0x200D]:  # Variation selectors and joiners
                    i += 1
                    continue
                
                # Standard emoji detection for other characters
                if (
                    # Main emoji blocks
                    (0x1F600 <= char_code <= 0x1F64F) or  # Emoticons  
                    (0x1F300 <= char_code <= 0x1F5FF) or  # Misc Symbols and Pictographs
                    (0x1F680 <= char_code <= 0x1F6FF) or  # Transport and Map Symbols
                    (0x1F700 <= char_code <= 0x1F77F) or  # Alchemical Symbols
                    (0x1F780 <= char_code <= 0x1F7FF) or  # Geometric Shapes Extended
                    (0x1F800 <= char_code <= 0x1F8FF) or  # Supplemental Arrows-C
                    (0x1F900 <= char_code <= 0x1F9FF) or  # Supplemental Symbols and Pictographs
                    (0x1FA00 <= char_code <= 0x1FA6F) or  # Chess Symbols
                    (0x1FA70 <= char_code <= 0x1FAFF) or  # Symbols and Pictographs Extended-A
                    # Additional emoji ranges
                    (0x2600 <= char_code <= 0x26FF) or   # Miscellaneous Symbols
                    (0x2700 <= char_code <= 0x27BF)      # Dingbats (✅ etc)
                ):
                    width += 2
                elif unicodedata.east_asian_width(char) in ('F', 'W'):  # Full-width or wide
                    width += 2
                else:
                    width += 1
                i += 1
            return width
        
        # Function to pad text to exact display width
        def pad_to_width(text, target_width, align='left'):
            """Pad text to exact display width, handling unicode and ANSI codes properly."""
            current_width = get_display_width(text)
            if current_width >= target_width:
                # Truncate if too long - need to handle ANSI codes properly
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_text = ansi_escape.sub('', text)
                
                result = ""
                width = 0
                text_pos = 0
                clean_pos = 0
                
                while text_pos < len(text) and clean_pos < len(clean_text):
                    # Check if we're at an ANSI sequence
                    if text[text_pos:text_pos+1] == '\x1B':
                        # Find the end of the ANSI sequence
                        ansi_match = ansi_escape.match(text[text_pos:])
                        if ansi_match:
                            # Add the entire ANSI sequence without counting width
                            result += ansi_match.group()
                            text_pos += len(ansi_match.group())
                            continue
                    
                    # Regular character - handle specific emojis first
                    char = clean_text[clean_pos]
                    char_code = ord(char)
                    
                    # Handle specific problematic emojis
                    if char == '🗳':  # Ballot box (often 1 width)
                        # Check if this will be followed by variation selector in original text
                        char_width = 1  # 🗳️ often renders as 1 width despite being an emoji
                    elif char_code in [0xFE0F, 0x200D]:  # Zero-width characters
                        char_width = 0
                    elif (
                        # Main emoji blocks
                        (0x1F600 <= char_code <= 0x1F64F) or  # Emoticons
                        (0x1F300 <= char_code <= 0x1F5FF) or  # Misc Symbols and Pictographs
                        (0x1F680 <= char_code <= 0x1F6FF) or  # Transport and Map Symbols
                        (0x1F700 <= char_code <= 0x1F77F) or  # Alchemical Symbols
                        (0x1F780 <= char_code <= 0x1F7FF) or  # Geometric Shapes Extended
                        (0x1F800 <= char_code <= 0x1F8FF) or  # Supplemental Arrows-C
                        (0x1F900 <= char_code <= 0x1F9FF) or  # Supplemental Symbols and Pictographs
                        (0x1FA00 <= char_code <= 0x1FA6F) or  # Chess Symbols
                        (0x1FA70 <= char_code <= 0x1FAFF) or  # Symbols and Pictographs Extended-A
                        # Additional emoji ranges
                        (0x2600 <= char_code <= 0x26FF) or   # Miscellaneous Symbols
                        (0x2700 <= char_code <= 0x27BF) or   # Dingbats (✅ etc)
                        unicodedata.east_asian_width(char) in ('F', 'W')  # Full-width or wide
                    ):
                        char_width = 2
                    else:
                        char_width = 1
                    
                    if width + char_width <= target_width - 1:  # Leave room for ellipsis
                        result += text[text_pos]
                        width += char_width
                        text_pos += 1
                        clean_pos += 1
                    else:
                        result += "…"
                        break
                
                return result
            
            # Add padding
            padding_needed = target_width - current_width
            if align == 'center':
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                return " " * left_pad + text + " " * right_pad
            elif align == 'right':
                return " " * padding_needed + text
            else:  # left align
                return text + " " * padding_needed
        
        # Print MASS system header with enhanced styling
        # ANSI color codes
        BRIGHT_CYAN = '\033[96m'
        BRIGHT_BLUE = '\033[94m'
        BRIGHT_GREEN = '\033[92m'
        BRIGHT_YELLOW = '\033[93m'
        BRIGHT_MAGENTA = '\033[95m'
        BRIGHT_RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        # Create enhanced multi-line header
        print("")
        
        # Top border with gradient effect
        top_border = f"{BRIGHT_CYAN}{BOLD}╔{'═' * (actual_width - 2)}╗{RESET}"
        print(top_border)
        
        # Empty line
        empty_line = f"{BRIGHT_CYAN}║{' ' * (actual_width - 2)}║{RESET}"
        print(empty_line)
        
        # Main title line
        title_text = "🚀 MASS - Multi-Agent Scaling System 🚀"
        title_display_width = get_display_width(title_text)
        title_padding = (actual_width - 2 - title_display_width) // 2
        title_line = f"{BRIGHT_CYAN}║{' ' * title_padding}{BRIGHT_YELLOW}{BOLD}{title_text}{RESET}{' ' * (actual_width - 2 - title_padding - title_display_width)}{BRIGHT_CYAN}║{RESET}"
        print(title_line)
        
        # Subtitle line
        subtitle_text = "🔬 Advanced Agent Collaboration Framework"
        subtitle_display_width = get_display_width(subtitle_text)
        subtitle_padding = (actual_width - 2 - subtitle_display_width) // 2
        subtitle_line = f"{BRIGHT_CYAN}║{' ' * subtitle_padding}{BRIGHT_GREEN}{subtitle_text}{RESET}{' ' * (actual_width - 2 - subtitle_padding - subtitle_display_width)}{BRIGHT_CYAN}║{RESET}"
        print(subtitle_line)
        
        # Another empty line
        print(empty_line)
        
        # Bottom border
        bottom_border = f"{BRIGHT_CYAN}{BOLD}╚{'═' * (actual_width - 2)}╝{RESET}"
        print(bottom_border)
        
        # Print agent column headers
        header_sep = "─" * actual_width
        print(f"\n{header_sep}")
        
        header_parts = []
        for agent_id in agent_ids:
            agent_name = f"🤖 Agent {agent_id}"
            # Center the agent name in the column with proper display width
            header_content = pad_to_width(agent_name, col_width, 'center')
            header_parts.append(header_content)
        
        header_line = " " + " │ ".join(header_parts) + " "
        print(header_line)
        print(header_sep)
        
        # Print content rows with exact column alignment
        for line_idx in range(max_lines):
            content_parts = []
            for agent_id in agent_ids:
                lines = agent_lines[agent_id]
                content = lines[line_idx] if line_idx < len(lines) else ""
                
                # Pad content to exact column width with proper display width
                padded_content = pad_to_width(content, col_width, 'left')
                content_parts.append(padded_content)
            
            content_line = " " + " │ ".join(content_parts) + " "
            print(content_line)
        
        # Print system messages with consistent formatting
        if self.system_messages:
            print(f"\n{header_sep}")
            system_header = f" 📋 SYSTEM MESSAGES{' ' * (actual_width - 19)} "
            print(system_header)
            print(header_sep)
            for message in self.system_messages:
                # Wrap long system messages to fit width
                max_message_width = actual_width - 4
                if len(message) > max_message_width:
                    words = message.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > max_message_width:
                            print(f" {current_line.strip()} ")
                            current_line = word + " "
                        else:
                            current_line += word + " "
                    if current_line.strip():
                        print(f" {current_line.strip()} ")
                else:
                    padded_message = f" {message}{' ' * (actual_width - len(message) - 2)} "
                    print(padded_message)
        print(header_sep)

class StreamingOrchestrator:
    def __init__(self, display_enabled: bool = True, stream_callback: Optional[Callable] = None, max_lines: int = 40):
        self.display = MultiRegionDisplay(display_enabled, max_lines)
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
                           stream_callback: Optional[Callable] = None,
                           max_lines: int = 40):
    return StreamingOrchestrator(display_enabled, stream_callback, max_lines)

def integrate_with_workflow_manager(workflow_manager, streaming_orchestrator):
    workflow_manager.streaming_orchestrator = streaming_orchestrator