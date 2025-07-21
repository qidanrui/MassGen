"""
MASS Streaming Display System

Provides real-time multi-region display for MASS agents with:
- Individual agent columns showing streaming conversations
- System status panel with phase transitions and voting
- File logging for all conversations and events
"""

import os
import time
import threading
import unicodedata
from typing import Dict, List, Optional, Callable, Union
from datetime import datetime

class MultiRegionDisplay:
    def __init__(self, display_enabled: bool = True, max_lines: int = 40, save_logs: bool = True):
        self.display_enabled = display_enabled
        self.max_lines = max_lines
        self.save_logs = save_logs
        self.agent_outputs: Dict[int, str] = {}
        self.agent_models: Dict[int, str] = {}
        self.agent_statuses: Dict[int, str] = {}  # Track agent statuses
        self.system_messages: List[str] = []
        self.start_time = time.time()
        self._lock = threading.RLock()  # Use reentrant lock to prevent deadlock
        
        # MASS-specific state tracking
        self.current_phase = "collaboration"
        self.vote_distribution: Dict[int, int] = {}
        self.consensus_reached = False
        self.representative_agent_id: Optional[int] = None
        
        # Initialize logging directory and files
        if self.save_logs:
            self._setup_logging()
    
    def set_agent_model(self, agent_id: int, model_name: str):
        """Set the model name for a specific agent."""
        with self._lock:
            self.agent_models[agent_id] = model_name
            # Ensure agent appears in display even if no content yet
            if agent_id not in self.agent_outputs:
                self.agent_outputs[agent_id] = ""
                
    def update_agent_status(self, agent_id: int, status: str):
        """Update agent status (working, voted, failed)."""
        with self._lock:
            old_status = self.agent_statuses.get(agent_id, "unknown")
            self.agent_statuses[agent_id] = status
            
            # Ensure agent appears in display even if no content yet
            if agent_id not in self.agent_outputs:
                self.agent_outputs[agent_id] = ""
            
            # Log status change
            status_msg = f"Agent {agent_id}: {old_status} → {status}"
            self.add_system_message(status_msg)
            
    def update_phase(self, old_phase: str, new_phase: str):
        """Update system phase."""
        with self._lock:
            self.current_phase = new_phase
            phase_msg = f"Phase: {old_phase} → {new_phase}"
            self.add_system_message(phase_msg)
            
    def update_vote_distribution(self, vote_dist: Dict[int, int]):
        """Update vote distribution."""
        with self._lock:
            self.vote_distribution = vote_dist.copy()
            
    def update_consensus_status(self, representative_id: int, vote_dist: Dict[int, int]):
        """Update when consensus is reached."""
        with self._lock:
            self.consensus_reached = True
            self.representative_agent_id = representative_id
            self.vote_distribution = vote_dist.copy()
            
            consensus_msg = f"🎉 CONSENSUS REACHED! Agent {representative_id} selected as representative"
            self.add_system_message(consensus_msg)
            
    def reset_consensus(self):
        """Reset consensus state for new debate round."""
        with self._lock:
            self.consensus_reached = False
            self.representative_agent_id = None
            self.vote_distribution.clear()
    
    def _setup_logging(self):
        """Set up the logging directory and initialize log files."""
        # Create logs directory if it doesn't exist
        base_logs_dir = "logs"
        os.makedirs(base_logs_dir, exist_ok=True)
        
        # Create timestamped subdirectory for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_logs_dir = os.path.join(base_logs_dir, timestamp)
        os.makedirs(self.session_logs_dir, exist_ok=True)
        
        # Initialize log file paths with simple names
        self.agent_log_files = {}
        self.system_log_file = os.path.join(self.session_logs_dir, "system.txt")
        
        # Initialize system log file
        with open(self.system_log_file, 'w', encoding='utf-8') as f:
            f.write(f"MASS System Messages Log\n")
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def _get_agent_log_file(self, agent_id: int) -> str:
        """Get or create the log file path for a specific agent."""
        if agent_id not in self.agent_log_files:
            # Use simple filename: agent_0.txt, agent_1.txt, etc.
            self.agent_log_files[agent_id] = os.path.join(self.session_logs_dir, f"agent_{agent_id}.txt")
            
            # Initialize agent log file
            with open(self.agent_log_files[agent_id], 'w', encoding='utf-8') as f:
                f.write(f"MASS Agent {agent_id} Output Log\n")
                f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
        
        return self.agent_log_files[agent_id]
    
    def get_agent_log_path_for_display(self, agent_id: int) -> str:
        """Get the log file path for display purposes (clickable link)."""
        if not self.save_logs:
            return ""
        
        # Ensure the log file exists by calling _get_agent_log_file
        log_path = self._get_agent_log_file(agent_id)
        
        # Return relative path for better display
        return log_path
    
    def get_system_log_path_for_display(self) -> str:
        """Get the system log file path for display purposes (clickable link)."""
        if not self.save_logs:
            return ""
        
        return self.system_log_file
    
    def _write_agent_log(self, agent_id: int, content: str):
        """Write content to the agent's log file."""
        if not self.save_logs:
            return
            
        try:
            log_file = self._get_agent_log_file(agent_id)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(content)
                f.flush()  # Ensure immediate write
        except Exception as e:
            print(f"Error writing to agent {agent_id} log: {e}")
    
    def _write_system_log(self, message: str):
        """Write a system message to the system log file."""
        if not self.save_logs:
            return
            
        try:
            with open(self.system_log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")
                f.flush()  # Ensure immediate write
        except Exception as e:
            print(f"Error writing to system log: {e}")
        
    async def stream_output(self, agent_id: int, content: str):
        if not self.display_enabled:
            return
            
        with self._lock:
            if agent_id not in self.agent_outputs:
                self.agent_outputs[agent_id] = ""
            
            # Handle special content markers for display vs logging
            display_content = content
            log_content = content
            
            # Check for special markers (keep text markers for backward compatibility)
            if content.startswith('[CODE_DISPLAY_ONLY]'):
                # This content should only be shown in display, not logged
                display_content = content[len('[CODE_DISPLAY_ONLY]'):]
                log_content = ""  # Don't log this content
            elif content.startswith('[CODE_LOG_ONLY]'):
                # This content should only be logged, not displayed
                display_content = ""  # Don't display this content
                log_content = content[len('[CODE_LOG_ONLY]'):]
            
            # Add to display output only if there's display content
            if display_content:
                self.agent_outputs[agent_id] += display_content
            
            # Write to log file only if there's log content
            if log_content:
                self._write_agent_log(agent_id, log_content)
            
            # Update display if there was display content
            if display_content:
                self._update_display()
    
    def stream_output_sync(self, agent_id: int, content: str):
        """Synchronous version for non-async code."""
        if not self.display_enabled:
            return
            
        with self._lock:
            if agent_id not in self.agent_outputs:
                self.agent_outputs[agent_id] = ""
            
            # Handle special content markers for display vs logging
            display_content = content
            log_content = content
            
            # Check for special markers (keep text markers for backward compatibility)
            if content.startswith('[CODE_DISPLAY_ONLY]'):
                # This content should only be shown in display, not logged
                display_content = content[len('[CODE_DISPLAY_ONLY]'):]
                log_content = ""  # Don't log this content
            elif content.startswith('[CODE_LOG_ONLY]'):
                # This content should only be logged, not displayed
                display_content = ""  # Don't display this content
                log_content = content[len('[CODE_LOG_ONLY]'):]
            
            # Add to display output only if there's display content
            if display_content:
                self.agent_outputs[agent_id] += display_content
            
            # Write to log file only if there's log content
            if log_content:
                self._write_agent_log(agent_id, log_content)
            
            # Update display if there was display content
            if display_content:
                self._update_display()
    
    def finalize_message(self, agent_id: int):
        if not self.display_enabled:
            return
            
        with self._lock:
            if agent_id in self.agent_outputs:
                self.agent_outputs[agent_id] += "\n"
                
                # Write newline to log file
                self._write_agent_log(agent_id, "\n")
                
                self._update_display()
    
    def finalize_message_sync(self, agent_id: int):
        """Synchronous version for non-async code."""
        if not self.display_enabled:
            return
            
        with self._lock:
            if agent_id in self.agent_outputs:
                self.agent_outputs[agent_id] += "\n"
                
                # Write newline to log file
                self._write_agent_log(agent_id, "\n")
                
                self._update_display()
    
    def add_system_message(self, message: str):
        """Add a system message with timestamp."""
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
            self.system_messages.append(formatted_message)
            
            # Keep only recent messages
            if len(self.system_messages) > 20:
                self.system_messages = self.system_messages[-20:]
                
            # Write to system log
            self._write_system_log(formatted_message + "\n")
    
    def format_agent_notification(self, agent_id: int, notification_type: str, content: str):
        """Format agent notifications for display."""
        notification_emoji = {
            "update": "📢",
            "debate": "🗣️", 
            "presentation": "🎯",
            "prompt": "💡"
        }
        
        emoji = notification_emoji.get(notification_type, "📨")
        notification_msg = f"{emoji} Agent {agent_id} received {notification_type} notification"
        self.add_system_message(notification_msg)
    
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
                
                while text_pos < len(text) and clean_pos < len(clean_text) and width < target_width - 1:
                    # Check if we're at an ANSI sequence
                    if text_pos < len(text) and text[text_pos:text_pos+1] == '\x1B':
                        # Find the end of the ANSI sequence
                        ansi_match = ansi_escape.match(text[text_pos:])
                        if ansi_match:
                            # Add the entire ANSI sequence without counting width
                            result += ansi_match.group()
                            text_pos += len(ansi_match.group())
                            continue
                        else:
                            # If no ANSI match, treat as regular character and advance
                            text_pos += 1
                            continue
                    
                    # Regular character processing
                    if clean_pos < len(clean_text):
                        char = clean_text[clean_pos]
                        char_code = ord(char)
                        
                        # Calculate character width using same logic as get_display_width
                        if char == '🗳':  # Ballot box (often 1 width)
                            # Check if followed by variation selector in clean text
                            if clean_pos + 1 < len(clean_text) and ord(clean_text[clean_pos + 1]) == 0xFE0F:
                                char_width = 1  # 🗳️ often renders as 1 width despite being an emoji
                                char_count = 2  # Skip both characters in clean text
                            else:
                                char_width = 1
                                char_count = 1
                        elif char_code in [0xFE0F, 0x200D]:  # Zero-width characters
                            char_width = 0
                            char_count = 1
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
                            char_count = 1
                        else:
                            char_width = 1
                            char_count = 1
                        
                        if width + char_width <= target_width - 1:  # Leave room for ellipsis
                            # Find the corresponding character(s) in original text
                            chars_to_add = ""
                            start_pos = text_pos
                            remaining_clean_chars = char_count
                            
                            # Advance through original text to find the character(s)
                            while text_pos < len(text) and remaining_clean_chars > 0:
                                if text[text_pos:text_pos+1] == '\x1B':
                                    # Skip ANSI sequence
                                    ansi_match = ansi_escape.match(text[text_pos:])
                                    if ansi_match:
                                        chars_to_add += ansi_match.group()
                                        text_pos += len(ansi_match.group())
                                    else:
                                        chars_to_add += text[text_pos]
                                        text_pos += 1
                                        remaining_clean_chars -= 1
                                else:
                                    chars_to_add += text[text_pos]
                                    text_pos += 1
                                    remaining_clean_chars -= 1
                            
                            result += chars_to_add
                            width += char_width
                            clean_pos += char_count
                        else:
                            result += "…"
                            break
                    else:
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
        
        # First row: Agent names with model information and status
        header_parts = []
        for agent_id in agent_ids:
            model_name = self.agent_models.get(agent_id, "")
            status = self.agent_statuses.get(agent_id, "unknown")
            
            # Status emoji mapping
            status_emoji = {
                "working": "🔄",
                "voted": "🗳️", 
                "failed": "❌",
                "unknown": "❓"
            }
            
            # Create agent header with status
            if model_name:
                agent_name = f"{status_emoji.get(status, '❓')} Agent {agent_id} ({model_name}) [{status}]"
            else:
                agent_name = f"{status_emoji.get(status, '❓')} Agent {agent_id} [{status}]"
                
            # Center the agent name in the column with proper display width
            header_content = pad_to_width(agent_name, col_width, 'center')
            header_parts.append(header_content)
        
        header_line = " " + " │ ".join(header_parts) + " "
        print(header_line)
        
        # Second row: Log file links (if logging is enabled) with underlines
        if self.save_logs and hasattr(self, 'session_logs_dir'):
            # ANSI escape codes for formatting
            UNDERLINE = '\033[4m'
            RESET = '\033[0m'
            
            link_parts = []
            for agent_id in agent_ids:
                log_path = self.get_agent_log_path_for_display(agent_id)
                if log_path:
                    # Create a shortened display version of the path
                    display_path = log_path.replace(os.getcwd() + "/", "") if log_path.startswith(os.getcwd()) else log_path
                    
                    # Fix path truncation to use display width instead of len()
                    folder_prefix = f"📁 {UNDERLINE}"
                    folder_suffix = f"{RESET}"
                    base_width = get_display_width(folder_prefix) + get_display_width(folder_suffix)
                    max_path_width = col_width - base_width - 2  # Account for padding
                    
                    if get_display_width(display_path) > max_path_width:
                        # Truncate path properly considering display width
                        truncated_path = "..."
                        remaining_width = max_path_width - get_display_width(truncated_path)
                        if remaining_width > 0:
                            # Take characters from the end, being careful about multi-byte characters
                            for i in range(len(display_path)):
                                suffix = display_path[-(i+1):]
                                if get_display_width(suffix) <= remaining_width:
                                    truncated_path = "..." + suffix
                                else:
                                    break
                        display_path = truncated_path
                    
                    link_content = pad_to_width(f"📁 {UNDERLINE}{display_path}{RESET}", col_width, 'center')
                else:
                    link_content = pad_to_width("", col_width, 'center')
                link_parts.append(link_content)
            
            link_line = " " + " │ ".join(link_parts) + " "
            print(link_line)
        
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
        
        # Print system messages with MASS-specific information
        if self.system_messages or self.current_phase or self.vote_distribution:
            print(f"\n{header_sep}")
            
            # Fix system header padding calculation to use display width
            system_header_text = f" 📋 SYSTEM MESSAGES - Phase: {self.current_phase.upper()}"
            system_header_display_width = get_display_width(system_header_text)
            system_header_padding = actual_width - system_header_display_width - 1
            system_header = f"{system_header_text}{' ' * max(0, system_header_padding)} "
            print(system_header)
            
            # Add system log file link if logging is enabled
            if self.save_logs and hasattr(self, 'system_log_file'):
                system_log_path = self.get_system_log_path_for_display()
                if system_log_path:
                    # ANSI escape codes for formatting
                    UNDERLINE = '\033[4m'
                    RESET = '\033[0m'
                    
                    # Create a shortened display version of the path
                    display_path = system_log_path.replace(os.getcwd() + "/", "") if system_log_path.startswith(os.getcwd()) else system_log_path
                    
                    # Fix path truncation to use display width instead of len()
                    folder_prefix = f" 📁 {UNDERLINE}"
                    folder_suffix = f"{RESET}"
                    base_width = get_display_width(folder_prefix) + get_display_width(folder_suffix)
                    max_path_width = actual_width - base_width - 2  # Account for padding
                    
                    if get_display_width(display_path) > max_path_width:
                        # Truncate path properly considering display width
                        truncated_path = "..."
                        remaining_width = max_path_width - get_display_width(truncated_path)
                        if remaining_width > 0:
                            # Take characters from the end, being careful about multi-byte characters
                            for i in range(len(display_path)):
                                suffix = display_path[-(i+1):]
                                if get_display_width(suffix) <= remaining_width:
                                    truncated_path = "..." + suffix
                                else:
                                    break
                        display_path = truncated_path
                    
                    system_link_text = f" 📁 {UNDERLINE}{display_path}{RESET}"
                    system_link_display_width = get_display_width(system_link_text)
                    system_link_padding = actual_width - system_link_display_width - 1
                    system_link_line = f"{system_link_text}{' ' * max(0, system_link_padding)} "
                    print(system_link_line)
            
            print(header_sep)
            
            # Show current system status
            if self.consensus_reached and self.representative_agent_id is not None:
                consensus_msg = f"🎉 CONSENSUS REACHED! Representative: Agent {self.representative_agent_id}"
                # Fix padding calculation to use display width instead of len()
                consensus_display_width = get_display_width(consensus_msg)
                consensus_padding = actual_width - consensus_display_width - 3  # Account for leading space and trailing spaces
                print(f" {consensus_msg}{' ' * max(0, consensus_padding)} ")
            
            # Show vote distribution if available
            if self.vote_distribution:
                vote_msg = "🗳️  Vote Distribution: " + ", ".join([f"Agent {k}→{v} votes" for k, v in self.vote_distribution.items()])
                vote_msg_display_width = get_display_width(vote_msg)
                
                if vote_msg_display_width > actual_width - 4:
                    # Wrap long vote distribution
                    print(f" 🗳️  Vote Distribution:")
                    for agent_id, votes in self.vote_distribution.items():
                        vote_detail = f"   Agent {agent_id}: {votes} votes"
                        # Fix padding calculation to use display width
                        vote_detail_display_width = get_display_width(vote_detail)
                        vote_detail_padding = actual_width - vote_detail_display_width - 3
                        print(f" {vote_detail}{' ' * max(0, vote_detail_padding)} ")
                else:
                    # Fix padding calculation to use display width
                    vote_msg_padding = actual_width - vote_msg_display_width - 3
                    print(f" {vote_msg}{' ' * max(0, vote_msg_padding)} ")
            
            # Show regular system messages
            for message in self.system_messages:
                # Wrap long system messages to fit width
                max_message_width = actual_width - 4
                message_display_width = get_display_width(message)
                
                if message_display_width > max_message_width:
                    # Word wrapping with proper display width calculation
                    words = message.split()
                    current_line = ""
                    
                    for word in words:
                        test_line = f"{current_line} {word}".strip()
                        if get_display_width(test_line) > max_message_width:
                            if current_line.strip():
                                # Print current line with proper padding
                                current_line_display_width = get_display_width(current_line.strip())
                                current_line_padding = actual_width - current_line_display_width - 3
                                print(f" {current_line.strip()}{' ' * max(0, current_line_padding)} ")
                            current_line = word + " "
                        else:
                            current_line = test_line + " "
                    
                    if current_line.strip():
                        # Print final line with proper padding
                        final_line_display_width = get_display_width(current_line.strip())
                        final_line_padding = actual_width - final_line_display_width - 3
                        print(f" {current_line.strip()}{' ' * max(0, final_line_padding)} ")
                else:
                    # Fix padding calculation to use display width
                    message_padding = actual_width - message_display_width - 3
                    print(f" {message}{' ' * max(0, message_padding)} ")
        print(header_sep)

class StreamingOrchestrator:
    def __init__(self, display_enabled: bool = True, stream_callback: Optional[Callable] = None, max_lines: int = 40, save_logs: bool = True):
        self.display = MultiRegionDisplay(display_enabled, max_lines, save_logs)
        self.stream_callback = stream_callback
        self.active_agents: Dict[int, str] = {}
        
    async def stream_agent_output(self, agent_id: int, content: str):
        await self.display.stream_output(agent_id, content)
        if self.stream_callback:
            try:
                self.stream_callback(agent_id, content)
            except Exception:
                pass
    
    def stream_output(self, agent_id: int, content: str):
        """Synchronous version for non-async code."""
        self.display.stream_output_sync(agent_id, content)
        if self.stream_callback:
            try:
                self.stream_callback(agent_id, content)
            except Exception:
                pass
    
    def set_agent_model(self, agent_id: int, model_name: str):
        self.display.set_agent_model(agent_id, model_name)
    
    def update_agent_status(self, agent_id: int, status: str):
        self.display.update_agent_status(agent_id, status)
    
    def update_phase(self, old_phase: str, new_phase: str):
        self.display.update_phase(old_phase, new_phase)
    
    def update_vote_distribution(self, vote_dist: Dict[int, int]):
        self.display.update_vote_distribution(vote_dist)
    
    def update_consensus_status(self, representative_id: int, vote_dist: Dict[int, int]):
        self.display.update_consensus_status(representative_id, vote_dist)
    
    def reset_consensus(self):
        self.display.reset_consensus()
    
    def add_system_message(self, message: str):
        """Add a custom system message to the display"""
        self.display.add_system_message(message)
    
    def format_agent_notification(self, agent_id: int, notification_type: str, content: str):
        """Format agent notifications for display."""
        self.display.format_agent_notification(agent_id, notification_type, content)
    
    def get_agent_log_path(self, agent_id: int) -> str:
        """Get the log file path for a specific agent."""
        return self.display.get_agent_log_path_for_display(agent_id)
    
    def get_system_log_path(self) -> str:
        """Get the system log file path."""
        return self.display.get_system_log_path_for_display()

def create_streaming_display(display_enabled: bool = True, stream_callback: Optional[Callable] = None, max_lines: int = 40, save_logs: bool = True) -> StreamingOrchestrator:
    """Create a streaming orchestrator with display capabilities."""
    return StreamingOrchestrator(display_enabled, stream_callback, max_lines, save_logs)

def integrate_with_workflow_manager(workflow_manager, streaming_orchestrator):
    """Integrate streaming display with workflow manager."""
    workflow_manager.streaming_orchestrator = streaming_orchestrator