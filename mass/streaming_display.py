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
    def __init__(self, display_enabled: bool = True, max_lines: int = 30, save_logs: bool = True):
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
        self.debate_rounds: int = 0  # Track debate rounds
        
        # Detailed agent state tracking for display
        self._agent_vote_targets: Dict[int, Optional[int]] = {}
        self._agent_chat_rounds: Dict[int, int] = {}
        self._agent_update_counts: Dict[int, int] = {}  # Track update history count
        
        # Border consistency tracking - improved tracking
        self._terminal_width = None
        self._actual_display_width = None
        self._col_width = None
        self._border_error_count = 0
        
        # Initialize logging directory and files
        if self.save_logs:
            self._setup_logging()
    
    def _get_terminal_dimensions(self):
        """Get and lock terminal dimensions for consistent display."""
        if self._terminal_width is None:
            try:
                detected_width = os.get_terminal_size().columns
            except:
                detected_width = 120
            
            # Conservative width calculation to prevent overflow
            self._terminal_width = max(80, min(detected_width - 4, 180))  # Extra margin
        
        return self._terminal_width
    
    def _calculate_display_dimensions(self, num_agents: int):
        """Calculate and lock all display dimensions."""
        if self._actual_display_width is None or self._col_width is None:
            terminal_width = self._get_terminal_dimensions()
            
            # Calculate border characters needed: │ before each column + │ at end
            border_chars = num_agents + 1
            
            # Calculate column width with safety margin
            available_width = terminal_width - border_chars - 4  # Extra safety margin
            col_width = max(30, available_width // num_agents)  # Minimum column width
            
            # Lock the actual display width
            actual_display_width = (col_width * num_agents) + border_chars
            
            # Final safety check - ensure we don't exceed terminal width
            if actual_display_width > terminal_width:
                col_width = max(25, (terminal_width - border_chars) // num_agents)
                actual_display_width = (col_width * num_agents) + border_chars
                
            # Additional safety check - if still too wide, use minimum values
            if actual_display_width > terminal_width:
                col_width = max(20, (terminal_width - border_chars - 2) // num_agents)
                actual_display_width = (col_width * num_agents) + border_chars
            
            self._col_width = col_width
            self._actual_display_width = actual_display_width
        
        return self._col_width, self._actual_display_width
    
    def _get_display_width_unified(self, text: str) -> int:
        """Unified, simplified width calculation with consistent behavior."""
        if not text:
            return 0
        
        # Strip ANSI escape sequences completely
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        
        # Simple, reliable width calculation
        width = 0
        for char in clean_text:
            char_code = ord(char)
            
            # Handle emojis and wide characters conservatively
            if (
                (char_code >= 0x1F000 and char_code <= 0x1FAFF) or  # Emoji blocks
                (char_code >= 0x2600 and char_code <= 0x27BF) or   # Misc symbols  
                (char_code >= 0x1F300 and char_code <= 0x1F5FF) or  # More emojis
                (char_code >= 0x1F900 and char_code <= 0x1F9FF) or  # Supplemental symbols
                unicodedata.east_asian_width(char) in ('F', 'W')    # Full-width
            ):
                width += 2
            elif char_code < 32 or char_code == 127:  # Control characters
                width += 0
            else:
                width += 1
        
        return width
    
    def _pad_to_exact_width_safe(self, text: str, target_width: int, align: str = 'left') -> str:
        """Safe, reliable padding function that guarantees exact width."""
        if target_width <= 0:
            return ""
        
        if not text:
            return " " * target_width
        
        current_width = self._get_display_width_unified(text)
        
        # Handle text that's too long - simple truncation
        if current_width > target_width:
            # Strip ANSI codes for safe truncation
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_text = ansi_escape.sub('', text)
            
            # Truncate character by character to be safe
            truncated = ""
            width = 0
            for char in clean_text:
                char_width = self._get_display_width_unified(char)
                if width + char_width > target_width - 1:  # Leave room for ellipsis
                    truncated += "…"
                    break
                truncated += char
                width += char_width
            
            text = truncated
            current_width = self._get_display_width_unified(text)
        
        # Calculate padding needed
        padding_needed = max(0, target_width - current_width)
        
        # Apply padding based on alignment
        if align == 'center':
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            result = " " * left_pad + text + " " * right_pad
        elif align == 'right':
            result = " " * padding_needed + text
        else:  # left align
            result = text + " " * padding_needed
        
        # Final validation and correction - use simple character counting for final check
        final_width = self._get_display_width_unified(result)
        if final_width != target_width:
            # Emergency correction - force exact width using simple padding
            if final_width > target_width:
                # Cut from the end, preserving important content at start
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_result = ansi_escape.sub('', result)
                if len(clean_result) >= target_width:
                    clean_result = clean_result[:target_width-1] + "…"
                result = clean_result + " " * max(0, target_width - len(clean_result))
            else:
                # Add simple spaces to reach exact width
                result += " " * (target_width - final_width)
        
        return result
    
    def _create_system_line_safe(self, content: str, width: int) -> str:
        """Create a system region line with guaranteed proper borders and width."""
        # Calculate content width (total width minus border characters)
        content_width = width - 2  # Subtract 2 for the │ characters on each side
        
        if content_width <= 0:
            return "│" + " " * max(0, width - 2) + "│"
        
        # Pad content to exact width
        padded_content = self._pad_to_exact_width_safe(content, content_width, 'left')
        
        # Create the line with borders
        line = f"│{padded_content}│"
        
        # Final validation - ensure line is exactly the right width
        actual_width = self._get_display_width_unified(line)
        if actual_width != width:
            # Emergency fallback - create a simple line with proper width
            safe_content = self._pad_to_exact_width_safe(content[:max(0, content_width-3)] + "...", content_width, 'left')
            line = f"│{safe_content}│"
        
        return line
    
    def _wrap_and_print_system_message(self, message: str, width: int):
        """Safely wrap and print a system message with consistent borders."""
        max_content_width = width - 2  # Account for border characters
        
        if max_content_width <= 0:
            return
        
        # Check if message fits on one line
        if self._get_display_width_unified(message) <= max_content_width:
            line = self._create_system_line_safe(message, width)
            print(line)
        else:
            # Simple word wrapping
            words = message.split()
            current_line = ""
            
            for word in words:
                test_line = f"{current_line} {word}".strip()
                if self._get_display_width_unified(test_line) > max_content_width:
                    # Print current line if it has content
                    if current_line.strip():
                        line = self._create_system_line_safe(current_line.strip(), width)
                        print(line)
                    current_line = word
                else:
                    current_line = test_line
            
            # Print final line if it has content
            if current_line.strip():
                line = self._create_system_line_safe(current_line.strip(), width)
                print(line)
    
    def _validate_border_consistency(self, line: str, expected_width: int) -> str:
        """Validate and correct border alignment issues using unified width calculation."""
        actual_width = self._get_display_width_unified(line)
        
        if actual_width != expected_width:
            self._border_error_count += 1
            
            # More robust border correction
            if line.startswith('│') and line.endswith('│'):
                # Extract content between borders
                content = line[1:-1] if len(line) >= 2 else ""
                
                # Use the safe line creation method for consistency
                corrected_line = self._create_system_line_safe(content, expected_width)
                
                # Double-check the corrected line
                corrected_width = self._get_display_width_unified(corrected_line)
                if corrected_width == expected_width:
                    return corrected_line
                else:
                    # Emergency fallback - create a simple bordered line with exact width
                    content_width = expected_width - 2
                    if content_width > 0:
                        safe_content = self._pad_to_exact_width_safe("", content_width, 'left')
                        return f"│{safe_content}│"
                    else:
                        return "│" * max(1, expected_width)
            else:
                # Fallback for non-bordered lines
                return self._pad_to_exact_width_safe(line, expected_width, 'left')
        
        return line
    
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
            
            # Status emoji mapping for system messages
            status_change_emoji = {
                "working": "🔄",
                "voted": "✅", 
                "failed": "❌",
                "unknown": "❓"
            }
            
            # Log status change with emoji
            old_emoji = status_change_emoji.get(old_status, "❓")
            new_emoji = status_change_emoji.get(status, "❓")
            status_msg = f"{old_emoji}→{new_emoji} Agent {agent_id}: {old_status} → {status}"
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
    
    def update_agent_vote_target(self, agent_id: int, target_id: Optional[int]):
        """Update which agent this agent voted for."""
        with self._lock:
            self._agent_vote_targets[agent_id] = target_id
    
    def update_agent_chat_round(self, agent_id: int, round_num: int):
        """Update the chat round for an agent."""
        with self._lock:
            self._agent_chat_rounds[agent_id] = round_num
    
    def update_agent_update_count(self, agent_id: int, count: int):
        """Update the update count for an agent."""
        with self._lock:
            self._agent_update_counts[agent_id] = count
    
    def update_debate_rounds(self, rounds: int):
        """Update the debate rounds count."""
        with self._lock:
            self.debate_rounds = rounds


    
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
    
    def _handle_terminal_resize(self):
        """Handle terminal resize by resetting cached dimensions."""
        try:
            current_width = os.get_terminal_size().columns
            if self._terminal_width and abs(current_width - self._terminal_width) > 5:
                # Significant change detected, reset cache
                self._reset_display_cache()
                return True
        except:
            pass
        return False
    
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
    
    def _reset_display_cache(self):
        """Reset cached display dimensions to handle terminal resize."""
        self._terminal_width = None
        self._actual_display_width = None
        self._col_width = None
        
    def _update_display(self):
        if not self.display_enabled:
            return
            
        try:
            # Handle potential terminal resize
            self._handle_terminal_resize()
            
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get sorted agent IDs for consistent ordering
            agent_ids = sorted(self.agent_outputs.keys())
            if not agent_ids:
                return
            
            # Get terminal dimensions and calculate display dimensions  
            num_agents = len(agent_ids)
            col_width, actual_display_width = self._calculate_display_dimensions(num_agents)
        except Exception as e:
            # Fallback to simple text output if display fails
            print(f"Display error: {e}")
            for agent_id in sorted(self.agent_outputs.keys()):
                print(f"Agent {agent_id}: {self.agent_outputs[agent_id][-100:]}")  # Last 100 chars
            return
        
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
        
        # Create horizontal border line - use the locked width
        border_line = "─" * actual_display_width
        
        # Enhanced MASS system header with fixed width
        print("")
        
        # ANSI color codes
        BRIGHT_CYAN = '\033[96m'
        BRIGHT_BLUE = '\033[94m'
        BRIGHT_GREEN = '\033[92m'
        BRIGHT_YELLOW = '\033[93m'
        BRIGHT_MAGENTA = '\033[95m'
        BRIGHT_RED = '\033[91m'
        BRIGHT_WHITE = '\033[97m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        # Header with exact width
        header_top = f"{BRIGHT_CYAN}{BOLD}╔{'═' * (actual_display_width - 2)}╗{RESET}"
        print(header_top)
        
        # Empty line
        header_empty = f"{BRIGHT_CYAN}║{' ' * (actual_display_width - 2)}║{RESET}"
        print(header_empty)
        
        # Title line with exact centering
        title_text = "🚀 MASS - Multi-Agent Scaling System 🚀"
        title_line_content = self._pad_to_exact_width_safe(title_text, actual_display_width - 2, 'center')
        title_line = f"{BRIGHT_CYAN}║{BRIGHT_YELLOW}{BOLD}{title_line_content}{RESET}{BRIGHT_CYAN}║{RESET}"
        print(title_line)
        
        # Subtitle line
        subtitle_text = "🔬 Advanced Agent Collaboration Framework"
        subtitle_line_content = self._pad_to_exact_width_safe(subtitle_text, actual_display_width - 2, 'center')
        subtitle_line = f"{BRIGHT_CYAN}║{BRIGHT_GREEN}{subtitle_line_content}{RESET}{BRIGHT_CYAN}║{RESET}"
        print(subtitle_line)
        
        # Empty line and bottom border
        print(header_empty)
        header_bottom = f"{BRIGHT_CYAN}{BOLD}╚{'═' * (actual_display_width - 2)}╝{RESET}"
        print(header_bottom)
        
        # Agent section with perfect alignment
        print(f"\n{border_line}")
        
        # Agent headers with exact column widths
        header_parts = []
        for agent_id in agent_ids:
            model_name = self.agent_models.get(agent_id, "")
            status = self.agent_statuses.get(agent_id, "unknown")
            
            # Status configuration
            status_config = {
                "working": {"emoji": "🔄", "color": BRIGHT_YELLOW},
                "voted": {"emoji": "🗳️", "color": BRIGHT_GREEN}, 
                "failed": {"emoji": "❌", "color": BRIGHT_RED},
                "unknown": {"emoji": "❓", "color": BRIGHT_WHITE}
            }
            
            config = status_config.get(status, status_config["unknown"])
            emoji = config["emoji"]
            status_color = config["color"]
            
            # Create agent header with exact width
            if model_name:
                agent_header = f"{emoji} {BRIGHT_CYAN}Agent {agent_id}{RESET} {BRIGHT_MAGENTA}({model_name}){RESET} {status_color}[{status}]{RESET}"
            else:
                agent_header = f"{emoji} {BRIGHT_CYAN}Agent {agent_id}{RESET} {status_color}[{status}]{RESET}"
            
            header_content = self._pad_to_exact_width_safe(agent_header, col_width, 'center')
            header_parts.append(header_content)
        
        # Print agent header line with exact borders
        header_line = "│" + "│".join(header_parts) + "│"
        header_line = self._validate_border_consistency(header_line, actual_display_width)
        print(header_line)
        
        # Agent state information line
        state_parts = []
        for agent_id in agent_ids:
            chat_round = getattr(self, '_agent_chat_rounds', {}).get(agent_id, 0)
            vote_target = getattr(self, '_agent_vote_targets', {}).get(agent_id)
            update_count = getattr(self, '_agent_update_counts', {}).get(agent_id, 0)
            
            # Format state info with better handling of color codes (removed redundant status)
            state_info = []
            state_info.append(f"{BRIGHT_WHITE}Round:{RESET} {BRIGHT_GREEN}{chat_round}{RESET}")
            state_info.append(f"{BRIGHT_WHITE}Updates:{RESET} {BRIGHT_MAGENTA}{update_count}{RESET}")
            if vote_target:
                state_info.append(f"{BRIGHT_WHITE}Vote →{RESET} {BRIGHT_GREEN}{vote_target}{RESET}")
            else:
                state_info.append(f"{BRIGHT_WHITE}Vote:{RESET} None")
            
            state_text = f"📊 {' | '.join(state_info)}"
            # Ensure exact column width with improved padding
            state_content = self._pad_to_exact_width_safe(state_text, col_width, 'center')
            state_parts.append(state_content)
        
        # Validate state line consistency before printing
        state_line = "│" + "│".join(state_parts) + "│"
        expected_state_width = actual_display_width
        state_line = self._validate_border_consistency(state_line, expected_state_width)
        print(state_line)
        
        # Log file information
        if self.save_logs and hasattr(self, 'session_logs_dir'):
            UNDERLINE = '\033[4m'
            link_parts = []
            for agent_id in agent_ids:
                log_path = self.get_agent_log_path_for_display(agent_id)
                if log_path:
                    # Shortened display path
                    display_path = log_path.replace(os.getcwd() + "/", "") if log_path.startswith(os.getcwd()) else log_path
                    
                    # Safe path truncation with better width handling
                    prefix = "📁 Log: "
                    # More conservative calculation
                    max_path_len = max(10, col_width - self._get_display_width_unified(prefix) - 8)
                    if len(display_path) > max_path_len:
                        display_path = "..." + display_path[-(max_path_len-3):]
                    
                    link_text = f"{prefix}{UNDERLINE}{display_path}{RESET}"
                    link_content = self._pad_to_exact_width_safe(link_text, col_width, 'center')
                else:
                    link_content = self._pad_to_exact_width_safe("", col_width, 'center')
                link_parts.append(link_content)
            
            # Validate log line consistency
            log_line = "│" + "│".join(link_parts) + "│"
            log_line = self._validate_border_consistency(log_line, expected_state_width)
            print(log_line)
        
        print(border_line)
        
        # Content area with perfect column alignment - Apply validation to every content line
        for line_idx in range(max_lines):
            content_parts = []
            for agent_id in agent_ids:
                lines = agent_lines[agent_id]
                content = lines[line_idx] if line_idx < len(lines) else ""
                
                # Ensure exact column width for each content piece
                padded_content = self._pad_to_exact_width_safe(content, col_width, 'left')
                content_parts.append(padded_content)
            
            # Apply border validation to every content line for consistency
            content_line = "│" + "│".join(content_parts) + "│"
            content_line = self._validate_border_consistency(content_line, actual_display_width)
            print(content_line)
        
        # System status section with exact width
        if self.system_messages or self.current_phase or self.vote_distribution:
            print(f"\n{border_line}")
            
            # System state header
            phase_color = BRIGHT_YELLOW if self.current_phase == "collaboration" else BRIGHT_GREEN
            consensus_color = BRIGHT_GREEN if self.consensus_reached else BRIGHT_RED
            consensus_text = "✅ YES" if self.consensus_reached else "❌ NO"
            
            system_state_info = []
            system_state_info.append(f"{BRIGHT_WHITE}Phase:{RESET} {phase_color}{self.current_phase.upper()}{RESET}")
            system_state_info.append(f"{BRIGHT_WHITE}Consensus:{RESET} {consensus_color}{consensus_text}{RESET}")
            system_state_info.append(f"{BRIGHT_WHITE}Debate Rounds:{RESET} {BRIGHT_CYAN}{self.debate_rounds}{RESET}")
            if self.representative_agent_id:
                system_state_info.append(f"{BRIGHT_WHITE}Rep →{RESET} {BRIGHT_GREEN}{self.representative_agent_id}{RESET}")
            else:
                system_state_info.append(f"{BRIGHT_WHITE}Representative Agent:{RESET} None")
            
            system_header_text = f"{BRIGHT_CYAN}📋 SYSTEM STATE{RESET} - {' | '.join(system_state_info)}"
            system_header_line = self._create_system_line_safe(system_header_text, actual_display_width)
            print(system_header_line)
            
            # System log file link
            if self.save_logs and hasattr(self, 'system_log_file'):
                system_log_path = self.get_system_log_path_for_display()
                if system_log_path:
                    UNDERLINE = '\033[4m'
                    display_path = system_log_path.replace(os.getcwd() + "/", "") if system_log_path.startswith(os.getcwd()) else system_log_path
                    
                    # Safe path truncation with consistent width handling
                    prefix = "📁 Log: "
                    max_path_len = max(10, actual_display_width - self._get_display_width_unified(prefix) - 15)
                    if len(display_path) > max_path_len:
                        display_path = "..." + display_path[-(max_path_len-3):]
                    
                    system_link_text = f"{prefix}{UNDERLINE}{display_path}{RESET}"
                    system_link_line = self._create_system_line_safe(system_link_text, actual_display_width)
                    print(system_link_line)
            
            print(border_line)
            
            # System messages with exact width and validation
            if self.consensus_reached and self.representative_agent_id is not None:
                consensus_msg = f"🎉 CONSENSUS REACHED! Representative: Agent {self.representative_agent_id}"
                consensus_line = self._create_system_line_safe(consensus_msg, actual_display_width)
                print(consensus_line)
            
            # Vote distribution with validation
            if self.vote_distribution:
                vote_msg = "🗳️  Vote Distribution: " + ", ".join([f"Agent {k}→{v} votes" for k, v in self.vote_distribution.items()])
                
                # Use the new safe wrapping method
                max_content_width = actual_display_width - 2
                if self._get_display_width_unified(vote_msg) <= max_content_width:
                    vote_line = self._create_system_line_safe(vote_msg, actual_display_width)
                    print(vote_line)
                else:
                    # Wrap vote distribution using safe method
                    vote_header = "🗳️  Vote Distribution:"
                    header_line = self._create_system_line_safe(vote_header, actual_display_width)
                    print(header_line)
                    
                    for agent_id, votes in self.vote_distribution.items():
                        vote_detail = f"   Agent {agent_id}: {votes} votes"
                        detail_line = self._create_system_line_safe(vote_detail, actual_display_width)
                        print(detail_line)
            
            # Regular system messages with validation
            for message in self.system_messages:
                self._wrap_and_print_system_message(message, actual_display_width)
        
        # Final border
        print(border_line)

class StreamingOrchestrator:
    def __init__(self, display_enabled: bool = True, stream_callback: Optional[Callable] = None, max_lines: int = 30, save_logs: bool = True):
        self.display = MultiRegionDisplay(display_enabled, max_lines, save_logs)
        self.stream_callback = stream_callback
    
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
    
    def update_agent_vote_target(self, agent_id: int, target_id: Optional[int]):
        """Update which agent this agent voted for."""
        self.display.update_agent_vote_target(agent_id, target_id)
    
    def update_agent_chat_round(self, agent_id: int, round_num: int):
        """Update the chat round for an agent."""
        self.display.update_agent_chat_round(agent_id, round_num)
    
    def update_agent_update_count(self, agent_id: int, count: int):
        """Update the update count for an agent."""
        self.display.update_agent_update_count(agent_id, count)
    
    def update_debate_rounds(self, rounds: int):
        """Update the debate rounds count."""
        self.display.update_debate_rounds(rounds)
    
    def format_agent_notification(self, agent_id: int, notification_type: str, content: str):
        """Format agent notifications for display."""
        self.display.format_agent_notification(agent_id, notification_type, content)
    
    def get_agent_log_path(self, agent_id: int) -> str:
        """Get the log file path for a specific agent."""
        return self.display.get_agent_log_path_for_display(agent_id)
    
    def get_system_log_path(self) -> str:
        """Get the system log file path."""
        return self.display.get_system_log_path_for_display()

def create_streaming_display(display_enabled: bool = True, stream_callback: Optional[Callable] = None, max_lines: int = 30, save_logs: bool = True) -> StreamingOrchestrator:
    """Create a streaming orchestrator with display capabilities."""
    return StreamingOrchestrator(display_enabled, stream_callback, max_lines, save_logs)