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
        
        # Detailed agent state tracking for display
        self._agent_vote_targets: Dict[int, Optional[int]] = {}
        self._agent_chat_rounds: Dict[int, int] = {}
        
        # Border consistency tracking
        self._last_display_width = None
        self._border_error_count = 0
        
        # Initialize logging directory and files
        if self.save_logs:
            self._setup_logging()
    
    def _validate_border_consistency(self, line: str, expected_width: int) -> str:
        """Validate and correct border alignment issues."""
        import re
        
        # Count actual visual width more accurately
        ansi_escape = re.compile(r'\x1B\[[0-9;]*[a-zA-Z]')
        clean_line = ansi_escape.sub('', line)
        
        # Calculate actual display width
        actual_width = 0
        for char in clean_line:
            char_code = ord(char)
            if (
                (char_code >= 0x1F000 and char_code <= 0x1FAFF) or  # Emoji blocks
                (char_code >= 0x2600 and char_code <= 0x27BF) or   # Misc symbols
                unicodedata.east_asian_width(char) in ('F', 'W')
            ):
                actual_width += 2
            else:
                actual_width += 1
        
        if actual_width != expected_width:
            self._border_error_count += 1
            
            # More sophisticated correction
            if actual_width > expected_width:
                # Truncate safely from the right, preserving border chars
                excess = actual_width - expected_width
                if line.endswith('│'):
                    # Keep the border character, truncate content
                    content = line[:-1]
                    truncated_content = content[:max(1, len(content) - excess)]
                    # Pad to exact width
                    padding_needed = expected_width - len(truncated_content) - 1
                    return truncated_content + " " * max(0, padding_needed) + "│"
                else:
                    return line[:expected_width]
            else:
                # Pad to correct width
                padding_needed = expected_width - actual_width
                if line.endswith('│'):
                    # Insert padding before the final border
                    content = line[:-1]
                    return content + " " * padding_needed + "│"
                else:
                    return line + " " * padding_needed
        
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
        
        # Fixed terminal width calculation - lock it for the entire display update
        try:
            detected_width = os.get_terminal_size().columns
        except:
            detected_width = 120
            
        # Ensure minimum width and maximum practical width - make it more conservative
        terminal_width = max(80, min(detected_width, 180))
        
        agent_ids = sorted(self.agent_outputs.keys())
        if not agent_ids:
            return
        
        # Fixed border system - calculate once and use consistently
        num_agents = len(agent_ids)
        border_chars = num_agents + 1  # │ chars (one before each column + one at the end)
        
        # Calculate column width with safety margin to prevent overflow
        available_width = terminal_width - border_chars - 2  # Extra safety margin
        col_width = max(35, available_width // num_agents)  # Minimum column width
        
        # Lock the actual display width - this will be used consistently throughout
        actual_display_width = (col_width * num_agents) + border_chars
        
        # Ensure we don't exceed terminal width
        if actual_display_width > terminal_width:
            col_width = max(30, (terminal_width - border_chars) // num_agents)
            actual_display_width = (col_width * num_agents) + border_chars
        
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
        
        # Simplified display width calculation with better error handling
        def get_display_width_safe(text):
            """Simplified display width calculation that handles edge cases better."""
            if not text:
                return 0
            
            # Strip ANSI escape sequences first - improved regex
            import re
            ansi_escape = re.compile(r'\x1B\[[0-9;]*[a-zA-Z]')
            clean_text = ansi_escape.sub('', text)
            
            # Try wcwidth first if available
            try:
                import wcwidth
                width = wcwidth.wcswidth(clean_text)
                if width is not None and width >= 0:
                    return width
            except ImportError:
                pass
            
            # Simple fallback that's more reliable for complex content
            width = 0
            for char in clean_text:
                char_code = ord(char)
                # Conservative emoji detection - treat all potential emoji as width 2
                if (
                    (char_code >= 0x1F000 and char_code <= 0x1FAFF) or  # All emoji blocks
                    (char_code >= 0x2600 and char_code <= 0x27BF) or   # Misc symbols
                    unicodedata.east_asian_width(char) in ('F', 'W')    # Full-width
                ):
                    width += 2
                elif char_code < 32 or char_code == 127:  # Control characters
                    width += 0
                else:
                    width += 1
            return width
        
        # Robust padding function that ensures exact width
        def pad_to_exact_width(text, target_width, align='left'):
            """Pad text to exact width with strong guarantees."""
            if not text:
                return " " * target_width
            
            current_width = get_display_width_safe(text)
            
            # Handle text that's too long - truncate safely
            if current_width > target_width:
                # Simple truncation that's safe
                import re
                ansi_escape = re.compile(r'\x1B\[[0-9;]*[a-zA-Z]')
                clean_text = ansi_escape.sub('', text)
                
                # Truncate clean text to fit, being conservative
                truncated = ""
                width = 0
                for char in clean_text:
                    char_width = 2 if get_display_width_safe(char) == 2 else 1
                    if width + char_width > target_width - 1:  # Leave room for ellipsis
                        truncated += "…"
                        break
                    truncated += char
                    width += char_width
                
                # For text with ANSI codes, try to preserve formatting but truncate safely
                if '\x1B' in text and len(truncated) > 0:
                    # Find ANSI codes in original and try to preserve color resets
                    if '\033[0m' in text:
                        text = truncated + '\033[0m'  # Add reset to prevent color bleeding
                    else:
                        text = truncated
                else:
                    text = truncated
                current_width = get_display_width_safe(text)
            
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
            
            # Final safety check - ensure exact width
            final_width = get_display_width_safe(result)
            if final_width != target_width:
                # Force correct width by truncating or padding as needed
                if final_width > target_width:
                    # More aggressive truncation for safety
                    result = result[:target_width]
                    # Ensure no broken ANSI codes at the end
                    if result.endswith('\x1B'):
                        result = result[:-1] + " "
                    elif len(result) > 1 and result[-2:].startswith('\x1B'):
                        result = result[:-2] + "  "
                else:
                    result += " " * (target_width - final_width)
            
            return result
        
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
        title_line_content = pad_to_exact_width(title_text, actual_display_width - 2, 'center')
        title_line = f"{BRIGHT_CYAN}║{BRIGHT_YELLOW}{BOLD}{title_line_content}{RESET}{BRIGHT_CYAN}║{RESET}"
        print(title_line)
        
        # Subtitle line
        subtitle_text = "🔬 Advanced Agent Collaboration Framework"
        subtitle_line_content = pad_to_exact_width(subtitle_text, actual_display_width - 2, 'center')
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
            
            header_content = pad_to_exact_width(agent_header, col_width, 'center')
            header_parts.append(header_content)
        
        # Print agent header line with exact borders
        print("│" + "│".join(header_parts) + "│")
        
        # Agent state information line
        state_parts = []
        for agent_id in agent_ids:
            status = self.agent_statuses.get(agent_id, "unknown")
            chat_round = getattr(self, '_agent_chat_rounds', {}).get(agent_id, 0)
            vote_target = getattr(self, '_agent_vote_targets', {}).get(agent_id)
            
            # Format state info with better handling of color codes
            state_info = []
            state_info.append(f"{BRIGHT_WHITE}Status:{RESET} {BRIGHT_BLUE}{status}{RESET}")
            state_info.append(f"{BRIGHT_WHITE}Round:{RESET} {BRIGHT_GREEN}{chat_round}{RESET}")
            if vote_target:
                state_info.append(f"{BRIGHT_WHITE}Vote →{RESET} {BRIGHT_GREEN}{vote_target}{RESET}")
            else:
                state_info.append(f"{BRIGHT_WHITE}Vote:{RESET} None")
            
            state_text = f"📊 {' | '.join(state_info)}"
            # Ensure exact column width with improved padding
            state_content = pad_to_exact_width(state_text, col_width, 'center')
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
                    max_path_len = max(10, col_width - get_display_width_safe(prefix) - 8)
                    if len(display_path) > max_path_len:
                        display_path = "..." + display_path[-(max_path_len-3):]
                    
                    link_text = f"{prefix}{UNDERLINE}{display_path}{RESET}"
                    link_content = pad_to_exact_width(link_text, col_width, 'center')
                else:
                    link_content = pad_to_exact_width("", col_width, 'center')
                link_parts.append(link_content)
            
            # Validate log line consistency
            log_line = "│" + "│".join(link_parts) + "│"
            log_line = self._validate_border_consistency(log_line, expected_state_width)
            print(log_line)
        
        print(border_line)
        
        # Content area with perfect column alignment
        for line_idx in range(max_lines):
            content_parts = []
            for agent_id in agent_ids:
                lines = agent_lines[agent_id]
                content = lines[line_idx] if line_idx < len(lines) else ""
                
                # Ensure exact column width for each content piece
                padded_content = pad_to_exact_width(content, col_width, 'left')
                content_parts.append(padded_content)
            
            # Print content line with exact borders
            print("│" + "│".join(content_parts) + "│")
        
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
            if self.representative_agent_id:
                system_state_info.append(f"{BRIGHT_WHITE}Rep →{RESET} {BRIGHT_GREEN}{self.representative_agent_id}{RESET}")
            else:
                system_state_info.append(f"{BRIGHT_WHITE}Rep:{RESET} None")
            
            system_header_text = f"{BRIGHT_CYAN}📋 SYSTEM STATE{RESET} - {' | '.join(system_state_info)}"
            system_header_content = pad_to_exact_width(system_header_text, actual_display_width - 2, 'left')
            system_header_line = f"│{system_header_content}│"
            system_header_line = self._validate_border_consistency(system_header_line, actual_display_width)
            print(system_header_line)
            
            # System log file link
            if self.save_logs and hasattr(self, 'system_log_file'):
                system_log_path = self.get_system_log_path_for_display()
                if system_log_path:
                    UNDERLINE = '\033[4m'
                    display_path = system_log_path.replace(os.getcwd() + "/", "") if system_log_path.startswith(os.getcwd()) else system_log_path
                    
                    # Safe path truncation with consistent width handling
                    prefix = "📁 Log: "
                    max_path_len = max(10, actual_display_width - get_display_width_safe(prefix) - 15)
                    if len(display_path) > max_path_len:
                        display_path = "..." + display_path[-(max_path_len-3):]
                    
                    system_link_text = f"{prefix}{UNDERLINE}{display_path}{RESET}"
                    system_link_content = pad_to_exact_width(system_link_text, actual_display_width - 2, 'left')
                    system_link_line = f"│{system_link_content}│"
                    system_link_line = self._validate_border_consistency(system_link_line, actual_display_width)
                    print(system_link_line)
            
            print(border_line)
            
            # System messages with exact width and validation
            if self.consensus_reached and self.representative_agent_id is not None:
                consensus_msg = f"🎉 CONSENSUS REACHED! Representative: Agent {self.representative_agent_id}"
                consensus_content = pad_to_exact_width(consensus_msg, actual_display_width - 2, 'left')
                consensus_line = f"│{consensus_content}│"
                consensus_line = self._validate_border_consistency(consensus_line, actual_display_width)
                print(consensus_line)
            
            # Vote distribution with validation
            if self.vote_distribution:
                vote_msg = "🗳️  Vote Distribution: " + ", ".join([f"Agent {k}→{v} votes" for k, v in self.vote_distribution.items()])
                
                # Handle long vote messages by wrapping
                max_content_width = actual_display_width - 2
                if get_display_width_safe(vote_msg) <= max_content_width:
                    vote_content = pad_to_exact_width(vote_msg, max_content_width, 'left')
                    vote_line = f"│{vote_content}│"
                    vote_line = self._validate_border_consistency(vote_line, actual_display_width)
                    print(vote_line)
                else:
                    # Wrap vote distribution
                    vote_header = "🗳️  Vote Distribution:"
                    header_content = pad_to_exact_width(vote_header, max_content_width, 'left')
                    header_line = f"│{header_content}│"
                    header_line = self._validate_border_consistency(header_line, actual_display_width)
                    print(header_line)
                    
                    for agent_id, votes in self.vote_distribution.items():
                        vote_detail = f"   Agent {agent_id}: {votes} votes"
                        detail_content = pad_to_exact_width(vote_detail, max_content_width, 'left')
                        detail_line = f"│{detail_content}│"
                        detail_line = self._validate_border_consistency(detail_line, actual_display_width)
                        print(detail_line)
            
            # Regular system messages with validation
            for message in self.system_messages:
                max_content_width = actual_display_width - 2
                
                # Handle long messages by wrapping
                if get_display_width_safe(message) <= max_content_width:
                    message_content = pad_to_exact_width(message, max_content_width, 'left')
                    message_line = f"│{message_content}│"
                    message_line = self._validate_border_consistency(message_line, actual_display_width)
                    print(message_line)
                else:
                    # Simple word wrapping with validation
                    words = message.split()
                    current_line = ""
                    
                    for word in words:
                        test_line = f"{current_line} {word}".strip()
                        if get_display_width_safe(test_line) > max_content_width:
                            if current_line.strip():
                                line_content = pad_to_exact_width(current_line.strip(), max_content_width, 'left')
                                wrapped_line = f"│{line_content}│"
                                wrapped_line = self._validate_border_consistency(wrapped_line, actual_display_width)
                                print(wrapped_line)
                            current_line = word + " "
                        else:
                            current_line = test_line + " "
                    
                    if current_line.strip():
                        final_content = pad_to_exact_width(current_line.strip(), max_content_width, 'left')
                        final_line = f"│{final_content}│"
                        final_line = self._validate_border_consistency(final_line, actual_display_width)
                        print(final_line)
        
        # Final border
        print(border_line)

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
    
    def update_agent_vote_target(self, agent_id: int, target_id: Optional[int]):
        """Update which agent this agent voted for."""
        self.display.update_agent_vote_target(agent_id, target_id)
    
    def update_agent_chat_round(self, agent_id: int, round_num: int):
        """Update the chat round for an agent."""
        self.display.update_agent_chat_round(agent_id, round_num)
    

    
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