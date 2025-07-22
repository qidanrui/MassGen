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
import sys
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
        self._agent_votes_cast: Dict[int, int] = {}  # Track number of votes cast by each agent
        
        # Simplified, consistent border tracking
        self._display_cache = None  # Single cache object for all dimensions
        self._last_agent_count = 0  # Track when to invalidate cache
        
        # CRITICAL FIX: Debounced display updates to prevent race conditions
        self._update_timer = None
        self._update_delay = 0.1  # 100ms debounce
        self._display_updating = False
        self._pending_update = False
        
        # Initialize logging directory and files
        if self.save_logs:
            self._setup_logging()
    
    def _get_terminal_width(self):
        """Get terminal width with conservative fallback."""
        try:
            return os.get_terminal_size().columns
        except:
            return 120  # Safe default
    
    def _calculate_layout(self, num_agents: int):
        """
        Calculate all layout dimensions in one place for consistency.
        Returns: (col_width, total_width, terminal_width)
        """
        # Invalidate cache if agent count changed or no cache exists
        if (self._display_cache is None or 
            self._last_agent_count != num_agents):
            
            terminal_width = self._get_terminal_width()
            
            # Conservative calculation to prevent overflow
            # Each column needs: content + left border (│)
            # Plus one final border (│) at the end
            border_chars = num_agents + 1  # │col1│col2│col3│
            safety_margin = 6  # Extra safety for terminal variations
            
            available_width = terminal_width - border_chars - safety_margin
            col_width = max(25, available_width // num_agents)  # Minimum 25 chars per column
            
            # Calculate actual total width used
            total_width = (col_width * num_agents) + border_chars
            
            # Final safety check - ensure we don't exceed terminal
            if total_width > terminal_width:
                col_width = max(20, (terminal_width - border_chars - 2) // num_agents)
                total_width = (col_width * num_agents) + border_chars
            
            # Cache the results
            self._display_cache = {
                'col_width': col_width,
                'total_width': total_width,
                'terminal_width': terminal_width,
                'num_agents': num_agents,
                'border_chars': border_chars
            }
            self._last_agent_count = num_agents
        
        cache = self._display_cache
        return cache['col_width'], cache['total_width'], cache['terminal_width']
    
    def _simple_display_width(self, text: str) -> int:
        """
        Simplified, reliable width calculation that focuses on accuracy over complexity.
        """
        if not text:
            return 0
        
        # Remove ANSI escape sequences
        import re
        ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_pattern.sub('', text)
        
        width = 0
        for char in clean_text:
            # Handle common wide characters and emojis conservatively
            char_code = ord(char)
            if (
                # Common emoji ranges
                (0x1F600 <= char_code <= 0x1F64F) or  # Emoticons
                (0x1F300 <= char_code <= 0x1F5FF) or  # Misc symbols
                (0x1F680 <= char_code <= 0x1F6FF) or  # Transport
                (0x1F1E6 <= char_code <= 0x1F1FF) or  # Flags
                (0x2600 <= char_code <= 0x26FF) or   # Misc symbols
                (0x2700 <= char_code <= 0x27BF) or   # Dingbats
                # East Asian wide characters
                unicodedata.east_asian_width(char) in ('F', 'W')
            ):
                width += 2
            elif char_code >= 32:  # Printable characters
                width += 1
            # Control characters add 0 width
        
        return width
    
    def _pad_to_width(self, text: str, target_width: int, align: str = 'left') -> str:
        """
        Pad text to exact target width using simplified logic.
        """
        if target_width <= 0:
            return ""
        
        current_width = self._simple_display_width(text)
        
        # Truncate if too long
        if current_width > target_width:
            # Simple truncation approach
            import re
            ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_text = ansi_pattern.sub('', text)
            
            truncated = ""
            width = 0
            for char in clean_text:
                char_width = self._simple_display_width(char)
                if width + char_width > target_width - 1:  # Save space for …
                    truncated += "…"
                    break
                truncated += char
                width += char_width
            text = truncated
            current_width = self._simple_display_width(text)
        
        # Add padding
        padding = target_width - current_width
        if padding > 0:
            if align == 'center':
                left_pad = padding // 2
                right_pad = padding - left_pad
                return " " * left_pad + text + " " * right_pad
            elif align == 'right':
                return " " * padding + text
            else:  # left
                return text + " " * padding
        
        return text
    
    def _create_bordered_line(self, content_parts: List[str], total_width: int) -> str:
        """
        Create a single bordered line with guaranteed correct width.
        """
        # Join content with borders: │content1│content2│content3│
        line = "│" + "│".join(content_parts) + "│"
        
        # Verify width matches expectation
        actual_width = self._simple_display_width(line)
        if actual_width != total_width:
            # Simple fix: truncate or pad the entire line
            if actual_width > total_width:
                # Emergency truncation
                import re
                ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_line = ansi_pattern.sub('', line)
                if len(clean_line) > total_width:
                    clean_line = clean_line[:total_width-1] + "│"
                line = clean_line
            else:
                # Add padding to reach exact width
                line += " " * (total_width - actual_width)
        
        return line
    
    def _create_system_bordered_line(self, content: str, total_width: int) -> str:
        """
        Create a system section line with borders.
        """
        content_width = total_width - 2  # Account for │ on each side
        if content_width <= 0:
            return "│" + " " * max(0, total_width - 2) + "│"
        
        padded_content = self._pad_to_width(content, content_width, 'left')
        return f"│{padded_content}│"
    
    def _invalidate_display_cache(self):
        """Reset display cache when terminal is resized."""
        self._display_cache = None
    
    def cleanup(self):
        """Clean up resources when display is no longer needed."""
        with self._lock:
            if self._update_timer:
                self._update_timer.cancel()
                self._update_timer = None
            self._pending_update = False
            self._display_updating = False
    
    def _clear_terminal_atomic(self):
        """Atomically clear terminal using proper ANSI sequences."""
        try:
            # Use ANSI escape sequences for atomic terminal clearing
            # This is more reliable than os.system('clear')
            sys.stdout.write('\033[2J')  # Clear entire screen
            sys.stdout.write('\033[H')   # Move cursor to home position
            sys.stdout.flush()           # Ensure immediate execution
        except Exception:
            # Fallback to os.system if ANSI sequences fail
            try:
                os.system('clear' if os.name == 'posix' else 'cls')
            except Exception:
                pass  # Silent fallback if all clearing methods fail
    
    def _schedule_display_update(self):
        """Schedule a debounced display update to prevent rapid refreshes."""
        with self._lock:
            if self._update_timer:
                self._update_timer.cancel()
            
            # Set pending update flag
            self._pending_update = True
            
            # Schedule update after delay
            self._update_timer = threading.Timer(self._update_delay, self._execute_display_update)
            self._update_timer.start()
    
    def _execute_display_update(self):
        """Execute the actual display update."""
        with self._lock:
            if not self._pending_update:
                return
                
            # Prevent concurrent updates
            if self._display_updating:
                # Reschedule if another update is in progress
                self._update_timer = threading.Timer(self._update_delay, self._execute_display_update)
                self._update_timer.start()
                return
            
            self._display_updating = True
            self._pending_update = False
        
        try:
            self._update_display_immediate()
        finally:
            with self._lock:
                self._display_updating = False
    
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
    
    def update_agent_votes_cast(self, agent_id: int, votes_cast: int):
        """Update the number of votes cast by an agent."""
        with self._lock:
            self._agent_votes_cast[agent_id] = votes_cast
    
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
        self.session_logs_dir = os.path.join(base_logs_dir, timestamp, "display")
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
        """FIXED: Buffered streaming with debounced display updates."""
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
            
            # CRITICAL FIX: Use debounced updates instead of immediate updates
            if display_content:
                self._schedule_display_update()
    
    def _handle_terminal_resize(self):
        """Handle terminal resize by resetting cached dimensions."""
        try:
            current_width = os.get_terminal_size().columns
            if self._display_cache and abs(current_width - self._display_cache['terminal_width']) > 5:
                # Significant change detected, reset cache
                self._invalidate_display_cache()
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
    
    def _update_display_immediate(self):
        """Immediate display update - called by the debounced scheduler."""
        if not self.display_enabled:
            return
            
        try:
            # Handle potential terminal resize
            self._handle_terminal_resize()
            
            # Use atomic terminal clearing
            self._clear_terminal_atomic()
            
            # Get sorted agent IDs for consistent ordering
            agent_ids = sorted(self.agent_outputs.keys())
            if not agent_ids:
                return
            
            # Get terminal dimensions and calculate display dimensions  
            num_agents = len(agent_ids)
            col_width, total_width, terminal_width = self._calculate_layout(num_agents)
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
        border_line = "─" * total_width
        
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
        header_top = f"{BRIGHT_CYAN}{BOLD}╔{'═' * (total_width - 2)}╗{RESET}"
        print(header_top)
        
        # Empty line
        header_empty = f"{BRIGHT_CYAN}║{' ' * (total_width - 2)}║{RESET}"
        print(header_empty)
        
        # Title line with exact centering
        title_text = "🚀 MASS - Multi-Agent Scaling System 🚀"
        title_line_content = self._pad_to_width(title_text, total_width - 2, 'center')
        title_line = f"{BRIGHT_CYAN}║{BRIGHT_YELLOW}{BOLD}{title_line_content}{RESET}{BRIGHT_CYAN}║{RESET}"
        print(title_line)
        
        # Subtitle line
        subtitle_text = "🔬 Advanced Agent Collaboration Framework"
        subtitle_line_content = self._pad_to_width(subtitle_text, total_width - 2, 'center')
        subtitle_line = f"{BRIGHT_CYAN}║{BRIGHT_GREEN}{subtitle_line_content}{RESET}{BRIGHT_CYAN}║{RESET}"
        print(subtitle_line)
        
        # Empty line and bottom border
        print(header_empty)
        header_bottom = f"{BRIGHT_CYAN}{BOLD}╚{'═' * (total_width - 2)}╝{RESET}"
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
            
            header_content = self._pad_to_width(agent_header, col_width, 'center')
            header_parts.append(header_content)
        
        # Print agent header line with exact borders
        header_line = self._create_bordered_line(header_parts, total_width)
        print(header_line)
        
        # Agent state information line
        state_parts = []
        for agent_id in agent_ids:
            chat_round = getattr(self, '_agent_chat_rounds', {}).get(agent_id, 0)
            vote_target = getattr(self, '_agent_vote_targets', {}).get(agent_id)
            update_count = getattr(self, '_agent_update_counts', {}).get(agent_id, 0)
            votes_cast = getattr(self, '_agent_votes_cast', {}).get(agent_id, 0)
            
            # Format state info with better handling of color codes (removed redundant status)
            state_info = []
            state_info.append(f"{BRIGHT_WHITE}Round:{RESET} {BRIGHT_GREEN}{chat_round}{RESET}")
            state_info.append(f"{BRIGHT_WHITE}#Updates:{RESET} {BRIGHT_MAGENTA}{update_count}{RESET}")
            state_info.append(f"{BRIGHT_WHITE}#Votes:{RESET} {BRIGHT_CYAN}{votes_cast}{RESET}")
            if vote_target:
                state_info.append(f"{BRIGHT_WHITE}Vote →{RESET} {BRIGHT_GREEN}{vote_target}{RESET}")
            else:
                state_info.append(f"{BRIGHT_WHITE}Vote:{RESET} None")
            
            state_text = f"📊 {' | '.join(state_info)}"
            # Ensure exact column width with improved padding
            state_content = self._pad_to_width(state_text, col_width, 'center')
            state_parts.append(state_content)
        
        # Validate state line consistency before printing
        state_line = self._create_bordered_line(state_parts, total_width)
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
                    max_path_len = max(10, col_width - self._simple_display_width(prefix) - 8)
                    if len(display_path) > max_path_len:
                        display_path = "..." + display_path[-(max_path_len-3):]
                    
                    link_text = f"{prefix}{UNDERLINE}{display_path}{RESET}"
                    link_content = self._pad_to_width(link_text, col_width, 'center')
                else:
                    link_content = self._pad_to_width("", col_width, 'center')
                link_parts.append(link_content)
            
            # Validate log line consistency
            log_line = self._create_bordered_line(link_parts, total_width)
            print(log_line)
        
        print(border_line)
        
        # Content area with perfect column alignment - Apply validation to every content line
        for line_idx in range(max_lines):
            content_parts = []
            for agent_id in agent_ids:
                lines = agent_lines[agent_id]
                content = lines[line_idx] if line_idx < len(lines) else ""
                
                # Ensure exact column width for each content piece
                padded_content = self._pad_to_width(content, col_width, 'left')
                content_parts.append(padded_content)
            
            # Apply border validation to every content line for consistency
            content_line = self._create_bordered_line(content_parts, total_width)
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
            system_header_line = self._create_system_bordered_line(system_header_text, total_width)
            print(system_header_line)
            
            # System log file link
            if self.save_logs and hasattr(self, 'system_log_file'):
                system_log_path = self.get_system_log_path_for_display()
                if system_log_path:
                    UNDERLINE = '\033[4m'
                    display_path = system_log_path.replace(os.getcwd() + "/", "") if system_log_path.startswith(os.getcwd()) else system_log_path
                    
                    # Safe path truncation with consistent width handling
                    prefix = "📁 Log: "
                    max_path_len = max(10, total_width - self._simple_display_width(prefix) - 15)
                    if len(display_path) > max_path_len:
                        display_path = "..." + display_path[-(max_path_len-3):]
                    
                    system_link_text = f"{prefix}{UNDERLINE}{display_path}{RESET}"
                    system_link_line = self._create_system_bordered_line(system_link_text, total_width)
                    print(system_link_line)
            
            print(border_line)
            
            # System messages with exact width and validation
            if self.consensus_reached and self.representative_agent_id is not None:
                consensus_msg = f"🎉 CONSENSUS REACHED! Representative: Agent {self.representative_agent_id}"
                consensus_line = self._create_system_bordered_line(consensus_msg, total_width)
                print(consensus_line)
            
            # Vote distribution with validation
            if self.vote_distribution:
                vote_msg = "🗳️  Vote Distribution: " + ", ".join([f"Agent {k}→{v} votes" for k, v in self.vote_distribution.items()])
                
                # Use the new safe wrapping method
                max_content_width = total_width - 2
                if self._simple_display_width(vote_msg) <= max_content_width:
                    vote_line = self._create_system_bordered_line(vote_msg, total_width)
                    print(vote_line)
                else:
                    # Wrap vote distribution using safe method
                    vote_header = "🗳️  Vote Distribution:"
                    header_line = self._create_system_bordered_line(vote_header, total_width)
                    print(header_line)
                    
                    for agent_id, votes in self.vote_distribution.items():
                        vote_detail = f"   Agent {agent_id}: {votes} votes"
                        detail_line = self._create_system_bordered_line(vote_detail, total_width)
                        print(detail_line)
            
            # Regular system messages with validation
            for message in self.system_messages:
                # The original _wrap_and_print_system_message used _get_display_width_unified,
                # which was a simplified width calculation.
                # Now we use _simple_display_width for consistency.
                max_content_width = total_width - 2
                if self._simple_display_width(message) <= max_content_width:
                    line = self._create_system_bordered_line(message, total_width)
                    print(line)
                else:
                    # Simple word wrapping
                    words = message.split()
                    current_line = ""
                    
                    for word in words:
                        test_line = f"{current_line} {word}".strip()
                        if self._simple_display_width(test_line) > max_content_width:
                            # Print current line if it has content
                            if current_line.strip():
                                line = self._create_system_bordered_line(current_line.strip(), total_width)
                                print(line)
                            current_line = word
                        else:
                            current_line = test_line
                    
                    # Print final line if it has content
                    if current_line.strip():
                        line = self._create_system_bordered_line(current_line.strip(), total_width)
                        print(line)
        
        # Final border
        print(border_line)
        
        # Force output to be written immediately
        sys.stdout.flush()
    
    def force_update_display(self):
        """Force an immediate display update (for status changes)."""
        with self._lock:
            if self._update_timer:
                self._update_timer.cancel()
            self._pending_update = True
        self._execute_display_update()

class StreamingOrchestrator:
    def __init__(self, display_enabled: bool = True, stream_callback: Optional[Callable] = None, max_lines: int = 30, save_logs: bool = True):
        self.display = MultiRegionDisplay(display_enabled, max_lines, save_logs)
        self.stream_callback = stream_callback
    
    def stream_output(self, agent_id: int, content: str):
        """Streaming content - uses debounced updates."""
        self.display.stream_output_sync(agent_id, content)
        if self.stream_callback:
            try:
                self.stream_callback(agent_id, content)
            except Exception:
                pass
    
    def set_agent_model(self, agent_id: int, model_name: str):
        """Set agent model - immediate update."""
        self.display.set_agent_model(agent_id, model_name)
        self.display.force_update_display()
    
    def update_agent_status(self, agent_id: int, status: str):
        """Update agent status - immediate update for critical state changes."""
        self.display.update_agent_status(agent_id, status)
        self.display.force_update_display()
    
    def update_phase(self, old_phase: str, new_phase: str):
        """Update phase - immediate update for critical state changes."""
        self.display.update_phase(old_phase, new_phase)
        self.display.force_update_display()
    
    def update_vote_distribution(self, vote_dist: Dict[int, int]):
        """Update vote distribution - immediate update for critical state changes."""
        self.display.update_vote_distribution(vote_dist)
        self.display.force_update_display()
    
    def update_consensus_status(self, representative_id: int, vote_dist: Dict[int, int]):
        """Update consensus status - immediate update for critical state changes."""
        self.display.update_consensus_status(representative_id, vote_dist)
        self.display.force_update_display()
    
    def reset_consensus(self):
        """Reset consensus - immediate update for critical state changes."""
        self.display.reset_consensus()
        self.display.force_update_display()
    
    def add_system_message(self, message: str):
        """Add system message - immediate update for important messages."""
        self.display.add_system_message(message)
        self.display.force_update_display()
    
    def update_agent_vote_target(self, agent_id: int, target_id: Optional[int]):
        """Update agent vote target - immediate update for critical state changes."""
        self.display.update_agent_vote_target(agent_id, target_id)
        self.display.force_update_display()
    
    def update_agent_chat_round(self, agent_id: int, round_num: int):
        """Update agent chat round - debounced update."""
        self.display.update_agent_chat_round(agent_id, round_num)
        # Don't force immediate update for chat rounds
    
    def update_agent_update_count(self, agent_id: int, count: int):
        """Update agent update count - debounced update."""
        self.display.update_agent_update_count(agent_id, count)
        # Don't force immediate update for update counts
    
    def update_agent_votes_cast(self, agent_id: int, votes_cast: int):
        """Update agent votes cast - immediate update for vote-related changes."""
        self.display.update_agent_votes_cast(agent_id, votes_cast)
        self.display.force_update_display()
    
    def update_debate_rounds(self, rounds: int):
        """Update debate rounds - immediate update for critical state changes."""
        self.display.update_debate_rounds(rounds)
        self.display.force_update_display()
    
    def format_agent_notification(self, agent_id: int, notification_type: str, content: str):
        """Format agent notifications - immediate update for notifications."""
        self.display.format_agent_notification(agent_id, notification_type, content)
        self.display.force_update_display()
    
    def get_agent_log_path(self, agent_id: int) -> str:
        """Get the log file path for a specific agent."""
        return self.display.get_agent_log_path_for_display(agent_id)
    
    def get_system_log_path(self) -> str:
        """Get the system log file path."""
        return self.display.get_system_log_path_for_display()
    
    def cleanup(self):
        """Clean up resources when orchestrator is no longer needed."""
        self.display.cleanup()

def create_streaming_display(display_enabled: bool = True, stream_callback: Optional[Callable] = None, max_lines: int = 30, save_logs: bool = True) -> StreamingOrchestrator:
    """Create a streaming orchestrator with display capabilities."""
    return StreamingOrchestrator(display_enabled, stream_callback, max_lines, save_logs)