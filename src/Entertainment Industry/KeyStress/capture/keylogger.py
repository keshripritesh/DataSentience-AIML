"""
Keystroke logger for capturing typing behavior.

This module provides a non-intrusive keystroke logger that captures
typing patterns for stress analysis.
"""

import time
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Callable
from pynput import keyboard
from pynput.keyboard import Key, KeyCode


class KeyLogger:
    """
    A keystroke logger that captures typing behavior for stress analysis.
    
    This logger records:
    - Key press events with timestamps
    - Inter-key intervals
    - Error corrections (backspace, delete)
    - Session metadata
    """
    
    def __init__(self, session_name: str = None, output_dir: str = "data/logs"):
        """
        Initialize the key logger.
        
        Args:
            session_name: Name for this typing session
            output_dir: Directory to save log files
        """
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, f"{self.session_name}.json")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Session data
        self.keystrokes: List[Dict] = []
        self.session_start = None
        self.session_end = None
        self.is_recording = False
        
        # Keyboard listener
        self.listener: Optional[keyboard.Listener] = None
        
        # Callbacks
        self.on_key_press: Optional[Callable] = None
        self.on_session_end: Optional[Callable] = None
    
    def start_recording(self) -> None:
        """Start recording keystrokes."""
        if self.is_recording:
            print("Already recording!")
            return
        
        self.session_start = time.time()
        self.is_recording = True
        self.keystrokes = []
        
        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()
        
        print(f"Started recording session: {self.session_name}")
        print("Press 'Ctrl+Shift+Q' to stop recording")
    
    def stop_recording(self) -> None:
        """Stop recording keystrokes and save the session."""
        if not self.is_recording:
            print("Not currently recording!")
            return
        
        self.is_recording = False
        self.session_end = time.time()
        
        if self.listener:
            self.listener.stop()
            self.listener = None
        
        # Save session data
        self._save_session()
        
        if self.on_session_end:
            self.on_session_end(self.keystrokes)
        
        print(f"Stopped recording. Session saved to: {self.log_file}")
        print(f"Session duration: {self.session_end - self.session_start:.2f} seconds")
        print(f"Total keystrokes: {len(self.keystrokes)}")
    
    def _on_key_press(self, key) -> None:
        """Handle key press events."""
        if not self.is_recording:
            return
        
        # Check for stop combination (Ctrl+Shift+Q)
        if hasattr(key, 'char') and key.char == 'q':
            # Check if Ctrl and Shift are pressed
            with keyboard.Controller() as controller:
                if controller.alt_pressed and controller.shift_pressed:
                    self.stop_recording()
                    return
        
        # Record the keystroke
        keystroke_data = {
            "key": self._key_to_string(key),
            "time": time.time() - self.session_start,
            "timestamp": datetime.now().isoformat()
        }
        
        self.keystrokes.append(keystroke_data)
        
        if self.on_key_press:
            self.on_key_press(keystroke_data)
    
    def _on_key_release(self, key) -> None:
        """Handle key release events (currently not used but required by pynput)."""
        pass
    
    def _key_to_string(self, key) -> str:
        """Convert key object to string representation."""
        if hasattr(key, 'char') and key.char:
            return key.char
        elif key == Key.space:
            return "space"
        elif key == Key.enter:
            return "enter"
        elif key == Key.backspace:
            return "backspace"
        elif key == Key.delete:
            return "delete"
        elif key == Key.tab:
            return "tab"
        elif key == Key.esc:
            return "escape"
        elif hasattr(key, 'name'):
            return key.name
        else:
            return str(key)
    
    def _save_session(self) -> None:
        """Save session data to JSON file."""
        session_data = {
            "session_name": self.session_name,
            "session_start": self.session_start,
            "session_end": self.session_end,
            "duration": self.session_end - self.session_start,
            "total_keystrokes": len(self.keystrokes),
            "keystrokes": self.keystrokes
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def get_session_summary(self) -> Dict:
        """Get a summary of the recorded session."""
        if not self.keystrokes:
            return {}
        
        # Calculate basic statistics
        intervals = []
        for i in range(1, len(self.keystrokes)):
            interval = self.keystrokes[i]["time"] - self.keystrokes[i-1]["time"]
            intervals.append(interval)
        
        # Count errors (backspace, delete)
        errors = sum(1 for k in self.keystrokes if k["key"] in ["backspace", "delete"])
        
        return {
            "session_name": self.session_name,
            "duration": self.session_end - self.session_start if self.session_end else 0,
            "total_keystrokes": len(self.keystrokes),
            "avg_interval": sum(intervals) / len(intervals) if intervals else 0,
            "error_count": errors,
            "error_rate": errors / len(self.keystrokes) if self.keystrokes else 0
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KeyStress Keystroke Logger")
    parser.add_argument("--session", type=str, help="Session name")
    parser.add_argument("--output-dir", type=str, default="data/logs", help="Output directory")
    
    args = parser.parse_args()
    
    logger = KeyLogger(session_name=args.session, output_dir=args.output_dir)
    
    try:
        logger.start_recording()
        # Keep the script running
        while logger.is_recording:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.stop_recording()


if __name__ == "__main__":
    main()
