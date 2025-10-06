"""
Session recorder for managing typing sessions and metadata.

This module handles session management, including session metadata,
stress level annotations, and session organization.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class SessionMetadata:
    """Metadata for a typing session."""
    session_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    stress_level: Optional[int] = None  # 1-5 scale
    fatigue_level: Optional[int] = None  # 1-5 scale
    notes: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()


class SessionRecorder:
    """
    Manages typing sessions and their metadata.
    
    This class handles:
    - Session creation and management
    - Stress/fatigue level annotations
    - Session metadata storage
    - Session retrieval and organization
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the session recorder.
        
        Args:
            data_dir: Directory to store session data
        """
        self.data_dir = data_dir
        self.sessions_dir = os.path.join(data_dir, "sessions")
        self.metadata_file = os.path.join(data_dir, "sessions_metadata.json")
        
        # Ensure directories exist
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Load existing metadata
        self.sessions_metadata = self._load_metadata()
    
    def create_session(self, session_name: str, tags: List[str] = None) -> SessionMetadata:
        """
        Create a new typing session.
        
        Args:
            session_name: Name for the session
            tags: Optional tags for categorization
            
        Returns:
            SessionMetadata object for the new session
        """
        if tags is None:
            tags = []
        
        session = SessionMetadata(
            session_name=session_name,
            start_time=datetime.now(),
            tags=tags
        )
        
        # Add to metadata
        self.sessions_metadata[session_name] = asdict(session)
        self._save_metadata()
        
        return session
    
    def end_session(self, session_name: str, stress_level: int = None, 
                   fatigue_level: int = None, notes: str = None) -> bool:
        """
        End a typing session and add annotations.
        
        Args:
            session_name: Name of the session to end
            stress_level: Self-reported stress level (1-5)
            fatigue_level: Self-reported fatigue level (1-5)
            notes: Optional notes about the session
            
        Returns:
            True if session was found and ended successfully
        """
        if session_name not in self.sessions_metadata:
            return False
        
        session_data = self.sessions_metadata[session_name]
        session_data["end_time"] = datetime.now().isoformat()
        session_data["stress_level"] = stress_level
        session_data["fatigue_level"] = fatigue_level
        session_data["notes"] = notes
        
        # Calculate duration
        start_time = datetime.fromisoformat(session_data["start_time"])
        end_time = datetime.fromisoformat(session_data["end_time"])
        session_data["duration"] = (end_time - start_time).total_seconds()
        
        self._save_metadata()
        return True
    
    def get_session(self, session_name: str) -> Optional[SessionMetadata]:
        """
        Get session metadata by name.
        
        Args:
            session_name: Name of the session
            
        Returns:
            SessionMetadata object or None if not found
        """
        if session_name not in self.sessions_metadata:
            return None
        
        data = self.sessions_metadata[session_name]
        return SessionMetadata(
            session_name=data["session_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            duration=data.get("duration"),
            stress_level=data.get("stress_level"),
            fatigue_level=data.get("fatigue_level"),
            notes=data.get("notes"),
            tags=data.get("tags", [])
        )
    
    def list_sessions(self, tags: List[str] = None, completed_only: bool = True) -> List[SessionMetadata]:
        """
        List all sessions, optionally filtered by tags.
        
        Args:
            tags: Filter sessions by these tags
            completed_only: Only return completed sessions
            
        Returns:
            List of SessionMetadata objects
        """
        sessions = []
        
        for session_data in self.sessions_metadata.values():
            # Filter by completion status
            if completed_only and not session_data.get("end_time"):
                continue
            
            # Filter by tags
            if tags and not any(tag in session_data.get("tags", []) for tag in tags):
                continue
            
            session = SessionMetadata(
                session_name=session_data["session_name"],
                start_time=datetime.fromisoformat(session_data["start_time"]),
                end_time=datetime.fromisoformat(session_data["end_time"]) if session_data.get("end_time") else None,
                duration=session_data.get("duration"),
                stress_level=session_data.get("stress_level"),
                fatigue_level=session_data.get("fatigue_level"),
                notes=session_data.get("notes"),
                tags=session_data.get("tags", [])
            )
            sessions.append(session)
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.start_time, reverse=True)
        return sessions
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all sessions.
        
        Returns:
            Dictionary with session statistics
        """
        completed_sessions = self.list_sessions(completed_only=True)
        
        if not completed_sessions:
            return {
                "total_sessions": 0,
                "completed_sessions": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "stress_levels": {},
                "fatigue_levels": {}
            }
        
        total_duration = sum(s.duration or 0 for s in completed_sessions)
        stress_levels = {}
        fatigue_levels = {}
        
        for session in completed_sessions:
            if session.stress_level:
                stress_levels[session.stress_level] = stress_levels.get(session.stress_level, 0) + 1
            if session.fatigue_level:
                fatigue_levels[session.fatigue_level] = fatigue_levels.get(session.fatigue_level, 0) + 1
        
        return {
            "total_sessions": len(self.sessions_metadata),
            "completed_sessions": len(completed_sessions),
            "total_duration": total_duration,
            "avg_duration": total_duration / len(completed_sessions) if completed_sessions else 0,
            "stress_levels": stress_levels,
            "fatigue_levels": fatigue_levels
        }
    
    def delete_session(self, session_name: str) -> bool:
        """
        Delete a session and its metadata.
        
        Args:
            session_name: Name of the session to delete
            
        Returns:
            True if session was found and deleted
        """
        if session_name not in self.sessions_metadata:
            return False
        
        # Remove from metadata
        del self.sessions_metadata[session_name]
        self._save_metadata()
        
        # Try to delete associated log file
        log_file = os.path.join(self.data_dir, "logs", f"{session_name}.json")
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except OSError:
                pass  # Ignore errors if file can't be deleted
        
        return True
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load session metadata from file."""
        if not os.path.exists(self.metadata_file):
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_metadata(self) -> None:
        """Save session metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.sessions_metadata, f, indent=2)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KeyStress Session Recorder")
    parser.add_argument("--list", action="store_true", help="List all sessions")
    parser.add_argument("--stats", action="store_true", help="Show session statistics")
    parser.add_argument("--create", type=str, help="Create a new session")
    parser.add_argument("--end", type=str, help="End a session")
    parser.add_argument("--stress", type=int, help="Stress level (1-5)")
    parser.add_argument("--fatigue", type=int, help="Fatigue level (1-5)")
    parser.add_argument("--notes", type=str, help="Session notes")
    
    args = parser.parse_args()
    
    recorder = SessionRecorder()
    
    if args.list:
        sessions = recorder.list_sessions()
        print(f"Found {len(sessions)} completed sessions:")
        for session in sessions:
            print(f"  {session.session_name}: {session.duration:.1f}s")
            if session.stress_level:
                print(f"    Stress: {session.stress_level}/5")
            if session.fatigue_level:
                print(f"    Fatigue: {session.fatigue_level}/5")
    
    elif args.stats:
        stats = recorder.get_session_stats()
        print("Session Statistics:")
        print(f"  Total sessions: {stats['total_sessions']}")
        print(f"  Completed sessions: {stats['completed_sessions']}")
        print(f"  Total duration: {stats['total_duration']:.1f}s")
        print(f"  Average duration: {stats['avg_duration']:.1f}s")
    
    elif args.create:
        session = recorder.create_session(args.create)
        print(f"Created session: {session.session_name}")
    
    elif args.end:
        success = recorder.end_session(
            args.end, 
            stress_level=args.stress,
            fatigue_level=args.fatigue,
            notes=args.notes
        )
        if success:
            print(f"Ended session: {args.end}")
        else:
            print(f"Session not found: {args.end}")


if __name__ == "__main__":
    main()
