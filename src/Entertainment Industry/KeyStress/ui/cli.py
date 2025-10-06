"""
Command-line interface for KeyStress.

This module provides a command-line interface for:
- Starting/stopping keystroke monitoring
- Real-time stress indicators
- Session management
- Model training and evaluation
"""

import argparse
import sys
import time
import threading
from typing import Optional
import click

from capture.keylogger import KeyLogger
from capture.session_recorder import SessionRecorder
from features.extractor import FeatureExtractor
from features.stress_indicators import StressIndicators
from ml.dataset import Dataset
from ml.train import ModelTrainer


class CLI:
    """
    Command-line interface for KeyStress.
    
    This class provides an interactive CLI for managing keystroke monitoring,
    stress analysis, and model training.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.keylogger = None
        self.session_recorder = SessionRecorder()
        self.feature_extractor = FeatureExtractor()
        self.stress_analyzer = StressIndicators()
        self.is_monitoring = False
        self.monitoring_thread = None
    
    def start_monitoring(self, session_name: str, duration: Optional[int] = None):
        """
        Start keystroke monitoring.
        
        Args:
            session_name: Name for the monitoring session
            duration: Duration in seconds (optional)
        """
        if self.is_monitoring:
            print("Monitoring is already active!")
            return
        
        print(f"Starting keystroke monitoring for session: {session_name}")
        print("Press Ctrl+C to stop monitoring")
        
        # Create session
        session = self.session_recorder.create_session(session_name)
        
        # Initialize keylogger
        self.keylogger = KeyLogger(session_name=session_name)
        
        # Start monitoring
        self.is_monitoring = True
        self.keylogger.start_recording()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(duration,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        try:
            # Keep main thread alive
            while self.is_monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop keystroke monitoring."""
        if not self.is_monitoring:
            print("No active monitoring session")
            return
        
        print("\nStopping monitoring...")
        self.is_monitoring = False
        
        if self.keylogger:
            self.keylogger.stop_recording()
        
        # Get stress level from user
        stress_level = self._get_stress_level()
        
        # End session with stress level
        session_name = self.keylogger.session_name if self.keylogger else "unknown"
        self.session_recorder.end_session(session_name, stress_level=stress_level)
        
        print("Monitoring stopped. Session saved.")
    
    def _monitoring_loop(self, duration: Optional[int]):
        """Monitoring loop for real-time analysis."""
        start_time = time.time()
        
        while self.is_monitoring:
            if duration and (time.time() - start_time) > duration:
                print(f"\nMonitoring duration ({duration}s) reached. Stopping...")
                self.stop_monitoring()
                break
            
            # Real-time analysis (every 10 seconds)
            if self.keylogger and len(self.keylogger.keystrokes) > 10:
                self._analyze_realtime()
            
            time.sleep(10)
    
    def _analyze_realtime(self):
        """Perform real-time stress analysis."""
        if not self.keylogger or len(self.keylogger.keystrokes) < 10:
            return
        
        # Extract features from recent keystrokes
        features = self.feature_extractor.extract_features(self.keylogger.keystrokes)
        
        # Calculate stress score
        stress_score = self.stress_analyzer.calculate_stress_score(features)
        stress_level = self.stress_analyzer.classify_stress_level(features)
        
        # Display real-time indicator
        self._display_stress_indicator(stress_score, stress_level)
    
    def _display_stress_indicator(self, stress_score: float, stress_level):
        """Display real-time stress indicator."""
        # Clear line and display indicator
        print(f"\rStress Level: {stress_level.name} ({stress_score:.2f})", end="", flush=True)
    
    def _get_stress_level(self) -> Optional[int]:
        """Get stress level from user input."""
        print("\nPlease rate your stress level during this session:")
        print("1 - Low stress")
        print("2 - Medium stress")
        print("3 - High stress")
        
        while True:
            try:
                level = input("Enter stress level (1-3): ").strip()
                if level in ['1', '2', '3']:
                    return int(level)
                else:
                    print("Please enter 1, 2, or 3")
            except KeyboardInterrupt:
                return None
    
    def list_sessions(self):
        """List all recorded sessions."""
        sessions = self.session_recorder.list_sessions()
        
        if not sessions:
            print("No sessions found")
            return
        
        print(f"Found {len(sessions)} sessions:")
        print("-" * 60)
        
        for session in sessions:
            print(f"Session: {session.session_name}")
            print(f"  Duration: {session.duration:.1f}s")
            if session.stress_level:
                print(f"  Stress Level: {session.stress_level}/3")
            if session.fatigue_level:
                print(f"  Fatigue Level: {session.fatigue_level}/3")
            print()
    
    def analyze_session(self, session_name: str):
        """
        Analyze a specific session.
        
        Args:
            session_name: Name of the session to analyze
        """
        # Load session data
        log_file = f"data/logs/{session_name}.json"
        
        try:
            features = self.feature_extractor.extract_features_from_file(log_file)
            report = self.stress_analyzer.generate_stress_report(features, session_name)
            
            self._display_analysis_report(report)
            
        except FileNotFoundError:
            print(f"Session {session_name} not found")
        except Exception as e:
            print(f"Error analyzing session: {e}")
    
    def _display_analysis_report(self, report: dict):
        """Display stress analysis report."""
        print(f"\n{'='*50}")
        print(f"STRESS ANALYSIS: {report['session_name']}")
        print(f"{'='*50}")
        
        print(f"Stress Score: {report['stress_score']:.3f}")
        print(f"Stress Level: {report['stress_level']}")
        print(f"Stressful Indicators: {report['stressful_indicators_count']}")
        
        print("\nTop Contributors:")
        for contrib in report['top_contributors']:
            print(f"  {contrib['category']}: {contrib['contribution']:.3f}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    def train_model(self, model_type: str = "random_forest"):
        """
        Train a stress detection model.
        
        Args:
            model_type: Type of model to train
        """
        print(f"Training {model_type} model...")
        
        trainer = ModelTrainer()
        
        # Prepare data
        if not trainer.prepare_data():
            print("Failed to prepare data for training")
            return
        
        # Train model
        try:
            results = trainer.train_model(model_type=model_type)
            print("Model training completed successfully!")
            
            # Display results
            accuracy = results['evaluation_results']['accuracy']
            print(f"Model accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Error training model: {e}")
    
    def show_stats(self):
        """Show session statistics."""
        stats = self.session_recorder.get_session_stats()
        
        print("Session Statistics:")
        print(f"  Total sessions: {stats['total_sessions']}")
        print(f"  Completed sessions: {stats['completed_sessions']}")
        print(f"  Total duration: {stats['total_duration']:.1f}s")
        print(f"  Average duration: {stats['avg_duration']:.1f}s")
        
        if stats['stress_levels']:
            print("\nStress Level Distribution:")
            for level, count in stats['stress_levels'].items():
                print(f"  Level {level}: {count} sessions")


@click.group()
def cli():
    """KeyStress - Keystroke-based stress detection system."""
    pass


@cli.command()
@click.option('--session', '-s', required=True, help='Session name')
@click.option('--duration', '-d', type=int, help='Duration in seconds')
def monitor(session, duration):
    """Start keystroke monitoring."""
    cli_instance = CLI()
    cli_instance.start_monitoring(session, duration)


@cli.command()
def sessions():
    """List all recorded sessions."""
    cli_instance = CLI()
    cli_instance.list_sessions()


@cli.command()
@click.option('--session', '-s', required=True, help='Session name to analyze')
def analyze(session):
    """Analyze a specific session."""
    cli_instance = CLI()
    cli_instance.analyze_session(session)


@cli.command()
@click.option('--model-type', '-m', default='random_forest', 
              type=click.Choice(['logistic', 'random_forest', 'svm']),
              help='Type of model to train')
def train(model_type):
    """Train a stress detection model."""
    cli_instance = CLI()
    cli_instance.train_model(model_type)


@cli.command()
def stats():
    """Show session statistics."""
    cli_instance = CLI()
    cli_instance.show_stats()


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
