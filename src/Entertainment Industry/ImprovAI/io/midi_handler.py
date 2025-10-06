"""
Advanced MIDI handler for ImprovAI.
Handles MIDI file reading, writing, and processing with sophisticated features.
"""

import pretty_midi
import mido
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import json

from core.encoders.music_encoder import Note, MusicalSequence
from utils.config import get_config

logger = logging.getLogger(__name__)


class MIDIHandler:
    """
    Advanced MIDI handler for reading, writing, and processing MIDI files.
    
    Features:
    - Multi-format MIDI support
    - Real-time MIDI processing
    - Advanced metadata handling
    - Quality assessment
    - Batch processing capabilities
    """
    
    def __init__(self, config=None):
        """Initialize the MIDI handler."""
        self.config = config or get_config()
        self.supported_formats = ['.mid', '.midi', '.kar']
        
        logger.info("MIDIHandler initialized")
    
    def read_midi_file(self, filepath: str) -> MusicalSequence:
        """
        Read a MIDI file and convert to MusicalSequence.
        
        Args:
            filepath: Path to MIDI file
            
        Returns:
            MusicalSequence object
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"MIDI file not found: {filepath}")
            
            if filepath.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            # Use pretty_midi for reading
            midi_data = pretty_midi.PrettyMIDI(str(filepath))
            
            # Extract notes from all instruments
            all_notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    # Filter notes within our range
                    if (self.config.audio.note_range[0] <= note.pitch <= 
                        self.config.audio.note_range[1]):
                        note_obj = Note(
                            pitch=note.pitch,
                            velocity=note.velocity,
                            start_time=note.start,
                            end_time=note.end,
                            duration=note.end - note.start
                        )
                        all_notes.append(note_obj)
            
            # Sort notes by start time
            all_notes.sort(key=lambda x: x.start_time)
            
            # Extract metadata
            tempo = midi_data.estimate_tempo() if midi_data.tempo_changes else 120.0
            time_signature = self._extract_time_signature(midi_data)
            key_signature = self._extract_key_signature(midi_data)
            
            # Create musical sequence
            sequence = MusicalSequence(
                notes=all_notes,
                tempo=tempo,
                time_signature=time_signature,
                key_signature=key_signature,
                total_duration=midi_data.get_end_time()
            )
            
            logger.info(f"Successfully read MIDI file: {filepath}")
            logger.info(f"Extracted {len(all_notes)} notes, duration: {sequence.total_duration:.2f}s")
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error reading MIDI file {filepath}: {e}")
            raise
    
    def write_midi_file(self, sequence: MusicalSequence, filepath: str,
                       metadata: Optional[Dict] = None) -> bool:
        """
        Write a MusicalSequence to a MIDI file.
        
        Args:
            sequence: MusicalSequence to write
            filepath: Output file path
            metadata: Optional metadata to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Create PrettyMIDI object
            midi_data = pretty_midi.PrettyMIDI(initial_tempo=sequence.tempo)
            
            # Create piano instrument
            piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
            piano = pretty_midi.Instrument(program=piano_program, name='Piano')
            
            # Convert notes to PrettyMIDI format
            for note in sequence.notes:
                pretty_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start_time,
                    end=note.end_time
                )
                piano.notes.append(pretty_note)
            
            # Add instrument to MIDI data
            midi_data.instruments.append(piano)
            
            # Add time signature
            if sequence.time_signature:
                time_sig = pretty_midi.TimeSignature(
                    numerator=sequence.time_signature[0],
                    denominator=sequence.time_signature[1],
                    time=0.0
                )
                midi_data.time_signature_changes.append(time_sig)
            
            # Add key signature
            if sequence.key_signature:
                key_sig = self._create_key_signature(sequence.key_signature, 0.0)
                if key_sig:
                    midi_data.key_signature_changes.append(key_sig)
            
            # Add metadata
            if metadata:
                self._add_metadata(midi_data, metadata)
            
            # Write to file
            midi_data.write(str(filepath))
            
            logger.info(f"Successfully wrote MIDI file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing MIDI file {filepath}: {e}")
            return False
    
    def _extract_time_signature(self, midi_data: pretty_midi.PrettyMIDI) -> Tuple[int, int]:
        """Extract time signature from MIDI data."""
        if midi_data.time_signature_changes:
            # Use the first time signature
            ts = midi_data.time_signature_changes[0]
            return (ts.numerator, ts.denominator)
        else:
            # Default to 4/4
            return (4, 4)
    
    def _extract_key_signature(self, midi_data: pretty_midi.PrettyMIDI) -> str:
        """Extract key signature from MIDI data."""
        if midi_data.key_signature_changes:
            # Use the first key signature
            ks = midi_data.key_signature_changes[0]
            return self._key_number_to_name(ks.key_number)
        else:
            # Try to estimate from notes
            all_notes = []
            for instrument in midi_data.instruments:
                all_notes.extend(instrument.notes)
            
            if all_notes:
                return self._estimate_key_from_notes(all_notes)
            else:
                return "C"
    
    def _key_number_to_name(self, key_number: int) -> str:
        """Convert key number to key name."""
        # Key numbers: -7 to 7, where 0 = C major
        key_names = {
            -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
            0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#"
        }
        return key_names.get(key_number, "C")
    
    def _estimate_key_from_notes(self, notes: List[pretty_midi.Note]) -> str:
        """Estimate key signature from note distribution."""
        # Extract pitch classes
        pitch_classes = [note.pitch % 12 for note in notes]
        
        # Create histogram
        hist = np.zeros(12)
        for pc in pitch_classes:
            hist[pc] += 1
        
        # Normalize
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        # Use Krumhansl-Kessler profiles for key estimation
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        major_profile = major_profile / np.sum(major_profile)
        
        # Calculate correlations for all possible keys
        correlations = []
        for i in range(12):
            rotated_hist = np.roll(hist, i)
            corr = np.corrcoef(rotated_hist, major_profile)[0, 1]
            correlations.append(corr)
        
        # Find best match
        best_key = np.argmax(correlations)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        return key_names[best_key]
    
    def _create_key_signature(self, key_name: str, time: float) -> Optional[pretty_midi.KeySignature]:
        """Create a key signature object."""
        try:
            # Convert key name to key number
            key_numbers = {
                'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
                'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7
            }
            
            # Extract base key (remove 'maj' or 'min')
            base_key = key_name.replace('maj', '').replace('min', '')
            
            if base_key in key_numbers:
                key_number = key_numbers[base_key]
                return pretty_midi.KeySignature(key_number, time)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not create key signature for {key_name}: {e}")
            return None
    
    def _add_metadata(self, midi_data: pretty_midi.PrettyMIDI, metadata: Dict):
        """Add metadata to MIDI file."""
        # Add text events for metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                text_event = pretty_midi.TextEvent(
                    text=f"{key}: {value}",
                    time=0.0
                )
                midi_data.text_events.append(text_event)
    
    def process_midi_batch(self, input_dir: str, output_dir: str,
                          processor_func=None) -> Dict[str, bool]:
        """
        Process multiple MIDI files in batch.
        
        Args:
            input_dir: Directory containing input MIDI files
            output_dir: Directory for output files
            processor_func: Optional function to process each sequence
            
        Returns:
            Dictionary mapping filenames to success status
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for midi_file in input_path.glob("*.mid*"):
            try:
                # Read MIDI file
                sequence = self.read_midi_file(str(midi_file))
                
                # Apply processor function if provided
                if processor_func:
                    sequence = processor_func(sequence)
                
                # Write processed file
                output_file = output_path / f"processed_{midi_file.name}"
                success = self.write_midi_file(sequence, str(output_file))
                
                results[midi_file.name] = success
                
            except Exception as e:
                logger.error(f"Error processing {midi_file}: {e}")
                results[midi_file.name] = False
        
        return results
    
    def validate_midi_file(self, filepath: str) -> Dict[str, Union[bool, str, float]]:
        """
        Validate a MIDI file and return quality metrics.
        
        Args:
            filepath: Path to MIDI file
            
        Returns:
            Dictionary with validation results
        """
        try:
            sequence = self.read_midi_file(filepath)
            
            validation = {
                'is_valid': True,
                'num_notes': len(sequence.notes),
                'duration': sequence.total_duration,
                'tempo': sequence.tempo,
                'time_signature': sequence.time_signature,
                'key_signature': sequence.key_signature,
                'issues': []
            }
            
            # Check for common issues
            if len(sequence.notes) == 0:
                validation['is_valid'] = False
                validation['issues'].append("No notes found")
            
            if sequence.total_duration < 0.1:
                validation['is_valid'] = False
                validation['issues'].append("File too short")
            
            if sequence.total_duration > 600:  # 10 minutes
                validation['issues'].append("File very long")
            
            # Check note range
            if sequence.notes:
                pitches = [note.pitch for note in sequence.notes]
                min_pitch = min(pitches)
                max_pitch = max(pitches)
                
                if min_pitch < 21 or max_pitch > 108:
                    validation['issues'].append("Notes outside piano range")
                
                validation['pitch_range'] = (min_pitch, max_pitch)
            
            # Check velocity range
            if sequence.notes:
                velocities = [note.velocity for note in sequence.notes]
                min_vel = min(velocities)
                max_vel = max(velocities)
                
                if min_vel < 1 or max_vel > 127:
                    validation['issues'].append("Invalid velocity values")
                
                validation['velocity_range'] = (min_vel, max_vel)
            
            return validation
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'issues': [f"File reading error: {e}"]
            }
    
    def extract_features(self, sequence: MusicalSequence) -> Dict:
        """
        Extract musical features from a sequence.
        
        Args:
            sequence: MusicalSequence to analyze
            
        Returns:
            Dictionary of extracted features
        """
        if not sequence.notes:
            return {}
        
        features = {
            'basic_stats': self._extract_basic_stats(sequence),
            'rhythm_features': self._extract_rhythm_features(sequence),
            'melody_features': self._extract_melody_features(sequence),
            'harmonic_features': self._extract_harmonic_features(sequence)
        }
        
        return features
    
    def _extract_basic_stats(self, sequence: MusicalSequence) -> Dict:
        """Extract basic statistical features."""
        notes = sequence.notes
        
        pitches = [note.pitch for note in notes]
        velocities = [note.velocity for note in notes]
        durations = [note.duration for note in notes]
        
        return {
            'num_notes': len(notes),
            'duration': sequence.total_duration,
            'tempo': sequence.tempo,
            'mean_pitch': np.mean(pitches),
            'std_pitch': np.std(pitches),
            'mean_velocity': np.mean(velocities),
            'std_velocity': np.std(velocities),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations)
        }
    
    def _extract_rhythm_features(self, sequence: MusicalSequence) -> Dict:
        """Extract rhythm-related features."""
        notes = sequence.notes
        
        # Onset times
        onset_times = sorted(list(set([note.start_time for note in notes])))
        
        # Inter-onset intervals
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            
            return {
                'num_onsets': len(onset_times),
                'mean_ioi': np.mean(intervals),
                'std_ioi': np.std(intervals),
                'min_ioi': np.min(intervals),
                'max_ioi': np.max(intervals),
                'rhythm_entropy': -np.sum(np.histogram(intervals, bins=20)[0] * 
                                        np.log(np.histogram(intervals, bins=20)[0] + 1e-10))
            }
        else:
            return {'num_onsets': len(onset_times)}
    
    def _extract_melody_features(self, sequence: MusicalSequence) -> Dict:
        """Extract melody-related features."""
        notes = sequence.notes
        
        if len(notes) < 2:
            return {}
        
        # Pitch intervals
        pitches = [note.pitch for note in notes]
        intervals = np.diff(pitches)
        
        # Melodic contour
        contour = np.sign(intervals)
        
        return {
            'mean_interval': np.mean(np.abs(intervals)),
            'std_interval': np.std(intervals),
            'max_interval': np.max(np.abs(intervals)),
            'melodic_range': max(pitches) - min(pitches),
            'contour_direction': np.mean(contour),
            'repeated_notes': np.sum(intervals == 0)
        }
    
    def _extract_harmonic_features(self, sequence: MusicalSequence) -> Dict:
        """Extract harmonic features."""
        notes = sequence.notes
        
        # Pitch class distribution
        pitch_classes = [note.pitch % 12 for note in notes]
        pc_hist = np.zeros(12)
        for pc in pitch_classes:
            pc_hist[pc] += 1
        
        if np.sum(pc_hist) > 0:
            pc_hist = pc_hist / np.sum(pc_hist)
        
        return {
            'pitch_class_entropy': -np.sum(pc_hist * np.log(pc_hist + 1e-10)),
            'most_common_pc': np.argmax(pc_hist),
            'pc_variety': np.sum(pc_hist > 0),
            'key_signature': sequence.key_signature
        }
    
    def save_features(self, features: Dict, filepath: str) -> bool:
        """Save extracted features to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(features, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving features to {filepath}: {e}")
            return False
    
    def load_features(self, filepath: str) -> Optional[Dict]:
        """Load features from JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading features from {filepath}: {e}")
            return None
