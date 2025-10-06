"""
Advanced music encoder for ImprovAI.
Handles MIDI note encoding, timing, velocity, and harmonic analysis.
"""

import numpy as np
import pretty_midi
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from utils.config import get_config

logger = logging.getLogger(__name__)


class NoteEvent(Enum):
    """Types of note events."""
    NOTE_ON = 1
    NOTE_OFF = 2
    VELOCITY = 3
    TIMING = 4


@dataclass
class Note:
    """Represents a musical note with timing and velocity information."""
    pitch: int
    velocity: int
    start_time: float
    end_time: float
    duration: Optional[float] = None
    
    def __post_init__(self):
        """Calculate duration if not provided."""
        if self.duration is None:
            self.duration = self.end_time - self.start_time


@dataclass
class MusicalSequence:
    """Represents a sequence of musical notes with metadata."""
    notes: List[Note]
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    key_signature: str = "C"
    total_duration: float = 0.0
    
    def __post_init__(self):
        """Calculate total duration if not provided."""
        if self.total_duration == 0.0 and self.notes:
            self.total_duration = max(note.end_time for note in self.notes)


class MusicEncoder:
    """
    Advanced music encoder for converting between MIDI and numerical representations.
    
    Features:
    - Multi-dimensional encoding (pitch, velocity, timing)
    - Harmonic analysis and chord detection
    - Rhythm pattern extraction
    - Style-aware encoding
    """
    
    def __init__(self, config=None):
        """Initialize the music encoder."""
        self.config = config or get_config()
        self.pitch_range = self.config.audio.note_range
        self.velocity_range = self.config.audio.velocity_range
        self.midi_resolution = self.config.audio.midi_resolution
        
        # Vocabulary sizes
        self.pitch_vocab_size = self.pitch_range[1] - self.pitch_range[0] + 1
        self.velocity_vocab_size = self.velocity_range[1] - self.velocity_range[0] + 1
        
        # Special tokens
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.sep_token = 3
        
        # Timing quantization
        self.time_quantization = 0.125  # 8th note resolution
        
        logger.info(f"MusicEncoder initialized with pitch range {self.pitch_range}, "
                   f"velocity range {self.velocity_range}")
    
    def encode_notes(self, notes: List[Note]) -> np.ndarray:
        """
        Encode a list of notes into a numerical sequence.
        
        Args:
            notes: List of Note objects
            
        Returns:
            numpy array of encoded notes
        """
        if not notes:
            return np.array([])
        
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda x: x.start_time)
        
        encoded_sequence = []
        
        for i, note in enumerate(sorted_notes):
            # Encode pitch (shifted to start from 0)
            pitch_encoded = note.pitch - self.pitch_range[0]
            
            # Encode velocity (shifted to start from 0)
            velocity_encoded = note.velocity - self.velocity_range[0]
            
            # Encode timing (quantized to nearest time step)
            time_encoded = int(round(note.start_time / self.time_quantization))
            
            # Encode duration (quantized)
            duration_encoded = int(round(note.duration / self.time_quantization))
            
            # Combine into a single token (multi-dimensional encoding)
            encoded_note = self._combine_encodings(
                pitch_encoded, velocity_encoded, time_encoded, duration_encoded
            )
            
            encoded_sequence.append(encoded_note)
        
        return np.array(encoded_sequence)
    
    def decode_notes(self, encoded_sequence: np.ndarray) -> List[Note]:
        """
        Decode a numerical sequence back to notes.
        
        Args:
            encoded_sequence: numpy array of encoded notes
            
        Returns:
            List of Note objects
        """
        notes = []
        current_time = 0.0
        
        for encoded_note in encoded_sequence:
            # Skip special tokens
            if encoded_note in [self.pad_token, self.start_token, self.end_token, self.sep_token]:
                continue
            
            # Decode individual components
            pitch, velocity, time_offset, duration = self._separate_encodings(encoded_note)
            
            # Convert back to original values
            pitch += self.pitch_range[0]
            velocity += self.velocity_range[0]
            start_time = current_time + (time_offset * self.time_quantization)
            note_duration = duration * self.time_quantization
            
            # Create note object
            note = Note(
                pitch=pitch,
                velocity=velocity,
                start_time=start_time,
                end_time=start_time + note_duration,
                duration=note_duration
            )
            
            notes.append(note)
            current_time = start_time + note_duration
        
        return notes
    
    def _combine_encodings(self, pitch: int, velocity: int, time: int, duration: int) -> int:
        """
        Combine multiple encodings into a single integer token.
        
        This creates a multi-dimensional encoding that preserves all information
        in a format suitable for neural network processing.
        """
        # Use bit shifting to combine encodings
        # Assuming reasonable ranges for each component
        combined = (
            pitch +
            (velocity << 8) +
            (time << 16) +
            (duration << 24)
        )
        
        return combined
    
    def _separate_encodings(self, combined: int) -> Tuple[int, int, int, int]:
        """
        Separate a combined encoding back into individual components.
        """
        pitch = combined & 0xFF
        velocity = (combined >> 8) & 0xFF
        time = (combined >> 16) & 0xFF
        duration = (combined >> 24) & 0xFF
        
        return pitch, velocity, time, duration
    
    def encode_midi_file(self, midi_path: str) -> MusicalSequence:
        """
        Encode a MIDI file into a MusicalSequence.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            MusicalSequence object
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    # Filter notes within our range
                    if self.pitch_range[0] <= note.pitch <= self.pitch_range[1]:
                        note_obj = Note(
                            pitch=note.pitch,
                            velocity=note.velocity,
                            start_time=note.start,
                            end_time=note.end,
                            duration=note.end - note.start
                        )
                        notes.append(note_obj)
            
            # Get tempo and time signature
            tempo = midi_data.estimate_tempo() if midi_data.tempo_changes else 120.0
            time_signature = (4, 4)  # Default, could be extracted from MIDI
            
            return MusicalSequence(
                notes=notes,
                tempo=tempo,
                time_signature=time_signature,
                total_duration=midi_data.get_end_time()
            )
            
        except Exception as e:
            logger.error(f"Error encoding MIDI file {midi_path}: {e}")
            return MusicalSequence(notes=[])
    
    def extract_harmonic_features(self, notes: List[Note]) -> Dict[str, np.ndarray]:
        """
        Extract harmonic features from a sequence of notes.
        
        Args:
            notes: List of Note objects
            
        Returns:
            Dictionary of harmonic features
        """
        if not notes:
            return {}
        
        # Extract pitch classes (modulo 12)
        pitch_classes = [note.pitch % 12 for note in notes]
        
        # Create pitch class histogram
        pitch_class_hist = np.zeros(12)
        for pc in pitch_classes:
            pitch_class_hist[pc] += 1
        
        # Normalize histogram
        if np.sum(pitch_class_hist) > 0:
            pitch_class_hist = pitch_class_hist / np.sum(pitch_class_hist)
        
        # Extract chord information (simplified)
        chords = self._detect_chords(notes)
        
        # Extract rhythm patterns
        rhythm_pattern = self._extract_rhythm_pattern(notes)
        
        return {
            'pitch_class_histogram': pitch_class_hist,
            'chords': chords,
            'rhythm_pattern': rhythm_pattern,
            'key_signature': self._estimate_key_signature(pitch_class_hist)
        }
    
    def _detect_chords(self, notes: List[Note]) -> List[str]:
        """
        Detect chords from overlapping notes.
        """
        chords = []
        time_windows = np.arange(0, max(note.end_time for note in notes), 0.25)
        
        for t in time_windows:
            # Find notes active at this time
            active_notes = [
                note.pitch for note in notes
                if note.start_time <= t <= note.end_time
            ]
            
            if len(active_notes) >= 3:
                # Simple chord detection (major/minor triads)
                chord_name = self._identify_chord(active_notes)
                chords.append(chord_name)
            else:
                chords.append("")
        
        return chords
    
    def _identify_chord(self, pitches: List[int]) -> str:
        """
        Identify chord type from a list of pitches.
        """
        if len(pitches) < 3:
            return ""
        
        # Convert to pitch classes
        pitch_classes = sorted(list(set([p % 12 for p in pitches])))
        
        # Simple chord identification
        if len(pitch_classes) >= 3:
            root = pitch_classes[0]
            third = pitch_classes[1] if len(pitch_classes) > 1 else None
            fifth = pitch_classes[2] if len(pitch_classes) > 2 else None
            
            if third and fifth:
                # Check for major triad
                if (third - root) % 12 == 4 and (fifth - root) % 12 == 7:
                    return f"{self._pitch_class_to_name(root)}maj"
                # Check for minor triad
                elif (third - root) % 12 == 3 and (fifth - root) % 12 == 7:
                    return f"{self._pitch_class_to_name(root)}min"
        
        return "unknown"
    
    def _pitch_class_to_name(self, pc: int) -> str:
        """Convert pitch class to note name."""
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return names[pc]
    
    def _extract_rhythm_pattern(self, notes: List[Note]) -> np.ndarray:
        """
        Extract rhythm pattern from notes.
        """
        if not notes:
            return np.array([])
        
        # Get onset times
        onset_times = sorted(list(set([note.start_time for note in notes])))
        
        # Calculate inter-onset intervals
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            # Quantize intervals
            quantized_intervals = np.round(intervals / self.time_quantization)
            return quantized_intervals
        
        return np.array([])
    
    def _estimate_key_signature(self, pitch_class_hist: np.ndarray) -> str:
        """
        Estimate key signature from pitch class histogram.
        """
        # Simple key estimation based on pitch class distribution
        # This is a simplified version - more sophisticated methods exist
        
        # Major key profiles (Krumhansl-Kessler profiles)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        # Calculate correlations for all possible keys
        correlations = []
        for i in range(12):
            # Rotate histogram for each key
            rotated_hist = np.roll(pitch_class_hist, i)
            
            # Calculate correlation with major and minor profiles
            major_corr = np.corrcoef(rotated_hist, major_profile)[0, 1]
            minor_corr = np.corrcoef(rotated_hist, minor_profile)[0, 1]
            
            correlations.append((major_corr, minor_corr))
        
        # Find best match
        best_key = 0
        best_corr = -1
        is_major = True
        
        for i, (major_corr, minor_corr) in enumerate(correlations):
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = i
                is_major = True
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = i
                is_major = False
        
        key_name = self._pitch_class_to_name(best_key)
        return f"{key_name}{'maj' if is_major else 'min'}"
    
    def create_training_sequences(self, notes: List[Note], sequence_length: int = 64) -> List[np.ndarray]:
        """
        Create training sequences from notes for model training.
        
        Args:
            notes: List of Note objects
            sequence_length: Length of each training sequence
            
        Returns:
            List of encoded sequences
        """
        if len(notes) < sequence_length:
            return []
        
        encoded_notes = self.encode_notes(notes)
        sequences = []
        
        # Create overlapping sequences
        for i in range(len(encoded_notes) - sequence_length + 1):
            sequence = encoded_notes[i:i + sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def get_vocabulary_size(self) -> int:
        """Get the total vocabulary size for the encoder."""
        # Calculate based on the combined encoding scheme
        # Use a reasonable vocabulary size for music tokens
        return 1000  # Reasonable size for music vocabulary
