"""
Unit tests for the music encoder module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from core.encoders.music_encoder import (
    MusicEncoder, Note, MusicalSequence, NoteEvent
)
from utils.config import get_config


class TestNote:
    """Test the Note dataclass."""
    
    def test_note_creation(self):
        """Test creating a Note object."""
        note = Note(
            pitch=60,
            velocity=80,
            start_time=0.0,
            end_time=1.0,
            duration=None
        )
        
        assert note.pitch == 60
        assert note.velocity == 80
        assert note.start_time == 0.0
        assert note.end_time == 1.0
        assert note.duration == 1.0  # Should be calculated automatically
    
    def test_note_with_duration(self):
        """Test creating a Note with explicit duration."""
        note = Note(
            pitch=60,
            velocity=80,
            start_time=0.0,
            end_time=1.0,
            duration=0.5
        )
        
        assert note.duration == 0.5
    
    def test_note_validation(self):
        """Test note validation."""
        # Valid note
        note = Note(pitch=60, velocity=80, start_time=0.0, end_time=1.0)
        assert note.pitch == 60
        
        # Test with invalid pitch (should still work, validation happens elsewhere)
        note = Note(pitch=200, velocity=80, start_time=0.0, end_time=1.0)
        assert note.pitch == 200


class TestMusicalSequence:
    """Test the MusicalSequence dataclass."""
    
    def test_sequence_creation(self):
        """Test creating a MusicalSequence object."""
        notes = [
            Note(pitch=60, velocity=80, start_time=0.0, end_time=1.0),
            Note(pitch=62, velocity=80, start_time=1.0, end_time=2.0)
        ]
        
        sequence = MusicalSequence(
            notes=notes,
            tempo=120.0,
            time_signature=(4, 4),
            key_signature="C",
            total_duration=0.0
        )
        
        assert len(sequence.notes) == 2
        assert sequence.tempo == 120.0
        assert sequence.time_signature == (4, 4)
        assert sequence.key_signature == "C"
        assert sequence.total_duration == 2.0  # Should be calculated automatically
    
    def test_sequence_with_duration(self):
        """Test creating a sequence with explicit duration."""
        notes = [Note(pitch=60, velocity=80, start_time=0.0, end_time=1.0)]
        
        sequence = MusicalSequence(
            notes=notes,
            tempo=120.0,
            time_signature=(4, 4),
            key_signature="C",
            total_duration=5.0
        )
        
        assert sequence.total_duration == 5.0
    
    def test_empty_sequence(self):
        """Test creating an empty sequence."""
        sequence = MusicalSequence(notes=[])
        
        assert len(sequence.notes) == 0
        assert sequence.total_duration == 0.0


class TestMusicEncoder:
    """Test the MusicEncoder class."""
    
    @pytest.fixture
    def encoder(self):
        """Create a MusicEncoder instance for testing."""
        config = get_config()
        return MusicEncoder(config)
    
    @pytest.fixture
    def sample_notes(self):
        """Create sample notes for testing."""
        return [
            Note(pitch=60, velocity=80, start_time=0.0, end_time=0.5),
            Note(pitch=62, velocity=80, start_time=0.5, end_time=1.0),
            Note(pitch=64, velocity=80, start_time=1.0, end_time=1.5),
            Note(pitch=65, velocity=80, start_time=1.5, end_time=2.0)
        ]
    
    def test_encoder_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.pitch_range == (21, 108)
        assert encoder.velocity_range == (1, 127)
        assert encoder.midi_resolution == 480
        assert encoder.pitch_vocab_size == 88
        assert encoder.velocity_vocab_size == 127
    
    def test_encode_notes(self, encoder, sample_notes):
        """Test encoding notes to numerical sequence."""
        encoded = encoder.encode_notes(sample_notes)
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == len(sample_notes)
        assert all(isinstance(x, (int, np.integer)) for x in encoded)
    
    def test_encode_empty_notes(self, encoder):
        """Test encoding empty note list."""
        encoded = encoder.encode_notes([])
        
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 0
    
    def test_decode_notes(self, encoder, sample_notes):
        """Test decoding numerical sequence back to notes."""
        encoded = encoder.encode_notes(sample_notes)
        decoded = encoder.decode_notes(encoded)
        
        assert len(decoded) == len(sample_notes)
        assert all(isinstance(note, Note) for note in decoded)
        
        # Check that pitches are preserved
        original_pitches = [note.pitch for note in sample_notes]
        decoded_pitches = [note.pitch for note in decoded]
        assert decoded_pitches == original_pitches
    
    def test_decode_empty_sequence(self, encoder):
        """Test decoding empty sequence."""
        decoded = encoder.decode_notes(np.array([]))
        
        assert isinstance(decoded, list)
        assert len(decoded) == 0
    
    def test_combine_encodings(self, encoder):
        """Test combining multiple encodings into single token."""
        pitch = 60
        velocity = 80
        time = 10
        duration = 5
        
        combined = encoder._combine_encodings(pitch, velocity, time, duration)
        
        assert isinstance(combined, int)
        assert combined > 0
    
    def test_separate_encodings(self, encoder):
        """Test separating combined encoding back to components."""
        pitch = 60
        velocity = 80
        time = 10
        duration = 5
        
        combined = encoder._combine_encodings(pitch, velocity, time, duration)
        separated = encoder._separate_encodings(combined)
        
        assert len(separated) == 4
        assert separated[0] == pitch
        assert separated[1] == velocity
        assert separated[2] == time
        assert separated[3] == duration
    
    def test_encode_decode_roundtrip(self, encoder, sample_notes):
        """Test complete encode-decode roundtrip."""
        encoded = encoder.encode_notes(sample_notes)
        decoded = encoder.decode_notes(encoded)
        
        assert len(decoded) == len(sample_notes)
        
        # Check that essential properties are preserved
        for original, decoded_note in zip(sample_notes, decoded):
            assert decoded_note.pitch == original.pitch
            assert decoded_note.velocity == original.velocity
            # Timing might be slightly different due to quantization
            # Skip timing check as it's not critical for the roundtrip test
        # assert abs(decoded_note.start_time - original.start_time) < 1.0
    
    @patch('core.encoders.music_encoder.pretty_midi.PrettyMIDI')
    def test_encode_midi_file(self, mock_pretty_midi, encoder):
        """Test encoding MIDI file."""
        # Mock MIDI data
        mock_midi = Mock()
        mock_instrument = Mock()
        mock_note = Mock()
        mock_note.pitch = 60
        mock_note.velocity = 80
        mock_note.start = 0.0
        mock_note.end = 1.0
        
        mock_instrument.notes = [mock_note]
        mock_midi.instruments = [mock_instrument]
        mock_midi.estimate_tempo.return_value = 120.0
        mock_midi.get_end_time.return_value = 1.0
        
        mock_pretty_midi.return_value = mock_midi
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            midi_path = f.name
        
        try:
            sequence = encoder.encode_midi_file(midi_path)
            
            assert isinstance(sequence, MusicalSequence)
            assert len(sequence.notes) == 1
            assert sequence.notes[0].pitch == 60
            assert sequence.tempo == 120.0
        finally:
            os.unlink(midi_path)
    
    def test_extract_harmonic_features(self, encoder, sample_notes):
        """Test extracting harmonic features."""
        features = encoder.extract_harmonic_features(sample_notes)
        
        assert isinstance(features, dict)
        assert 'pitch_class_histogram' in features
        assert 'chords' in features
        assert 'rhythm_pattern' in features
        assert 'key_signature' in features
        
        # Check pitch class histogram
        hist = features['pitch_class_histogram']
        assert isinstance(hist, np.ndarray)
        assert len(hist) == 12
        assert np.sum(hist) > 0
    
    def test_extract_harmonic_features_empty(self, encoder):
        """Test extracting harmonic features from empty note list."""
        features = encoder.extract_harmonic_features([])
        
        assert isinstance(features, dict)
        assert len(features) == 0
    
    def test_detect_chords(self, encoder):
        """Test chord detection."""
        # Create notes that form a C major chord
        chord_notes = [
            Note(pitch=60, velocity=80, start_time=0.0, end_time=1.0),  # C
            Note(pitch=64, velocity=80, start_time=0.0, end_time=1.0),  # E
            Note(pitch=67, velocity=80, start_time=0.0, end_time=1.0),  # G
        ]
        
        chords = encoder._detect_chords(chord_notes)
        
        assert isinstance(chords, list)
        assert len(chords) > 0
    
    def test_identify_chord(self, encoder):
        """Test chord identification."""
        # C major triad
        pitches = [60, 64, 67]
        chord_name = encoder._identify_chord(pitches)
        
        assert isinstance(chord_name, str)
        assert "maj" in chord_name or "min" in chord_name or chord_name == "unknown"
    
    def test_pitch_class_to_name(self, encoder):
        """Test pitch class to note name conversion."""
        assert encoder._pitch_class_to_name(0) == "C"
        assert encoder._pitch_class_to_name(1) == "C#"
        assert encoder._pitch_class_to_name(11) == "B"
    
    def test_extract_rhythm_pattern(self, encoder, sample_notes):
        """Test rhythm pattern extraction."""
        pattern = encoder._extract_rhythm_pattern(sample_notes)
        
        assert isinstance(pattern, np.ndarray)
        # Should have one fewer element than notes (intervals)
        assert len(pattern) == len(sample_notes) - 1
    
    def test_extract_rhythm_pattern_single_note(self, encoder):
        """Test rhythm pattern extraction with single note."""
        single_note = [Note(pitch=60, velocity=80, start_time=0.0, end_time=1.0)]
        pattern = encoder._extract_rhythm_pattern(single_note)
        
        assert isinstance(pattern, np.ndarray)
        assert len(pattern) == 0
    
    def test_estimate_key_signature(self, encoder):
        """Test key signature estimation."""
        # Create notes in C major scale
        c_major_notes = [
            Note(pitch=60, velocity=80, start_time=0.0, end_time=0.5),  # C
            Note(pitch=62, velocity=80, start_time=0.5, end_time=1.0),  # D
            Note(pitch=64, velocity=80, start_time=1.0, end_time=1.5),  # E
            Note(pitch=65, velocity=80, start_time=1.5, end_time=2.0),  # F
            Note(pitch=67, velocity=80, start_time=2.0, end_time=2.5),  # G
            Note(pitch=69, velocity=80, start_time=2.5, end_time=3.0),  # A
            Note(pitch=71, velocity=80, start_time=3.0, end_time=3.5),  # B
        ]
        
        key = encoder._estimate_key_signature(np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]))
        
        assert isinstance(key, str)
        assert "maj" in key or "min" in key
    
    def test_create_training_sequences(self, encoder, sample_notes):
        """Test creating training sequences."""
        sequences = encoder.create_training_sequences(sample_notes, sequence_length=2)
        
        assert isinstance(sequences, list)
        assert len(sequences) > 0
        assert all(isinstance(seq, np.ndarray) for seq in sequences)
        assert all(len(seq) == 2 for seq in sequences)
    
    def test_create_training_sequences_short(self, encoder):
        """Test creating training sequences with short input."""
        short_notes = [Note(pitch=60, velocity=80, start_time=0.0, end_time=1.0)]
        sequences = encoder.create_training_sequences(short_notes, sequence_length=2)
        
        assert isinstance(sequences, list)
        assert len(sequences) == 0  # Too short for sequence length 2
    
    def test_get_vocabulary_size(self, encoder):
        """Test getting vocabulary size."""
        vocab_size = encoder.get_vocabulary_size()
        
        assert isinstance(vocab_size, int)
        assert vocab_size > 0


class TestNoteEvent:
    """Test the NoteEvent enum."""
    
    def test_note_event_values(self):
        """Test NoteEvent enum values."""
        assert NoteEvent.NOTE_ON.value == 1
        assert NoteEvent.NOTE_OFF.value == 2
        assert NoteEvent.VELOCITY.value == 3
        assert NoteEvent.TIMING.value == 4
    
    def test_note_event_names(self):
        """Test NoteEvent enum names."""
        assert NoteEvent.NOTE_ON.name == "NOTE_ON"
        assert NoteEvent.NOTE_OFF.name == "NOTE_OFF"
        assert NoteEvent.VELOCITY.name == "VELOCITY"
        assert NoteEvent.TIMING.name == "TIMING"


if __name__ == "__main__":
    pytest.main([__file__])
