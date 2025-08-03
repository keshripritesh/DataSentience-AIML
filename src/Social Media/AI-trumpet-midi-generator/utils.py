"""
Utility functions for trumpet MIDI generation and processing
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from config import TRUMPET_CONFIG, EXERCISE_TYPES

def validate_trumpet_note(midi_note: int, range_type: str = "full") -> bool:
    """
    Validate if a MIDI note is within trumpet range

    Args:
        midi_note: MIDI note number
        range_type: "full", "comfortable", or "beginner"

    Returns:
        bool: True if note is valid for trumpet
    """
    ranges = {
        "full": (TRUMPET_CONFIG["note_range"]["lowest"], TRUMPET_CONFIG["note_range"]["highest"]),
        "comfortable": (TRUMPET_CONFIG["note_range"]["comfortable_low"], TRUMPET_CONFIG["note_range"]["comfortable_high"]),
        "beginner": (TRUMPET_CONFIG["note_range"]["beginner_low"], TRUMPET_CONFIG["note_range"]["beginner_high"])
    }

    low, high = ranges.get(range_type, ranges["full"])
    return low <= midi_note <= high

def transpose_for_trumpet(notes: List[int], target_range: str = "comfortable") -> List[int]:
    """
    Transpose notes to fit within trumpet range

    Args:
        notes: List of MIDI note numbers
        target_range: Target range type

    Returns:
        List of transposed MIDI notes
    """
    if not notes:
        return notes

    range_config = TRUMPET_CONFIG["note_range"]
    if target_range == "comfortable":
        target_low = range_config["comfortable_low"]
        target_high = range_config["comfortable_high"]
    elif target_range == "beginner":
        target_low = range_config["beginner_low"]
        target_high = range_config["beginner_high"]
    else:
        target_low = range_config["lowest"]
        target_high = range_config["highest"]

    # Find the best octave transposition
    transposed_notes = []
    for note in notes:
        # Try different octave transpositions
        best_note = note
        best_distance = float('inf')

        for octave_shift in range(-3, 4):  # Try shifting up to 3 octaves up/down
            candidate = note + (octave_shift * 12)
            if target_low <= candidate <= target_high:
                distance = abs(candidate - (target_low + target_high) // 2)
                if distance < best_distance:
                    best_distance = distance
                    best_note = candidate

        transposed_notes.append(best_note)

    return transposed_notes

def create_trumpet_friendly_rhythm(num_notes: int, complexity: str = "simple") -> List[float]:
    """
    Generate trumpet-friendly rhythm patterns

    Args:
        num_notes: Number of notes to generate rhythm for
        complexity: "simple", "moderate", or "complex"

    Returns:
        List of note durations
    """
    if complexity == "simple":
        # Mostly quarter and half notes
        durations = [0.25, 0.5, 1.0]
        weights = [0.6, 0.3, 0.1]
    elif complexity == "moderate":
        # Include eighth notes and dotted rhythms
        durations = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
        weights = [0.2, 0.4, 0.1, 0.2, 0.05, 0.05]
    else:  # complex
        # Include sixteenth notes and syncopation
        durations = [0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5]
        weights = [0.1, 0.25, 0.1, 0.25, 0.1, 0.1, 0.05, 0.03, 0.02]

    return np.random.choice(durations, size=num_notes, p=weights).tolist()

def analyze_trumpet_difficulty(midi_file: MidiFile) -> Dict[str, any]:
    """
    Analyze the difficulty level of a trumpet piece

    Args:
        midi_file: MIDI file to analyze

    Returns:
        Dictionary with difficulty metrics
    """
    analysis = {
        "difficulty_score": 0,
        "factors": {},
        "recommendations": []
    }

    # Extract notes and timing
    notes = []
    durations = []

    for track in midi_file.tracks:
        current_time = 0
        note_starts = {}

        for msg in track:
            current_time += msg.time

            if hasattr(msg, 'type'):
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_starts[msg.note] = current_time
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in note_starts:
                        duration = current_time - note_starts[msg.note]
                        notes.append(msg.note)
                        durations.append(duration)
                        del note_starts[msg.note]

    if not notes:
        return analysis

    # Range analysis
    note_range = max(notes) - min(notes)
    analysis["factors"]["range_semitones"] = note_range

    if note_range > 24:  # More than 2 octaves
        analysis["difficulty_score"] += 3
        analysis["recommendations"].append("Wide range - practice scales across full range")
    elif note_range > 12:  # More than 1 octave
        analysis["difficulty_score"] += 1

    # High notes analysis
    high_notes = [n for n in notes if n > TRUMPET_CONFIG["note_range"]["comfortable_high"]]
    if high_notes:
        analysis["difficulty_score"] += len(high_notes) / len(notes) * 5
        analysis["factors"]["high_note_percentage"] = len(high_notes) / len(notes) * 100
        analysis["recommendations"].append("Contains high notes - practice lip flexibility exercises")

    # Low notes analysis
    low_notes = [n for n in notes if n < TRUMPET_CONFIG["note_range"]["comfortable_low"]]
    if low_notes:
        analysis["difficulty_score"] += len(low_notes) / len(notes) * 3
        analysis["factors"]["low_note_percentage"] = len(low_notes) / len(notes) * 100
        analysis["recommendations"].append("Contains low notes - focus on proper air support")

    # Rhythm complexity
    if durations:
        unique_durations = len(set(durations))
        short_notes = [d for d in durations if d < 240]  # Shorter than quarter note at 120 BPM

        if unique_durations > 5:
            analysis["difficulty_score"] += 2
            analysis["recommendations"].append("Complex rhythm - practice with metronome")

        if short_notes and len(short_notes) / len(durations) > 0.3:
            analysis["difficulty_score"] += 2
            analysis["factors"]["fast_note_percentage"] = len(short_notes) / len(durations) * 100
            analysis["recommendations"].append("Fast passages - practice slowly and gradually increase tempo")

    # Interval analysis
    intervals = []
    for i in range(1, len(notes)):
        intervals.append(abs(notes[i] - notes[i-1]))

    if intervals:
        large_leaps = [i for i in intervals if i > 7]  # Perfect 5th or larger
        if large_leaps:
            analysis["difficulty_score"] += len(large_leaps) / len(intervals) * 3
            analysis["factors"]["large_leap_percentage"] = len(large_leaps) / len(intervals) * 100
            analysis["recommendations"].append("Large intervals - practice lip slurs and flexibility")

    # Overall difficulty rating
    if analysis["difficulty_score"] < 2:
        analysis["difficulty_level"] = "Beginner"
    elif analysis["difficulty_score"] < 5:
        analysis["difficulty_level"] = "Intermediate"
    elif analysis["difficulty_score"] < 8:
        analysis["difficulty_level"] = "Advanced"
    else:
        analysis["difficulty_level"] = "Professional"

    return analysis

def suggest_practice_exercises(analysis: Dict) -> List[str]:
    """
    Suggest practice exercises based on difficulty analysis

    Args:
        analysis: Output from analyze_trumpet_difficulty

    Returns:
        List of practice suggestions
    """
    suggestions = []

    # Add general suggestions based on difficulty level
    difficulty = analysis.get("difficulty_level", "Intermediate")

    if difficulty == "Beginner":
        suggestions.extend([
            "Practice long tones to build embouchure strength",
            "Work on basic major scales",
            "Focus on steady rhythm with quarter notes"
        ])
    elif difficulty == "Intermediate":
        suggestions.extend([
            "Practice chromatic scales for finger dexterity",
            "Work on lip slurs for flexibility",
            "Practice with different articulations (legato, staccato)"
        ])
    elif difficulty in ["Advanced", "Professional"]:
        suggestions.extend([
            "Practice advanced etudes for technical development",
            "Work on extended range exercises",
            "Practice complex rhythmic patterns with subdivision"
        ])

    # Add specific suggestions from analysis
    suggestions.extend(analysis.get("recommendations", []))

    return list(set(suggestions))  # Remove duplicates

def create_practice_variation(original_notes: List[Tuple[int, float]], variation_type: str) -> List[Tuple[int, float]]:
    """
    Create practice variations of a musical exercise

    Args:
        original_notes: List of (midi_note, duration) tuples
        variation_type: Type of variation to create

    Returns:
        Modified exercise as list of (midi_note, duration) tuples
    """
    if not original_notes:
        return original_notes

    notes, durations = zip(*original_notes)

    if variation_type == "rhythm_double":
        # Double the rhythm speed
        new_durations = [d / 2 for d in durations]
        return list(zip(notes, new_durations))

    elif variation_type == "rhythm_half":
        # Halve the rhythm speed
        new_durations = [d * 2 for d in durations]
        return list(zip(notes, new_durations))

    elif variation_type == "octave_up":
        # Transpose up an octave (if possible)
        new_notes = []
        for note in notes:
            new_note = note + 12
            if validate_trumpet_note(new_note):
                new_notes.append(new_note)
            else:
                new_notes.append(note)  # Keep original if transposition goes out of range
        return list(zip(new_notes, durations))

    elif variation_type == "octave_down":
        # Transpose down an octave (if possible)
        new_notes = []
        for note in notes:
            new_note = note - 12
            if validate_trumpet_note(new_note):
                new_notes.append(new_note)
            else:
                new_notes.append(note)  # Keep original if transposition goes out of range
        return list(zip(new_notes, durations))

    elif variation_type == "reverse":
        # Reverse the note order
        return list(zip(reversed(notes), durations))

    elif variation_type == "staccato":
        # Make all notes shorter (staccato effect)
        new_durations = [min(d * 0.5, 0.25) for d in durations]  # Cap at quarter note
        return list(zip(notes, new_durations))

    elif variation_type == "legato":
        # Make all notes longer (legato effect)
        new_durations = [d * 1.5 for d in durations]
        return list(zip(notes, new_durations))

    else:
        return original_notes

def midi_note_to_trumpet_fingering(midi_note: int) -> str:
    """
    Convert MIDI note to trumpet fingering notation

    Args:
        midi_note: MIDI note number

    Returns:
        String representation of trumpet fingering
    """
    # Trumpet fingering chart (Bb trumpet)
    # Note: This is a simplified chart for common fingerings
    fingering_chart = {
        58: "1+3",      # Bb3
        59: "1+2",      # B3
        60: "1",        # C4
        61: "1+2+3",    # C#4
        62: "1+3",      # D4
        63: "2+3",      # D#4
        64: "2",        # E4
        65: "1",        # F4
        66: "2",        # F#4
        67: "0",        # G4
        68: "2+3",      # G#4
        69: "1+2",      # A4
        70: "1",        # Bb4
        71: "2",        # B4
        72: "0",        # C5
        73: "1+2+3",    # C#5
        74: "1+3",      # D5
        75: "2+3",      # D#5
        76: "2",        # E5
        77: "1",        # F5
        78: "2",        # F#5
        79: "0",        # G5
        80: "2+3",      # G#5
        81: "1+2",      # A5
        82: "1",        # Bb5
        83: "2",        # B5
        84: "0",        # C6
    }

    return fingering_chart.get(midi_note, "?")

def generate_fingering_chart(notes: List[int]) -> Dict[str, str]:
    """
    Generate a fingering chart for a list of notes

    Args:
        notes: List of MIDI note numbers

    Returns:
        Dictionary mapping note names to fingerings
    """
    from app import midi_to_note_name  # Import here to avoid circular import

    chart = {}
    for note in set(notes):  # Remove duplicates
        if validate_trumpet_note(note):
            note_name = midi_to_note_name(note)
            fingering = midi_note_to_trumpet_fingering(note)
            chart[note_name] = fingering

    return chart
