#!/usr/bin/env python3
"""
Demo script for Trumpet MIDI Generator
Shows key features without requiring Streamlit interface
"""

import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

from app import json_to_midi, note_name_to_midi, midi_to_note_name
from utils import validate_trumpet_note, analyze_trumpet_difficulty, midi_note_to_trumpet_fingering
from config import TRUMPET_CONFIG

def demo_note_validation():
    """Demonstrate note validation for trumpet"""
    print("🎺 Note Validation Demo")
    print("-" * 30)

    test_notes = [40, 52, 60, 72, 84, 96]  # Various MIDI notes

    for note in test_notes:
        note_name = midi_to_note_name(note)
        full_range = validate_trumpet_note(note, "full")
        comfortable = validate_trumpet_note(note, "comfortable")
        beginner = validate_trumpet_note(note, "beginner")

        print(f"{note_name:4} (MIDI {note:2}): ", end="")
        print(f"Full: {'✅' if full_range else '❌'} ", end="")
        print(f"Comfortable: {'✅' if comfortable else '❌'} ", end="")
        print(f"Beginner: {'✅' if beginner else '❌'}")

def demo_fingering_chart():
    """Demonstrate trumpet fingering chart"""
    print("\n🎺 Trumpet Fingering Demo")
    print("-" * 30)

    # Show fingerings for common trumpet notes
    notes = [58, 60, 62, 64, 65, 67, 69, 70, 72, 74, 76, 77, 79, 81, 82, 84]

    print("Note | Fingering")
    print("-----|----------")
    for note in notes:
        if validate_trumpet_note(note):
            note_name = midi_to_note_name(note)
            fingering = midi_note_to_trumpet_fingering(note)
            print(f"{note_name:4} | {fingering}")

def demo_json_to_midi():
    """Demonstrate JSON to MIDI conversion"""
    print("\n🎵 JSON to MIDI Demo")
    print("-" * 30)

    # Create a simple trumpet exercise
    exercise_data = [
        ["C4", 1.0],    # Whole note C
        ["D4", 0.5],    # Half note D
        ["E4", 0.5],    # Half note E
        ["F4", 0.25],   # Quarter note F
        ["G4", 0.25],   # Quarter note G
        ["A4", 0.25],   # Quarter note A
        ["Bb4", 0.25],  # Quarter note Bb
        ["C5", 2.0]     # Breve C
    ]

    print("Exercise JSON:")
    print(json.dumps(exercise_data, indent=2))

    # Convert to MIDI
    midi_file = json_to_midi(exercise_data, instrument=56, tempo=120)

    # Save demo file
    output_path = Path("demo_exercise.mid")
    midi_file.save(str(output_path))
    print(f"\n✅ MIDI file saved: {output_path}")

    return midi_file

def demo_difficulty_analysis(midi_file):
    """Demonstrate difficulty analysis"""
    print("\n📊 Difficulty Analysis Demo")
    print("-" * 30)

    analysis = analyze_trumpet_difficulty(midi_file)

    print(f"Difficulty Level: {analysis['difficulty_level']}")
    print(f"Difficulty Score: {analysis['difficulty_score']:.2f}")

    if analysis['factors']:
        print("\nFactors:")
        for factor, value in analysis['factors'].items():
            if isinstance(value, float):
                print(f"  • {factor}: {value:.1f}")
            else:
                print(f"  • {factor}: {value}")

    if analysis['recommendations']:
        print("\nPractice Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")

def demo_exercise_types():
    """Demonstrate different exercise types"""
    print("\n🎼 Exercise Types Demo")
    print("-" * 30)

    # Warm-up exercise (simple, comfortable range)
    warmup = [
        ["Bb3", 2.0], ["C4", 2.0], ["D4", 2.0], ["Eb4", 2.0],
        ["F4", 2.0], ["Eb4", 2.0], ["D4", 2.0], ["C4", 2.0], ["Bb3", 4.0]
    ]

    # Technical exercise (faster, wider range)
    technical = [
        ["C4", 0.25], ["E4", 0.25], ["G4", 0.25], ["C5", 0.25],
        ["G4", 0.25], ["E4", 0.25], ["C4", 0.25], ["G3", 0.25],
        ["C4", 0.25], ["F4", 0.25], ["A4", 0.25], ["C5", 0.25],
        ["A4", 0.25], ["F4", 0.25], ["C4", 0.25], ["F3", 0.25]
    ]

    # Jazz exercise (syncopated, blue notes)
    jazz = [
        ["C4", 0.25], ["Eb4", 0.375], ["F4", 0.125], ["G4", 0.5],
        ["Bb4", 0.25], ["A4", 0.125], ["G4", 0.125], ["F4", 0.25],
        ["Eb4", 0.375], ["D4", 0.125], ["C4", 1.0]
    ]

    exercises = [
        ("Warm-up", warmup),
        ("Technical", technical),
        ("Jazz", jazz)
    ]

    for name, exercise in exercises:
        print(f"\n{name} Exercise:")
        midi_file = json_to_midi(exercise, instrument=56, tempo=120)
        analysis = analyze_trumpet_difficulty(midi_file)
        print(f"  Difficulty: {analysis['difficulty_level']}")
        print(f"  Score: {analysis['difficulty_score']:.2f}")

        # Save exercise
        filename = f"demo_{name.lower()}.mid"
        midi_file.save(filename)
        print(f"  Saved: {filename}")

def demo_range_comparison():
    """Demonstrate different range types"""
    print("\n🎯 Range Comparison Demo")
    print("-" * 30)

    ranges = {
        "beginner": (TRUMPET_CONFIG["note_range"]["beginner_low"],
                    TRUMPET_CONFIG["note_range"]["beginner_high"]),
        "comfortable": (TRUMPET_CONFIG["note_range"]["comfortable_low"],
                       TRUMPET_CONFIG["note_range"]["comfortable_high"]),
        "full": (TRUMPET_CONFIG["note_range"]["lowest"],
                TRUMPET_CONFIG["note_range"]["highest"])
    }

    for range_name, (low, high) in ranges.items():
        low_note = midi_to_note_name(low)
        high_note = midi_to_note_name(high)
        semitones = high - low
        print(f"{range_name.capitalize():12}: {low_note} - {high_note} ({semitones} semitones)")

def main():
    """Run all demos"""
    print("🎺 Trumpet MIDI Generator - Feature Demo")
    print("=" * 50)

    # Run demonstrations
    demo_range_comparison()
    demo_note_validation()
    demo_fingering_chart()
    midi_file = demo_json_to_midi()
    demo_difficulty_analysis(midi_file)
    demo_exercise_types()

    print("\n🎉 Demo completed!")
    print("\nGenerated files:")
    for file in Path('.').glob('demo_*.mid'):
        print(f"  • {file}")

    print("\n📋 Next steps:")
    print("  1. Run: streamlit run app.py")
    print("  2. Try different prompts and generation methods")
    print("  3. Upload custom soundfonts for better audio")
    print("  4. Experiment with different difficulty levels")

if __name__ == "__main__":
    main()
