#!/usr/bin/env python3
"""
Basic usage example for ImprovAI.
Demonstrates how to use the music generation system.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.encoders.music_encoder import Note, MusicalSequence
from core.generators.music_generator import MusicGenerator, GenerationConfig
from io.midi_handler import MIDIHandler
from utils.config import get_config


def create_sample_melody():
    """Create a sample melody for demonstration."""
    notes = [
        Note(pitch=60, velocity=80, start_time=0.0, end_time=0.5),   # C4
        Note(pitch=62, velocity=80, start_time=0.5, end_time=1.0),   # D4
        Note(pitch=64, velocity=80, start_time=1.0, end_time=1.5),   # E4
        Note(pitch=65, velocity=80, start_time=1.5, end_time=2.0),   # F4
        Note(pitch=67, velocity=80, start_time=2.0, end_time=2.5),   # G4
        Note(pitch=69, velocity=80, start_time=2.5, end_time=3.0),   # A4
        Note(pitch=71, velocity=80, start_time=3.0, end_time=3.5),   # B4
        Note(pitch=72, velocity=80, start_time=3.5, end_time=4.0),   # C5
    ]
    
    return MusicalSequence(
        notes=notes,
        tempo=120.0,
        time_signature=(4, 4),
        key_signature="C"
    )


def main():
    """Main demonstration function."""
    print("🎵 ImprovAI - Basic Usage Example")
    print("=" * 50)
    
    # Initialize components
    print("🔧 Initializing components...")
    config = get_config()
    generator = MusicGenerator(config)
    midi_handler = MIDIHandler(config)
    
    # Create sample input melody
    print("🎼 Creating sample input melody...")
    input_sequence = create_sample_melody()
    print(f"   Created melody with {len(input_sequence.notes)} notes")
    print(f"   Duration: {input_sequence.total_duration:.2f} seconds")
    print(f"   Key: {input_sequence.key_signature}")
    print(f"   Tempo: {input_sequence.tempo} BPM")
    
    # Configure generation parameters
    print("\n⚙️  Configuring generation parameters...")
    generation_config = GenerationConfig(
        model_type="hybrid",
        max_length=32,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        style="classical",
        creativity=0.7,
        tempo=120.0,
        time_signature=(4, 4)
    )
    
    # Generate continuation
    print("🎵 Generating musical continuation...")
    try:
        generated_sequence = generator.generate_continuation(
            input_sequence.notes,
            generation_config
        )
        
        print(f"   Generated {len(generated_sequence.notes)} notes")
        print(f"   Duration: {generated_sequence.total_duration:.2f} seconds")
        
        # Assess quality
        print("\n📊 Assessing generation quality...")
        quality_metrics = generator.assess_quality(generated_sequence)
        
        print("   Quality Metrics:")
        for metric, value in quality_metrics.items():
            print(f"     {metric}: {value:.3f}")
        
        # Save to MIDI file
        print("\n💾 Saving generated music...")
        output_path = "examples/generated_music.mid"
        success = midi_handler.write_midi_file(generated_sequence, output_path)
        
        if success:
            print(f"   ✅ Saved to {output_path}")
        else:
            print("   ❌ Failed to save MIDI file")
        
        # Extract and save features
        print("\n📈 Extracting musical features...")
        features = midi_handler.extract_features(generated_sequence)
        
        features_path = "examples/musical_features.json"
        success = midi_handler.save_features(features, features_path)
        
        if success:
            print(f"   ✅ Saved features to {features_path}")
        else:
            print("   ❌ Failed to save features")
        
        # Generate variations
        print("\n🔄 Generating variations...")
        variations = generator.generate_multiple_variations(
            input_sequence.notes,
            generation_config,
            num_variations=2
        )
        
        print(f"   Generated {len(variations)} variations")
        
        for i, variation in enumerate(variations):
            variation_path = f"examples/variation_{i+1}.mid"
            midi_handler.write_midi_file(variation, variation_path)
            print(f"   ✅ Saved variation {i+1} to {variation_path}")
        
        print("\n🎉 Example completed successfully!")
        print("\n📁 Generated files:")
        print("   - examples/generated_music.mid")
        print("   - examples/musical_features.json")
        print("   - examples/variation_1.mid")
        print("   - examples/variation_2.mid")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    main()
