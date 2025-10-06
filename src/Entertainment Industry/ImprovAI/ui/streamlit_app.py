"""
Advanced Streamlit application for ImprovAI.
Provides a sophisticated web interface for AI-powered music improvisation.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
import time
from pathlib import Path

# Import our modules
from core.encoders.music_encoder import Note, MusicalSequence
from core.generators.music_generator import MusicGenerator, GenerationConfig
from io.midi_handler import MIDIHandler
from utils.config import get_config
from utils.visualization import create_piano_roll, create_waveform_plot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ImprovAI - Advanced AI Music Improviser",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .piano-key {
        display: inline-block;
        width: 40px;
        height: 120px;
        margin: 2px;
        border: 1px solid #ccc;
        border-radius: 0 0 5px 5px;
        text-align: center;
        line-height: 120px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .piano-key:hover {
        background-color: #e0e0e0;
    }
    .piano-key.active {
        background-color: #1f77b4;
        color: white;
    }
    .black-key {
        background-color: #333;
        color: white;
        width: 30px;
        height: 80px;
        line-height: 80px;
        position: relative;
        z-index: 1;
    }
    .black-key:hover {
        background-color: #555;
    }
    .black-key.active {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class ImprovAIApp:
    """Main application class for ImprovAI Streamlit interface."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = get_config()
        self.generator = MusicGenerator(self.config)
        self.midi_handler = MIDIHandler(self.config)
        
        # Initialize session state
        if 'input_notes' not in st.session_state:
            st.session_state.input_notes = []
        if 'generated_sequence' not in st.session_state:
            st.session_state.generated_sequence = None
        if 'generation_config' not in st.session_state:
            st.session_state.generation_config = self._get_default_config()
    
    def _get_default_config(self) -> GenerationConfig:
        """Get default generation configuration."""
        return GenerationConfig(
            model_type="hybrid",
            max_length=64,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            style="classical",
            creativity=0.7,
            tempo=120.0,
            time_signature=(4, 4)
        )
    
    def run(self):
        """Run the main application."""
        # Header
        st.markdown('<h1 class="main-header">🎵 ImprovAI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced AI-Powered Musical Improvisation System</p>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._create_input_section()
            self._create_generation_section()
        
        with col2:
            self._create_controls_section()
            self._create_visualization_section()
        
        # Bottom section
        self._create_output_section()
    
    def _create_sidebar(self):
        """Create the sidebar with advanced controls."""
        st.sidebar.title("🎛️ Advanced Controls")
        
        # Model Configuration
        st.sidebar.subheader("🤖 AI Model")
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["hybrid", "lstm", "transformer"],
            help="Choose the AI model for generation"
        )
        
        # Generation Parameters
        st.sidebar.subheader("🎼 Generation Parameters")
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Controls randomness in generation"
        )
        
        creativity = st.sidebar.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls creative vs. conservative generation"
        )
        
        max_length = st.sidebar.slider(
            "Max Length",
            min_value=16,
            max_value=128,
            value=64,
            step=8,
            help="Maximum number of notes to generate"
        )
        
        # Style Configuration
        st.sidebar.subheader("🎨 Style Settings")
        style = st.sidebar.selectbox(
            "Musical Style",
            ["classical", "jazz", "pop", "custom"],
            help="Choose the musical style for generation"
        )
        
        tempo = st.sidebar.slider(
            "Tempo (BPM)",
            min_value=60,
            max_value=200,
            value=120,
            step=5,
            help="Tempo in beats per minute"
        )
        
        # Harmonic Constraints
        st.sidebar.subheader("🎶 Harmonic Constraints")
        key_signature = st.sidebar.selectbox(
            "Key Signature",
            ["", "C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"],
            help="Optional key signature constraint"
        )
        
        # Update generation config
        st.session_state.generation_config = GenerationConfig(
            model_type=model_type,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            style=style,
            creativity=creativity,
            key_signature=key_signature if key_signature else None,
            tempo=tempo,
            time_signature=(4, 4)
        )
        
        # Export Options
        st.sidebar.subheader("💾 Export Options")
        if st.sidebar.button("Export as MIDI"):
            self._export_midi()
        
        if st.sidebar.button("Export Features"):
            self._export_features()
    
    def _create_input_section(self):
        """Create the input section with virtual piano."""
        st.subheader("🎹 Input Melody")
        
        # Virtual piano
        self._create_virtual_piano()
        
        # MIDI file upload
        st.markdown("---")
        st.markdown("**Or upload a MIDI file:**")
        uploaded_file = st.file_uploader(
            "Choose a MIDI file",
            type=['mid', 'midi'],
            help="Upload a MIDI file to use as input"
        )
        
        if uploaded_file is not None:
            self._handle_midi_upload(uploaded_file)
        
        # Display input notes
        if st.session_state.input_notes:
            st.markdown("**Current Input Notes:**")
            self._display_notes_table(st.session_state.input_notes, "Input")
    
    def _create_virtual_piano(self):
        """Create a virtual piano interface."""
        st.markdown("**Click keys to add notes:**")
        
        # Piano octaves
        octaves = range(3, 6)  # C3 to C6
        
        for octave in octaves:
            st.markdown(f"**Octave {octave}:**")
            
            # White keys
            white_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
            cols = st.columns(7)
            
            for i, (col, note) in enumerate(zip(cols, white_keys)):
                with col:
                    if st.button(f"{note}{octave}", key=f"white_{note}{octave}"):
                        self._add_note(note, octave, 80)
            
            # Black keys (simplified layout)
            if octave < 6:  # Don't show black keys for highest octave
                st.markdown("&nbsp;")  # Spacing
                black_cols = st.columns(7)
                
                # Position black keys
                black_positions = [0, 1, 3, 4, 5]  # C#, D#, F#, G#, A#
                black_notes = ['C#', 'D#', 'F#', 'G#', 'A#']
                
                for i, (pos, note) in enumerate(zip(black_positions, black_notes)):
                    with black_cols[pos]:
                        if st.button(f"{note}{octave}", key=f"black_{note}{octave}"):
                            self._add_note(note, octave, 80)
        
        # Clear button
        if st.button("Clear Input"):
            st.session_state.input_notes = []
            st.rerun()
    
    def _add_note(self, note_name: str, octave: int, velocity: int):
        """Add a note to the input sequence."""
        # Convert note name to MIDI pitch
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_class = note_names.index(note_name)
        pitch = pitch_class + (octave * 12)
        
        # Calculate timing
        current_time = len(st.session_state.input_notes) * 0.5  # 0.5 seconds per note
        
        note = Note(
            pitch=pitch,
            velocity=velocity,
            start_time=current_time,
            end_time=current_time + 0.5,
            duration=0.5
        )
        
        st.session_state.input_notes.append(note)
        st.rerun()
    
    def _handle_midi_upload(self, uploaded_file):
        """Handle MIDI file upload."""
        try:
            # Save uploaded file temporarily
            temp_path = Path("temp_upload.mid")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read MIDI file
            sequence = self.midi_handler.read_midi_file(str(temp_path))
            
            # Update input notes
            st.session_state.input_notes = sequence.notes
            
            # Clean up
            temp_path.unlink()
            
            st.success(f"Successfully loaded {len(sequence.notes)} notes from MIDI file")
            
        except Exception as e:
            st.error(f"Error loading MIDI file: {e}")
    
    def _create_generation_section(self):
        """Create the generation section."""
        st.subheader("🎼 Generate Continuation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎵 Generate", type="primary", use_container_width=True):
                self._generate_continuation()
        
        with col2:
            if st.button("🔄 Generate Variations", use_container_width=True):
                self._generate_variations()
        
        with col3:
            if st.button("🎯 Assess Quality", use_container_width=True):
                self._assess_quality()
        
        # Generation progress
        if 'generation_progress' in st.session_state:
            st.progress(st.session_state.generation_progress)
    
    def _generate_continuation(self):
        """Generate a musical continuation."""
        if not st.session_state.input_notes:
            st.warning("Please add some input notes first!")
            return
        
        with st.spinner("Generating musical continuation..."):
            try:
                # Update progress
                st.session_state.generation_progress = 0.3
                
                # Generate continuation
                generated_sequence = self.generator.generate_continuation(
                    st.session_state.input_notes,
                    st.session_state.generation_config
                )
                
                st.session_state.generation_progress = 0.7
                
                # Store result
                st.session_state.generated_sequence = generated_sequence
                
                st.session_state.generation_progress = 1.0
                
                st.success(f"Generated {len(generated_sequence.notes)} notes!")
                
            except Exception as e:
                st.error(f"Error during generation: {e}")
                logger.error(f"Generation error: {e}")
            finally:
                if 'generation_progress' in st.session_state:
                    del st.session_state.generation_progress
    
    def _generate_variations(self):
        """Generate multiple variations."""
        if not st.session_state.input_notes:
            st.warning("Please add some input notes first!")
            return
        
        with st.spinner("Generating variations..."):
            try:
                variations = self.generator.generate_multiple_variations(
                    st.session_state.input_notes,
                    st.session_state.generation_config,
                    num_variations=3
                )
                
                # Store variations
                st.session_state.variations = variations
                
                st.success(f"Generated {len(variations)} variations!")
                
            except Exception as e:
                st.error(f"Error generating variations: {e}")
    
    def _assess_quality(self):
        """Assess the quality of generated music."""
        if not st.session_state.generated_sequence:
            st.warning("No generated sequence to assess!")
            return
        
        try:
            quality_metrics = self.generator.assess_quality(st.session_state.generated_sequence)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Quality", f"{quality_metrics.get('overall_quality', 0):.2f}")
            
            with col2:
                st.metric("Harmonic Variety", f"{quality_metrics.get('harmonic_variety', 0):.2f}")
            
            with col3:
                st.metric("Rhythm Consistency", f"{quality_metrics.get('rhythm_consistency', 0):.2f}")
            
            with col4:
                st.metric("Melodic Smoothness", f"{quality_metrics.get('melodic_smoothness', 0):.2f}")
            
        except Exception as e:
            st.error(f"Error assessing quality: {e}")
    
    def _create_controls_section(self):
        """Create the controls section."""
        st.subheader("🎛️ Generation Controls")
        
        # Model info
        st.markdown(f"**Model:** {st.session_state.generation_config.model_type.title()}")
        st.markdown(f"**Style:** {st.session_state.generation_config.style.title()}")
        st.markdown(f"**Tempo:** {st.session_state.generation_config.tempo} BPM")
        
        if st.session_state.generation_config.key_signature:
            st.markdown(f"**Key:** {st.session_state.generation_config.key_signature}")
        
        # Real-time parameters
        st.markdown("---")
        st.markdown("**Real-time Controls:**")
        
        # Creativity slider
        creativity = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.generation_config.creativity,
            step=0.1
        )
        
        # Update config
        st.session_state.generation_config.creativity = creativity
    
    def _create_visualization_section(self):
        """Create the visualization section."""
        st.subheader("📊 Visualizations")
        
        if st.session_state.input_notes:
            # Input visualization
            st.markdown("**Input Melody:**")
            fig = create_piano_roll(st.session_state.input_notes, "Input Melody")
            st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.generated_sequence:
            # Generated visualization
            st.markdown("**Generated Continuation:**")
            fig = create_piano_roll(st.session_state.generated_sequence.notes, "Generated Continuation")
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_output_section(self):
        """Create the output section."""
        st.subheader("🎵 Generated Music")
        
        if st.session_state.generated_sequence:
            # Display generated notes
            st.markdown("**Generated Notes:**")
            self._display_notes_table(st.session_state.generated_sequence.notes, "Generated")
            
            # Playback controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▶️ Play"):
                    st.info("Audio playback would be implemented here")
            
            with col2:
                if st.button("⏸️ Pause"):
                    st.info("Pause functionality would be implemented here")
            
            with col3:
                if st.button("⏹️ Stop"):
                    st.info("Stop functionality would be implemented here")
        
        # Variations
        if hasattr(st.session_state, 'variations') and st.session_state.variations:
            st.markdown("---")
            st.subheader("🔄 Variations")
            
            for i, variation in enumerate(st.session_state.variations):
                with st.expander(f"Variation {i+1} ({len(variation.notes)} notes)"):
                    self._display_notes_table(variation.notes, f"Variation {i+1}")
    
    def _display_notes_table(self, notes: List[Note], title: str):
        """Display notes in a table format."""
        if not notes:
            st.info("No notes to display")
            return
        
        # Create DataFrame
        data = []
        for i, note in enumerate(notes):
            note_name = self._pitch_to_note_name(note.pitch)
            data.append({
                "Index": i + 1,
                "Note": note_name,
                "Pitch": note.pitch,
                "Velocity": note.velocity,
                "Start Time": f"{note.start_time:.2f}s",
                "Duration": f"{note.duration:.2f}s"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    def _pitch_to_note_name(self, pitch: int) -> str:
        """Convert MIDI pitch to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_class = pitch % 12
        octave = pitch // 12
        return f"{note_names[pitch_class]}{octave}"
    
    def _export_midi(self):
        """Export generated music as MIDI file."""
        if not st.session_state.generated_sequence:
            st.warning("No generated sequence to export!")
            return
        
        try:
            # Create MIDI file
            timestamp = int(time.time())
            filename = f"improvai_generated_{timestamp}.mid"
            
            success = self.midi_handler.write_midi_file(
                st.session_state.generated_sequence,
                filename
            )
            
            if success:
                # Read file for download
                with open(filename, "rb") as f:
                    st.download_button(
                        label="📥 Download MIDI",
                        data=f.read(),
                        file_name=filename,
                        mime="audio/midi"
                    )
                
                # Clean up
                Path(filename).unlink()
            else:
                st.error("Failed to create MIDI file")
                
        except Exception as e:
            st.error(f"Error exporting MIDI: {e}")
    
    def _export_features(self):
        """Export musical features."""
        if not st.session_state.generated_sequence:
            st.warning("No generated sequence to analyze!")
            return
        
        try:
            # Extract features
            features = self.midi_handler.extract_features(st.session_state.generated_sequence)
            
            # Save features
            timestamp = int(time.time())
            filename = f"improvai_features_{timestamp}.json"
            
            success = self.midi_handler.save_features(features, filename)
            
            if success:
                # Read file for download
                with open(filename, "r") as f:
                    st.download_button(
                        label="📊 Download Features",
                        data=f.read(),
                        file_name=filename,
                        mime="application/json"
                    )
                
                # Clean up
                Path(filename).unlink()
            else:
                st.error("Failed to save features")
                
        except Exception as e:
            st.error(f"Error exporting features: {e}")


def main():
    """Main function to run the application."""
    try:
        app = ImprovAIApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
