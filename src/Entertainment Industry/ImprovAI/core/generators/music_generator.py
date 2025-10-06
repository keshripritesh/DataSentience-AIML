"""
Advanced music generator for ImprovAI.
Combines LSTM and Transformer models for sophisticated music generation.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass

from core.encoders.music_encoder import MusicEncoder, Note, MusicalSequence
from core.models.lstm_model import MusicLSTM
from core.models.transformer_model import MusicTransformer
from utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for music generation."""
    
    # Model selection
    model_type: str = "hybrid"  # "lstm", "transformer", "hybrid"
    
    # Generation parameters
    max_length: int = 64
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    
    # Style parameters
    style: str = "classical"  # "classical", "jazz", "pop", "custom"
    creativity: float = 0.7  # 0.0 to 1.0
    
    # Harmonic constraints
    key_signature: Optional[str] = None
    chord_progression: Optional[List[str]] = None
    
    # Rhythm parameters
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)


class MusicGenerator:
    """
    Advanced music generator that combines multiple AI models.
    
    Features:
    - Hybrid LSTM + Transformer architecture
    - Style-aware generation
    - Harmonic constraint enforcement
    - Real-time generation capabilities
    - Quality assessment and filtering
    """
    
    def __init__(self, config=None):
        """Initialize the music generator."""
        self.config = config or get_config()
        self.encoder = MusicEncoder(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.lstm_model = None
        self.transformer_model = None
        self.hybrid_model = None
        
        # Load pre-trained models if available
        self._load_models()
        
        logger.info(f"MusicGenerator initialized on device {self.device}")
    
    def _load_models(self):
        """Load pre-trained models."""
        try:
            # Load LSTM model
            lstm_path = self.config.models_dir / "lstm_model.pth"
            if lstm_path.exists():
                self.lstm_model = MusicLSTM.load_model(str(lstm_path), self.device)
                logger.info("LSTM model loaded successfully")
            
            # Load Transformer model
            transformer_path = self.config.models_dir / "transformer_model.pth"
            if transformer_path.exists():
                self.transformer_model = MusicTransformer.load_model(str(transformer_path), self.device)
                logger.info("Transformer model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Will use randomly initialized models")
    
    def _initialize_models(self, vocab_size: int):
        """Initialize models with given vocabulary size."""
        if self.lstm_model is None:
            self.lstm_model = MusicLSTM(
                vocab_size=vocab_size,
                embedding_dim=256,
                hidden_size=self.config.model.lstm_hidden_size,
                num_layers=self.config.model.lstm_num_layers,
                dropout=self.config.model.lstm_dropout
            ).to(self.device)
        
        if self.transformer_model is None:
            self.transformer_model = MusicTransformer(
                vocab_size=vocab_size,
                d_model=self.config.model.transformer_d_model,
                n_heads=self.config.model.transformer_nhead,
                n_layers=self.config.model.transformer_num_layers,
                dropout=self.config.model.transformer_dropout
            ).to(self.device)
    
    def generate_continuation(self, 
                            input_notes: List[Note],
                            generation_config: GenerationConfig) -> MusicalSequence:
        """
        Generate a musical continuation from input notes.
        
        Args:
            input_notes: List of input notes
            generation_config: Generation configuration
            
        Returns:
            MusicalSequence with generated continuation
        """
        if not input_notes:
            logger.warning("No input notes provided")
            return MusicalSequence(notes=[])
        
        # Encode input notes
        encoded_input = self.encoder.encode_notes(input_notes)
        
        if len(encoded_input) == 0:
            logger.warning("Failed to encode input notes")
            return MusicalSequence(notes=[])
        
        # Initialize models if needed
        vocab_size = self.encoder.get_vocabulary_size()
        self._initialize_models(vocab_size)
        
        # Convert to tensor
        input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate continuation based on model type
        if generation_config.model_type == "lstm":
            generated_sequence = self._generate_with_lstm(input_tensor, generation_config)
        elif generation_config.model_type == "transformer":
            generated_sequence = self._generate_with_transformer(input_tensor, generation_config)
        elif generation_config.model_type == "hybrid":
            generated_sequence = self._generate_with_hybrid(input_tensor, generation_config)
        else:
            raise ValueError(f"Unknown model type: {generation_config.model_type}")
        
        # Decode generated sequence
        generated_notes = self.encoder.decode_notes(generated_sequence)
        
        # Apply post-processing
        processed_notes = self._post_process_notes(generated_notes, generation_config)
        
        # Create musical sequence
        result = MusicalSequence(
            notes=processed_notes,
            tempo=generation_config.tempo,
            time_signature=generation_config.time_signature,
            key_signature=generation_config.key_signature or self._estimate_key(input_notes)
        )
        
        logger.info(f"Generated continuation with {len(processed_notes)} notes")
        return result
    
    def _generate_with_lstm(self, input_tensor: torch.Tensor, 
                           config: GenerationConfig) -> np.ndarray:
        """Generate continuation using LSTM model."""
        self.lstm_model.eval()
        
        with torch.no_grad():
            generated = self.lstm_model.generate(
                input_ids=input_tensor,
                max_length=config.max_length,
                temperature=config.temperature * config.creativity,
                top_k=config.top_k,
                top_p=config.top_p,
                do_sample=config.do_sample
            )
        
        return generated.squeeze(0).cpu().numpy()
    
    def _generate_with_transformer(self, input_tensor: torch.Tensor,
                                  config: GenerationConfig) -> np.ndarray:
        """Generate continuation using Transformer model."""
        self.transformer_model.eval()
        
        with torch.no_grad():
            generated = self.transformer_model.generate(
                input_ids=input_tensor,
                max_length=config.max_length,
                temperature=config.temperature * config.creativity,
                top_k=config.top_k,
                top_p=config.top_p,
                do_sample=config.do_sample
            )
        
        return generated.squeeze(0).cpu().numpy()
    
    def _generate_with_hybrid(self, input_tensor: torch.Tensor,
                             config: GenerationConfig) -> np.ndarray:
        """
        Generate continuation using hybrid approach.
        Combines LSTM and Transformer outputs for better quality.
        """
        # Generate with both models
        lstm_output = self._generate_with_lstm(input_tensor, config)
        transformer_output = self._generate_with_transformer(input_tensor, config)
        
        # Combine outputs (weighted average based on sequence length)
        if len(lstm_output) <= 32:
            # Use more LSTM for shorter sequences
            weight_lstm = 0.7
            weight_transformer = 0.3
        else:
            # Use more Transformer for longer sequences
            weight_lstm = 0.3
            weight_transformer = 0.7
        
        # Interpolate between outputs
        combined_output = []
        max_len = max(len(lstm_output), len(transformer_output))
        
        for i in range(max_len):
            if i < len(lstm_output) and i < len(transformer_output):
                # Weighted combination
                combined_token = int(
                    weight_lstm * lstm_output[i] + 
                    weight_transformer * transformer_output[i]
                )
            elif i < len(lstm_output):
                combined_token = lstm_output[i]
            else:
                combined_token = transformer_output[i]
            
            combined_output.append(combined_token)
        
        return np.array(combined_output)
    
    def _post_process_notes(self, notes: List[Note], 
                           config: GenerationConfig) -> List[Note]:
        """
        Apply post-processing to generated notes.
        
        Features:
        - Harmonic constraint enforcement
        - Rhythm smoothing
        - Style-specific adjustments
        - Quality filtering
        """
        if not notes:
            return notes
        
        # Apply harmonic constraints
        if config.key_signature or config.chord_progression:
            notes = self._apply_harmonic_constraints(notes, config)
        
        # Apply style-specific processing
        notes = self._apply_style_processing(notes, config.style)
        
        # Smooth rhythm patterns
        notes = self._smooth_rhythm(notes)
        
        # Filter low-quality notes
        notes = self._filter_notes(notes)
        
        return notes
    
    def _apply_harmonic_constraints(self, notes: List[Note], 
                                   config: GenerationConfig) -> List[Note]:
        """Apply harmonic constraints to notes."""
        if not config.key_signature and not config.chord_progression:
            return notes
        
        processed_notes = []
        
        for note in notes:
            # Check if note fits the key signature
            if config.key_signature:
                if self._note_in_key(note.pitch, config.key_signature):
                    processed_notes.append(note)
                else:
                    # Transpose to nearest in-key note
                    corrected_pitch = self._correct_pitch_to_key(note.pitch, config.key_signature)
                    corrected_note = Note(
                        pitch=corrected_pitch,
                        velocity=note.velocity,
                        start_time=note.start_time,
                        end_time=note.end_time,
                        duration=note.duration
                    )
                    processed_notes.append(corrected_note)
            else:
                processed_notes.append(note)
        
        return processed_notes
    
    def _note_in_key(self, pitch: int, key_signature: str) -> bool:
        """Check if a note is in the given key signature."""
        # Simplified key signature checking
        pitch_class = pitch % 12
        
        # Define notes in major keys (simplified)
        major_keys = {
            'C': [0, 2, 4, 5, 7, 9, 11],  # C, D, E, F, G, A, B
            'G': [0, 2, 4, 5, 7, 9, 11],  # G, A, B, C, D, E, F#
            'D': [0, 2, 4, 5, 7, 9, 11],  # D, E, F#, G, A, B, C#
            'A': [0, 2, 4, 5, 7, 9, 11],  # A, B, C#, D, E, F#, G#
            'E': [0, 2, 4, 5, 7, 9, 11],  # E, F#, G#, A, B, C#, D#
            'B': [0, 2, 4, 5, 7, 9, 11],  # B, C#, D#, E, F#, G#, A#
            'F#': [0, 2, 4, 5, 7, 9, 11], # F#, G#, A#, B, C#, D#, E#
            'C#': [0, 2, 4, 5, 7, 9, 11], # C#, D#, E#, F#, G#, A#, B#
        }
        
        # Extract key name (remove 'maj' or 'min')
        key_name = key_signature.replace('maj', '').replace('min', '')
        
        if key_name in major_keys:
            return pitch_class in major_keys[key_name]
        
        return True  # Default to allowing all notes
    
    def _correct_pitch_to_key(self, pitch: int, key_signature: str) -> int:
        """Correct a pitch to fit the key signature."""
        pitch_class = pitch % 12
        octave = pitch // 12
        
        # Find nearest in-key note
        key_name = key_signature.replace('maj', '').replace('min', '')
        
        # Simplified correction (move to nearest scale degree)
        scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # Major scale
        
        # Find closest scale degree
        distances = [abs(pitch_class - degree) for degree in scale_degrees]
        closest_degree = scale_degrees[np.argmin(distances)]
        
        return closest_degree + (octave * 12)
    
    def _apply_style_processing(self, notes: List[Note], style: str) -> List[Note]:
        """Apply style-specific processing to notes."""
        if style == "jazz":
            return self._apply_jazz_style(notes)
        elif style == "pop":
            return self._apply_pop_style(notes)
        elif style == "classical":
            return self._apply_classical_style(notes)
        else:
            return notes
    
    def _apply_jazz_style(self, notes: List[Note]) -> List[Note]:
        """Apply jazz-style processing."""
        processed_notes = []
        
        for note in notes:
            # Add swing rhythm (simplified)
            if note.start_time % 0.5 == 0:  # On beat
                # Keep original timing
                processed_notes.append(note)
            else:
                # Add slight delay for off-beat notes
                swung_note = Note(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start_time=note.start_time + 0.05,
                    end_time=note.end_time + 0.05,
                    duration=note.duration
                )
                processed_notes.append(swung_note)
        
        return processed_notes
    
    def _apply_pop_style(self, notes: List[Note]) -> List[Note]:
        """Apply pop-style processing."""
        processed_notes = []
        
        for note in notes:
            # Emphasize strong beats
            if note.start_time % 1.0 == 0:  # On strong beat
                # Increase velocity
                pop_note = Note(
                    pitch=note.pitch,
                    velocity=min(127, note.velocity + 10),
                    start_time=note.start_time,
                    end_time=note.end_time,
                    duration=note.duration
                )
                processed_notes.append(pop_note)
            else:
                processed_notes.append(note)
        
        return processed_notes
    
    def _apply_classical_style(self, notes: List[Note]) -> List[Note]:
        """Apply classical-style processing."""
        processed_notes = []
        
        for note in notes:
            # More conservative velocity changes
            classical_note = Note(
                pitch=note.pitch,
                velocity=max(40, min(100, note.velocity)),  # Moderate dynamics
                start_time=note.start_time,
                end_time=note.end_time,
                duration=note.duration
            )
            processed_notes.append(classical_note)
        
        return processed_notes
    
    def _smooth_rhythm(self, notes: List[Note]) -> List[Note]:
        """Smooth rhythm patterns for more natural flow."""
        if len(notes) < 2:
            return notes
        
        processed_notes = []
        
        for i, note in enumerate(notes):
            if i == 0:
                processed_notes.append(note)
                continue
            
            # Check for unrealistic timing gaps
            time_gap = note.start_time - notes[i-1].end_time
            
            if time_gap > 2.0:  # Gap too large
                # Adjust timing
                adjusted_note = Note(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start_time=notes[i-1].end_time + 0.5,  # Reasonable gap
                    end_time=notes[i-1].end_time + 0.5 + note.duration,
                    duration=note.duration
                )
                processed_notes.append(adjusted_note)
            else:
                processed_notes.append(note)
        
        return processed_notes
    
    def _filter_notes(self, notes: List[Note]) -> List[Note]:
        """Filter out low-quality or unrealistic notes."""
        filtered_notes = []
        
        for note in notes:
            # Check for valid pitch range
            if not (21 <= note.pitch <= 108):
                continue
            
            # Check for valid velocity
            if not (1 <= note.velocity <= 127):
                continue
            
            # Check for valid duration
            if note.duration <= 0 or note.duration > 10:
                continue
            
            # Check for valid timing
            if note.start_time < 0 or note.end_time <= note.start_time:
                continue
            
            filtered_notes.append(note)
        
        return filtered_notes
    
    def _estimate_key(self, notes: List[Note]) -> str:
        """Estimate key signature from notes."""
        if not notes:
            return "C"
        
        # Extract pitch classes
        pitch_classes = [note.pitch % 12 for note in notes]
        
        # Create pitch class histogram
        hist = np.zeros(12)
        for pc in pitch_classes:
            hist[pc] += 1
        
        # Normalize
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        # Use encoder's key estimation
        return self.encoder._estimate_key_signature(hist)
    
    def assess_quality(self, sequence: MusicalSequence) -> Dict[str, float]:
        """
        Assess the quality of a generated musical sequence.
        
        Returns:
            Dictionary with quality metrics
        """
        if not sequence.notes:
            return {"overall_quality": 0.0}
        
        metrics = {}
        
        # Harmonic coherence
        harmonic_features = self.encoder.extract_harmonic_features(sequence.notes)
        if 'pitch_class_histogram' in harmonic_features:
            # Calculate entropy as measure of harmonic variety
            hist = harmonic_features['pitch_class_histogram']
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            metrics['harmonic_variety'] = min(1.0, entropy / 2.5)  # Normalize
        
        # Rhythmic consistency
        rhythm_pattern = harmonic_features.get('rhythm_pattern', np.array([]))
        if len(rhythm_pattern) > 0:
            # Calculate rhythm consistency
            rhythm_std = np.std(rhythm_pattern)
            metrics['rhythm_consistency'] = max(0.0, 1.0 - rhythm_std / 2.0)
        
        # Melodic contour
        pitches = [note.pitch for note in sequence.notes]
        if len(pitches) > 1:
            # Calculate melodic smoothness
            pitch_diffs = np.diff(pitches)
            large_jumps = np.sum(np.abs(pitch_diffs) > 12)  # Octave jumps
            metrics['melodic_smoothness'] = max(0.0, 1.0 - large_jumps / len(pitches))
        
        # Overall quality (weighted average)
        weights = {
            'harmonic_variety': 0.3,
            'rhythm_consistency': 0.3,
            'melodic_smoothness': 0.4
        }
        
        overall_quality = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                overall_quality += metrics[metric] * weight
        
        metrics['overall_quality'] = overall_quality
        
        return metrics
    
    def generate_multiple_variations(self, 
                                   input_notes: List[Note],
                                   generation_config: GenerationConfig,
                                   num_variations: int = 3) -> List[MusicalSequence]:
        """
        Generate multiple variations of a musical continuation.
        
        Args:
            input_notes: Input notes
            generation_config: Generation configuration
            num_variations: Number of variations to generate
            
        Returns:
            List of MusicalSequence variations
        """
        variations = []
        
        for i in range(num_variations):
            # Vary generation parameters slightly
            varied_config = GenerationConfig(
                model_type=generation_config.model_type,
                max_length=generation_config.max_length,
                temperature=generation_config.temperature * (0.8 + 0.4 * np.random.random()),
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                do_sample=generation_config.do_sample,
                style=generation_config.style,
                creativity=generation_config.creativity,
                key_signature=generation_config.key_signature,
                chord_progression=generation_config.chord_progression,
                tempo=generation_config.tempo,
                time_signature=generation_config.time_signature
            )
            
            variation = self.generate_continuation(input_notes, varied_config)
            variations.append(variation)
        
        return variations
