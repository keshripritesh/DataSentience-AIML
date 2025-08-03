"""
Configuration settings for the Trumpet MIDI Generator
"""

import os
from pathlib import Path

# Application settings
APP_NAME = "Trumpet MIDI Generator"
VERSION = "1.0.0"

# Model settings
HUGGINGFACE_REPO = "amaai-lab/text2midi"
OLLAMA_BASE_URL = "http://localhost:11434"

# Trumpet-specific MIDI settings
TRUMPET_CONFIG = {
    "instrument_program": 56,  # MIDI program number for trumpet
    "note_range": {
        "lowest": 52,           # E3 - absolute lowest note
        "highest": 84,          # C6 - absolute highest note
        "comfortable_low": 58,  # Bb3 - comfortable low note
        "comfortable_high": 77, # F5 - comfortable high note
        "beginner_low": 60,     # C4 - beginner range low
        "beginner_high": 72     # C5 - beginner range high
    },
    "default_velocity": 64,
    "default_tempo": 120,
    "default_time_signature": (4, 4)
}

# Exercise types and their characteristics
EXERCISE_TYPES = {
    "warm_up": {
        "tempo_range": (60, 100),
        "note_range": "comfortable",
        "rhythm_complexity": "simple",
        "suggested_duration": 16  # beats
    },
    "technical": {
        "tempo_range": (80, 140),
        "note_range": "full",
        "rhythm_complexity": "moderate",
        "suggested_duration": 32
    },
    "lyrical": {
        "tempo_range": (60, 90),
        "note_range": "comfortable",
        "rhythm_complexity": "simple",
        "suggested_duration": 24
    },
    "jazz": {
        "tempo_range": (100, 180),
        "note_range": "full",
        "rhythm_complexity": "complex",
        "suggested_duration": 32
    },
    "classical": {
        "tempo_range": (60, 120),
        "note_range": "full",
        "rhythm_complexity": "moderate",
        "suggested_duration": 40
    }
}

# Audio settings
AUDIO_CONFIG = {
    "sample_rate": 44100,
    "bit_depth": 16,
    "format": "mp3",
    "quality": "medium",
    "default_soundfont": None  # Path to default soundfont
}

# MIDI conversion settings
MIDI_CONFIG = {
    "ticks_per_beat": 480,
    "default_velocity": 64,
    "max_duration": 4.0,  # Maximum note duration in beats
    "min_duration": 0.125  # Minimum note duration in beats
}

# File paths
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
SOUNDS_DIR = BASE_DIR / "sounds"
EXPORTS_DIR = BASE_DIR / "exports"

# Ensure directories exist
for dir_path in [TEMP_DIR, SOUNDS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Suggested prompts for different skill levels
PROMPT_TEMPLATES = {
    "beginner": [
        "A simple warm-up exercise for beginning trumpet players using only C, D, E, F, G",
        "A basic scale exercise in C major for trumpet",
        "Simple quarter note patterns for trumpet practice",
        "Beginner trumpet exercise with repeated notes for valve practice",
        "Easy melody in the middle register for new trumpet players"
    ],
    "intermediate": [
        "A trumpet arpeggio exercise in the key of Bb major",
        "Intermediate trumpet exercise with eighth note patterns",
        "A melodic trumpet phrase with simple rhythmic variations",
        "Trumpet exercise focusing on smooth slur connections",
        "Modal exercise for trumpet in the Dorian mode"
    ],
    "advanced": [
        "A technical trumpet etude with sixteenth note passages",
        "Advanced trumpet exercise with complex rhythmic patterns",
        "A virtuosic trumpet passage with rapid articulation",
        "Jazz trumpet lick with blue notes and syncopation",
        "Classical trumpet fanfare with elaborate ornamentations"
    ],
    "style_specific": {
        "classical": [
            "A baroque-style trumpet fanfare in D major",
            "Classical trumpet sonata excerpt with elegant phrasing",
            "Orchestral trumpet excerpt from a symphony",
            "Chamber music trumpet part with precise articulation"
        ],
        "jazz": [
            "A bebop trumpet lick with chromatic passing tones",
            "Swing-style trumpet phrase with syncopated rhythms",
            "Blues trumpet solo with expressive bends",
            "Latin jazz trumpet montuno pattern"
        ],
        "contemporary": [
            "Modern trumpet piece with extended techniques",
            "Minimalist trumpet pattern with repeated motifs",
            "Contemporary trumpet piece with irregular meters",
            "Avant-garde trumpet gesture with microtonal elements"
        ]
    }
}

# Model-specific configurations
MODEL_CONFIGS = {
    "text2midi": {
        "max_length": 2000,
        "temperature": 1.0,
        "top_k": 50,
        "vocab_size": 5000  # Will be updated when model loads
    },
    "ollama": {
        "timeout": 30,
        "max_retries": 3,
        "system_prompt_template": """
You are a professional trumpet instructor and music composer specializing in creating educational exercises for brass players.

Generate a trumpet exercise in JSON format with the following structure:
[
  ["note_name", duration],
  ["note_name", duration],
  ...
]

Guidelines:
- Use standard note names (C4, Bb3, F#5, etc.)
- Keep notes within practical trumpet range: {note_range}
- Duration values: 0.125=eighth, 0.25=quarter, 0.5=half, 1=whole, 2=breve
- Create musically logical progressions
- Consider trumpet valve combinations and fingerings
- Ensure the exercise serves the specified pedagogical purpose

Exercise type: {exercise_type}
Difficulty level: {difficulty}
Tempo: {tempo} BPM

Only return the JSON array, no additional text.
"""
    }
}

# Validation settings
VALIDATION = {
    "max_prompt_length": 500,
    "max_midi_duration": 300,  # seconds
    "min_notes_per_exercise": 4,
    "max_notes_per_exercise": 100
}

# UI Settings
UI_CONFIG = {
    "page_title": "🎺 Trumpet MIDI Generator",
    "page_icon": "🎺",
    "layout": "wide",
    "sidebar_state": "expanded",
    "theme": "light"
}

# Export settings
EXPORT_FORMATS = {
    "midi": {
        "extension": ".mid",
        "mime_type": "audio/midi"
    },
    "musicxml": {
        "extension": ".xml",
        "mime_type": "application/vnd.recordare.musicxml+xml"
    },
    "audio": {
        "extension": ".mp3",
        "mime_type": "audio/mpeg"
    }
}
