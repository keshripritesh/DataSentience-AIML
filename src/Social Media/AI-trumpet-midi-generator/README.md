# 🎺 Trumpet MIDI Generator

An advanced Streamlit application for generating trumpet exercises and melodies using multiple AI approaches, including text-to-MIDI generation and JSON-based composition.

## Features

### 🎵 Multiple Generation Methods
- **Text-to-MIDI Model**: Uses Hugging Face's pre-trained text-to-MIDI transformer
- **JSON-based Generation**: Leverages Ollama models for structured MIDI output
- **Hybrid Approach**: Combine both methods for comparison

### 🎺 Trumpet-Specific Features
- **Range Validation**: Ensures all generated notes are playable on trumpet
- **Multiple Ranges**: Full range (E3-C6), comfortable range (Bb3-F5), beginner range (C4-C5)
- **Fingering Charts**: Automatic trumpet fingering notation
- **Difficulty Analysis**: AI-powered difficulty assessment with practice recommendations
- **Exercise Variations**: Generate rhythm, transposition, and articulation variations

### 🎧 Audio & Export
- **Audio Playback**: Convert MIDI to MP3 with custom soundfonts
- **Multiple Formats**: Export as MIDI, audio, or MusicXML
- **Real-time Analysis**: View note distributions, ranges, and difficulty metrics

## Installation

### Prerequisites
- Python 3.8 or higher
- Optional: Ollama (for JSON-based generation)
- Optional: FluidSynth (for audio generation)

### Quick Setup

1. **Clone or download the repository**
```bash
git clone <repository-url>
cd trumpet-midi-generator
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install system dependencies (Ubuntu/Debian)**
```bash
sudo apt-get update
sudo apt-get install fluidsynth
```

4. **Install system dependencies (macOS)**
```bash
brew install fluidsynth
```

5. **Install system dependencies (Windows)**
- Download FluidSynth from [official website](http://www.fluidsynth.org/)
- Add to PATH or place in project directory

### Optional: Ollama Setup

For JSON-based generation with local models:

1. **Install Ollama**
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

2. **Pull recommended models**
```bash
ollama pull llama2
ollama pull codellama
ollama pull mistral
```

## Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Choose Generation Method**
   - Text-to-MIDI: For natural language descriptions
   - JSON-based: For structured, deterministic output
   - Both: Compare different approaches

2. **Configure Settings**
   - Select note range (full/comfortable/beginner)
   - Set tempo and other parameters
   - Upload custom soundfont (optional)

3. **Enter Prompt**
   - Choose from suggested prompts
   - Or write your own description
   - Be specific about style, difficulty, and techniques

4. **Generate & Analyze**
   - Click "Generate Trumpet MIDI"
   - Listen to audio playback
   - Review difficulty analysis
   - Download MIDI files

### Example Prompts

**Beginner Level:**
- "A simple warm-up exercise for beginning trumpet players using only C, D, E, F, G"
- "Basic scale exercise in C major for trumpet"
- "Simple quarter note patterns for trumpet practice"

**Intermediate Level:**
- "A trumpet arpeggio exercise in the key of Bb major"
- "Melodic trumpet phrase with simple rhythmic variations"
- "Jazz trumpet lick with blue notes and syncopation"

**Advanced Level:**
- "A technical trumpet etude with sixteenth note passages"
- "Classical trumpet fanfare with elaborate ornamentations"
- "Contemporary trumpet piece with extended techniques"

## Configuration

### Trumpet Settings

Edit `config.py` to customize:

```python
TRUMPET_CONFIG = {
    "instrument_program": 56,  # MIDI program for trumpet
    "note_range": {
        "lowest": 52,           # E3
        "highest": 84,          # C6
        "comfortable_low": 58,  # Bb3
        "comfortable_high": 77, # F5
    },
    "default_velocity": 64,
    "default_tempo": 120
}
```

### Model Configuration

```python
MODEL_CONFIGS = {
    "text2midi": {
        "max_length": 2000,
        "temperature": 1.0,
        "top_k": 50
    },
    "ollama": {
        "timeout": 30,
        "max_retries": 3
    }
}
```

## File Structure

```
trumpet-midi-generator/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── model/
│   ├── __init__.py
│   ├── transformer_model.py    # Transformer architecture
│   └── remi_tokenizer.py       # MIDI tokenization
├── temp/                 # Temporary files
├── sounds/              # Soundfont storage
└── exports/             # Generated files
```

## API Reference

### Core Functions

**`generate_midi_from_text(prompt, model, tokenizer, device)`**
- Generate MIDI from natural language description
- Returns: MidiFile object

**`json_to_midi(json_data, instrument, tempo)`**
- Convert JSON note data to MIDI
- Format: `[["C4", 0.5], ["D4", 0.25], ...]`
- Returns: MidiFile object

**`analyze_trumpet_difficulty(midi_file)`**
- Analyze difficulty and provide recommendations
- Returns: Dictionary with metrics and suggestions

### Utility Functions

**`validate_trumpet_note(midi_note, range_type)`**
- Check if note is playable on trumpet
- Range types: "full", "comfortable", "beginner"

**`transpose_for_trumpet(notes, target_range)`**
- Automatically transpose to trumpet range

**`create_practice_variation(notes, variation_type)`**
- Generate exercise variations
- Types: "rhythm_double", "octave_up", "staccato", etc.

## Troubleshooting

### Common Issues

**"No module named 'midi2audio'"**
```bash
pip install midi2audio
# Also install FluidSynth system package
```

**"Ollama connection failed"**
- Ensure Ollama is running: `ollama serve`
- Check URL in config.py
- Verify models are pulled: `ollama list`

**"Audio conversion failed"**
- Install FluidSynth system package
- Check soundfont file path
- Try without custom soundfont

**"Model loading failed"**
- Check internet connection
- Verify Hugging Face repository access
- Ensure sufficient disk space

### Performance Tips

1. **Use GPU if available** - PyTorch will automatically use CUDA/MPS
2. **Limit generation length** - Shorter sequences generate faster
3. **Cache models** - Streamlit caches loaded models automatically
4. **Use appropriate ranges** - Smaller note ranges improve relevance

## Contributing

### Development Setup

1. **Install development dependencies**
```bash
pip install -r requirements.txt
pip install black pytest streamlit-testing
```

2. **Run tests**
```bash
pytest tests/
```

3. **Format code**
```bash
black .
```

### Adding New Features

- **New generation methods**: Extend `app.py` with additional model integrations
- **Exercise types**: Add configurations to `config.py`
- **Analysis features**: Enhance `utils.py` with new metrics
- **UI improvements**: Modify Streamlit interface in `app.py`

## License

This project is open source. See LICENSE file for details.

## Acknowledgments

- **Hugging Face**: Text-to-MIDI model from amaai-lab/text2midi
- **Ollama**: Local LLM inference
- **Streamlit**: Web application framework
- **mido**: MIDI file handling
- **FluidSynth**: Audio synthesis

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description

---

**Happy practicing! 🎺**
