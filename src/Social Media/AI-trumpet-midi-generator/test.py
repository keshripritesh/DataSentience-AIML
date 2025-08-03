import streamlit as st
import pickle
import torch
from transformers import T5Tokenizer
from huggingface_hub import hf_hub_download

# Note mapping for trumpet
NOTE_MAP = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11
}

# --- App Title ---
st.title("🎺 Trumpet MIDI Generator")
st.markdown("Generate trumpet melodies from text prompts using AI")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Creativity (temperature)", 0.1, 2.0, 1.0, 0.1)
    max_length = st.slider("Max Length", 100, 2000, 1000, 100)

# --- Main Input ---
prompt = st.text_area(
    "Enter your music description:",
    height=150,
    value="A jazzy trumpet solo in Bb major with syncopated rhythms"
)

# --- Generate Button ---
if st.button("Generate MIDI", type="primary"):
    with st.spinner("Generating trumpet melody..."):
        try:
            # --- Load Model (cached) ---
            @st.cache_resource
            def load_model():
                # Initialize device
                device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

                # Download and load tokenizer
                tokenizer_path = hf_hub_download(repo_id="amaai-lab/text2midi", filename="vocab_remi.pkl")
                with open(tokenizer_path, "rb") as f:
                    r_tokenizer = pickle.load(f)

                # Load model with correct parameters
                model = Transformer(
                    vocab_size=len(r_tokenizer),
                    d_model=768,
                    num_heads=8,  # Changed from nhead to num_heads
                    dim_feedforward=2048,
                    num_encoder_layers=12,  # Reduced from 18 to match typical implementations
                    max_seq_length=1024,
                    device=device
                )
                model_path = hf_hub_download(repo_id="amaai-lab/text2midi", filename="pytorch_model.bin")
                model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                model.eval()

                # Load text tokenizer
                tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

                return model, tokenizer, r_tokenizer, device


            model, tokenizer, r_tokenizer, device = load_model()

            # --- Process Input ---
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # --- Generate MIDI ---
            output = model.generate(
                input_ids,
                attention_mask,
                max_len=max_length,
                temperature=temperature
            )

            # --- Convert to MIDI ---
            output_list = output[0].tolist()
            midi = r_tokenizer.decode(output_list)

            # --- Save and Download ---
            midi_path = "trumpet_output.mid"
            midi.dump_midi(midi_path)

            st.success("MIDI generated successfully!")

            # Audio preview
            st.audio(midi_path, format="audio/midi")

            # Download button
            with open(midi_path, "rb") as f:
                st.download_button(
                    label="Download MIDI",
                    data=f,
                    file_name="trumpet_solo.mid",
                    mime="audio/midi"
                )

        except Exception as e:
            st.error(f"Error generating MIDI: {str(e)}")

# --- Trumpet Note Reference ---
st.divider()
st.subheader("Trumpet Note Reference")
st.write("Standard MIDI note numbers for trumpet (C4 = 60):")

cols = st.columns(4)
for i, (note, val) in enumerate(NOTE_MAP.items()):
    with cols[i % 4]:
        st.metric(label=note, value=f"MIDI {60 + val} (C4 + {val})")