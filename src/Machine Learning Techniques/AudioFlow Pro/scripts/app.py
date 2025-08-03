"""
AudioFlow Pro - Main Application
Advanced PDF to Speech Conversion Platform
"""

import streamlit as st
import logging
from typing import Optional, Tuple
from pathlib import Path
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
from Pipeline import AudioFlowPipeline
from config import AudioFlowConfig

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AudioFlow Pro - Document to Speech",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def create_sidebar():
    """Create the sidebar with configuration options."""
    st.sidebar.title("🎵 AudioFlow Pro")
    st.sidebar.markdown("---")
    
    # Language selection
    st.sidebar.subheader("Language Settings")
    language = st.sidebar.selectbox(
        "Select Language",
        options=AudioFlowConfig.SUPPORTED_LANGUAGES,
        index=0,
        help="Choose the language for text-to-speech conversion"
    )
    
    # Voice settings
    st.sidebar.subheader("Voice Settings")
    speech_rate = st.sidebar.slider(
        "Speech Rate",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Adjust the speed of speech (0.5 = slow, 2.0 = fast)"
    )
    
    # Quality settings
    st.sidebar.subheader("Quality Settings")
    quality = st.sidebar.selectbox(
        "Audio Quality",
        options=["Standard", "Premium"],
        index=0,
        help="Choose audio quality (Premium requires API key)"
    )
    
    return {
        "language": language,
        "speech_rate": speech_rate,
        "quality": quality
    }

def display_header():
    """Display the main header and description."""
    st.title("🎵 AudioFlow Pro")
    st.markdown("### Intelligent Document-to-Speech Conversion Platform")
    
    st.markdown("""
    Transform your PDF documents into engaging audio content with our advanced AI-powered platform.
    Support for 50+ languages with natural-sounding voice synthesis.
    """)

def display_upload_section():
    """Display the file upload section."""
    st.subheader("📄 Upload Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file to convert",
            type=['pdf'],
            help="Upload a PDF file (max 100MB)"
        )
    
    with col2:
        st.markdown("""
        **Supported Formats:**
        - PDF documents
        - Max file size: 100MB
        - Text-based PDFs work best
        """)
    
    return uploaded_file

def display_processing_status():
    """Display processing status and progress."""
    with st.spinner("Processing your document..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate processing steps
        steps = [
            "Extracting text from PDF...",
            "Cleaning and formatting text...",
            "Generating speech audio...",
            "Optimizing audio quality..."
        ]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) * 25)
            st.empty()  # Small delay for visual effect
        
        return progress_bar, status_text

def display_results(audio_file, error_message: Optional[str] = None):
    """Display conversion results."""
    if error_message:
        st.error(f"❌ Conversion Error: {error_message}")
        st.info("💡 **Tips:** Ensure your PDF contains readable text and is not corrupted.")
        return
    
    if audio_file:
        st.success("✅ Conversion completed successfully!")
        
        # Display audio player
        st.subheader("🎧 Preview Audio")
        st.audio(audio_file, format='audio/mp3')
        
        # Download section
        st.subheader("📥 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download MP3",
                data=audio_file,
                file_name=f"audioflow_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                mime="audio/mp3",
                help="Download the converted audio file"
            )
        
        with col2:
            if st.button("🔄 Convert Another File"):
                st.rerun()

def display_analytics():
    """Display usage analytics and statistics."""
    st.subheader("📊 Usage Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents Processed", "1,247", "+12%")
    
    with col2:
        st.metric("Languages Supported", "50+", "")
    
    with col3:
        st.metric("Processing Speed", "1000 words/min", "")

def main():
    """Main application function."""
    try:
        # Setup
        setup_page_config()
        config = create_sidebar()
        
        # Display header
        display_header()
        
        # File upload
        uploaded_file = display_upload_section()
        
        if uploaded_file is not None:
            # Validate file
            if uploaded_file.size > 100 * 1024 * 1024:  # 100MB limit
                st.error("❌ File size exceeds 100MB limit. Please upload a smaller file.")
                return
            
            # Process file
            progress_bar, status_text = display_processing_status()
            
            try:
                # Initialize pipeline
                pipeline = AudioFlowPipeline(config)
                
                # Process the file
                audio_file, error_message = pipeline.process_file(uploaded_file)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                display_results(audio_file, error_message)
                
            except Exception as e:
                logger.error(f"Pipeline processing error: {e}")
                st.error(f"❌ Processing error: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")
        
        # Display analytics
        st.markdown("---")
        display_analytics()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("❌ Application error. Please refresh the page and try again.")
        st.info("If the problem persists, please contact support.")

if __name__ == "__main__":
    main()
