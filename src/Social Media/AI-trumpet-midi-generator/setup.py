#!/usr/bin/env python3
"""
Setup script for Trumpet MIDI Generator
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def check_system_dependencies():
    """Check and provide instructions for system dependencies"""
    print("\n🔧 Checking system dependencies...")

    system = platform.system().lower()

    if system == "linux":
        print("📋 For Linux (Ubuntu/Debian), run:")
        print("   sudo apt-get update")
        print("   sudo apt-get install fluidsynth")
    elif system == "darwin":  # macOS
        print("📋 For macOS, run:")
        print("   brew install fluidsynth")
    elif system == "windows":
        print("📋 For Windows:")
        print("   Download FluidSynth from: http://www.fluidsynth.org/")
        print("   Add to PATH or place in project directory")

    print("\n⚠️  FluidSynth is optional but recommended for audio playback")

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")

    directories = ["temp", "sounds", "exports", "model"]

    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {dir_name}")

def check_ollama():
    """Check if Ollama is available"""
    print("\n🦙 Checking Ollama installation...")

    try:
        result = subprocess.run(["ollama", "--version"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print("📋 To use Ollama features, make sure it's running:")
            print("   ollama serve")
            print("📋 Recommended models to pull:")
            print("   ollama pull llama2")
            print("   ollama pull mistral")
            return True
        else:
            print("⚠️  Ollama not found (optional)")
            print("📋 To install Ollama:")
            print("   curl -fsSL https://ollama.ai/install.sh | sh")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  Ollama not found (optional)")
        print("📋 To install Ollama, visit: https://ollama.ai/")
        return False

def test_installation():
    """Test basic functionality"""
    print("\n🧪 Testing installation...")

    try:
        # Test basic imports
        import streamlit
        import torch
        import transformers
        import mido
        print("✅ Core dependencies imported successfully")

        # Test optional imports
        try:
            import midi2audio
            from pydub import AudioSegment
            print("✅ Audio dependencies available")
        except ImportError:
            print("⚠️  Audio dependencies not available (optional)")

        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def create_launch_script():
    """Create a launch script for easy startup"""
    print("\n🚀 Creating launch script...")

    script_content = """#!/bin/bash
# Trumpet MIDI Generator Launch Script

echo "🎺 Starting Trumpet MIDI Generator..."

# Check if virtual environment should be activated
if [ -d "venv" ]; then
    echo "📁 Activating virtual environment..."
    source venv/bin/activate
fi

# Start Streamlit app
echo "🌐 Launching Streamlit app..."
streamlit run app.py

echo "👋 Goodbye!"
"""

    with open("launch.sh", "w") as f:
        f.write(script_content)

    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("launch.sh", 0o755)

    print("✅ Created launch.sh script")

    # Create Windows batch file
    if platform.system() == "Windows":
        batch_content = """@echo off
echo 🎺 Starting Trumpet MIDI Generator...

REM Check if virtual environment should be activated
if exist "venv" (
    echo 📁 Activating virtual environment...
    call venv\\Scripts\\activate
)

REM Start Streamlit app
echo 🌐 Launching Streamlit app...
streamlit run app.py

echo 👋 Goodbye!
pause
"""
        with open("launch.bat", "w") as f:
            f.write(batch_content)
        print("✅ Created launch.bat script")

def main():
    """Main setup function"""
    print("🎺 Trumpet MIDI Generator Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Setup directories
    setup_directories()

    # Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Setup failed due to dependency installation issues")
        sys.exit(1)

    # Test installation
    if not test_installation():
        print("\n❌ Setup failed due to import issues")
        sys.exit(1)

    # Check system dependencies
    check_system_dependencies()

    # Check Ollama
    check_ollama()

    # Create launch script
    create_launch_script()

    # Final instructions
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Install system dependencies (FluidSynth) if needed")
    print("2. (Optional) Install and configure Ollama")
    print("3. Run the application:")
    print("   • Linux/macOS: ./launch.sh")
    print("   • Windows: launch.bat")
    print("   • Or directly: streamlit run app.py")

    print("\n🎺 Happy music making!")

if __name__ == "__main__":
    main()
