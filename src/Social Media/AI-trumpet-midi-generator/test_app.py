#!/usr/bin/env python3
"""
Test script for Trumpet MIDI Generator
Run this to verify the installation is working correctly
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")

    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

    try:
        import torch
        print(f"✅ PyTorch imported (version: {torch.__version__})")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import transformers
        print(f"✅ Transformers imported (version: {transformers.__version__})")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False

    try:
        import mido
        try:
            version = mido.__version__
            print(f"✅ Mido imported (version: {version})")
        except AttributeError:
            print("✅ Mido imported (version: unknown)")
    except ImportError as e:
        print(f"❌ Mido import failed: {e}")
        return False

    try:
        import huggingface_hub
        print("✅ Hugging Face Hub imported")
    except ImportError as e:
        print(f"❌ Hugging Face Hub import failed: {e}")
        return False

    # Test optional imports
    try:
        import midi2audio
        from pydub import AudioSegment
        print("✅ Audio libraries available")
    except ImportError:
        print("⚠️  Audio libraries not available (audio playback will be disabled)")

    try:
        import numpy as np
        print("✅ NumPy imported")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    return True

def test_model_imports():
    """Test custom model imports"""
    print("\n🧪 Testing custom model imports...")

    try:
        from model.transformer_model import Transformer
        print("✅ Transformer model imported")
    except ImportError as e:
        print(f"❌ Transformer model import failed: {e}")
        return False

    try:
        from model.remi_tokenizer import REMITokenizer
        print("✅ REMI tokenizer imported")
    except ImportError as e:
        print(f"❌ REMI tokenizer import failed: {e}")
        return False

    return True

def test_config():
    """Test configuration"""
    print("\n🧪 Testing configuration...")

    try:
        import config
        print("✅ Config imported")
        print(f"   - Trumpet range: {config.TRUMPET_CONFIG['note_range']['lowest']}-{config.TRUMPET_CONFIG['note_range']['highest']}")
        print(f"   - Default tempo: {config.TRUMPET_CONFIG['default_tempo']}")
        return True
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\n🧪 Testing utility functions...")

    try:
        import utils

        # Test note validation
        result = utils.validate_trumpet_note(60, "full")  # C4
        print(f"✅ Note validation works (C4 in full range: {result})")

        # Test fingering
        fingering = utils.midi_note_to_trumpet_fingering(60)  # C4
        print(f"✅ Fingering chart works (C4: {fingering})")

        return True
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Utils function test failed: {e}")
        return False

def test_basic_midi_generation():
    """Test basic MIDI generation"""
    print("\n🧪 Testing basic MIDI generation...")

    try:
        # Import required functions
        sys.path.append('.')
        from app import json_to_midi, note_name_to_midi

        # Test note conversion
        midi_note = note_name_to_midi("C4")
        print(f"✅ Note conversion works (C4 = MIDI {midi_note})")

        # Test JSON to MIDI conversion
        test_json = [["C4", 1.0], ["D4", 0.5], ["E4", 0.5], ["F4", 1.0]]
        midi_file = json_to_midi(test_json)
        print("✅ JSON to MIDI conversion works")

        # Save test MIDI file
        test_output = Path("test_output.mid")
        midi_file.save(str(test_output))
        print(f"✅ Test MIDI file saved: {test_output}")

        # Clean up
        if test_output.exists():
            test_output.unlink()
            print("✅ Test file cleaned up")

        return True
    except Exception as e:
        print(f"❌ Basic MIDI generation test failed: {e}")
        return False

def test_device_availability():
    """Test device availability for ML models"""
    print("\n🧪 Testing device availability...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA available (GPU: {torch.cuda.get_device_name()})")
            device = "cuda"
        elif torch.backends.mps.is_available():
            print("✅ MPS available (Apple Silicon)")
            device = "mps"
        else:
            print("✅ CPU available")
            device = "cpu"

        print(f"   Selected device: {device}")
        return True
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\n🧪 Testing directory structure...")

    required_dirs = ["model", "temp", "sounds", "exports"]
    required_files = ["app.py", "config.py", "utils.py", "requirements.txt"]

    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            return False

    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✅ File exists: {file_name}")
        else:
            print(f"❌ File missing: {file_name}")
            return False

    return True

def run_all_tests():
    """Run all tests"""
    print("🎺 Trumpet MIDI Generator - Installation Test")
    print("=" * 50)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Python Imports", test_imports),
        ("Model Imports", test_model_imports),
        ("Configuration", test_config),
        ("Utilities", test_utils),
        ("Device Availability", test_device_availability),
        ("Basic MIDI Generation", test_basic_midi_generation),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The installation looks good.")
        print("You can now run: streamlit run app.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("You may need to install missing dependencies or fix configuration issues.")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
