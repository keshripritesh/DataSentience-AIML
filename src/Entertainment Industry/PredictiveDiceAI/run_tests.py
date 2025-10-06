#!/usr/bin/env python3
"""
Test runner script for NeuralDicePredictor.

This script runs all tests and provides a summary of results.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests and return results."""
    print("🧪 Running NeuralDicePredictor Tests...")
    print("=" * 50)
    
    # Add src to path for imports
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Test files to run
    test_files = [
        "tests/test_game_state.py",
        "tests/test_game_engine.py"
    ]
    
    results = {}
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            continue
        
        print(f"\n📋 Running tests from: {test_file}")
        print("-" * 40)
        
        try:
            # Run pytest on the test file
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Tests passed!")
                results[test_file] = "PASSED"
                passed_tests += 1
            else:
                print("❌ Tests failed!")
                print("Error output:")
                print(result.stdout)
                print("Error details:")
                print(result.stderr)
                results[test_file] = "FAILED"
                failed_tests += 1
                
        except subprocess.TimeoutExpired:
            print("⏰ Tests timed out!")
            results[test_file] = "TIMEOUT"
            failed_tests += 1
        except Exception as e:
            print(f"💥 Error running tests: {e}")
            results[test_file] = "ERROR"
            failed_tests += 1
        
        total_tests += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_file, status in results.items():
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {test_file}: {status}")
    
    print(f"\nTotal test files: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed successfully!")
        return True
    else:
        print(f"\n⚠️  {failed_tests} test file(s) failed.")
        return False

def run_simple_tests():
    """Run simple import tests to check basic functionality."""
    print("\n🔍 Running Simple Import Tests...")
    print("-" * 40)
    
    try:
        # Test basic imports
        from src.core.game_state import GamePhase, DiceAction, DiceState, PlayerState, GameState
        print("✅ Core game state imports successful")
        
        from src.core.game_engine import ScoringRule, AdvancedScoringEngine, GameEngine
        print("✅ Core game engine imports successful")
        
        # Test basic object creation
        dice_state = DiceState((1, 2, 3, 4, 5, 6))
        print("✅ DiceState creation successful")
        
        player_state = PlayerState(0, 0, 0, dice_state)
        print("✅ PlayerState creation successful")
        
        game_state = GameState(
            players=(player_state,),
            current_player=0,
            phase=GamePhase.ROLLING,
            turn_number=0,
            max_turns=10
        )
        print("✅ GameState creation successful")
        
        # Test game engine
        engine = GameEngine()
        print("✅ GameEngine creation successful")
        
        # Test scoring engine
        scoring_engine = AdvancedScoringEngine()
        score = scoring_engine.calculate_score((1, 1, 1, 2, 3, 4))
        print(f"✅ Scoring engine test successful (score: {score})")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 NeuralDicePredictor Test Suite")
    print("=" * 50)
    
    # Run simple tests first
    simple_success = run_simple_tests()
    
    if simple_success:
        # Run full test suite
        test_success = run_tests()
        
        if test_success:
            print("\n🎯 All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed. Please check the output above.")
            sys.exit(1)
    else:
        print("\n💥 Basic functionality tests failed. Cannot run full test suite.")
        sys.exit(1)
