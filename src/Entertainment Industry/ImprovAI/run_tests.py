#!/usr/bin/env python3
"""
Test runner for ImprovAI project.
"""

import sys
import os
import subprocess
import pytest
from pathlib import Path

def main():
    """Run all tests for the ImprovAI project."""
    print("🎵 ImprovAI Test Suite")
    print("=" * 50)
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Test categories
    test_categories = [
        ("Unit Tests", "tests/unit/"),
        ("Integration Tests", "tests/integration/"),
        ("Performance Tests", "tests/performance/")
    ]
    
    total_passed = 0
    total_failed = 0
    
    for category_name, test_path in test_categories:
        print(f"\n📋 Running {category_name}...")
        print("-" * 30)
        
        if not os.path.exists(test_path):
            print(f"⚠️  No tests found in {test_path}")
            continue
        
        try:
            # Run tests with pytest
            result = pytest.main([
                test_path,
                "-v",
                "--tb=short",
                "--color=yes"
            ])
            
            if result == 0:
                print(f"✅ {category_name} passed!")
                total_passed += 1
            else:
                print(f"❌ {category_name} failed!")
                total_failed += 1
                
        except Exception as e:
            print(f"❌ Error running {category_name}: {e}")
            total_failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📈 Total: {total_passed + total_failed}")
    
    if total_failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test category(ies) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
