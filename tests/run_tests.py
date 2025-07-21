#!/usr/bin/env python3
"""
Simple Test Runner for AI Recognition System
Runs tests from project root for proper path resolution
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Run tests from project root"""
    # Get project root (parent of tests directory)
    project_root = Path(__file__).parent.parent
    print(f"🧪 AI Recognition System - Test Suite")
    print(f"📁 Project root: {project_root}")
    
    # Change to project root
    os.chdir(project_root)
    
    # Run the production test (most important)
    print(f"\n{'='*60}")
    print(f"🚀 Production Recognition Test")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            'python3', 'tests/system/test_recognition_final.py'
        ], check=True)
        print(f"✅ Production test PASSED")
        production_passed = True
    except subprocess.CalledProcessError as e:
        print(f"❌ Production test FAILED")
        production_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    if production_passed:
        print("✅ Production Recognition Test: PASS")
        print("\n🎉 Core system functionality verified!")
        print("✅ System ready for production use")
    else:
        print("❌ Production Recognition Test: FAIL") 
        print("\n⚠️  Core system requires attention")
        print("❌ Check system setup and configuration")
    
    print(f"{'='*60}")
    
    return 0 if production_passed else 1

if __name__ == "__main__":
    sys.exit(main())