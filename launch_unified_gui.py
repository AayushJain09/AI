#!/usr/bin/env python3
"""
Unified GUI Launcher

Launch the modern GUI interface for the AI recognition system.
This provides real-time item addition with complete preprocessing pipeline.
"""

import sys
import os
from pathlib import Path

# Fix OpenMP library conflict before importing torch-based modules
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Main launcher function"""
    try:
        print("🚀 Starting AI Recognition System - Unified GUI...")
        print("Loading unified preprocessing components...")
        
        from unified_storage.gui.unified_gui import UnifiedGUI
        
        print("✅ Components loaded successfully!")
        print("🎯 Launching GUI interface...")
        
        # Create and run GUI
        app = UnifiedGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  - tkinter (usually comes with Python)")
        print("  - PIL/Pillow")
        print("  - numpy")
        print("  - torch")
        print("  - faiss")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()