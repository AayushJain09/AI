#!/usr/bin/env python3
"""
Production-Ready Recognition Test Script
Tests the complete AI Recognition System with all items
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from inference.recognize import create_pipeline

def test_recognition_system():
    """Test the complete recognition system"""
    print("🚀 AI Recognition System - Production Test")
    print("=" * 50)
    
    # Initialize pipeline
    print("📊 Loading recognition pipeline...")
    start_time = time.time()
    # Change to relative path from project root
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    pipeline = create_pipeline(str(config_path))
    load_time = time.time() - start_time
    print(f"✅ Pipeline loaded in {load_time:.2f}s")
    
    # System stats
    print(f"📈 Index: {pipeline.index.ntotal} vectors, {pipeline.index.d} dimensions")
    
    # Test all known items (with absolute paths from project root)
    project_root = Path(__file__).parent.parent.parent
    test_items = {
        'item_001': str(project_root / 'data/raw/item_001/Copy of IMG_8388.JPG'),
        'item_002': str(project_root / 'data/raw/item_002/Copy of IMG_8403.JPG'),
        'item_003': str(project_root / 'data/raw/item_003/Copy of IMG_8428.JPG'),
        'item_004': str(project_root / 'data/raw/item_004/1752950980813_IMG_8416.JPG')
    }
    
    print("\n🔍 Testing Known Items:")
    print("-" * 30)
    all_correct = True
    total_time = 0
    
    for expected_item, image_path in test_items.items():
        start = time.time()
        result = pipeline.recognize(image_path)
        end = time.time()
        total_time += (end - start)
        
        if result.item_id == expected_item:
            print(f"✅ {expected_item}: {result.confidence:.3f} ({end-start:.3f}s)")
        else:
            print(f"❌ {expected_item}: got {result.item_id} ({result.confidence:.3f})")
            all_correct = False
    
    # Test unknown items
    print("\n🔍 Testing Unknown Items:")
    print("-" * 30)
    unknown_tests = [
        str(project_root / 'environments/env2/lib/python3.13/site-packages/networkx/drawing/tests/baseline/test_display_empty_graph.png')
    ]
    
    for unknown_image in unknown_tests:
        try:
            start = time.time()
            result = pipeline.recognize(unknown_image)
            end = time.time()
            total_time += (end - start)
            
            if result.item_id == "unknown":
                print(f"✅ Unknown item correctly rejected ({end-start:.3f}s)")
            else:
                print(f"❌ Unknown item incorrectly matched to {result.item_id}")
                all_correct = False
        except Exception as e:
            print(f"⚠️  Test failed: {e}")
    
    # Performance summary
    avg_time = total_time / (len(test_items) + len(unknown_tests))
    print(f"\n📊 Performance Summary:")
    print(f"   Average recognition time: {avg_time:.3f}s")
    print(f"   Total test time: {total_time:.3f}s")
    print(f"   System accuracy: {'PASS' if all_correct else 'FAIL'}")
    
    return all_correct

if __name__ == "__main__":
    success = test_recognition_system()
    sys.exit(0 if success else 1)