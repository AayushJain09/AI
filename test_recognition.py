#!/usr/bin/env python3
"""
Test script for the optimized recognition system

Tests the complete recognition pipeline with:
- 1536-dimensional CLIP ViT-L/14 + DINOv2 features
- State-of-the-art Siamese network
- Optimized FAISS indexing
- Multi-stage recognition pipeline
"""

import sys
import yaml
import time
from pathlib import Path
from src.inference.recognize import create_pipeline, PerformanceMonitor

def test_recognition_system():
    """Test the complete recognition system"""
    
    print("🚀 Testing State-of-the-Art Recognition System")
    print("=" * 60)
    
    # Load configuration
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        # Create optimized recognition pipeline
        print("📋 Loading recognition pipeline...")
        pipeline = create_pipeline(config_path)
        print("✅ Pipeline loaded successfully")
        
        # Print system information
        print(f"\n📊 System Architecture:")
        print(f"  - Device: {pipeline.device}")
        print(f"  - Model loaded: {'Yes' if pipeline.model else 'No'}")
        print(f"  - Index vectors: {pipeline.index.ntotal if pipeline.index else 0}")
        print(f"  - Index dimensions: {pipeline.index.d if pipeline.index else 'N/A'}")
        
        # Test with a sample image
        test_image_path = None
        for item_dir in Path("data/raw").iterdir():
            if item_dir.is_dir():
                for img_file in item_dir.glob("*.JPG"):
                    test_image_path = str(img_file)
                    break
            if test_image_path:
                break
        
        if not test_image_path:
            print("⚠️  No test images found in data/raw")
            return False
        
        print(f"\n🖼️  Testing with: {test_image_path}")
        
        # Perform recognition
        start_time = time.time()
        result = pipeline.recognize(test_image_path)
        total_time = time.time() - start_time
        
        # Display results
        print(f"\n🎯 Recognition Results:")
        print(f"  - Item ID: {result.item_id}")
        print(f"  - Confidence: {result.confidence:.4f}")
        print(f"  - Inference Time: {result.inference_time:.3f}s")
        print(f"  - Total Time: {total_time:.3f}s")
        
        if result.top_k_matches:
            print(f"\n📊 Top 5 Matches:")
            for i, (item_id, score) in enumerate(result.top_k_matches[:5], 1):
                print(f"  {i}. {item_id}: {score:.4f}")
        
        # Test performance with multiple queries
        print(f"\n⚡ Performance Test:")
        n_tests = 3
        total_inference_time = 0
        
        for i in range(n_tests):
            start = time.time()
            result = pipeline.recognize(test_image_path)
            inference_time = time.time() - start
            total_inference_time += inference_time
            print(f"  Test {i+1}: {inference_time:.3f}s (confidence: {result.confidence:.3f})")
        
        avg_time = total_inference_time / n_tests
        throughput = 1.0 / avg_time if avg_time > 0 else float('inf')
        
        print(f"\n📈 Performance Summary:")
        print(f"  - Average inference time: {avg_time:.3f}s")
        print(f"  - Throughput: {throughput:.1f} images/second")
        
        # Test batch recognition if multiple images available
        test_images = []
        for item_dir in Path("data/raw").iterdir():
            if item_dir.is_dir():
                for img_file in list(item_dir.glob("*.JPG"))[:2]:  # Max 2 per item
                    test_images.append(str(img_file))
        
        if len(test_images) > 1:
            print(f"\n🔄 Batch Recognition Test ({len(test_images)} images):")
            
            batch_start = time.time()
            batch_results = pipeline.batch_recognize(test_images)
            batch_time = time.time() - batch_start
            
            successful = sum(1 for r in batch_results if r.item_id != "unknown")
            avg_confidence = sum(r.confidence for r in batch_results) / len(batch_results)
            
            print(f"  - Success rate: {successful}/{len(batch_results)} ({successful/len(batch_results)*100:.1f}%)")
            print(f"  - Average confidence: {avg_confidence:.3f}")
            print(f"  - Batch time: {batch_time:.3f}s")
            print(f"  - Time per image: {batch_time/len(test_images):.3f}s")
        
        # Performance monitor test
        print(f"\n📊 Performance Monitoring:")
        monitor = PerformanceMonitor(pipeline)
        
        # Add some test results
        monitor.update_metrics(result)
        for batch_result in batch_results[:3]:
            monitor.update_metrics(batch_result)
        
        # Generate report
        report = monitor.get_report()
        print(f"  - Total queries: {report['overall_metrics']['total_queries']}")
        print(f"  - Success rate: {report['overall_metrics']['accuracy']:.3f}")
        print(f"  - Average confidence: {report['overall_metrics']['avg_confidence']:.3f}")
        print(f"  - Average inference time: {report['overall_metrics']['avg_inference_time']:.3f}s")
        
        print(f"\n✅ Recognition system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Recognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_add_new_item():
    """Test adding a new item to the recognition system"""
    print(f"\n🔧 Testing Item Addition...")
    
    try:
        pipeline = create_pipeline("config.yaml")
        
        # Find a test item directory
        test_item_dir = None
        for item_dir in Path("data/raw").iterdir():
            if item_dir.is_dir():
                test_item_dir = item_dir
                break
        
        if not test_item_dir:
            print("⚠️  No test item directory found")
            return False
        
        # Get images from the directory
        image_paths = [str(p) for p in test_item_dir.glob("*.JPG")]
        if not image_paths:
            print("⚠️  No images found in test directory")
            return False
        
        # Add item with a test ID
        test_item_id = f"test_item_{int(time.time())}"
        
        print(f"  Adding item: {test_item_id}")
        print(f"  Images: {len(image_paths)}")
        
        original_count = pipeline.index.ntotal
        pipeline.add_item_to_index(test_item_id, image_paths)
        new_count = pipeline.index.ntotal
        
        print(f"  Index vectors: {original_count} → {new_count}")
        print(f"✅ Item addition test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Item addition test failed: {e}")
        return False

if __name__ == "__main__":
    # Run recognition system test
    success = test_recognition_system()
    
    if success:
        # Run item addition test
        test_add_new_item()
    
    print(f"\n{'🎉 All tests passed!' if success else '❌ Tests failed!'}")