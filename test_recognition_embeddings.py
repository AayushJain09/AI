#!/usr/bin/env python3
"""
Test script for the optimized recognition system with embedding-based indexing
"""

import sys
import yaml
import time
import torch
import numpy as np
from pathlib import Path
from src.inference.recognize import RecognitionPipeline, PerformanceMonitor

def test_recognition_system():
    """Test the complete recognition system with embedding-based indexing"""
    
    print("🚀 Testing State-of-the-Art Recognition System (Embedding-Based)")
    print("=" * 70)
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Create manual config for embedding-based recognition
    config = {
        'model_path': 'checkpoints/best_model.pth',
        'index_path': 'data/models/faiss_index_corrected.bin',
        'metadata_path': 'data/models/index_metadata_corrected.pkl',
        'threshold': 0.85,
        'batch_size': 32,
        'cache_size': 1000,
        'confidence_threshold': 0.85,
        'high_confidence_threshold': 0.95,
        'batch_confidence_threshold': 0.9,
        'clip_model': full_config.get('model', {}).get('clip_variant', 'ViT-L/14'),
        'mobile_mode': full_config.get('features', {}).get('mobile_mode', False),
        'embedding_dim': full_config.get('model', {}).get('embedding_dim', 512)
    }
    
    # Create recognition pipeline
    pipeline = RecognitionPipeline(config)
    
    # Load the pipeline
    try:
        print("📋 Loading recognition pipeline...")
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
        n_tests = 5
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
        
        # Test different images from the dataset
        print(f"\n🔄 Multi-Image Test:")
        test_images = []
        for item_dir in Path("data/raw").iterdir():
            if item_dir.is_dir():
                for img_file in list(item_dir.glob("*.JPG"))[:2]:  # Max 2 per item
                    test_images.append(str(img_file))
                if len(test_images) >= 10:  # Test with 10 different images
                    break
        
        if len(test_images) > 1:
            print(f"Testing {len(test_images)} different images...")
            
            successful = 0
            confidences = []
            times = []
            
            for img_path in test_images:
                start = time.time()
                result = pipeline.recognize(img_path)
                elapsed = time.time() - start
                
                times.append(elapsed)
                confidences.append(result.confidence)
                
                if result.item_id != "unknown":
                    successful += 1
                
                print(f"  {Path(img_path).name}: {result.item_id} (conf: {result.confidence:.3f}, time: {elapsed:.3f}s)")
            
            avg_confidence = sum(confidences) / len(confidences)
            avg_time = sum(times) / len(times)
            
            print(f"\n📊 Multi-Image Summary:")
            print(f"  - Success rate: {successful}/{len(test_images)} ({successful/len(test_images)*100:.1f}%)")
            print(f"  - Average confidence: {avg_confidence:.3f}")
            print(f"  - Average time: {avg_time:.3f}s")
        
        print(f"\n✅ Recognition system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Recognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_recognition_system()
    print(f"\n{'🎉 Test passed!' if success else '❌ Test failed!'}")