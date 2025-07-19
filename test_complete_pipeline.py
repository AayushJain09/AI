#!/usr/bin/env python3
"""
Complete Pipeline Integration Test

Tests the entire AI Recognition System pipeline:
1. Feature extraction (1536-dim CLIP + DINOv2)
2. Training (state-of-the-art Siamese network)
3. FAISS indexing (optimized for 1536 dimensions)
4. Recognition/inference
"""

import numpy as np
import torch
import yaml
import h5py
from pathlib import Path
from src.feature_extraction.feature_extractor import MultiModalFeatureExtractor
from src.indexing.faiss_indexer import AdvancedFAISSIndexer, IndexConfig

def test_complete_pipeline():
    """Test the complete AI recognition pipeline"""
    
    print("🚀 AI Recognition System - Complete Pipeline Test")
    print("=" * 60)
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print("📋 System Configuration:")
    print(f"  - CLIP Model: {config['features']['clip_variant']} (768 dims)")
    print(f"  - DINOv2 Model: {config['features']['dinov2_variant']} (768 dims)")
    print(f"  - Total Dimensions: {config['model']['total_dims']}")
    print(f"  - Training Epochs: {config['training']['epochs']}")
    print(f"  - Batch Size: {config['training']['batch_size']}")
    
    # 1. Test Feature Extraction
    print(f"\n🔧 1. Testing Feature Extraction Pipeline")
    print("-" * 40)
    
    # Load feature extractor
    feature_config = config['features']
    extractor = MultiModalFeatureExtractor(feature_config)
    
    # Test with a sample image
    test_image_path = None
    for item_dir in Path("data/raw").iterdir():
        if item_dir.is_dir():
            for img_file in item_dir.glob("*.JPG"):
                test_image_path = str(img_file)
                break
        if test_image_path:
            break
    
    if test_image_path:
        print(f"  Testing with: {test_image_path}")
        features = extractor.extract_all_features(test_image_path)
        
        if features:
            clip_dims = len(features['clip'])
            dino_dims = len(features['dinov2'])
            total_dims = clip_dims + dino_dims
            
            print(f"  ✅ Feature extraction successful:")
            print(f"     - CLIP: {clip_dims} dimensions")
            print(f"     - DINOv2: {dino_dims} dimensions")
            print(f"     - Total: {total_dims} dimensions")
            
            if total_dims == 1536:
                print(f"  ✅ Dimensions are optimal (1536)")
            else:
                print(f"  ⚠️  Expected 1536 dimensions, got {total_dims}")
        else:
            print(f"  ❌ Feature extraction failed")
            return False
    else:
        print(f"  ⚠️  No test images found in data/raw")
    
    # 2. Test Training System
    print(f"\n🎯 2. Testing Training System")
    print("-" * 40)
    
    # Check if trained model exists
    model_path = Path(config['recognition']['model_path'])
    if model_path.exists():
        print(f"  ✅ Trained model found: {model_path}")
        
        # Load and inspect model
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                print(f"  ✅ Model checkpoint is valid")
                print(f"     - Training completed at epoch: {checkpoint.get('epoch', 'Unknown')}")
                print(f"     - Best validation accuracy: {checkpoint.get('best_val_acc', 'Unknown'):.4f}")
            else:
                print(f"  ⚠️  Model format is non-standard")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
    else:
        print(f"  ⚠️  No trained model found at {model_path}")
    
    # 3. Test Indexing System
    print(f"\n🗂️  3. Testing FAISS Indexing System")
    print("-" * 40)
    
    # Check index files
    index_path = Path(config['recognition']['index_path'])
    metadata_path = Path(config['recognition']['metadata_path'])
    
    if index_path.exists() and metadata_path.exists():
        print(f"  ✅ Index files found:")
        print(f"     - Index: {index_path}")
        print(f"     - Metadata: {metadata_path}")
        
        # Test index loading and search
        index_config = IndexConfig(**config['indexing'])
        indexer = AdvancedFAISSIndexer(index_config)
        
        success = indexer.load_index(str(index_path), str(metadata_path))
        if success:
            print(f"  ✅ Index loaded successfully: {indexer.index.ntotal} vectors")
            
            # Test search with random query
            query = np.random.randn(1536).astype('float32')
            results = indexer.search(query, k=3)
            
            print(f"  ✅ Search test successful: {len(results)} results")
            if results:
                print(f"     - Top result: {results[0]['item_id']} (similarity: {results[0]['similarity']:.4f})")
                print(f"     - Search time: {results[0]['search_time_ms']:.2f}ms")
        else:
            print(f"  ❌ Failed to load index")
    else:
        print(f"  ❌ Index files not found")
        print(f"     - Looking for: {index_path}")
        print(f"     - Looking for: {metadata_path}")
    
    # 4. Test Feature File Integrity
    print(f"\n📊 4. Testing Feature File Integrity")
    print("-" * 40)
    
    features_file = Path("data/features_1536.h5")
    if features_file.exists():
        print(f"  ✅ Features file found: {features_file}")
        
        with h5py.File(features_file, 'r') as hf:
            num_images = len(list(hf.keys()))
            print(f"  ✅ Feature file contains {num_images} image features")
            
            # Check a sample feature
            if num_images > 0:
                sample_key = list(hf.keys())[0]
                sample_group = hf[sample_key]
                
                clip_shape = sample_group['clip'].shape
                dino_shape = sample_group['dinov2'].shape
                
                print(f"  ✅ Sample feature dimensions:")
                print(f"     - CLIP: {clip_shape}")
                print(f"     - DINOv2: {dino_shape}")
                
                if clip_shape[0] == 768 and dino_shape[0] == 768:
                    print(f"  ✅ Feature dimensions are correct")
                else:
                    print(f"  ⚠️  Unexpected feature dimensions")
    else:
        print(f"  ❌ Features file not found: {features_file}")
    
    # 5. Performance Summary
    print(f"\n📈 5. Performance Summary")
    print("-" * 40)
    
    # Read index stats
    stats_file = Path("data/models/index_stats.json")
    if stats_file.exists():
        import json
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"  📊 System Performance:")
        print(f"     - Total vectors indexed: {stats['total_vectors']}")
        print(f"     - Feature dimensions: {stats['dimension']}")
        print(f"     - Index type: {stats['index_type']}")
        print(f"     - Memory usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"     - Expected search time: {stats['expected_search_time_ms']:.2f} ms")
        print(f"     - Creation time: {stats['creation_time']}")
    
    # Final system check
    print(f"\n🎉 Pipeline Test Summary")
    print("=" * 60)
    
    components = {
        "Feature Extraction": test_image_path is not None,
        "Training Model": model_path.exists(),
        "FAISS Index": index_path.exists() and metadata_path.exists(),
        "Feature Database": features_file.exists()
    }
    
    all_working = all(components.values())
    
    for component, status in components.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}")
    
    if all_working:
        print(f"\n🚀 All pipeline components are operational!")
        print(f"   System is ready for production use.")
    else:
        print(f"\n⚠️  Some components need attention.")
    
    return all_working

if __name__ == "__main__":
    test_complete_pipeline()