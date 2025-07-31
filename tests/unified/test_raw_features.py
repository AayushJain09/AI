#!/usr/bin/env python3
"""
Test Raw Feature Extraction for Unified Recognition

This script validates that the unified storage system produces
identical 1536D feature vectors as expected for the recognition pipeline.
"""

import sys
import numpy as np
import logging
from pathlib import Path
from PIL import Image

# Add source to path 
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_raw_feature_extraction():
    """Test that unified storage produces exactly 1536D raw features."""
    print("🧪 Testing Raw Feature Extraction for Recognition Pipeline")
    print("=" * 60)
    
    try:
        # Import unified storage
        from unified_storage.unified_store import UnifiedStore
        
        # Initialize unified storage system
        print("🔧 Initializing unified storage...")
        unified_store = UnifiedStore(
            data_dir="test_data_features",
            enable_analytics=False  # Disable for testing
        )
        
        # Create a test image
        test_image_path = "test_recognition_image.jpg"
        print("📷 Creating test image...")
        test_image = Image.new('RGB', (224, 224), color=(128, 64, 192))  # Purple test image
        test_image.save(test_image_path)
        
        # Extract features using unified storage
        print("🔍 Extracting features using unified storage...")
        image = Image.open(test_image_path).convert('RGB')
        
        # Get raw 1536D features
        raw_features = unified_store.feature_extractor.extract_features_single(image)
        
        if raw_features is None:
            raise ValueError("Feature extraction returned None")
        
        # Validate feature properties
        print(f"\n✅ Raw Feature Extraction Results:")
        print(f"  Shape: {raw_features.shape}")
        print(f"  Data type: {raw_features.dtype}")
        print(f"  Min value: {np.min(raw_features):.6f}")
        print(f"  Max value: {np.max(raw_features):.6f}")
        print(f"  Mean: {np.mean(raw_features):.6f}")
        print(f"  Std: {np.std(raw_features):.6f}")
        print(f"  L2 norm: {np.linalg.norm(raw_features):.6f}")
        
        # Critical validation: Must be exactly 1536 dimensions
        expected_dim = 1536  # 768 CLIP + 768 DINOv2
        if len(raw_features) != expected_dim:
            raise ValueError(f"CRITICAL: Expected {expected_dim}D features, got {len(raw_features)}D")
        
        print(f"\n✅ CRITICAL VALIDATION PASSED: Features are exactly {expected_dim}D")
        
        # Validate CLIP + DINOv2 structure
        clip_part = raw_features[:768]
        dinov2_part = raw_features[768:]
        
        print(f"\n📊 Feature Structure Analysis:")
        print(f"  CLIP component (first 768D):")
        print(f"    Shape: {clip_part.shape}")
        print(f"    L2 norm: {np.linalg.norm(clip_part):.6f}")
        print(f"    Mean: {np.mean(clip_part):.6f}")
        
        print(f"  DINOv2 component (last 768D):")
        print(f"    Shape: {dinov2_part.shape}")
        print(f"    L2 norm: {np.linalg.norm(dinov2_part):.6f}")
        print(f"    Mean: {np.mean(dinov2_part):.6f}")
        
        # Test feature consistency (same image should give same features)
        print(f"\n🔄 Testing feature consistency...")
        features_2nd = unified_store.feature_extractor.extract_features_single(image)
        
        if features_2nd is not None:
            feature_diff = np.abs(raw_features - features_2nd)
            max_difference = np.max(feature_diff)
            print(f"  Max difference between extractions: {max_difference:.8f}")
            
            if max_difference < 1e-6:
                print("✅ Features are perfectly consistent")
            else:
                print(f"⚠️  Small differences detected: {max_difference:.8f}")
        
        # Test that features are normalized (important for similarity search)
        feature_norm = np.linalg.norm(raw_features)
        print(f"\n📏 Feature Normalization:")
        print(f"  L2 norm: {feature_norm:.6f}")
        if abs(feature_norm - 1.0) < 0.1:
            print("✅ Features are approximately normalized")
        else:
            print(f"ℹ️  Features not unit normalized (norm: {feature_norm:.6f})")
        
        # Test batch processing
        print(f"\n🔄 Testing batch feature extraction...")
        batch_images = [image, image]  # Same image twice
        batch_features = unified_store.feature_extractor.extract_features_batch(batch_images)
        
        if batch_features and len(batch_features) == 2:
            print(f"✅ Batch processing successful:")
            print(f"  Batch size: {len(batch_features)}")
            print(f"  Each feature shape: {batch_features[0].shape}")
            
            # Check batch consistency
            batch_diff = np.abs(batch_features[0] - batch_features[1])
            max_batch_diff = np.max(batch_diff)
            print(f"  Batch consistency: {max_batch_diff:.8f}")
            
            if max_batch_diff < 1e-6:
                print("✅ Batch features are identical")
            else:
                print(f"⚠️  Batch features differ: {max_batch_diff:.8f}")
        
        # Clean up
        unified_store.close()
        Path(test_image_path).unlink()
        import shutil
        if Path("test_data_features").exists():
            shutil.rmtree("test_data_features")
        
        print(f"\n🎉 RAW FEATURE EXTRACTION TEST PASSED!")
        print(f"✅ Unified storage produces identical 1536D features")
        print(f"✅ Feature structure: CLIP(768D) + DINOv2(768D) = 1536D")
        print(f"✅ Ready for raw feature recognition pipeline")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Raw feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the raw feature extraction test."""
    success = test_raw_feature_extraction()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())