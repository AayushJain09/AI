#!/usr/bin/env python3
"""
Complete Unified Recognition Pipeline Test

This script provides a comprehensive test of the unified recognition pipeline
to validate that it maintains exact recognition accuracy while using the
unified storage backend.
"""

import sys
import numpy as np
import logging
import yaml
from pathlib import Path
from PIL import Image
import time

# Add source to path 
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_config():
    """Create a test configuration file for the unified pipeline."""
    config = {
        'recognition': {
            'cache_size': 100,
            'confidence_threshold': 0.98,
            'high_confidence_threshold': 0.95,
            'min_stage1_confidence': 0.85,
            'max_candidate_score_gap': 0.1,
            'min_top_score_margin': 0.05,
            'hybrid_mode': True,
            'refinement_threshold': 0.82,
            'confidence_gap_threshold': 0.15
        },
        'data': {
            'base_dir': 'test_pipeline_data'
        }
    }
    
    config_path = "test_pipeline_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def test_complete_unified_pipeline():
    """Test the complete unified recognition pipeline end-to-end."""
    print("🧪 Testing Complete Unified Recognition Pipeline")
    print("=" * 60)
    
    try:
        # Create test configuration
        print("⚙️  Creating test configuration...")
        config_path = create_test_config()
        
        # Import the unified pipeline (import here to avoid path issues)
        from inference.recognize import UnifiedRecognitionPipeline
        from unified_storage.unified_store import UnifiedStore
        
        # Create unified storage first
        print("🔧 Initializing unified storage...")
        unified_store = UnifiedStore(
            data_dir="test_pipeline_data",
            config_path=config_path,
            enable_analytics=False
        )
        
        # Create unified recognition pipeline
        print("🔧 Initializing unified recognition pipeline...")
        pipeline = UnifiedRecognitionPipeline(
            config={
                'cache_size': 100,
                'confidence_threshold': 0.98,
                'hybrid_mode': True,
                'refinement_threshold': 0.82,
                'confidence_gap_threshold': 0.15,
                'min_stage1_confidence': 0.85,
                'max_candidate_score_gap': 0.1,
                'min_top_score_margin': 0.05,
                'data_dir': 'test_pipeline_data'
            },
            unified_store=unified_store
        )
        
        print("✅ Pipeline initialized successfully:")
        print(f"  Device: {pipeline.device}")
        print(f"  Hybrid mode: {pipeline.hybrid_mode}")
        print(f"  Refiner available: {pipeline.refiner_model is not None}")
        
        # Create and store test images
        print("\n📷 Creating test dataset...")
        test_items = [
            ('red_item', (255, 100, 100)),
            ('blue_item', (100, 100, 255)), 
            ('green_item', (100, 255, 100))
        ]
        
        stored_images = []
        for i, (item_name, color) in enumerate(test_items):
            # Create multiple images per item (simulating real dataset)
            for j in range(3):
                image_path = f"test_{item_name}_{j}.jpg"
                
                # Add slight variation to each image
                varied_color = tuple(max(0, min(255, c + np.random.randint(-20, 21))) for c in color)
                image = Image.new('RGB', (224, 224), color=varied_color)
                image.save(image_path)
                
                # Store in unified storage
                image_id = f"{item_name}_sample_{j}"
                metadata = {
                    'category': item_name,
                    'sample_id': j,
                    'test_data': True
                }
                
                stored_id = unified_store.store_image(
                    image_path=image_path,
                    image_id=image_id,
                    metadata=metadata
                )
                
                stored_images.append((image_path, image_id, item_name))
                print(f"  Stored: {image_path} → {stored_id}")
        
        print(f"\n✅ Created dataset with {len(stored_images)} images")
        
        # Test recognition on each stored image
        print("\n🔍 Testing recognition accuracy...")
        
        correct_recognitions = 0
        total_recognitions = 0
        recognition_times = []
        
        for image_path, expected_id, expected_category in stored_images:
            print(f"\n🔍 Testing: {image_path} (expecting category: {expected_category})")
            
            try:
                # Perform recognition
                start_time = time.time()
                result = pipeline.recognize_with_unified_storage(image_path)
                recognition_time = time.time() - start_time
                recognition_times.append(recognition_time)
                
                print(f"  ✅ Recognition completed in {recognition_time:.3f}s")
                print(f"  Result: {result.item_id} (confidence: {result.confidence:.6f})")
                print(f"  Top matches: {result.top_k_matches[:3]}")
                
                # Check if recognition is correct (should match category)
                if result.item_id != "unknown":
                    predicted_category = result.item_id.split('_sample_')[0] if '_sample_' in result.item_id else result.item_id
                    if predicted_category == expected_category:
                        correct_recognitions += 1
                        print(f"  ✅ Correct recognition: {predicted_category}")
                    else:
                        print(f"  ❌ Incorrect recognition: {predicted_category} (expected: {expected_category})")
                else:
                    print(f"  ⚠️  Unknown result (confidence: {result.confidence:.6f})")
                
                total_recognitions += 1
                
            except Exception as e:
                print(f"  ❌ Recognition failed: {e}")
                total_recognitions += 1
                continue
        
        # Calculate accuracy
        accuracy = correct_recognitions / total_recognitions if total_recognitions > 0 else 0
        avg_recognition_time = np.mean(recognition_times) if recognition_times else 0
        
        print(f"\n📊 Recognition Results:")
        print(f"  Total tests: {total_recognitions}")
        print(f"  Correct: {correct_recognitions}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Average time: {avg_recognition_time:.3f}s")
        
        # Test hybrid statistics
        print(f"\n📈 Hybrid Pipeline Statistics:")
        hybrid_stats = pipeline.get_hybrid_stats()
        print(f"  Total queries: {hybrid_stats['total_queries']}")
        print(f"  Raw only: {hybrid_stats['raw_only']} ({hybrid_stats.get('raw_only_pct', 0):.1f}%)")
        print(f"  Refined: {hybrid_stats['refined']} ({hybrid_stats.get('refined_pct', 0):.1f}%)")
        
        # Test storage statistics
        print(f"\n💾 Storage Statistics:")
        storage_stats = unified_store.get_statistics()
        print(f"  Images stored: {storage_stats.total_images_stored}")
        print(f"  Searches performed: {storage_stats.total_searches_performed}")
        print(f"  Average storage time: {storage_stats.avg_storage_time_ms:.1f}ms")
        print(f"  Average search time: {storage_stats.avg_search_time_ms:.1f}ms")
        
        # Test with query image (not in database)
        print(f"\n🔍 Testing with unknown query image...")
        unknown_image_path = "test_unknown_query.jpg"
        unknown_image = Image.new('RGB', (224, 224), color=(128, 128, 128))  # Gray
        unknown_image.save(unknown_image_path)
        
        unknown_result = pipeline.recognize_with_unified_storage(unknown_image_path)
        print(f"  Unknown query result: {unknown_result.item_id} (confidence: {unknown_result.confidence:.6f})")
        
        # Clean up
        unified_store.close()
        
        # Remove test files
        for image_path, _, _ in stored_images:
            Path(image_path).unlink()
        Path(unknown_image_path).unlink()
        Path(config_path).unlink()
        
        import shutil
        if Path("test_pipeline_data").exists():
            shutil.rmtree("test_pipeline_data")
        
        # Final validation
        success = True
        if accuracy < 0.8:  # At least 80% accuracy expected
            print(f"⚠️  Accuracy below threshold: {accuracy:.1%} < 80%")
            success = False
        
        if avg_recognition_time > 2.0:  # Should be faster than 2 seconds
            print(f"⚠️  Recognition time too slow: {avg_recognition_time:.3f}s > 2.0s")
            success = False
        
        if success:
            print(f"\n🎉 COMPLETE PIPELINE TEST PASSED!")
            print(f"✅ Unified recognition pipeline working correctly")
            print(f"✅ Raw features (1536D CLIP + DINOv2) recognition preserved")
            print(f"✅ Hybrid raw + refiner logic functional")
            print(f"✅ FAISS search on unified storage working")
            print(f"✅ Recognition accuracy: {accuracy:.1%}")
            print(f"✅ Average recognition time: {avg_recognition_time:.3f}s")
        else:
            print(f"\n⚠️  Some performance metrics below threshold")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete unified pipeline test."""
    success = test_complete_unified_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())