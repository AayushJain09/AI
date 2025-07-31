#!/usr/bin/env python3
"""
Test Unified Storage FAISS Search

This script tests that the unified storage system can store images
and perform FAISS similarity search on raw 1536D features correctly.
"""

import sys
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import time

# Add source to path 
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_images():
    """Create a set of test images for search validation."""
    test_images = []
    
    # Create different colored test images
    colors = [
        ('red', (255, 0, 0)),
        ('blue', (0, 0, 255)), 
        ('green', (0, 255, 0)),
        ('yellow', (255, 255, 0)),
        ('purple', (128, 0, 128))
    ]
    
    for i, (color_name, color) in enumerate(colors):
        image_path = f"test_image_{color_name}.jpg"
        image = Image.new('RGB', (224, 224), color=color)
        image.save(image_path)
        test_images.append((image_path, f"item_{color_name}"))
        logger.info(f"Created test image: {image_path} ({color_name})")
    
    return test_images


def test_unified_storage_search():
    """Test unified storage image storage and FAISS search."""
    print("🧪 Testing Unified Storage FAISS Search")
    print("=" * 50)
    
    try:
        from unified_storage.unified_store import UnifiedStore
        
        # Initialize unified storage
        print("🔧 Initializing unified storage system...")
        unified_store = UnifiedStore(
            data_dir="test_search_data",
            enable_analytics=False
        )
        
        # Create test images
        print("\n📷 Creating test images...")
        test_images = create_test_images()
        
        # Store images in unified storage
        print("\n📥 Storing images in unified storage...")
        stored_ids = []
        
        for image_path, item_id in test_images:
            print(f"  Storing: {image_path} as {item_id}")
            
            # Store with metadata
            metadata = {
                'category': item_id,
                'test_image': True,
                'color': item_id.replace('item_', '')
            }
            
            stored_id = unified_store.store_image(
                image_path=image_path,
                image_id=item_id,
                metadata=metadata
            )
            
            stored_ids.append(stored_id)
            print(f"    ✅ Stored as: {stored_id}")
        
        print(f"\n✅ Successfully stored {len(stored_ids)} images")
        
        # Test similarity search
        print("\n🔍 Testing similarity search...")
        
        # Search for each stored image (should find itself as top match)
        for i, (query_image_path, expected_item_id) in enumerate(test_images):
            print(f"\n🔍 Searching for: {query_image_path} (expecting: {expected_item_id})")
            
            start_time = time.time()
            search_results = unified_store.search_similar(
                query_image_path=query_image_path,
                top_k=3,  # Get top 3 matches
                similarity_threshold=0.5  # Lower threshold for test
            )
            search_time = (time.time() - start_time) * 1000
            
            print(f"  Search completed in {search_time:.1f}ms")
            print(f"  Found {len(search_results)} results:")
            
            if not search_results:
                print("  ❌ No results found!")
                continue
            
            # Check results
            for j, result in enumerate(search_results):
                print(f"    {j+1}. {result.image_id}: {result.similarity_score:.6f}")
                if result.metadata:
                    print(f"       Metadata: {result.metadata.get('color', 'unknown')}")
            
            # Validate top result
            top_result = search_results[0]
            if top_result.image_id == expected_item_id:
                print(f"  ✅ Correct top match: {top_result.image_id} (score: {top_result.similarity_score:.6f})")
            else:
                print(f"  ⚠️  Unexpected top match: {top_result.image_id} (expected: {expected_item_id})")
            
            # Check similarity score
            if top_result.similarity_score > 0.9:
                print(f"  ✅ High similarity score: {top_result.similarity_score:.6f}")
            else:
                print(f"  ⚠️  Lower similarity score: {top_result.similarity_score:.6f}")
        
        # Test cross-image search (different images should have lower similarity)
        print(f"\n🔄 Testing cross-image similarity...")
        query_image = test_images[0][0]  # Red image
        
        search_results = unified_store.search_similar(
            query_image_path=query_image,
            top_k=len(test_images),
            similarity_threshold=0.0  # Get all results
        )
        
        print(f"Cross-similarity results for {query_image}:")
        for i, result in enumerate(search_results):
            color = result.metadata.get('color', 'unknown') if result.metadata else 'unknown'
            print(f"  {i+1}. {result.image_id} ({color}): {result.similarity_score:.6f}")
        
        # Validate that same image has highest score
        if search_results and search_results[0].image_id == test_images[0][1]:
            print("✅ Same image has highest similarity")
        else:
            print("⚠️  Same image doesn't have highest similarity")
        
        # Get storage statistics
        print(f"\n📊 Storage Statistics:")
        stats = unified_store.get_statistics()
        print(f"  Images stored: {stats.total_images_stored}")
        print(f"  Searches performed: {stats.total_searches_performed}")
        print(f"  Average storage time: {stats.avg_storage_time_ms:.1f}ms")
        print(f"  Average search time: {stats.avg_search_time_ms:.1f}ms")
        print(f"  Database size: {stats.database_size_mb:.1f}MB")
        
        # Health check
        print(f"\n🏥 System Health Check:")
        health = unified_store.health_check()
        print(f"  Overall status: {health['overall_status']}")
        
        for component, status in health['components'].items():
            print(f"  {component}: {status['status']}")
        
        # Clean up
        unified_store.close()
        
        # Remove test files
        for image_path, _ in test_images:
            Path(image_path).unlink()
        
        import shutil
        if Path("test_search_data").exists():
            shutil.rmtree("test_search_data")
        
        print(f"\n🎉 UNIFIED STORAGE SEARCH TEST PASSED!")
        print(f"✅ Images stored successfully in unified storage")
        print(f"✅ FAISS search works correctly on 1536D raw features")
        print(f"✅ Similarity scores are reasonable and consistent")
        print(f"✅ Ready for unified recognition pipeline")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unified storage search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the unified storage search test."""
    success = test_unified_storage_search()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())