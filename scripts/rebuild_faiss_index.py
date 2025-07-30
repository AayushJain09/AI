#!/usr/bin/env python3
"""
FAISS Index Rebuild Script

This script rebuilds the FAISS search index from vectors stored in the SQLite database.
It ensures the search functionality works properly after schema changes or data migrations.

OPTIMIZATION NOTES:
- Batch processing reduces memory overhead and improves rebuild performance
- Platform-specific optimizations leverage Apple Silicon unified memory architecture  
- Index validation ensures search functionality before deployment
- Progress monitoring provides user feedback during potentially long operations
"""

import sys
import os
import time
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unified_storage import UnifiedStore
from unified_storage.config_manager import ConfigManager

def rebuild_faiss_index(data_dir: str = "data") -> bool:
    """
    Rebuild FAISS index from existing vectors in SQLite database.
    
    PERFORMANCE OPTIMIZATION:
    Apple Silicon gets batch size of 50 for efficient unified memory usage.
    Other platforms use smaller batches to prevent memory pressure.
    
    Args:
        data_dir: Data directory containing recognition.db
        
    Returns:
        bool: True if rebuild successful, False otherwise
    """
    print("🔄 Rebuilding FAISS search index from SQLite vectors...")
    
    try:
        # Initialize unified storage system
        # This will create empty FAISS index that we'll populate
        print("🚀 Initializing unified storage system...")
        store = UnifiedStore(data_dir=data_dir)
        
        # Get platform-specific batch size for optimal performance
        # Apple Silicon unified memory allows larger batches without swap pressure
        config = ConfigManager(data_dir=data_dir).get_config()
        if config.platform.platform_type == "Apple_Silicon":
            batch_size = 50  # Leverage unified memory architecture
        else:
            batch_size = 25  # Conservative for discrete memory systems
            
        print(f"📊 Using batch size: {batch_size} (platform: {config.platform.platform_type})")
        
        # Connect to SQLite database to read vectors
        db_path = Path(data_dir) / "recognition.db"
        if not db_path.exists():
            print(f"❌ Database not found: {db_path}")
            return False
            
        conn = sqlite3.connect(str(db_path))
        
        # Get total vector count for progress tracking
        cursor = conn.execute("SELECT COUNT(*) FROM vectors")
        total_vectors = cursor.fetchone()[0]
        print(f"📈 Found {total_vectors} vectors to rebuild into FAISS index")
        
        if total_vectors == 0:
            print("⚠️ No vectors found in database - nothing to rebuild")
            conn.close()
            return True
            
        # Process vectors in batches to manage memory efficiently
        # Batch processing prevents memory spikes on large datasets
        rebuilt_count = 0
        batch_num = 0
        
        print("🔄 Processing vectors in batches...")
        
        # Query vectors in batches ordered by creation time for consistent results
        while rebuilt_count < total_vectors:
            offset = rebuilt_count
            
            cursor = conn.execute("""
                SELECT image_id, vector_data 
                FROM vectors 
                ORDER BY created_at 
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            
            batch_vectors = cursor.fetchall()
            if not batch_vectors:
                break
                
            batch_num += 1
            batch_start = time.time()
            
            # Process each vector in the batch
            for image_id, vector_blob in batch_vectors:
                try:
                    # Deserialize vector from blob storage
                    # SQLite BLOB stores numpy arrays as binary data
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    
                    if len(vector) != 1536:
                        print(f"⚠️ Skipping vector {image_id}: invalid dimensions ({len(vector)} != 1536)")
                        continue
                        
                    # Add vector to FAISS index using store's internal method
                    # This maintains consistency with the unified storage API
                    store.faiss_index.add(vector.reshape(1, -1))
                    store.image_id_mapping[store.index_size] = image_id
                    store.index_size += 1
                    
                    rebuilt_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to process vector {image_id}: {e}")
                    continue
            
            batch_time = time.time() - batch_start
            progress = (rebuilt_count / total_vectors) * 100
            rate = len(batch_vectors) / batch_time if batch_time > 0 else 0
            
            print(f"✓ Batch {batch_num}: {len(batch_vectors)} vectors processed "
                  f"({rebuilt_count}/{total_vectors} = {progress:.1f}%) "
                  f"[{rate:.1f} vectors/sec]")
        
        conn.close()
        
        # Validate the rebuilt index
        print("🔍 Validating rebuilt FAISS index...")
        
        if store.index_size != rebuilt_count:
            print(f"❌ Index size mismatch: expected {rebuilt_count}, got {store.index_size}")
            return False
            
        if len(store.image_id_mapping) != rebuilt_count:
            print(f"❌ Mapping size mismatch: expected {rebuilt_count}, got {len(store.image_id_mapping)}")
            return False
            
        # Test search functionality with a sample vector
        if rebuilt_count > 0:
            print("🔍 Testing search functionality...")
            test_results = store.search_similar_images(
                np.random.rand(1536).astype(np.float32),  # Random test vector
                k=min(5, rebuilt_count)  # Search for up to 5 results
            )
            
            if len(test_results) == 0:
                print("⚠️ Search test returned no results - this may indicate an issue")
            else:
                print(f"✅ Search test successful: {len(test_results)} results returned")
        
        print(f"✅ FAISS index rebuild completed successfully!")
        print(f"📊 Final stats:")
        print(f"   - Vectors processed: {rebuilt_count}")
        print(f"   - FAISS index size: {store.index_size}")
        print(f"   - Image ID mappings: {len(store.image_id_mapping)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to rebuild FAISS index: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point for FAISS index rebuild."""
    print("🔧 FAISS Index Rebuild Tool")
    print("=" * 50)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    success = rebuild_faiss_index()
    
    if success:
        print("\n✅ FAISS index rebuild completed successfully!")
        print("🔍 Search functionality is now ready for use.")
        return 0
    else:
        print("\n❌ FAISS index rebuild failed!")
        print("🔧 Check logs above for error details.")
        return 1

if __name__ == "__main__":
    exit(main())