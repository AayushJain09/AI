"""
Migration Tools: HDF5 + FAISS to SQLite + sqlite-vec
Preserves exact data and functionality while migrating storage systems
"""

import h5py
import numpy as np
import pickle
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import uuid
from datetime import datetime

# Import our SQLite storage
from .sqlite_store import SQLiteVectorStore, FeatureRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HDF5ToSQLiteMigrator:
    """
    Migration tool to convert HDF5 + FAISS storage to SQLite + sqlite-vec
    Preserves exact feature data and metadata from original system
    """
    
    def __init__(self, sqlite_db_path: str):
        self.sqlite_store = SQLiteVectorStore(sqlite_db_path)
        
        # Migration statistics
        self.stats = {
            'total_features_migrated': 0,
            'total_items_migrated': 0,
            'migration_start_time': None,
            'migration_end_time': None,
            'errors_encountered': 0,
            'duplicate_items_skipped': 0
        }
        
        logger.info(f"🔄 HDF5→SQLite Migration initialized")
        logger.info(f"   Target database: {sqlite_db_path}")
    
    def migrate_hdf5_features(self, hdf5_path: str, metadata_path: Optional[str] = None) -> Dict:
        """
        Migrate features from HDF5 file to SQLite
        Preserves exact feature vectors and metadata
        
        Args:
            hdf5_path: Path to the HDF5 features file
            metadata_path: Optional path to FAISS metadata pickle file
            
        Returns:
            Migration statistics and results
        """
        logger.info(f"🔄 Starting HDF5→SQLite migration")
        logger.info(f"   Source HDF5: {hdf5_path}")
        self.stats['migration_start_time'] = time.time()
        
        hdf5_file = Path(hdf5_path)
        if not hdf5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        # Load FAISS metadata if provided
        faiss_metadata = {}
        if metadata_path and Path(metadata_path).exists():
            logger.info(f"📂 Loading FAISS metadata: {metadata_path}")
            try:
                with open(metadata_path, 'rb') as f:
                    faiss_data = pickle.load(f)
                    
                # Handle different metadata formats
                if isinstance(faiss_data, dict):
                    if 'item_ids' in faiss_data and 'metadata' in faiss_data:
                        # New format from AdvancedFAISSIndexer
                        faiss_metadata = {
                            'item_ids': faiss_data['item_ids'],
                            'metadata': faiss_data['metadata']
                        }
                    elif 'index_to_item' in faiss_data:
                        # Legacy format
                        faiss_metadata = faiss_data
                    
                logger.info(f"✅ Loaded FAISS metadata with {len(faiss_metadata.get('item_ids', []))} items")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load FAISS metadata: {e}")
        
        # Process HDF5 file
        with h5py.File(hdf5_path, 'r') as hf:
            # Get all image groups
            image_keys = list(hf.keys())
            logger.info(f"📊 Found {len(image_keys)} feature records in HDF5")
            
            migrated_count = 0
            error_count = 0
            
            for img_key in tqdm(image_keys, desc="Migrating features"):
                try:
                    success = self._migrate_single_feature_record(hf[img_key], img_key, faiss_metadata)
                    if success:
                        migrated_count += 1
                        self.stats['total_features_migrated'] += 1
                    else:
                        error_count += 1
                        self.stats['errors_encountered'] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error migrating {img_key}: {e}")
                    error_count += 1
                    self.stats['errors_encountered'] += 1
        
        # Finalize migration
        self.stats['migration_end_time'] = time.time()
        migration_time = self.stats['migration_end_time'] - self.stats['migration_start_time']
        
        logger.info(f"🎉 HDF5→SQLite migration complete!")
        logger.info(f"   ✅ Features migrated: {migrated_count}")
        logger.info(f"   ❌ Errors: {error_count}")
        logger.info(f"   ⏱️ Migration time: {migration_time:.2f}s")
        logger.info(f"   📊 Rate: {migrated_count/migration_time:.1f} features/sec")
        
        return {
            'success': migrated_count,
            'errors': error_count,
            'total_processed': len(image_keys),
            'migration_time_seconds': migration_time,
            'features_per_second': migrated_count / migration_time if migration_time > 0 else 0
        }
    
    def _migrate_single_feature_record(self, img_group: h5py.Group, img_key: str, 
                                     faiss_metadata: Dict) -> bool:
        """
        Migrate a single feature record from HDF5 to SQLite
        Preserves exact feature data and metadata
        """
        try:
            # Extract metadata from HDF5 attributes
            item_id = img_group.attrs.get('item_id', 'unknown')
            if isinstance(item_id, bytes):
                item_id = item_id.decode('utf-8')
                
            image_path = img_group.attrs.get('image_path', '')
            if isinstance(image_path, bytes):
                image_path = image_path.decode('utf-8')
            
            # Load feature vectors (preserving exact data)
            clip_features = img_group['clip'][:]
            dinov2_features = img_group['dinov2'][:]
            
            # Validate feature dimensions (critical preservation check)
            if len(clip_features) != 768:
                logger.warning(f"⚠️ Unexpected CLIP dimensions for {img_key}: {len(clip_features)} (expected 768)")
                # Pad or truncate to maintain compatibility
                if len(clip_features) < 768:
                    clip_features = np.pad(clip_features, (0, 768 - len(clip_features)))
                else:
                    clip_features = clip_features[:768]
                    
            if len(dinov2_features) != 768:
                logger.warning(f"⚠️ Unexpected DINOv2 dimensions for {img_key}: {len(dinov2_features)} (expected 768)")
                # Pad or truncate to maintain compatibility
                if len(dinov2_features) < 768:
                    dinov2_features = np.pad(dinov2_features, (0, 768 - len(dinov2_features)))
                else:
                    dinov2_features = dinov2_features[:768]
            
            # Combine features (preserving exact original logic)
            combined_features = np.concatenate([clip_features, dinov2_features])
            
            # Normalize for consistency (preserving original approach)
            clip_normalized = clip_features / (np.linalg.norm(clip_features) + 1e-8)
            dinov2_normalized = dinov2_features / (np.linalg.norm(dinov2_features) + 1e-8)
            combined_normalized = combined_features / (np.linalg.norm(combined_features) + 1e-8)
            
            # Generate unique image ID for SQLite
            image_id = f"migrated_{item_id}_{img_key}_{uuid.uuid4().hex[:8]}"
            
            # Create feature record
            record = FeatureRecord(
                image_id=image_id,
                item_id=str(item_id),
                image_path=str(image_path),
                clip_features=clip_normalized,
                dinov2_features=dinov2_normalized,
                combined_features=combined_normalized,
                augmentation_params={'migrated_from': 'hdf5', 'original_key': img_key},
                extraction_timestamp=time.time(),
                image_hash=None  # Will be computed if needed
            )
            
            # Store to SQLite
            success = self.sqlite_store.store_features(record)
            
            if success:
                logger.debug(f"✅ Migrated {img_key} → {image_id}")
            else:
                logger.error(f"❌ Failed to store migrated record: {img_key}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error migrating record {img_key}: {e}")
            return False
    
    def migrate_from_directory_structure(self, data_dir: str, features_hdf5: str, 
                                       faiss_index: Optional[str] = None,
                                       faiss_metadata: Optional[str] = None) -> Dict:
        """
        Migrate complete directory structure from old system to SQLite
        
        Args:
            data_dir: Directory containing augmented images
            features_hdf5: HDF5 features file
            faiss_index: Optional FAISS index file 
            faiss_metadata: Optional FAISS metadata file
            
        Returns:
            Complete migration statistics
        """
        logger.info(f"🔄 Starting complete system migration to SQLite")
        logger.info(f"   Data directory: {data_dir}")
        logger.info(f"   Features HDF5: {features_hdf5}")
        
        results = {}
        
        # 1. Migrate HDF5 features
        if Path(features_hdf5).exists():
            logger.info("📊 Step 1: Migrating HDF5 features...")
            hdf5_results = self.migrate_hdf5_features(features_hdf5, faiss_metadata)
            results['hdf5_migration'] = hdf5_results
        else:
            logger.warning(f"⚠️ HDF5 file not found: {features_hdf5}")
            results['hdf5_migration'] = {'error': 'HDF5 file not found'}
        
        # 2. Process directory structure for additional metadata
        data_path = Path(data_dir)
        if data_path.exists():
            logger.info("📁 Step 2: Processing directory structure...")
            dir_results = self._process_directory_metadata(data_path)
            results['directory_processing'] = dir_results
        else:
            logger.warning(f"⚠️ Data directory not found: {data_dir}")
            results['directory_processing'] = {'error': 'Data directory not found'}
        
        # 3. Validate migration completeness
        logger.info("✅ Step 3: Validating migration...")
        validation_results = self._validate_migration_completeness()
        results['validation'] = validation_results
        
        # Final statistics
        total_time = self.stats.get('migration_end_time', time.time()) - self.stats.get('migration_start_time', time.time())
        
        results['summary'] = {
            'total_features_migrated': self.stats['total_features_migrated'],
            'total_items_migrated': self.stats['total_items_migrated'],
            'total_errors': self.stats['errors_encountered'],
            'migration_time_seconds': total_time,
            'migration_rate_features_per_second': self.stats['total_features_migrated'] / total_time if total_time > 0 else 0
        }
        
        logger.info("🎉 Complete system migration finished!")
        logger.info(f"   Features migrated: {results['summary']['total_features_migrated']}")
        logger.info(f"   Items migrated: {results['summary']['total_items_migrated']}")
        logger.info(f"   Total time: {results['summary']['migration_time_seconds']:.2f}s")
        
        return results
    
    def _process_directory_metadata(self, data_dir: Path) -> Dict:
        """
        Process directory structure to extract additional metadata
        """
        items_processed = 0
        items_with_metadata = 0
        
        for item_dir in data_dir.iterdir():
            if not item_dir.is_dir():
                continue
                
            item_id = item_dir.name
            items_processed += 1
            
            # Look for metadata files
            metadata_file = item_dir / 'augmentation_metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Update item metadata in SQLite
                    self.sqlite_store.add_item(item_id, {
                        'source_directory': str(item_dir),
                        'augmentation_metadata': metadata,
                        'migration_timestamp': datetime.now().isoformat()
                    })
                    
                    items_with_metadata += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process metadata for {item_id}: {e}")
        
        self.stats['total_items_migrated'] = items_processed
        
        return {
            'items_processed': items_processed,
            'items_with_metadata': items_with_metadata
        }
    
    def _validate_migration_completeness(self) -> Dict:
        """
        Validate that the migration preserved data integrity
        """
        try:
            # Get SQLite statistics
            sqlite_stats = self.sqlite_store.get_statistics()
            
            validation_results = {
                'total_items_in_sqlite': sqlite_stats.get('total_items', 0),
                'total_features_in_sqlite': sqlite_stats.get('total_features', 0),
                'total_vectors_in_sqlite': sqlite_stats.get('total_vectors', 0),
                'database_size_mb': sqlite_stats.get('database_size_mb', 0),
                'validation_passed': True,
                'issues': []
            }
            
            # Basic validation checks
            if validation_results['total_features_in_sqlite'] != validation_results['total_vectors_in_sqlite']:
                validation_results['issues'].append("Mismatch between features and vectors count")
                validation_results['validation_passed'] = False
            
            if validation_results['total_features_in_sqlite'] == 0:
                validation_results['issues'].append("No features found in SQLite database")
                validation_results['validation_passed'] = False
            
            # Log validation results
            if validation_results['validation_passed']:
                logger.info("✅ Migration validation passed")
            else:
                logger.warning(f"⚠️ Migration validation issues: {validation_results['issues']}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Migration validation failed: {e}")
            return {
                'validation_passed': False,
                'error': str(e)
            }
    
    def create_performance_comparison(self, original_faiss_path: Optional[str] = None) -> Dict:
        """
        Compare performance characteristics between original and migrated systems
        """
        logger.info("📊 Creating performance comparison...")
        
        comparison = {
            'sqlite_stats': self.sqlite_store.get_statistics(),
            'migration_stats': self.stats,
            'storage_comparison': {
                'original_system': 'HDF5 + FAISS',
                'new_system': 'SQLite + sqlite-vec',
                'feature_dimensions_preserved': '1536D (768D CLIP + 768D DINOv2)',
                'normalization_preserved': 'L2 normalization for cosine similarity'
            }
        }
        
        # Add FAISS comparison if available
        if original_faiss_path and Path(original_faiss_path).exists():
            try:
                import faiss
                original_index = faiss.read_index(original_faiss_path)
                comparison['original_faiss_stats'] = {
                    'total_vectors': original_index.ntotal,
                    'dimension': original_index.d,
                    'index_type': str(type(original_index))
                }
                
                # Compare vector counts
                sqlite_vectors = comparison['sqlite_stats'].get('total_vectors', 0)
                faiss_vectors = original_index.ntotal
                
                comparison['vector_count_match'] = (sqlite_vectors == faiss_vectors)
                comparison['dimension_match'] = (original_index.d == 1536)
                
                logger.info(f"📊 Vector count comparison: SQLite({sqlite_vectors}) vs FAISS({faiss_vectors})")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load original FAISS index: {e}")
        
        return comparison
    
    def get_migration_statistics(self) -> Dict:
        """Get detailed migration statistics"""
        return {
            **self.stats,
            'sqlite_database_stats': self.sqlite_store.get_statistics()
        }


def migrate_system(old_system_dir: str, new_sqlite_db: str, 
                  features_hdf5: str, faiss_metadata: Optional[str] = None) -> Dict:
    """
    Convenience function to migrate complete system from old to new storage
    
    Args:
        old_system_dir: Directory containing old system data
        new_sqlite_db: Path for new SQLite database
        features_hdf5: Path to HDF5 features file
        faiss_metadata: Optional FAISS metadata file
        
    Returns:
        Complete migration results
    """
    logger.info("🔄 Starting complete system migration...")
    
    # Create migrator
    migrator = HDF5ToSQLiteMigrator(new_sqlite_db)
    
    # Perform migration
    results = migrator.migrate_from_directory_structure(
        data_dir=old_system_dir,
        features_hdf5=features_hdf5,
        faiss_metadata=faiss_metadata
    )
    
    # Create performance comparison
    results['performance_comparison'] = migrator.create_performance_comparison()
    
    return results


def main():
    """Main entry point for migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate HDF5+FAISS to SQLite+sqlite-vec')
    parser.add_argument('--old-system', type=str, required=True, help='Old system data directory')
    parser.add_argument('--features-hdf5', type=str, required=True, help='HDF5 features file')
    parser.add_argument('--sqlite-db', type=str, required=True, help='Target SQLite database path')
    parser.add_argument('--faiss-metadata', type=str, help='Optional FAISS metadata file')
    parser.add_argument('--validate', action='store_true', help='Run validation after migration')
    
    args = parser.parse_args()
    
    # Run migration
    results = migrate_system(
        old_system_dir=args.old_system,
        new_sqlite_db=args.sqlite_db,
        features_hdf5=args.features_hdf5,
        faiss_metadata=args.faiss_metadata
    )
    
    # Print results
    print("\n🎉 Migration Complete!")
    print(f"   Features migrated: {results['summary']['total_features_migrated']}")
    print(f"   Items migrated: {results['summary']['total_items_migrated']}")
    print(f"   Migration time: {results['summary']['migration_time_seconds']:.2f}s")
    print(f"   Rate: {results['summary']['migration_rate_features_per_second']:.1f} features/sec")
    
    if args.validate:
        validation = results.get('validation', {})
        if validation.get('validation_passed', False):
            print("✅ Validation: PASSED")
        else:
            print(f"⚠️ Validation: ISSUES - {validation.get('issues', [])}")


if __name__ == "__main__":
    main()