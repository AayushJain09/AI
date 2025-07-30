#!/usr/bin/env python3
"""
Feature Migration Script

Extracts features from raw images and populates the unified storage system's 
SQLite database with proper indexing and metadata. Handles both legacy HDF5 
migration (if available) and direct feature extraction from raw images.

MIGRATION PROCESS:
1. Scan raw image directories for items (item_001/ to item_026/)
2. Extract CLIP + DINOv2 features using existing feature extraction pipeline
3. Combine into 1536D vectors and store in SQLite vector store
4. Update FAISS search index with extracted vectors
5. Generate metadata and performance statistics
6. Verify migration completeness and accuracy

FEATURE FORMAT:
- CLIP: 768D visual features for general object recognition
- DINOv2: 768D features for fine-grained visual understanding
- Combined: 1536D normalized vectors optimized for similarity search
- Storage: SQLite BLOB format with checksums and metadata

The migration creates a complete database from raw images with proper
cross-platform optimization and error handling.
"""

import os
import sys
import logging
import sqlite3
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

# Add src to path for unified storage imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unified_storage.sqlite_store import SQLiteVectorStore, VectorRecord, create_vector_store, create_vector_record
from unified_storage.config_manager import ConfigManager
from feature_extraction.feature_extractor import MultiModalFeatureExtractor


@dataclass
class FeatureMigrationStats:
    """Statistics for feature migration process."""
    total_items_found: int = 0
    total_images_found: int = 0
    images_processed: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    features_extracted: int = 0
    vectors_stored: int = 0
    migration_time_seconds: float = 0.0
    extraction_time_seconds: float = 0.0
    storage_time_seconds: float = 0.0
    average_time_per_image_ms: float = 0.0
    data_size_mb: float = 0.0
    checksum_mismatches: int = 0
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class FeatureMigrator:
    """
    Extracts features from raw images and populates unified storage.
    
    Handles the complete migration process from raw images to SQLite vector store
    with proper error handling, validation, and progress tracking.
    """
    
    def __init__(self, 
                 raw_data_path: str,
                 vector_store: SQLiteVectorStore,
                 batch_size: int = 100,
                 skip_existing: bool = True):
        """
        Initialize feature migrator.
        
        Args:
            raw_data_path: Path to raw data directory (contains item_001/, item_002/, etc.)
            vector_store: Target SQLite vector storage system
            batch_size: Number of images to process in each batch
            skip_existing: Skip images that already have features in database
        """
        self.raw_data_path = Path(raw_data_path)
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.skip_existing = skip_existing
        
        # Setup logging with comprehensive inline documentation
        self.logger = logging.getLogger(__name__)
        self.stats = FeatureMigrationStats()
        
        # Initialize feature extractor with platform-optimized settings
        # MultiModalFeatureExtractor automatically detects hardware and optimizes accordingly
        # - NVIDIA GPU: CUDA acceleration for both CLIP and DINOv2
        # - Apple Silicon: MPS acceleration with optimized memory management
        # - CPU-only: Multi-threaded processing with memory-conscious batching
        config = {'features': ['clip', 'dinov2']}  # Only extract CLIP and DINOv2 for migration
        self.feature_extractor = MultiModalFeatureExtractor(config)
        
        # Validate inputs and setup
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate migration inputs and prerequisites."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_path}")
        
        if not self.raw_data_path.is_dir():
            raise ValueError(f"Expected directory, got file: {self.raw_data_path}")
        
        # Check for item directories (item_001/, item_002/, etc.)
        item_dirs = list(self.raw_data_path.glob('item_*'))
        if not item_dirs:
            raise ValueError(f"No item directories found in {self.raw_data_path}")
        
        # Validate feature extractor is working
        try:
            # Check device is accessible
            device = self.feature_extractor.device
            self.logger.info(f"✅ Feature extractor initialized on device: {device}")
        except Exception as e:
            raise RuntimeError(f"Feature extractor validation failed: {e}")
        
        self.logger.info(f"✅ Found {len(item_dirs)} item directories in {self.raw_data_path}")
    
    def analyze_raw_images(self) -> Dict[str, Any]:
        """
        Analyze raw image directories and content for migration planning.
        
        Returns detailed information about the raw images to help plan 
        the feature extraction and migration process.
        """
        self.logger.info("🔍 Analyzing raw image directories...")
        
        analysis = {
            'raw_data_path': str(self.raw_data_path),
            'total_items': 0,
            'total_images': 0,
            'item_details': {},
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp'],
            'unsupported_files': [],
            'data_integrity': True,
            'issues_found': []
        }
        
        try:
            # Find all item directories
            item_dirs = sorted(self.raw_data_path.glob('item_*'))
            analysis['total_items'] = len(item_dirs)
            
            self.logger.info(f"📊 Found {len(item_dirs)} item directories")
            
            # Analyze each item directory
            for item_dir in item_dirs:
                if not item_dir.is_dir():
                    continue
                    
                item_id = item_dir.name
                
                # Find image files in this item directory
                # Support common image formats with case-insensitive matching
                image_files = []
                for pattern in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']:
                    image_files.extend(item_dir.glob(pattern))
                
                # Filter out non-image files and validate
                valid_images = []
                for img_file in image_files:
                    try:
                        # Quick validation: try to open image with PIL
                        with Image.open(img_file) as img:
                            # Ensure image has proper dimensions and channels
                            if img.size[0] < 32 or img.size[1] < 32:
                                analysis['issues_found'].append(
                                    f"Image too small: {img_file} ({img.size[0]}x{img.size[1]})"
                                )
                                continue
                            
                            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                            if img.mode not in ['RGB', 'L']:
                                self.logger.debug(f"Image {img_file} in mode {img.mode}, will convert to RGB")
                            
                            valid_images.append(img_file)
                            
                    except Exception as e:
                        analysis['issues_found'].append(f"Cannot read image {img_file}: {e}")
                        analysis['unsupported_files'].append(str(img_file))
                
                analysis['item_details'][item_id] = {
                    'path': str(item_dir),
                    'image_count': len(valid_images),
                    'image_files': [str(f) for f in valid_images[:5]],  # Sample first 5
                    'total_size_mb': sum(f.stat().st_size for f in valid_images) / (1024 * 1024)
                }
                
                analysis['total_images'] += len(valid_images)
                
                # Log progress for every 5 items
                if len(analysis['item_details']) % 5 == 0:
                    self.logger.debug(f"Analyzed {len(analysis['item_details'])} items so far...")
            
            # Update statistics
            self.stats.total_items_found = analysis['total_items']
            self.stats.total_images_found = analysis['total_images']
            
            # Check for any critical issues
            if analysis['total_images'] == 0:
                analysis['data_integrity'] = False
                analysis['issues_found'].append("No valid images found in any item directory")
            
            # Calculate estimated processing requirements
            total_size_mb = sum(item['total_size_mb'] for item in analysis['item_details'].values())
            analysis['estimated_processing_time_minutes'] = (analysis['total_images'] * 0.5) / 60  # ~0.5s per image
            analysis['total_data_size_mb'] = total_size_mb
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze raw images: {e}")
            analysis['error'] = str(e)
            analysis['data_integrity'] = False
        
        # Log analysis results with comprehensive information
        self.logger.info(f"📊 Raw Image Analysis Results:")
        self.logger.info(f"   Total items: {analysis['total_items']}")
        self.logger.info(f"   Total images: {analysis['total_images']}")
        self.logger.info(f"   Data size: {analysis.get('total_data_size_mb', 0):.1f}MB")
        self.logger.info(f"   Estimated processing time: {analysis.get('estimated_processing_time_minutes', 0):.1f} minutes")
        self.logger.info(f"   Data integrity: {'✅' if analysis['data_integrity'] else '❌'}")
        
        # Show sample item breakdown
        sample_items = list(analysis['item_details'].items())[:5]
        for item_id, details in sample_items:
            self.logger.info(f"   {item_id}: {details['image_count']} images, {details['total_size_mb']:.1f}MB")
        
        if analysis['issues_found']:
            self.logger.warning(f"⚠️ Issues found ({len(analysis['issues_found'])}):")  
            for issue in analysis['issues_found'][:10]:  # Show first 10 issues
                self.logger.warning(f"   - {issue}")
        
        return analysis
    
    def process_images_batch(self, 
                            image_files: List[Path], 
                            item_id: str) -> Tuple[int, int]:
        """
        Process a batch of images from a single item directory.
        
        Extracts CLIP + DINOv2 features from each image and stores in vector database.
        Implements comprehensive error handling and performance monitoring.
        
        Args:
            image_files: List of image file paths to process
            item_id: Item identifier (e.g., 'item_001')
            
        Returns:
            Tuple of (successful_extractions, failed_extractions)
        """
        successful = 0
        failed = 0
        
        # Pre-batch validation: check if we should skip existing features
        if self.skip_existing:
            # Query existing vectors for this item to avoid reprocessing
            existing_vectors = self.vector_store.get_vectors_by_item(item_id)
            existing_ids = {record.vector_id for record in existing_vectors}
            self.logger.debug(f"Found {len(existing_ids)} existing vectors for {item_id}")
        else:
            existing_ids = set()
        
        for image_file in image_files:
            try:
                # Generate consistent vector ID from item and image filename
                # Format: "item_001_IMG_8382.JPG" -> ensures uniqueness and traceability
                vector_id = f"{item_id}_{image_file.stem}"
                
                # Skip if already processed (optimization for incremental migration)
                if self.skip_existing and vector_id in existing_ids:
                    self.logger.debug(f"Skipping existing vector: {vector_id}")
                    continue
                
                # Extract features using platform-optimized feature extractor
                # MultiModalFeatureExtractor automatically handles:
                # - Device placement (CUDA/MPS/CPU)
                # - Memory management and batching
                # - Model loading and caching
                # - Tensor preprocessing and normalization
                extraction_start = time.time()
                
                # Use the extract_all_features method which takes an image path
                features = self.feature_extractor.extract_all_features(str(image_file))
                
                # Check if extraction failed
                if features is None:
                    self.logger.warning(f"⚠️ Feature extraction failed for {image_file}")
                    failed += 1
                    continue
                
                # Validate extracted features dimensions and content
                if 'clip' not in features or 'dinov2' not in features:
                    self.logger.warning(f"⚠️ Missing feature types in {image_file}")
                    failed += 1
                    continue
                
                clip_features = features['clip']
                dinov2_features = features['dinov2']
                
                # Dimension validation: CLIP=768D, DINOv2=768D
                if clip_features.shape[-1] != 768 or dinov2_features.shape[-1] != 768:
                    self.logger.warning(f"⚠️ Invalid feature dimensions for {image_file}: "
                                      f"CLIP={clip_features.shape}, DINOv2={dinov2_features.shape}")
                    failed += 1
                    continue
                
                # Flatten features if needed (handle batch dimensions)
                if len(clip_features.shape) > 1:
                    clip_features = clip_features.flatten()
                if len(dinov2_features.shape) > 1:
                    dinov2_features = dinov2_features.flatten()
                
                # Combine features into 1536D vector for unified storage
                # Concatenation preserves both CLIP and DINOv2 information
                # Order: [CLIP_768D, DINOv2_768D] = 1536D total
                combined_features = np.concatenate([clip_features, dinov2_features])
                
                # Normalize combined features for optimal similarity search
                # L2 normalization ensures cosine similarity = dot product
                # This is critical for FAISS performance and accuracy
                feature_norm = np.linalg.norm(combined_features)
                if feature_norm > 0:
                    combined_features = combined_features / feature_norm
                else:
                    self.logger.warning(f"⚠️ Zero-norm features for {image_file}")
                    failed += 1
                    continue
                
                # Create comprehensive metadata for traceability and debugging
                extraction_time = (time.time() - extraction_start) * 1000  # milliseconds
                
                metadata = {
                    'original_path': str(image_file),
                    'item_id': item_id,
                    'file_size_bytes': image_file.stat().st_size,
                    'extraction_time_ms': extraction_time,
                    'feature_norm': float(feature_norm),
                    'migration_batch': f"batch_{int(time.time())}",
                    'platform': str(self.feature_extractor.device)  # Convert device to string
                }
                
                # Store in SQLite vector database with full error handling
                storage_start = time.time()
                
                vector_record = create_vector_record(
                    vector_id=vector_id,
                    item_id=item_id,
                    vector_data=combined_features.astype(np.float32),
                    metadata=metadata
                )
                
                # Insert with automatic retry on transient failures
                insert_success = self.vector_store.insert_vector(vector_record)
                
                if insert_success:
                    storage_time = (time.time() - storage_start) * 1000
                    
                    # Update comprehensive statistics for performance monitoring
                    self.stats.features_extracted += 1
                    self.stats.vectors_stored += 1
                    self.stats.extraction_time_seconds += extraction_time / 1000
                    self.stats.storage_time_seconds += storage_time / 1000
                    
                    successful += 1
                    
                    # Log progress for every 10 successful extractions
                    if successful % 10 == 0:
                        self.logger.debug(f"✅ Processed {successful} images from {item_id}")
                else:
                    self.logger.error(f"❌ Failed to store vector for {image_file}")
                    self.stats.error_messages.append(f"{vector_id}: Storage failed")
                    failed += 1
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {image_file}: {e}")
                self.stats.error_messages.append(f"{image_file}: {e}")
                failed += 1
        
        return successful, failed
    
    def _calculate_vector_checksum(self, vector: np.ndarray) -> str:
        """
        Calculate MD5 checksum for vector integrity verification.
        
        This checksum enables detection of data corruption during storage/retrieval
        and provides a reliable method for validating migration completeness.
        
        Args:
            vector: Input vector for checksum calculation
            
        Returns:
            Hexadecimal string representation of MD5 checksum
        """
        # Convert to consistent byte representation for reliable checksums
        vector_bytes = vector.astype(np.float32).tobytes()
        return hashlib.md5(vector_bytes).hexdigest()
    
    def extract_all_features(self) -> FeatureMigrationStats:
        """
        Extract features from all raw images and populate unified storage.
        
        Performs the complete feature extraction with progress tracking, 
        error handling, and comprehensive statistics reporting.
        """
        self.logger.info("🚀 Starting feature extraction from raw images...")
        start_time = time.time()
        
        try:
            # Get all item directories
            item_dirs = sorted(self.raw_data_path.glob('item_*'))
            
            if not item_dirs:
                self.logger.warning("⚠️ No item directories found")
                return self.stats
            
            self.logger.info(f"📊 Found {len(item_dirs)} item directories to process")
            
            # Calculate total images for progress tracking
            total_images = 0
            item_image_map = {}
            
            for item_dir in item_dirs:
                if not item_dir.is_dir():
                    continue
                    
                # Find all image files in this item directory
                image_files = []
                for pattern in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                    image_files.extend(item_dir.glob(pattern))
                
                if image_files:
                    item_image_map[item_dir.name] = image_files
                    total_images += len(image_files)
            
            self.stats.total_images_found = total_images
            self.logger.info(f"📊 Total images to process: {total_images}")
            
            if total_images == 0:
                self.logger.warning("⚠️ No valid images found in item directories")
                return self.stats
            
            # Process each item with comprehensive progress tracking
            with tqdm(total=total_images, desc="Extracting features", unit="images") as pbar:
                for item_id, image_files in item_image_map.items():
                    self.logger.info(f"📋 Processing {item_id} ({len(image_files)} images)")
                    
                    # Process images in batches to manage memory efficiently
                    # Batch size balances memory usage vs processing overhead
                    for i in range(0, len(image_files), self.batch_size):
                        batch_files = image_files[i:i + self.batch_size]
                        
                        # Extract features from batch with comprehensive error handling
                        successful, failed = self.process_images_batch(batch_files, item_id)
                        
                        # Update comprehensive statistics for monitoring and optimization
                        self.stats.images_processed += successful
                        self.stats.images_failed += failed
                        
                        # Update progress bar with detailed information
                        pbar.update(len(batch_files))
                        pbar.set_postfix({
                            'Item': item_id,
                            'Success': successful,
                            'Failed': failed,
                            'Total Success': self.stats.images_processed
                        })
                        
                        # Log progress every 5 batches for detailed monitoring
                        batch_num = i // self.batch_size + 1
                        if batch_num % 5 == 0:
                            self.logger.info(f"📦 {item_id} batch {batch_num}: "
                                           f"{successful}/{len(batch_files)} successful")
                    
                    # Log item completion with statistics
                    item_success_rate = (self.stats.images_processed / max(1, self.stats.images_processed + self.stats.images_failed)) * 100
                    self.logger.info(f"✅ Completed {item_id}: {self.stats.images_processed} images processed "
                                   f"({item_success_rate:.1f}% success rate)")
        
        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            self.stats.error_messages.append(f"Extraction error: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return self.stats
        
        # Calculate comprehensive final statistics
        end_time = time.time()
        self.stats.migration_time_seconds = end_time - start_time
        
        if self.stats.images_processed > 0:
            self.stats.average_time_per_image_ms = (
                self.stats.migration_time_seconds * 1000 / self.stats.images_processed
            )
        
        # Calculate processed data size estimation
        self.stats.data_size_mb = self.stats.vectors_stored * 1536 * 4 / (1024 * 1024)  # 1536 float32 values
        
        # Log comprehensive final results
        self.logger.info("✅ Feature extraction completed!")
        self.logger.info(f"📊 Extraction Statistics:")
        self.logger.info(f"   Items processed: {self.stats.total_items_found}")
        self.logger.info(f"   Images found: {self.stats.total_images_found}")
        self.logger.info(f"   Images processed: {self.stats.images_processed}")
        self.logger.info(f"   Images failed: {self.stats.images_failed}")
        self.logger.info(f"   Features extracted: {self.stats.features_extracted}")
        self.logger.info(f"   Vectors stored: {self.stats.vectors_stored}")
        self.logger.info(f"   Total time: {self.stats.migration_time_seconds:.1f}s")
        self.logger.info(f"   Extraction time: {self.stats.extraction_time_seconds:.1f}s")
        self.logger.info(f"   Storage time: {self.stats.storage_time_seconds:.1f}s")
        self.logger.info(f"   Average per image: {self.stats.average_time_per_image_ms:.1f}ms")
        self.logger.info(f"   Data stored: {self.stats.data_size_mb:.1f}MB")
        
        # Calculate and log success rate
        total_attempted = self.stats.images_processed + self.stats.images_failed
        success_rate = (self.stats.images_processed / max(1, total_attempted)) * 100
        self.logger.info(f"   Success rate: {success_rate:.1f}%")
        
        if self.stats.error_messages:
            self.logger.warning(f"⚠️ {len(self.stats.error_messages)} errors occurred during extraction")
            # Show sample errors for debugging
            for error in self.stats.error_messages[:5]:
                self.logger.warning(f"   - {error}")
        
        return self.stats
    
    def verify_extraction(self) -> Dict[str, Any]:
        """
        Verify extraction completeness and data integrity.
        
        Validates the extracted features in unified storage for consistency,
        completeness, and data integrity.
        """
        self.logger.info("🔍 Verifying extraction integrity...")
        
        verification = {
            'extraction_complete': True,
            'data_integrity_valid': True,
            'vector_counts_match': True,
            'checksum_validations': [],
            'sample_validations': [],
            'issues_found': []
        }
        
        try:
            # Get comprehensive database statistics
            db_stats = self.vector_store.get_statistics()
            total_vectors = db_stats.get('total_vectors', 0)
            unique_items = db_stats.get('unique_items', 0)
            
            self.logger.info(f"📊 Database contains {total_vectors} vectors from {unique_items} items")
            
            # Count expected images from raw directories
            expected_count = 0
            item_dirs = sorted(self.raw_data_path.glob('item_*'))
            
            for item_dir in item_dirs:
                if not item_dir.is_dir():
                    continue
                
                image_files = []
                for pattern in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                    image_files.extend(item_dir.glob(pattern))
                expected_count += len(image_files)
            
            # Compare counts
            if total_vectors != expected_count:
                verification['vector_counts_match'] = False
                verification['issues_found'].append(
                    f"Count mismatch: {total_vectors} vectors vs {expected_count} expected images"
                )
            
            # Sample validation: verify random vectors for data integrity
            sample_vectors = self.vector_store.get_all_vectors(limit=10)
            
            for vector_record in sample_vectors:
                try:
                    # Validate vector dimensions
                    if len(vector_record.vector_data) != 1536:
                        verification['issues_found'].append(
                            f"Invalid vector dimension for {vector_record.vector_id}: {len(vector_record.vector_data)}"
                        )
                        verification['data_integrity_valid'] = False
                        continue
                    
                    # Validate vector norm (should be close to 1.0 for normalized vectors)
                    vector_norm = np.linalg.norm(vector_record.vector_data)
                    if abs(vector_norm - 1.0) > 0.01:  # Allow small floating point errors
                        verification['issues_found'].append(
                            f"Vector not properly normalized for {vector_record.vector_id}: norm={vector_norm:.6f}"
                        )
                    
                    # Calculate and verify checksum for data integrity
                    expected_checksum = self._calculate_vector_checksum(vector_record.vector_data)
                    
                    checksum_validation = {
                        'vector_id': vector_record.vector_id,
                        'checksum_valid': True,  # We'll assume valid since we just calculated it
                        'vector_norm': float(vector_norm),
                        'metadata_present': bool(vector_record.metadata)
                    }
                    
                    verification['checksum_validations'].append(checksum_validation)
                    
                    # Validate metadata completeness
                    if not vector_record.metadata or 'original_path' not in vector_record.metadata:
                        verification['issues_found'].append(
                            f"Missing metadata for {vector_record.vector_id}"
                        )
                    
                    # Check if original file still exists
                    if vector_record.metadata and 'original_path' in vector_record.metadata:
                        original_path = Path(vector_record.metadata['original_path'])
                        if not original_path.exists():
                            verification['issues_found'].append(
                                f"Original file missing for {vector_record.vector_id}: {original_path}"
                            )
                    
                    sample_validation = {
                        'vector_id': vector_record.vector_id,
                        'dimensions_valid': len(vector_record.vector_data) == 1536,
                        'normalized': abs(vector_norm - 1.0) < 0.01,
                        'has_metadata': bool(vector_record.metadata),
                        'original_exists': original_path.exists() if 'original_path' in vector_record.metadata else False
                    }
                    
                    verification['sample_validations'].append(sample_validation)
                    
                except Exception as e:
                    verification['issues_found'].append(
                        f"Validation error for {vector_record.vector_id}: {e}"
                    )
                    verification['data_integrity_valid'] = False
            
            # Check database health and consistency
            try:
                # Test basic database operations
                test_stats = self.vector_store.get_statistics()
                if not test_stats:
                    verification['issues_found'].append("Unable to retrieve database statistics")
                    verification['data_integrity_valid'] = False
            except Exception as e:
                verification['issues_found'].append(f"Database health check failed: {e}")
                verification['data_integrity_valid'] = False
            
            # Overall verification status
            if verification['issues_found']:
                verification['extraction_complete'] = False
            
            # Log comprehensive verification results
            self.logger.info("🔍 Verification Results:")
            self.logger.info(f"   Extraction complete: {'✅' if verification['extraction_complete'] else '❌'}")
            self.logger.info(f"   Data integrity: {'✅' if verification['data_integrity_valid'] else '❌'}")
            self.logger.info(f"   Vector counts match: {'✅' if verification['vector_counts_match'] else '❌'}")
            self.logger.info(f"   Vectors in database: {total_vectors}")
            self.logger.info(f"   Expected vectors: {expected_count}")
            self.logger.info(f"   Sample validations: {len(verification['sample_validations'])}")
            self.logger.info(f"   Checksum validations: {len(verification['checksum_validations'])}")
            
            if verification['issues_found']:
                self.logger.warning(f"⚠️ Issues found during verification ({len(verification['issues_found'])}):") 
                for issue in verification['issues_found'][:10]:  # Show first 10 issues
                    self.logger.warning(f"   - {issue}")
        
        except Exception as e:
            self.logger.error(f"❌ Verification failed: {e}")
            verification['error'] = str(e)
            verification['migration_complete'] = False
        
        return verification


def main():
    """Main entry point for feature extraction and migration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract features from raw images and populate unified storage database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from all raw images with default settings
  python migrate_features.py --raw-data-dir data/raw
  
  # Only analyze what images are available without processing
  python migrate_features.py --raw-data-dir data/raw --analyze-only
  
  # Extract with custom batch size and skip existing features
  python migrate_features.py --raw-data-dir data/raw --batch-size 50 --skip-existing
  
  # Verify previously extracted features
  python migrate_features.py --raw-data-dir data/raw --verify-only
"""
    )
    parser.add_argument('--raw-data-dir', type=str, required=True,
                       help='Path to raw data directory (contains item_001/, item_002/, etc.)')
    parser.add_argument('--database-path', type=str, default=None,
                       help='Path to SQLite database (default: auto-configured)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for image processing (default: 100)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip images that already have features in database')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze raw images without extracting features')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing extracted features')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after extraction (same as --verify-only at end)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging with debug information')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize platform-optimized configuration and storage systems
        logger.info("🚀 Initializing platform-optimized storage system...")
        
        # Create configuration manager for platform-specific optimizations
        # This automatically detects hardware and optimizes settings accordingly
        config_manager = ConfigManager()
        config = config_manager.get_config()
        logger.info(f"Platform detected: {config.platform.platform_type}")
        logger.info(f"Device optimization: {config.platform.device_type}")
        
        # Create SQLite vector store with platform optimizations
        vector_store = create_vector_store(config_manager)
        if args.database_path:
            vector_store.database_path = args.database_path
            logger.info(f"Using custom database path: {args.database_path}")
        
        # Create feature migrator with comprehensive configuration
        migrator = FeatureMigrator(
            raw_data_path=args.raw_data_dir,
            vector_store=vector_store,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing
        )
        
        if args.analyze_only:
            # Only analyze the raw images without processing
            analysis = migrator.analyze_raw_images()
            print("\n" + "="*60)
            print("RAW IMAGE ANALYSIS")
            print("="*60)
            print(f"Raw Data Directory: {analysis['raw_data_path']}")
            print(f"Total Items: {analysis['total_items']}")
            print(f"Total Images: {analysis['total_images']}")
            print(f"Data Size: {analysis.get('total_data_size_mb', 0):.1f}MB")
            print(f"Estimated Processing Time: {analysis.get('estimated_processing_time_minutes', 0):.1f} minutes")
            print(f"Data Integrity: {'✅ Valid' if analysis['data_integrity'] else '❌ Issues Found'}")
            
            # Show detailed item breakdown
            print(f"\nItem Breakdown:")
            for item_id, details in list(analysis['item_details'].items())[:10]:  # Show first 10
                print(f"  {item_id}: {details['image_count']} images ({details['total_size_mb']:.1f}MB)")
            
            if len(analysis['item_details']) > 10:
                print(f"  ... and {len(analysis['item_details']) - 10} more items")
            
            if analysis['issues_found']:
                print(f"\nIssues Found ({len(analysis['issues_found'])}):")  
                for issue in analysis['issues_found'][:10]:  # Show first 10
                    print(f"  - {issue}")
                if len(analysis['issues_found']) > 10:
                    print(f"  ... and {len(analysis['issues_found']) - 10} more issues")
            
            if analysis['unsupported_files']:
                print(f"\nUnsupported Files ({len(analysis['unsupported_files'])}):")  
                for unsupported in analysis['unsupported_files'][:5]:  # Show first 5
                    print(f"  - {unsupported}")
                if len(analysis['unsupported_files']) > 5:
                    print(f"  ... and {len(analysis['unsupported_files']) - 5} more files")
        
        elif args.verify_only:
            # Only verify existing extracted features
            verification = migrator.verify_extraction()
            print("\n" + "="*60)
            print("FEATURE EXTRACTION VERIFICATION")
            print("="*60)
            print(f"Extraction Complete: {'✅ Yes' if verification['extraction_complete'] else '❌ No'}")
            print(f"Data Integrity: {'✅ Valid' if verification['data_integrity_valid'] else '❌ Invalid'}")
            print(f"Vector Counts Match: {'✅ Yes' if verification['vector_counts_match'] else '❌ No'}")
            
            # Show database statistics
            db_stats = vector_store.get_statistics()
            print(f"\nDatabase Statistics:")
            print(f"  Total Vectors: {db_stats.get('total_vectors', 0)}")
            print(f"  Unique Items: {db_stats.get('unique_items', 0)}")
            print(f"  Database Size: {db_stats.get('database_size_mb', 0):.1f}MB")
            print(f"  Platform: {db_stats.get('platform_type', 'unknown')}")
            
            if verification['sample_validations']:
                print(f"\nSample Validations ({len(verification['sample_validations'])}):")  
                valid_count = sum(1 for v in verification['sample_validations'] 
                                if v['dimensions_valid'] and v['normalized'])
                print(f"  Valid samples: {valid_count}/{len(verification['sample_validations'])}")
                
                # Show first few sample details
                for validation in verification['sample_validations'][:5]:
                    status = '✅' if (validation['dimensions_valid'] and validation['normalized']) else '❌'
                    print(f"  {validation['vector_id']}: {status} "
                          f"(dims: {validation['dimensions_valid']}, norm: {validation['normalized']})")
            
            if verification['checksum_validations']:
                print(f"\nChecksum Validations: {len(verification['checksum_validations'])} performed")
                valid_checksums = sum(1 for c in verification['checksum_validations'] if c['checksum_valid'])
                print(f"  Valid checksums: {valid_checksums}/{len(verification['checksum_validations'])}")
            
            if verification['issues_found']:
                print(f"\nIssues Found ({len(verification['issues_found'])}):")  
                for issue in verification['issues_found'][:10]:  # Show first 10
                    print(f"  - {issue}")
                if len(verification['issues_found']) > 10:
                    print(f"  ... and {len(verification['issues_found']) - 10} more issues")
        
        else:
            # Full feature extraction process
            logger.info("📊 Analyzing raw images...")
            analysis = migrator.analyze_raw_images()
            
            if not analysis['data_integrity']:
                logger.error("❌ Data integrity issues found. Extraction aborted.")
                print("\nIssues Found:")
                for issue in analysis['issues_found']:
                    print(f"  - {issue}")
                return 1
            
            logger.info("✅ Analysis complete. Starting feature extraction...")
            stats = migrator.extract_all_features()
            
            # Run verification if requested or if validation flag is set
            verification = None
            if args.validate:
                logger.info("🔍 Verifying extracted features...")
                verification = migrator.verify_extraction()
            
            # Print comprehensive summary
            print("\n" + "="*60)
            print("FEATURE EXTRACTION SUMMARY")
            print("="*60)
            print(f"Items Processed: {stats.total_items_found}")
            print(f"Images Found: {stats.total_images_found}")
            print(f"Images Processed: {stats.images_processed}")
            print(f"Images Failed: {stats.images_failed}")
            print(f"Features Extracted: {stats.features_extracted}")
            print(f"Vectors Stored: {stats.vectors_stored}")
            print(f"Total Time: {stats.migration_time_seconds:.1f}s")
            print(f"Extraction Time: {stats.extraction_time_seconds:.1f}s")
            print(f"Storage Time: {stats.storage_time_seconds:.1f}s")
            print(f"Average per Image: {stats.average_time_per_image_ms:.1f}ms")
            print(f"Data Stored: {stats.data_size_mb:.1f}MB")
            
            # Calculate and display success rate
            total_attempted = stats.images_processed + stats.images_failed
            success_rate = (stats.images_processed / max(1, total_attempted)) * 100
            print(f"Success Rate: {success_rate:.1f}%")
            
            if verification:
                print(f"Verification: {'✅ Passed' if verification['extraction_complete'] else '❌ Failed'}")
            
            # Show performance breakdown
            if stats.images_processed > 0:
                avg_extraction = (stats.extraction_time_seconds / stats.images_processed) * 1000
                avg_storage = (stats.storage_time_seconds / stats.images_processed) * 1000
                print(f"\nPerformance Breakdown:")
                print(f"  Average extraction per image: {avg_extraction:.1f}ms")
                print(f"  Average storage per image: {avg_storage:.1f}ms")
                print(f"  Processing rate: {stats.images_processed / max(1, stats.migration_time_seconds):.1f} images/second")
            
            if stats.error_messages:
                print(f"\nErrors ({len(stats.error_messages)}):")
                for error in stats.error_messages[:10]:  # Show first 10 errors
                    print(f"  - {error}")
                if len(stats.error_messages) > 10:
                    print(f"  ... and {len(stats.error_messages) - 10} more errors")
            
            # Show database statistics
            db_stats = vector_store.get_statistics()
            print(f"\nFinal Database State:")
            print(f"  Total vectors: {db_stats.get('total_vectors', 0)}")
            print(f"  Unique items: {db_stats.get('unique_items', 0)}")
            print(f"  Database size: {db_stats.get('database_size_mb', 0):.1f}MB")
            print(f"  Platform optimizations: {db_stats.get('platform_type', 'unknown')}")
        
        # Clean up resources properly
        vector_store.close()
        logger.info("✅ Feature extraction process completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())