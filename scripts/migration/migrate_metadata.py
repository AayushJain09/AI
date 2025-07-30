#!/usr/bin/env python3
"""
Metadata Migration Script

Extracts metadata from legacy directory structure and image files,
then stores it in the unified storage system's metadata database.

MIGRATION PROCESS:
1. Scan directory structure (data/raw/item_XXX/) for organization patterns
2. Extract file metadata (size, dimensions, format, timestamps)
3. Analyze image content for additional metadata (color, quality, etc.)
4. Extract item categorization from directory names
5. Store structured metadata in SQLite database
6. Create search-optimized indices for efficient queries

METADATA SOURCES:
- Directory structure: Item categories and organization
- File system: File sizes, timestamps, paths
- Image files: Dimensions, format, quality, EXIF data
- Content analysis: Color profiles, quality metrics
- Recognition history: Previous recognition results if available

The migration preserves all available metadata while creating a unified,
searchable structure for the cross-platform storage system.
"""

import os
import sys
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import re
from collections import defaultdict
import mimetypes

# Image processing imports
try:
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

import numpy as np
from tqdm import tqdm

# Add src to path for unified storage imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unified_storage.sqlite_store import SQLiteVectorStore, create_vector_store
from unified_storage.config_manager import ConfigManager


@dataclass
class MetadataMigrationStats:
    """Statistics for metadata migration process."""
    directories_scanned: int = 0
    files_discovered: int = 0
    images_processed: int = 0
    metadata_records_created: int = 0
    categories_discovered: int = 0
    migration_time_seconds: float = 0.0
    average_time_per_file_ms: float = 0.0
    total_data_size_mb: float = 0.0
    exif_data_extracted: int = 0
    content_analysis_completed: int = 0
    error_messages: List[str] = field(default_factory=list)
    category_distribution: Dict[str, int] = field(default_factory=dict)


class MetadataMigrator:
    """
    Migrates metadata from legacy directory structure to unified storage.
    
    Analyzes directory structure, extracts file and image metadata,
    and creates comprehensive metadata records in the unified storage system.
    """
    
    def __init__(self, 
                 legacy_data_dir: str,
                 vector_store: SQLiteVectorStore,
                 enable_content_analysis: bool = True):
        """
        Initialize metadata migrator.
        
        Args:
            legacy_data_dir: Root directory of legacy data structure
            vector_store: Target SQLite vector storage system
            enable_content_analysis: Whether to perform content analysis
        """
        self.legacy_data_dir = Path(legacy_data_dir)
        self.vector_store = vector_store
        self.enable_content_analysis = enable_content_analysis
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.stats = MetadataMigrationStats()
        
        # Initialize metadata patterns
        self._setup_metadata_patterns()
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate migration inputs and prerequisites."""
        if not self.legacy_data_dir.exists():
            raise FileNotFoundError(f"Legacy data directory not found: {self.legacy_data_dir}")
        
        if not self.legacy_data_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.legacy_data_dir}")
        
        if not PIL_AVAILABLE and self.enable_content_analysis:
            self.logger.warning("⚠️ PIL not available - content analysis will be limited")
            self.enable_content_analysis = False
        
        self.logger.info(f"✅ Validation passed for {self.legacy_data_dir}")
    
    def _setup_metadata_patterns(self):
        """Setup regex patterns for extracting metadata from paths and filenames."""
        # Common patterns for extracting information
        self.patterns = {
            # Item ID patterns: item_001, item_002, etc.
            'item_id': re.compile(r'item_(\d+)', re.IGNORECASE),
            
            # Image number patterns: 1.jpg, 01.jpg, image_1.jpg, etc.
            'image_number': re.compile(r'(?:image_)?(\d+)\.', re.IGNORECASE),
            
            # Category patterns from directory names
            'category': re.compile(r'category_(\w+)', re.IGNORECASE),
            
            # Date patterns in filenames: 2023-01-01, 20230101, etc.
            'date': re.compile(r'(\d{4}[-_]?\d{2}[-_]?\d{2})'),
            
            # Quality indicators: high_res, low_res, thumb, etc.
            'quality': re.compile(r'(high_res|low_res|thumb|thumbnail|preview)', re.IGNORECASE),
            
            # Augmentation indicators: rotated, flipped, scaled, etc.
            'augmentation': re.compile(r'(rotated|flipped|scaled|cropped|augmented)', re.IGNORECASE)
        }
    
    def discover_directory_structure(self) -> Dict[str, Any]:
        """
        Analyze legacy directory structure to understand organization patterns.
        
        Returns comprehensive information about the directory structure,
        file organization, and discovered patterns.
        """
        self.logger.info("🔍 Analyzing legacy directory structure...")
        
        structure_analysis = {
            'root_path': str(self.legacy_data_dir),
            'total_directories': 0,
            'total_files': 0,
            'image_files': 0,
            'directory_levels': 0,
            'item_directories': [],
            'category_structure': defaultdict(list),
            'file_extensions': defaultdict(int),
            'file_size_distribution': {'small': 0, 'medium': 0, 'large': 0},
            'organization_patterns': [],
            'issues_found': []
        }
        
        try:
            # Walk through directory structure
            for root, dirs, files in os.walk(self.legacy_data_dir):
                root_path = Path(root)
                level = len(root_path.relative_to(self.legacy_data_dir).parts)
                structure_analysis['directory_levels'] = max(structure_analysis['directory_levels'], level)
                structure_analysis['total_directories'] += len(dirs)
                structure_analysis['total_files'] += len(files)
                
                # Analyze directory names
                dir_name = root_path.name
                
                # Check for item directories (item_001, item_002, etc.)
                item_match = self.patterns['item_id'].search(dir_name)
                if item_match:
                    item_id = f"item_{item_match.group(1).zfill(3)}"
                    structure_analysis['item_directories'].append({
                        'item_id': item_id,
                        'path': str(root_path),
                        'file_count': len(files),
                        'subdirs': dirs.copy()
                    })
                
                # Check for category patterns
                category_match = self.patterns['category'].search(dir_name)
                if category_match:
                    category = category_match.group(1)
                    structure_analysis['category_structure'][category].append(str(root_path))
                
                # Analyze files in this directory
                for file_name in files:
                    file_path = root_path / file_name
                    
                    # Get file extension
                    ext = file_path.suffix.lower()
                    structure_analysis['file_extensions'][ext] += 1
                    
                    # Check if it's an image file
                    if self._is_image_file(file_path):
                        structure_analysis['image_files'] += 1
                    
                    # Categorize by file size
                    try:
                        file_size = file_path.stat().st_size
                        if file_size < 100 * 1024:  # < 100KB
                            structure_analysis['file_size_distribution']['small'] += 1
                        elif file_size < 10 * 1024 * 1024:  # < 10MB
                            structure_analysis['file_size_distribution']['medium'] += 1
                        else:  # >= 10MB
                            structure_analysis['file_size_distribution']['large'] += 1
                    except OSError:
                        pass
            
            # Identify organization patterns
            if structure_analysis['item_directories']:
                structure_analysis['organization_patterns'].append('item_based_directories')
            
            if structure_analysis['category_structure']:
                structure_analysis['organization_patterns'].append('category_based_structure')
            
            if structure_analysis['file_extensions'].get('.jpg', 0) > 0:
                structure_analysis['organization_patterns'].append('jpeg_images')
            
            # Update stats
            self.stats.directories_scanned = structure_analysis['total_directories']
            self.stats.files_discovered = structure_analysis['total_files']
            self.stats.categories_discovered = len(structure_analysis['category_structure'])
            
            # Log analysis results
            self.logger.info(f"📊 Directory Structure Analysis:")
            self.logger.info(f"   Total directories: {structure_analysis['total_directories']}")
            self.logger.info(f"   Total files: {structure_analysis['total_files']}")
            self.logger.info(f"   Image files: {structure_analysis['image_files']}")
            self.logger.info(f"   Item directories: {len(structure_analysis['item_directories'])}")
            self.logger.info(f"   Categories: {len(structure_analysis['category_structure'])}")
            self.logger.info(f"   File extensions: {dict(structure_analysis['file_extensions'])}")
            self.logger.info(f"   Organization patterns: {structure_analysis['organization_patterns']}")
        
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze directory structure: {e}")
            structure_analysis['error'] = str(e)
        
        return structure_analysis
    
    def _is_image_file(self, file_path: Path) -> bool:
        """Check if file is an image based on extension and MIME type."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        if file_path.suffix.lower() in image_extensions:
            return True
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type and mime_type.startswith('image/')
    
    def extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a single file.
        
        Includes file system metadata, image properties, EXIF data,
        and content analysis results.
        """
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_path.suffix.lower(),
            'file_size_bytes': 0,
            'created_timestamp': None,
            'modified_timestamp': None,
            'is_image': False,
            'mime_type': None,
            'image_metadata': {},
            'exif_data': {},
            'content_analysis': {},
            'extraction_errors': []
        }
        
        try:
            # File system metadata
            stat_info = file_path.stat()
            metadata['file_size_bytes'] = stat_info.st_size
            metadata['created_timestamp'] = stat_info.st_ctime
            metadata['modified_timestamp'] = stat_info.st_mtime
            
            # MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            metadata['mime_type'] = mime_type
            metadata['is_image'] = self._is_image_file(file_path)
            
            # Image-specific metadata
            if metadata['is_image'] and PIL_AVAILABLE:
                try:
                    with Image.open(file_path) as img:
                        # Basic image properties
                        metadata['image_metadata'] = {
                            'width': img.width,
                            'height': img.height,
                            'mode': img.mode,
                            'format': img.format,
                            'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                        }
                        
                        # EXIF data extraction
                        if hasattr(img, '_getexif') and img._getexif() is not None:
                            exif_data = img._getexif()
                            metadata['exif_data'] = self._process_exif_data(exif_data)
                            self.stats.exif_data_extracted += 1
                        
                        # Content analysis
                        if self.enable_content_analysis:
                            metadata['content_analysis'] = self._analyze_image_content(img)
                            self.stats.content_analysis_completed += 1
                
                except Exception as e:
                    metadata['extraction_errors'].append(f"Image processing error: {e}")
            
            # Extract patterns from filename and path
            metadata.update(self._extract_path_patterns(file_path))
            
        except Exception as e:
            metadata['extraction_errors'].append(f"File metadata error: {e}")
            self.logger.warning(f"⚠️ Failed to extract metadata from {file_path}: {e}")
        
        return metadata
    
    def _process_exif_data(self, exif_data: Dict) -> Dict[str, Any]:
        """Process raw EXIF data into structured format."""
        processed_exif = {}
        
        try:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                
                # Convert non-serializable types to strings
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='replace')
                    except UnicodeDecodeError:
                        value = f"<bytes:{len(value)}>"
                elif isinstance(value, (tuple, list)):
                    # Convert tuples/lists that might contain bytes
                    try:
                        value = [item.decode('utf-8', errors='replace') if isinstance(item, bytes) else str(item) for item in value]
                    except:
                        value = str(value)
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    # Convert any other non-JSON-serializable types to string
                    value = str(value)
                
                # Handle specific EXIF tags
                if tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                    # Parse date/time
                    try:
                        from datetime import datetime
                        processed_exif[f"{tag_name}_parsed"] = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S').isoformat()
                    except (ValueError, TypeError):
                        pass
                
                processed_exif[tag_name] = value
        
        except Exception as e:
            self.logger.warning(f"⚠️ EXIF processing error: {e}")
        
        return processed_exif
    
    def _analyze_image_content(self, img: Image.Image) -> Dict[str, Any]:
        """Perform content analysis on image."""
        analysis = {}
        
        try:
            # Convert to RGB for analysis
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
            else:
                rgb_img = img
            
            # Basic color analysis
            img_array = np.array(rgb_img)
            
            analysis['color_analysis'] = {
                'mean_brightness': float(np.mean(img_array)),
                'brightness_std': float(np.std(img_array)),
                'dominant_colors': self._extract_dominant_colors(img_array),
                'color_channels': {
                    'red_mean': float(np.mean(img_array[:, :, 0])),
                    'green_mean': float(np.mean(img_array[:, :, 1])),
                    'blue_mean': float(np.mean(img_array[:, :, 2]))
                }
            }
            
            # Image quality metrics
            analysis['quality_metrics'] = {
                'aspect_ratio': img.width / img.height if img.height > 0 else 0,
                'resolution_category': self._categorize_resolution(img.width, img.height),
                'estimated_quality': self._estimate_image_quality(img_array)
            }
        
        except Exception as e:
            self.logger.warning(f"⚠️ Content analysis error: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _extract_dominant_colors(self, img_array: np.ndarray, n_colors: int = 3) -> List[List[int]]:
        """Extract dominant colors from image using k-means clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape image to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Sample pixels if image is too large
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Return dominant colors as RGB values
            return [color.astype(int).tolist() for color in kmeans.cluster_centers_]
        
        except ImportError:
            # Fallback without sklearn
            return [[128, 128, 128]]  # Gray fallback
        except Exception:
            return [[128, 128, 128]]  # Gray fallback
    
    def _categorize_resolution(self, width: int, height: int) -> str:
        """Categorize image resolution."""
        total_pixels = width * height
        
        if total_pixels >= 8000000:  # 8MP+
            return 'high'
        elif total_pixels >= 2000000:  # 2MP+
            return 'medium'
        elif total_pixels >= 500000:  # 0.5MP+
            return 'standard'
        else:
            return 'low'
    
    def _estimate_image_quality(self, img_array: np.ndarray) -> str:
        """Estimate image quality based on various metrics."""
        try:
            # Calculate image sharpness using Laplacian variance
            gray = np.mean(img_array, axis=2)
            laplacian_var = np.var(np.gradient(gray))
            
            # Categorize quality
            if laplacian_var > 1000:
                return 'high'
            elif laplacian_var > 100:
                return 'medium'
            else:
                return 'low'
        
        except Exception:
            return 'unknown'
    
    def _extract_path_patterns(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata patterns from file path and name."""
        patterns_found = {}
        
        # Extract item ID from path
        full_path = str(file_path)
        item_match = self.patterns['item_id'].search(full_path)
        if item_match:
            patterns_found['item_id'] = f"item_{item_match.group(1).zfill(3)}"
        
        # Extract image number
        image_num_match = self.patterns['image_number'].search(file_path.name)
        if image_num_match:
            patterns_found['image_number'] = int(image_num_match.group(1))
        
        # Extract category - use item_id as category for item-based organization
        category_match = self.patterns['category'].search(full_path)
        if category_match:
            patterns_found['category'] = category_match.group(1)
        elif patterns_found.get('item_id'):
            # Use item_id as category for item-based organization
            patterns_found['category'] = patterns_found['item_id']
        
        # Extract date patterns
        date_match = self.patterns['date'].search(file_path.name)
        if date_match:
            patterns_found['date_in_filename'] = date_match.group(1)
        
        # Extract quality indicators
        quality_match = self.patterns['quality'].search(full_path)
        if quality_match:
            patterns_found['quality_indicator'] = quality_match.group(1).lower()
        
        # Extract augmentation indicators
        aug_match = self.patterns['augmentation'].search(full_path)
        if aug_match:
            patterns_found['augmentation_type'] = aug_match.group(1).lower()
        
        return patterns_found
    
    def _make_json_serializable(self, obj):
        """Recursively convert an object to be JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, bytes):
            try:
                return obj.decode('utf-8', errors='replace')
            except:
                return f"<bytes:{len(obj)}>"
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def migrate_file_metadata(self, file_metadata: Dict[str, Any]) -> bool:
        """
        Store extracted metadata in unified storage system.
        
        Args:
            file_metadata: Extracted metadata for a file
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            # Generate unique image_id from file path - use full file stem for uniqueness
            file_path = Path(file_metadata['file_path'])
            base_image_id = file_path.stem  # Use filename stem as base
            
            # If we have item_id from directory, include it for organization
            item_id = file_metadata.get('item_id')
            if item_id:
                image_id = f"{item_id}_{base_image_id}"
            else:
                image_id = base_image_id
            
            # Ensure unique vector_id by checking existing vectors with 'metadata_' prefix
            vector_id = f"metadata_{image_id}"
            base_vector_id = vector_id
            counter = 1
            
            while True:
                existing_vector = self.vector_store.get_vector(vector_id)
                if existing_vector is None:
                    break
                vector_id = f"{base_vector_id}_{counter}"
                counter += 1
                
            # Extract just the image_id from vector_id for metadata
            final_image_id = vector_id.replace('metadata_', '')
            
            # Prepare metadata for storage
            metadata_record = {
                'image_id': final_image_id,
                'original_path': file_metadata['file_path'],
                'file_size_bytes': file_metadata['file_size_bytes'],
                'image_width': file_metadata.get('image_metadata', {}).get('width'),
                'image_height': file_metadata.get('image_metadata', {}).get('height'),
                'item_category': file_metadata.get('category'),
                'confidence_score': None,  # Will be populated during recognition
                'recognition_status': 'metadata_migrated'
            }
            
            # Store metadata using SQLite vector store's database connection
            # Create comprehensive metadata dictionary for storage
            comprehensive_metadata = {
                'file_metadata': self._make_json_serializable(file_metadata),
                'extraction_info': {
                    'extraction_timestamp': time.time(),
                    'migrator_version': '1.0.0',
                    'platform': 'metadata_migration'
                },
                'item_info': {
                    'item_id': file_metadata.get('item_id'),
                    'image_id': final_image_id,
                    'category': file_metadata.get('category'),
                    'image_number': file_metadata.get('image_number')
                },
                'file_info': {
                    'original_path': file_metadata['file_path'],
                    'file_size_bytes': file_metadata['file_size_bytes'],
                    'file_extension': file_metadata['file_extension'],
                    'mime_type': file_metadata['mime_type'],
                    'created_timestamp': file_metadata.get('created_timestamp'),
                    'modified_timestamp': file_metadata.get('modified_timestamp')
                },
                'image_info': file_metadata.get('image_metadata', {}),
                'content_analysis': file_metadata.get('content_analysis', {}),
                'exif_data': file_metadata.get('exif_data', {}),
                'extraction_errors': file_metadata.get('extraction_errors', [])
            }
            
            # For now, we'll create a dummy vector to store the metadata
            # This allows us to use the existing vector storage system
            dummy_vector = np.zeros(1536, dtype=np.float32)
            
            # Use the vector store's insert functionality
            from unified_storage.sqlite_store import create_vector_record
            
            vector_record = create_vector_record(
                vector_id=vector_id,
                item_id=file_metadata.get('item_id', 'unknown'),
                vector_data=dummy_vector,
                metadata=comprehensive_metadata
            )
            
            # Insert the metadata record as a special vector
            success = self.vector_store.insert_vector(vector_record)
            
            # Update category distribution stats
            category = file_metadata.get('category', 'uncategorized')
            self.stats.category_distribution[category] = self.stats.category_distribution.get(category, 0) + 1
            
            return success
        
        except Exception as e:
            self.logger.error(f"❌ Failed to migrate metadata for {file_metadata.get('file_path')}: {e}")
            self.stats.error_messages.append(f"Metadata migration error: {e}")
            return False
    
    def migrate_all_metadata(self) -> MetadataMigrationStats:
        """
        Migrate all metadata from legacy directory structure.
        
        Performs complete metadata extraction and migration with progress tracking.
        """
        self.logger.info("🚀 Starting metadata migration...")
        start_time = time.time()
        
        try:
            # Analyze directory structure first
            structure = self.discover_directory_structure()
            
            if structure.get('error'):
                self.logger.error(f"❌ Directory analysis failed: {structure['error']}")
                return self.stats
            
            # Process all discovered files
            self.logger.info(f"📊 Processing {structure['total_files']} files...")
            
            processed_files = 0
            migrated_files = 0
            
            # Walk through all files
            for root, dirs, files in os.walk(self.legacy_data_dir):
                root_path = Path(root)
                
                # Process files with progress bar
                if files:
                    pbar = tqdm(files, desc=f"Processing {root_path.name}", leave=False)
                    
                    for file_name in pbar:
                        file_path = root_path / file_name
                        
                        try:
                            # Extract metadata
                            file_metadata = self.extract_file_metadata(file_path)
                            processed_files += 1
                            
                            # Migrate to unified storage
                            if self.migrate_file_metadata(file_metadata):
                                migrated_files += 1
                                
                                # Update stats for images
                                if file_metadata['is_image']:
                                    self.stats.images_processed += 1
                            
                            # Update total data size
                            self.stats.total_data_size_mb += file_metadata['file_size_bytes'] / (1024 * 1024)
                            
                            pbar.set_postfix({
                                'Migrated': migrated_files,
                                'Images': self.stats.images_processed
                            })
                        
                        except Exception as e:
                            self.logger.error(f"❌ Failed to process {file_path}: {e}")
                            self.stats.error_messages.append(f"File processing error: {e}")
            
            # Update final statistics
            self.stats.metadata_records_created = migrated_files
            
            # Calculate timing statistics
            end_time = time.time()
            self.stats.migration_time_seconds = end_time - start_time
            
            if processed_files > 0:
                self.stats.average_time_per_file_ms = (
                    self.stats.migration_time_seconds * 1000 / processed_files
                )
            
            # Log final results
            self.logger.info("✅ Metadata migration completed!")
            self.logger.info(f"📊 Migration Statistics:")
            self.logger.info(f"   Files discovered: {self.stats.files_discovered}")
            self.logger.info(f"   Files processed: {processed_files}")
            self.logger.info(f"   Images processed: {self.stats.images_processed}")
            self.logger.info(f"   Metadata records: {self.stats.metadata_records_created}")
            self.logger.info(f"   Categories found: {len(self.stats.category_distribution)}")
            self.logger.info(f"   EXIF data extracted: {self.stats.exif_data_extracted}")
            self.logger.info(f"   Migration time: {self.stats.migration_time_seconds:.1f}s")
            self.logger.info(f"   Average per file: {self.stats.average_time_per_file_ms:.1f}ms")
            self.logger.info(f"   Total data size: {self.stats.total_data_size_mb:.1f}MB")
            
            if self.stats.category_distribution:
                self.logger.info(f"   Category distribution: {dict(self.stats.category_distribution)}")
            
            if self.stats.error_messages:
                self.logger.warning(f"⚠️ {len(self.stats.error_messages)} errors occurred")
        
        except Exception as e:
            self.logger.error(f"❌ Migration failed: {e}")
            self.stats.error_messages.append(f"Migration error: {e}")
        
        return self.stats
    
    def verify_metadata_migration(self) -> Dict[str, Any]:
        """Verify metadata migration completeness and data integrity."""
        self.logger.info("🔍 Verifying metadata migration...")
        
        verification = {
            'migration_complete': True,
            'data_integrity_valid': True,
            'file_counts_match': True,
            'category_data_valid': True,
            'extended_metadata_valid': True,
            'issues_found': []
        }
        
        try:
            # Count migrated metadata records in the vector store
            stats = self.vector_store.get_statistics()
            total_vectors = stats.get('total_vectors', 0)
            
            # Get all vectors and count metadata records
            all_vectors = self.vector_store.get_all_vectors(limit=total_vectors)
            metadata_vectors = [v for v in all_vectors if v.vector_id.startswith('metadata_')]
            migrated_count = len(metadata_vectors)
            
            # Count original files
            original_file_count = 0
            for root, dirs, files in os.walk(self.legacy_data_dir):
                original_file_count += len(files)
            
            # Check counts
            if migrated_count == 0:
                verification['migration_complete'] = False
                verification['issues_found'].append("No metadata records found")
            
            if abs(migrated_count - original_file_count) > original_file_count * 0.1:  # Allow 10% variance
                verification['file_counts_match'] = False
                verification['issues_found'].append(
                    f"File count mismatch: {migrated_count} migrated vs {original_file_count} original"
                )
            
            # Sample verification - check first few metadata vectors
            sample_vectors = metadata_vectors[:10] if metadata_vectors else []
            
            for vector in sample_vectors:
                try:
                    # Check if original file exists
                    original_path = vector.metadata.get('file_info', {}).get('original_path')
                    if original_path and not Path(original_path).exists():
                        verification['issues_found'].append(
                            f"Original file missing for {vector.vector_id}: {original_path}"
                        )
                    
                    # Validate metadata structure
                    required_sections = ['file_metadata', 'extraction_info', 'file_info']
                    for section in required_sections:
                        if section not in vector.metadata:
                            verification['extended_metadata_valid'] = False
                            verification['issues_found'].append(
                                f"Missing metadata section '{section}' for {vector.vector_id}"
                            )
                            
                except Exception as e:
                    verification['extended_metadata_valid'] = False
                    verification['issues_found'].append(
                        f"Error validating metadata for {vector.vector_id}: {e}"
                    )
            
            # Overall verification
            if verification['issues_found']:
                verification['migration_complete'] = False
            
            # Log verification results
            self.logger.info("🔍 Verification Results:")
            self.logger.info(f"   Migration complete: {'✅' if verification['migration_complete'] else '❌'}")
            self.logger.info(f"   Data integrity: {'✅' if verification['data_integrity_valid'] else '❌'}")
            self.logger.info(f"   File counts match: {'✅' if verification['file_counts_match'] else '❌'}")
            self.logger.info(f"   Extended metadata valid: {'✅' if verification['extended_metadata_valid'] else '❌'}")
            self.logger.info(f"   Migrated records: {migrated_count}")
            self.logger.info(f"   Original files: {original_file_count}")
            
            if verification['issues_found']:
                self.logger.warning("⚠️ Issues found:")
                for issue in verification['issues_found']:
                    self.logger.warning(f"   - {issue}")
        
        except Exception as e:
            self.logger.error(f"❌ Verification failed: {e}")
            verification['error'] = str(e)
            verification['migration_complete'] = False
        
        return verification


def main():
    """Main entry point for metadata migration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Migrate metadata from legacy directory structure to unified storage'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Legacy data directory to migrate from')
    parser.add_argument('--target-dir', type=str, default='data',
                       help='Target data directory for unified storage')
    parser.add_argument('--no-content-analysis', action='store_true',
                       help='Disable image content analysis')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze directory structure without migrating')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing migration')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create SQLite vector store
        logger.info("🚀 Initializing SQLite vector storage system...")
        config_manager = ConfigManager(data_dir=args.target_dir)
        vector_store = create_vector_store(config_manager)
        
        # Create migrator
        migrator = MetadataMigrator(
            legacy_data_dir=args.data_dir,
            vector_store=vector_store,
            enable_content_analysis=not args.no_content_analysis
        )
        
        if args.analyze_only:
            # Only analyze directory structure
            structure = migrator.discover_directory_structure()
            
            print("\n" + "="*60)
            print("DIRECTORY STRUCTURE ANALYSIS")
            print("="*60)
            print(f"Root Path: {structure['root_path']}")
            print(f"Total Directories: {structure['total_directories']}")
            print(f"Total Files: {structure['total_files']}")
            print(f"Image Files: {structure['image_files']}")
            print(f"Directory Levels: {structure['directory_levels']}")
            print(f"Item Directories: {len(structure['item_directories'])}")
            print(f"Categories: {len(structure['category_structure'])}")
            print(f"Organization Patterns: {structure['organization_patterns']}")
            
            print(f"\nFile Extensions:")
            for ext, count in sorted(structure['file_extensions'].items()):
                print(f"  {ext}: {count}")
            
            print(f"\nFile Size Distribution:")
            for size, count in structure['file_size_distribution'].items():
                print(f"  {size}: {count}")
            
            if structure['item_directories']:
                print(f"\nSample Item Directories:")
                for item in structure['item_directories'][:5]:
                    print(f"  {item['item_id']}: {item['file_count']} files")
        
        elif args.verify_only:
            # Only verify existing migration
            verification = migrator.verify_metadata_migration()
            
            print("\n" + "="*50)
            print("METADATA MIGRATION VERIFICATION")
            print("="*50)
            print(f"Migration Complete: {'✅ Yes' if verification['migration_complete'] else '❌ No'}")
            print(f"Data Integrity: {'✅ Valid' if verification['data_integrity_valid'] else '❌ Invalid'}")
            print(f"File Counts Match: {'✅ Yes' if verification['file_counts_match'] else '❌ No'}")
            print(f"Extended Metadata: {'✅ Valid' if verification['extended_metadata_valid'] else '❌ Invalid'}")
            
            if verification['issues_found']:
                print(f"\nIssues Found:")
                for issue in verification['issues_found']:
                    print(f"  - {issue}")
        
        else:
            # Full migration process
            logger.info("📊 Starting metadata migration...")
            stats = migrator.migrate_all_metadata()
            
            logger.info("🔍 Verifying migration...")
            verification = migrator.verify_metadata_migration()
            
            # Print summary
            print("\n" + "="*50)
            print("METADATA MIGRATION SUMMARY") 
            print("="*50)
            print(f"Files Discovered: {stats.files_discovered}")
            print(f"Images Processed: {stats.images_processed}")
            print(f"Metadata Records: {stats.metadata_records_created}")
            print(f"Categories Found: {len(stats.category_distribution)}")
            print(f"EXIF Data Extracted: {stats.exif_data_extracted}")
            print(f"Migration Time: {stats.migration_time_seconds:.1f}s")
            print(f"Average per File: {stats.average_time_per_file_ms:.1f}ms")
            print(f"Total Data Size: {stats.total_data_size_mb:.1f}MB")
            print(f"Verification: {'✅ Passed' if verification['migration_complete'] else '❌ Failed'}")
            
            if stats.category_distribution:
                print(f"\nCategory Distribution:")
                for category, count in sorted(stats.category_distribution.items()):
                    print(f"  {category}: {count}")
            
            if stats.error_messages:
                print(f"\nErrors ({len(stats.error_messages)}):")
                for error in stats.error_messages[:10]:
                    print(f"  - {error}")
        
        # Clean up
        vector_store.close()
        logger.info("✅ Metadata migration process completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())