"""
Enhanced Input Manager with Data Persistence Tracking

This module provides unified handling of all image input sources with guaranteed
data persistence and real-time item addition capabilities for GUI integration.

CRITICAL: Maintains 100% compatibility with existing augmentation system
"""

import os
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageSourceType(Enum):
    """Types of image input sources"""
    CAMERA = "camera"
    FILE = "file" 
    URL = "url"
    BATCH = "batch"
    MEMORY = "memory"


@dataclass
class ItemInformation:
    """
    Complete item information for GUI-based addition
    Contains all metadata needed for proper data persistence
    """
    item_id: str
    item_name: str
    category: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    creation_timestamp: datetime = field(default_factory=datetime.now)
    user_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQLite storage"""
        return {
            'item_id': self.item_id,
            'item_name': self.item_name,
            'category': self.category,
            'description': self.description,
            'tags': ','.join(self.tags) if self.tags else '',
            'creation_timestamp': self.creation_timestamp.isoformat(),
            'user_metadata': str(self.user_metadata)
        }


@dataclass 
class ImageSource:
    """
    Enhanced image source with persistence tracking
    Supports all input types while maintaining data lineage
    """
    source_type: ImageSourceType
    path_or_data: Union[Path, bytes, str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    item_info: Optional[ItemInformation] = None
    save_original: bool = True  # Always save original for 100% accuracy
    processing_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_hash(self) -> str:
        """Generate unique hash for the image data"""
        if self.source_type == ImageSourceType.FILE:
            # Hash file path and modification time for files
            path = Path(self.path_or_data)
            if path.exists():
                stat = path.stat()
                hash_data = f"{path.absolute()}_{stat.st_mtime}_{stat.st_size}".encode()
            else:
                hash_data = str(path.absolute()).encode()
        elif self.source_type == ImageSourceType.MEMORY:
            # Hash the numpy array data
            if isinstance(self.path_or_data, np.ndarray):
                hash_data = self.path_or_data.tobytes()
            else:
                hash_data = str(self.path_or_data).encode()
        else:
            # Hash the data directly
            if isinstance(self.path_or_data, bytes):
                hash_data = self.path_or_data
            else:
                hash_data = str(self.path_or_data).encode()
        
        return hashlib.sha256(hash_data).hexdigest()
    
    def load_image(self) -> Image.Image:
        """Load the image from the source"""
        try:
            if self.source_type == ImageSourceType.FILE:
                return Image.open(self.path_or_data)
            elif self.source_type == ImageSourceType.MEMORY:
                if isinstance(self.path_or_data, np.ndarray):
                    return Image.fromarray(self.path_or_data)
                elif isinstance(self.path_or_data, bytes):
                    from io import BytesIO
                    return Image.open(BytesIO(self.path_or_data))
            elif self.source_type == ImageSourceType.CAMERA:
                if isinstance(self.path_or_data, bytes):
                    from io import BytesIO
                    return Image.open(BytesIO(self.path_or_data))
            
            raise ValueError(f"Unsupported source type: {self.source_type}")
            
        except Exception as e:
            logger.error(f"Failed to load image from {self.source_type}: {e}")
            raise


@dataclass
class ProcessingMetadata:
    """Metadata about the processing applied to an image"""
    processing_mode: str
    platform_info: Dict[str, Any]
    augmentation_params: Optional[Dict[str, Any]] = None
    feature_extraction_params: Optional[Dict[str, Any]] = None
    processing_duration: float = 0.0
    gpu_used: bool = False
    memory_peak_mb: float = 0.0


@dataclass
class ProcessedImageWithPersistence:
    """
    Processed image with guaranteed SQLite storage
    Tracks all processing steps and maintains data lineage
    """
    image_data: np.ndarray  # Normalized image array
    original_source: ImageSource
    processing_metadata: ProcessingMetadata
    feature_vector: Optional[np.ndarray] = None  # 1536D features (PRESERVE existing dimension)
    augmentation_params: Optional[Dict[str, Any]] = None  # Track augmentation strategy used
    storage_id: Optional[str] = None  # SQLite storage reference
    integrity_checksum: Optional[str] = None  # Data integrity verification
    parent_item_id: Optional[str] = None  # Link to original item
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    def compute_checksum(self) -> str:
        """Compute integrity checksum for the processed image"""
        # Combine image data and feature vector for checksum
        checksum_data = self.image_data.tobytes()
        if self.feature_vector is not None:
            checksum_data += self.feature_vector.tobytes()
        
        return hashlib.sha256(checksum_data).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify data integrity using stored checksum"""
        if self.integrity_checksum is None:
            return False
        
        current_checksum = self.compute_checksum()
        return current_checksum == self.integrity_checksum
    
    def update_checksum(self):
        """Update the integrity checksum"""
        self.integrity_checksum = self.compute_checksum()


class UnifiedImageInputManager:
    """
    Unified handling of all image input sources with data persistence
    
    CRITICAL FEATURES:
    - Supports camera, file, batch, and memory inputs
    - Maintains data lineage and integrity
    - Prepares images for existing augmentation pipeline (NO CHANGES to augmentation)
    - Real-time processing for GUI integration
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.temp_dir = self.data_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'images_processed': 0,
            'items_created': 0,
            'processing_errors': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"UnifiedImageInputManager initialized with data_dir: {data_dir}")
    
    def create_item_from_images(self, 
                               images: List[Union[str, Path, Image.Image, np.ndarray]],
                               item_info: ItemInformation,
                               validate_images: bool = True) -> List[ImageSource]:
        """
        Create image sources for a new item from various input types
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            item_info: Complete item information
            validate_images: Whether to validate image data
            
        Returns:
            List of ImageSource objects ready for processing
        """
        start_time = time.time()
        image_sources = []
        
        try:
            for idx, image in enumerate(images):
                try:
                    # Determine source type and create ImageSource
                    if isinstance(image, (str, Path)):
                        # File path
                        path = Path(image)
                        if not path.exists():
                            logger.warning(f"Image file not found: {path}")
                            continue
                            
                        source = ImageSource(
                            source_type=ImageSourceType.FILE,
                            path_or_data=path,
                            item_info=item_info,
                            metadata={
                                'original_filename': path.name,
                                'file_size': path.stat().st_size,
                                'image_index': idx
                            }
                        )
                        
                    elif isinstance(image, Image.Image):
                        # PIL Image
                        source = ImageSource(
                            source_type=ImageSourceType.MEMORY,
                            path_or_data=np.array(image),
                            item_info=item_info,
                            metadata={
                                'format': image.format,
                                'mode': image.mode,
                                'size': image.size,
                                'image_index': idx
                            }
                        )
                        
                    elif isinstance(image, np.ndarray):
                        # Numpy array
                        source = ImageSource(
                            source_type=ImageSourceType.MEMORY,
                            path_or_data=image,
                            item_info=item_info,
                            metadata={
                                'shape': image.shape,
                                'dtype': str(image.dtype),
                                'image_index': idx
                            }
                        )
                        
                    else:
                        logger.warning(f"Unsupported image type: {type(image)}")
                        continue
                    
                    # Validate image if requested
                    if validate_images:
                        try:
                            test_image = source.load_image()
                            if test_image.size[0] < 32 or test_image.size[1] < 32:
                                logger.warning(f"Image too small: {test_image.size}")
                                continue
                        except Exception as e:
                            logger.warning(f"Image validation failed: {e}")
                            continue
                    
                    image_sources.append(source)
                    logger.debug(f"Created ImageSource for image {idx}: {source.source_type}")
                    
                except Exception as e:
                    logger.error(f"Failed to process image {idx}: {e}")
                    self.stats['processing_errors'] += 1
                    continue
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['images_processed'] += len(image_sources)
            self.stats['items_created'] += 1 if image_sources else 0
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"Created {len(image_sources)} image sources for item {item_info.item_id} in {processing_time:.2f}s")
            return image_sources
            
        except Exception as e:
            logger.error(f"Failed to create item from images: {e}")
            self.stats['processing_errors'] += 1
            raise
    
    def process_camera_input(self, 
                           camera_data: bytes,
                           item_info: ItemInformation) -> ImageSource:
        """
        Process camera input for real-time recognition
        
        Args:
            camera_data: Raw camera image data
            item_info: Item information
            
        Returns:
            ImageSource ready for processing
        """
        try:
            source = ImageSource(
                source_type=ImageSourceType.CAMERA,
                path_or_data=camera_data,
                item_info=item_info,
                metadata={
                    'capture_timestamp': datetime.now().isoformat(),
                    'data_size': len(camera_data)
                }
            )
            
            # Validate camera image
            test_image = source.load_image()
            logger.info(f"Camera image processed: {test_image.size}")
            
            self.stats['images_processed'] += 1
            return source
            
        except Exception as e:
            logger.error(f"Failed to process camera input: {e}")
            self.stats['processing_errors'] += 1
            raise
    
    def process_batch_upload(self, 
                           file_paths: List[str],
                           item_info: ItemInformation) -> List[ImageSource]:
        """
        Process multiple uploaded files for batch item creation
        
        Args:
            file_paths: List of file paths to process
            item_info: Item information
            
        Returns:
            List of ImageSource objects
        """
        try:
            # Convert file paths to Path objects
            paths = [Path(path) for path in file_paths]
            
            # Use existing create_item_from_images method
            return self.create_item_from_images(paths, item_info, validate_images=True)
            
        except Exception as e:
            logger.error(f"Failed to process batch upload: {e}")
            self.stats['processing_errors'] += 1
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            'average_processing_time': (
                self.stats['total_processing_time'] / max(1, self.stats['images_processed'])
            ),
            'success_rate': (
                (self.stats['images_processed'] - self.stats['processing_errors']) / 
                max(1, self.stats['images_processed'])
            )
        }
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")