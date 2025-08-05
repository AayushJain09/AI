"""
Augmentation Adapter

This module provides an adapter that preserves the existing AdvancedAugmentationPipeline
exactly as it is, while integrating it with the unified storage system.

CRITICAL: NO CHANGES to existing augmentation logic - only adapts the interface
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)


class AugmentationAdapter:
    """
    Adapter for existing AdvancedAugmentationPipeline
    
    CRITICAL: Preserves exact existing augmentation behavior
    - Uses exact same AdvancedAugmentationPipeline class
    - Maintains all existing parameters and strategy weights
    - Only adapts file I/O to work with unified storage
    """
    
    def __init__(self, augmentation_config: Dict[str, Any]):
        """
        Initialize adapter with existing augmentation configuration
        
        Args:
            augmentation_config: Configuration dictionary with exact same parameters
                                as existing system (augmentations_per_image=50, etc.)
        """
        self.config = augmentation_config
        
        # Import the existing augmentation system
        import sys
        parent_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from data_preparation.prepare import AdvancedAugmentationPipeline
        
        # Initialize existing pipeline with EXACT same config
        self.pipeline = AdvancedAugmentationPipeline(augmentation_config)
        
        # Create temporary directory for file I/O adapter
        self.temp_dir = Path(tempfile.mkdtemp(prefix="augmentation_adapter_"))
        self.temp_input_dir = self.temp_dir / "input"
        self.temp_output_dir = self.temp_dir / "output"
        self.temp_input_dir.mkdir(parents=True, exist_ok=True)
        self.temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AugmentationAdapter initialized with temp dir: {self.temp_dir}")
    
    def process_image_array(self, 
                           image_array: np.ndarray, 
                           item_id: str,
                           image_index: int = 0) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process a numpy image array using existing augmentation pipeline
        
        CRITICAL: Uses existing augment_single_image method exactly as is
        
        Args:
            image_array: Input image as numpy array (RGB)
            item_id: Item identifier
            image_index: Index of image within item
            
        Returns:
            List of (augmented_image_array, augmentation_params) tuples
        """
        try:
            # Step 1: Convert numpy array to temporary file for existing pipeline
            temp_input_path = self.temp_input_dir / f"{item_id}_{image_index}_input.jpg"
            
            # Convert numpy array to PIL Image and save
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_array)
            pil_image.save(temp_input_path, format='JPEG', quality=95)
            
            logger.debug(f"Saved input image to: {temp_input_path}")
            
            # Step 2: Use existing augment_single_image method (NO CHANGES)
            augmented_paths = self.pipeline.augment_single_image(
                image_path=str(temp_input_path),
                output_dir=self.temp_output_dir,
                item_id=item_id,
                image_idx=image_index
            )
            
            logger.info(f"Generated {len(augmented_paths)} augmented images for {item_id}_{image_index}")
            
            # Step 3: Load augmented images back to numpy arrays
            augmented_results = []
            
            for aug_path in augmented_paths:
                try:
                    # Load augmented image
                    aug_image = cv2.imread(aug_path)
                    if aug_image is not None:
                        # Convert BGR to RGB
                        aug_image_rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
                        
                        # Extract augmentation parameters from filename
                        filename = Path(aug_path).stem
                        # Parse: {item_id}_aug_{image_idx}_{aug_idx}
                        parts = filename.split('_')
                        aug_idx = int(parts[-1]) if parts[-1].isdigit() else 0
                        
                        # Create augmentation parameters record
                        aug_params = {
                            'augmentation_index': aug_idx,
                            'original_filename': temp_input_path.name,
                            'augmented_filename': Path(aug_path).name,
                            'strategy_weights': self.config.get('strategy_weights', self.pipeline.strategy_weights),
                            'quality_level': self.config.get('quality', 95),
                            'diversity_factor': self.config.get('diversity_factor', 0.8),
                            'augmentation_intensity': self.config.get('augmentation_intensity', 0.6),
                            'background_removal': self.config.get('use_background_removal', True),
                            'target_size': self.config.get('image_size', (1024, 1024))
                        }
                        
                        augmented_results.append((aug_image_rgb, aug_params))
                    
                except Exception as e:
                    logger.warning(f"Failed to load augmented image {aug_path}: {e}")
                    continue
            
            # Clean up temporary files
            self._cleanup_temp_files([temp_input_path] + augmented_paths)
            
            return augmented_results
            
        except Exception as e:
            logger.error(f"Failed to process image array: {e}")
            # Clean up on error
            self._cleanup_temp_files()
            raise
    
    def process_pil_image(self, 
                         pil_image: Image.Image, 
                         item_id: str,
                         image_index: int = 0) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process a PIL Image using existing augmentation pipeline
        
        Args:
            pil_image: Input PIL Image
            item_id: Item identifier
            image_index: Index of image within item
            
        Returns:
            List of (augmented_image_array, augmentation_params) tuples
        """
        # Convert PIL Image to numpy array
        image_array = np.array(pil_image)
        
        return self.process_image_array(image_array, item_id, image_index)
    
    def get_augmentation_statistics(self) -> Dict[str, Any]:
        """Get statistics from the existing augmentation pipeline"""
        return {
            'pipeline_stats': self.pipeline.stats,
            'config_used': self.config,
            'device': str(self.pipeline.device),
            'strategy_weights': self.pipeline.strategy_weights,
            'augmentations_per_image': self.pipeline.augmentations_per_image,
            'temp_dir': str(self.temp_dir)
        }
    
    def _cleanup_temp_files(self, specific_files: List[str] = None):
        """Clean up temporary files"""
        try:
            if specific_files:
                # Clean up specific files
                for file_path in specific_files:
                    try:
                        Path(file_path).unlink(missing_ok=True)
                    except Exception as e:
                        logger.debug(f"Failed to remove temp file {file_path}: {e}")
            else:
                # Clean up all temp files in directories
                for temp_path in [self.temp_input_dir, self.temp_output_dir]:
                    if temp_path.exists():
                        for file in temp_path.glob("*"):
                            try:
                                file.unlink(missing_ok=True)
                            except Exception as e:
                                logger.debug(f"Failed to remove temp file {file}: {e}")
                                
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
    
    def __del__(self):
        """Cleanup when adapter is destroyed"""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.debug(f"Failed to cleanup temp directory in destructor: {e}")


def create_exact_augmentation_config(augmentations_per_image: int = 50, 
                                   quality_level: int = 95) -> Dict[str, Any]:
    """
    Create augmentation configuration with EXACT same parameters as existing system
    
    Args:
        augmentations_per_image: Number of augmentations per image (PRESERVE: 50)
        quality_level: JPEG quality level (PRESERVE: 95)
        
    Returns:
        Configuration dictionary matching existing system exactly
    """
    return {
        # CRITICAL: Exact same parameters as existing system
        'augmentations_per_image': augmentations_per_image,  # PRESERVE: 50
        'image_size': (1024, 1024),  # PRESERVE: exact size
        'quality': quality_level,  # PRESERVE: 95
        
        # PRESERVE: Exact same strategy weights - DO NOT CHANGE
        'geometric_weight': 0.30,
        'perspective_weight': 0.25,
        'lighting_weight': 0.25,
        'noise_blur_weight': 0.15,
        'effects_weight': 0.05,
        
        # PRESERVE: Other existing parameters - DO NOT CHANGE
        'diversity_factor': 0.8,
        'augmentation_intensity': 0.6,
        'multi_strategy_prob': 0.3,
        'num_backgrounds': 25,
        'background_complexity': 0.5,
        'use_background_removal': True,
        'batch_processing': True,
        'parallel_workers': 4,
        'memory_efficient': True,
        'cache_backgrounds': True,
        'advanced_augmentation': True,
        'preserve_aspect_ratio': True
    }