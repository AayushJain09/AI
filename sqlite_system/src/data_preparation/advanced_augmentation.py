"""
Advanced Augmentation Pipeline with SQLite Storage
Preserves exact augmentation strategy and weights from the original proven system
Now stores results directly to SQLite instead of file system
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import json
import uuid
from datetime import datetime
import logging
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Background removal - optional dependency
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg not available - background removal will be skipped")
import yaml

# Import our SQLite storage and platform detection
from ..storage.sqlite_store import SQLiteVectorStore, FeatureRecord
from ..utils.platform_detector import get_platform_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedAugmentationPipeline:
    """
    GPU-accelerated diverse augmented dataset generator with SQLite storage
    Preserves exact augmentation logic and strategy weights from original proven system
    """
    
    def __init__(self, config: Dict, vector_store: SQLiteVectorStore):
        self.config = config
        self.vector_store = vector_store
        
        # === CORE AUGMENTATION CONTROLS (PRESERVED FROM ORIGINAL) ===
        self.augmentations_per_image = config.get('augmentations_per_image', 50)
        self.target_size = tuple(config.get('target_image_size', [1024, 1024]))
        self.quality = config.get('jpeg_quality', 95)
        
        # === CRITICAL: PRESERVED STRATEGY WEIGHTS (MUST SUM TO 1.0) ===
        self.strategy_weights = {
            'geometric': config.get('strategy_weights', {}).get('geometric', 0.30), # Rotation, flip, scale
            'perspective': config.get('strategy_weights', {}).get('perspective', 0.25), # Perspective, distortion
            'lighting': config.get('strategy_weights', {}).get('lighting', 0.25),  # Brightness, contrast
            'noise_blur': config.get('strategy_weights', {}).get('noise_blur', 0.15), # Noise, blur
            'effects': config.get('strategy_weights', {}).get('effects', 0.05)  # Sun flare, shadows
        }
        
        # Validate strategy weights sum to 1.0
        total_weight = sum(self.strategy_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Strategy weights sum to {total_weight:.3f}, normalizing to 1.0")
            for key in self.strategy_weights:
                self.strategy_weights[key] /= total_weight
        
        # === GENERALIZATION CONTROLS (PRESERVED) ===
        self.diversity_factor = config.get('diversity_factor', 0.8)  # 0.0=identical, 1.0=max variety
        self.augmentation_intensity = config.get('augmentation_intensity', 0.6)  # 0.0=subtle, 1.0=extreme
        self.multi_strategy_probability = config.get('multi_strategy_probability', 0.3)  # Mix multiple strategies
        
        # === BACKGROUND VARIETY ===
        self.num_synthetic_backgrounds = config.get('synthetic_backgrounds', 25)
        self.background_complexity = config.get('background_complexity', 0.5)  # 0.0=simple, 1.0=complex
        self.use_background_removal = config.get('background_removal', True)
        
        # === PERFORMANCE & SCALABILITY ===
        platform_config = get_platform_config()
        self.batch_processing = config.get('batch_processing', True)
        # Use cpu_cores if available, otherwise fallback to cpu_count or default of 4
        cpu_cores = platform_config.get('cpu_cores') or platform_config.get('cpu_count', 4)
        self.parallel_workers = min(config.get('parallel_workers', 4), cpu_cores)
        self.memory_efficient = config.get('memory_efficient', True)
        self.cache_backgrounds = config.get('cache_backgrounds', True)
        
        # GPU acceleration setup (preserved from original)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("🚀 Using CUDA GPU for augmentation acceleration")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("🚀 Using Apple MPS GPU for augmentation acceleration")
        else:
            self.device = torch.device('cpu')
            logger.info("⚠️ Using CPU for augmentation (consider GPU for 5x speedup)")
        
        # Statistics tracking
        self.stats = {
            'total_images_processed': 0,
            'total_augmentations_created': 0,
            'items_processed': 0,
            'gpu_accelerated': self.device.type != 'cpu',
            'backgrounds_generated': 0,
            'storage_method': 'SQLite'
        }
        
        # Pre-generate synthetic backgrounds for efficiency
        if self.cache_backgrounds:
            self.synthetic_backgrounds = self.generate_synthetic_backgrounds()
        else:
            self.synthetic_backgrounds = []
        
        # Setup GPU-accelerated transforms
        self._setup_gpu_transforms()
        
        # Log configuration for monitoring generalization
        self._log_configuration()
    
    def _log_configuration(self):
        """Log current configuration for generalization monitoring"""
        logger.info("🔧 ANTI-OVERFITTING CONFIGURATION:")
        logger.info(f"   📊 Augmentations per image: {self.augmentations_per_image}")
        logger.info(f"   🎯 Diversity factor: {self.diversity_factor:.2f}")
        logger.info(f"   ⚡ Intensity: {self.augmentation_intensity:.2f}")
        logger.info(f"   🔄 Multi-strategy prob: {self.multi_strategy_probability:.2f}")
        logger.info(f"   🏗️  Strategy weights: {dict(self.strategy_weights)}")
        logger.info(f"   🖼️  Backgrounds: {self.num_synthetic_backgrounds} (complexity: {self.background_complexity:.2f})")
        logger.info(f"   ⚙️  Workers: {self.parallel_workers}, GPU: {self.device.type}")
        logger.info(f"   🗄️  Storage: SQLite Vector Store")
    
    def _setup_gpu_transforms(self):
        """Setup GPU-accelerated PyTorch transforms for maximum speed (preserved)"""
        self.gpu_transforms = {
            'geometric': transforms.Compose([
                transforms.RandomRotation(degrees=45, fill=0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), 
                    shear=10, fill=0
                ),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.7),
            ]),
            'color': transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.3),
            ]),
            'advanced': transforms.Compose([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        }

# TODO: for GPU systems
    # def gpu_augment_batch(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
    #     """Apply GPU-accelerated augmentations to a batch of images"""
    #     if not images:
    #         return []
            
    #     # Stack images into batch tensor
    #     batch = torch.stack(images).to(self.device)
    #     augmented_batch = []
        
    #     # Apply different transform strategies
    #     strategies = ['geometric', 'color', 'advanced']
        
    #     for img in batch:
    #         strategy = random.choice(strategies)
            
    #         if self.use_advanced_augmentation:
    #             # Apply multiple strategies with random mixing
    #             if random.random() < 0.3:  # 30% chance for multi-strategy
    #                 for s in random.sample(strategies, 2):
    #                     img = self.gpu_transforms[s](img.unsqueeze(0)).squeeze(0)
    #             else:
    #                 img = self.gpu_transforms[strategy](img.unsqueeze(0)).squeeze(0)
    #         else:
    #             img = self.gpu_transforms[strategy](img.unsqueeze(0)).squeeze(0)
            
    #         augmented_batch.append(img)
        
    #     return augmented_batch
    
    def create_augmentation_strategies(self) -> Dict[str, A.Compose]:
        """
        Create diverse augmentation strategies with configurable intensity for optimal generalization
        Preserves exact augmentation parameters from original proven system
        """
        
        # Scale augmentation parameters based on intensity setting
        intensity = self.augmentation_intensity
        
        strategies = {}
        
        # === GEOMETRIC TRANSFORMATIONS (30% weight) ===
        # Essential for viewpoint invariance
        strategies['geometric'] = A.Compose([
            A.RandomRotate90(p=0.5 * self.diversity_factor),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3 * self.diversity_factor),
            A.Transpose(p=0.4 * self.diversity_factor),
            A.ShiftScaleRotate(
                shift_limit=0.1 * intensity,
                scale_limit=0.2 * intensity,
                rotate_limit=int(45 * intensity),
                border_mode=cv2.BORDER_REFLECT,
                p=0.8
            ),
            A.Resize(self.target_size[0], self.target_size[1])
        ])
        
        # === PERSPECTIVE AND DISTORTION (25% weight) ===  
        # Critical for 3D generalization
        strategies['perspective'] = A.Compose([
            A.Perspective(scale=(0.05 * intensity, 0.15 * intensity), p=0.7),
            A.OpticalDistortion(distort_limit=0.3 * intensity, p=0.5),
            A.GridDistortion(distort_limit=0.2 * intensity, p=0.5),
            A.Resize(self.target_size[0], self.target_size[1])
        ])
        
        # === LIGHTING VARIATIONS (25% weight) ===
        # Robust to different environments
        strategies['lighting'] = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3 * intensity, 
                contrast_limit=0.3 * intensity, 
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(max(50, 100-50*intensity), min(150, 100+50*intensity)), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=int(20 * intensity),
                sat_shift_limit=int(30 * intensity),
                val_shift_limit=int(20 * intensity),
                p=0.6
            ),
            A.Resize(self.target_size[0], self.target_size[1])
        ])
        
        # === NOISE AND BLUR (15% weight) ===
        # Simulate camera/sensor variations
        strategies['noise_blur'] = A.Compose([
            A.GaussianBlur(blur_limit=(3, int(7 * intensity)), p=0.5),
            A.GaussNoise(var_limit=(10, int(50 * intensity)), p=0.5),
            A.ISONoise(
                color_shift=(0.01, 0.05 * intensity), 
                intensity=(0.1, 0.5 * intensity), 
                p=0.3
            ),
            A.MotionBlur(blur_limit=int(7 * intensity), p=0.3),
            A.Resize(self.target_size[0], self.target_size[1])
        ])
        
        # === ENVIRONMENTAL EFFECTS (5% weight) ===
        # Real-world conditions
        strategies['effects'] = A.Compose([
            A.RandomSunFlare(p=0.3 * self.diversity_factor),
            A.RandomShadow(p=0.3 * self.diversity_factor),
            A.RandomFog(p=0.2 * self.diversity_factor),
            A.ImageCompression(
                quality_lower=max(70, 90-20*intensity), 
                quality_upper=100, 
                p=0.5
            ),
            A.Resize(self.target_size[0], self.target_size[1])
        ])
        
        return strategies
    
    def select_augmentation_strategy(self, strategies: Dict[str, A.Compose]) -> A.Compose:
        """
        Select augmentation strategy based on configured weights for optimal distribution
        Preserves exact strategy selection from original system
        """
        strategy_names = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(weights)
        if total_weight != 1.0:
            weights = [w/total_weight for w in weights]
            
        # Weighted random selection
        selected_strategy = np.random.choice(strategy_names, p=weights)
        
        # Apply multi-strategy mixing for enhanced generalization
        if random.random() < self.multi_strategy_probability:
            # Mix two strategies
            second_strategy = np.random.choice(strategy_names, p=weights)
            if second_strategy != selected_strategy:
                # Create mixed strategy by combining transforms
                mixed_transforms = []
                mixed_transforms.extend(strategies[selected_strategy].transforms[:2])
                mixed_transforms.extend(strategies[second_strategy].transforms[:2])
                mixed_transforms.append(A.Resize(self.target_size[0], self.target_size[1]))
                return A.Compose(mixed_transforms)
        
        return strategies[selected_strategy]
    
    def generate_synthetic_backgrounds(self, num_backgrounds: int = None) -> List[np.ndarray]:
        """
        Generate diverse synthetic backgrounds with configurable complexity
        Preserves exact background generation from original system
        """
        if num_backgrounds is None:
            num_backgrounds = self.num_synthetic_backgrounds
            
        backgrounds = []
        creators = [
            self._create_solid_background,
            self._create_gradient_background,
            self._create_texture_background,
            self._create_pattern_background
        ]
        
        # Adjust creator selection based on complexity setting
        if self.background_complexity < 0.3:
            # Simple backgrounds - mostly solid and gradients
            creator_weights = [0.6, 0.3, 0.05, 0.05]
        elif self.background_complexity < 0.7:
            # Medium complexity - balanced mix
            creator_weights = [0.3, 0.3, 0.2, 0.2]
        else:
            # High complexity - more textures and patterns
            creator_weights = [0.1, 0.2, 0.35, 0.35]
            
        for i in range(num_backgrounds):
            creator = np.random.choice(creators, p=creator_weights)
            backgrounds.append(creator())
        
        self.stats['backgrounds_generated'] = len(backgrounds)
        logger.info(f"Generated {len(backgrounds)} synthetic backgrounds")
        return backgrounds

    def _create_gradient_background(self) -> np.ndarray:
        """Creates a two-color linear gradient background (preserved)"""
        h, w = self.target_size
        color1 = [random.randint(0, 255) for _ in range(3)]
        color2 = [random.randint(0, 255) for _ in range(3)]
        
        background = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            ratio = y / h
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            background[y, :] = [r, g, b]
            
        return background

    def _create_texture_background(self) -> np.ndarray:
        """Creates a procedural texture background using simplified noise (preserved)"""
        h, w = self.target_size
        background = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate simplified noise for each channel
        for i in range(3):
            noise = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            background[:, :, i] = noise
            
        return background

    def _create_pattern_background(self) -> np.ndarray:
        """Creates a simple geometric pattern background (preserved)"""
        h, w = self.target_size
        background = self._create_solid_background()
        
        pattern_type = random.choice(['circles', 'lines', 'rects'])
        num_shapes = random.randint(20, 50)
        
        for _ in range(num_shapes):
            color = [random.randint(0, 255) for _ in range(3)]
            thickness = random.randint(1, 5)
            
            if pattern_type == 'circles':
                center = (random.randint(0, w), random.randint(0, h))
                radius = random.randint(10, 100)
                cv2.circle(background, center, radius, color, thickness)
            elif pattern_type == 'lines':
                pt1 = (random.randint(0, w), random.randint(0, h))
                pt2 = (random.randint(0, w), random.randint(0, h))
                cv2.line(background, pt1, pt2, color, thickness)
            elif pattern_type == 'rects':
                pt1 = (random.randint(0, w), random.randint(0, h))
                pt2 = (random.randint(pt1[0], w), random.randint(pt1[1], h))
                cv2.rectangle(background, pt1, pt2, color, -1)
                
        return background

    def _create_solid_background(self) -> np.ndarray:
        """Creates a solid color background (preserved)"""
        h, w = self.target_size
        color = [random.randint(100, 255) for _ in range(3)]
        return np.full((h, w, 3), color, dtype=np.uint8)

    def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes background using rembg and returns foreground and mask
        Preserves exact background removal from original system
        """
        # rembg expects RGB, OpenCV uses BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Remove background (disable alpha matting to avoid Cholesky warnings)
        if not REMBG_AVAILABLE:
            # Fallback: return original image without background removal
            logger.debug("Background removal not available - using original image")
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            return image, mask
        
        foreground_rgba = remove(image_rgb)
        
        # Separate foreground and mask
        foreground = cv2.cvtColor(foreground_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        mask = foreground_rgba[:, :, 3]
        
        return foreground, mask

    def create_composite_image(self, foreground: np.ndarray, background: np.ndarray, 
                             mask: np.ndarray) -> np.ndarray:
        """
        Composites a foreground onto a background using a mask
        Preserves exact compositing logic from original system
        """
        # Ensure background is the correct size
        background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
        
        # Convert mask to 3 channels for blending
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        # Blend images
        composite = (foreground.astype(np.float32) * mask_3ch + 
                     background.astype(np.float32) * (1 - mask_3ch))
        
        return composite.astype(np.uint8)

    def create_augmented_image_record(self, image_path: str, item_id: str, 
                                    image_idx: int, aug_idx: int) -> Optional[Dict]:
        """
        Create a single augmented image and return its metadata for SQLite storage
        """
        try:
            # Read image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading {image_path}: {e}")
            return None

        try:
            # 1. Remove background if enabled
            if self.use_background_removal:
                foreground, mask = self.remove_background(image)
            else:
                foreground = image
                mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
                
        except Exception as e:
            logger.error(f"Background removal failed for {image_path}: {e}")
            return None

        try:
            # 2. Create composite with a random synthetic background
            if self.cache_backgrounds and self.synthetic_backgrounds:
                background = random.choice(self.synthetic_backgrounds)
            else:
                # Generate background on-demand for memory efficiency
                creators = [self._create_solid_background, self._create_gradient_background, 
                           self._create_texture_background, self._create_pattern_background]
                background = random.choice(creators)()
            
            if self.use_background_removal:
                composite_image = self.create_composite_image(foreground, background, mask)
            else:
                composite_image = image
                
            # 3. Apply weighted augmentation strategy for optimal generalization
            strategies = self.create_augmentation_strategies()
            strategy = self.select_augmentation_strategy(strategies)
            augmented = strategy(image=composite_image)
            augmented_image = augmented['image']

            # 4. Generate unique image ID for this augmentation
            aug_image_id = f"{item_id}_aug_{image_idx}_{aug_idx}_{uuid.uuid4().hex[:8]}"
            
            # 5. Create augmentation parameters record
            augmentation_params = {
                'original_path': image_path,
                'augmentation_index': aug_idx,
                'image_index': image_idx,
                'strategy_used': 'mixed' if random.random() < self.multi_strategy_probability else 'single',
                'background_removal_used': self.use_background_removal,
                'composite_created': True,
                'target_size': list(self.target_size),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # 6. Convert to RGB for consistency (many models expect RGB)
            augmented_image_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
            
            return {
                'image_id': aug_image_id,
                'item_id': item_id,
                'image_data': augmented_image_rgb,  # RGB format
                'augmentation_params': augmentation_params,
                'original_path': image_path
            }
            
        except Exception as e:
            logger.error(f"Error augmenting {image_path} (aug {aug_idx}): {e}")
            return None

    def store_augmented_image_with_features(self, aug_record: Dict, feature_extractor=None) -> bool:
        """
        Store augmented image and extract features using the multimodal feature extractor
        """
        try:
            # Import here to avoid circular imports
            from ..storage.sqlite_store import FeatureRecord
            import time
            
            # Use provided feature extractor (reuse existing models)
            if feature_extractor is None:
                raise ValueError("Feature extractor must be provided to avoid model loading overhead")
            
            # Extract features from the augmented image data
            features = feature_extractor.extract_features_from_image(aug_record['image_data'])
            
            if features is None:
                logger.error(f"Failed to extract features for {aug_record['image_id']}")
                return False
            
            # Create combined features (preserved logic)
            clip_features = features['clip']
            dinov2_features = features['dinov2']
            combined_features = np.concatenate([clip_features, dinov2_features])
            
            # Store the augmented image as BLOB in the database
            import io
            from PIL import Image
            
            # Convert numpy array to PIL Image and then to bytes
            pil_image = Image.fromarray(aug_record['image_data'])
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
            
            # Create feature record
            record = FeatureRecord(
                image_id=aug_record['image_id'],
                item_id=aug_record['item_id'],
                image_path=aug_record['original_path'],
                clip_features=clip_features,
                dinov2_features=dinov2_features,
                combined_features=combined_features,
                augmentation_params=aug_record['augmentation_params'],
                extraction_timestamp=time.time(),
                image_hash=None  # Will be computed by the vector store
            )
            
            # Store features to SQLite (which also handles the images table)
            features_stored = self.vector_store.store_features(record)
            
            # Update the images table to store the actual augmented image data
            if features_stored:
                image_stored = self._store_image_data(
                    aug_record['image_id'],
                    image_bytes
                )
            else:
                image_stored = False
            
            success = features_stored and image_stored
            
            if success:
                logger.debug(f"✅ Stored augmented image and features for {aug_record['image_id']}")
            else:
                logger.error(f"❌ Failed to store data for {aug_record['image_id']}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error storing augmented image with features: {e}")
            return False

    def _store_image_data(self, image_id: str, image_bytes: bytes) -> bool:
        """
        Store the actual image data as BLOB in the images table
        """
        try:
            cursor = self.vector_store.connection.cursor()
            
            # Update the existing image record to include the actual image data
            cursor.execute('''
            UPDATE images 
            SET image_data = ?, image_type = 'augmented'
            WHERE image_id = ?
            ''', (image_bytes, image_id))
            
            self.vector_store.connection.commit()
            
            if cursor.rowcount > 0:
                logger.debug(f"✅ Stored image data for {image_id}")
                return True
            else:
                logger.warning(f"⚠️ No image record found for {image_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error storing image data for {image_id}: {e}")
            return False

    def _store_original_image_with_features(self, image_path: str, image_id: str, item_id: str, image_idx: int) -> bool:
        """
        Store original image with features (no augmentation applied)
        """
        try:
            # Read original image
            from PIL import Image
            import io
            
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            
            # Convert to bytes for storage
            img_byte_arr = io.BytesIO()
            original_image.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
            
            # Extract features from original image
            features = self._feature_extractor.extract_features_from_image(original_image)
            
            if features is None:
                logger.error(f"Failed to extract features for original image {image_id}")
                return False
            
            # Create combined features
            clip_features = features['clip']
            dinov2_features = features['dinov2']
            combined_features = np.concatenate([clip_features, dinov2_features])
            
            # Create feature record for original image
            from ..storage.sqlite_store import FeatureRecord
            import time
            
            record = FeatureRecord(
                image_id=image_id,
                item_id=item_id,
                image_path=image_path,
                clip_features=clip_features,
                dinov2_features=dinov2_features,
                combined_features=combined_features,
                augmentation_params=None,  # No augmentation for original
                extraction_timestamp=time.time(),
                image_hash=None
            )
            
            # Store features to SQLite
            features_stored = self.vector_store.store_features(record)
            
            # Store original image data
            if features_stored:
                image_stored = self._store_original_image_data(image_id, image_bytes)
            else:
                image_stored = False
            
            return features_stored and image_stored
            
        except Exception as e:
            logger.error(f"❌ Error storing original image {image_id}: {e}")
            return False

    def _store_original_image_data(self, image_id: str, image_bytes: bytes) -> bool:
        """
        Store original image data as BLOB in the images table
        """
        try:
            cursor = self.vector_store.connection.cursor()
            
            # Update the existing image record to include the actual image data
            cursor.execute('''
            UPDATE images 
            SET image_data = ?, image_type = 'original'
            WHERE image_id = ?
            ''', (image_bytes, image_id))
            
            self.vector_store.connection.commit()
            
            if cursor.rowcount > 0:
                logger.debug(f"✅ Stored original image data for {image_id}")
                return True
            else:
                logger.warning(f"⚠️ No image record found for {image_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error storing original image data for {image_id}: {e}")
            return False

    def process_item_to_sqlite(self, item_dir: Path, item_id: str = None) -> Dict:
        """
        Process all images for a single item and store directly to SQLite
        Replaces file-based storage with SQLite Vector Store
        """
        if item_id is None:
            item_id = item_dir.name
        
        # Check if this item already exists in SQLite
        existing_features = self.vector_store.get_item_features(item_id)
        if existing_features:
            expected_augmentations = len(existing_features)
            logger.info(f"Item {item_id} already processed - found {expected_augmentations} features in SQLite")
            self.stats['total_augmentations_created'] += expected_augmentations
            return {
                'status': 'skipped',
                'item_id': item_id,
                'reason': 'already_processed',
                'augmented_count': expected_augmentations
            }
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(item_dir.glob(f"*{ext}"))
            image_files.extend(item_dir.glob(f"*{ext.upper()}"))
            
        if len(image_files) == 0:
            logger.warning(f"No images found in {item_dir}")
            return {'status': 'failed', 'reason': 'no_images'}
            
        logger.info(f"Processing {len(image_files)} images for item {item_id}")
        
        # Add item to SQLite
        self.vector_store.add_item(item_id, {
            'source_directory': str(item_dir),
            'original_images': len(image_files),
            'processing_started': datetime.now().isoformat()
        })
        
        total_augmentations_created = 0
        total_originals_stored = 0
        
        # Process each original image
        for image_idx, image_file in enumerate(image_files):
            logger.info(f"Processing image {image_idx + 1}/{len(image_files)}: {image_file.name}")
            
            # Store original image with features first
            original_image_id = f"{item_id}_orig_{image_idx}_{uuid.uuid4().hex[:8]}"
            
            # Create feature extractor only once per item (not per image)
            if not hasattr(self, '_feature_extractor'):
                from ..feature_extraction.multimodal_extractor import MultiModalFeatureExtractor
                config = {'device': 'auto'}
                self._feature_extractor = MultiModalFeatureExtractor(config, self.vector_store)
                logger.info("✅ Feature extractor created and cached for reuse")
            
            # Store original image with features
            original_success = self._store_original_image_with_features(
                str(image_file), original_image_id, item_id, image_idx
            )
            
            if original_success:
                total_originals_stored += 1
                logger.debug(f"✅ Stored original image {image_idx + 1}/{len(image_files)}")
            else:
                logger.warning(f"⚠️ Failed to store original image {image_idx + 1}/{len(image_files)}")
            
            # Create augmented versions and extract features
            for aug_idx in range(self.augmentations_per_image):
                aug_record = self.create_augmented_image_record(
                    str(image_file), item_id, image_idx, aug_idx
                )
                
                if aug_record:
                    # Store the augmented image data and extract features
                    success = self.store_augmented_image_with_features(aug_record, self._feature_extractor)
                    if success:
                        total_augmentations_created += 1
                        logger.debug(f"Processed and stored augmentation {aug_idx + 1}/{self.augmentations_per_image}")
                    else:
                        logger.warning(f"Failed to store augmentation {aug_idx + 1}/{self.augmentations_per_image}")
        
        # Update statistics
        self.stats['total_images_processed'] += len(image_files)
        self.stats['total_augmentations_created'] += total_augmentations_created
        self.stats['items_processed'] += 1
        
        logger.info(f"✅ Processing complete for item {item_id}: {total_originals_stored} originals + {total_augmentations_created} augmentations = {total_originals_stored + total_augmentations_created} total images")
        
        return {
            'status': 'success',
            'item_id': item_id,
            'original_count': total_originals_stored,
            'augmented_count': total_augmentations_created, 
            'total_images_processed': total_originals_stored + total_augmentations_created,
            'storage_method': 'sqlite'
        }
    
    def process_dataset_to_sqlite(self, input_dir: Path) -> Dict:
        """
        Process entire dataset and store to SQLite
        Replaces HDF5-based processing with SQLite Vector Store
        """
        input_path = Path(input_dir)
        
        # Find all item directories
        item_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(item_dirs)} items to process with SQLite storage")
        
        results = []
        
        # Process each item
        for item_dir in tqdm(item_dirs, desc="Processing items to SQLite"):
            result = self.process_item_to_sqlite(item_dir)
            results.append(result)

        # Save overall statistics to SQLite
        self.stats['timestamp'] = datetime.now().isoformat()
        self.stats['success_count'] = sum(1 for r in results if r['status'] == 'success')
        self.stats['failure_count'] = sum(1 for r in results if r['status'] == 'failed')
        self.stats['skipped_count'] = sum(1 for r in results if r['status'] == 'skipped')
        
        # Store statistics in the vector store
        # Note: This could be expanded to use the recognition_stats table
        
        logger.info(f"🎉 SQLite Processing complete!")
        logger.info(f"Total items processed: {self.stats['items_processed']}")
        logger.info(f"Items skipped (already processed): {self.stats.get('skipped_count', 0)}")
        logger.info(f"Total images processed: {self.stats['total_images_processed']}")
        logger.info(f"Total augmentations created: {self.stats['total_augmentations_created']}")
        
        return {
            'statistics': self.stats,
            'results': results,
            'storage_method': 'sqlite'
        }


def create_augmentation_pipeline(config_path: str, vector_store: SQLiteVectorStore) -> AdvancedAugmentationPipeline:
    """
    Factory function to create augmentation pipeline with SQLite storage
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    augmentation_config = config.get('augmentation', {})
    
    return AdvancedAugmentationPipeline(augmentation_config, vector_store)


def main():
    """Main entry point for augmentation with SQLite storage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced augmentation pipeline with SQLite storage',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True, help='Input directory with item folders')
    parser.add_argument('--config', type=str, required=True, help='Configuration YAML file')
    parser.add_argument('--database', type=str, required=True, help='SQLite database path')
    
    args = parser.parse_args()
    
    # Create vector store
    vector_store = SQLiteVectorStore(args.database)
    
    # Create augmentation pipeline
    pipeline = create_augmentation_pipeline(args.config, vector_store)
    
    # Process dataset
    results = pipeline.process_dataset_to_sqlite(Path(args.input))
    
    print(f"✅ Processing complete!")
    print(f"   Success: {results['statistics']['success_count']}")
    print(f"   Failed: {results['statistics']['failure_count']}")
    print(f"   Skipped: {results['statistics']['skipped_count']}")
    print(f"   Storage: SQLite Vector Store")


if __name__ == "__main__":
    main()