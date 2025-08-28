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
import time
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

# Import our SQLite storage, platform detection, and color extraction
from ..storage.sqlite_store import SQLiteVectorStore, FeatureRecord
from ..utils.platform_detector import get_platform_config
from ..utils.color_extractor import ModernColorExtractor, create_color_extractor

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
        
        # === COLOR EXTRACTION SETUP ===
        color_config = config.get('color_extraction', {})
        color_config['remove_background'] = self.use_background_removal  # Sync with background removal setting
        self.color_extractor = create_color_extractor(color_config)
        self.extract_colors = config.get('extract_colors', True)  # Enable color extraction by default
        
        # GPU acceleration setup (preserved from original)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            # Optimize CUDA for augmentation workloads
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
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
        
        # Setup GPU-accelerated transforms and batch processing
        self._setup_gpu_transforms()
        self._setup_gpu_batch_processing()
        
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
        logger.info(f"   📦 GPU batch size: {getattr(self, 'gpu_batch_size', 'N/A')}")
        logger.info(f"   🎨 Color extraction: {'✅ Enabled' if self.extract_colors else '❌ Disabled'}")
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
    
    def _setup_gpu_batch_processing(self):
        """Setup GPU batch processing parameters for optimal performance"""
        if self.device.type == 'cuda':
            # Detect GPU memory for optimal batch sizing
            try:
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                # Conservative estimate: ~50-100MB per image for augmentation
                self.gpu_batch_size = min(8, max(2, gpu_memory_mb // 200))
                logger.info(f"🚀 CUDA GPU detected with {gpu_memory_mb}MB memory - using batch size {self.gpu_batch_size}")
            except:
                self.gpu_batch_size = 4  # Safe fallback
                logger.info("🚀 CUDA GPU detected - using default batch size 4")
        elif self.device.type == 'mps':
            # Apple Silicon - unified memory, be conservative
            self.gpu_batch_size = 4
            logger.info("🍎 Apple Silicon MPS detected - using batch size 4")
        else:
            self.gpu_batch_size = 2  # CPU fallback
            logger.info("💻 CPU processing - using batch size 2")
        
        # Enable memory optimization for GPU processing
        if self.device.type in ['cuda', 'mps']:
            self.enable_gpu_optimizations = True
            # Pre-allocate some GPU memory to avoid repeated allocations
            try:
                dummy_tensor = torch.randn(1, 3, *self.target_size, device=self.device)
                del dummy_tensor
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
            except:
                pass

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
            A.Affine(
                translate_percent={'x': (-0.1 * intensity, 0.1 * intensity), 'y': (-0.1 * intensity, 0.1 * intensity)},
                scale=(1.0 - 0.2 * intensity, 1.0 + 0.2 * intensity),
                rotate=(-45 * intensity, 45 * intensity),
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
            A.GaussianBlur(blur_limit=(3, max(5, int(7 * intensity) | 1)), p=0.5),  # Ensure odd numbers
            A.MultiplicativeNoise(multiplier=(0.9, 1.0 + 0.1 * intensity), p=0.5),  # Alternative noise that's more stable
            A.ISONoise(
                color_shift=(0.01, 0.05 * intensity), 
                intensity=(0.1, 0.5 * intensity), 
                p=0.3
            ),
            A.MotionBlur(blur_limit=max(5, int(7 * intensity) | 1), p=0.3),  # Ensure odd numbers
            A.Resize(self.target_size[0], self.target_size[1])
        ])
        
        # === ENVIRONMENTAL EFFECTS (5% weight) ===
        # Real-world conditions
        strategies['effects'] = A.Compose([
            A.RandomSunFlare(p=0.3 * self.diversity_factor),
            A.RandomShadow(p=0.3 * self.diversity_factor),
            A.RandomFog(p=0.2 * self.diversity_factor),
            A.ImageCompression(
                quality_range=(max(70, 90-20*intensity), 100), 
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
                                    image_idx: int, aug_idx: int, cached_background_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Create a single augmented image and return its metadata for SQLite storage
        """
        try:
            # Use cached background removal data if provided (MAJOR OPTIMIZATION)
            if cached_background_data is not None:
                # Reuse pre-computed background removal - saves ~10 seconds per augmentation!
                image = cached_background_data['original_image']
                foreground = cached_background_data['foreground'] 
                mask = cached_background_data['mask']
                logger.debug(f"🚀 Using cached background removal for aug {aug_idx}")
            else:
                # Read image and perform background removal (slow path)
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not read image: {image_path}")
                    return None
                    
                # 1. Remove background if enabled (this is the slow operation)
                if self.use_background_removal:
                    logger.debug(f"⏳ Removing background for {image_path} (this may take ~10 seconds)")
                    foreground, mask = self.remove_background(image)
                else:
                    foreground = image
                    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
                    
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
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
            
            # 7. Extract colors after background removal and augmentation
            color_data = {}
            if self.extract_colors:
                try:
                    # For color extraction, use the image after background removal but before heavy augmentation
                    # This gives more accurate colors of the actual object
                    color_source = cv2.cvtColor(foreground if self.use_background_removal else image, cv2.COLOR_BGR2RGB)
                    color_result = self.color_extractor.extract_colors_from_image(color_source)
                    color_data = {
                        'dominant_colors': color_result.get('dominant_color'),
                        'color_palette': color_result.get('color_palette', []),
                        'background_removed': color_result.get('background_removed', False)
                    }
                except Exception as e:
                    logger.warning(f"Color extraction failed for {aug_image_id}: {e}")
                    color_data = {
                        'dominant_colors': (128, 128, 128),  # Gray fallback
                        'color_palette': [(128, 128, 128)],
                        'background_removed': False
                    }
            
            return {
                'image_id': aug_image_id,
                'item_id': item_id,
                'image_data': augmented_image_rgb,  # RGB format
                'augmentation_params': augmentation_params,
                'original_path': image_path,
                'color_data': color_data  # Include extracted colors
            }
            
        except Exception as e:
            logger.error(f"Error augmenting {image_path} (aug {aug_idx}): {e}")
            return None

    def process_augmentation_batch(self, batch_records: List[Dict], feature_extractor) -> int:
        """
        Process a batch of augmentation records using GPU batch feature extraction
        Returns the number of successfully processed augmentations
        """
        if not batch_records:
            return 0
            
        try:
            from ..storage.sqlite_store import FeatureRecord
            import time
            import io
            from PIL import Image
            
            # Extract image data from all records in the batch
            batch_images = [rec['image_data'] for rec in batch_records]
            
            # Perform batch feature extraction (major GPU optimization)
            batch_features = self._extract_batch_features(batch_images, feature_extractor)
            
            if batch_features is None or len(batch_features) != len(batch_records):
                logger.error(f"Batch feature extraction failed for {len(batch_records)} images")
                return 0
            
            success_count = 0
            for i, (aug_record, features) in enumerate(zip(batch_records, batch_features)):
                if features is None:
                    continue
                    
                try:
                    # Create combined features
                    clip_features = features['clip']
                    dinov2_features = features['dinov2']
                    combined_features = np.concatenate([clip_features, dinov2_features])
                    
                    # Convert image to bytes for storage
                    pil_image = Image.fromarray(aug_record['image_data'])
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()
                    
                    # Create feature record (using only supported fields)
                    record = FeatureRecord(
                        image_id=aug_record['image_id'],
                        item_id=aug_record['item_id'],
                        image_path=aug_record['original_path'],
                        clip_features=clip_features,
                        dinov2_features=dinov2_features,
                        combined_features=combined_features,
                        augmentation_params=aug_record['augmentation_params'],
                        extraction_timestamp=time.time(),
                        image_hash=None
                    )
                    
                    # Store to SQLite with extended data
                    success = self.vector_store.store_features(record)
                    if success:
                        # Store additional image and color data
                        self._store_additional_image_data(
                            aug_record['image_id'],
                            image_bytes,
                            aug_record.get('color_data', {}),
                            'augmented'
                        )
                        success_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to store augmentation {i}: {e}")
                    continue
                    
            return success_count
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return 0
    
    def _extract_batch_features(self, batch_images: List[np.ndarray], feature_extractor) -> List[Dict]:
        """
        Extract features from a batch of images using optimized GPU processing
        This is the key optimization that reduces 30 individual GPU calls to ~4 batch calls
        """
        try:
            logger.info(f"🚀 Starting TRUE GPU batch processing for {len(batch_images)} images")
            
            # Use the new batch processing method in the feature extractor
            batch_features = feature_extractor.extract_features_batch(batch_images)
            
            logger.info(f"✅ GPU batch processing complete: {len(batch_features)} feature sets extracted")
            return batch_features
            
        except Exception as e:
            logger.error(f"Batch feature extraction failed: {e}")
            # Fallback to individual processing if batch fails
            logger.warning("Falling back to individual feature extraction")
            batch_features = []
            for image_data in batch_images:
                features = feature_extractor.extract_features_from_image(image_data)
                batch_features.append(features)
            return batch_features

    def _store_additional_image_data(self, image_id: str, image_bytes: bytes, color_data: Dict, image_type: str = 'augmented'):
        """
        Store additional image data (image bytes, colors) that aren't in FeatureRecord
        """
        try:
            import json
            cursor = self.vector_store.connection.cursor()
            
            # Update the existing image record with additional data
            cursor.execute('''
            UPDATE images SET 
                image_data = ?,
                image_type = ?,
                dominant_colors = ?,
                color_palette = ?,
                background_removed = ?
            WHERE image_id = ?
            ''', (
                image_bytes,
                image_type,
                json.dumps(color_data.get('dominant_colors')),
                json.dumps(color_data.get('color_palette', [])),
                color_data.get('background_removed', False),
                image_id
            ))
            
            self.vector_store.connection.commit()
            return True
            
        except Exception as e:
            logger.warning(f"Failed to store additional image data for {image_id}: {e}")
            return False

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
            
            # Update the images table to store the actual augmented image data with color info
            if features_stored:
                image_stored = self._store_image_data(
                    aug_record['image_id'],
                    image_bytes,
                    aug_record.get('color_data', {})
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

    def _store_image_data(self, image_id: str, image_bytes: bytes, color_data: Dict = None) -> bool:
        """
        Store the actual image data as BLOB in the images table with color information
        """
        try:
            cursor = self.vector_store.connection.cursor()
            
            # Prepare color data for JSON storage
            if color_data:
                dominant_colors_json = json.dumps(color_data.get('dominant_colors'))
                color_palette_json = json.dumps(color_data.get('color_palette', []))
                background_removed = color_data.get('background_removed', False)
            else:
                dominant_colors_json = None
                color_palette_json = None
                background_removed = False
            
            # Update the existing image record to include the actual image data and color info
            cursor.execute('''
            UPDATE images 
            SET image_data = ?, 
                image_type = 'augmented', 
                dominant_colors = ?,
                color_palette = ?,
                background_removed = ?
            WHERE image_id = ?
            ''', (image_bytes, dominant_colors_json, color_palette_json, background_removed, image_id))
            
            self.vector_store.connection.commit()
            
            if cursor.rowcount > 0:
                logger.debug(f"✅ Stored image data with colors for {image_id}")
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
        item_start_time = time.time()
        
        if item_id is None:
            item_id = item_dir.name
        
        logger.info(f"⏱️ PREPROCESSING TIMING - Starting item: {item_id}")
        
        # Check if this item already exists in SQLite
        check_start_time = time.time()
        existing_features = self.vector_store.get_item_features(item_id)
        check_time = (time.time() - check_start_time) * 1000
        logger.info(f"⏱️ SQLite check: {check_time:.2f}ms")
        
        if existing_features:
            expected_augmentations = len(existing_features)
            total_time = (time.time() - item_start_time) * 1000
            logger.info(f"Item {item_id} already processed - found {expected_augmentations} features in SQLite")
            logger.info(f"⏱️ Total time (skipped): {total_time:.2f}ms")
            self.stats['total_augmentations_created'] += expected_augmentations
            return {
                'status': 'skipped',
                'item_id': item_id,
                'reason': 'already_processed',
                'augmented_count': expected_augmentations,
                'timing_ms': total_time
            }
        
        # Find all images
        file_scan_start = time.time()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(item_dir.glob(f"*{ext}"))
            image_files.extend(item_dir.glob(f"*{ext.upper()}"))
        file_scan_time = (time.time() - file_scan_start) * 1000
        logger.info(f"⏱️ File scanning: {file_scan_time:.2f}ms")
            
        if len(image_files) == 0:
            total_time = (time.time() - item_start_time) * 1000
            logger.warning(f"No images found in {item_dir}")
            logger.info(f"⏱️ Total time (no images): {total_time:.2f}ms")
            return {'status': 'failed', 'reason': 'no_images', 'timing_ms': total_time}
            
        logger.info(f"Processing {len(image_files)} images for item {item_id}")
        
        # Add item to SQLite
        sqlite_add_start = time.time()
        self.vector_store.add_item(item_id, {
            'source_directory': str(item_dir),
            'original_images': len(image_files),
            'processing_started': datetime.now().isoformat()
        })
        sqlite_add_time = (time.time() - sqlite_add_start) * 1000
        logger.info(f"⏱️ SQLite item creation: {sqlite_add_time:.2f}ms")
        
        total_augmentations_created = 0
        total_originals_stored = 0
        
        # Initialize feature extractor timing
        extractor_init_start = time.time()
        if not hasattr(self, '_feature_extractor'):
            from ..feature_extraction.multimodal_extractor import MultiModalFeatureExtractor
            # Force GPU usage if available - prioritize performance
            device_name = 'auto'
            if torch.cuda.is_available():
                device_name = 'cuda'
                logger.info("🚀 Forcing CUDA device for feature extraction")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_name = 'mps'
                logger.info("🍎 Forcing MPS device for feature extraction")
            else:
                device_name = 'cpu'
                logger.warning("⚠️ No GPU available - using CPU for feature extraction")
                
            config = {'device': device_name}
            self._feature_extractor = MultiModalFeatureExtractor(config, self.vector_store)
            logger.info("✅ Feature extractor created and cached for reuse")
        extractor_init_time = (time.time() - extractor_init_start) * 1000
        logger.info(f"⏱️ Feature extractor initialization: {extractor_init_time:.2f}ms")
        
        # Detailed timing for each processing stage
        total_original_processing_time = 0
        total_augmentation_time = 0
        total_feature_extraction_time = 0
        total_storage_time = 0
        
        # Process each original image
        for image_idx, image_file in enumerate(image_files):
            image_start_time = time.time()
            logger.info(f"Processing image {image_idx + 1}/{len(image_files)}: {image_file.name}")
            
            # Store original image with features first
            original_image_id = f"{item_id}_orig_{image_idx}_{uuid.uuid4().hex[:8]}"
            
            original_processing_start = time.time()
            original_success = self._store_original_image_with_features(
                str(image_file), original_image_id, item_id, image_idx
            )
            original_processing_time = (time.time() - original_processing_start) * 1000
            total_original_processing_time += original_processing_time
            
            if original_success:
                total_originals_stored += 1
                logger.debug(f"✅ Stored original image {image_idx + 1}/{len(image_files)} ({original_processing_time:.1f}ms)")
            else:
                logger.warning(f"⚠️ Failed to store original image {image_idx + 1}/{len(image_files)} ({original_processing_time:.1f}ms)")
            
            # Create augmented versions and extract features using GPU batch processing
            augmentation_batch_start = time.time()
            
            # MAJOR OPTIMIZATION: Pre-compute background removal ONCE per image
            # This saves 30 x 10 seconds = 300 seconds per image!
            cached_background_data = None
            if self.use_background_removal:
                bg_removal_start = time.time()
                original_image = cv2.imread(str(image_file))
                if original_image is not None:
                    logger.info(f"⏳ Pre-computing background removal for {image_file.name} (one-time cost)")
                    foreground, mask = self.remove_background(original_image)
                    cached_background_data = {
                        'original_image': original_image,
                        'foreground': foreground,
                        'mask': mask
                    }
                    bg_removal_time = (time.time() - bg_removal_start) * 1000
                    logger.info(f"✅ Background removal cached in {bg_removal_time:.1f}ms - will reuse for all 30 augmentations")
                else:
                    logger.error(f"Failed to read image for background removal: {image_file}")
            
            # Process augmentations in GPU-optimized batches for massive speedup
            batch_records = []
            for aug_idx in range(self.augmentations_per_image):
                aug_creation_start = time.time()
                aug_record = self.create_augmented_image_record(
                    str(image_file), item_id, image_idx, aug_idx, cached_background_data
                )
                aug_creation_time = (time.time() - aug_creation_start) * 1000
                
                if aug_record:
                    batch_records.append(aug_record)
                    
                    # Process batch when it reaches GPU batch size or is the last augmentation
                    if len(batch_records) >= self.gpu_batch_size or aug_idx == self.augmentations_per_image - 1:
                        batch_start = time.time()
                        batch_success_count = self.process_augmentation_batch(batch_records, self._feature_extractor)
                        batch_time = (time.time() - batch_start) * 1000
                        
                        total_storage_time += batch_time
                        total_augmentations_created += batch_success_count
                        
                        logger.debug(f"Processed batch of {len(batch_records)} augmentations ({batch_time:.1f}ms, {batch_success_count}/{len(batch_records)} success)")
                        batch_records = []  # Clear batch for next iteration
            
            augmentation_batch_time = (time.time() - augmentation_batch_start) * 1000
            total_augmentation_time += augmentation_batch_time
            
            image_total_time = (time.time() - image_start_time) * 1000
            logger.info(f"⏱️ Image {image_idx + 1} complete: {image_total_time:.1f}ms (orig: {original_processing_time:.1f}ms, augs: {augmentation_batch_time:.1f}ms)")
        
        # Update statistics
        self.stats['total_images_processed'] += len(image_files)
        self.stats['total_augmentations_created'] += total_augmentations_created
        self.stats['items_processed'] += 1
        
        # Calculate final timing
        total_item_time = (time.time() - item_start_time) * 1000
        
        logger.info(f"✅ Processing complete for item {item_id}: {total_originals_stored} originals + {total_augmentations_created} augmentations = {total_originals_stored + total_augmentations_created} total images")
        
        # === DETAILED PREPROCESSING TIMING BREAKDOWN ===
        logger.info(f"📊 PREPROCESSING TIMING BREAKDOWN for {item_id}:")
        logger.info(f"   📁 File scanning: {file_scan_time:.1f}ms")
        logger.info(f"   🗄️  SQLite setup: {sqlite_add_time:.1f}ms") 
        logger.info(f"   🧠 Feature extractor init: {extractor_init_time:.1f}ms")
        logger.info(f"   📷 Original processing: {total_original_processing_time:.1f}ms")
        logger.info(f"   🔄 Augmentation creation: {total_augmentation_time:.1f}ms")
        logger.info(f"   💾 Storage operations: {total_storage_time:.1f}ms")
        logger.info(f"   ⏱️  TOTAL TIME: {total_item_time:.1f}ms ({total_item_time/1000:.2f}s)")
        logger.info(f"   📈 Avg per augmentation: {total_item_time/(total_augmentations_created or 1):.1f}ms")
        
        return {
            'status': 'success',
            'item_id': item_id,
            'original_count': total_originals_stored,
            'augmented_count': total_augmentations_created, 
            'total_images_processed': total_originals_stored + total_augmentations_created,
            'storage_method': 'sqlite',
            'timing_ms': {
                'total_time': total_item_time,
                'file_scanning': file_scan_time,
                'sqlite_setup': sqlite_add_time,
                'extractor_init': extractor_init_time,
                'original_processing': total_original_processing_time,
                'augmentation_creation': total_augmentation_time,
                'storage_operations': total_storage_time,
                'avg_per_augmentation': total_item_time/(total_augmentations_created or 1)
            }
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