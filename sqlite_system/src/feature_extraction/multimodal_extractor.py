"""
Multi-Modal Feature Extraction with SQLite Storage
Preserves exact CLIP + DINOv2 architecture and feature processing from original system
Now stores features directly to SQLite instead of HDF5
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import clip
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import uuid
import time
from tqdm import tqdm
import logging
from sklearn.preprocessing import normalize
import yaml

# Import our SQLite storage and platform detection
from ..storage.sqlite_store import SQLiteVectorStore, FeatureRecord
from ..utils.platform_detector import get_platform_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalFeatureExtractor:
    """
    Multi-modal feature extraction using CLIP + DINOv2 with SQLite storage
    Preserves exact feature extraction logic from original proven system
    """
    
    def __init__(self, config: Dict, vector_store: SQLiteVectorStore):
        self.config = config
        self.vector_store = vector_store
        
        # Platform-specific optimization
        platform_config = get_platform_config()
        self.device_name = config.get('device', 'auto')
        
        if self.device_name == 'auto':
            self.device_name = platform_config.get('feature_extraction_device', 'cpu')
        
        # Set device (preserved from original)
        if self.device_name == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif self.device_name == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            
        logger.info(f"🖥️ Feature extraction device: {self.device}")
        
        # Load models with exact same architecture as original
        self._load_models()
        
        # Define preprocessing (preserved from original)
        self._setup_preprocessing()
        
        # Performance tracking
        self.stats = {
            'features_extracted': 0,
            'items_processed': 0,
            'avg_extraction_time_ms': 0.0,
            'device_used': str(self.device),
            'clip_model': self.config.get('clip_model', 'ViT-L/14'),
            'dinov2_model': self.config.get('dinov2_model', 'dinov2_vitb14')
        }
        
    def _load_models(self):
        """
        Load optimized feature extraction models with exact preservation:
        - CLIP: ViT-L/14 (768 dimensions native)
        - DINOv2: dinov2_vitb14 (768 dimensions native)
        Total: 1536 dimensions for maximum accuracy and speed
        """
        
        # === CLIP MODEL (CORE MODEL #1 - ALWAYS LOADED) ===
        logger.info("🚀 Loading CLIP model (optimized 768-dim vision-language features)...")
        
        try:
            # Temporarily disable SSL verification for CLIP download (preserved from original)
            import ssl
            import urllib.request
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))
            
            # Load exact same CLIP variant for 768 dimensions (preserved)
            clip_variant = self.config.get('clip_model', 'ViT-L/14')
            self.clip_model, self.clip_preprocess = clip.load(clip_variant, device=self.device)
            self.clip_model.eval()
            
            # Verify output dimensions (preserved validation)
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_output = self.clip_model.encode_image(dummy_input)
                actual_clip_dims = dummy_output.shape[1]
                logger.info(f"✅ CLIP {clip_variant} loaded: {actual_clip_dims} native dims (using full 768)")
                
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise
        
        # === DINOv2 MODEL (CORE MODEL #2 - ALWAYS LOADED) ===
        logger.info("🔥 Loading DINOv2 model (optimized 768-dim self-supervised features)...")
        
        try:
            # Use exact same DINOv2 variant for 768 dimensions (preserved)
            dinov2_variant = self.config.get('dinov2_model', 'dinov2_vitb14')
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_variant, trust_repo=True)
            self.dinov2 = self.dinov2.to(self.device)
            self.dinov2.eval()
            
            # Verify output dimensions (preserved validation)
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_output = self.dinov2(dummy_input)
                actual_dino_dims = dummy_output.shape[1]
                logger.info(f"✅ DINOv2 {dinov2_variant} loaded: {actual_dino_dims} native dims (using full 768)")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load DINOv2: {e}. Proceeding with CLIP-only mode")
            self.dinov2 = None
        
        # === REMOVED MODELS FOR OPTIMIZATION (PRESERVED) ===
        self.resnet = None
        self.efficientnet = None
        logger.info("🗑️ ResNet and EfficientNet REMOVED for maximum performance")
        logger.info("🎯 Optimized Architecture: CLIP (768) + DINOv2 (768) = 1536 total dimensions")
        
        # Initialize feature dimensions (preserved)
        self.target_clip_dims = 768      # Use native CLIP ViT-L/14 dimensions
        self.target_dinov2_dims = 768    # Use native DINOv2-base dimensions
        
        logger.info(f"🔧 Feature dimensions set: CLIP={self.target_clip_dims}, DINOv2={self.target_dinov2_dims} (native, no compression)")
        
    def _setup_preprocessing(self):
        """Setup preprocessing pipelines for different models (preserved)"""
        # Standard preprocessing for non-CLIP models
        self.standard_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_clip_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract optimized CLIP features (768 dimensions - native ViT-L/14)
        Preserves exact feature extraction from original system
        """
        with torch.no_grad():
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_input)
            
            # Use native 768 dimensions (no compression needed) - preserved
            # ViT-L/14 naturally outputs 768-dimensional features
            
            # Normalize for optimal similarity computation (preserved)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
    
    def extract_dinov2_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract optimized DINOv2 self-supervised features (768 dimensions - native DINOv2-base)
        Preserves exact feature extraction from original system
        
        DINOv2 provides complementary features to CLIP:
        - CLIP: Multi-modal (vision + language) understanding  
        - DINOv2: Pure visual self-supervised features with excellent fine-grained recognition
        
        Returns 768-dimensional feature vector from DINOv2-base (native dimensions)
        """
        if self.dinov2 is None:
            return np.zeros(768)  # Return zeros if DINOv2 failed to load
        
        with torch.no_grad():
            # DINOv2 expects normalized RGB images of size 224x224 (preserved)
            dinov2_transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            image_input = dinov2_transform(image).unsqueeze(0).to(self.device)
            features = self.dinov2(image_input)
            
            # Use native 768 dimensions (no compression needed) - preserved
            # DINOv2-base naturally outputs 768-dimensional features
            
            # Normalize for optimal similarity computation (preserved)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
    
    def extract_features_from_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract optimized features from an image using CLIP + DINOv2 architecture
        Preserves exact feature extraction logic from original system
        
        Optimized Architecture: 
        - CLIP: 768-dimensional multi-modal vision-language features (native ViT-L/14)
        - DINOv2: 768-dimensional self-supervised visual features (native DINOv2-base)
        - Total: 1536 dimensions for maximum accuracy and speed
        
        Args:
            image: Image input (file path, PIL Image, or numpy array)
            
        Returns:
            Dictionary with 'clip' and 'dinov2' feature vectors (768 dims each)
        """
        start_time = time.time()
        
        # Load and preprocess image (preserved logic)
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Use high-resolution processing for maximum accuracy (preserved)
        target_size = (self.config.get('image_size', 768), self.config.get('image_size', 768))
        pil_image_resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        features = {}
        
        try:
            # === OPTIMIZED DUAL-MODEL ARCHITECTURE (PRESERVED) ===
            
            # Extract CLIP features (768 dimensions)
            logger.debug("Extracting CLIP features (768D)")
            features['clip'] = self.extract_clip_features(pil_image_resized)
            
            # Extract DINOv2 features (768 dimensions)
            logger.debug("Extracting DINOv2 features (768D)")
            features['dinov2'] = self.extract_dinov2_features(pil_image_resized)
            
            # Verify dimensions (preserved validation)
            clip_dims = len(features['clip']) if features['clip'] is not None else 0
            dino_dims = len(features['dinov2']) if features['dinov2'] is not None else 0
            total_dims = clip_dims + dino_dims
            
            logger.debug(f"Feature extraction complete: CLIP({clip_dims}D) + DINOv2({dino_dims}D) = {total_dims}D total")
            
            # Normalize all feature vectors for optimal similarity computation (preserved)
            for key in features:
                if features[key] is not None and len(features[key]) > 0:
                    features[key] = normalize(features[key].reshape(1, -1))[0]
            
            # Update performance stats
            extraction_time_ms = (time.time() - start_time) * 1000
            self.stats['features_extracted'] += 1
            self.stats['avg_extraction_time_ms'] = (
                (self.stats['avg_extraction_time_ms'] * (self.stats['features_extracted'] - 1) + extraction_time_ms) 
                / self.stats['features_extracted']
            )
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return None
        
        return features
    
    def store_features_to_sqlite(self, image_id: str, item_id: str, image_path: str, 
                               image_data: Optional[np.ndarray] = None,
                               augmentation_params: Optional[Dict] = None) -> bool:
        """
        Extract features and store directly to SQLite
        Replaces HDF5 storage with SQLite Vector Store
        """
        try:
            # Extract features from image
            if image_data is not None:
                features = self.extract_features_from_image(image_data)
            else:
                features = self.extract_features_from_image(image_path)
            
            if features is None:
                logger.error(f"❌ Failed to extract features for {image_id}")
                return False
            
            # Create combined features (preserved logic)
            clip_features = features['clip']
            dinov2_features = features['dinov2']
            combined_features = np.concatenate([clip_features, dinov2_features])
            
            # Validate dimensions (preserved validation)
            assert len(clip_features) == 768, f"CLIP features must be 768D, got {len(clip_features)}"
            assert len(dinov2_features) == 768, f"DINOv2 features must be 768D, got {len(dinov2_features)}"
            assert len(combined_features) == 1536, f"Combined features must be 1536D, got {len(combined_features)}"
            
            # Create feature record
            record = FeatureRecord(
                image_id=image_id,
                item_id=item_id,
                image_path=image_path,
                clip_features=clip_features,
                dinov2_features=dinov2_features,
                combined_features=combined_features,
                augmentation_params=augmentation_params,
                extraction_timestamp=time.time(),
                image_hash=None  # Will be computed by the vector store
            )
            
            # Store to SQLite
            success = self.vector_store.store_features(record)
            
            if success:
                logger.debug(f"✅ Stored features for {image_id} (item: {item_id})")
            else:
                logger.error(f"❌ Failed to store features for {image_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error storing features for {image_id}: {e}")
            return False
    
    def process_augmented_images_to_sqlite(self, augmented_images: List[Dict]) -> Dict:
        """
        Process a list of augmented images and store their features to SQLite
        """
        success_count = 0
        failure_count = 0
        
        for aug_record in tqdm(augmented_images, desc="Extracting features"):
            success = self.store_features_to_sqlite(
                image_id=aug_record['image_id'],
                item_id=aug_record['item_id'],
                image_path=aug_record['original_path'],
                image_data=aug_record['image_data'],
                augmentation_params=aug_record['augmentation_params']
            )
            
            if success:
                success_count += 1
            else:
                failure_count += 1
        
        return {
            'success_count': success_count,
            'failure_count': failure_count,
            'total_processed': len(augmented_images)
        }
    
    def process_item_directory_to_sqlite(self, item_dir: Path, item_id: str = None) -> Dict:
        """
        Process all images in an item directory and store features to SQLite
        Preserves original processing logic but with SQLite storage
        """
        if item_id is None:
            item_id = item_dir.name
        
        # Check if features already exist
        existing_features = self.vector_store.get_item_features(item_id)
        if existing_features:
            logger.info(f"Item {item_id} already has {len(existing_features)} features in SQLite")
            return {
                'status': 'skipped',
                'item_id': item_id,
                'reason': 'already_processed',
                'feature_count': len(existing_features)
            }
        
        # Find all images (preserved logic)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(item_dir.glob(f"*{ext}"))
            image_files.extend(item_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No images found in {item_dir}")
            return {'status': 'failed', 'reason': 'no_images'}
        
        logger.info(f"Processing {len(image_files)} images for item {item_id}")
        
        success_count = 0
        failure_count = 0
        
        # Process each image
        for idx, image_file in enumerate(image_files):
            image_id = f"{item_id}_orig_{idx}_{uuid.uuid4().hex[:8]}"
            
            success = self.store_features_to_sqlite(
                image_id=image_id,
                item_id=item_id,
                image_path=str(image_file),
                augmentation_params=None  # Original images have no augmentation
            )
            
            if success:
                success_count += 1
            else:
                failure_count += 1
        
        # Update item statistics
        self.stats['items_processed'] += 1
        
        logger.info(f"✅ Processed {success_count}/{len(image_files)} images for item {item_id}")
        
        return {
            'status': 'success',
            'item_id': item_id,
            'success_count': success_count,
            'failure_count': failure_count,
            'total_images': len(image_files)
        }
    
    def process_dataset_to_sqlite(self, input_dir: Path) -> Dict:
        """
        Process entire dataset and store features to SQLite
        Replaces HDF5-based processing
        """
        input_path = Path(input_dir)
        
        # Find all item directories
        item_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        logger.info(f"Processing {len(item_dirs)} items with SQLite feature storage")
        
        results = []
        total_success = 0
        total_failure = 0
        
        # Process each item
        for item_dir in tqdm(item_dirs, desc="Processing items"):
            result = self.process_item_directory_to_sqlite(item_dir)
            results.append(result)
            
            if result['status'] == 'success':
                total_success += result.get('success_count', 0)
                total_failure += result.get('failure_count', 0)
        
        # Final statistics
        final_stats = {
            'total_items_processed': len(item_dirs),
            'total_features_extracted': total_success,
            'total_failures': total_failure,
            'avg_extraction_time_ms': self.stats['avg_extraction_time_ms'],
            'device_used': self.stats['device_used'],
            'storage_method': 'sqlite'
        }
        
        logger.info("🎉 SQLite Feature extraction complete!")
        logger.info(f"   Items processed: {final_stats['total_items_processed']}")
        logger.info(f"   Features extracted: {final_stats['total_features_extracted']}")
        logger.info(f"   Failures: {final_stats['total_failures']}")
        logger.info(f"   Average extraction time: {final_stats['avg_extraction_time_ms']:.2f}ms")
        
        return {
            'statistics': final_stats,
            'results': results
        }
    
    def get_statistics(self) -> Dict:
        """Get feature extraction statistics"""
        return {
            **self.stats,
            'vector_store_stats': self.vector_store.get_statistics()
        }


def create_feature_extractor(config_path: str, vector_store: SQLiteVectorStore) -> MultiModalFeatureExtractor:
    """
    Factory function to create feature extractor with SQLite storage
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    feature_config = config.get('feature_extraction', {})
    
    return MultiModalFeatureExtractor(feature_config, vector_store)


def main():
    """Main entry point for feature extraction with SQLite storage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features with SQLite storage')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--config', type=str, required=True, help='Configuration YAML file')
    parser.add_argument('--database', type=str, required=True, help='SQLite database path')
    
    args = parser.parse_args()
    
    # Create vector store
    vector_store = SQLiteVectorStore(args.database)
    
    # Create feature extractor
    extractor = create_feature_extractor(args.config, vector_store)
    
    # Process dataset
    results = extractor.process_dataset_to_sqlite(Path(args.input))
    
    print(f"✅ Feature extraction complete!")
    print(f"   Items: {results['statistics']['total_items_processed']}")
    print(f"   Features: {results['statistics']['total_features_extracted']}")
    print(f"   Storage: SQLite Vector Store")


if __name__ == "__main__":
    main()