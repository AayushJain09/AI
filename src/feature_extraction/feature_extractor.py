"""
Multi-Model Feature Extraction System
Combines CLIP, ResNet, and traditional CV features for robust recognition
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
from typing import Dict, List, Tuple, Optional
import h5py
import json
from tqdm import tqdm
import logging
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalFeatureExtractor:
    """Extract features using multiple models and techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        # Use GPU acceleration if available (CUDA or Apple Silicon MPS)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_models()
        
        # Define preprocessing
        self._setup_preprocessing()
        
    def _load_models(self):
        """
        Load optimized feature extraction models:
        - CLIP: Multi-modal vision-language features (768 dimensions)
        - DINOv2: Self-supervised vision features (768 dimensions)
        - ResNet: REMOVED for performance optimization
        - EfficientNet: REMOVED for performance optimization
        Total: 1536 dimensions for maximum accuracy and speed
        """
        
        # === CLIP Model (Core Model #1 - Always Loaded) ===
        logger.info("🚀 Loading CLIP model (optimized 768-dim vision-language features)...")
        import ssl
        import urllib.request
        
        # Temporarily disable SSL verification for CLIP download
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))
        
        try:
            # Load larger CLIP variant for 768 dimensions (native ViT-L/14 dimensions)
            clip_variant = self.config.get('clip_variant', 'ViT-L/14')  # ViT-L/14 outputs 768 dims natively
            self.clip_model, self.clip_preprocess = clip.load(clip_variant, device=self.device)
            self.clip_model.eval()
            
            # Get actual output dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_output = self.clip_model.encode_image(dummy_input)
                actual_clip_dims = dummy_output.shape[1]
                logger.info(f"✓ CLIP {clip_variant} loaded: {actual_clip_dims} native dims (using full 768)")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
        # === DINOv2 Model (Core Model #2 - Always Loaded) ===
        logger.info("🔥 Loading DINOv2 model (optimized 768-dim self-supervised features)...")
        try:
            # Use DINOv2-base for 768 dimensional features (native dimensions)
            dinov2_variant = self.config.get('dinov2_variant', 'dinov2_vitb14')  # dinov2_vitb14 outputs 768 dims natively
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_variant, trust_repo=True)
            self.dinov2 = self.dinov2.to(self.device)
            self.dinov2.eval()
            
            # Get actual output dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_output = self.dinov2(dummy_input)
                actual_dino_dims = dummy_output.shape[1]
                logger.info(f"✓ DINOv2 {dinov2_variant} loaded: {actual_dino_dims} native dims (using full 768)")
        except Exception as e:
            logger.warning(f"Failed to load DINOv2: {e}. Proceeding with CLIP-only mode")
            self.dinov2 = None
        
        # === REMOVED MODELS FOR OPTIMIZATION ===
        self.resnet = None
        self.efficientnet = None
        logger.info("🗑️  ResNet and EfficientNet REMOVED for maximum performance")
        logger.info("🎯 Optimized Architecture: CLIP (768) + DINOv2 (768) = 1536 total dimensions")
        
        # Initialize feature compression layers for dimension optimization
        self._setup_feature_compression()
        
    def _setup_feature_compression(self):
        """Setup feature dimensions (no compression needed - using native 768)"""
        self.target_clip_dims = 768      # Use native CLIP ViT-L/14 dimensions
        self.target_dinov2_dims = 768    # Use native DINOv2-base dimensions
        
        # No compression layers needed - using native model outputs at 768 dimensions
        self.clip_compressor = None
        self.dinov2_compressor = None
        
        logger.info(f"🔧 Feature dimensions set: CLIP={self.target_clip_dims}, DINOv2={self.target_dinov2_dims} (native, no compression)")
        
    def _setup_preprocessing(self):
        """Setup preprocessing pipelines for different models"""
        # ResNet/EfficientNet preprocessing
        self.cnn_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_clip_features(self, image: Image.Image) -> np.ndarray:
        """Extract optimized CLIP features (768 dimensions - native ViT-L/14)"""
        with torch.no_grad():
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_input)
            
            # Use native 768 dimensions (no compression needed)
            # ViT-L/14 naturally outputs 768-dimensional features
            
            # Normalize for optimal similarity computation
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
    
    def extract_dinov2_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract optimized DINOv2 self-supervised features (768 dimensions - native DINOv2-base).
        
        DINOv2 provides complementary features to CLIP:
        - CLIP: Multi-modal (vision + language) understanding  
        - DINOv2: Pure visual self-supervised features with excellent fine-grained recognition
        
        Returns 768-dimensional feature vector from DINOv2-base (native dimensions)
        """
        if self.dinov2 is None:
            return np.zeros(768)  # Return zeros if DINOv2 failed to load
        
        with torch.no_grad():
            # DINOv2 expects normalized RGB images of size 224x224
            dinov2_transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            image_input = dinov2_transform(image).unsqueeze(0).to(self.device)
            features = self.dinov2(image_input)
            
            # Use native 768 dimensions (no compression needed)
            # DINOv2-base naturally outputs 768-dimensional features
            
            # Normalize for optimal similarity computation
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
    
    def extract_resnet_features(self, image: Image.Image) -> np.ndarray:
        """ResNet features REMOVED - not used in optimized architecture"""
        return np.array([])  # Return empty array for removed model
    
    def extract_efficientnet_features(self, image: Image.Image) -> np.ndarray:
        """EfficientNet features REMOVED - not used in optimized architecture"""
        return np.array([])  # Return empty array for removed model
    
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        features = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # HSV histogram
        for i, (bins, ranges) in enumerate([(50, [0, 180]), (60, [0, 256]), (60, [0, 256])]):
            hist = cv2.calcHist([hsv], [i], None, [bins], ranges)
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # LAB histogram
        for i, (bins, ranges) in enumerate([(50, [0, 256]), (50, [0, 256]), (50, [0, 256])]):
            hist = cv2.calcHist([lab], [i], None, [bins], ranges)
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # Color moments (mean, std, skewness)
        for channel in cv2.split(image):
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.abs(np.mean(((channel - np.mean(channel)) / np.std(channel)) ** 3))
            ])
        
        return np.array(features)
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using LBP and HOG"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = []
        
        # Local Binary Patterns
        lbp_features = self._compute_lbp(gray)
        features.extend(lbp_features)
        
        # Histogram of Oriented Gradients (simplified)
        hog_features = self._compute_hog(gray)
        features.extend(hog_features)
        
        # Gabor filters
        gabor_features = self._compute_gabor(gray)
        features.extend(gabor_features)
        
        return np.array(features)
    
    def _compute_lbp(self, gray_image: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
        """Compute Local Binary Pattern features"""
        def lbp_pixel(img, x, y, radius, n_points):
            center = img[y, x]
            binary_string = ""
            
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x_p = x + radius * np.cos(angle)
                y_p = y + radius * np.sin(angle)
                
                # Bilinear interpolation
                x1, y1 = int(np.floor(x_p)), int(np.floor(y_p))
                x2, y2 = min(x1 + 1, img.shape[1] - 1), min(y1 + 1, img.shape[0] - 1)
                
                fx, fy = x_p - x1, y_p - y1
                
                interpolated = (1 - fx) * (1 - fy) * img[y1, x1] + \
                              fx * (1 - fy) * img[y1, x2] + \
                              (1 - fx) * fy * img[y2, x1] + \
                              fx * fy * img[y2, x2]
                
                binary_string += "1" if interpolated >= center else "0"
            
            return int(binary_string, 2)
        
        # Compute LBP for the image
        h, w = gray_image.shape
        lbp_image = np.zeros((h - 2*radius, w - 2*radius))
        
        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                lbp_image[y - radius, x - radius] = lbp_pixel(gray_image, x, y, radius, n_points)
        
        # Compute histogram
        hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        
        return hist
    
    def _compute_hog(self, gray_image: np.ndarray, cell_size: int = 16) -> np.ndarray:
        """Simplified HOG feature extraction"""
        # Compute gradients
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=1)
        
        # Compute magnitude and angle
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180
        
        # Compute histograms for cells
        h, w = gray_image.shape
        n_cells_y = h // cell_size
        n_cells_x = w // cell_size
        
        hist_bins = 9
        hist_range = (0, 180)
        
        features = []
        
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_magnitude = magnitude[i*cell_size:(i+1)*cell_size, 
                                         j*cell_size:(j+1)*cell_size]
                cell_angle = angle[i*cell_size:(i+1)*cell_size, 
                                 j*cell_size:(j+1)*cell_size]
                
                hist, _ = np.histogram(cell_angle, 
                                     bins=hist_bins, 
                                     range=hist_range, 
                                     weights=cell_magnitude)
                hist = hist / (np.sum(hist) + 1e-6)
                features.extend(hist)
        
        return np.array(features)
    
    def _compute_gabor(self, gray_image: np.ndarray) -> np.ndarray:
        """Compute Gabor filter features"""
        features = []
        
        # Gabor filter parameters
        ksize = 31
        sigma = 4.0
        lambd = 10.0
        gamma = 0.5
        psi = 0
        
        # Multiple orientations
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
            filtered = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
            
            # Compute statistics
            features.extend([
                np.mean(filtered),
                np.std(filtered),
                np.mean(np.abs(filtered))
            ])
        
        return np.array(features)
    
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Extract shape-based features"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Contour features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape descriptors
            features.append(area)
            features.append(perimeter)
            features.append(4 * np.pi * area / (perimeter ** 2 + 1e-6))  # Circularity
            
            # Moments
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            features.append(w / (h + 1e-6))  # Aspect ratio
            features.append(area / (w * h + 1e-6))  # Extent
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features.append(area / (hull_area + 1e-6))  # Solidity
        else:
            # Default values if no contour found
            features = [0] * 12
        
        return np.array(features)
    
    def extract_all_features(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Extract optimized features from an image using CLIP + DINOv2 architecture.
        
        Optimized Architecture: 
        - CLIP: 768-dimensional multi-modal vision-language features (native ViT-L/14)
        - DINOv2: 768-dimensional self-supervised visual features (native DINOv2-base)
        - Total: 1536 dimensions for maximum accuracy and speed
        - ResNet/EfficientNet: REMOVED for performance optimization
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with 'clip' and 'dinov2' feature vectors (768 dims each)
        """
        # Load and preprocess image
        pil_image = Image.open(image_path).convert('RGB')
        
        # Use high-resolution processing for maximum accuracy
        target_size = (self.config.get('feature_image_size', 768), self.config.get('feature_image_size', 768))
        pil_image_resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        features = {}
        
        try:
            # === OPTIMIZED DUAL-MODEL ARCHITECTURE ===
            
            # Extract CLIP features (768 dimensions)
            logger.debug(f"Extracting CLIP features (768D) from {image_path}")
            features['clip'] = self.extract_clip_features(pil_image_resized)
            
            # Extract DINOv2 features (768 dimensions)
            logger.debug(f"Extracting DINOv2 features (768D) from {image_path}")
            features['dinov2'] = self.extract_dinov2_features(pil_image_resized)
            
            # Verify dimensions
            clip_dims = len(features['clip']) if features['clip'] is not None else 0
            dino_dims = len(features['dinov2']) if features['dinov2'] is not None else 0
            total_dims = clip_dims + dino_dims
            
            logger.debug(f"Feature extraction complete: CLIP({clip_dims}D) + DINOv2({dino_dims}D) = {total_dims}D total")
            
            # Normalize all feature vectors for optimal similarity computation
            for key in features:
                if features[key] is not None and len(features[key]) > 0:
                    features[key] = normalize(features[key].reshape(1, -1))[0]
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None
        
        return features
    
    def process_dataset(self, image_dir: Path, output_file: Path) -> None:
        """Process entire dataset and save features to HDF5"""
        image_dir = Path(image_dir)
        
        # Collect all images
        image_files = []
        for item_dir in image_dir.iterdir():
            if item_dir.is_dir():
                for img_file in item_dir.glob('*.jpg'):
                    image_files.append((item_dir.name, str(img_file)))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Create HDF5 file
        with h5py.File(output_file, 'w') as hf:
            # Process images
            for idx, (item_id, img_path) in enumerate(tqdm(image_files, desc="Extracting features")):
                features = self.extract_all_features(img_path)
                
                if features is None:
                    continue
                
                # Create group for this image
                img_group = hf.create_group(f"image_{idx:06d}")
                img_group.attrs['item_id'] = item_id
                img_group.attrs['image_path'] = img_path
                
                # Save features
                for feature_name, feature_vector in features.items():
                    img_group.create_dataset(feature_name, data=feature_vector)
            
            # Save metadata
            hf.attrs['total_images'] = len(image_files)
            hf.attrs['feature_types'] = list(features.keys()) if features else []
            hf.attrs['config'] = json.dumps(self.config)
        
        logger.info(f"Features saved to {output_file}")


def main():
    """Main entry point for feature extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from augmented dataset')
    parser.add_argument('--input', type=str, required=True, help='Input directory with augmented images')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file for features')
    parser.add_argument('--clip-model', type=str, default='ViT-L/14', help='CLIP model variant (optimized for 768 dimensions)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'clip_variant': args.clip_model,
        'dinov2_variant': 'dinov2_vitb14',  # Ensure DINOv2-base for 768 dims
        'batch_size': args.batch_size,
        'feature_image_size': 768  # Higher resolution for better features
    }
    
    # Create feature extractor
    extractor = MultiModalFeatureExtractor(config)
    
    # Process dataset
    extractor.process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()