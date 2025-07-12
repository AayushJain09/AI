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
        """Load feature extraction models (mobile mode support)"""
        # Check for mobile mode
        self.mobile_mode = self.config.get('mobile_mode', False)
        
        # CLIP model with SSL fix (always loaded)
        logger.info("Loading CLIP model...")
        import ssl
        import urllib.request
        
        # Temporarily disable SSL verification for CLIP download
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Apply SSL context
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))
        
        try:
            self.clip_model, self.clip_preprocess = clip.load(
                self.config.get('clip_variant', 'ViT-B/32'), 
                device=self.device
            )
            self.clip_model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
        # Additional models (only in full mode)
        if not self.mobile_mode:
            # ResNet model (fix deprecated pretrained parameter)
            logger.info("Loading ResNet model...")
            from torchvision.models import ResNet50_Weights
            self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove FC layer
            self.resnet = self.resnet.to(self.device)
            self.resnet.eval()
            
            # EfficientNet model for additional features
            logger.info("Loading EfficientNet model...")
            from torchvision.models import EfficientNet_B4_Weights
            self.efficientnet = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            self.efficientnet.classifier = nn.Identity()  # Remove classifier
            self.efficientnet = self.efficientnet.to(self.device)
            self.efficientnet.eval()
        else:
            logger.info("Mobile mode: Skipping ResNet and EfficientNet models")
            self.resnet = None
            self.efficientnet = None
        
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
        """Extract CLIP features"""
        with torch.no_grad():
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
            return features.cpu().numpy().flatten()
    
    def extract_resnet_features(self, image: Image.Image) -> np.ndarray:
        """Extract ResNet features (mobile mode aware)"""
        if self.resnet is None:
            return np.zeros(2048)  # Return zeros in mobile mode
        
        with torch.no_grad():
            image_input = self.cnn_preprocess(image).unsqueeze(0).to(self.device)
            features = self.resnet(image_input)
            features = features.flatten()
            return features.cpu().numpy()
    
    def extract_efficientnet_features(self, image: Image.Image) -> np.ndarray:
        """Extract EfficientNet features (mobile mode aware)"""
        if self.efficientnet is None:
            return np.zeros(1792)  # Return zeros in mobile mode
        
        with torch.no_grad():
            image_input = self.cnn_preprocess(image).unsqueeze(0).to(self.device)
            features = self.efficientnet(image_input)
            return features.cpu().numpy().flatten()
    
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
        """Extract features from an image (mobile mode support)"""
        # Load image
        pil_image = Image.open(image_path).convert('RGB')
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize for consistent feature extraction (mobile uses smaller size)
        if self.mobile_mode:
            target_size = (512, 512)  # Mobile optimized
        else:
            target_size = (self.config.get('feature_image_size', 512),) * 2
        
        pil_image_resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        cv_image_resized = cv2.resize(cv_image_rgb, target_size)
        
        features = {}
        
        try:
            # CLIP features (always extracted)
            features['clip'] = self.extract_clip_features(pil_image_resized)
            
            if self.mobile_mode:
                # Mobile mode: CLIP only for speed
                logger.debug(f"Mobile mode: Using CLIP features only for {image_path}")
            else:
                # Full mode: Extract all features
                features['resnet'] = self.extract_resnet_features(pil_image_resized)
                features['efficientnet'] = self.extract_efficientnet_features(pil_image_resized)
                
                # Traditional CV features (skip in mobile mode for speed)
                features['color'] = self.extract_color_features(cv_image_resized)
                features['texture'] = self.extract_texture_features(cv_image_resized)
                features['shape'] = self.extract_shape_features(cv_image_resized)
            
            # Normalize all features
            for key in features:
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
    parser.add_argument('--clip-model', type=str, default='ViT-B/32', help='CLIP model variant')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'clip_variant': args.clip_model,
        'batch_size': args.batch_size,
        'feature_image_size': 512
    }
    
    # Create feature extractor
    extractor = MultiModalFeatureExtractor(config)
    
    # Process dataset
    extractor.process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()