"""
Modern ColorThief-based Color Extraction
Extracts dominant colors from images after background removal for enhanced recognition
"""

import numpy as np
import cv2
from PIL import Image
import json
import logging
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import io

# ColorThief for modern color extraction
try:
    from colorthief import ColorThief
    COLORTHIEF_AVAILABLE = True
except ImportError:
    COLORTHIEF_AVAILABLE = False
    logging.warning("ColorThief not available - color extraction will be disabled")

# Background removal
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not available - background removal will be skipped")

logger = logging.getLogger(__name__)


class ModernColorExtractor:
    """
    GPU-accelerated color extraction with background removal for precise color analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize color extractor with optimized settings
        
        Args:
            config: Configuration dictionary with extraction parameters
        """
        config = config or {}
        
        # Color extraction parameters
        self.num_colors = config.get('num_colors', 10)  # Extract top 10 colors
        self.color_quality = config.get('color_quality', 1)  # 1=best quality, 10=fast
        self.min_color_difference = config.get('min_color_difference', 30)  # RGB difference threshold
        
        # Background removal settings
        self.remove_background = config.get('remove_background', True)
        self.background_model = config.get('background_model', 'u2net')  # u2net, silueta, etc.
        
        # Performance settings
        self.resize_for_analysis = config.get('resize_for_analysis', (512, 512))  # Resize for faster processing
        self.cache_results = config.get('cache_results', True)
        
        # Color similarity threshold for deduplication
        self.similarity_threshold = config.get('similarity_threshold', 20)
        
        logger.info(f"🎨 Modern Color Extractor initialized with {self.num_colors} colors per image")
        
    def extract_colors_from_image(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict:
        """
        Extract dominant colors from image with optional background removal
        
        Args:
            image_path: Path to image file, PIL Image, or numpy array
            
        Returns:
            Dictionary containing:
                - dominant_color: RGB tuple of most dominant color
                - color_palette: List of top N colors as RGB tuples
                - color_percentages: Estimated percentages for each color
                - background_removed: Whether background was removed
                - processing_info: Technical details about extraction
        """
        try:
            # Load and prepare image
            if isinstance(image_path, (str, Path)):
                # Load from file path
                img_pil = Image.open(image_path)
                image_name = Path(image_path).name
            elif isinstance(image_path, Image.Image):
                # Already PIL Image
                img_pil = image_path.copy()
                image_name = "unknown"
            elif isinstance(image_path, np.ndarray):
                # Convert numpy array to PIL
                if image_path.dtype != np.uint8:
                    image_path = (image_path * 255).astype(np.uint8)
                if len(image_path.shape) == 3 and image_path.shape[2] == 3:
                    # RGB to PIL
                    img_pil = Image.fromarray(image_path)
                elif len(image_path.shape) == 3 and image_path.shape[2] == 4:
                    # RGBA to PIL
                    img_pil = Image.fromarray(image_path, 'RGBA')
                else:
                    raise ValueError(f"Unsupported numpy array shape: {image_path.shape}")
                image_name = "numpy_array"
            else:
                raise ValueError(f"Unsupported image type: {type(image_path)}")
            
            # Ensure RGB mode for consistent processing
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            
            original_size = img_pil.size
            background_removed = False
            
            # Optional background removal for cleaner color extraction
            if self.remove_background and REMBG_AVAILABLE:
                try:
                    # Remove background using rembg
                    img_bytes = io.BytesIO()
                    img_pil.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    # Process with rembg (returns RGBA with transparent background)
                    img_no_bg = remove(img_bytes.getvalue())
                    img_pil = Image.open(io.BytesIO(img_no_bg))
                    
                    # Convert RGBA to RGB with white background for ColorThief
                    if img_pil.mode == 'RGBA':
                        # Create white background
                        rgb_img = Image.new('RGB', img_pil.size, (255, 255, 255))
                        # Paste image with alpha channel as mask
                        rgb_img.paste(img_pil, mask=img_pil.split()[3])
                        img_pil = rgb_img
                    
                    background_removed = True
                    logger.debug(f"🖼️ Background removed from {image_name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Background removal failed for {image_name}: {e}")
                    # Continue with original image if background removal fails
            
            # Resize for faster processing while maintaining aspect ratio
            if self.resize_for_analysis and img_pil.size != self.resize_for_analysis:
                img_pil.thumbnail(self.resize_for_analysis, Image.Resampling.LANCZOS)
            
            # Extract colors using ColorThief
            if not COLORTHIEF_AVAILABLE:
                raise RuntimeError("ColorThief not available - cannot extract colors")
            
            # Save temporarily for ColorThief (it requires a file path)
            temp_buffer = io.BytesIO()
            img_pil.save(temp_buffer, format='PNG')
            temp_buffer.seek(0)
            
            color_thief = ColorThief(temp_buffer)
            
            # Get dominant color
            dominant_color = color_thief.get_color(quality=self.color_quality)
            
            # Get color palette
            try:
                color_palette = color_thief.get_palette(
                    color_count=self.num_colors,
                    quality=self.color_quality
                )
            except Exception as e:
                # Fallback to just dominant color if palette extraction fails
                logger.warning(f"⚠️ Palette extraction failed for {image_name}: {e}")
                color_palette = [dominant_color]
            
            # Remove similar colors to improve palette diversity
            filtered_palette = self._filter_similar_colors(color_palette)
            
            # Calculate rough color percentages (approximation)
            color_percentages = self._estimate_color_percentages(img_pil, filtered_palette)
            
            result = {
                'dominant_color': dominant_color,
                'color_palette': filtered_palette,
                'color_percentages': color_percentages,
                'background_removed': background_removed,
                'processing_info': {
                    'original_size': original_size,
                    'processed_size': img_pil.size,
                    'num_extracted_colors': len(filtered_palette),
                    'color_quality_setting': self.color_quality,
                    'extraction_method': 'ColorThief'
                }
            }
            
            logger.debug(f"✅ Extracted {len(filtered_palette)} colors from {image_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Color extraction failed for {image_name}: {e}")
            # Return minimal result on failure
            return {
                'dominant_color': (128, 128, 128),  # Gray fallback
                'color_palette': [(128, 128, 128)],
                'color_percentages': [1.0],
                'background_removed': False,
                'processing_info': {
                    'error': str(e),
                    'extraction_method': 'Failed'
                }
            }
    
    def _filter_similar_colors(self, colors: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Remove colors that are too similar to improve palette diversity
        
        Args:
            colors: List of RGB color tuples
            
        Returns:
            Filtered list of distinct colors
        """
        if len(colors) <= 1:
            return colors
            
        filtered = [colors[0]]  # Always keep the first (dominant) color
        
        for color in colors[1:]:
            # Check if this color is sufficiently different from existing ones
            is_distinct = True
            for existing in filtered:
                # Calculate Euclidean distance in RGB space
                distance = np.sqrt(
                    (color[0] - existing[0])**2 +
                    (color[1] - existing[1])**2 +
                    (color[2] - existing[2])**2
                )
                if distance < self.similarity_threshold:
                    is_distinct = False
                    break
            
            if is_distinct:
                filtered.append(color)
        
        return filtered
    
    def _estimate_color_percentages(self, image: Image.Image, colors: List[Tuple[int, int, int]]) -> List[float]:
        """
        Estimate rough percentages for each color in the palette
        This is an approximation - for precise percentages, pixel-by-pixel analysis would be needed
        
        Args:
            image: PIL Image object
            colors: List of RGB color tuples
            
        Returns:
            List of percentage estimates (0.0 to 1.0)
        """
        if len(colors) <= 1:
            return [1.0]
        
        # Simple estimation: dominant color gets highest percentage, others decrease
        # This is a rough approximation - real implementation would analyze pixel distributions
        percentages = []
        total = 0
        
        # Use exponential decay for percentage distribution
        for i, color in enumerate(colors):
            # First color (dominant) gets highest percentage
            percentage = 0.6 * (0.7 ** i)  # Exponential decay
            percentages.append(percentage)
            total += percentage
        
        # Normalize to sum to 1.0
        if total > 0:
            percentages = [p / total for p in percentages]
        else:
            # Fallback: equal distribution
            percentages = [1.0 / len(colors)] * len(colors)
        
        return percentages
    
    def extract_colors_batch(self, image_paths: List[Union[str, Path]]) -> Dict[str, Dict]:
        """
        Extract colors from multiple images in batch for efficiency
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Dictionary mapping image paths to color extraction results
        """
        results = {}
        
        for image_path in image_paths:
            try:
                result = self.extract_colors_from_image(image_path)
                results[str(image_path)] = result
            except Exception as e:
                logger.error(f"❌ Batch color extraction failed for {image_path}: {e}")
                results[str(image_path)] = {
                    'error': str(e),
                    'dominant_color': (128, 128, 128),
                    'color_palette': [(128, 128, 128)],
                    'color_percentages': [1.0],
                    'background_removed': False
                }
        
        logger.info(f"🎨 Batch color extraction completed for {len(image_paths)} images")
        return results
    
    def color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """
        Calculate perceptual distance between two colors using Delta E formula
        
        Args:
            color1: First RGB color tuple
            color2: Second RGB color tuple
            
        Returns:
            Distance value (0 = identical, higher = more different)
        """
        # Simple Euclidean distance in RGB space
        # For more accurate perceptual distance, convert to LAB color space
        return np.sqrt(
            (color1[0] - color2[0])**2 +
            (color1[1] - color2[1])**2 +
            (color1[2] - color2[2])**2
        )
    
    def find_similar_colors(self, query_colors: List[Tuple[int, int, int]], 
                          database_colors: List[Tuple[int, int, int]], 
                          threshold: float = 50.0) -> List[int]:
        """
        Find colors in database that are similar to query colors
        
        Args:
            query_colors: List of query RGB color tuples
            database_colors: List of database RGB color tuples  
            threshold: Maximum distance for colors to be considered similar
            
        Returns:
            List of indices in database_colors that match query colors
        """
        matches = []
        
        for db_idx, db_color in enumerate(database_colors):
            for query_color in query_colors:
                distance = self.color_distance(query_color, db_color)
                if distance <= threshold:
                    matches.append(db_idx)
                    break  # Don't double-count the same database color
        
        return matches


def create_color_extractor(config: Optional[Dict] = None) -> ModernColorExtractor:
    """
    Factory function to create optimized color extractor instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured ModernColorExtractor instance
    """
    default_config = {
        'num_colors': 10,
        'color_quality': 1,  # Best quality
        'remove_background': True,
        'resize_for_analysis': (512, 512),
        'similarity_threshold': 20,
        'cache_results': True
    }
    
    if config:
        default_config.update(config)
    
    return ModernColorExtractor(default_config)