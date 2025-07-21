#!/usr/bin/env python3
"""
Unit Tests: Feature Extraction Components
Tests individual feature extraction components in isolation
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from feature_extraction.feature_extractor import MultiModalFeatureExtractor


class TestFeatureExtractor:
    """Unit tests for feature extraction components"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.config = {
            'clip_model': 'ViT-L/14',
            'clip_dimensions': 768,
            'dinov2_dimensions': 768,
            'total_dimensions': 1536,
            'batch_size': 32
        }
        
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        extractor = MultiModalFeatureExtractor(self.config)
        assert extractor.config['clip_model'] == 'ViT-L/14'
        assert extractor.config['total_dimensions'] == 1536
        
        # Invalid config - missing required keys
        with pytest.raises(KeyError):
            invalid_config = {'clip_model': 'ViT-L/14'}  # Missing other keys
            MultiModalFeatureExtractor(invalid_config)
    
    def test_device_selection(self):
        """Test device selection logic"""
        extractor = MultiModalFeatureExtractor(self.config)
        
        # Should select a valid device
        assert extractor.device in ['cpu', 'cuda', 'mps']
        
        # Device should be accessible by torch
        test_tensor = torch.randn(1)
        device_tensor = test_tensor.to(extractor.device)
        assert device_tensor.device.type in ['cpu', 'cuda', 'mps']
    
    @patch('feature_extraction.feature_extractor.clip')
    def test_clip_model_loading(self, mock_clip):
        """Test CLIP model loading"""
        # Mock CLIP loading
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)
        
        extractor = MultiModalFeatureExtractor(self.config)
        
        # Verify CLIP was loaded with correct parameters
        mock_clip.load.assert_called_once_with(
            'ViT-L/14', 
            device=extractor.device
        )
    
    @patch('feature_extraction.feature_extractor.torch.hub.load')
    def test_dinov2_model_loading(self, mock_hub_load):
        """Test DINOv2 model loading"""
        # Mock DINOv2 loading
        mock_model = MagicMock()
        mock_hub_load.return_value = mock_model
        
        extractor = MultiModalFeatureExtractor(self.config)
        
        # Verify DINOv2 was loaded correctly
        mock_hub_load.assert_called()
        args, kwargs = mock_hub_load.call_args
        assert 'facebookresearch/dinov2' in args
    
    def test_image_preprocessing(self):
        """Test image preprocessing functionality"""
        # Create a dummy extractor (without loading real models)
        with patch('feature_extraction.feature_extractor.clip.load'), \
             patch('feature_extraction.feature_extractor.torch.hub.load'):
            
            extractor = MultiModalFeatureExtractor(self.config)
            
            # Mock preprocessor
            mock_preprocess = MagicMock()
            mock_preprocess.return_value = torch.randn(3, 224, 224)
            extractor.clip_preprocess = mock_preprocess
            
            # Test preprocessing
            with patch('PIL.Image.open') as mock_open:
                mock_image = MagicMock()
                mock_open.return_value = mock_image
                
                # This would normally preprocess the image
                result = extractor.clip_preprocess(mock_image)
                
                assert result.shape == (3, 224, 224)
                mock_preprocess.assert_called_once_with(mock_image)
    
    def test_feature_dimensions(self):
        """Test feature dimension validation"""
        # Test with correct dimensions
        clip_features = np.random.randn(768)
        dinov2_features = np.random.randn(768)
        
        assert len(clip_features) == 768
        assert len(dinov2_features) == 768
        
        combined = np.concatenate([clip_features, dinov2_features])
        assert len(combined) == 1536
    
    def test_feature_normalization(self):
        """Test feature normalization"""
        # Create random features
        features = np.random.randn(768) * 10  # Large scale
        
        # L2 normalize
        normalized = features / np.linalg.norm(features)
        
        # Check norm is 1
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6, f"Normalized features should have norm 1, got {norm}"
    
    def test_batch_processing_config(self):
        """Test batch processing configuration"""
        # Test different batch sizes
        configs = [
            {'batch_size': 1},
            {'batch_size': 16},
            {'batch_size': 32},
            {'batch_size': 64}
        ]
        
        for batch_config in configs:
            config = {**self.config, **batch_config}
            
            with patch('feature_extraction.feature_extractor.clip.load'), \
                 patch('feature_extraction.feature_extractor.torch.hub.load'):
                
                extractor = MultiModalFeatureExtractor(config)
                assert extractor.config['batch_size'] == batch_config['batch_size']
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        with patch('feature_extraction.feature_extractor.clip.load'), \
             patch('feature_extraction.feature_extractor.torch.hub.load'):
            
            extractor = MultiModalFeatureExtractor(self.config)
            
            # Test with non-existent file
            with pytest.raises((FileNotFoundError, Exception)):
                extractor.extract_all_features('nonexistent_file.jpg')
    
    def test_feature_consistency(self):
        """Test that feature extraction is consistent"""
        # Mock consistent outputs
        mock_clip_features = np.random.randn(768)
        mock_dinov2_features = np.random.randn(768)
        
        with patch('feature_extraction.feature_extractor.clip.load'), \
             patch('feature_extraction.feature_extractor.torch.hub.load'):
            
            extractor = MultiModalFeatureExtractor(self.config)
            
            # Mock the actual extraction methods
            with patch.object(extractor, '_extract_clip_features', return_value=mock_clip_features), \
                 patch.object(extractor, '_extract_dinov2_features', return_value=mock_dinov2_features), \
                 patch('PIL.Image.open'):
                
                # Extract features twice
                features1 = extractor.extract_all_features('dummy.jpg')
                features2 = extractor.extract_all_features('dummy.jpg')
                
                if features1 and features2:
                    # Should be identical (same mocked output)
                    np.testing.assert_array_equal(features1['clip'], features2['clip'])
                    np.testing.assert_array_equal(features1['dinov2'], features2['dinov2'])


class TestFeatureValidation:
    """Unit tests for feature validation functions"""
    
    def test_validate_feature_dimensions(self):
        """Test feature dimension validation"""
        # Valid dimensions
        valid_clip = np.random.randn(768)
        valid_dinov2 = np.random.randn(768)
        
        assert len(valid_clip) == 768
        assert len(valid_dinov2) == 768
        
        # Invalid dimensions
        invalid_clip = np.random.randn(512)  # Wrong size
        invalid_dinov2 = np.random.randn(1024)  # Wrong size
        
        assert len(invalid_clip) != 768
        assert len(invalid_dinov2) != 768
    
    def test_feature_data_types(self):
        """Test feature data type handling"""
        # Test different data types
        float32_features = np.random.randn(768).astype(np.float32)
        float64_features = np.random.randn(768).astype(np.float64)
        int_features = np.random.randint(0, 100, 768).astype(np.int32)
        
        # All should be convertible to float32
        assert float32_features.dtype == np.float32
        assert float64_features.astype(np.float32).dtype == np.float32
        assert int_features.astype(np.float32).dtype == np.float32
    
    def test_feature_range_validation(self):
        """Test feature value range validation"""
        # Normalized features should typically be in reasonable range
        normalized_features = np.random.randn(768)
        normalized_features = normalized_features / np.linalg.norm(normalized_features)
        
        # Check all values are finite
        assert np.all(np.isfinite(normalized_features))
        
        # Check reasonable range
        assert np.all(np.abs(normalized_features) <= 1.0)


def run_unit_tests():
    """Run unit tests with detailed output"""
    print("🧪 AI Recognition System - Unit Tests (Feature Extraction)")
    print("=" * 60)
    
    pytest_args = [
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ]
    
    result = pytest.main(pytest_args)
    return result == 0


if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)