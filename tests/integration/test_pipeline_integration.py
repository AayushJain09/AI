#!/usr/bin/env python3
"""
Integration Test: Complete Pipeline Components
Tests integration between feature extraction, indexing, and recognition
"""

import sys
import pytest
import numpy as np
import yaml
import h5py
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from feature_extraction.feature_extractor import MultiModalFeatureExtractor
from indexing.faiss_indexer import AdvancedFAISSIndexer, IndexConfig


class TestPipelineIntegration:
    """Integration tests for pipeline components"""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures"""
        # Load configuration
        with open("config.yaml", 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Initialize components
        cls.feature_extractor = MultiModalFeatureExtractor(cls.config['features'])
        
        index_config = IndexConfig(**cls.config.get('indexing', {}))
        cls.indexer = AdvancedFAISSIndexer(index_config)
    
    def test_feature_extraction_integration(self):
        """Test feature extraction with real images"""
        # Find test image
        test_image_path = None
        for item_dir in Path("data/raw").iterdir():
            if item_dir.is_dir():
                for img_file in item_dir.glob("*.JPG"):
                    test_image_path = str(img_file)
                    break
            if test_image_path:
                break
        
        if not test_image_path:
            pytest.skip("No test images found in data/raw")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(test_image_path)
        
        # Validate features
        assert features is not None, "Feature extraction failed"
        assert 'clip' in features, "CLIP features missing"
        assert 'dinov2' in features, "DINOv2 features missing"
        
        # Check dimensions
        clip_dims = len(features['clip'])
        dino_dims = len(features['dinov2'])
        total_dims = clip_dims + dino_dims
        
        assert clip_dims == 768, f"Expected 768 CLIP dims, got {clip_dims}"
        assert dino_dims == 768, f"Expected 768 DINOv2 dims, got {dino_dims}"
        assert total_dims == 1536, f"Expected 1536 total dims, got {total_dims}"
    
    def test_index_loading_integration(self):
        """Test FAISS index loading and search"""
        index_path = "data/models/faiss_index.bin"
        metadata_path = "data/models/index_metadata.pkl"
        
        if not (Path(index_path).exists() and Path(metadata_path).exists()):
            pytest.skip("Index files not found")
        
        # Load index
        success = self.indexer.load_index(index_path, metadata_path)
        assert success, "Failed to load index"
        
        # Verify index properties
        assert self.indexer.index.ntotal > 0, "Index is empty"
        assert self.indexer.index.d == 1536, f"Expected 1536 dims, got {self.indexer.index.d}"
        
        # Test search
        query = np.random.randn(1536).astype('float32')
        results = self.indexer.search(query, k=3)
        
        assert len(results) > 0, "Search returned no results"
        assert all('item_id' in r for r in results), "Results missing item_id"
        assert all('similarity' in r for r in results), "Results missing similarity"
    
    def test_feature_file_integration(self):
        """Test feature file format and content"""
        features_file = Path("data/features.h5")
        if not features_file.exists():
            features_file = Path("data/features_1536.h5")
        
        if not features_file.exists():
            pytest.skip("No feature files found")
        
        with h5py.File(features_file, 'r') as hf:
            num_images = len(list(hf.keys()))
            assert num_images > 0, "Feature file is empty"
            
            # Check sample feature structure
            sample_key = list(hf.keys())[0]
            sample_group = hf[sample_key]
            
            assert 'clip' in sample_group, "CLIP features missing from file"
            assert 'dinov2' in sample_group, "DINOv2 features missing from file"
            
            clip_shape = sample_group['clip'].shape
            dino_shape = sample_group['dinov2'].shape
            
            assert clip_shape[0] == 768, f"Wrong CLIP dims in file: {clip_shape}"
            assert dino_shape[0] == 768, f"Wrong DINOv2 dims in file: {dino_shape}"
    
    def test_config_integration(self):
        """Test configuration consistency across components"""
        # Check feature extraction config
        feature_config = self.config['features']
        assert 'clip_model' in feature_config, "CLIP model not configured"
        assert feature_config.get('total_dimensions', 0) == 1536, "Wrong total dimensions"
        
        # Check indexing config
        index_config = self.config.get('indexing', {})
        assert 'index_type' in index_config, "Index type not configured"
        
        # Check recognition config
        recognition_config = self.config.get('recognition', {})
        if recognition_config:
            assert 'confidence_threshold' in recognition_config, "Confidence threshold not configured"
    
    def test_end_to_end_integration(self):
        """Test complete end-to-end pipeline"""
        # Find test image
        test_image_path = None
        for item_dir in Path("data/raw").iterdir():
            if item_dir.is_dir():
                for img_file in item_dir.glob("*.JPG"):
                    test_image_path = str(img_file)
                    break
            if test_image_path:
                break
        
        if not test_image_path:
            pytest.skip("No test images found")
        
        # 1. Extract features
        features = self.feature_extractor.extract_all_features(test_image_path)
        assert features is not None, "Feature extraction failed"
        
        # 2. Combine features (as done in recognition pipeline)
        clip_features = np.array(features['clip'])
        dino_features = np.array(features['dinov2'])
        combined_features = np.concatenate([clip_features, dino_features]).astype('float32')
        
        assert len(combined_features) == 1536, "Combined features wrong size"
        
        # 3. Load index and search
        index_path = "data/models/faiss_index.bin"
        metadata_path = "data/models/index_metadata.pkl"
        
        if Path(index_path).exists() and Path(metadata_path).exists():
            success = self.indexer.load_index(index_path, metadata_path)
            assert success, "Failed to load index"
            
            # 4. Perform search
            results = self.indexer.search(combined_features, k=1)
            assert len(results) > 0, "Search failed to return results"
            
            # 5. Validate result format
            result = results[0]
            assert 'item_id' in result, "Result missing item_id"
            assert 'similarity' in result, "Result missing similarity"
            assert isinstance(result['similarity'], (int, float)), "Similarity not numeric"


def run_integration_tests():
    """Run integration tests with detailed output"""
    print("🔧 AI Recognition System - Integration Tests")
    print("=" * 50)
    
    pytest_args = [
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ]
    
    result = pytest.main(pytest_args)
    return result == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)