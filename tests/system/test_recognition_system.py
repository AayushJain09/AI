#!/usr/bin/env python3
"""
System Test: Production Recognition System
Tests the complete AI Recognition System end-to-end
"""

import sys
import time
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from inference.recognize import create_pipeline


class TestRecognitionSystem:
    """System-level tests for the complete recognition pipeline"""
    
    @classmethod
    def setup_class(cls):
        """Initialize the recognition pipeline once for all tests"""
        cls.pipeline = create_pipeline('config.yaml')
        
        # Test data paths
        cls.test_items = {
            'item_001': 'data/raw/item_001/Copy of IMG_8388.JPG',
            'item_002': 'data/raw/item_002/Copy of IMG_8403.JPG', 
            'item_003': 'data/raw/item_003/Copy of IMG_8428.JPG',
            'item_004': 'data/raw/item_004/1752950980813_IMG_8416.JPG'
        }
        
        cls.unknown_test_image = './environments/env2/lib/python3.13/site-packages/networkx/drawing/tests/baseline/test_display_empty_graph.png'
    
    def test_pipeline_initialization(self):
        """Test that the pipeline initializes correctly"""
        assert self.pipeline is not None
        assert hasattr(self.pipeline, 'index')
        assert hasattr(self.pipeline, 'recognize')
        
        # Check index properties
        assert self.pipeline.index.ntotal > 0
        assert self.pipeline.index.d == 1536  # CLIP(768) + DINOv2(768)
    
    @pytest.mark.parametrize("item_id,image_path", [
        ('item_001', 'data/raw/item_001/Copy of IMG_8388.JPG'),
        ('item_002', 'data/raw/item_002/Copy of IMG_8403.JPG'),
        ('item_003', 'data/raw/item_003/Copy of IMG_8428.JPG'),
        ('item_004', 'data/raw/item_004/1752950980813_IMG_8416.JPG')
    ])
    def test_known_item_recognition(self, item_id, image_path):
        """Test recognition of known items"""
        if not Path(image_path).exists():
            pytest.skip(f"Test image not found: {image_path}")
            
        start_time = time.time()
        result = self.pipeline.recognize(image_path)
        processing_time = time.time() - start_time
        
        # Assertions
        assert result.item_id == item_id, f"Expected {item_id}, got {result.item_id}"
        assert result.confidence > 1.0, f"Low confidence: {result.confidence}"
        assert processing_time < 1.0, f"Slow recognition: {processing_time:.3f}s"
    
    def test_unknown_item_rejection(self):
        """Test that unknown items are properly rejected"""
        if not Path(self.unknown_test_image).exists():
            pytest.skip(f"Unknown test image not found: {self.unknown_test_image}")
            
        result = self.pipeline.recognize(self.unknown_test_image)
        
        assert result.item_id == "unknown", f"Unknown item incorrectly recognized as {result.item_id}"
        assert result.confidence < 0.7, f"Unknown item confidence too high: {result.confidence}"
    
    def test_recognition_performance(self):
        """Test recognition performance across all items"""
        processing_times = []
        confidences = []
        
        for item_id, image_path in self.test_items.items():
            if not Path(image_path).exists():
                continue
                
            start_time = time.time()
            result = self.pipeline.recognize(image_path)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            if result.item_id != "unknown":
                confidences.append(result.confidence)
        
        # Performance assertions
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            assert avg_time < 0.5, f"Average processing time too slow: {avg_time:.3f}s"
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            assert avg_confidence > 1.0, f"Average confidence too low: {avg_confidence:.3f}"
    
    def test_system_accuracy(self):
        """Test overall system accuracy"""
        total_tests = 0
        correct_results = 0
        
        # Test known items
        for expected_item, image_path in self.test_items.items():
            if not Path(image_path).exists():
                continue
                
            result = self.pipeline.recognize(image_path)
            total_tests += 1
            
            if result.item_id == expected_item:
                correct_results += 1
        
        # Test unknown item
        if Path(self.unknown_test_image).exists():
            result = self.pipeline.recognize(self.unknown_test_image)
            total_tests += 1
            
            if result.item_id == "unknown":
                correct_results += 1
        
        # Calculate accuracy
        accuracy = correct_results / total_tests if total_tests > 0 else 0
        assert accuracy >= 1.0, f"System accuracy below 100%: {accuracy:.1%}"
    
    def test_index_properties(self):
        """Test FAISS index properties"""
        index = self.pipeline.index
        
        # Check index dimensions
        assert index.d == 1536, f"Expected 1536 dimensions, got {index.d}"
        
        # Check vector count (should have vectors for all items)
        assert index.ntotal >= 4, f"Expected at least 4 vectors for 4 items, got {index.ntotal}"
        
        # Verify index is trained (for some index types)
        if hasattr(index, 'is_trained'):
            assert index.is_trained, "Index is not trained"


def run_system_tests():
    """Run system tests with detailed output"""
    print("🚀 AI Recognition System - System Tests")
    print("=" * 50)
    
    # Run tests with pytest
    pytest_args = [
        __file__,
        '-v',  # verbose output
        '--tb=short',  # shorter traceback format
        '--color=yes'  # colored output
    ]
    
    result = pytest.main(pytest_args)
    return result == 0


if __name__ == "__main__":
    success = run_system_tests()
    sys.exit(0 if success else 1)