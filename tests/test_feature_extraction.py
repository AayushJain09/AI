#!/usr/bin/env python3
"""
Comprehensive test suite for cross-platform feature extraction.

This test suite validates the CrossPlatformFeatureExtractor across different
hardware configurations and ensures optimal performance on all supported platforms:

- NVIDIA GPU systems (Windows/Linux with CUDA)
- Apple Silicon systems (macOS with MPS)
- Intel/AMD CPU systems (Cross-platform CPU-only)

Test Coverage:
- Platform detection and device selection
- Model compilation and optimization
- Batch processing with memory management
- Performance benchmarking across platforms
- Error handling and graceful degradation
- Memory pressure scenarios and cleanup

Usage:
    python tests/test_feature_extraction.py --platform-test
    python -m pytest tests/test_feature_extraction.py -v
"""

import os
import sys
import time
import logging
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import gc

import numpy as np
from PIL import Image
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unified_storage.config_manager import ConfigManager
from unified_storage.platform_detector import PlatformDetector
from unified_storage.cross_platform_extractor import (
    CrossPlatformFeatureExtractor,
    ExtractionConfiguration,
    ExtractionStatistics,
    PlatformOptimizer,
    MemoryMonitor,
    create_cross_platform_extractor
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)


class TestPlatformDetection(unittest.TestCase):
    """Test platform detection and device selection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        self.platform_detector = PlatformDetector()
    
    def test_platform_detection_basic(self):
        """Test basic platform detection functionality."""
        config = self.platform_detector.get_platform_config()
        
        # Verify basic platform information is detected
        self.assertIsNotNone(config.platform_type)
        self.assertIsNotNone(config.os_name)
        self.assertGreater(config.cpu_cores, 0)
        self.assertGreater(config.memory_gb, 0)
        self.assertIn(config.device_type, ['cuda', 'mps', 'cpu'])
        
        logger.info(f"✓ Platform detected: {config.platform_type} with {config.device_type} device")
    
    def test_device_selection_priority(self):
        """Test device selection follows correct priority: CUDA > MPS > CPU."""
        config = self.platform_detector.get_platform_config()
        
        # Verify device selection logic
        if torch.cuda.is_available():
            expected_device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            expected_device = 'mps'
        else:
            expected_device = 'cpu'
        
        self.assertEqual(config.device_type, expected_device)
        logger.info(f"✓ Device selection correct: {expected_device}")
    
    def test_optimization_settings(self):
        """Test platform-specific optimization settings are applied."""
        config = self.platform_detector.get_platform_config()
        
        # Verify optimization settings are reasonable
        self.assertGreater(config.optimal_batch_size, 0)
        self.assertLessEqual(config.optimal_batch_size, 32)
        self.assertGreater(config.cache_size_mb, 0)
        self.assertGreater(config.faiss_threads, 0)
        self.assertIn(config.faiss_mode, ['gpu', 'cpu_optimized', 'cpu_standard'])
        
        logger.info(f"✓ Optimization settings: batch_size={config.optimal_batch_size}, "
                   f"cache={config.cache_size_mb}MB, faiss_mode={config.faiss_mode}")


class TestFeatureExtractorInitialization(unittest.TestCase):
    """Test feature extractor initialization and model loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        
        # Use conservative settings for testing
        self.extraction_config = ExtractionConfiguration(
            batch_size=2,  # Small batch for testing
            max_memory_usage_gb=2.0,  # Conservative memory limit
            compile_models=False,  # Skip compilation for faster tests
            use_mixed_precision=False,  # Skip mixed precision for compatibility
        )
    
    def test_extractor_initialization(self):
        """Test basic extractor initialization."""
        extractor = CrossPlatformFeatureExtractor(
            self.config_manager, 
            self.extraction_config
        )
        
        # Verify extractor is properly initialized
        self.assertIsNotNone(extractor.device)
        self.assertIsNotNone(extractor.models)
        self.assertIsNotNone(extractor.statistics)
        self.assertIsNotNone(extractor._memory_monitor)
        
        # Verify models are loaded
        self.assertIn('clip', extractor.models)
        self.assertIsNotNone(extractor.models['clip'])
        
        logger.info(f"✓ Extractor initialized on {extractor.device}")
        
        # Cleanup
        extractor.close()
    
    def test_model_loading_fallback(self):
        """Test model loading with fallback for missing models."""
        extractor = CrossPlatformFeatureExtractor(
            self.config_manager, 
            self.extraction_config
        )
        
        # CLIP should always be available
        self.assertIsNotNone(extractor.models.get('clip'))
        
        # DINOv2 might not be available, but should handle gracefully
        dinov2_available = extractor.models.get('dinov2') is not None
        logger.info(f"✓ DINOv2 availability: {dinov2_available}")
        
        # Cleanup
        extractor.close()
    
    def test_platform_optimizer_configuration(self):
        """Test platform optimizer configuration."""
        extractor = CrossPlatformFeatureExtractor(
            self.config_manager, 
            self.extraction_config
        )
        
        optimizer = extractor.platform_optimizer
        
        # Verify optimizer has proper configuration
        self.assertIsNotNone(optimizer.optimizations)
        self.assertIn('device_preference', optimizer.optimizations)
        self.assertIn('batch_size_multiplier', optimizer.optimizations)
        
        # Verify device configuration
        device, device_config = optimizer.get_device_configuration()
        self.assertIsNotNone(device)
        self.assertIsInstance(device_config, dict)
        
        logger.info(f"✓ Platform optimizer configured for {device}")
        
        # Cleanup
        extractor.close()


class TestMemoryMonitoring(unittest.TestCase):
    """Test memory monitoring and management features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device_type = 'cuda' if torch.cuda.is_available() else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
        self.memory_monitor = MemoryMonitor(max_memory_gb=4.0, device_type=self.device_type)
    
    def test_memory_info_collection(self):
        """Test memory information collection."""
        memory_info = self.memory_monitor.get_memory_info()
        
        # Verify required fields are present
        required_fields = ['rss_mb', 'vms_mb', 'percent', 'available_gb', 'total_gb', 'pressure_level', 'memory_trend']
        for field in required_fields:
            self.assertIn(field, memory_info)
            self.assertIsNotNone(memory_info[field])
        
        # Verify reasonable values
        self.assertGreater(memory_info['rss_mb'], 0)
        self.assertGreater(memory_info['available_gb'], 0)
        self.assertIn(memory_info['pressure_level'], ['normal', 'warning', 'critical', 'emergency'])
        
        logger.info(f"✓ Memory info: {memory_info['rss_mb']:.1f}MB used, "
                   f"{memory_info['available_gb']:.1f}GB available, "
                   f"pressure: {memory_info['pressure_level']}")
    
    def test_batch_size_suggestions(self):
        """Test batch size adjustment suggestions."""
        # Test various batch sizes
        for current_batch in [1, 4, 8, 16, 32]:
            suggested = self.memory_monitor.suggest_batch_size_adjustment(current_batch)
            self.assertGreaterEqual(suggested, 1)
            self.assertLessEqual(suggested, 32)
        
        logger.info("✓ Batch size adjustment suggestions working")
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        # Allocate some memory to clean up
        test_data = [np.random.randn(1000, 1000) for _ in range(10)]
        
        # Trigger cleanup
        cleanup_result = self.memory_monitor.trigger_memory_cleanup()
        
        # Verify cleanup result structure
        required_fields = ['cleanup_time_ms', 'actions_taken', 'success']
        for field in required_fields:
            self.assertIn(field, cleanup_result)
        
        self.assertIsInstance(cleanup_result['actions_taken'], list)
        self.assertGreater(len(cleanup_result['actions_taken']), 0)
        
        logger.info(f"✓ Memory cleanup: {cleanup_result['actions_taken']}, "
                   f"time: {cleanup_result['cleanup_time_ms']:.1f}ms")
        
        # Clean up test data
        del test_data
        gc.collect()


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction functionality across platforms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        
        # Create test extraction configuration
        self.extraction_config = ExtractionConfiguration(
            batch_size=4,
            max_memory_usage_gb=3.0,
            compile_models=False,  # Skip compilation for faster tests
            use_mixed_precision=False,
        )
        
        self.extractor = CrossPlatformFeatureExtractor(
            self.config_manager, 
            self.extraction_config
        )
        
        # Create test images
        self.test_images = self._create_test_images(5)
    
    def tearDown(self):
        """Clean up after tests."""
        self.extractor.close()
        # Clean up test images
        for img_path in self.test_images:
            if os.path.exists(img_path):
                os.unlink(img_path)
    
    def _create_test_images(self, count: int) -> List[str]:
        """Create test images for feature extraction."""
        test_images = []
        
        for i in range(count):
            # Create random test image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                test_images.append(tmp.name)
        
        return test_images
    
    def test_single_image_extraction(self):
        """Test single image feature extraction."""
        test_image = self.test_images[0]
        
        # Extract features
        start_time = time.time()
        features = self.extractor.extract_features_single(test_image)
        extraction_time = (time.time() - start_time) * 1000
        
        # Verify feature extraction
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 1536)  # CLIP (768) + DINOv2 (768)
        self.assertEqual(features.dtype, np.float32)
        
        # Verify feature normalization
        feature_norm = np.linalg.norm(features)
        self.assertAlmostEqual(feature_norm, 1.0, places=5)  # Should be normalized
        
        logger.info(f"✓ Single image extraction: {features.shape} features in {extraction_time:.1f}ms")
    
    def test_batch_extraction(self):
        """Test batch feature extraction."""
        # Extract features from batch
        start_time = time.time()
        features_list = self.extractor.extract_features_batch(self.test_images)
        extraction_time = (time.time() - start_time) * 1000
        
        # Verify batch results
        self.assertEqual(len(features_list), len(self.test_images))
        
        successful_extractions = sum(1 for f in features_list if f is not None)
        success_rate = successful_extractions / len(features_list)
        
        # Should have high success rate
        self.assertGreaterEqual(success_rate, 0.8)
        
        # Verify feature dimensions for successful extractions
        for features in features_list:
            if features is not None:
                self.assertEqual(features.shape[0], 1536)
                self.assertEqual(features.dtype, np.float32)
        
        avg_time_per_image = extraction_time / len(self.test_images)
        
        logger.info(f"✓ Batch extraction: {successful_extractions}/{len(features_list)} successful, "
                   f"{avg_time_per_image:.1f}ms per image")
    
    def test_memory_pressure_handling(self):
        """Test feature extraction under memory pressure."""
        # Create a large batch to stress memory
        large_batch = self.test_images * 10  # 50 images total
        
        # Monitor memory during extraction
        initial_memory = self.extractor._memory_monitor.get_memory_info()
        
        # Extract features
        features_list = self.extractor.extract_features_batch(large_batch)
        
        final_memory = self.extractor._memory_monitor.get_memory_info()
        memory_growth = final_memory['rss_mb'] - initial_memory['rss_mb']
        
        # Verify extraction completed without crashes
        self.assertEqual(len(features_list), len(large_batch))
        
        # Verify memory management
        successful_extractions = sum(1 for f in features_list if f is not None)
        self.assertGreater(successful_extractions, 0)
        
        logger.info(f"✓ Memory pressure test: {successful_extractions}/{len(large_batch)} successful, "
                   f"memory growth: +{memory_growth:.1f}MB")
    
    def test_performance_statistics(self):
        """Test performance statistics collection."""
        # Extract some features to generate statistics
        _ = self.extractor.extract_features_batch(self.test_images[:3])
        
        # Get performance statistics
        stats = self.extractor.get_performance_statistics()
        
        # Verify statistics structure
        self.assertIsInstance(stats, ExtractionStatistics)
        self.assertGreater(stats.total_images_processed, 0)
        self.assertGreaterEqual(stats.avg_extraction_time_ms, 0)
        self.assertGreaterEqual(stats.throughput_images_per_sec, 0)
        self.assertIsNotNone(stats.platform_type)
        self.assertIsNotNone(stats.device_used)
        
        logger.info(f"✓ Performance stats: {stats.total_images_processed} images, "
                   f"{stats.avg_extraction_time_ms:.1f}ms avg, "
                   f"{stats.throughput_images_per_sec:.1f} img/s")


class TestPlatformCompatibility(unittest.TestCase):
    """Test compatibility across different platforms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        self.platform_detector = PlatformDetector()
    
    def test_platform_compatibility_check(self):
        """Test platform compatibility validation."""
        compatibility = self.platform_detector.validate_platform_compatibility()
        
        # Verify compatibility check structure
        required_checks = ['python_version_ok', 'memory_sufficient', 'pytorch_available', 
                          'faiss_available', 'sqlite_available', 'performance_acceptable']
        
        for check in required_checks:
            self.assertIn(check, compatibility)
            self.assertIsInstance(compatibility[check], bool)
        
        # At minimum, Python and SQLite should be available
        self.assertTrue(compatibility['python_version_ok'])
        self.assertTrue(compatibility['sqlite_available'])
        
        failed_checks = [check for check, result in compatibility.items() if not result]
        
        if failed_checks:
            logger.warning(f"⚠️ Failed compatibility checks: {failed_checks}")
        else:
            logger.info("✅ All compatibility checks passed")
    
    def test_performance_estimation(self):
        """Test performance estimation for current platform."""
        summary = self.platform_detector.get_performance_summary()
        
        # Verify performance summary structure
        required_fields = ['platform_type', 'expected_recognition_time_s', 
                          'scalability', 'acceleration', 'optimization_level']
        
        for field in required_fields:
            self.assertIn(field, summary)
            self.assertIsNotNone(summary[field])
        
        # Verify reasonable values
        self.assertGreater(summary['expected_recognition_time_s'], 0)
        self.assertLess(summary['expected_recognition_time_s'], 5.0)  # Should be under 5 seconds
        self.assertIn(summary['acceleration'], ['cuda', 'mps', 'cpu'])
        
        logger.info(f"✓ Performance estimate: {summary['expected_recognition_time_s']:.2f}s recognition, "
                   f"{summary['scalability']} scalability, {summary['optimization_level']} optimization")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and graceful degradation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
    
    def test_invalid_image_handling(self):
        """Test handling of invalid or corrupted images."""
        extraction_config = ExtractionConfiguration(
            batch_size=2,
            compile_models=False,
        )
        
        extractor = CrossPlatformFeatureExtractor(self.config_manager, extraction_config)
        
        try:
            # Test with non-existent image
            features = extractor.extract_features_single("/nonexistent/image.jpg")
            self.assertIsNone(features)
            
            # Test with invalid image data
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                tmp.write(b"This is not an image")
                tmp.flush()
                
                features = extractor.extract_features_single(tmp.name)
                self.assertIsNone(features)
                
                os.unlink(tmp.name)
            
            logger.info("✓ Invalid image handling works correctly")
            
        finally:
            extractor.close()
    
    def test_low_memory_scenario(self):
        """Test behavior under low memory conditions."""
        # Create extractor with very low memory limit
        extraction_config = ExtractionConfiguration(
            batch_size=1,
            max_memory_usage_gb=0.1,  # Very low limit
            compile_models=False,
        )
        
        extractor = CrossPlatformFeatureExtractor(self.config_manager, extraction_config)
        
        try:
            # Create a test image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_image = Image.fromarray(img_array)
            
            # Should still work with single image
            features = extractor.extract_features_single(test_image)
            
            # May fail but shouldn't crash
            if features is not None:
                self.assertEqual(features.shape[0], 1536)
                logger.info("✓ Low memory scenario handled successfully")
            else:
                logger.info("✓ Low memory scenario handled gracefully (extraction failed but no crash)")
                
        finally:
            extractor.close()


def run_platform_benchmarks():
    """Run comprehensive platform benchmarks."""
    logger.info("🚀 Starting platform-specific benchmarks...")
    
    config_manager = ConfigManager()
    platform_detector = PlatformDetector()
    platform_config = platform_detector.get_platform_config()
    
    logger.info(f"📊 Benchmarking platform: {platform_config.platform_type}")
    logger.info(f"   Device: {platform_config.device_type}")
    logger.info(f"   CPU cores: {platform_config.cpu_cores}")
    logger.info(f"   Memory: {platform_config.memory_gb:.1f}GB")
    if platform_config.gpu_name:
        logger.info(f"   GPU: {platform_config.gpu_name}")
    
    # Create extractor with optimal settings
    extraction_config = ExtractionConfiguration(
        batch_size=platform_config.optimal_batch_size,
        compile_models=True,
        use_mixed_precision=True,
    )
    
    extractor = CrossPlatformFeatureExtractor(config_manager, extraction_config)
    
    try:
        # Create test images
        test_images = []
        for i in range(20):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_images.append(Image.fromarray(img_array))
        
        # Warm up the extractor
        logger.info("🔥 Warming up models...")
        _ = extractor.extract_features_single(test_images[0])
        
        # Benchmark single image extraction
        logger.info("📈 Benchmarking single image extraction...")
        single_times = []
        for i in range(10):
            start_time = time.time()
            features = extractor.extract_features_single(test_images[i])
            end_time = time.time()
            
            if features is not None:
                single_times.append((end_time - start_time) * 1000)
        
        avg_single_time = np.mean(single_times) if single_times else 0
        
        # Benchmark batch extraction
        logger.info("📦 Benchmarking batch extraction...")
        start_time = time.time()
        batch_features = extractor.extract_features_batch(test_images)
        end_time = time.time()
        
        batch_time = (end_time - start_time) * 1000
        successful_batch = sum(1 for f in batch_features if f is not None)
        avg_batch_time = batch_time / len(test_images) if test_images else 0
        
        # Get performance statistics
        stats = extractor.get_performance_statistics()
        
        # Report results
        logger.info("📊 Benchmark Results:")
        logger.info(f"   Single image: {avg_single_time:.1f}ms average")
        logger.info(f"   Batch processing: {avg_batch_time:.1f}ms per image")
        logger.info(f"   Throughput: {stats.throughput_images_per_sec:.1f} images/second")
        logger.info(f"   Success rate: {successful_batch}/{len(test_images)} ({successful_batch/len(test_images):.1%})")
        logger.info(f"   Peak memory: {stats.peak_memory_usage_mb:.1f}MB")
        logger.info(f"   CLIP inference: {stats.clip_inference_time_ms:.1f}ms")
        logger.info(f"   DINOv2 inference: {stats.dinov2_inference_time_ms:.1f}ms")
        
        # Performance assessment
        target_time = {"cuda": 150, "mps": 250, "cpu": 350}.get(platform_config.device_type, 350)
        performance_grade = "EXCELLENT" if avg_single_time < target_time * 0.8 else \
                           "GOOD" if avg_single_time < target_time else \
                           "ACCEPTABLE" if avg_single_time < target_time * 1.5 else "NEEDS_OPTIMIZATION"
        
        logger.info(f"🎯 Performance Grade: {performance_grade}")
        logger.info(f"   Target: <{target_time}ms, Actual: {avg_single_time:.1f}ms")
        
    finally:
        extractor.close()


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-Platform Feature Extraction Tests')
    parser.add_argument('--platform-test', action='store_true', 
                       help='Run platform compatibility test')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmarks')
    parser.add_argument('--unittest', action='store_true', 
                       help='Run unit tests')
    parser.add_argument('--all', action='store_true', 
                       help='Run all tests')
    
    args = parser.parse_args()
    
    if args.platform_test or args.all:
        logger.info("🧪 Running platform compatibility test...")
        
        # Quick platform test
        config_manager = ConfigManager()
        extraction_config = ExtractionConfiguration(
            batch_size=2,
            compile_models=False,
        )
        
        extractor = CrossPlatformFeatureExtractor(config_manager, extraction_config)
        
        try:
            # Create test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_pil = Image.fromarray(test_image)
            
            # Test feature extraction
            features = extractor.extract_features_single(test_pil)
            
            if features is not None:
                print(f"✅ Platform test PASSED - Features shape: {features.shape}")
                print(f"📊 Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
            else:
                print("❌ Platform test FAILED")
            
            # Print performance statistics
            stats = extractor.get_performance_statistics()
            print(f"\n📈 Performance Statistics:")
            print(f"   Platform: {stats.platform_type}")
            print(f"   Device: {stats.device_used}")
            print(f"   Hardware acceleration: {stats.hardware_acceleration}")
            print(f"   Extraction time: {stats.avg_extraction_time_ms:.2f}ms")
            
        finally:
            extractor.close()
    
    if args.benchmark or args.all:
        run_platform_benchmarks()
    
    if args.unittest or args.all:
        # Run unit tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    
    if not any([args.platform_test, args.benchmark, args.unittest, args.all]):
        print("ℹ️ No test specified. Use --platform-test, --benchmark, --unittest, or --all")
        print("📖 Available options:")
        print("   --platform-test: Quick platform compatibility test")
        print("   --benchmark: Performance benchmarks for current platform")
        print("   --unittest: Comprehensive unit test suite")
        print("   --all: Run all tests")


if __name__ == "__main__":
    main()