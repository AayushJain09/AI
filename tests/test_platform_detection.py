#!/usr/bin/env python3
"""
Test Platform Detection and Configuration Manager

Validates the core functionality of the unified storage infrastructure.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from unified_storage.platform_detector import PlatformDetector, detect_platform
from unified_storage.config_manager import ConfigManager, get_default_config


def test_platform_detection():
    """Test platform detection functionality"""
    print("🔍 Testing Platform Detection...")
    
    # Test basic detection
    detector = PlatformDetector()
    config = detector.get_platform_config()
    
    assert config.platform_type is not None
    assert config.cpu_cores > 0
    assert config.memory_gb > 0
    assert config.device_type in ['cuda', 'mps', 'cpu']
    assert config.faiss_mode in ['gpu', 'cpu_optimized', 'cpu_standard']
    
    print(f"✅ Platform: {config.platform_type}")
    print(f"✅ Device: {config.device_type}")
    print(f"✅ Memory: {config.memory_gb:.1f} GB")
    
    # Test convenience function
    quick_config = detect_platform()
    assert quick_config.platform_type == config.platform_type
    
    # Test performance summary
    summary = detector.get_performance_summary()
    assert 'expected_recognition_time_s' in summary
    assert summary['expected_recognition_time_s'] > 0
    
    print("✅ Platform detection tests passed")


def test_configuration_manager():
    """Test configuration manager functionality"""
    print("\n🔧 Testing Configuration Manager...")
    
    # Test basic configuration
    manager = ConfigManager(data_dir="test_data")
    config = manager.get_config()
    
    assert config.platform is not None
    assert config.database is not None
    assert config.vector_storage is not None
    assert config.search is not None
    assert config.performance is not None
    
    print(f"✅ SQLite Path: {config.database.sqlite_path}")
    print(f"✅ Cache Size: {config.vector_storage.cache_size_mb} MB")
    print(f"✅ FAISS Threads: {config.search.faiss_threads}")
    
    # Test validation
    validation = manager.validate_configuration()
    assert isinstance(validation, dict)
    
    # Test optimization summary
    summary = manager.get_optimization_summary()
    assert 'platform_type' in summary
    assert 'device_acceleration' in summary
    
    # Test convenience function
    default_config = get_default_config(data_dir="test_data")
    assert default_config.platform.platform_type == config.platform.platform_type
    
    print("✅ Configuration manager tests passed")


def test_platform_specific_optimizations():
    """Test platform-specific optimization logic"""
    print("\n⚡ Testing Platform-Specific Optimizations...")
    
    detector = PlatformDetector()
    config = detector.get_platform_config()
    
    # Test optimization logic based on platform type
    if config.device_type == 'cuda':
        assert config.faiss_mode == 'gpu'
        assert config.optimal_batch_size >= 16
        print("✅ NVIDIA GPU optimizations detected")
    
    elif config.device_type == 'mps':
        assert config.faiss_mode == 'cpu_optimized'
        assert config.faiss_threads <= 16
        print("✅ Apple Silicon optimizations detected")
    
    else:
        assert config.faiss_mode == 'cpu_standard'
        assert config.faiss_threads == config.cpu_cores
        print("✅ CPU optimizations detected")
    
    # Test memory-based optimizations
    if config.memory_gb >= 16:
        assert config.cache_size_mb >= 300
        print("✅ High memory optimizations applied")
    
    elif config.memory_gb >= 8:
        assert config.cache_size_mb >= 100
        print("✅ Medium memory optimizations applied")
    
    else:
        assert config.cache_size_mb <= 100
        print("✅ Low memory optimizations applied")
    
    print("✅ Platform-specific optimization tests passed")


def test_compatibility_validation():
    """Test compatibility validation"""
    print("\n🔍 Testing Compatibility Validation...")
    
    detector = PlatformDetector()
    compatibility = detector.validate_platform_compatibility()
    
    # Check required compatibility items
    required_checks = [
        'python_version_ok',
        'memory_sufficient', 
        'pytorch_available',
        'sqlite_available'
    ]
    
    for check in required_checks:
        assert check in compatibility
        if not compatibility[check]:
            print(f"⚠️  Warning: {check} failed")
        else:
            print(f"✅ {check}")
    
    print("✅ Compatibility validation tests passed")


def performance_benchmark():
    """Quick performance benchmark of the components"""
    print("\n📊 Performance Benchmark...")
    
    import time
    
    # Benchmark platform detection
    start = time.time()
    detector = PlatformDetector()
    config = detector.get_platform_config()
    detection_time = time.time() - start
    
    print(f"✅ Platform detection: {detection_time:.3f}s")
    
    # Benchmark configuration building
    start = time.time()
    manager = ConfigManager(data_dir="test_data")
    unified_config = manager.get_config()
    config_time = time.time() - start
    
    print(f"✅ Configuration building: {config_time:.3f}s")
    
    # Benchmark repeated access (should be cached)
    start = time.time()
    for _ in range(10):
        _ = manager.get_config()
    cache_time = (time.time() - start) / 10
    
    print(f"✅ Cached config access: {cache_time:.6f}s")
    
    # Performance should be acceptable
    assert detection_time < 1.0, "Platform detection too slow"
    assert config_time < 0.5, "Configuration building too slow"
    assert cache_time < 0.001, "Cached access too slow"
    
    print("✅ Performance benchmark passed")


def main():
    """Run all tests"""
    print("🚀 Testing Unified Storage Infrastructure")
    print("=" * 50)
    
    try:
        test_platform_detection()
        test_configuration_manager()
        test_platform_specific_optimizations()
        test_compatibility_validation()
        performance_benchmark()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Platform detection working correctly")
        print("✅ Configuration manager functioning properly")
        print("✅ Performance optimizations applied")
        print("✅ System ready for unified storage implementation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()