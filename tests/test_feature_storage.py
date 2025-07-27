#!/usr/bin/env python3
"""
Comprehensive Test Suite for MultiFormatFeatureStorage

Tests all aspects of the multi-format feature storage system:
- SQLite primary storage functionality
- HDF5 backup storage operations
- Metadata management and versioning
- Integrity validation and corruption detection
- Cross-storage consistency validation
- Performance benchmarks and optimization
- Platform-specific optimizations

Usage:
    python tests/test_feature_storage.py                    # Run all tests
    python tests/test_feature_storage.py --benchmark        # Run performance benchmarks
    python tests/test_feature_storage.py --features=1000    # Benchmark with specific feature count
    python tests/test_feature_storage.py --integrity        # Run integrity validation tests
"""

import os
import sys
import argparse
import unittest
import time
import tempfile
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import test subjects
from unified_storage.feature_storage import (
    MultiFormatFeatureStorage,
    FeatureRecord,
    IntegrityValidator,
    SQLiteFeatureStore,
    create_feature_storage
)
from unified_storage.config_manager import ConfigManager

# Check HDF5 availability
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


class TestIntegrityValidator(unittest.TestCase):
    """Test the integrity validation component."""
    
    def setUp(self):
        """Set up test integrity validator."""
        self.validator = IntegrityValidator()
    
    def test_checksum_computation(self):
        """Test checksum computation for different data types."""
        # Test with numpy array
        test_vector = np.random.random(1536).astype(np.float32)
        checksum1 = self.validator.compute_checksum(test_vector)
        
        self.assertIsInstance(checksum1, str)
        self.assertEqual(len(checksum1), 64)  # SHA256 produces 64-character hex
        
        # Same data should produce same checksum
        checksum2 = self.validator.compute_checksum(test_vector)
        self.assertEqual(checksum1, checksum2)
        
        # Different data should produce different checksum
        different_vector = test_vector + 0.1
        checksum3 = self.validator.compute_checksum(different_vector)
        self.assertNotEqual(checksum1, checksum3)
    
    def test_integrity_verification(self):
        """Test integrity verification functionality."""
        test_vector = np.random.random(1536).astype(np.float32)
        correct_checksum = self.validator.compute_checksum(test_vector)
        
        # Valid data should pass verification
        self.assertTrue(self.validator.verify_integrity(test_vector, correct_checksum))
        
        # Corrupted data should fail verification
        corrupted_vector = test_vector + 0.1
        self.assertFalse(self.validator.verify_integrity(corrupted_vector, correct_checksum))
        
        # Check statistics tracking
        stats = self.validator.get_statistics()
        self.assertGreater(stats['checksums_computed'], 0)
        self.assertGreater(stats['checksums_verified'], 0)
        self.assertGreater(stats['corruptions_detected'], 0)
    
    def test_storage_consistency_validation(self):
        """Test cross-storage consistency validation."""
        # Identical arrays should validate
        array1 = np.random.random(100).astype(np.float32)
        array2 = array1.copy()
        
        self.assertTrue(self.validator.validate_storage_consistency(array1, array2))
        
        # Different arrays should fail validation
        array3 = array1 + 0.1
        self.assertFalse(self.validator.validate_storage_consistency(array1, array3))
        
        # Shape mismatch should fail
        array4 = np.random.random(200).astype(np.float32)
        self.assertFalse(self.validator.validate_storage_consistency(array1, array4))


class TestSQLiteFeatureStore(unittest.TestCase):
    """Test the SQLite storage component."""
    
    def setUp(self):
        """Set up test SQLite store."""
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_dir, "test_features.db")
        self.config_manager = ConfigManager()
        self.sqlite_store = SQLiteFeatureStore(self.test_db_path, self.config_manager)
    
    def tearDown(self):
        """Clean up test resources."""
        self.sqlite_store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_feature_storage_and_retrieval(self):
        """Test basic feature storage and retrieval."""
        # Create test feature record
        test_vector = np.random.random(1536).astype(np.float32)
        test_metadata = {"test": True, "confidence": 0.95}
        
        record = FeatureRecord(
            feature_id="test_feature_001",
            item_id="test_item",
            model_version="test_v1.0",
            feature_vector=test_vector,
            extraction_time=time.time(),
            metadata=test_metadata,
            checksum="test_checksum",
            storage_format="v1.0",
            vector_shape=test_vector.shape,
            vector_dtype=str(test_vector.dtype),
            compression_used=False,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Store feature
        success = self.sqlite_store.insert_feature(record)
        self.assertTrue(success)
        
        # Retrieve feature
        retrieved_record = self.sqlite_store.get_feature("test_feature_001")
        self.assertIsNotNone(retrieved_record)
        self.assertEqual(retrieved_record.feature_id, "test_feature_001")
        self.assertEqual(retrieved_record.item_id, "test_item")
        self.assertEqual(retrieved_record.model_version, "test_v1.0")
        
        # Verify vector data
        np.testing.assert_array_equal(retrieved_record.feature_vector, test_vector)
        
        # Verify metadata
        self.assertEqual(retrieved_record.metadata["test"], True)
        self.assertEqual(retrieved_record.metadata["confidence"], 0.95)
    
    def test_features_by_item(self):
        """Test retrieving multiple features for an item."""
        item_id = "multi_feature_item"
        feature_count = 5
        
        # Store multiple features for the same item
        for i in range(feature_count):
            test_vector = np.random.random(1536).astype(np.float32)
            record = FeatureRecord(
                feature_id=f"feature_{i}",
                item_id=item_id,
                model_version=f"v{i}.0",
                feature_vector=test_vector,
                extraction_time=time.time() + i,  # Different timestamps
                metadata={"index": i},
                checksum=f"checksum_{i}",
                storage_format="v1.0",
                vector_shape=test_vector.shape,
                vector_dtype=str(test_vector.dtype),
                compression_used=False,
                created_at=time.time(),
                updated_at=time.time()
            )
            
            success = self.sqlite_store.insert_feature(record)
            self.assertTrue(success)
        
        # Retrieve all features for the item
        features = self.sqlite_store.get_features_by_item(item_id)
        self.assertEqual(len(features), feature_count)
        
        # Verify ordering (should be by extraction_time DESC)
        for i in range(len(features) - 1):
            self.assertGreaterEqual(features[i].extraction_time, features[i + 1].extraction_time)
    
    def test_feature_deletion(self):
        """Test feature deletion functionality."""
        # Store test feature
        test_vector = np.random.random(1536).astype(np.float32)
        record = FeatureRecord(
            feature_id="delete_test",
            item_id="test_item",
            model_version="v1.0",
            feature_vector=test_vector,
            extraction_time=time.time(),
            metadata={"test": "delete"},
            checksum="delete_checksum",
            storage_format="v1.0",
            vector_shape=test_vector.shape,
            vector_dtype=str(test_vector.dtype),
            compression_used=False,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.sqlite_store.insert_feature(record)
        
        # Verify it exists
        retrieved = self.sqlite_store.get_feature("delete_test")
        self.assertIsNotNone(retrieved)
        
        # Delete it
        success = self.sqlite_store.delete_feature("delete_test")
        self.assertTrue(success)
        
        # Verify it's gone
        retrieved_after_delete = self.sqlite_store.get_feature("delete_test")
        self.assertIsNone(retrieved_after_delete)
    
    def test_storage_statistics(self):
        """Test storage statistics collection."""
        # Store some test features
        for i in range(10):
            test_vector = np.random.random(1536).astype(np.float32)
            record = FeatureRecord(
                feature_id=f"stats_test_{i}",
                item_id=f"item_{i % 3}",
                model_version=f"v{i % 2}.0",
                feature_vector=test_vector,
                extraction_time=time.time(),
                metadata={"index": i},
                checksum=f"checksum_{i}",
                storage_format="v1.0",
                vector_shape=test_vector.shape,
                vector_dtype=str(test_vector.dtype),
                compression_used=False,
                created_at=time.time(),
                updated_at=time.time()
            )
            
            self.sqlite_store.insert_feature(record)
        
        # Get statistics
        stats = self.sqlite_store.get_storage_statistics()
        
        self.assertEqual(stats['total_features'], 10)
        self.assertGreater(stats['database_size_mb'], 0)
        self.assertIn('version_distribution', stats)
        self.assertIn('performance_stats', stats)
        
        # Check version distribution
        version_dist = stats['version_distribution']
        self.assertIn('v0.0', version_dist)
        self.assertIn('v1.0', version_dist)


@unittest.skipIf(not HDF5_AVAILABLE, "HDF5 not available")
class TestHDF5FeatureStore(unittest.TestCase):
    """Test the HDF5 storage component."""
    
    def setUp(self):
        """Set up test HDF5 store."""
        self.test_dir = tempfile.mkdtemp()
        self.test_hdf5_path = os.path.join(self.test_dir, "test_features.h5")
        self.config_manager = ConfigManager()
        
        from unified_storage.feature_storage import HDF5FeatureStore
        self.hdf5_store = HDF5FeatureStore(self.test_hdf5_path, self.config_manager)
    
    def tearDown(self):
        """Clean up test resources."""
        self.hdf5_store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_hdf5_storage_and_retrieval(self):
        """Test HDF5 feature storage and retrieval."""
        feature_id = "hdf5_test_001"
        test_vector = np.random.random(1536).astype(np.float32)
        test_metadata = {"test": True, "source": "hdf5_test"}
        
        # Store feature
        success = self.hdf5_store.store_feature_vector(feature_id, test_vector, test_metadata)
        self.assertTrue(success)
        
        # Retrieve feature
        result = self.hdf5_store.get_feature_vector(feature_id)
        self.assertIsNotNone(result)
        
        retrieved_vector, retrieved_metadata = result
        
        # Verify vector data
        np.testing.assert_array_equal(retrieved_vector, test_vector)
        
        # Verify metadata
        self.assertEqual(retrieved_metadata["test"], True)
        self.assertEqual(retrieved_metadata["source"], "hdf5_test")
    
    def test_hdf5_storage_info(self):
        """Test HDF5 storage information."""
        # Store some test features
        for i in range(5):
            feature_id = f"info_test_{i}"
            test_vector = np.random.random(512).astype(np.float32)
            test_metadata = {"index": i}
            
            self.hdf5_store.store_feature_vector(feature_id, test_vector, test_metadata)
        
        # Get storage info
        info = self.hdf5_store.get_storage_info()
        
        self.assertGreater(info['file_size_mb'], 0)
        self.assertEqual(info['feature_count'], 5)
        self.assertIn('features', info['groups'])
        self.assertIn('metadata', info['groups'])
        self.assertIn('performance_stats', info)


class TestMultiFormatFeatureStorage(unittest.TestCase):
    """Test the main MultiFormatFeatureStorage class."""
    
    def setUp(self):
        """Set up test multi-format storage."""
        self.test_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        self.storage = create_feature_storage(
            self.test_dir, 
            self.config_manager, 
            enable_hdf5=HDF5_AVAILABLE
        )
    
    def tearDown(self):
        """Clean up test resources."""
        self.storage.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_feature_storage_workflow(self):
        """Test complete feature storage workflow."""
        # Store feature
        test_vector = np.random.random(1536).astype(np.float32)
        test_metadata = {
            "source": "test_workflow",
            "confidence": 0.92,
            "extraction_method": "test"
        }
        
        feature_id = self.storage.store_feature(
            item_id="workflow_item",
            model_version="test_v1.0",
            feature_vector=test_vector,
            metadata=test_metadata
        )
        
        self.assertIsNotNone(feature_id)
        self.assertIsInstance(feature_id, str)
        
        # Retrieve feature
        retrieved_record = self.storage.get_feature(feature_id)
        self.assertIsNotNone(retrieved_record)
        self.assertEqual(retrieved_record.item_id, "workflow_item")
        self.assertEqual(retrieved_record.model_version, "test_v1.0")
        
        # Verify vector data
        np.testing.assert_array_almost_equal(
            retrieved_record.feature_vector, test_vector, decimal=6
        )
        
        # Verify metadata
        self.assertEqual(retrieved_record.metadata["source"], "test_workflow")
        self.assertEqual(retrieved_record.metadata["confidence"], 0.92)
    
    def test_multiple_features_per_item(self):
        """Test storing multiple features for the same item."""
        item_id = "multi_version_item"
        model_versions = ["v1.0", "v1.1", "v2.0"]
        
        stored_feature_ids = []
        
        # Store features with different model versions
        for version in model_versions:
            test_vector = np.random.random(1536).astype(np.float32)
            metadata = {"model_version": version, "test": True}
            
            feature_id = self.storage.store_feature(
                item_id=item_id,
                model_version=version,
                feature_vector=test_vector,
                metadata=metadata
            )
            
            self.assertIsNotNone(feature_id)
            stored_feature_ids.append(feature_id)
        
        # Retrieve all features for item
        features = self.storage.get_features_by_item(item_id)
        self.assertEqual(len(features), len(model_versions))
        
        # Verify model versions
        retrieved_versions = {f.model_version for f in features}
        expected_versions = set(model_versions)
        self.assertEqual(retrieved_versions, expected_versions)
    
    def test_integrity_validation(self):
        """Test integrity validation functionality."""
        # Store feature with integrity checking
        test_vector = np.random.random(1536).astype(np.float32)
        
        feature_id = self.storage.store_feature(
            item_id="integrity_test",
            model_version="v1.0",
            feature_vector=test_vector,
            enable_integrity_check=True
        )
        
        self.assertIsNotNone(feature_id)
        
        # Retrieve with integrity verification
        retrieved_record = self.storage.get_feature(feature_id, verify_integrity=True)
        self.assertIsNotNone(retrieved_record)
        
        # Verify checksum was computed
        self.assertNotEqual(retrieved_record.checksum, "")
    
    def test_storage_integrity_validation(self):
        """Test comprehensive storage integrity validation."""
        # Store several features
        for i in range(10):
            test_vector = np.random.random(512).astype(np.float32)
            metadata = {"index": i, "test": "integrity"}
            
            feature_id = self.storage.store_feature(
                item_id=f"integrity_item_{i}",
                model_version="v1.0",
                feature_vector=test_vector,
                metadata=metadata,
                enable_integrity_check=True
            )
            
            self.assertIsNotNone(feature_id)
        
        # Run integrity validation
        validation_results = self.storage.validate_storage_integrity(sample_ratio=1.0)
        
        self.assertIn('status', validation_results)
        self.assertIn('total_features', validation_results)
        self.assertIn('validated_features', validation_results)
        self.assertEqual(validation_results['total_features'], 10)
        
        # Should be healthy with no failures
        self.assertIn(validation_results['status'], ['healthy', 'warning'])
        self.assertEqual(validation_results['integrity_failures'], 0)
    
    @unittest.skipIf(not HDF5_AVAILABLE, "HDF5 not available")
    def test_hdf5_backup_creation(self):
        """Test HDF5 backup creation."""
        # Store features in SQLite
        for i in range(5):
            test_vector = np.random.random(256).astype(np.float32)
            feature_id = self.storage.store_feature(
                item_id=f"backup_item_{i}",
                model_version="v1.0",
                feature_vector=test_vector,
                enable_hdf5_backup=False  # Don't auto-backup
            )
            self.assertIsNotNone(feature_id)
        
        # Create HDF5 backup
        backup_success = self.storage.create_hdf5_backup(force=True)
        self.assertTrue(backup_success)
        
        # Verify backup was created
        hdf5_stats = self.storage.hdf5_store.get_storage_info()
        self.assertEqual(hdf5_stats['feature_count'], 5)
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics collection."""
        # Store some features
        for i in range(15):
            test_vector = np.random.random(128).astype(np.float32)
            feature_id = self.storage.store_feature(
                item_id=f"stats_item_{i}",
                model_version=f"v{i % 3}.0",
                feature_vector=test_vector
            )
            self.assertIsNotNone(feature_id)
        
        # Get statistics
        stats = self.storage.get_comprehensive_statistics()
        
        self.assertEqual(stats.total_features, 15)
        self.assertGreater(stats.sqlite_size_mb, 0)
        self.assertGreaterEqual(stats.avg_insert_time_ms, 0)
        self.assertEqual(stats.version_count, 3)  # v0.0, v1.0, v2.0
    
    def test_feature_deletion(self):
        """Test feature deletion across storage backends."""
        # Store feature
        test_vector = np.random.random(256).astype(np.float32)
        feature_id = self.storage.store_feature(
            item_id="delete_test_item",
            model_version="v1.0",
            feature_vector=test_vector
        )
        
        self.assertIsNotNone(feature_id)
        
        # Verify it exists
        retrieved = self.storage.get_feature(feature_id)
        self.assertIsNotNone(retrieved)
        
        # Delete it
        success = self.storage.delete_feature(feature_id)
        self.assertTrue(success)
        
        # Verify it's gone
        retrieved_after_delete = self.storage.get_feature(feature_id)
        self.assertIsNone(retrieved_after_delete)


class FeatureStorageBenchmarks:
    """Performance benchmark tests for feature storage."""
    
    def __init__(self, feature_count: int = 1000):
        """Initialize benchmarks with specified feature count."""
        self.feature_count = feature_count
        self.test_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        self.storage = create_feature_storage(
            self.test_dir, 
            self.config_manager, 
            enable_hdf5=HDF5_AVAILABLE
        )
    
    def cleanup(self):
        """Clean up benchmark resources."""
        self.storage.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def benchmark_storage_performance(self) -> Dict:
        """Benchmark feature storage performance."""
        print(f"\\n📦 Benchmarking Storage Performance ({self.feature_count} features)")
        print("-" * 60)
        
        results = {}
        
        # Generate test data
        test_features = []
        for i in range(self.feature_count):
            vector = np.random.random(1536).astype(np.float32)
            metadata = {
                "index": i,
                "confidence": np.random.random(),
                "source": f"benchmark_{i % 10}",
                "test": True
            }
            test_features.append((f"bench_item_{i}", f"v{i % 5}.0", vector, metadata))
        
        # Benchmark storage
        print("   📊 Testing storage operations...")
        
        storage_times = []
        for item_id, model_version, vector, metadata in test_features:
            start_time = time.time()
            
            feature_id = self.storage.store_feature(
                item_id=item_id,
                model_version=model_version,
                feature_vector=vector,
                metadata=metadata,
                enable_integrity_check=True
            )
            
            storage_time = (time.time() - start_time) * 1000
            
            if feature_id:
                storage_times.append(storage_time)
        
        # Benchmark retrieval
        print("   📊 Testing retrieval operations...")
        
        # Get some feature IDs for retrieval testing
        stats = self.storage.get_comprehensive_statistics()
        sample_size = min(100, self.feature_count)
        
        retrieval_times = []
        for i in range(sample_size):
            # Try to retrieve features by constructing likely IDs
            test_feature_id = None
            
            # Get features by item to find valid IDs
            features = self.storage.get_features_by_item(f"bench_item_{i}")
            if features:
                test_feature_id = features[0].feature_id
                
                start_time = time.time()
                retrieved = self.storage.get_feature(test_feature_id, verify_integrity=True)
                retrieval_time = (time.time() - start_time) * 1000
                
                if retrieved:
                    retrieval_times.append(retrieval_time)
        
        # Calculate statistics
        if storage_times:
            avg_storage_time = sum(storage_times) / len(storage_times)
            p95_storage_time = sorted(storage_times)[int(len(storage_times) * 0.95)]
        else:
            avg_storage_time = p95_storage_time = 0
        
        if retrieval_times:
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
            p95_retrieval_time = sorted(retrieval_times)[int(len(retrieval_times) * 0.95)]
        else:
            avg_retrieval_time = p95_retrieval_time = 0
        
        results = {
            'features_stored': len(storage_times),
            'features_retrieved': len(retrieval_times),
            'avg_storage_time_ms': avg_storage_time,
            'p95_storage_time_ms': p95_storage_time,
            'avg_retrieval_time_ms': avg_retrieval_time,
            'p95_retrieval_time_ms': p95_retrieval_time,
            'storage_throughput': len(storage_times) / (sum(storage_times) / 1000) if storage_times else 0
        }
        
        # Print results
        print(f"   📊 Storage Results:")
        print(f"      Features stored: {results['features_stored']}/{self.feature_count}")
        print(f"      Avg storage time: {avg_storage_time:.3f}ms")
        print(f"      P95 storage time: {p95_storage_time:.3f}ms")
        print(f"      Storage throughput: {results['storage_throughput']:.1f} features/sec")
        
        print(f"   📊 Retrieval Results:")
        print(f"      Features retrieved: {results['features_retrieved']}")
        print(f"      Avg retrieval time: {avg_retrieval_time:.3f}ms")
        print(f"      P95 retrieval time: {p95_retrieval_time:.3f}ms")
        
        return results
    
    def benchmark_integrity_validation(self) -> Dict:
        """Benchmark integrity validation performance."""
        print(f"\\n🔒 Benchmarking Integrity Validation")
        print("-" * 60)
        
        # Run comprehensive integrity validation
        start_time = time.time()
        validation_results = self.storage.validate_storage_integrity(sample_ratio=0.5)
        validation_time = (time.time() - start_time) * 1000
        
        print(f"   📊 Validation Results:")
        print(f"      Total features: {validation_results['total_features']}")
        print(f"      Validated features: {validation_results['validated_features']}")
        print(f"      Validation time: {validation_time:.2f}ms")
        print(f"      Status: {validation_results['status']}")
        print(f"      Integrity failures: {validation_results['integrity_failures']}")
        
        if validation_results.get('recommendations'):
            print(f"   💡 Recommendations:")
            for rec in validation_results['recommendations']:
                print(f"      • {rec}")
        
        return {
            'validation_time_ms': validation_time,
            'validation_status': validation_results['status'],
            'features_validated': validation_results['validated_features'],
            'integrity_failures': validation_results['integrity_failures']
        }
    
    @unittest.skipIf(not HDF5_AVAILABLE, "HDF5 not available")
    def benchmark_hdf5_backup(self) -> Dict:
        """Benchmark HDF5 backup performance."""
        print(f"\\n💾 Benchmarking HDF5 Backup")
        print("-" * 60)
        
        # Create HDF5 backup
        start_time = time.time()
        backup_success = self.storage.create_hdf5_backup(force=True)
        backup_time = (time.time() - start_time) * 1000
        
        # Get storage statistics
        storage_stats = self.storage.get_comprehensive_statistics()
        
        print(f"   📊 Backup Results:")
        print(f"      Backup successful: {'✅' if backup_success else '❌'}")
        print(f"      Backup time: {backup_time:.2f}ms")
        print(f"      SQLite size: {storage_stats.sqlite_size_mb:.2f}MB")
        print(f"      HDF5 size: {storage_stats.hdf5_size_mb:.2f}MB")
        
        if storage_stats.hdf5_size_mb > 0:
            compression_ratio = storage_stats.sqlite_size_mb / storage_stats.hdf5_size_mb
            print(f"      Compression ratio: {compression_ratio:.2f}x")
        
        return {
            'backup_success': backup_success,
            'backup_time_ms': backup_time,
            'sqlite_size_mb': storage_stats.sqlite_size_mb,
            'hdf5_size_mb': storage_stats.hdf5_size_mb
        }
    
    def run_comprehensive_benchmarks(self) -> Dict:
        """Run comprehensive benchmark suite."""
        print(f"🚀 Comprehensive Feature Storage Benchmarks")
        print(f"   Feature count: {self.feature_count}")
        print(f"   Platform: {self.config_manager.get_config().platform.platform_type}")
        print("=" * 80)
        
        results = {}
        
        try:
            results['storage_performance'] = self.benchmark_storage_performance()
            results['integrity_validation'] = self.benchmark_integrity_validation()
            
            if HDF5_AVAILABLE:
                results['hdf5_backup'] = self.benchmark_hdf5_backup()
            
            # Get final comprehensive statistics
            results['final_stats'] = self.storage.get_comprehensive_statistics()
            
            return results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {}


def run_unit_tests():
    """Run unit tests for feature storage."""
    print("🧪 Running Feature Storage Unit Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(loader.loadTestsFromTestCase(TestIntegrityValidator))
    test_suite.addTest(loader.loadTestsFromTestCase(TestSQLiteFeatureStore))
    
    if HDF5_AVAILABLE:
        test_suite.addTest(loader.loadTestsFromTestCase(TestHDF5FeatureStore))
    else:
        print("⚠️  HDF5 tests skipped - h5py not available")
    
    test_suite.addTest(loader.loadTestsFromTestCase(TestMultiFormatFeatureStorage))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="MultiFormatFeatureStorage Test Suite")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--features", type=int, default=1000, help="Number of features for benchmarks")
    parser.add_argument("--integrity", action="store_true", help="Run integrity validation tests")
    parser.add_argument("--unit-tests", action="store_true", default=True, help="Run unit tests")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    success = True
    
    # Run unit tests by default
    if args.unit_tests and not (args.benchmark or args.integrity):
        success = run_unit_tests()
    
    # Run benchmarks if requested
    if args.benchmark or args.integrity:
        benchmark = FeatureStorageBenchmarks(feature_count=args.features)
        
        try:
            results = benchmark.run_comprehensive_benchmarks()
            
            if results:
                print("\\n" + "=" * 80)
                print("📋 BENCHMARK SUMMARY")
                print("=" * 80)
                
                # Print key performance metrics
                if 'storage_performance' in results:
                    perf = results['storage_performance']
                    print(f"🎯 Storage Performance:")
                    print(f"   Avg storage time: {perf['avg_storage_time_ms']:.3f}ms")
                    print(f"   Storage throughput: {perf['storage_throughput']:.1f} features/sec")
                    print(f"   Avg retrieval time: {perf['avg_retrieval_time_ms']:.3f}ms")
                
                if 'integrity_validation' in results:
                    integrity = results['integrity_validation']
                    print(f"🔒 Integrity Validation:")
                    print(f"   Validation status: {integrity['validation_status']}")
                    print(f"   Features validated: {integrity['features_validated']}")
                    print(f"   Validation time: {integrity['validation_time_ms']:.2f}ms")
                
                if 'final_stats' in results:
                    stats = results['final_stats']
                    print(f"📊 Storage Statistics:")
                    print(f"   Total features: {stats.total_features}")
                    print(f"   SQLite size: {stats.sqlite_size_mb:.2f}MB")
                    print(f"   HDF5 size: {stats.hdf5_size_mb:.2f}MB")
                    print(f"   Version count: {stats.version_count}")
        
        finally:
            benchmark.cleanup()
    
    if success:
        print("\\n✅ All feature storage tests completed successfully!")
        return 0
    else:
        print("\\n❌ Some feature storage tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())