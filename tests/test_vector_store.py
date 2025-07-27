#!/usr/bin/env python3
"""
Comprehensive Test Suite for HighPerformanceVectorStore

Tests all aspects of the high-performance vector storage system:
- Vector storage and retrieval operations
- Batch operations and throughput
- Integrity checking and corruption detection
- Performance benchmarks with real data
- Memory usage profiling
- Comparison with existing system baseline

Usage:
    python3 tests/test_vector_store.py                    # Run all tests
    python3 tests/test_vector_store.py --benchmark        # Run performance benchmarks
    python3 tests/test_vector_store.py --vectors=1000     # Benchmark with specific vector count
    python3 tests/test_vector_store.py --profile          # Memory profiling
    python3 tests/test_vector_store.py --compare          # Compare with baseline
"""

import os
import sys
import argparse
import unittest
import time
import logging
import statistics
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import test subjects
from unified_storage.vector_store import (
    HighPerformanceVectorStore, 
    create_high_performance_vector_store,
    LRUCache,
    VectorCompressor,
    IntegrityChecker
)
from unified_storage.sqlite_store import SQLiteVectorStore, create_vector_store
from unified_storage.config_manager import ConfigManager


class TestLRUCache(unittest.TestCase):
    """Test the LRU cache component."""
    
    def setUp(self):
        """Set up test cache."""
        self.cache = LRUCache(max_size=3, max_memory_mb=10)
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        from unified_storage.vector_store import CachedVector
        
        # Create test cached vectors
        vector1 = CachedVector(
            vector_id="test1",
            vector_data=np.random.random(100).astype(np.float32),
            compressed_size=400,
            access_count=1,
            last_access=time.time(),
            integrity_hash="hash1",
            metadata={}
        )
        
        # Test put and get
        self.assertTrue(self.cache.put("test1", vector1))
        retrieved = self.cache.get("test1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.vector_id, "test1")
        
        # Test cache miss
        self.assertIsNone(self.cache.get("nonexistent"))
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        from unified_storage.vector_store import CachedVector
        
        # Fill cache to capacity
        for i in range(3):
            vector = CachedVector(
                vector_id=f"test{i}",
                vector_data=np.random.random(100).astype(np.float32),
                compressed_size=400,
                access_count=1,
                last_access=time.time(),
                integrity_hash=f"hash{i}",
                metadata={}
            )
            self.cache.put(f"test{i}", vector)
        
        # Access test1 to make it most recently used
        self.cache.get("test1")
        
        # Add new item - should evict test0 (least recently used)
        vector3 = CachedVector(
            vector_id="test3",
            vector_data=np.random.random(100).astype(np.float32),
            compressed_size=400,
            access_count=1,
            last_access=time.time(),
            integrity_hash="hash3",
            metadata={}
        )
        self.cache.put("test3", vector3)
        
        # test0 should be evicted, others should remain
        self.assertIsNone(self.cache.get("test0"))
        self.assertIsNotNone(self.cache.get("test1"))
        self.assertIsNotNone(self.cache.get("test2"))
        self.assertIsNotNone(self.cache.get("test3"))
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        from unified_storage.vector_store import CachedVector
        
        vector = CachedVector(
            vector_id="test",
            vector_data=np.random.random(100).astype(np.float32),
            compressed_size=400,
            access_count=1,
            last_access=time.time(),
            integrity_hash="hash",
            metadata={}
        )
        
        # Test hit and miss tracking
        self.cache.put("test", vector)
        self.cache.get("test")  # Hit
        self.cache.get("nonexistent")  # Miss
        
        stats = self.cache.get_statistics()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertGreater(stats['hit_rate'], 0)


class TestVectorCompressor(unittest.TestCase):
    """Test the vector compression component."""
    
    def setUp(self):
        """Set up test compressor."""
        self.compressor = VectorCompressor()
    
    def test_compression_decompression(self):
        """Test vector compression and decompression."""
        # Create test vector
        original_vector = np.random.random(1536).astype(np.float32)
        
        # Compress
        compressed_data, compression_stats = self.compressor.compress_vector(original_vector)
        
        # Verify compression stats
        self.assertGreater(compression_stats.original_size, 0)
        self.assertGreater(compression_stats.compressed_size, 0)
        self.assertGreater(compression_stats.compression_time_ms, 0)
        
        # Decompress
        decompressed_vector, decompression_time = self.compressor.decompress_vector(
            compressed_data, original_vector.shape, original_vector.dtype
        )
        
        # Verify decompression
        self.assertEqual(decompressed_vector.shape, original_vector.shape)
        self.assertEqual(decompressed_vector.dtype, original_vector.dtype)
        np.testing.assert_array_almost_equal(original_vector, decompressed_vector, decimal=6)
        self.assertGreater(decompression_time, 0)
    
    def test_compression_efficiency(self):
        """Test compression efficiency with different vector types."""
        # Test with sparse vector (should compress well)
        sparse_vector = np.zeros(1536, dtype=np.float32)
        sparse_vector[:100] = np.random.random(100).astype(np.float32)
        
        compressed_data, stats = self.compressor.compress_vector(sparse_vector)
        
        # Sparse vectors should compress reasonably well
        self.assertLess(stats.compression_ratio, 1.0)  # Should achieve some compression
        self.assertGreater(len(compressed_data), 0)


class TestIntegrityChecker(unittest.TestCase):
    """Test the integrity checking component."""
    
    def setUp(self):
        """Set up test integrity checker."""
        self.checker = IntegrityChecker()
    
    def test_hash_computation(self):
        """Test hash computation."""
        test_vector = np.random.random(1536).astype(np.float32)
        
        # Compute hash
        hash1 = self.checker.compute_hash(test_vector)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64)  # SHA256 produces 64-character hex string
        
        # Same data should produce same hash
        hash2 = self.checker.compute_hash(test_vector)
        self.assertEqual(hash1, hash2)
        
        # Different data should produce different hash
        different_vector = test_vector + 0.1
        hash3 = self.checker.compute_hash(different_vector)
        self.assertNotEqual(hash1, hash3)
    
    def test_integrity_verification(self):
        """Test integrity verification."""
        test_vector = np.random.random(1536).astype(np.float32)
        correct_hash = self.checker.compute_hash(test_vector)
        
        # Valid data should pass verification
        self.assertTrue(self.checker.verify_integrity(test_vector, correct_hash))
        
        # Corrupted data should fail verification
        corrupted_vector = test_vector + 0.1
        self.assertFalse(self.checker.verify_integrity(corrupted_vector, correct_hash))
        
        # Check corruption detection tracking
        stats = self.checker.get_statistics()
        self.assertEqual(stats['corruption_detected'], 1)


class TestHighPerformanceVectorStore(unittest.TestCase):
    """Test the main HighPerformanceVectorStore class."""
    
    def setUp(self):
        """Set up test vector store."""
        # Create temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_dir, "test_vectors.db")
        
        # Create test store
        self.config_manager = ConfigManager()
        self.vector_store = HighPerformanceVectorStore(
            config_manager=self.config_manager,
            database_path=self.test_db_path
        )
    
    def tearDown(self):
        """Clean up test resources."""
        self.vector_store.close()
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_vector_storage_and_retrieval(self):
        """Test basic vector storage and retrieval."""
        # Create test vector
        vector_id = "test_vector_001"
        item_id = "test_item"
        vector_data = np.random.random(1536).astype(np.float32)
        metadata = {"test": True, "category": "unit_test"}
        
        # Store vector
        success = self.vector_store.store_vector(vector_id, item_id, vector_data, metadata)
        self.assertTrue(success)
        
        # Retrieve vector
        retrieved_vector = self.vector_store.get_vector(vector_id)
        self.assertIsNotNone(retrieved_vector)
        self.assertEqual(retrieved_vector.shape, vector_data.shape)
        self.assertEqual(retrieved_vector.dtype, vector_data.dtype)
        
        # Vectors should be approximately equal (allowing for compression artifacts)
        np.testing.assert_array_almost_equal(vector_data, retrieved_vector, decimal=5)
    
    def test_vector_storage_with_compression(self):
        """Test vector storage with compression enabled."""
        vector_id = "test_compressed"
        item_id = "test_item"
        vector_data = np.random.random(1536).astype(np.float32)
        
        # Store with compression
        success = self.vector_store.store_vector(
            vector_id, item_id, vector_data, 
            metadata={}, enable_compression=True
        )
        self.assertTrue(success)
        
        # Retrieve and verify
        retrieved_vector = self.vector_store.get_vector(vector_id, verify_integrity=True)
        self.assertIsNotNone(retrieved_vector)
        np.testing.assert_array_almost_equal(vector_data, retrieved_vector, decimal=5)
    
    def test_vector_storage_without_compression(self):
        """Test vector storage without compression."""
        vector_id = "test_uncompressed"
        item_id = "test_item"
        vector_data = np.random.random(1536).astype(np.float32)
        
        # Store without compression
        success = self.vector_store.store_vector(
            vector_id, item_id, vector_data, 
            metadata={}, enable_compression=False
        )
        self.assertTrue(success)
        
        # Retrieve and verify
        retrieved_vector = self.vector_store.get_vector(vector_id)
        self.assertIsNotNone(retrieved_vector)
        np.testing.assert_array_almost_equal(vector_data, retrieved_vector, decimal=7)
    
    def test_cache_functionality(self):
        """Test L1 cache functionality."""
        vector_id = "test_cache"
        item_id = "test_item"
        vector_data = np.random.random(1536).astype(np.float32)
        
        # Store vector
        self.vector_store.store_vector(vector_id, item_id, vector_data)
        
        # First retrieval should cache the vector
        retrieved1 = self.vector_store.get_vector(vector_id)
        self.assertIsNotNone(retrieved1)
        
        # Second retrieval should hit cache
        start_time = time.time()
        retrieved2 = self.vector_store.get_vector(vector_id)
        cache_time = time.time() - start_time
        
        self.assertIsNotNone(retrieved2)
        # Cache retrieval should be very fast
        self.assertLess(cache_time, 0.001)  # Less than 1ms
        
        # Verify cache statistics
        stats = self.vector_store.get_comprehensive_statistics()
        self.assertGreater(stats['performance']['cache_hit_rate'], 0)
    
    def test_batch_operations(self):
        """Test batch vector operations."""
        # Create batch of test vectors
        batch_size = 50
        batch_vectors = []
        
        for i in range(batch_size):
            vector_id = f"batch_test_{i:03d}"
            item_id = f"batch_item_{i % 10}"
            vector_data = np.random.random(1536).astype(np.float32)
            metadata = {"batch_index": i, "test_batch": True}
            batch_vectors.append((vector_id, item_id, vector_data, metadata))
        
        # Test batch insertion
        start_time = time.time()
        successful_inserts = self.vector_store.store_vectors_batch(batch_vectors)
        batch_time = time.time() - start_time
        
        # Verify batch results
        self.assertEqual(successful_inserts, batch_size)
        
        # Calculate throughput
        throughput = successful_inserts / batch_time if batch_time > 0 else 0
        self.assertGreater(throughput, 100)  # At least 100 vectors/second
        
        # Verify all vectors can be retrieved
        for vector_id, _, original_vector, _ in batch_vectors:
            retrieved_vector = self.vector_store.get_vector(vector_id)
            self.assertIsNotNone(retrieved_vector)
            np.testing.assert_array_almost_equal(original_vector, retrieved_vector, decimal=5)
    
    def test_integrity_checking(self):
        """Test integrity checking functionality."""
        vector_id = "test_integrity"
        item_id = "test_item"
        vector_data = np.random.random(1536).astype(np.float32)
        
        # Store vector
        self.vector_store.store_vector(vector_id, item_id, vector_data)
        
        # Retrieve with integrity verification
        retrieved_vector = self.vector_store.get_vector(vector_id, verify_integrity=True)
        self.assertIsNotNone(retrieved_vector)
        
        # Check integrity statistics
        integrity_stats = self.vector_store.integrity_checker.get_statistics()
        self.assertGreater(integrity_stats['hashes_computed'], 0)
        self.assertGreater(integrity_stats['hashes_verified'], 0)
        self.assertEqual(integrity_stats['corruption_detected'], 0)
    
    def test_vector_deletion(self):
        """Test vector deletion."""
        vector_id = "test_delete"
        item_id = "test_item"
        vector_data = np.random.random(1536).astype(np.float32)
        
        # Store vector
        self.vector_store.store_vector(vector_id, item_id, vector_data)
        
        # Verify it exists
        retrieved = self.vector_store.get_vector(vector_id)
        self.assertIsNotNone(retrieved)
        
        # Delete vector
        success = self.vector_store.delete_vector(vector_id)
        self.assertTrue(success)
        
        # Verify it's gone
        retrieved_after_delete = self.vector_store.get_vector(vector_id)
        self.assertIsNone(retrieved_after_delete)
    
    def test_get_vectors_by_item(self):
        """Test retrieving all vectors for a specific item."""
        item_id = "multi_vector_item"
        vector_count = 5
        
        # Store multiple vectors for the same item
        original_vectors = []
        for i in range(vector_count):
            vector_id = f"item_vector_{i}"
            vector_data = np.random.random(1536).astype(np.float32)
            original_vectors.append(vector_data)
            
            self.vector_store.store_vector(vector_id, item_id, vector_data)
        
        # Retrieve all vectors for the item
        retrieved_vectors = self.vector_store.get_vectors_by_item(item_id)
        
        # Verify count
        self.assertEqual(len(retrieved_vectors), vector_count)
        
        # Verify all vectors are present (order may vary)
        for retrieved in retrieved_vectors:
            found_match = False
            for original in original_vectors:
                try:
                    np.testing.assert_array_almost_equal(original, retrieved, decimal=5)
                    found_match = True
                    break
                except AssertionError:
                    continue
            self.assertTrue(found_match, "Retrieved vector doesn't match any original vector")


class PerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def __init__(self, vector_count: int = 1000):
        """Initialize benchmarks with specified vector count."""
        self.vector_count = vector_count
        self.test_dir = tempfile.mkdtemp()
        self.hp_test_db = os.path.join(self.test_dir, "hp_benchmark.db")
        self.baseline_test_db = os.path.join(self.test_dir, "baseline_benchmark.db")
        
        # Create stores for comparison
        self.config_manager = ConfigManager()
        self.hp_store = HighPerformanceVectorStore(
            config_manager=self.config_manager,
            database_path=self.hp_test_db
        )
        self.baseline_store = SQLiteVectorStore(
            config_manager=self.config_manager,
            database_path=self.baseline_test_db
        )
    
    def cleanup(self):
        """Clean up benchmark resources."""
        self.hp_store.close()
        self.baseline_store.close()
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def generate_test_vectors(self) -> List[Tuple[str, str, np.ndarray, Dict]]:
        """Generate test vectors for benchmarking."""
        vectors = []
        for i in range(self.vector_count):
            vector_id = f"bench_vector_{i:06d}"
            item_id = f"bench_item_{i % 100}"  # 100 items with multiple vectors each
            vector_data = np.random.random(1536).astype(np.float32)
            metadata = {
                "benchmark": True,
                "index": i,
                "category": f"cat_{i % 10}",
                "timestamp": time.time()
            }
            vectors.append((vector_id, item_id, vector_data, metadata))
        return vectors
    
    def benchmark_single_operations(self, vectors: List) -> Dict:
        """Benchmark single vector operations."""
        print(f"\n🔬 Benchmarking Single Operations ({len(vectors)} vectors)")
        print("-" * 60)
        
        # Test high-performance store
        hp_insert_times = []
        hp_retrieve_times = []
        
        print("   📦 Testing HighPerformanceVectorStore...")
        
        # Insert performance
        for vector_id, item_id, vector_data, metadata in vectors:
            start_time = time.time()
            success = self.hp_store.store_vector(vector_id, item_id, vector_data, metadata)
            insert_time = (time.time() - start_time) * 1000
            
            if success:
                hp_insert_times.append(insert_time)
        
        # Retrieval performance (clear cache first)
        self.hp_store.l1_cache.clear()
        
        for vector_id, _, _, _ in vectors[:100]:  # Test subset for retrieval
            start_time = time.time()
            retrieved = self.hp_store.get_vector(vector_id)
            retrieve_time = (time.time() - start_time) * 1000
            
            if retrieved is not None:
                hp_retrieve_times.append(retrieve_time)
        
        # Test baseline store for comparison
        baseline_insert_times = []
        baseline_retrieve_times = []
        
        print("   📦 Testing baseline SQLiteVectorStore...")
        
        from unified_storage.sqlite_store import VectorRecord
        
        # Convert vectors to VectorRecord format for baseline
        for vector_id, item_id, vector_data, metadata in vectors:
            record = VectorRecord(
                vector_id=vector_id,
                item_id=item_id,
                vector_data=vector_data,
                metadata=metadata,
                created_at=time.time(),
                updated_at=time.time()
            )
            
            start_time = time.time()
            success = self.baseline_store.insert_vector(record)
            insert_time = (time.time() - start_time) * 1000
            
            if success:
                baseline_insert_times.append(insert_time)
        
        # Baseline retrieval performance
        for vector_id, _, _, _ in vectors[:100]:
            start_time = time.time()
            retrieved = self.baseline_store.get_vector(vector_id)
            retrieve_time = (time.time() - start_time) * 1000
            
            if retrieved is not None:
                baseline_retrieve_times.append(retrieve_time)
        
        # Calculate statistics
        results = {
            'hp_store': {
                'insert_avg_ms': statistics.mean(hp_insert_times) if hp_insert_times else 0,
                'insert_p95_ms': statistics.quantiles(hp_insert_times, n=20)[18] if len(hp_insert_times) > 20 else 0,
                'retrieve_avg_ms': statistics.mean(hp_retrieve_times) if hp_retrieve_times else 0,
                'retrieve_p95_ms': statistics.quantiles(hp_retrieve_times, n=20)[18] if len(hp_retrieve_times) > 20 else 0,
                'successful_inserts': len(hp_insert_times),
                'successful_retrievals': len(hp_retrieve_times)
            },
            'baseline_store': {
                'insert_avg_ms': statistics.mean(baseline_insert_times) if baseline_insert_times else 0,
                'insert_p95_ms': statistics.quantiles(baseline_insert_times, n=20)[18] if len(baseline_insert_times) > 20 else 0,
                'retrieve_avg_ms': statistics.mean(baseline_retrieve_times) if baseline_retrieve_times else 0,
                'retrieve_p95_ms': statistics.quantiles(baseline_retrieve_times, n=20)[18] if len(baseline_retrieve_times) > 20 else 0,
                'successful_inserts': len(baseline_insert_times),
                'successful_retrievals': len(baseline_retrieve_times)
            }
        }
        
        # Print results
        print(f"\n   📊 HighPerformanceVectorStore Results:")
        print(f"      Insert: avg {results['hp_store']['insert_avg_ms']:.3f}ms, p95 {results['hp_store']['insert_p95_ms']:.3f}ms")
        print(f"      Retrieve: avg {results['hp_store']['retrieve_avg_ms']:.3f}ms, p95 {results['hp_store']['retrieve_p95_ms']:.3f}ms")
        
        print(f"\n   📊 Baseline SQLiteVectorStore Results:")
        print(f"      Insert: avg {results['baseline_store']['insert_avg_ms']:.3f}ms, p95 {results['baseline_store']['insert_p95_ms']:.3f}ms")
        print(f"      Retrieve: avg {results['baseline_store']['retrieve_avg_ms']:.3f}ms, p95 {results['baseline_store']['retrieve_p95_ms']:.3f}ms")
        
        # Performance improvement calculation
        if results['baseline_store']['insert_avg_ms'] > 0:
            insert_improvement = (results['baseline_store']['insert_avg_ms'] - results['hp_store']['insert_avg_ms']) / results['baseline_store']['insert_avg_ms'] * 100
            print(f"\n   🚀 Insert Performance Improvement: {insert_improvement:.1f}%")
        
        if results['baseline_store']['retrieve_avg_ms'] > 0:
            retrieve_improvement = (results['baseline_store']['retrieve_avg_ms'] - results['hp_store']['retrieve_avg_ms']) / results['baseline_store']['retrieve_avg_ms'] * 100
            print(f"   🚀 Retrieve Performance Improvement: {retrieve_improvement:.1f}%")
        
        return results
    
    def benchmark_batch_operations(self, vectors: List) -> Dict:
        """Benchmark batch operations."""
        print(f"\n🚀 Benchmarking Batch Operations")
        print("-" * 60)
        
        batch_sizes = [50, 100, 500]
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(vectors):
                continue
                
            print(f"\n   📦 Testing batch size: {batch_size}")
            
            # Test high-performance store batch
            batch_subset = vectors[:batch_size]
            
            start_time = time.time()
            hp_successful = self.hp_store.store_vectors_batch(batch_subset)
            hp_batch_time = time.time() - start_time
            hp_throughput = hp_successful / hp_batch_time if hp_batch_time > 0 else 0
            
            # Test baseline store batch
            from unified_storage.sqlite_store import VectorRecord
            baseline_records = []
            for vector_id, item_id, vector_data, metadata in batch_subset:
                record = VectorRecord(
                    vector_id=f"baseline_{vector_id}",
                    item_id=item_id,
                    vector_data=vector_data,
                    metadata=metadata,
                    created_at=time.time(),
                    updated_at=time.time()
                )
                baseline_records.append(record)
            
            start_time = time.time()
            baseline_successful = self.baseline_store.insert_vectors_batch(baseline_records)
            baseline_batch_time = time.time() - start_time
            baseline_throughput = baseline_successful / baseline_batch_time if baseline_batch_time > 0 else 0
            
            results[batch_size] = {
                'hp_throughput': hp_throughput,
                'baseline_throughput': baseline_throughput,
                'hp_time_ms': hp_batch_time * 1000,
                'baseline_time_ms': baseline_batch_time * 1000,
                'hp_successful': hp_successful,
                'baseline_successful': baseline_successful
            }
            
            print(f"      HighPerformance: {hp_throughput:.1f} vec/sec ({hp_batch_time*1000:.2f}ms)")
            print(f"      Baseline: {baseline_throughput:.1f} vec/sec ({baseline_batch_time*1000:.2f}ms)")
            
            if baseline_throughput > 0:
                improvement = (hp_throughput - baseline_throughput) / baseline_throughput * 100
                print(f"      Improvement: {improvement:.1f}%")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage."""
        print(f"\n🧠 Benchmarking Memory Usage")
        print("-" * 60)
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        print(f"   📊 Initial memory usage: {initial_memory_mb:.1f}MB")
        
        # Generate and store vectors
        test_vectors = self.generate_test_vectors()
        
        # Store vectors and monitor memory
        memory_samples = []
        for i, (vector_id, item_id, vector_data, metadata) in enumerate(test_vectors):
            self.hp_store.store_vector(vector_id, item_id, vector_data, metadata)
            
            if i % 100 == 0:  # Sample every 100 vectors
                current_memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_samples.append(current_memory_mb)
        
        final_memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_increase_mb = final_memory_mb - initial_memory_mb
        
        # Get cache statistics
        cache_stats = self.hp_store.l1_cache.get_statistics()
        
        print(f"   📊 Final memory usage: {final_memory_mb:.1f}MB")
        print(f"   📊 Memory increase: {memory_increase_mb:.1f}MB")
        print(f"   📊 Cache memory usage: {cache_stats['memory_usage_mb']:.1f}MB")
        print(f"   📊 Cache utilization: {cache_stats['memory_utilization']:.1f}%")
        
        # Memory efficiency calculation
        vector_memory_mb = (self.vector_count * 1536 * 4) / (1024 * 1024)  # Raw vector data size
        memory_overhead = (memory_increase_mb / vector_memory_mb) * 100 if vector_memory_mb > 0 else 0
        
        print(f"   📊 Raw vector data size: {vector_memory_mb:.1f}MB")
        print(f"   📊 Memory overhead: {memory_overhead:.1f}%")
        
        return {
            'initial_memory_mb': initial_memory_mb,
            'final_memory_mb': final_memory_mb,
            'memory_increase_mb': memory_increase_mb,
            'cache_memory_mb': cache_stats['memory_usage_mb'],
            'memory_overhead_percent': memory_overhead,
            'memory_samples': memory_samples
        }
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark suite."""
        print(f"🚀 Comprehensive Performance Benchmark")
        print(f"   Vector count: {self.vector_count}")
        print(f"   Platform: {self.config_manager.get_config().platform.platform_type}")
        print("=" * 80)
        
        # Generate test data
        test_vectors = self.generate_test_vectors()
        
        # Run benchmarks
        results = {}
        
        try:
            results['single_operations'] = self.benchmark_single_operations(test_vectors)
            results['batch_operations'] = self.benchmark_batch_operations(test_vectors)
            results['memory_usage'] = self.benchmark_memory_usage()
            
            # Get comprehensive statistics
            results['final_stats'] = self.hp_store.get_comprehensive_statistics()
            
            return results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {}


def run_unit_tests():
    """Run unit tests."""
    print("🧪 Running Unit Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(loader.loadTestsFromTestCase(TestLRUCache))
    test_suite.addTest(loader.loadTestsFromTestCase(TestVectorCompressor))
    test_suite.addTest(loader.loadTestsFromTestCase(TestIntegrityChecker))
    test_suite.addTest(loader.loadTestsFromTestCase(TestHighPerformanceVectorStore))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="HighPerformanceVectorStore Test Suite")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--vectors", type=int, default=1000, help="Number of vectors for benchmarks")
    parser.add_argument("--profile", action="store_true", help="Run memory profiling")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline")
    parser.add_argument("--unit-tests", action="store_true", default=True, help="Run unit tests")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    success = True
    
    # Run unit tests by default
    if args.unit_tests and not (args.benchmark or args.profile or args.compare):
        success = run_unit_tests()
    
    # Run benchmarks if requested
    if args.benchmark or args.profile or args.compare:
        benchmark = PerformanceBenchmarks(vector_count=args.vectors)
        
        try:
            results = benchmark.run_comprehensive_benchmark()
            
            if results:
                print("\n" + "=" * 80)
                print("📋 BENCHMARK SUMMARY")
                print("=" * 80)
                
                # Print key results
                if 'single_operations' in results:
                    hp_results = results['single_operations']['hp_store']
                    print(f"🎯 Performance Targets:")
                    print(f"   Insert time: {hp_results['insert_avg_ms']:.3f}ms (target: <0.5ms)")
                    print(f"   Retrieve time: {hp_results['retrieve_avg_ms']:.3f}ms (target: <0.5ms)")
                
                if 'batch_operations' in results:
                    max_throughput = max(r['hp_throughput'] for r in results['batch_operations'].values())
                    print(f"   Batch throughput: {max_throughput:.1f} vec/sec (target: >1000 vec/sec)")
                
                if 'memory_usage' in results:
                    memory_overhead = results['memory_usage']['memory_overhead_percent']
                    print(f"   Memory overhead: {memory_overhead:.1f}% (target: <50%)")
                
                # Save results to file
                results_file = "benchmark_results.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n💾 Results saved to: {results_file}")
            
        finally:
            benchmark.cleanup()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())