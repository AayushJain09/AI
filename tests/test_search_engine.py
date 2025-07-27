#!/usr/bin/env python3
"""
Comprehensive Test Suite for AdaptiveSearchEngine

Tests all aspects of the adaptive search engine:
- Search tier transitions (Linear -> FAISS Flat -> FAISS IVF)
- Platform-specific optimizations
- Search performance and accuracy
- Memory usage and scalability
- Incremental vector addition
- Query caching and optimization

Usage:
    python tests/test_search_engine.py                    # Run all tests
    python tests/test_search_engine.py --benchmark        # Run performance benchmarks
    python tests/test_search_engine.py --scalability      # Run scalability tests
"""

import os
import sys
import argparse
import unittest
import time
import logging
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import test subjects
from unified_storage.search_engine import (
    AdaptiveSearchEngine,
    LinearSearchTier,
    FAISSFlatTier,
    FAISSIVFTier,
    SearchResult,
    create_adaptive_search_engine
)
from unified_storage.vector_store import create_high_performance_vector_store
from unified_storage.config_manager import ConfigManager


class TestLinearSearchTier(unittest.TestCase):
    """Test the linear search tier component."""
    
    def setUp(self):
        """Set up test linear search tier."""
        self.linear_tier = LinearSearchTier("Apple_Silicon")
        
        # Create test vectors
        self.test_vectors = np.random.random((50, 1536)).astype(np.float32)
        self.test_metadata = [{'index': i, 'test': True} for i in range(50)]
    
    def test_initialization(self):
        """Test linear search tier initialization."""
        success = self.linear_tier.initialize(self.test_vectors, self.test_metadata)
        self.assertTrue(success)
        self.assertEqual(self.linear_tier.vector_count, 50)
        self.assertIsNotNone(self.linear_tier.vectors)
    
    def test_search_functionality(self):
        """Test linear search functionality."""
        self.linear_tier.initialize(self.test_vectors, self.test_metadata)
        
        # Test search with first vector as query
        query_vector = self.test_vectors[0]
        distances, indices = self.linear_tier.search(query_vector, k=5)
        
        # Should return 5 results
        self.assertEqual(len(distances), 5)
        self.assertEqual(len(indices), 5)
        
        # First result should be the query vector itself (distance ≈ 0)
        self.assertEqual(indices[0], 0)
        self.assertLess(distances[0], 0.01)
    
    def test_add_vectors(self):
        """Test adding vectors to linear search."""
        self.linear_tier.initialize(self.test_vectors[:25], self.test_metadata[:25])
        
        # Add more vectors
        new_vectors = self.test_vectors[25:]
        success = self.linear_tier.add_vectors(new_vectors)
        
        self.assertTrue(success)
        self.assertEqual(self.linear_tier.vector_count, 50)
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        self.linear_tier.initialize(self.test_vectors, self.test_metadata)
        
        memory_usage = self.linear_tier.get_memory_usage()
        self.assertGreater(memory_usage, 0)
        
        # Should be roughly the size of vectors
        expected_memory = (50 * 1536 * 4) / (1024 * 1024)  # 50 vectors * 1536 dims * 4 bytes
        self.assertAlmostEqual(memory_usage, expected_memory, delta=1.0)


class TestFAISSFlatTier(unittest.TestCase):
    """Test the FAISS Flat search tier component."""
    
    def setUp(self):
        """Set up test FAISS Flat tier."""
        self.faiss_tier = FAISSFlatTier("Apple_Silicon", 1536)
        
        # Create test vectors
        self.test_vectors = np.random.random((500, 1536)).astype(np.float32)
        self.test_metadata = [{'index': i, 'test': True} for i in range(500)]
    
    def test_initialization(self):
        """Test FAISS Flat tier initialization."""
        success = self.faiss_tier.initialize(self.test_vectors, self.test_metadata)
        self.assertTrue(success)
        self.assertEqual(self.faiss_tier.vector_count, 500)
        self.assertIsNotNone(self.faiss_tier.index)
    
    def test_search_functionality(self):
        """Test FAISS Flat search functionality."""
        self.faiss_tier.initialize(self.test_vectors, self.test_metadata)
        
        # Test search with first vector as query
        query_vector = self.test_vectors[0]
        distances, indices = self.faiss_tier.search(query_vector, k=10)
        
        # Should return 10 results
        self.assertEqual(len(distances), 10)
        self.assertEqual(len(indices), 10)
        
        # First result should be the query vector itself
        self.assertEqual(indices[0], 0)
        self.assertLess(distances[0], 0.01)
    
    def test_add_vectors(self):
        """Test adding vectors to FAISS Flat."""
        self.faiss_tier.initialize(self.test_vectors[:250], self.test_metadata[:250])
        
        # Add more vectors
        new_vectors = self.test_vectors[250:]
        success = self.faiss_tier.add_vectors(new_vectors)
        
        self.assertTrue(success)
        self.assertEqual(self.faiss_tier.vector_count, 500)


class TestFAISSIVFTier(unittest.TestCase):
    """Test the FAISS IVF search tier component."""
    
    def setUp(self):
        """Set up test FAISS IVF tier."""
        self.ivf_tier = FAISSIVFTier("Apple_Silicon", 1536)
        
        # Create larger test vectors for IVF
        self.test_vectors = np.random.random((5000, 1536)).astype(np.float32)
        self.test_metadata = [{'index': i, 'test': True} for i in range(5000)]
    
    def test_initialization(self):
        """Test FAISS IVF tier initialization."""
        success = self.ivf_tier.initialize(self.test_vectors, self.test_metadata)
        self.assertTrue(success)
        self.assertEqual(self.ivf_tier.vector_count, 5000)
        self.assertIsNotNone(self.ivf_tier.index)
        self.assertTrue(self.ivf_tier.index.is_trained)
    
    def test_search_functionality(self):
        """Test FAISS IVF search functionality."""
        self.ivf_tier.initialize(self.test_vectors, self.test_metadata)
        
        # Test search with first vector as query
        query_vector = self.test_vectors[0]
        distances, indices = self.ivf_tier.search(query_vector, k=10)
        
        # Should return up to 10 results (may be fewer with IVF)
        self.assertGreaterEqual(len(distances), 1)
        self.assertLessEqual(len(distances), 10)
        
        # Check that results are valid
        self.assertTrue(all(idx >= 0 for idx in indices))
        self.assertTrue(all(dist >= 0 for dist in distances))


class TestAdaptiveSearchEngine(unittest.TestCase):
    """Test the main AdaptiveSearchEngine class."""
    
    def setUp(self):
        """Set up test adaptive search engine."""
        # Create temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_dir, "test_search.db")
        
        # Create test vector store and search engine
        self.config_manager = ConfigManager()
        self.vector_store = create_high_performance_vector_store(
            config_manager=self.config_manager
        )
        self.search_engine = AdaptiveSearchEngine(
            vector_store=self.vector_store,
            config_manager=self.config_manager
        )
    
    def tearDown(self):
        """Clean up test resources."""
        self.search_engine.close()
        self.vector_store.close()
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initial_tier_selection(self):
        """Test initial tier selection for empty dataset."""
        # Should start with linear search for empty dataset
        self.assertEqual(self.search_engine.current_tier_type, 'linear')
    
    def test_tier_transition_to_faiss_flat(self):
        """Test automatic transition from linear to FAISS Flat."""
        # Add vectors to trigger transition
        for i in range(120):  # Above linear threshold
            test_vector = np.random.random(1536).astype(np.float32)
            success = self.search_engine.add_vector_incremental(f"item_{i}", test_vector)
            self.assertTrue(success)
        
        # Should transition to FAISS Flat
        self.assertEqual(self.search_engine.current_tier_type, 'faiss_flat')
        self.assertEqual(self.search_engine.vector_count, 120)
    
    def test_search_functionality(self):
        """Test search functionality across different tiers."""
        # Test with small dataset (linear)
        for i in range(10):
            test_vector = np.random.random(1536).astype(np.float32)
            self.search_engine.add_vector_incremental(f"small_item_{i}", test_vector)
        
        query_vector = np.random.random(1536).astype(np.float32)
        results = self.search_engine.search_adaptive(query_vector, k=5)
        
        self.assertLessEqual(len(results), 5)
        self.assertEqual(self.search_engine.current_tier_type, 'linear')
        
        # Add more vectors to trigger FAISS Flat
        for i in range(150):
            test_vector = np.random.random(1536).astype(np.float32)
            self.search_engine.add_vector_incremental(f"medium_item_{i}", test_vector)
        
        results = self.search_engine.search_adaptive(query_vector, k=10)
        self.assertEqual(self.search_engine.current_tier_type, 'faiss_flat')
    
    def test_search_result_format(self):
        """Test search result format and metadata."""
        # Add some test vectors
        for i in range(5):
            test_vector = np.random.random(1536).astype(np.float32)
            self.search_engine.add_vector_incremental(
                f"test_item_{i}", 
                test_vector, 
                metadata={'category': f'cat_{i}', 'test': True}
            )
        
        query_vector = np.random.random(1536).astype(np.float32)
        results = self.search_engine.search_adaptive(query_vector, k=3)
        
        # Check result format
        for result in results:
            self.assertIsInstance(result, SearchResult)
            self.assertIsInstance(result.item_id, str)
            self.assertIsInstance(result.distance, float)
            self.assertIsInstance(result.confidence, float)
            self.assertIsInstance(result.search_tier, str)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
    
    def test_search_caching(self):
        """Test search result caching."""
        # Add some vectors
        for i in range(10):
            test_vector = np.random.random(1536).astype(np.float32)
            self.search_engine.add_vector_incremental(f"cache_item_{i}", test_vector)
        
        query_vector = np.random.random(1536).astype(np.float32)
        
        # First search
        start_time = time.time()
        results1 = self.search_engine.search_adaptive(query_vector, k=5)
        first_time = time.time() - start_time
        
        # Second search (should hit cache)
        start_time = time.time()
        results2 = self.search_engine.search_adaptive(query_vector, k=5)
        second_time = time.time() - start_time
        
        # Cache hit should be faster
        self.assertLess(second_time, first_time)
        
        # Results should be the same
        self.assertEqual(len(results1), len(results2))
    
    def test_performance_statistics(self):
        """Test performance statistics collection."""
        # Add vectors and perform searches
        for i in range(20):
            test_vector = np.random.random(1536).astype(np.float32)
            self.search_engine.add_vector_incremental(f"perf_item_{i}", test_vector)
        
        # Perform multiple searches
        for _ in range(5):
            query_vector = np.random.random(1536).astype(np.float32)
            self.search_engine.search_adaptive(query_vector, k=3)
        
        # Check statistics
        stats = self.search_engine.get_comprehensive_statistics()
        
        self.assertIn('search_engine', stats)
        self.assertIn('current_tier_stats', stats)
        self.assertIn('cache_stats', stats)
        
        search_stats = stats['search_engine']
        self.assertGreater(search_stats['total_searches'], 0)
        self.assertGreater(search_stats['avg_search_time_ms'], 0)
        self.assertEqual(search_stats['vector_count'], 20)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        # Add vectors and check memory usage
        initial_stats = self.search_engine.get_comprehensive_statistics()
        initial_memory = initial_stats['current_tier_stats'].get('memory_usage_mb', 0)
        
        # Add more vectors
        for i in range(100):
            test_vector = np.random.random(1536).astype(np.float32)
            self.search_engine.add_vector_incremental(f"memory_item_{i}", test_vector)
        
        final_stats = self.search_engine.get_comprehensive_statistics()
        final_memory = final_stats['current_tier_stats'].get('memory_usage_mb', 0)
        
        # Memory usage should increase
        self.assertGreaterEqual(final_memory, initial_memory)


class SearchEngineBenchmarks:
    """Performance benchmark tests for search engine."""
    
    def __init__(self, vector_counts: List[int] = [100, 1000, 5000]):
        """Initialize benchmarks with specified vector counts."""
        self.vector_counts = vector_counts
        self.test_dir = tempfile.mkdtemp()
        
        # Create search engine
        self.config_manager = ConfigManager()
        self.vector_store = create_high_performance_vector_store(self.config_manager)
        self.search_engine = create_adaptive_search_engine(
            self.vector_store, self.config_manager
        )
    
    def cleanup(self):
        """Clean up benchmark resources."""
        self.search_engine.close()
        self.vector_store.close()
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def benchmark_tier_transitions(self) -> Dict:
        """Benchmark search performance across tier transitions."""
        print(f"\n🔀 Benchmarking Tier Transitions")
        print("-" * 50)
        
        results = {}
        
        for vector_count in self.vector_counts:
            print(f"\n   📊 Testing with {vector_count} vectors...")
            
            # Reset search engine for clean test
            self.search_engine.close()
            self.vector_store.close()
            
            self.vector_store = create_high_performance_vector_store(self.config_manager)
            self.search_engine = create_adaptive_search_engine(
                self.vector_store, self.config_manager
            )
            
            # Add vectors and measure time
            start_time = time.time()
            
            for i in range(vector_count):
                test_vector = np.random.random(1536).astype(np.float32)
                self.search_engine.add_vector_incremental(
                    f"bench_item_{i}", 
                    test_vector, 
                    metadata={'index': i, 'batch': vector_count}
                )
            
            build_time = time.time() - start_time
            
            # Perform search benchmark
            query_vector = np.random.random(1536).astype(np.float32)
            
            search_times = []
            for _ in range(10):  # 10 search queries
                start_time = time.time()
                results_list = self.search_engine.search_adaptive(query_vector, k=10)
                search_time = (time.time() - start_time) * 1000
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            
            # Get statistics
            stats = self.search_engine.get_comprehensive_statistics()
            tier_type = stats['search_engine']['current_tier']
            memory_usage = stats['current_tier_stats'].get('memory_usage_mb', 0)
            
            results[vector_count] = {
                'tier_type': tier_type,
                'build_time_ms': build_time * 1000,
                'avg_search_time_ms': avg_search_time,
                'memory_usage_mb': memory_usage,
                'tier_transitions': stats['search_engine']['tier_transitions']
            }
            
            print(f"      Tier: {tier_type}")
            print(f"      Build time: {build_time*1000:.2f}ms")
            print(f"      Avg search time: {avg_search_time:.2f}ms")
            print(f"      Memory usage: {memory_usage:.2f}MB")
            print(f"      Tier transitions: {stats['search_engine']['tier_transitions']}")
        
        return results
    
    def benchmark_search_accuracy(self) -> Dict:
        """Benchmark search accuracy across different tiers."""
        print(f"\n🎯 Benchmarking Search Accuracy")
        print("-" * 50)
        
        # Create ground truth with linear search
        test_vectors = np.random.random((1000, 1536)).astype(np.float32)
        query_vector = np.random.random(1536).astype(np.float32)
        
        # Linear search ground truth
        linear_tier = LinearSearchTier("Apple_Silicon")
        linear_tier.initialize(test_vectors, [{'index': i} for i in range(1000)])
        ground_truth_distances, ground_truth_indices = linear_tier.search(query_vector, k=50)
        
        # Test different tiers
        accuracy_results = {}
        
        # FAISS Flat
        flat_tier = FAISSFlatTier("Apple_Silicon", 1536)
        flat_tier.initialize(test_vectors, [{'index': i} for i in range(1000)])
        flat_distances, flat_indices = flat_tier.search(query_vector, k=50)
        
        # Calculate recall@50
        flat_recall = len(set(flat_indices) & set(ground_truth_indices)) / len(ground_truth_indices)
        accuracy_results['faiss_flat'] = {'recall_at_50': flat_recall}
        
        # FAISS IVF
        ivf_tier = FAISSIVFTier("Apple_Silicon", 1536)
        ivf_tier.initialize(test_vectors, [{'index': i} for i in range(1000)])
        ivf_distances, ivf_indices = ivf_tier.search(query_vector, k=50)
        
        # Calculate recall@50
        ivf_recall = len(set(ivf_indices) & set(ground_truth_indices)) / len(ground_truth_indices)
        accuracy_results['faiss_ivf'] = {'recall_at_50': ivf_recall}
        
        print(f"   📊 FAISS Flat Recall@50: {flat_recall:.3f}")
        print(f"   📊 FAISS IVF Recall@50: {ivf_recall:.3f}")
        
        return accuracy_results
    
    def run_comprehensive_benchmarks(self) -> Dict:
        """Run comprehensive benchmark suite."""
        print(f"🚀 Comprehensive Search Engine Benchmarks")
        print(f"   Platform: {self.config_manager.get_config().platform.platform_type}")
        print("=" * 70)
        
        results = {}
        
        try:
            results['tier_transitions'] = self.benchmark_tier_transitions()
            results['search_accuracy'] = self.benchmark_search_accuracy()
            
            return results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {}


def run_unit_tests():
    """Run unit tests for search engine."""
    print("🧪 Running Search Engine Unit Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(loader.loadTestsFromTestCase(TestLinearSearchTier))
    test_suite.addTest(loader.loadTestsFromTestCase(TestFAISSFlatTier))
    test_suite.addTest(loader.loadTestsFromTestCase(TestFAISSIVFTier))
    test_suite.addTest(loader.loadTestsFromTestCase(TestAdaptiveSearchEngine))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_tier_validation_test(dataset_size: int):
    """Run tier validation test with specific dataset size."""
    print(f"🎯 Tier Validation Test - Dataset Size: {dataset_size}")
    print("=" * 60)
    
    # Expected tier based on dataset size
    if dataset_size <= 100:
        expected_tier = 'linear'
    elif dataset_size <= 10000:
        expected_tier = 'faiss_flat'
    else:
        expected_tier = 'faiss_ivf'
    
    print(f"   Expected tier for {dataset_size} vectors: {expected_tier}")
    
    try:
        # Create search engine
        config_manager = ConfigManager()
        vector_store = create_high_performance_vector_store(config_manager)
        search_engine = create_adaptive_search_engine(vector_store, config_manager)
        
        # Add vectors progressively and monitor tier transitions
        print(f"   Adding {dataset_size} vectors...")
        
        tier_transitions = []
        batch_size = min(50, dataset_size // 10) if dataset_size > 100 else dataset_size
        
        for i in range(0, dataset_size, batch_size):
            # Add batch of vectors
            for j in range(batch_size):
                if i + j >= dataset_size:
                    break
                    
                vector_id = f"test_vector_{i + j:06d}"
                test_vector = np.random.random(1536).astype(np.float32)
                metadata = {'batch': i // batch_size, 'index': i + j}
                
                success = search_engine.add_vector_incremental(vector_id, test_vector, metadata)
                if not success:
                    print(f"   ⚠️  Failed to add vector {i + j}")
            
            # Check current tier
            current_tier = search_engine.current_tier_type
            vector_count = search_engine.vector_count
            
            if not tier_transitions or tier_transitions[-1]['tier'] != current_tier:
                tier_transitions.append({
                    'vector_count': vector_count,
                    'tier': current_tier,
                    'batch': i // batch_size
                })
                print(f"   📊 Vectors: {vector_count:>5} → Tier: {current_tier}")
        
        # Final validation
        final_tier = search_engine.current_tier_type
        final_count = search_engine.vector_count
        
        print(f"\n   🎯 Final Results:")
        print(f"      Added vectors: {final_count}/{dataset_size}")
        print(f"      Final tier: {final_tier}")
        print(f"      Expected tier: {expected_tier}")
        
        # Tier transition summary
        print(f"\n   🔀 Tier Transitions:")
        for i, transition in enumerate(tier_transitions):
            print(f"      {i+1}. {transition['vector_count']:>5} vectors → {transition['tier']}")
        
        # Performance test
        print(f"\n   ⚡ Performance Test:")
        test_query = np.random.random(1536).astype(np.float32)
        
        start_time = time.time()
        results = search_engine.search_adaptive(test_query, k=min(10, final_count))
        search_time_ms = (time.time() - start_time) * 1000
        
        print(f"      Search time: {search_time_ms:.2f}ms")
        print(f"      Results returned: {len(results)}")
        
        # Validate tier correctness
        tier_correct = final_tier == expected_tier
        
        # Index integrity validation
        integrity_results = search_engine.validate_index_integrity()
        
        print(f"\n   ✅ Validation Results:")
        print(f"      Tier correct: {'✅' if tier_correct else '❌'} ({final_tier} vs {expected_tier})")
        print(f"      Index integrity: {'✅' if integrity_results['status'] == 'healthy' else '⚠️'} ({integrity_results['status']})")
        print(f"      Search functional: {'✅' if len(results) > 0 or final_count == 0 else '❌'}")
        
        # Performance targets
        performance_targets = {
            'linear': 1.0,      # <1ms for linear search
            'faiss_flat': 5.0,  # <5ms for FAISS flat
            'faiss_ivf': 10.0   # <10ms for FAISS IVF
        }
        
        target_time = performance_targets.get(final_tier, 10.0)
        performance_ok = search_time_ms < target_time
        
        print(f"      Performance: {'✅' if performance_ok else '⚠️'} ({search_time_ms:.2f}ms < {target_time}ms)")
        
        if integrity_results['recommendations']:
            print(f"\n   💡 Recommendations:")
            for rec in integrity_results['recommendations']:
                print(f"      • {rec}")
        
        # Cleanup
        search_engine.close()
        vector_store.close()
        
        # Return overall success
        overall_success = tier_correct and integrity_results['status'] in ['healthy', 'warning'] and performance_ok
        
        if overall_success:
            print(f"\n✅ Tier validation test PASSED for {dataset_size} vectors")
        else:
            print(f"\n❌ Tier validation test FAILED for {dataset_size} vectors")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ Tier validation test ERROR: {e}")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="AdaptiveSearchEngine Test Suite")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--scalability", action="store_true", help="Run scalability tests")
    parser.add_argument("--unit-tests", action="store_true", default=True, help="Run unit tests")
    parser.add_argument("--size", type=int, help="Test with specific dataset size for tier validation")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    success = True
    
    # Run tier validation test if specific size requested
    if args.size:
        success = run_tier_validation_test(args.size)
        return 0 if success else 1
    
    # Run unit tests by default
    if args.unit_tests and not (args.benchmark or args.scalability):
        success = run_unit_tests()
    
    # Run benchmarks if requested
    if args.benchmark or args.scalability:
        vector_counts = [100, 1000, 5000] if args.scalability else [100, 500, 1000]
        benchmark = SearchEngineBenchmarks(vector_counts)
        
        try:
            results = benchmark.run_comprehensive_benchmarks()
            
            if results:
                print("\n" + "=" * 70)
                print("📋 BENCHMARK SUMMARY")
                print("=" * 70)
                
                # Print tier transition results
                if 'tier_transitions' in results:
                    print("\n🔀 Tier Transition Performance:")
                    for count, data in results['tier_transitions'].items():
                        print(f"   {count:>5} vectors: {data['tier_type']:>12} "
                              f"({data['avg_search_time_ms']:>6.2f}ms search)")
                
                # Print accuracy results
                if 'search_accuracy' in results:
                    print("\n🎯 Search Accuracy:")
                    for tier, data in results['search_accuracy'].items():
                        print(f"   {tier:>12}: {data['recall_at_50']:>6.3f} recall@50")
            
        finally:
            benchmark.cleanup()
    
    if success:
        print("\n✅ All search engine tests completed successfully!")
        return 0
    else:
        print("\n❌ Some search engine tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())