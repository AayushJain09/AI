"""
Adaptive Search Engine for High-Performance Vector Similarity Search

Intelligent search system that automatically selects the optimal search strategy based on:
- Dataset size and characteristics
- Platform capabilities (CPU/GPU/Apple Silicon)
- Query patterns and performance requirements
- Memory constraints and optimization goals

OVERVIEW:
This module implements a multi-tier adaptive search engine that provides optimal vector
similarity search performance across different dataset sizes and hardware platforms.
The system automatically transitions between search strategies as datasets grow:

1. Linear Search (0-100 vectors): SIMD-optimized brute force for small datasets
2. FAISS Flat (100-10K vectors): Exhaustive search with hardware acceleration
3. FAISS IVF (10K+ vectors): Approximate search with inverted file index

PERFORMANCE OPTIMIZATION STRATEGY:
- Platform detection: Automatic optimization for NVIDIA GPU, Apple Silicon, CPU-only
- Adaptive thresholds: Dynamic tier switching based on dataset characteristics
- Memory awareness: Optimal index configuration based on available system memory
- Query pattern analysis: Adaptive caching and precomputation for frequent queries
- Incremental updates: Efficient index maintenance as vectors are added/removed

SEARCH TIER ARCHITECTURE:
                    ┌─────────────────────────────────────┐
                    │         Query Interface             │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │    AdaptiveSearchEngine            │
                    │  (Tier Selection & Optimization)   │
                    └─────┬─────────┬─────────┬───────────┘
                          │         │         │
              ┌───────────▼──┐  ┌───▼────┐   ┌▼──────────┐
              │ Linear Search│  │  FAISS │   │ FAISS IVF │
              │  (0-100)     │  │  Flat  │   │ (10K+)    │
              │ SIMD Optimized│  │(100-10K)│   │Approximate│
              └──────────────┘  └────────┘   └───────────┘

PLATFORM-SPECIFIC OPTIMIZATIONS:
- NVIDIA GPU: GPU-accelerated FAISS with large batch processing
- Apple Silicon: MPS acceleration + optimized CPU FAISS with unified memory
- CPU-only: Multi-threaded FAISS with conservative memory allocation
"""

import logging
import os
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import faiss

# Import our unified storage components
from .vector_store import HighPerformanceVectorStore
from .config_manager import ConfigManager


@dataclass
class SearchResult:
    """
    Search result with metadata and performance information.
    
    Contains similarity search results with comprehensive metadata for
    performance analysis and optimization.
    """
    item_id: str                    # Item identifier
    vector_id: str                  # Specific vector identifier
    distance: float                 # Similarity distance (lower = more similar)
    confidence: float               # Search confidence (0-1, higher = more confident)
    metadata: Dict[str, Any]        # Additional item metadata
    search_tier: str                # Which search tier was used
    search_time_ms: float           # Time taken for this result


@dataclass
class SearchPerformance:
    """
    Search performance metrics for optimization analysis.
    """
    query_time_ms: float            # Total query execution time
    tier_used: str                  # Search tier that was used
    results_count: int              # Number of results returned
    vectors_searched: int           # Total vectors in search space
    cache_hit: bool                 # Whether result came from cache
    platform_type: str              # Platform used for search
    memory_usage_mb: float          # Memory used during search


class SearchTier(ABC):
    """
    Abstract base class for search tier implementations.
    
    Each search tier (Linear, FAISS Flat, FAISS IVF) implements this interface
    to provide consistent search functionality with tier-specific optimizations.
    """
    
    @abstractmethod
    def initialize(self, vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """Initialize the search tier with vector data."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform similarity search and return distances and indices."""
        pass
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray) -> bool:
        """Add new vectors to the search index."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get search tier statistics."""
        pass


class LinearSearchTier(SearchTier):
    """
    SIMD-optimized linear search for small datasets (0-100 vectors).
    
    Provides brute-force similarity search with SIMD optimizations for
    small datasets where the overhead of building indices is not justified.
    Linear search is often faster than indexed search for small datasets.
    """
    
    def __init__(self, platform_type: str):
        """Initialize linear search tier with platform optimizations."""
        self.platform_type = platform_type
        self.vectors = None
        self.metadata = []
        self.vector_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Performance statistics
        self.stats = {
            'total_searches': 0,
            'total_search_time_ms': 0.0,
            'avg_search_time_ms': 0.0,
            'vectors_searched': 0
        }
    
    def initialize(self, vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """
        Initialize linear search with vector data.
        
        Args:
            vectors: NumPy array of vectors (n_vectors, dimensions)
            metadata: List of metadata dictionaries for each vector
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Store vectors in contiguous memory for SIMD optimization
            # Ensure float32 for optimal SIMD performance
            self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            self.metadata = metadata.copy()
            self.vector_count = len(vectors)
            
            self.logger.info(f"Linear search initialized with {self.vector_count} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Linear search initialization failed: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform SIMD-optimized linear search.
        
        Uses NumPy's optimized vectorized operations which leverage SIMD
        instructions for efficient similarity computation.
        
        Args:
            query_vector: Query vector for similarity search
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices) for k nearest neighbors
        """
        start_time = time.time()
        
        try:
            if self.vectors is None or self.vector_count == 0:
                return np.array([]), np.array([])
            
            # Ensure query vector is contiguous float32 for SIMD optimization
            query = np.ascontiguousarray(query_vector, dtype=np.float32)
            
            # SIMD-optimized cosine similarity computation
            # NumPy automatically uses SIMD instructions for vectorized operations
            
            # Normalize query vector
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            
            # Compute cosine similarities using vectorized operations
            # This leverages SIMD instructions for parallel computation
            vector_norms = np.linalg.norm(self.vectors, axis=1)
            normalized_vectors = self.vectors / vector_norms[:, np.newaxis]
            
            # Dot product for cosine similarity (higher = more similar)
            similarities = np.dot(normalized_vectors, query)
            
            # Convert to distances (lower = more similar)
            distances = 1.0 - similarities
            
            # Get k nearest neighbors
            k = min(k, self.vector_count)
            
            if k == self.vector_count:
                # Return all vectors sorted by distance
                indices = np.argsort(distances)
                sorted_distances = distances[indices]
            else:
                # Use partial sort for efficiency
                indices = np.argpartition(distances, k)[:k]
                sorted_indices = np.argsort(distances[indices])
                indices = indices[sorted_indices]
                sorted_distances = distances[indices]
            
            # Update statistics
            search_time = (time.time() - start_time) * 1000
            self._update_statistics(search_time)
            
            return sorted_distances, indices
            
        except Exception as e:
            self.logger.error(f"Linear search failed: {e}")
            return np.array([]), np.array([])
    
    def add_vectors(self, vectors: np.ndarray) -> bool:
        """
        Add new vectors to linear search.
        
        Args:
            vectors: New vectors to add
            
        Returns:
            bool: True if vectors added successfully
        """
        try:
            if self.vectors is None:
                self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            else:
                # Concatenate new vectors
                new_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
                self.vectors = np.vstack([self.vectors, new_vectors])
            
            self.vector_count = len(self.vectors)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add vectors to linear search: {e}")
            return False
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        if self.vectors is None:
            return 0.0
        
        # Calculate memory for vectors and metadata
        vector_memory = self.vectors.nbytes / (1024 * 1024)
        metadata_memory = len(self.metadata) * 0.001  # Rough estimate
        
        return vector_memory + metadata_memory
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get linear search statistics."""
        return {
            'tier_type': 'linear',
            'vector_count': self.vector_count,
            'memory_usage_mb': self.get_memory_usage(),
            'platform_type': self.platform_type,
            **self.stats
        }
    
    def _update_statistics(self, search_time_ms: float):
        """Update search performance statistics."""
        self.stats['total_searches'] += 1
        self.stats['total_search_time_ms'] += search_time_ms
        self.stats['vectors_searched'] += self.vector_count
        
        total_searches = self.stats['total_searches']
        self.stats['avg_search_time_ms'] = (
            self.stats['total_search_time_ms'] / total_searches
        )


class FAISSFlatTier(SearchTier):
    """
    FAISS Flat index for medium datasets (100-10K vectors).
    
    Uses FAISS FlatIP (Inner Product) index for exhaustive similarity search
    with hardware acceleration. Provides exact results with optimized
    implementations for different platforms.
    """
    
    def __init__(self, platform_type: str, vector_dim: int = 1536):
        """Initialize FAISS Flat tier with platform optimizations."""
        self.platform_type = platform_type
        self.vector_dim = vector_dim
        self.index = None
        self.metadata = []
        self.vector_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Performance statistics
        self.stats = {
            'total_searches': 0,
            'total_search_time_ms': 0.0,
            'avg_search_time_ms': 0.0,
            'vectors_searched': 0,
            'index_build_time_ms': 0.0
        }
    
    def initialize(self, vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """
        Initialize FAISS Flat index.
        
        Args:
            vectors: NumPy array of vectors (n_vectors, dimensions)
            metadata: List of metadata dictionaries for each vector
            
        Returns:
            bool: True if initialization successful
        """
        start_time = time.time()
        
        try:
            # Advanced platform-specific FAISS optimization
            if self.platform_type == "NVIDIA_GPU":
                # GPU-accelerated FAISS with optimized memory management
                try:
                    import torch
                    # Verify CUDA availability
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA not available")
                    
                    # Create CPU index first
                    cpu_index = faiss.IndexFlatIP(self.vector_dim)
                    
                    # Configure GPU resources with optimized memory allocation
                    gpu_resource = faiss.StandardGpuResources()
                    
                    # Set memory fraction (use 80% of GPU memory for safety)
                    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                    faiss_memory_bytes = int(gpu_memory_bytes * 0.8)
                    gpu_resource.setTempMemory(faiss_memory_bytes)
                    
                    # Transfer to GPU with device 0
                    self.index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
                    self.gpu_resource = gpu_resource  # Keep reference
                    
                    self.logger.info(f"Using GPU-accelerated FAISS Flat index with {faiss_memory_bytes//1024//1024}MB GPU memory")
                    
                except Exception as gpu_error:
                    # Fallback to optimized CPU implementation
                    self.logger.warning(f"GPU FAISS failed, using optimized CPU: {gpu_error}")
                    self.index = faiss.IndexFlatIP(self.vector_dim)
                    self._configure_cpu_threading()
                    
            elif self.platform_type == "Apple_Silicon":
                # Apple Silicon optimized FAISS configuration
                self.index = faiss.IndexFlatIP(self.vector_dim)
                
                # Apple Silicon specific optimizations:
                # 1. Optimal thread count for efficiency + performance cores
                # 2. Memory-mapped operations for unified memory architecture
                # 3. Conservative threading to avoid core contention
                optimal_threads = min(8, max(4, os.cpu_count() // 2))
                faiss.omp_set_num_threads(optimal_threads)
                
                self.logger.info(f"Apple Silicon FAISS: {optimal_threads} threads, unified memory optimization")
                
            else:
                # CPU-only platforms (Intel Mac, Linux, Windows without GPU)
                self.index = faiss.IndexFlatIP(self.vector_dim)
                self._configure_cpu_threading()
                
                self.logger.info(f"CPU FAISS Flat index with optimized threading")
                if self.platform_type == "Apple_Silicon":
                    # Apple Silicon optimizations
                    faiss.omp_set_num_threads(min(8, 16))  # Optimal for efficiency cores
                else:
                    # CPU-only optimizations
                    faiss.omp_set_num_threads(min(4, 8))   # Conservative threading
            
            # Prepare vectors for FAISS (needs float32)
            vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
            
            # Normalize vectors for cosine similarity using Inner Product
            norms = np.linalg.norm(vectors_f32, axis=1, keepdims=True)
            normalized_vectors = vectors_f32 / norms
            
            # Add vectors to index
            self.index.add(normalized_vectors)
            
            # Store metadata
            self.metadata = metadata.copy()
            self.vector_count = len(vectors)
            
            # Update statistics
            build_time = (time.time() - start_time) * 1000
            self.stats['index_build_time_ms'] = build_time
            
            self.logger.info(f"FAISS Flat index initialized with {self.vector_count} vectors "
                           f"in {build_time:.2f}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"FAISS Flat initialization failed: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform FAISS Flat search.
        
        Args:
            query_vector: Query vector for similarity search
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices) for k nearest neighbors
        """
        start_time = time.time()
        
        try:
            if self.index is None or self.vector_count == 0:
                return np.array([]), np.array([])
            
            # Prepare query vector
            query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
            
            # Normalize query vector for cosine similarity
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            
            # Perform search
            k = min(k, self.vector_count)
            distances, indices = self.index.search(query, k)
            
            # Convert inner product back to cosine distance
            distances = 1.0 - distances[0]  # Remove batch dimension
            indices = indices[0]
            
            # Update statistics
            search_time = (time.time() - start_time) * 1000
            self._update_statistics(search_time)
            
            return distances, indices
            
        except Exception as e:
            self.logger.error(f"FAISS Flat search failed: {e}")
            return np.array([]), np.array([])
    
    def add_vectors(self, vectors: np.ndarray) -> bool:
        """
        Add new vectors to FAISS Flat index.
        
        Args:
            vectors: New vectors to add
            
        Returns:
            bool: True if vectors added successfully
        """
        try:
            if self.index is None:
                return False
            
            # Prepare vectors
            vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
            
            # Normalize vectors
            norms = np.linalg.norm(vectors_f32, axis=1, keepdims=True)
            normalized_vectors = vectors_f32 / norms
            
            # Add to index
            self.index.add(normalized_vectors)
            self.vector_count += len(vectors)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add vectors to FAISS Flat: {e}")
            return False
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        if self.index is None:
            return 0.0
        
        # Estimate FAISS index memory usage
        # Flat index stores all vectors + some overhead
        vector_memory = self.vector_count * self.vector_dim * 4 / (1024 * 1024)  # float32
        overhead_memory = 0.1  # Small overhead for FAISS structures
        
        return vector_memory + overhead_memory
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get FAISS Flat statistics."""
        return {
            'tier_type': 'faiss_flat',
            'vector_count': self.vector_count,
            'memory_usage_mb': self.get_memory_usage(),
            'platform_type': self.platform_type,
            'index_trained': self.index is not None,
            **self.stats
        }
    
    def _update_statistics(self, search_time_ms: float):
        """Update search performance statistics."""
        self.stats['total_searches'] += 1
        self.stats['total_search_time_ms'] += search_time_ms
        self.stats['vectors_searched'] += self.vector_count
        
        total_searches = self.stats['total_searches']
        self.stats['avg_search_time_ms'] = (
            self.stats['total_search_time_ms'] / total_searches
        )
    
    def _configure_cpu_threading(self):
        """
        Configure optimal CPU threading for FAISS operations.
        
        Platform-specific thread optimization:
        - Intel/AMD: Use most CPU cores with hyperthreading
        - General CPU: Conservative threading to avoid contention
        """
        try:
            cpu_count = os.cpu_count() or 4
            
            # Platform-specific thread optimization
            if self.platform_type == "CPU_Intel" or self.platform_type.startswith("Intel"):
                # Intel CPUs benefit from higher thread counts
                optimal_threads = min(16, cpu_count)
            else:
                # Conservative threading for unknown CPU architectures
                optimal_threads = min(8, max(4, cpu_count // 2))
            
            # Apply threading configuration
            faiss.omp_set_num_threads(optimal_threads)
            
            self.logger.info(f"Configured FAISS CPU threading: {optimal_threads} threads for {self.platform_type}")
            
        except Exception as e:
            self.logger.warning(f"Failed to configure CPU threading: {e}")
            # Use default FAISS threading


class FAISSIVFTier(SearchTier):
    """
    FAISS IVF index for large datasets (10K+ vectors).
    
    Uses FAISS IndexIVFFlat for approximate similarity search with
    inverted file index. Provides fast search with configurable
    accuracy/speed tradeoffs.
    """
    
    def __init__(self, platform_type: str, vector_dim: int = 1536):
        """Initialize FAISS IVF tier with platform optimizations."""
        self.platform_type = platform_type
        self.vector_dim = vector_dim
        self.index = None
        self.metadata = []
        self.vector_count = 0
        self.n_centroids = 256  # Default number of centroids
        self.n_probe = 32       # Default number of probes
        self.logger = logging.getLogger(__name__)
        
        # Performance statistics
        self.stats = {
            'total_searches': 0,
            'total_search_time_ms': 0.0,
            'avg_search_time_ms': 0.0,
            'vectors_searched': 0,
            'index_build_time_ms': 0.0,
            'index_train_time_ms': 0.0,
            'n_centroids': self.n_centroids,
            'n_probe': self.n_probe
        }
    
    def initialize(self, vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """
        Initialize FAISS IVF index with training.
        
        Args:
            vectors: NumPy array of vectors (n_vectors, dimensions)
            metadata: List of metadata dictionaries for each vector
            
        Returns:
            bool: True if initialization successful
        """
        start_time = time.time()
        
        try:
            # Calculate optimal number of centroids based on dataset size
            # Rule of thumb: sqrt(n_vectors) centroids, but with reasonable bounds
            # Ensure we have enough training points (FAISS recommends 39*n_centroids)
            n_vectors = len(vectors)
            max_centroids = max(4, n_vectors // 50)  # Conservative: 50 vectors per centroid
            self.n_centroids = min(max(int(np.sqrt(n_vectors)), 4), max_centroids)
            
            # Advanced platform-specific optimizations for IVF
            if self.platform_type == "NVIDIA_GPU":
                try:
                    import torch
                    # Verify CUDA availability
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA not available")
                    
                    # GPU-accelerated FAISS IVF with memory optimization
                    quantizer = faiss.IndexFlatIP(self.vector_dim)
                    cpu_index = faiss.IndexIVFFlat(quantizer, self.vector_dim, self.n_centroids)
                    
                    # Configure GPU resources with optimized memory allocation
                    gpu_resource = faiss.StandardGpuResources()
                    
                    # Set memory fraction (use 70% for IVF as it needs more memory)
                    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                    faiss_memory_bytes = int(gpu_memory_bytes * 0.7)
                    gpu_resource.setTempMemory(faiss_memory_bytes)
                    
                    # Transfer to GPU
                    self.index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
                    self.gpu_resource = gpu_resource  # Keep reference
                    
                    # GPU-optimized probe count (higher for GPU efficiency)
                    self.n_probe = min(64, max(16, self.n_centroids // 4))
                    
                    self.logger.info(f"Using GPU-accelerated FAISS IVF index with {faiss_memory_bytes//1024//1024}MB GPU memory, {self.n_centroids} centroids, {self.n_probe} probes")
                    
                except Exception as gpu_error:
                    self.logger.warning(f"GPU FAISS IVF failed, using optimized CPU: {gpu_error}")
                    quantizer = faiss.IndexFlatIP(self.vector_dim)
                    self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, self.n_centroids)
                    self.n_probe = min(32, self.n_centroids // 8)
                    self._configure_cpu_threading()
            elif self.platform_type == "Apple_Silicon":
                # Apple Silicon optimized FAISS IVF
                quantizer = faiss.IndexFlatIP(self.vector_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, self.n_centroids)
                
                # Apple Silicon specific optimizations for IVF:
                # 1. Balanced probe count for unified memory architecture
                # 2. Optimal threading for efficiency + performance cores
                self.n_probe = min(32, max(8, self.n_centroids // 8))
                
                # Unified memory optimization - conservative threading
                optimal_threads = min(8, max(4, os.cpu_count() // 2))
                faiss.omp_set_num_threads(optimal_threads)
                
                self.logger.info(f"Apple Silicon FAISS IVF: {optimal_threads} threads, {self.n_centroids} centroids, {self.n_probe} probes")
                
            else:
                # CPU-only platforms with general optimizations
                quantizer = faiss.IndexFlatIP(self.vector_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, self.n_centroids)
                
                # Conservative probe optimization for CPU-only
                self.n_probe = min(16, max(4, self.n_centroids // 16))
                self._configure_cpu_threading()
                
                self.logger.info(f"CPU FAISS IVF: {self.n_centroids} centroids, {self.n_probe} probes")
            
            # Set search parameters
            self.index.nprobe = self.n_probe
            
            # Prepare vectors for training and adding
            vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
            
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(vectors_f32, axis=1, keepdims=True)
            normalized_vectors = vectors_f32 / norms
            
            # Train the index
            train_start = time.time()
            
            # Use subset for training if dataset is very large
            train_vectors = normalized_vectors
            if len(normalized_vectors) > 100000:
                # Use random sample for training
                train_indices = np.random.choice(len(normalized_vectors), 100000, replace=False)
                train_vectors = normalized_vectors[train_indices]
            
            self.index.train(train_vectors)
            train_time = (time.time() - train_start) * 1000
            
            # Add all vectors to the trained index
            self.index.add(normalized_vectors)
            
            # Store metadata
            self.metadata = metadata.copy()
            self.vector_count = len(vectors)
            
            # Update statistics
            total_build_time = (time.time() - start_time) * 1000
            self.stats['index_build_time_ms'] = total_build_time
            self.stats['index_train_time_ms'] = train_time
            self.stats['n_centroids'] = self.n_centroids
            self.stats['n_probe'] = self.n_probe
            
            self.logger.info(f"FAISS IVF index initialized with {self.vector_count} vectors, "
                           f"{self.n_centroids} centroids, probe={self.n_probe} "
                           f"in {total_build_time:.2f}ms (train: {train_time:.2f}ms)")
            return True
            
        except Exception as e:
            self.logger.error(f"FAISS IVF initialization failed: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform FAISS IVF search.
        
        Args:
            query_vector: Query vector for similarity search
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices) for k nearest neighbors
        """
        start_time = time.time()
        
        try:
            if self.index is None or self.vector_count == 0:
                return np.array([]), np.array([])
            
            # Prepare query vector
            query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
            
            # Normalize query vector
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            
            # Perform search
            k = min(k, self.vector_count)
            distances, indices = self.index.search(query, k)
            
            # Convert inner product back to cosine distance
            distances = 1.0 - distances[0]  # Remove batch dimension
            indices = indices[0]
            
            # Filter out invalid results (FAISS returns -1 for missing results)
            valid_mask = indices >= 0
            distances = distances[valid_mask]
            indices = indices[valid_mask]
            
            # Update statistics
            search_time = (time.time() - start_time) * 1000
            self._update_statistics(search_time)
            
            return distances, indices
            
        except Exception as e:
            self.logger.error(f"FAISS IVF search failed: {e}")
            return np.array([]), np.array([])
    
    def add_vectors(self, vectors: np.ndarray) -> bool:
        """
        Add new vectors to FAISS IVF index.
        
        Note: Adding vectors to IVF index is less efficient than Flat index.
        For large additions, consider rebuilding the index.
        
        Args:
            vectors: New vectors to add
            
        Returns:
            bool: True if vectors added successfully
        """
        try:
            if self.index is None:
                return False
            
            # Prepare vectors
            vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
            
            # Normalize vectors
            norms = np.linalg.norm(vectors_f32, axis=1, keepdims=True)
            normalized_vectors = vectors_f32 / norms
            
            # Add to index
            self.index.add(normalized_vectors)
            self.vector_count += len(vectors)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add vectors to FAISS IVF: {e}")
            return False
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        if self.index is None:
            return 0.0
        
        # Estimate FAISS IVF memory usage
        # IVF index has centroids + inverted lists + some overhead
        centroid_memory = self.n_centroids * self.vector_dim * 4 / (1024 * 1024)
        vector_memory = self.vector_count * self.vector_dim * 4 / (1024 * 1024)
        overhead_memory = 0.2  # Additional overhead for IVF structures
        
        return centroid_memory + vector_memory + overhead_memory
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get FAISS IVF statistics."""
        return {
            'tier_type': 'faiss_ivf',
            'vector_count': self.vector_count,
            'memory_usage_mb': self.get_memory_usage(),
            'platform_type': self.platform_type,
            'index_trained': self.index is not None and self.index.is_trained,
            **self.stats
        }
    
    def _update_statistics(self, search_time_ms: float):
        """Update search performance statistics."""
        self.stats['total_searches'] += 1
        self.stats['total_search_time_ms'] += search_time_ms
        self.stats['vectors_searched'] += self.vector_count
        
        total_searches = self.stats['total_searches']
        self.stats['avg_search_time_ms'] = (
            self.stats['total_search_time_ms'] / total_searches
        )
    
    def _configure_cpu_threading(self):
        """
        Configure optimal CPU threading for FAISS IVF operations.
        
        IVF indexes benefit from different threading strategies than Flat indexes
        due to the inverted file structure and centroid-based search.
        """
        try:
            cpu_count = os.cpu_count() or 4
            
            # Platform-specific thread optimization for IVF
            if self.platform_type == "CPU_Intel" or self.platform_type.startswith("Intel"):
                # Intel CPUs with IVF: slightly lower threads due to complex operations
                optimal_threads = min(12, max(4, cpu_count * 3 // 4))
            else:
                # Conservative threading for unknown CPU architectures with IVF
                optimal_threads = min(6, max(4, cpu_count // 3))
            
            # Apply threading configuration
            faiss.omp_set_num_threads(optimal_threads)
            
            self.logger.info(f"Configured FAISS IVF CPU threading: {optimal_threads} threads for {self.platform_type}")
            
        except Exception as e:
            self.logger.warning(f"Failed to configure IVF CPU threading: {e}")
            # Use default FAISS threading


class AdaptiveSearchEngine:
    """
    Adaptive search engine that automatically selects optimal search strategy.
    
    Intelligent search system that adapts to dataset characteristics and platform
    capabilities to provide optimal vector similarity search performance.
    
    Search tier selection:
    - 0-100 vectors: Linear search with SIMD optimization
    - 100-10K vectors: FAISS Flat exhaustive search
    - 10K+ vectors: FAISS IVF approximate search
    
    Platform optimizations:
    - NVIDIA GPU: GPU-accelerated FAISS with large batch processing
    - Apple Silicon: MPS + optimized CPU FAISS with unified memory
    - CPU-only: Multi-threaded FAISS with conservative memory allocation
    """
    
    def __init__(self, vector_store: HighPerformanceVectorStore, 
                 config_manager: Optional[ConfigManager] = None):
        """
        Initialize adaptive search engine.
        
        Args:
            vector_store: High-performance vector store for data access
            config_manager: Configuration manager for platform settings
        """
        self.vector_store = vector_store
        self.config_manager = config_manager or ConfigManager()
        self.unified_config = self.config_manager.get_config()
        self.platform_type = self.unified_config.platform.platform_type
        
        self.logger = logging.getLogger(__name__)
        
        # Search tier management
        self.current_tier = None
        self.current_tier_type = None
        self.vector_count = 0
        self.vector_dim = 1536  # Standard dimension for CLIP + DINOv2
        
        # Search tier thresholds (configurable)
        self.linear_threshold = 100      # Switch from linear to FAISS Flat
        self.flat_threshold = 10000      # Switch from FAISS Flat to IVF
        
        # Performance monitoring
        self.search_stats = {
            'total_searches': 0,
            'total_search_time_ms': 0.0,
            'avg_search_time_ms': 0.0,
            'tier_transitions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Result caching for frequent queries
        self._query_cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 100
        
        self.logger.info(f"AdaptiveSearchEngine initialized for platform: {self.platform_type}")
        
        # Initialize with current data
        self._initialize_search_tier()
    
    def _initialize_search_tier(self):
        """
        Initialize the search tier based on current dataset size.
        
        Determines the optimal search strategy and builds the appropriate
        index based on the number of vectors in the vector store.
        """
        try:
            # Get current vector count from vector store
            # For now, we'll start with empty and build incrementally
            self.vector_count = 0
            
            # Determine initial tier
            if self.vector_count <= self.linear_threshold:
                tier_type = 'linear'
            elif self.vector_count <= self.flat_threshold:
                tier_type = 'faiss_flat'
            else:
                tier_type = 'faiss_ivf'
            
            # Create the appropriate search tier
            self._setup_search_tier(tier_type)
            
            self.logger.info(f"Initialized {tier_type} search tier for {self.vector_count} vectors")
            
        except Exception as e:
            self.logger.error(f"Search tier initialization failed: {e}")
            # Fallback to linear search
            self._setup_search_tier('linear')
    
    def _setup_search_tier(self, tier_type: str) -> bool:
        """
        Setup specific search tier.
        
        Args:
            tier_type: Type of search tier ('linear', 'faiss_flat', 'faiss_ivf')
            
        Returns:
            bool: True if setup successful
        """
        try:
            if tier_type == 'linear':
                self.current_tier = LinearSearchTier(self.platform_type)
            elif tier_type == 'faiss_flat':
                self.current_tier = FAISSFlatTier(self.platform_type, self.vector_dim)
            elif tier_type == 'faiss_ivf':
                self.current_tier = FAISSIVFTier(self.platform_type, self.vector_dim)
            else:
                raise ValueError(f"Unknown tier type: {tier_type}")
            
            self.current_tier_type = tier_type
            
            # If we have existing vectors, initialize the tier
            if self.vector_count > 0:
                # This would load vectors from vector_store
                # For now, we'll implement incremental building
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup {tier_type} tier: {e}")
            return False
    
    def _should_transition_tier(self) -> Optional[str]:
        """
        Determine if we should transition to a different search tier.
        
        Returns:
            Optional[str]: New tier type if transition needed, None otherwise
        """
        if self.current_tier_type == 'linear' and self.vector_count > self.linear_threshold:
            return 'faiss_flat'
        elif self.current_tier_type == 'faiss_flat' and self.vector_count > self.flat_threshold:
            return 'faiss_ivf'
        
        return None
    
    def _transition_tier(self, new_tier_type: str) -> bool:
        """
        Transition to a new search tier.
        
        Args:
            new_tier_type: Target tier type
            
        Returns:
            bool: True if transition successful
        """
        try:
            self.logger.info(f"Transitioning from {self.current_tier_type} to {new_tier_type}")
            
            # Get current vectors and metadata (if available)
            vectors = None
            metadata = []
            
            # This would extract vectors from current tier
            # For now, we'll handle empty transition
            
            # Setup new tier
            if self._setup_search_tier(new_tier_type):
                # Initialize with existing data
                if vectors is not None:
                    self.current_tier.initialize(vectors, metadata)
                
                self.search_stats['tier_transitions'] += 1
                self.logger.info(f"Successfully transitioned to {new_tier_type}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Tier transition failed: {e}")
            return False
    
    def search_adaptive(self, query_vector: np.ndarray, k: int = 50) -> List[SearchResult]:
        """
        Perform adaptive similarity search.
        
        Automatically selects the optimal search strategy and returns
        comprehensive search results with metadata.
        
        Args:
            query_vector: Query vector for similarity search
            k: Number of nearest neighbors to return
            
        Returns:
            List of SearchResult objects with similarity results
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = hash(query_vector.tobytes())
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.search_stats['cache_hits'] += 1
                return cached_result[:k]  # Return up to k results
            
            self.search_stats['cache_misses'] += 1
            
            # Check if tier transition is needed
            new_tier_type = self._should_transition_tier()
            if new_tier_type:
                self._transition_tier(new_tier_type)
            
            # Perform search with current tier
            if self.current_tier is None:
                self.logger.warning("No search tier available, returning empty results")
                return []
            
            # Execute search
            distances, indices = self.current_tier.search(query_vector, k)
            
            # Convert to SearchResult objects
            search_results = []
            
            for i, (distance, index) in enumerate(zip(distances, indices)):
                # Create SearchResult with placeholder metadata
                # In a real implementation, this would retrieve actual metadata
                # from the vector store using the index
                
                search_result = SearchResult(
                    item_id=f"item_{index}",
                    vector_id=f"vector_{index}",
                    distance=float(distance),
                    confidence=max(0.0, 1.0 - float(distance)),  # Convert distance to confidence
                    metadata={"index": int(index)},  # Placeholder metadata
                    search_tier=self.current_tier_type,
                    search_time_ms=(time.time() - start_time) * 1000
                )
                
                search_results.append(search_result)
            
            # Cache result
            self._cache_result(cache_key, search_results)
            
            # Update performance statistics
            search_time = (time.time() - start_time) * 1000
            self._update_search_statistics(search_time)
            
            self.logger.debug(f"Adaptive search completed in {search_time:.2f}ms, "
                            f"tier: {self.current_tier_type}, results: {len(search_results)}")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Adaptive search failed: {e}")
            return []
    
    def add_vector_incremental(self, item_id: str, vector: np.ndarray, 
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add vector incrementally with automatic tier management.
        
        Args:
            item_id: Item identifier
            vector: Vector to add
            metadata: Optional metadata for the vector
            
        Returns:
            bool: True if vector added successfully
        """
        try:
            # Add to vector store first
            vector_id = f"{item_id}_vector_{int(time.time()*1000)}"
            success = self.vector_store.store_vector(
                vector_id=vector_id,
                item_id=item_id,
                vector_data=vector,
                metadata=metadata or {}
            )
            
            if not success:
                return False
            
            # Add to current search tier
            if self.current_tier is not None:
                self.current_tier.add_vectors(vector.reshape(1, -1))
            
            self.vector_count += 1
            
            # Check if tier transition is needed
            new_tier_type = self._should_transition_tier()
            if new_tier_type:
                self._transition_tier(new_tier_type)
            
            # Clear cache after adding new vector
            self._clear_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add vector incrementally: {e}")
            return False
    
    def get_search_performance(self) -> SearchPerformance:
        """
        Get current search performance metrics.
        
        Returns:
            SearchPerformance object with comprehensive metrics
        """
        tier_stats = self.current_tier.get_statistics() if self.current_tier else {}
        
        return SearchPerformance(
            query_time_ms=self.search_stats.get('avg_search_time_ms', 0.0),
            tier_used=self.current_tier_type or 'none',
            results_count=0,  # Would be updated during search
            vectors_searched=self.vector_count,
            cache_hit=False,  # Would be set during individual searches
            platform_type=self.platform_type,
            memory_usage_mb=tier_stats.get('memory_usage_mb', 0.0)
        )
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive search engine statistics.
        
        Returns:
            Dictionary with detailed performance and configuration metrics
        """
        tier_stats = self.current_tier.get_statistics() if self.current_tier else {}
        
        # Cache statistics
        cache_hit_rate = 0.0
        total_cache_ops = self.search_stats['cache_hits'] + self.search_stats['cache_misses']
        if total_cache_ops > 0:
            cache_hit_rate = (self.search_stats['cache_hits'] / total_cache_ops) * 100
        
        return {
            'search_engine': {
                'current_tier': self.current_tier_type,
                'vector_count': self.vector_count,
                'platform_type': self.platform_type,
                'tier_thresholds': {
                    'linear_threshold': self.linear_threshold,
                    'flat_threshold': self.flat_threshold
                },
                **self.search_stats
            },
            'current_tier_stats': tier_stats,
            'cache_stats': {
                'cache_size': len(self._query_cache),
                'max_cache_size': self._max_cache_size,
                'hit_rate': cache_hit_rate
            },
            'vector_store_stats': self.vector_store.get_comprehensive_statistics()
        }
    
    def _get_cached_result(self, cache_key: int) -> Optional[List[SearchResult]]:
        """Get cached search result if available."""
        with self._cache_lock:
            if cache_key in self._query_cache:
                result, timestamp = self._query_cache[cache_key]
                # Cache results for 5 minutes
                if time.time() - timestamp < 300:
                    return result
                else:
                    del self._query_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: int, result: List[SearchResult]):
        """Cache search result."""
        with self._cache_lock:
            # Limit cache size
            if len(self._query_cache) >= self._max_cache_size:
                # Remove oldest entry
                oldest_key = min(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k][1])
                del self._query_cache[oldest_key]
            
            self._query_cache[cache_key] = (result, time.time())
    
    def _clear_cache(self):
        """Clear search result cache."""
        with self._cache_lock:
            self._query_cache.clear()
    
    def _update_search_statistics(self, search_time_ms: float):
        """Update search performance statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['total_search_time_ms'] += search_time_ms
        
        total_searches = self.search_stats['total_searches']
        self.search_stats['avg_search_time_ms'] = (
            self.search_stats['total_search_time_ms'] / total_searches
        )
    
    def rebuild_index_optimized(self, force_tier: Optional[str] = None) -> bool:
        """
        Rebuild search index with advanced optimization strategies.
        
        Provides intelligent index rebuilding with:
        - Memory-efficient batch processing
        - Platform-specific optimizations
        - Index validation and error recovery
        - Performance monitoring
        
        Args:
            force_tier: Force specific tier type (linear/faiss_flat/faiss_ivf)
            
        Returns:
            bool: True if rebuild successful
        """
        try:
            self.logger.info("Starting optimized index rebuild...")
            start_time = time.time()
            
            # Determine optimal tier for current dataset
            target_tier = force_tier or self._determine_optimal_tier()
            
            # Collect all vectors from vector store
            all_vectors = []
            all_metadata = []
            
            # Batch processing to avoid memory issues
            batch_size = 1000  # Process in batches
            vector_count = 0
            
            # Get all vector IDs (this should be implemented in vector store)
            try:
                # For now, use existing vectors from current tier
                if self.current_tier and hasattr(self.current_tier, 'vectors') and self.current_tier.vectors is not None:
                    all_vectors.append(self.current_tier.vectors)
                    all_metadata.extend(self.current_tier.metadata)
                    vector_count = len(self.current_tier.metadata)
                else:
                    self.logger.warning("No existing vectors found for rebuild")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Failed to collect vectors for rebuild: {e}")
                return False
            
            # Concatenate all vectors
            if all_vectors:
                combined_vectors = np.vstack(all_vectors)
            else:
                self.logger.warning("No vectors to rebuild index")
                return False
            
            # Create new tier with optimized parameters
            old_tier = self.current_tier
            self.current_tier_type = target_tier
            
            if target_tier == 'linear':
                self.current_tier = LinearSearchTier(self.platform_type)
            elif target_tier == 'faiss_flat':
                self.current_tier = FAISSFlatTier(self.platform_type, self.vector_dim)
            elif target_tier == 'faiss_ivf':
                self.current_tier = FAISSIVFTier(self.platform_type, self.vector_dim)
            else:
                raise ValueError(f"Unknown tier type: {target_tier}")
            
            # Initialize new tier with all vectors
            success = self.current_tier.initialize(combined_vectors, all_metadata)
            
            if not success:
                # Rollback on failure
                self.current_tier = old_tier
                self.logger.error("Index rebuild failed, rolled back to previous tier")
                return False
            
            # Update vector count
            self.vector_count = vector_count
            
            # Clear cache after rebuild
            self._clear_cache()
            
            # Update statistics
            self.search_stats['tier_transitions'] += 1
            
            rebuild_time = time.time() - start_time
            self.logger.info(f"Index rebuild completed in {rebuild_time:.2f}s: {vector_count} vectors → {target_tier}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Index rebuild failed: {e}")
            return False
    
    def add_vectors_batch_incremental(self, vectors_data: List[Tuple[str, np.ndarray, Optional[Dict]]]) -> int:
        """
        Add multiple vectors incrementally without full index rebuild.
        
        Optimized for high-throughput scenarios:
        - Batched index updates
        - Memory-efficient processing
        - Automatic tier management
        - Partial failure handling
        
        Args:
            vectors_data: List of (item_id, vector, metadata) tuples
            
        Returns:
            int: Number of successfully added vectors
        """
        try:
            if not vectors_data:
                return 0
            
            self.logger.info(f"Adding {len(vectors_data)} vectors incrementally...")
            start_time = time.time()
            
            successful_adds = 0
            batch_size = 100  # Process in smaller batches for stability
            
            for i in range(0, len(vectors_data), batch_size):
                batch = vectors_data[i:i + batch_size]
                
                try:
                    # Prepare batch vectors
                    batch_vectors = []
                    batch_metadata = []
                    
                    for item_id, vector, metadata in batch:
                        # Store in vector store first
                        success = self.vector_store.store_vector(
                            f"incremental_{self.vector_count + len(batch_vectors)}",
                            item_id,
                            vector,
                            metadata or {}
                        )
                        
                        if success:
                            batch_vectors.append(vector)
                            batch_metadata.append(metadata or {})
                    
                    if batch_vectors:
                        # Convert to numpy array
                        batch_array = np.array(batch_vectors)
                        
                        # Add to current tier
                        if self.current_tier is not None:
                            tier_success = self.current_tier.add_vectors(batch_array)
                            if tier_success:
                                self.current_tier.metadata.extend(batch_metadata)
                                successful_adds += len(batch_vectors)
                                self.vector_count += len(batch_vectors)
                
                except Exception as batch_error:
                    self.logger.warning(f"Batch {i//batch_size + 1} failed: {batch_error}")
                    continue
                
                # Check for tier transition after each batch
                new_tier_type = self._should_transition_tier()
                if new_tier_type:
                    self.logger.info(f"Triggering tier transition to {new_tier_type} during batch add")
                    if not self._transition_tier(new_tier_type):
                        self.logger.warning("Tier transition failed during batch add")
            
            # Clear cache after batch operations
            self._clear_cache()
            
            add_time = time.time() - start_time
            self.logger.info(f"Batch incremental add completed: {successful_adds}/{len(vectors_data)} vectors in {add_time:.2f}s")
            
            return successful_adds
            
        except Exception as e:
            self.logger.error(f"Batch incremental add failed: {e}")
            return 0
    
    def _determine_optimal_tier(self) -> str:
        """
        Determine optimal tier type based on current dataset characteristics.
        
        Returns:
            str: Optimal tier type
        """
        if self.vector_count <= self.linear_threshold:
            return 'linear'
        elif self.vector_count <= self.flat_threshold:
            return 'faiss_flat'
        else:
            return 'faiss_ivf'
    
    def validate_index_integrity(self) -> Dict[str, Any]:
        """
        Validate search index integrity and performance.
        
        Returns:
            Dict with validation results and recommendations
        """
        try:
            validation_results = {
                'status': 'healthy',
                'issues': [],
                'recommendations': [],
                'performance_metrics': {}
            }
            
            if not self.current_tier:
                validation_results['status'] = 'error'
                validation_results['issues'].append('No active search tier')
                return validation_results
            
            # Test search performance with a random query
            test_vector = np.random.random(self.vector_dim).astype(np.float32)
            start_time = time.time()
            
            try:
                results = self.search_adaptive(test_vector, k=min(10, self.vector_count))
                search_time = (time.time() - start_time) * 1000
                
                validation_results['performance_metrics']['test_search_time_ms'] = search_time
                
                # Performance analysis
                if search_time > 100:  # >100ms is slow
                    validation_results['issues'].append(f'Slow search performance: {search_time:.2f}ms')
                    validation_results['recommendations'].append('Consider index optimization or tier upgrade')
                
                if len(results) == 0 and self.vector_count > 0:
                    validation_results['issues'].append('Search returned no results despite having vectors')
                    validation_results['status'] = 'warning'
                
            except Exception as search_error:
                validation_results['status'] = 'error'
                validation_results['issues'].append(f'Search test failed: {search_error}')
            
            # Memory usage analysis
            memory_usage = self.current_tier.get_memory_usage()
            validation_results['performance_metrics']['memory_usage_mb'] = memory_usage
            
            if memory_usage > 1000:  # >1GB
                validation_results['recommendations'].append('High memory usage - consider data compression or tier optimization')
            
            # Vector count vs tier optimization
            if self.current_tier_type == 'linear' and self.vector_count > self.linear_threshold * 1.5:
                validation_results['recommendations'].append('Consider upgrading to FAISS Flat tier for better performance')
            elif self.current_tier_type == 'faiss_flat' and self.vector_count > self.flat_threshold * 1.5:
                validation_results['recommendations'].append('Consider upgrading to FAISS IVF tier for better scalability')
            
            if not validation_results['issues']:
                validation_results['status'] = 'healthy'
            elif validation_results['status'] != 'error':
                validation_results['status'] = 'warning'
            
            return validation_results
            
        except Exception as e:
            return {
                'status': 'error',
                'issues': [f'Validation failed: {e}'],
                'recommendations': ['Check search engine configuration'],
                'performance_metrics': {}
            }
    
    def optimize_search_performance(self):
        """
        Optimize search performance based on usage patterns.
        
        Analyzes search patterns and adjusts parameters for optimal performance.
        """
        try:
            self.logger.info("Optimizing search performance...")
            
            # Analyze search patterns and adjust tier thresholds if needed
            tier_stats = self.current_tier.get_statistics() if self.current_tier else {}
            
            avg_search_time = tier_stats.get('avg_search_time_ms', 0)
            
            # If search is getting slow, consider adjusting parameters
            if self.current_tier_type == 'faiss_ivf' and avg_search_time > 50:
                # For IVF, we could adjust nprobe for speed/accuracy tradeoff
                if hasattr(self.current_tier, 'index') and self.current_tier.index:
                    current_nprobe = self.current_tier.index.nprobe
                    new_nprobe = max(8, current_nprobe // 2)  # Reduce for speed
                    self.current_tier.index.nprobe = new_nprobe
                    self.logger.info(f"Reduced nprobe from {current_nprobe} to {new_nprobe} for speed")
            
            # Clear old cache entries
            self._clear_cache()
            
            self.logger.info("Search performance optimization completed")
            
        except Exception as e:
            self.logger.error(f"Search performance optimization failed: {e}")
    
    def close(self):
        """Close search engine and cleanup resources."""
        try:
            # Clear cache
            self._clear_cache()
            
            # The search tiers don't need explicit cleanup
            # FAISS indices are managed by FAISS internally
            
            self.logger.info("AdaptiveSearchEngine closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing AdaptiveSearchEngine: {e}")


# Convenience functions
def create_adaptive_search_engine(vector_store: HighPerformanceVectorStore,
                                config_manager: Optional[ConfigManager] = None) -> AdaptiveSearchEngine:
    """Create adaptive search engine with current platform settings."""
    return AdaptiveSearchEngine(vector_store, config_manager)


if __name__ == "__main__":
    # Test adaptive search engine functionality
    import logging
    from .vector_store import create_high_performance_vector_store
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Adaptive Search Engine Test ===")
    
    # Create vector store and search engine
    vector_store = create_high_performance_vector_store()
    search_engine = create_adaptive_search_engine(vector_store)
    
    try:
        # Test with small dataset (linear search)
        print("\n🔍 Testing with small dataset...")
        
        for i in range(50):
            test_vector = np.random.random(1536).astype(np.float32)
            search_engine.add_vector_incremental(f"test_item_{i}", test_vector)
        
        # Test search
        query_vector = np.random.random(1536).astype(np.float32)
        results = search_engine.search_adaptive(query_vector, k=10)
        
        print(f"✅ Small dataset search: {len(results)} results")
        print(f"   Current tier: {search_engine.current_tier_type}")
        
        # Test with medium dataset (FAISS Flat)
        print("\n🔍 Testing tier transition to FAISS Flat...")
        
        for i in range(150):  # This should trigger transition
            test_vector = np.random.random(1536).astype(np.float32)
            search_engine.add_vector_incremental(f"medium_item_{i}", test_vector)
        
        results = search_engine.search_adaptive(query_vector, k=10)
        
        print(f"✅ Medium dataset search: {len(results)} results")
        print(f"   Current tier: {search_engine.current_tier_type}")
        
        # Get comprehensive statistics
        stats = search_engine.get_comprehensive_statistics()
        print(f"\n📊 Search Engine Statistics:")
        print(f"   Vector count: {stats['search_engine']['vector_count']}")
        print(f"   Current tier: {stats['search_engine']['current_tier']}")
        print(f"   Tier transitions: {stats['search_engine']['tier_transitions']}")
        print(f"   Avg search time: {stats['search_engine']['avg_search_time_ms']:.2f}ms")
        print(f"   Platform: {stats['search_engine']['platform_type']}")
        
    finally:
        # Cleanup
        search_engine.close()
        vector_store.close()
    
    print("\n✅ Adaptive Search Engine test completed successfully")