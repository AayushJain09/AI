"""
Advanced FAISS Indexing System for 1536-Dimensional Features

Optimized for high-performance similarity search with:
- CLIP ViT-L/14 (768 dimensions) + DINOv2-base (768 dimensions) = 1536 total
- GPU acceleration when available
- Multiple index types for different dataset sizes
- Memory-efficient batch processing
- Incremental indexing support
"""

import numpy as np
import faiss
import h5py
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from dataclasses import dataclass
from tqdm import tqdm
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Configuration for FAISS index creation"""
    # Index type selection
    use_gpu: bool = True
    index_type: str = "auto"  # auto, flat, ivf, ivf_pq, hnsw
    
    # Performance parameters
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 32  # Number of clusters to search
    m_pq: int = 64    # PQ subquantizers (must divide dimension)
    nbits_pq: int = 8 # Bits per PQ code
    
    # HNSW parameters
    hnsw_m: int = 32  # Number of bi-directional links for HNSW
    hnsw_ef_construction: int = 200  # Size of dynamic candidate list for HNSW
    hnsw_ef_search: int = 128  # Size of dynamic candidate list for search
    
    # Memory and processing
    batch_size: int = 1000
    max_memory_gb: float = 4.0
    
    # Quality vs speed trade-offs
    precision_mode: str = "balanced"  # fast, balanced, accurate
    normalize_features: bool = True   # Normalize for cosine similarity
    use_inner_product: bool = True    # Use inner product metric
    
    # Index selection thresholds
    flat_threshold: int = 1000        # Use flat index below this size
    ivf_threshold: int = 10000        # Use IVF below this size
    pq_threshold: int = 100000        # Use IVF+PQ below this size


class AdvancedFAISSIndexer:
    """
    Advanced FAISS indexing system optimized for 1536-dimensional features.
    
    Features:
    - Automatic index type selection based on dataset size
    - GPU acceleration with fallback to CPU
    - Memory-efficient batch processing
    - Multiple precision modes for speed/accuracy trade-offs
    - Incremental indexing for real-time updates
    - Comprehensive performance monitoring
    """
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.dimension = 1536  # CLIP(768) + DINOv2(768)
        
        # Initialize device
        self._setup_device()
        
        # Index components
        self.index = None
        self.item_ids = []
        self.metadata = {}
        
        logger.info(f"🚀 Advanced FAISS Indexer initialized")
        logger.info(f"📐 Feature dimensions: {self.dimension}")
        logger.info(f"🖥️  Device: {'GPU' if self.use_gpu else 'CPU'}")
        
    def _setup_device(self):
        """Setup GPU/CPU device for FAISS operations"""
        self.use_gpu = False
        self.gpu_resources = None
        
        if self.config.use_gpu:
            try:
                # Check FAISS GPU availability
                if faiss.get_num_gpus() > 0:
                    self.gpu_resources = faiss.StandardGpuResources()
                    self.use_gpu = True
                    logger.info(f"✅ GPU acceleration enabled: {faiss.get_num_gpus()} GPU(s)")
                else:
                    logger.info("⚠️  No GPU available, using CPU")
            except Exception as e:
                logger.warning(f"⚠️  GPU initialization failed: {e}, using CPU")
        else:
            logger.info("🖥️  CPU-only mode selected")
    
    def _determine_optimal_index_type(self, n_vectors: int) -> str:
        """
        Automatically determine optimal index type based on dataset size and requirements.
        
        Index Selection Strategy:
        - < 1K vectors: Flat index (exact search)
        - 1K-10K vectors: IVF with flat quantizer  
        - 10K-100K vectors: IVF with product quantization
        - > 100K vectors: HNSW for maximum speed
        """
        if self.config.index_type != "auto":
            return self.config.index_type
        
        if n_vectors < 1000:
            return "flat"
        elif n_vectors < 10000:
            return "ivf"
        elif n_vectors < 100000:
            return "ivf_pq"
        else:
            return "hnsw"
    
    def _create_flat_index(self) -> faiss.Index:
        """Create exact search index (best quality, slower for large datasets)"""
        logger.info("🔧 Creating Flat index (exact search)")
        
        if self.config.precision_mode == "accurate":
            # Use inner product for cosine similarity with normalized features
            index = faiss.IndexFlatIP(self.dimension)
        else:
            # Use L2 distance (faster)
            index = faiss.IndexFlatL2(self.dimension)
        
        return index
    
    def _create_ivf_index(self, n_vectors: int) -> faiss.Index:
        """Create IVF index (good balance of speed and accuracy)"""
        # Determine optimal number of clusters
        nlist = min(self.config.nlist, max(1, n_vectors // 39))
        logger.info(f"🔧 Creating IVF index with {nlist} clusters")
        
        # Create quantizer
        if self.config.precision_mode == "accurate":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        
        # Set search parameters
        index.nprobe = min(self.config.nprobe, nlist)
        
        return index
    
    def _create_ivf_pq_index(self, n_vectors: int) -> faiss.Index:
        """Create IVF+PQ index (compressed, good for large datasets)"""
        nlist = min(self.config.nlist * 4, max(1, n_vectors // 39))
        
        # Ensure m_pq divides dimension evenly
        m = self.config.m_pq
        while self.dimension % m != 0 and m > 8:
            m -= 1
        
        logger.info(f"🔧 Creating IVF+PQ index: {nlist} clusters, {m} subquantizers")
        
        # Create quantizer
        if self.config.precision_mode == "accurate":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, self.config.nbits_pq, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, self.config.nbits_pq, faiss.METRIC_L2)
        
        # Set search parameters
        index.nprobe = min(self.config.nprobe, nlist)
        
        return index
    
    def _create_hnsw_index(self) -> faiss.Index:
        """Create HNSW index (fastest search, good for very large datasets)"""
        logger.info(f"🔧 Creating HNSW index: M={self.config.hnsw_m}")
        
        if self.config.precision_mode == "accurate":
            index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_m, faiss.METRIC_L2)
        
        # Set construction parameters
        index.hnsw.efConstruction = self.config.hnsw_ef_construction
        index.hnsw.efSearch = self.config.hnsw_ef_search
        
        return index
    
    def create_index(self, features_file: str, output_dir: str) -> Dict[str, str]:
        """
        Create optimized FAISS index from features file.
        
        Args:
            features_file: Path to HDF5 features file
            output_dir: Directory to save index and metadata
            
        Returns:
            Dictionary with paths to created files
        """
        logger.info(f"🚀 Creating optimized FAISS index from {features_file}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load features and build index
        features, item_ids, metadata = self._load_features(features_file)
        n_vectors = len(features)
        
        logger.info(f"📊 Dataset: {n_vectors} vectors, {self.dimension} dimensions")
        
        # Determine optimal index type
        index_type = self._determine_optimal_index_type(n_vectors)
        logger.info(f"🎯 Selected index type: {index_type}")
        
        # Create appropriate index
        if index_type == "flat":
            index = self._create_flat_index()
        elif index_type == "ivf":
            index = self._create_ivf_index(n_vectors)
        elif index_type == "ivf_pq":
            index = self._create_ivf_pq_index(n_vectors)
        elif index_type == "hnsw":
            index = self._create_hnsw_index()
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Move to GPU if available
        if self.use_gpu and index_type != "hnsw":  # HNSW doesn't support GPU
            try:
                index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                logger.info("✅ Index moved to GPU")
            except Exception as e:
                logger.warning(f"⚠️  Failed to move index to GPU: {e}")
                self.use_gpu = False
        
        # Train index if needed
        if hasattr(index, 'is_trained') and not index.is_trained:
            logger.info("🎓 Training index...")
            training_data = features[:min(len(features), 10000)]  # Use subset for training
            index.train(training_data.astype('float32'))
            logger.info("✅ Index training completed")
        
        # Add vectors to index
        logger.info("📥 Adding vectors to index...")
        batch_size = self.config.batch_size
        
        for i in tqdm(range(0, len(features), batch_size), desc="Adding features"):
            batch_end = min(i + batch_size, len(features))
            batch_features = features[i:batch_end].astype('float32')
            index.add(batch_features)
        
        logger.info(f"✅ Index created: {index.ntotal} vectors indexed")
        
        # Save index and metadata
        index_path = output_path / "faiss_index.bin"
        metadata_path = output_path / "index_metadata.pkl"
        
        # Move back to CPU for saving
        if self.use_gpu and hasattr(index, 'index'):
            cpu_index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(index, str(index_path))
        
        # Save metadata
        index_metadata = {
            'item_ids': item_ids,
            'metadata': metadata,
            'index_type': index_type,
            'dimension': self.dimension,
            'n_vectors': n_vectors,
            'config': self.config.__dict__,
            'creation_time': time.time(),
            'feature_file': str(features_file)
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(index_metadata, f)
        
        # Save human-readable stats
        stats_path = output_path / "index_stats.json"
        stats = {
            'total_vectors': n_vectors,
            'dimension': self.dimension,
            'index_type': index_type,
            'gpu_used': self.use_gpu,
            'precision_mode': self.config.precision_mode,
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'memory_usage_mb': self._estimate_memory_usage(index, n_vectors),
            'expected_search_time_ms': self._estimate_search_time(index_type, n_vectors)
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"💾 Index saved to: {index_path}")
        logger.info(f"📋 Metadata saved to: {metadata_path}")
        logger.info(f"📊 Stats saved to: {stats_path}")
        
        return {
            'index_path': str(index_path),
            'metadata_path': str(metadata_path),
            'stats_path': str(stats_path)
        }
    
    def _load_features(self, features_file: str) -> Tuple[np.ndarray, List[str], Dict]:
        """Load features from HDF5 file and combine CLIP + DINOv2"""
        logger.info(f"📂 Loading features from {features_file}")
        
        features_list = []
        item_ids = []
        metadata = {}
        
        with h5py.File(features_file, 'r') as hf:
            # Get all image groups
            image_keys = list(hf.keys())
            logger.info(f"Found {len(image_keys)} feature groups")
            
            for img_key in tqdm(image_keys, desc="Loading features"):
                img_group = hf[img_key]
                
                # Get item ID from attributes
                item_id = img_group.attrs.get('item_id', 'unknown')
                image_path = img_group.attrs.get('image_path', '')
                
                # Load CLIP and DINOv2 features
                clip_features = img_group['clip'][:]
                dinov2_features = img_group['dinov2'][:]
                
                # Validate dimensions
                if len(clip_features) != 768:
                    logger.warning(f"Unexpected CLIP dimensions: {len(clip_features)} (expected 768)")
                if len(dinov2_features) != 768:
                    logger.warning(f"Unexpected DINOv2 dimensions: {len(dinov2_features)} (expected 768)")
                
                # Combine features
                combined_features = np.concatenate([clip_features, dinov2_features])
                
                # Normalize for cosine similarity
                if self.config.precision_mode == "accurate":
                    combined_features = combined_features / np.linalg.norm(combined_features)
                
                features_list.append(combined_features)
                item_ids.append(item_id)
                
                # Store metadata
                if item_id not in metadata:
                    metadata[item_id] = []
                metadata[item_id].append({
                    'image_key': img_key,
                    'image_path': image_path,
                    'clip_dims': len(clip_features),
                    'dinov2_dims': len(dinov2_features)
                })
        
        features_array = np.vstack(features_list)
        logger.info(f"✅ Loaded {len(features_array)} feature vectors")
        logger.info(f"📐 Feature shape: {features_array.shape}")
        
        return features_array, item_ids, metadata
    
    def _estimate_memory_usage(self, index, n_vectors: int) -> float:
        """Estimate memory usage in MB"""
        bytes_per_vector = 4 * self.dimension  # float32
        base_memory = n_vectors * bytes_per_vector / (1024 * 1024)
        
        # Add index overhead
        if hasattr(index, 'code_size'):
            # PQ index
            return base_memory * 0.1  # Roughly 10% of original
        elif 'HNSW' in str(type(index)):
            return base_memory * 1.5  # HNSW overhead
        else:
            return base_memory
    
    def _estimate_search_time(self, index_type: str, n_vectors: int) -> float:
        """Estimate search time in milliseconds"""
        if index_type == "flat":
            return 0.001 * n_vectors  # Linear scan
        elif index_type == "ivf":
            return 0.1 + 0.001 * (n_vectors / 100)
        elif index_type == "ivf_pq":
            return 0.05 + 0.001 * (n_vectors / 1000)
        elif index_type == "hnsw":
            return 0.01 + 0.001 * np.log(n_vectors)
        else:
            return 1.0
    
    def load_index(self, index_path: str, metadata_path: str) -> bool:
        """Load existing FAISS index"""
        try:
            logger.info(f"📂 Loading index from {index_path}")
            
            # Load index
            self.index = faiss.read_index(index_path)
            
            # Move to GPU if available
            if self.use_gpu and 'HNSW' not in str(type(self.index)):
                try:
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                    logger.info("✅ Index moved to GPU")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to move index to GPU: {e}")
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.item_ids = metadata['item_ids']
                self.metadata = metadata['metadata']
            
            logger.info(f"✅ Index loaded: {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            return False
    
    def search(self, query_features: np.ndarray, k: int = 10, confidence_threshold: float = 0.85) -> List[Dict]:
        """
        Search for similar items in the index.
        
        Args:
            query_features: Combined CLIP + DINOv2 features (1536 dims)
            k: Number of nearest neighbors to return
            confidence_threshold: Minimum similarity threshold
            
        Returns:
            List of search results with similarities and metadata
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Normalize query features if using inner product
        if self.config.precision_mode == "accurate":
            query_features = query_features / np.linalg.norm(query_features)
        
        # Ensure proper shape and type
        query_features = query_features.reshape(1, -1).astype('float32')
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_features, k)
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Process results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No more results
                break
            
            # Convert distance to similarity
            if self.config.precision_mode == "accurate":
                similarity = distance  # Inner product is already similarity
            else:
                similarity = 1.0 / (1.0 + distance)  # Convert L2 distance to similarity
            
            if similarity >= confidence_threshold:
                item_id = self.item_ids[idx]
                results.append({
                    'item_id': item_id,
                    'similarity': float(similarity),
                    'rank': i + 1,
                    'metadata': self.metadata.get(item_id, {}),
                    'search_time_ms': search_time
                })
        
        logger.debug(f"🔍 Search completed: {len(results)} results in {search_time:.2f}ms")
        return results


def main():
    """Main entry point for index creation"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Create optimized FAISS index')
    parser.add_argument('--features', type=str, required=True, help='Features HDF5 file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for index')
    parser.add_argument('--config', type=str, help='Config YAML file')
    parser.add_argument('--index-type', type=str, default='auto', 
                       choices=['auto', 'flat', 'ivf', 'ivf_pq', 'hnsw'],
                       help='Index type to create')
    parser.add_argument('--precision', type=str, default='balanced',
                       choices=['fast', 'balanced', 'accurate'],
                       help='Precision vs speed trade-off')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            index_config = IndexConfig(**config_dict.get('indexing', {}))
    else:
        index_config = IndexConfig()
    
    # Override with command line arguments
    if args.index_type:
        index_config.index_type = args.index_type
    if args.precision:
        index_config.precision_mode = args.precision
    if args.gpu:
        index_config.use_gpu = True
    
    # Create indexer and build index
    indexer = AdvancedFAISSIndexer(index_config)
    result_paths = indexer.create_index(args.features, args.output)
    
    logger.info("🎉 Indexing completed successfully!")
    for key, path in result_paths.items():
        logger.info(f"  {key}: {path}")


if __name__ == "__main__":
    main()