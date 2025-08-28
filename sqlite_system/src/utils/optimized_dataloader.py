"""
GPU-Optimized DataLoader Configuration for Maximum Throughput
Implements industry best practices for CPU-GPU hybrid pipelines
"""

try:
    import cv2
    # CRITICAL: Fix OpenCV threading conflicts with DataLoader workers
    cv2.setNumThreads(0)
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available - some optimizations will be skipped")

import torch
import psutil
import logging
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import multiprocessing as mp

logger = logging.getLogger(__name__)

# OpenCV threading fix moved to import section above

class OptimizedDataLoaderConfig:
    """
    Factory for creating GPU-optimized DataLoader configurations
    Following industry best practices from Lightly AI and Albumentations teams
    """
    
    @staticmethod
    def create_optimized_config(
        batch_size: Optional[int] = None,
        platform_config: Optional[Dict] = None,
        dataset_size: Optional[int] = None,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Create optimized DataLoader configuration for maximum GPU utilization
        
        Key optimizations:
        1. Optimal worker count based on CPU cores and I/O patterns
        2. Pin memory for fast GPU transfers  
        3. Persistent workers to reduce startup overhead
        4. Prefetch factor for pipeline parallelism
        5. Platform-specific optimizations
        
        Args:
            batch_size: Target batch size (auto-detected if None)
            platform_config: Platform-specific settings
            dataset_size: Size of dataset for optimization
            is_training: Whether this is training or inference
            
        Returns:
            Optimized DataLoader configuration dictionary
        """
        
        # Get system information
        cpu_count = mp.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Platform-specific optimizations
        if platform_config is None:
            platform_config = {}
            
        platform_type = platform_config.get('platform_type', 'CPU')
        gpu_memory_gb = platform_config.get('gpu_memory_gb', 0)
        
        # OPTIMIZATION 1: Determine optimal worker count
        # Rule of thumb: 4-8 workers per GPU, but limited by CPU cores
        if platform_type == 'NVIDIA_GPU':
            # CUDA GPUs benefit from more workers for data loading
            optimal_workers = min(8, max(2, cpu_count - 2))  # Leave 2 cores for main process
        elif platform_type == 'APPLE_SILICON':
            # Apple Silicon: Balance efficiency/performance cores
            optimal_workers = min(6, max(2, cpu_count // 2))  # Conservative for unified memory
        else:
            # CPU-only: More conservative
            optimal_workers = min(4, max(1, cpu_count // 2))
        
        # OPTIMIZATION 2: Determine optimal batch size if not provided
        if batch_size is None:
            if platform_type == 'NVIDIA_GPU' and gpu_memory_gb > 0:
                # GPU batch sizing: 2-4 images per GB of GPU memory
                batch_size = min(32, max(4, int(gpu_memory_gb * 3)))
            elif platform_type == 'APPLE_SILICON':
                # Apple Silicon: Unified memory, be conservative
                batch_size = min(8, max(2, int(available_memory_gb * 0.5)))
            else:
                # CPU-only: Memory-based sizing
                batch_size = min(4, max(1, int(available_memory_gb * 0.25)))
        
        # OPTIMIZATION 3: Prefetch factor for pipeline parallelism
        # Higher prefetch for GPU systems to keep GPU busy
        if platform_type in ['NVIDIA_GPU', 'APPLE_SILICON']:
            prefetch_factor = 4  # Prefetch 4 batches ahead
        else:
            prefetch_factor = 2  # Conservative for CPU-only
        
        # OPTIMIZATION 4: Pin memory for fast GPU transfers
        pin_memory = platform_type in ['NVIDIA_GPU', 'APPLE_SILICON']
        
        # OPTIMIZATION 5: Persistent workers to reduce startup overhead
        # Especially beneficial for augmentation-heavy pipelines
        persistent_workers = optimal_workers > 0
        
        config = {
            'batch_size': batch_size,
            'num_workers': optimal_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'prefetch_factor': prefetch_factor,
            'drop_last': is_training,  # Drop incomplete batches during training
            'shuffle': is_training,    # Shuffle for training
            
            # Additional optimizations
            'collate_fn': None,  # Can be customized for specific needs
            'timeout': 30,       # Timeout for worker processes
            'worker_init_fn': _worker_init_fn,  # Initialize workers with OpenCV fix
        }
        
        # Log configuration for monitoring
        logger.info(f"🚀 Optimized DataLoader Configuration:")
        logger.info(f"   Platform: {platform_type}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Workers: {optimal_workers}")
        logger.info(f"   Pin memory: {pin_memory}")
        logger.info(f"   Prefetch factor: {prefetch_factor}")
        logger.info(f"   Persistent workers: {persistent_workers}")
        
        return config

    @staticmethod
    def create_dataloader(dataset, platform_config: Optional[Dict] = None, **kwargs) -> DataLoader:
        """
        Create optimized DataLoader with best practices applied
        
        Args:
            dataset: PyTorch dataset
            platform_config: Platform-specific configuration
            **kwargs: Additional DataLoader arguments (override defaults)
            
        Returns:
            Optimized DataLoader instance
        """
        # Get optimized configuration
        config = OptimizedDataLoaderConfig.create_optimized_config(
            platform_config=platform_config,
            dataset_size=len(dataset) if hasattr(dataset, '__len__') else None
        )
        
        # Override with user-provided kwargs
        config.update(kwargs)
        
        # Create DataLoader
        dataloader = DataLoader(dataset, **config)
        
        logger.info(f"✅ Created optimized DataLoader for {len(dataset)} samples")
        return dataloader


def _worker_init_fn(worker_id: int):
    """
    Initialize DataLoader worker processes with optimizations
    
    CRITICAL: This fixes OpenCV threading conflicts that can cause 
    significant performance degradation in DataLoader workers
    """
    # Fix OpenCV threading conflicts (CRITICAL for performance)
    if OPENCV_AVAILABLE:
        cv2.setNumThreads(0)
    
    # Set worker-specific random seeds for reproducibility
    import numpy as np
    import random
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    # Optional: Set worker process priority (lower priority to not interfere with main process)
    try:
        import os
        os.nice(1)  # Lower priority on Unix systems
    except:
        pass  # Ignore on Windows or if permission denied


def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 10) -> Dict[str, float]:
    """
    Benchmark DataLoader performance for optimization validation
    
    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to process for timing
        
    Returns:
        Performance metrics dictionary
    """
    import time
    
    logger.info(f"🔍 Benchmarking DataLoader performance...")
    
    # Warm up
    for i, batch in enumerate(dataloader):
        if i >= 2:  # Warm up with 2 batches
            break
    
    # Timing
    start_time = time.time()
    batch_times = []
    
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Simulate some processing
        if isinstance(batch, (list, tuple)):
            for item in batch:
                if torch.is_tensor(item):
                    _ = item.sum()
        elif torch.is_tensor(batch):
            _ = batch.sum()
        
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        
        if i >= num_batches - 1:
            break
    
    total_time = time.time() - start_time
    
    metrics = {
        'total_time_seconds': total_time,
        'avg_batch_time_ms': (sum(batch_times) / len(batch_times)) * 1000,
        'batches_per_second': len(batch_times) / total_time,
        'samples_per_second': (len(batch_times) * dataloader.batch_size) / total_time,
    }
    
    logger.info(f"📊 DataLoader Performance:")
    logger.info(f"   Average batch time: {metrics['avg_batch_time_ms']:.1f}ms")
    logger.info(f"   Batches per second: {metrics['batches_per_second']:.1f}")
    logger.info(f"   Samples per second: {metrics['samples_per_second']:.1f}")
    
    return metrics


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, grandparent_dir)
    
    try:
        # Try relative import first
        from ..utils.platform_detector import get_platform_config
    except ImportError:
        # Fallback to direct import
        try:
            from src.utils.platform_detector import get_platform_config
        except ImportError:
            # Final fallback - create mock config
            def get_platform_config():
                return {
                    'platform_type': 'CPU',
                    'cpu_cores': 4,
                    'gpu_memory_gb': 0,
                    'optimization_tier': 'basic'
                }
    
    print("🔍 Testing Optimized DataLoader Configuration...")
    print(f"OpenCV Available: {OPENCV_AVAILABLE}")
    
    # Test configuration generation
    try:
        platform_config = get_platform_config()
        print(f"Platform: {platform_config.get('platform_type', 'Unknown')}")
        
        config = OptimizedDataLoaderConfig.create_optimized_config(
            platform_config=platform_config
        )
        
        print("\n🚀 Generated optimized DataLoader configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print(f"\n✅ DataLoader optimization test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing DataLoader config: {e}")
        print("This is normal if platform_detector is not available.")
        
        # Show basic config anyway
        basic_config = OptimizedDataLoaderConfig.create_optimized_config()
        print("\n📋 Basic DataLoader configuration (no platform detection):")
        for key, value in basic_config.items():
            print(f"  {key}: {value}")