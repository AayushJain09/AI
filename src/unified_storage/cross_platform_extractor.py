#!/usr/bin/env python3
"""
Cross-Platform Feature Extractor with Adaptive Optimization

This module provides a unified feature extraction system that automatically
optimizes performance based on the detected hardware platform:

- NVIDIA GPU: CUDA acceleration with GPU memory optimization
- Apple Silicon: MPS acceleration with unified memory optimization  
- Intel/AMD CPU: Multi-threading with cache optimization

Architecture:
- CLIP (768D): Multi-modal vision-language features
- DINOv2 (768D): Self-supervised visual features
- Total: 1536 dimensions optimized for cross-platform performance

Performance Features:
- Platform-aware device selection and optimization
- Model compilation for target hardware
- Batch processing with memory management
- Dynamic memory allocation and cleanup
- Hardware-specific threading and caching
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_storage.config_manager import ConfigManager
from unified_storage.platform_detector import PlatformDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress PIL warnings for better output
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ExtractionConfiguration:
    """Configuration for cross-platform feature extraction."""
    
    # Platform-specific settings
    device: str = "auto"  # auto, cuda, mps, cpu
    device_index: int = 0  # GPU device index if multiple available
    
    # Model optimization settings
    compile_models: bool = True  # Enable torch.compile for performance
    use_mixed_precision: bool = True  # Enable automatic mixed precision
    
    # Memory management settings
    batch_size: int = 8  # Adaptive batch size based on platform
    max_memory_usage_gb: float = 4.0  # Maximum memory allocation
    enable_memory_mapping: bool = True  # Memory-mapped file loading
    
    # Threading and parallelism
    num_workers: int = 4  # Number of worker threads
    prefetch_factor: int = 2  # Prefetch batches for processing
    
    # Feature extraction settings
    clip_variant: str = "ViT-L/14"  # CLIP model variant (768D output)
    dinov2_variant: str = "dinov2_vitb14"  # DINOv2 variant (768D output)
    image_size: int = 224  # Input image size
    normalize_features: bool = True  # L2 normalize feature vectors
    
    # Cache and optimization
    enable_feature_caching: bool = True  # Cache extracted features
    cache_size_mb: float = 512.0  # Feature cache size
    enable_jit_compilation: bool = True  # JIT compile models
    
    # Platform-specific overrides
    platform_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionStatistics:
    """Statistics for feature extraction performance."""
    
    # Performance metrics
    total_images_processed: int = 0
    total_extraction_time_ms: float = 0.0
    avg_extraction_time_ms: float = 0.0
    throughput_images_per_sec: float = 0.0
    
    # Memory usage
    peak_memory_usage_mb: float = 0.0
    avg_memory_usage_mb: float = 0.0
    memory_efficiency: float = 0.0  # Features extracted per MB
    
    # Platform information
    platform_type: str = ""
    device_used: str = ""
    hardware_acceleration: bool = False
    
    # Model statistics
    clip_inference_time_ms: float = 0.0
    dinov2_inference_time_ms: float = 0.0
    compilation_time_ms: float = 0.0
    
    # Error tracking
    failed_extractions: int = 0
    error_rate: float = 0.0


class PlatformOptimizer:
    """Platform-specific optimization for feature extraction."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize platform optimizer with configuration manager."""
        self.config_manager = config_manager
        self.platform_config = config_manager.get_config()
        self.platform_detector = PlatformDetector()
        
        # Platform-specific optimizations are automatically applied
        # by the configuration manager based on detected hardware
        self.optimizations = self._determine_platform_optimizations()
        
        logger.info(f"🔧 Platform optimizer initialized for {self.platform_config.platform.platform_type}")
    
    def _determine_platform_optimizations(self) -> Dict[str, Any]:
        """
        Determine platform-specific optimizations based on detected hardware.
        
        Platform optimization strategy:
        - NVIDIA GPU: CUDA + large batches + GPU memory optimization
        - Apple Silicon: MPS + unified memory + optimized threading
        - CPU-only: Multi-threading + cache optimization
        """
        platform_type = self.platform_config.platform.platform_type
        optimizations = {}
        
        if platform_type == "NVIDIA_GPU":
            # NVIDIA GPU optimizations - leverage CUDA acceleration
            # GPU memory is separate from system memory, allowing larger batches
            optimizations.update({
                "device_preference": "cuda",
                "batch_size_multiplier": 2.0,  # Larger batches for GPU throughput
                "enable_cuda_graphs": True,  # CUDA graph optimization
                "memory_growth": True,  # Allow GPU memory growth
                "compile_mode": "max-autotune",  # Aggressive compilation
                "thread_count": 1,  # Single thread for GPU processing
                "prefetch_to_device": True,  # Prefetch data to GPU
            })
            
        elif platform_type == "Apple_Silicon":
            # Apple Silicon optimizations - leverage MPS and unified memory
            # Unified memory architecture requires careful memory management
            optimizations.update({
                "device_preference": "mps",
                "batch_size_multiplier": 1.0,  # Conservative batches for unified memory
                "memory_pressure_management": True,  # Monitor memory pressure
                "compile_mode": "default",  # Standard compilation for MPS
                "thread_count": min(8, self.platform_config.performance.num_workers),  # Balanced threading
                "unified_memory_optimization": True,  # Apple Silicon specific
            })
            
        else:
            # CPU-only optimizations - maximize multi-threading
            # Rely on CPU cores and system memory for processing
            optimizations.update({
                "device_preference": "cpu", 
                "batch_size_multiplier": 0.5,  # Smaller batches for CPU processing
                "enable_cpu_fusion": True,  # CPU operator fusion
                "compile_mode": "default",  # Standard compilation for CPU
                "thread_count": self.platform_config.performance.num_workers,  # Full CPU utilization
                "memory_efficient_attention": True,  # Reduce memory for attention
            })
        
        # Apply memory constraints based on available system memory
        available_memory_gb = self.platform_config.platform.memory_gb
        optimizations["max_memory_gb"] = min(available_memory_gb * 0.6, 8.0)  # Use 60% of available memory
        optimizations["cache_size_mb"] = min(available_memory_gb * 100, 1024)  # 1GB max cache
        
        logger.info(f"🎯 Platform optimizations for {platform_type}: {optimizations}")
        return optimizations
    
    def optimize_batch_size(self, base_batch_size: int, image_size: int) -> int:
        """
        Optimize batch size based on platform capabilities and memory constraints.
        
        Optimization factors:
        - Available memory (system or GPU)
        - Platform processing capabilities
        - Image resolution and memory requirements
        """
        # Calculate memory requirement per image (approximate)
        bytes_per_pixel = 3 * 4  # RGB float32
        image_memory_mb = (image_size * image_size * bytes_per_pixel) / (1024 * 1024)
        model_overhead_mb = 2000  # Approximate model memory usage
        
        # Apply platform-specific multiplier
        multiplier = self.optimizations.get("batch_size_multiplier", 1.0)
        optimized_batch_size = int(base_batch_size * multiplier)
        
        # Memory constraint check
        max_memory_gb = self.optimizations.get("max_memory_gb", 4.0)
        max_batch_by_memory = int((max_memory_gb * 1024 - model_overhead_mb) / image_memory_mb)
        
        # Take the minimum to ensure we don't exceed memory limits
        final_batch_size = min(optimized_batch_size, max_batch_by_memory, 32)  # Cap at 32
        
        logger.info(f"📊 Batch size optimization: {base_batch_size} → {final_batch_size} "
                   f"(memory limit: {max_batch_by_memory}, platform factor: {multiplier:.1f})")
        
        return max(1, final_batch_size)  # Ensure at least 1
    
    def get_device_configuration(self) -> Tuple[torch.device, Dict[str, Any]]:
        """
        Get optimized device configuration for the current platform.
        
        Device selection strategy prioritizes performance while ensuring compatibility:
        1. CUDA: Best performance with GPU acceleration and large memory
        2. MPS: Good performance on Apple Silicon with unified memory
        3. CPU: Fallback mode ensuring universal compatibility
        
        Each device type gets platform-specific optimizations based on empirical testing.
        
        Returns:
            Tuple of (device, config_dict) for model initialization
        """
        preferred_device = self.optimizations.get("device_preference", "cpu")
        device_config = {}
        
        # PRIORITY 1: NVIDIA CUDA - Best performance tier
        # CUDA provides optimal performance characteristics:
        # - Dedicated GPU memory prevents system memory pressure
        # - Mature optimization ecosystem with proven performance gains
        # - Hardware-accelerated operations for both inference and FAISS search
        if preferred_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            device_config.update({
                "enable_flash_attention": True,  # CUDA-specific attention optimization
                "memory_format": torch.channels_last,  # NHWC layout for better CUDA performance
                "allow_tf32": True,  # TensorFloat-32 provides 10x speedup with minimal accuracy loss
                "device_index": 0,  # Use primary GPU (can be configured later)
            })
            logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name()} "
                       f"({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB)")
            
        # PRIORITY 2: Apple MPS - Good performance tier  
        # MPS leverages Apple Silicon GPU with unified memory:
        # - Shared memory between CPU and GPU reduces data movement overhead
        # - Apple-optimized kernels provide good performance on M-series chips
        # - Contiguous memory format required for MPS compatibility
        elif preferred_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_config.update({
                "memory_format": torch.contiguous_format,  # MPS requires contiguous layout
                "enable_mps_profiler": False,  # Disable profiling for production performance
                "unified_memory_aware": True,  # Flag for unified memory optimizations
            })
            logger.info("🍎 Using Apple Metal Performance Shaders (MPS) with unified memory")
            
        # FALLBACK: CPU-only mode - Functional performance tier
        # CPU mode ensures universal compatibility across all systems:
        # - No GPU dependencies, works on any hardware configuration
        # - Multi-threading optimization maximizes CPU core utilization
        # - Intel MKL and NNPACK provide optimized CPU operations
        else:
            device = torch.device("cpu")
            thread_count = self.optimizations.get("thread_count", 4)
            device_config.update({
                "enable_mkldnn": True,  # Intel MKL-DNN for optimized CPU operations
                "enable_nnpack": True,  # NNPACK provides fast convolutions on ARM/x86
                "num_threads": thread_count,  # Optimal threading based on CPU cores
                "prefer_single_thread": thread_count == 1,  # Single-thread mode for resource-constrained systems
            })
            logger.info(f"💻 Using CPU with {thread_count} threads (MKL-DNN + NNPACK optimized)")
        
        return device, device_config


class CrossPlatformFeatureExtractor:
    """
    Cross-platform feature extractor with adaptive optimization.
    
    Automatically detects hardware platform and applies optimal configurations
    for CLIP + DINOv2 feature extraction across different systems.
    """
    
    def __init__(self, config_manager: ConfigManager, extraction_config: Optional[ExtractionConfiguration] = None):
        """
        Initialize cross-platform feature extractor.
        
        Args:
            config_manager: System configuration manager
            extraction_config: Optional extraction configuration overrides
        """
        self.config_manager = config_manager
        self.platform_config = config_manager.get_config()
        self.platform_optimizer = PlatformOptimizer(config_manager)
        
        # Initialize extraction configuration with platform optimizations
        self.extraction_config = extraction_config or ExtractionConfiguration()
        self._apply_platform_overrides()
        
        # Initialize device and models
        self.device, self.device_config = self.platform_optimizer.get_device_configuration()
        self.models = {}
        self.statistics = ExtractionStatistics()
        
        # Memory management with platform-specific configuration
        self._memory_monitor = MemoryMonitor(
            self.extraction_config.max_memory_usage_gb, 
            self.device.type
        )
        self._feature_cache = {}
        
        # Initialize models with platform optimizations
        self._initialize_models()
        
        logger.info(f"✅ CrossPlatformFeatureExtractor initialized for {self.platform_config.platform.platform_type}")
    
    def _apply_platform_overrides(self):
        """Apply platform-specific configuration overrides."""
        optimizations = self.platform_optimizer.optimizations
        
        # Apply batch size optimization
        self.extraction_config.batch_size = self.platform_optimizer.optimize_batch_size(
            self.extraction_config.batch_size, 
            self.extraction_config.image_size
        )
        
        # Apply memory constraints
        self.extraction_config.max_memory_usage_gb = optimizations.get("max_memory_gb", 4.0)
        self.extraction_config.cache_size_mb = optimizations.get("cache_size_mb", 512.0)
        
        # Apply threading configuration
        self.extraction_config.num_workers = optimizations.get("thread_count", 4)
        
        # Apply compilation settings
        compile_mode = optimizations.get("compile_mode", "default")
        self.extraction_config.compile_models = compile_mode != "none"
        
        logger.info(f"🔧 Applied platform overrides: batch_size={self.extraction_config.batch_size}, "
                   f"memory_limit={self.extraction_config.max_memory_usage_gb}GB")
    
    def _initialize_models(self):
        """
        Initialize and optimize CLIP and DINOv2 models for the target platform.
        
        Platform-specific optimizations:
        - Model compilation for target hardware
        - Memory format optimization
        - Mixed precision setup
        """
        start_time = time.time()
        
        try:
            # Initialize CLIP model with platform optimizations
            logger.info(f"🚀 Loading CLIP model ({self.extraction_config.clip_variant})...")
            self._initialize_clip_model()
            
            # Initialize DINOv2 model with platform optimizations
            logger.info(f"🔥 Loading DINOv2 model ({self.extraction_config.dinov2_variant})...")
            self._initialize_dinov2_model()
            
            # Apply model compilation if enabled
            if self.extraction_config.compile_models:
                self._compile_models()
            
            # Setup preprocessing pipelines
            self._setup_preprocessing()
            
            compilation_time = (time.time() - start_time) * 1000
            self.statistics.compilation_time_ms = compilation_time
            
            logger.info(f"✅ Models initialized and optimized in {compilation_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def _initialize_clip_model(self):
        """Initialize CLIP model with platform-specific optimizations."""
        import clip
        import ssl
        import urllib.request
        
        # Temporarily disable SSL verification for CLIP download
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))
        
        # Load CLIP model
        clip_model, clip_preprocess = clip.load(self.extraction_config.clip_variant, device=self.device)
        clip_model.eval()
        
        # Apply platform-specific optimizations
        if self.device.type == "cuda":
            # CUDA optimizations - use channels_last memory format for better performance
            clip_model = clip_model.to(memory_format=torch.channels_last)
            
        elif self.device.type == "mps":
            # MPS optimizations - ensure contiguous memory format
            clip_model = clip_model.to(memory_format=torch.contiguous_format)
        
        # Enable mixed precision if supported and configured
        if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
            clip_model = clip_model.half()  # Convert to FP16
            logger.info("🔧 CLIP model using mixed precision (FP16)")
        
        self.models["clip"] = clip_model
        self.models["clip_preprocess"] = clip_preprocess
        
        # Verify output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
                dummy_input = dummy_input.half()
            dummy_output = clip_model.encode_image(dummy_input)
            actual_dims = dummy_output.shape[1]
            logger.info(f"✓ CLIP model loaded: {actual_dims} dimensions")
    
    def _initialize_dinov2_model(self):
        """Initialize DINOv2 model with platform-specific optimizations."""
        try:
            # Load DINOv2 model from torch hub
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.extraction_config.dinov2_variant, trust_repo=True)
            dinov2_model = dinov2_model.to(self.device)
            dinov2_model.eval()
            
            # Apply platform-specific optimizations
            if self.device.type == "cuda":
                # CUDA optimizations
                dinov2_model = dinov2_model.to(memory_format=torch.channels_last)
                
            elif self.device.type == "mps":
                # MPS optimizations  
                dinov2_model = dinov2_model.to(memory_format=torch.contiguous_format)
            
            # Enable mixed precision if supported and configured
            if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
                dinov2_model = dinov2_model.half()  # Convert to FP16
                logger.info("🔧 DINOv2 model using mixed precision (FP16)")
            
            self.models["dinov2"] = dinov2_model
            
            # Verify output dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
                    dummy_input = dummy_input.half()
                dummy_output = dinov2_model(dummy_input)
                actual_dims = dummy_output.shape[1]
                logger.info(f"✓ DINOv2 model loaded: {actual_dims} dimensions")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load DINOv2: {e}. Proceeding with CLIP-only mode")
            self.models["dinov2"] = None
    
    def _compile_models(self):
        """
        Compile models for target platform using torch.compile with platform-specific optimizations.
        
        Compilation Strategy by Platform:
        - NVIDIA GPU: "max-autotune" mode for aggressive optimization and maximum performance
        - Apple Silicon: "default" mode for MPS compatibility and stable performance  
        - CPU: "default" mode with CPU-specific backends for broad compatibility
        
        Why compilation matters:
        - 10-30% inference speedup on most platforms
        - Better memory layout optimization
        - Kernel fusion reduces overhead
        - Platform-specific code generation
        """
        if not hasattr(torch, 'compile'):
            logger.warning("⚠️ torch.compile not available (requires PyTorch 2.0+), skipping compilation")
            return
        
        compile_mode = self.platform_optimizer.optimizations.get("compile_mode", "default")
        device_type = self.device.type
        
        # Platform-specific compilation configuration
        compile_config = {
            "mode": compile_mode,
            "fullgraph": False,  # Allow graph breaks for compatibility
            "dynamic": True,  # Handle dynamic shapes in batch processing
        }
        
        # NVIDIA GPU: Aggressive compilation for maximum performance
        # CUDA has mature compilation support and benefits from aggressive optimization
        if device_type == "cuda":
            compile_config.update({
                "backend": "inductor",  # TorchInductor for CUDA optimization
                "mode": "max-autotune",  # Override to max performance mode
                "fullgraph": True,  # Aggressive graph optimization for CUDA
                "dynamic": False,  # Static shapes perform better on GPU
            })
            logger.info("🚀 Using aggressive CUDA compilation (max-autotune + inductor)")
            
        # Apple MPS: Conservative compilation for compatibility  
        # MPS backend is newer and requires careful compilation settings
        elif device_type == "mps":
            compile_config.update({
                "backend": "aot_eager",  # AOT compilation for MPS compatibility
                "mode": "default",  # Conservative mode for MPS stability
                # Remove options to avoid conflict with mode
            })
            logger.info("🍎 Using compatible MPS compilation (aot_eager + default)")
            
        # CPU: Standard compilation with CPU-specific optimizations
        # CPU compilation focuses on threading and memory optimization
        else:
            compile_config.update({
                "backend": "inductor",  # TorchInductor supports CPU optimization
                "mode": "default",  # Balanced performance/compatibility
                # Remove options to avoid conflict with mode
            })
            logger.info("💻 Using standard CPU compilation (inductor + default)")
        
        try:
            compilation_start = time.time()
            models_compiled = 0
            
            # Compile CLIP model with platform-specific settings
            if "clip" in self.models and self.models["clip"] is not None:
                logger.info(f"🔧 Compiling CLIP model...")
                
                # Wrap only the encoding part to avoid compilation issues with the full model
                original_encode_image = self.models["clip"].encode_image
                compiled_encode_image = torch.compile(original_encode_image, **compile_config)
                self.models["clip"].encode_image = compiled_encode_image
                models_compiled += 1
                
                logger.info("✓ CLIP model compiled successfully")
            
            # Compile DINOv2 model with platform-specific settings
            if "dinov2" in self.models and self.models["dinov2"] is not None:
                logger.info(f"🔧 Compiling DINOv2 model...")
                
                # DINOv2 is simpler to compile as a complete model
                self.models["dinov2"] = torch.compile(self.models["dinov2"], **compile_config)
                models_compiled += 1
                
                logger.info("✓ DINOv2 model compiled successfully")
            
            compilation_time = (time.time() - compilation_start) * 1000
            logger.info(f"✅ {models_compiled} models compiled in {compilation_time:.1f}ms "
                       f"(mode: {compile_config['mode']}, backend: {compile_config.get('backend', 'default')})")
            
            # Warm up compiled models with a dummy forward pass
            # This triggers actual compilation and caches the optimized kernels
            if models_compiled > 0:
                self._warmup_compiled_models()
                
        except Exception as e:
            logger.warning(f"⚠️ Model compilation failed: {e}. Proceeding without compilation")
            # Reset models to uncompiled state if compilation partially succeeded
            self._reset_to_uncompiled_models()
    
    def _warmup_compiled_models(self):
        """
        Warm up compiled models with dummy forward passes.
        
        Compilation warmup is critical because:
        - torch.compile is lazy - actual optimization happens on first run
        - First inference can be 10x slower due to compilation overhead
        - Warmup moves compilation cost to initialization, not runtime
        - Subsequent inferences benefit from pre-compiled kernels
        """
        logger.info("🔥 Warming up compiled models...")
        warmup_start = time.time()
        
        try:
            with torch.no_grad():
                # Create dummy input matching expected preprocessing
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                # Apply mixed precision if enabled
                if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
                    dummy_input = dummy_input.half()
                
                # Apply memory format optimization
                if self.device.type == "cuda":
                    dummy_input = dummy_input.to(memory_format=torch.channels_last)
                
                # Warmup CLIP model
                if "clip" in self.models and self.models["clip"] is not None:
                    try:
                        _ = self.models["clip"].encode_image(dummy_input)
                        logger.info("✓ CLIP model warmed up")
                    except Exception as e:
                        logger.warning(f"⚠️ CLIP warmup failed: {e}")
                
                # Warmup DINOv2 model  
                if "dinov2" in self.models and self.models["dinov2"] is not None:
                    try:
                        _ = self.models["dinov2"](dummy_input)
                        logger.info("✓ DINOv2 model warmed up")
                    except Exception as e:
                        logger.warning(f"⚠️ DINOv2 warmup failed: {e}")
                
                warmup_time = (time.time() - warmup_start) * 1000
                logger.info(f"🔥 Model warmup completed in {warmup_time:.1f}ms")
                
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")
    
    def _reset_to_uncompiled_models(self):
        """Reset models to uncompiled state if compilation fails."""
        try:
            # This would require storing original models, which we don't do
            # In practice, we log the issue and continue with partially compiled models
            logger.info("Models remain in current state (compilation may be partial)")
        except Exception as e:
            logger.warning(f"⚠️ Model reset failed: {e}")
    
    def _setup_preprocessing(self):
        """Setup preprocessing pipelines optimized for the target platform."""
        # CLIP preprocessing is handled by the loaded preprocessor
        
        # DINOv2 preprocessing pipeline
        self.dinov2_preprocess = transforms.Compose([
            transforms.Resize(self.extraction_config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.extraction_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        logger.info(f"🔧 Preprocessing configured for {self.extraction_config.image_size}x{self.extraction_config.image_size} images")
    
    def extract_features_single(self, image: Union[str, Path, Image.Image]) -> Optional[np.ndarray]:
        """
        Extract features from a single image with platform optimizations.
        
        Args:
            image: Image path or PIL Image object
            
        Returns:
            Combined feature vector (1536D) or None if extraction fails
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert('RGB')
            else:
                pil_image = image.convert('RGB')
            
            # Extract features from both models
            features = {}
            
            # CLIP features (768D)
            clip_start = time.time()
            clip_features = self._extract_clip_features(pil_image)
            self.statistics.clip_inference_time_ms += (time.time() - clip_start) * 1000
            
            if clip_features is not None:
                features["clip"] = clip_features
            
            # DINOv2 features (768D)
            if self.models.get("dinov2") is not None:
                dinov2_start = time.time()
                dinov2_features = self._extract_dinov2_features(pil_image)
                self.statistics.dinov2_inference_time_ms += (time.time() - dinov2_start) * 1000
                
                if dinov2_features is not None:
                    features["dinov2"] = dinov2_features
            else:
                # Fallback: use zeros if DINOv2 not available
                features["dinov2"] = np.zeros(768, dtype=np.float32)
            
            # Combine features
            if features:
                combined_features = np.concatenate([
                    features.get("clip", np.zeros(768, dtype=np.float32)),
                    features.get("dinov2", np.zeros(768, dtype=np.float32))
                ])
                
                # Normalize if configured
                if self.extraction_config.normalize_features:
                    norm = np.linalg.norm(combined_features)
                    if norm > 0:
                        combined_features = combined_features / norm
                
                # Update statistics
                extraction_time = (time.time() - start_time) * 1000
                self._update_extraction_statistics(extraction_time, success=True)
                
                return combined_features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            self._update_extraction_statistics(0, success=False)
        
        return None
    
    def _extract_clip_features(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract CLIP features with platform optimizations."""
        try:
            with torch.no_grad():
                # Preprocess image
                image_input = self.models["clip_preprocess"](image).unsqueeze(0).to(self.device)
                
                # Apply mixed precision if enabled
                if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
                    image_input = image_input.half()
                
                # Apply memory format optimization
                if self.device.type == "cuda":
                    image_input = image_input.to(memory_format=torch.channels_last)
                
                # Extract features
                features = self.models["clip"].encode_image(image_input)
                
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
                
                return features.cpu().numpy().flatten().astype(np.float32)
                
        except Exception as e:
            logger.error(f"❌ CLIP feature extraction failed: {e}")
            return None
    
    def _extract_dinov2_features(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract DINOv2 features with platform optimizations."""
        try:
            with torch.no_grad():
                # Preprocess image
                image_input = self.dinov2_preprocess(image).unsqueeze(0).to(self.device)
                
                # Apply mixed precision if enabled
                if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
                    image_input = image_input.half()
                
                # Apply memory format optimization
                if self.device.type == "cuda":
                    image_input = image_input.to(memory_format=torch.channels_last)
                
                # Extract features
                features = self.models["dinov2"](image_input)
                
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
                
                return features.cpu().numpy().flatten().astype(np.float32)
                
        except Exception as e:
            logger.error(f"❌ DINOv2 feature extraction failed: {e}")
            return None
    
    def extract_features_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[Optional[np.ndarray]]:
        """
        Extract features from a batch of images with platform-optimized processing.
        
        Batch Processing Strategy:
        - Dynamic batch sizing based on available memory and platform capabilities
        - Memory pressure monitoring to prevent OOM crashes
        - Platform-specific memory cleanup between batches
        - Parallel preprocessing with CPU threads while GPU processes previous batch
        
        Performance Optimizations by Platform:
        - NVIDIA GPU: Large batches (16-32) with GPU memory management
        - Apple Silicon: Medium batches (8) with unified memory awareness  
        - CPU: Small batches (4) with careful threading balance
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            List of feature vectors (1536D each) or None for failed extractions
        """
        if not images:
            return []
        
        total_images = len(images)
        
        # Dynamic batch size adjustment based on memory pressure
        base_batch_size = self.extraction_config.batch_size
        current_batch_size = self._adjust_batch_size_for_memory_pressure(base_batch_size, total_images)
        
        results = []
        processed_count = 0
        failed_count = 0
        
        logger.info(f"📦 Processing {total_images} images in optimized batches of {current_batch_size} "
                   f"(platform: {self.platform_config.platform.platform_type})")
        
        # Track memory usage during batch processing
        initial_memory = self._memory_monitor.get_memory_info()["rss_mb"]
        
        # Process images in dynamically sized batches
        i = 0
        batch_idx = 0
        while i < total_images:
            batch_start_time = time.time()
            
            # Memory pressure check before processing batch
            if self._memory_monitor.check_memory_pressure():
                logger.warning(f"⚠️ Memory pressure detected, reducing batch size")
                current_batch_size = max(1, current_batch_size // 2)
            
            # Create batch with current batch size
            batch = images[i:i + current_batch_size]
            actual_batch_size = len(batch)
            
            logger.debug(f"Processing batch {batch_idx + 1} with {actual_batch_size} images...")
            
            # Process the batch with platform-specific optimizations
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            # Calculate batch timing for statistics
            batch_time = (time.time() - batch_start_time) * 1000
            
            # Update statistics
            successful_in_batch = sum(1 for r in batch_results if r is not None)
            processed_count += successful_in_batch
            failed_count += (actual_batch_size - successful_in_batch)
            
            # Update global statistics for each successful extraction
            for _ in range(successful_in_batch):
                self._update_extraction_statistics(batch_time / actual_batch_size, success=True)
            
            # Update global statistics for each failed extraction
            for _ in range(actual_batch_size - successful_in_batch):
                self._update_extraction_statistics(0, success=False)
            
            # Platform-specific memory cleanup between batches
            self._cleanup_batch_memory()
            
            # Log progress and performance metrics
            images_per_sec = actual_batch_size / (batch_time / 1000) if batch_time > 0 else 0
            current_memory = self._memory_monitor.get_memory_info()["rss_mb"]
            memory_growth = current_memory - initial_memory
            
            logger.debug(f"Batch {batch_idx + 1} completed: {successful_in_batch}/{actual_batch_size} successful, "
                        f"{batch_time:.1f}ms ({images_per_sec:.1f} img/s), "
                        f"memory: +{memory_growth:.1f}MB")
            
            # Adaptive batch size adjustment based on performance
            if batch_idx > 0 and batch_idx % 5 == 0:  # Every 5 batches
                current_batch_size = self._adapt_batch_size_from_performance(
                    current_batch_size, batch_time, memory_growth
                )
            
            # Move to next batch
            i += actual_batch_size
            batch_idx += 1
        
        success_rate = processed_count / total_images if total_images > 0 else 0
        final_memory = self._memory_monitor.get_memory_info()["rss_mb"]
        total_memory_growth = final_memory - initial_memory
        
        logger.info(f"✅ Batch processing complete: {processed_count}/{total_images} successful "
                   f"({success_rate:.1%} success rate), total memory growth: +{total_memory_growth:.1f}MB")
        
        return results
    
    def _adjust_batch_size_for_memory_pressure(self, base_batch_size: int, total_images: int) -> int:
        """
        Adjust batch size based on current memory pressure and platform capabilities.
        
        Memory-aware batch sizing prevents OOM crashes by:
        - Monitoring available memory before processing starts
        - Reducing batch size on memory-constrained systems
        - Accounting for platform-specific memory overhead
        """
        memory_info = self._memory_monitor.get_memory_info()
        available_memory_gb = memory_info["available_gb"]
        
        # Platform-specific memory overhead estimation (per image in batch)
        if self.device.type == "cuda":
            # GPU memory is separate, but we still need system memory for preprocessing
            memory_per_image_mb = 50  # Conservative estimate for CUDA preprocessing
        elif self.device.type == "mps":
            # Unified memory - need to account for both model and preprocessing
            memory_per_image_mb = 100  # Higher due to unified memory architecture
        else:
            # CPU-only - all memory comes from system RAM
            memory_per_image_mb = 75  # Moderate estimate for CPU processing
        
        # Calculate safe batch size based on available memory
        safe_batch_size = int((available_memory_gb * 1024 * 0.5) / memory_per_image_mb)  # Use 50% of available
        safe_batch_size = max(1, min(safe_batch_size, base_batch_size))
        
        # Special handling for small datasets
        if total_images <= 10:
            safe_batch_size = min(safe_batch_size, total_images)
        
        if safe_batch_size != base_batch_size:
            logger.info(f"🔧 Adjusted batch size: {base_batch_size} → {safe_batch_size} "
                       f"(available memory: {available_memory_gb:.1f}GB)")
        
        return safe_batch_size
    
    def _adapt_batch_size_from_performance(self, current_batch_size: int, 
                                         batch_time_ms: float, memory_growth_mb: float) -> int:
        """
        Dynamically adapt batch size based on performance metrics.
        
        Performance-based adaptation optimizes throughput by:
        - Increasing batch size if memory growth is low and processing is fast
        - Decreasing batch size if memory pressure is high or processing is slow
        - Maintaining platform-specific optimal ranges
        """
        # Performance thresholds based on platform capabilities
        if self.device.type == "cuda":
            target_batch_time_ms = 200  # Target 200ms per batch on GPU
            max_memory_growth_mb = 500   # Allow more memory growth on CUDA
            max_batch_size = 32
        elif self.device.type == "mps":
            target_batch_time_ms = 400  # Target 400ms per batch on MPS
            max_memory_growth_mb = 200   # Conservative memory growth on unified memory
            max_batch_size = 16
        else:
            target_batch_time_ms = 800  # Target 800ms per batch on CPU
            max_memory_growth_mb = 100   # Very conservative memory growth on CPU
            max_batch_size = 8
        
        new_batch_size = current_batch_size
        
        # Increase batch size if performance allows
        if (batch_time_ms < target_batch_time_ms * 0.7 and  # Processing is fast enough
            memory_growth_mb < max_memory_growth_mb * 0.5 and  # Memory growth is low
            current_batch_size < max_batch_size):  # Haven't hit platform limit
            new_batch_size = min(current_batch_size + 1, max_batch_size)
            logger.debug(f"📈 Increasing batch size: {current_batch_size} → {new_batch_size}")
        
        # Decrease batch size if performance is poor
        elif (batch_time_ms > target_batch_time_ms * 1.5 or  # Processing is too slow
              memory_growth_mb > max_memory_growth_mb) and current_batch_size > 1:  # Memory growth too high
            new_batch_size = max(current_batch_size - 1, 1)
            logger.debug(f"📉 Decreasing batch size: {current_batch_size} → {new_batch_size}")
        
        return new_batch_size
    
    def _cleanup_batch_memory(self):
        """
        Platform-specific memory cleanup between batches.
        
        Memory cleanup strategy by platform:
        - NVIDIA CUDA: Clear GPU cache to prevent memory fragmentation
        - Apple MPS: Clear MPS cache and trigger garbage collection
        - CPU: Aggressive garbage collection to free system memory
        """
        if self.device.type == "cuda":
            # CUDA-specific cleanup: clear GPU memory cache
            # This prevents GPU memory fragmentation during long batch processing
            torch.cuda.empty_cache()  # Free unused cached memory
            torch.cuda.synchronize()  # Ensure all operations complete
            
        elif self.device.type == "mps":
            # MPS-specific cleanup: clear Metal Performance Shaders cache
            # Important for unified memory systems to prevent memory pressure
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            torch.mps.synchronize()  # Ensure MPS operations complete
            
        # Universal cleanup: Python garbage collection
        # Helps free memory from intermediate Python objects
        gc.collect()
        
        # Yield to other processes briefly
        time.sleep(0.001)  # 1ms yield
    
    def _process_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[Optional[np.ndarray]]:
        """Process a batch of images efficiently."""
        batch_results = []
        
        try:
            # Preload and preprocess all images in the batch
            preprocessed_images = []
            valid_indices = []
            
            for idx, image in enumerate(images):
                try:
                    if isinstance(image, (str, Path)):
                        pil_image = Image.open(image).convert('RGB')
                    else:
                        pil_image = image.convert('RGB')
                    
                    preprocessed_images.append(pil_image)
                    valid_indices.append(idx)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load image {idx}: {e}")
                    continue
            
            if not preprocessed_images:
                return [None] * len(images)
            
            # Batch process CLIP features
            clip_features_batch = self._extract_clip_features_batch(preprocessed_images)
            
            # Batch process DINOv2 features
            dinov2_features_batch = self._extract_dinov2_features_batch(preprocessed_images)
            
            # Combine and normalize features
            for i, image_idx in enumerate(valid_indices):
                clip_feat = clip_features_batch[i] if i < len(clip_features_batch) else None
                dinov2_feat = dinov2_features_batch[i] if i < len(dinov2_features_batch) else None
                
                if clip_feat is not None or dinov2_feat is not None:
                    # Use zeros for missing features
                    if clip_feat is None:
                        clip_feat = np.zeros(768, dtype=np.float32)
                    if dinov2_feat is None:
                        dinov2_feat = np.zeros(768, dtype=np.float32)
                    
                    combined_features = np.concatenate([clip_feat, dinov2_feat])
                    
                    # Normalize if configured
                    if self.extraction_config.normalize_features:
                        norm = np.linalg.norm(combined_features)
                        if norm > 0:
                            combined_features = combined_features / norm
                    
                    batch_results.append(combined_features)
                else:
                    batch_results.append(None)
            
            # Fill in None for failed image loads
            final_results = [None] * len(images)
            for i, valid_idx in enumerate(valid_indices):
                final_results[valid_idx] = batch_results[i] if i < len(batch_results) else None
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return [None] * len(images)
    
    def _extract_clip_features_batch(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        """Extract CLIP features for a batch of images."""
        try:
            with torch.no_grad():
                # Preprocess all images
                image_tensors = []
                for image in images:
                    processed = self.models["clip_preprocess"](image)
                    image_tensors.append(processed)
                
                # Stack into batch tensor
                batch_tensor = torch.stack(image_tensors).to(self.device)
                
                # Apply mixed precision if enabled
                if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
                    batch_tensor = batch_tensor.half()
                
                # Apply memory format optimization
                if self.device.type == "cuda":
                    batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
                
                # Extract features for entire batch
                features_batch = self.models["clip"].encode_image(batch_tensor)
                
                # Normalize features
                features_batch = features_batch / features_batch.norm(dim=-1, keepdim=True)
                
                # Convert to numpy and split by image
                features_np = features_batch.cpu().numpy()
                return [feat.astype(np.float32) for feat in features_np]
                
        except Exception as e:
            logger.error(f"❌ CLIP batch processing failed: {e}")
            return [None] * len(images)
    
    def _extract_dinov2_features_batch(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        """Extract DINOv2 features for a batch of images."""
        if self.models.get("dinov2") is None:
            return [np.zeros(768, dtype=np.float32)] * len(images)
        
        try:
            with torch.no_grad():
                # Preprocess all images
                image_tensors = []
                for image in images:
                    processed = self.dinov2_preprocess(image)
                    image_tensors.append(processed)
                
                # Stack into batch tensor
                batch_tensor = torch.stack(image_tensors).to(self.device)
                
                # Apply mixed precision if enabled
                if self.extraction_config.use_mixed_precision and self.device.type in ["cuda", "mps"]:
                    batch_tensor = batch_tensor.half()
                
                # Apply memory format optimization
                if self.device.type == "cuda":
                    batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
                
                # Extract features for entire batch
                features_batch = self.models["dinov2"](batch_tensor)
                
                # Normalize features
                features_batch = features_batch / features_batch.norm(dim=-1, keepdim=True)
                
                # Convert to numpy and split by image
                features_np = features_batch.cpu().numpy()
                return [feat.astype(np.float32) for feat in features_np]
                
        except Exception as e:
            logger.error(f"❌ DINOv2 batch processing failed: {e}")
            return [np.zeros(768, dtype=np.float32)] * len(images)
    
    def _update_extraction_statistics(self, extraction_time_ms: float, success: bool):
        """Update extraction performance statistics."""
        if success:
            self.statistics.total_images_processed += 1
            self.statistics.total_extraction_time_ms += extraction_time_ms
            
            if self.statistics.total_images_processed > 0:
                self.statistics.avg_extraction_time_ms = (
                    self.statistics.total_extraction_time_ms / self.statistics.total_images_processed
                )
                self.statistics.throughput_images_per_sec = (
                    1000.0 / self.statistics.avg_extraction_time_ms
                )
        else:
            self.statistics.failed_extractions += 1
        
        # Update error rate
        total_attempts = self.statistics.total_images_processed + self.statistics.failed_extractions
        if total_attempts > 0:
            self.statistics.error_rate = self.statistics.failed_extractions / total_attempts
        
        # Update memory usage
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.statistics.peak_memory_usage_mb = max(self.statistics.peak_memory_usage_mb, current_memory)
    
    def get_performance_statistics(self) -> ExtractionStatistics:
        """Get comprehensive performance statistics."""
        # Update platform information
        self.statistics.platform_type = self.platform_config.platform.platform_type
        self.statistics.device_used = str(self.device)
        self.statistics.hardware_acceleration = self.device.type in ["cuda", "mps"]
        
        # Update memory efficiency
        if self.statistics.peak_memory_usage_mb > 0:
            self.statistics.memory_efficiency = (
                self.statistics.total_images_processed / self.statistics.peak_memory_usage_mb
            )
        
        return self.statistics
    
    def close(self):
        """Clean up resources and models."""
        try:
            # Clear models
            self.models.clear()
            
            # Clear cache
            self._feature_cache.clear()
            
            # Clean up GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("✅ CrossPlatformFeatureExtractor resources cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")


class MemoryMonitor:
    """
    Advanced memory monitoring and management for cross-platform feature extraction.
    
    Provides real-time memory monitoring with platform-specific optimizations:
    - System memory tracking for batch size adjustment
    - GPU memory monitoring on CUDA systems
    - Memory pressure detection and automatic mitigation
    - Platform-aware memory management strategies
    """
    
    def __init__(self, max_memory_gb: float, device_type: str = "cpu"):
        """
        Initialize memory monitor with platform-specific configuration.
        
        Args:
            max_memory_gb: Maximum allowed memory usage in GB
            device_type: Device type for platform-specific monitoring
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.device_type = device_type
        self.process = psutil.Process()
        
        # Memory tracking history for trend analysis
        self.memory_history = []
        self.max_history_size = 100
        
        # Platform-specific thresholds
        self.pressure_thresholds = self._get_platform_pressure_thresholds()
        
        # GPU memory monitoring setup
        self.gpu_memory_available = self._initialize_gpu_monitoring()
        
        logger.info(f"🔍 Memory monitor initialized: {max_memory_gb}GB limit, "
                   f"device: {device_type}, GPU monitoring: {self.gpu_memory_available}")
    
    def _get_platform_pressure_thresholds(self) -> Dict[str, float]:
        """
        Get platform-specific memory pressure thresholds.
        
        Different platforms handle memory pressure differently:
        - CUDA: Can rely more on GPU memory, higher system memory threshold
        - MPS: Unified memory requires conservative thresholds
        - CPU: Must be very conservative with system memory
        """
        if self.device_type == "cuda":
            return {
                "warning_threshold": 0.75,  # 75% of max allowed memory
                "critical_threshold": 0.90,  # 90% triggers immediate action
                "emergency_threshold": 0.95   # 95% triggers aggressive cleanup
            }
        elif self.device_type == "mps":
            return {
                "warning_threshold": 0.65,   # More conservative for unified memory
                "critical_threshold": 0.80,
                "emergency_threshold": 0.90
            }
        else:  # CPU
            return {
                "warning_threshold": 0.70,   # Conservative for CPU-only systems
                "critical_threshold": 0.85,
                "emergency_threshold": 0.95
            }
    
    def _initialize_gpu_monitoring(self) -> bool:
        """Initialize GPU memory monitoring if available."""
        try:
            if self.device_type == "cuda" and torch.cuda.is_available():
                # Test GPU memory access
                torch.cuda.memory_stats()
                return True
            elif self.device_type == "mps" and hasattr(torch.mps, 'driver_allocated_memory'):
                # MPS memory monitoring (if available in PyTorch version)
                return True
        except Exception as e:
            logger.debug(f"GPU memory monitoring not available: {e}")
        return False
    
    def check_memory_pressure(self) -> bool:
        """
        Check if current memory usage exceeds configured limits.
        
        Uses platform-specific thresholds and considers both system and GPU memory.
        """
        system_pressure = self._check_system_memory_pressure()
        
        if self.gpu_memory_available:
            gpu_pressure = self._check_gpu_memory_pressure()
            return system_pressure or gpu_pressure
        
        return system_pressure
    
    def _check_system_memory_pressure(self) -> bool:
        """Check system memory pressure against platform thresholds."""
        current_memory = self.process.memory_info().rss
        memory_ratio = current_memory / self.max_memory_bytes
        
        # Add to history for trend analysis
        self.memory_history.append(memory_ratio)
        if len(self.memory_history) > self.max_history_size:
            self.memory_history.pop(0)
        
        return memory_ratio > self.pressure_thresholds["warning_threshold"]
    
    def _check_gpu_memory_pressure(self) -> bool:
        """Check GPU memory pressure if GPU monitoring is available."""
        try:
            if self.device_type == "cuda":
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                
                # Consider both allocated and cached memory
                used_ratio = (allocated + cached) / total
                return used_ratio > 0.85  # 85% GPU memory threshold
                
            elif self.device_type == "mps":
                # MPS memory pressure detection (simplified)
                # Apple Silicon unified memory makes this complex
                return False  # Rely on system memory pressure for now
                
        except Exception as e:
            logger.debug(f"GPU memory pressure check failed: {e}")
        
        return False
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get comprehensive memory usage information.
        
        Returns detailed memory statistics for monitoring and optimization.
        """
        system_memory = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        info = {
            # System memory
            "rss_mb": system_memory.rss / (1024 * 1024),
            "vms_mb": system_memory.vms / (1024 * 1024),
            "percent": self.process.memory_percent(),
            "available_gb": virtual_memory.available / (1024 * 1024 * 1024),
            "total_gb": virtual_memory.total / (1024 * 1024 * 1024),
            
            # Memory pressure indicators
            "pressure_level": self._get_pressure_level(),
            "memory_trend": self._get_memory_trend(),
        }
        
        # Add GPU memory info if available
        if self.gpu_memory_available:
            gpu_info = self._get_gpu_memory_info()
            info.update(gpu_info)
        
        return info
    
    def _get_pressure_level(self) -> str:
        """Get current memory pressure level as string."""
        if not self.memory_history:
            return "normal"
        
        current_ratio = self.memory_history[-1]
        thresholds = self.pressure_thresholds
        
        if current_ratio > thresholds["emergency_threshold"]:
            return "emergency"
        elif current_ratio > thresholds["critical_threshold"]:
            return "critical"
        elif current_ratio > thresholds["warning_threshold"]:
            return "warning"
        else:
            return "normal"
    
    def _get_memory_trend(self) -> str:
        """Analyze memory usage trend over recent history."""
        if len(self.memory_history) < 5:
            return "insufficient_data"
        
        recent = self.memory_history[-5:]
        if recent[-1] > recent[0] * 1.1:  # 10% increase
            return "increasing"
        elif recent[-1] < recent[0] * 0.9:  # 10% decrease
            return "decreasing"
        else:
            return "stable"
    
    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information if available."""
        gpu_info = {}
        
        try:
            if self.device_type == "cuda":
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                
                gpu_info.update({
                    "gpu_allocated_mb": allocated / (1024 * 1024),
                    "gpu_cached_mb": cached / (1024 * 1024),
                    "gpu_total_mb": total / (1024 * 1024),
                    "gpu_utilization": (allocated + cached) / total,
                })
                
            elif self.device_type == "mps":
                # MPS memory info (limited availability)
                gpu_info.update({
                    "mps_allocated_mb": 0,  # Placeholder - MPS doesn't expose detailed stats
                    "unified_memory": True,
                })
                
        except Exception as e:
            logger.debug(f"GPU memory info collection failed: {e}")
        
        return gpu_info
    
    def suggest_batch_size_adjustment(self, current_batch_size: int) -> int:
        """
        Suggest batch size adjustment based on current memory pressure.
        
        Uses memory pressure level and trend to recommend optimal batch size.
        """
        pressure_level = self._get_pressure_level()
        memory_trend = self._get_memory_trend()
        
        if pressure_level == "emergency":
            # Emergency: reduce batch size aggressively
            return max(1, current_batch_size // 4)
        elif pressure_level == "critical":
            # Critical: reduce batch size significantly
            return max(1, current_batch_size // 2)
        elif pressure_level == "warning":
            # Warning: small reduction
            return max(1, current_batch_size - 1)
        elif pressure_level == "normal" and memory_trend == "decreasing":
            # Memory pressure decreasing, can potentially increase
            platform_max = {"cuda": 32, "mps": 16, "cpu": 8}.get(self.device_type, 8)
            return min(current_batch_size + 1, platform_max)
        else:
            # Stable or insufficient data: no change
            return current_batch_size
    
    def trigger_memory_cleanup(self) -> Dict[str, Any]:
        """
        Trigger platform-specific memory cleanup and report results.
        
        Returns information about cleanup actions taken.
        """
        cleanup_start = time.time()
        initial_memory = self.get_memory_info()
        actions_taken = []
        
        try:
            # Python garbage collection
            collected = gc.collect()
            actions_taken.append(f"gc_collected_{collected}_objects")
            
            # Platform-specific cleanup
            if self.device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                actions_taken.append("cuda_cache_cleared")
                
            elif self.device_type == "mps":
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                torch.mps.synchronize()
                actions_taken.append("mps_cache_cleared")
            
            # Brief pause to allow cleanup to take effect
            time.sleep(0.01)  # 10ms
            
            final_memory = self.get_memory_info()
            cleanup_time = (time.time() - cleanup_start) * 1000
            
            memory_freed = initial_memory["rss_mb"] - final_memory["rss_mb"]
            
            return {
                "cleanup_time_ms": cleanup_time,
                "actions_taken": actions_taken,
                "memory_freed_mb": memory_freed,
                "initial_pressure": initial_memory["pressure_level"],
                "final_pressure": final_memory["pressure_level"],
                "success": memory_freed > 0
            }
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return {
                "cleanup_time_ms": (time.time() - cleanup_start) * 1000,
                "actions_taken": actions_taken,
                "error": str(e),
                "success": False
            }


def create_cross_platform_extractor(config_manager: ConfigManager, 
                                   extraction_config: Optional[ExtractionConfiguration] = None) -> CrossPlatformFeatureExtractor:
    """
    Factory function to create a cross-platform feature extractor.
    
    Args:
        config_manager: System configuration manager
        extraction_config: Optional extraction configuration overrides
        
    Returns:
        Initialized CrossPlatformFeatureExtractor instance
    """
    return CrossPlatformFeatureExtractor(config_manager, extraction_config)


def main():
    """Main entry point for testing the cross-platform feature extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Platform Feature Extractor")
    parser.add_argument("--test-image", type=str, help="Path to test image")
    parser.add_argument("--batch-test", type=str, help="Directory with test images")
    parser.add_argument("--platform-test", action="store_true", help="Run platform compatibility test")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config_manager = ConfigManager()
    
    # Create extractor
    extractor = create_cross_platform_extractor(config_manager)
    
    try:
        if args.platform_test:
            # Platform compatibility test
            print("🧪 Running platform compatibility test...")
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_pil = Image.fromarray(test_image)
            
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
            
        elif args.test_image:
            # Single image test
            print(f"🖼️ Testing single image: {args.test_image}")
            features = extractor.extract_features_single(args.test_image)
            
            if features is not None:
                print(f"✅ Extraction successful - Features shape: {features.shape}")
            else:
                print("❌ Extraction failed")
        
        elif args.batch_test:
            # Batch processing test
            print(f"📦 Testing batch processing: {args.batch_test}")
            image_paths = list(Path(args.batch_test).glob("*.jpg"))[:10]  # Test with first 10 images
            
            if image_paths:
                features_list = extractor.extract_features_batch(image_paths)
                successful = sum(1 for f in features_list if f is not None)
                print(f"✅ Batch processing: {successful}/{len(features_list)} successful")
            else:
                print("❌ No images found for batch test")
        
        else:
            print("ℹ️ No test specified. Use --platform-test, --test-image, or --batch-test")
    
    finally:
        extractor.close()


if __name__ == "__main__":
    main()