"""
AI Recognition System - Unified Storage Module

Cross-platform storage architecture with automatic platform optimization,
unified interface for all storage operations, and comprehensive error handling.

Key Components:
- UnifiedStore: Single entry point for all storage operations
- CrossPlatformFeatureExtractor: Optimized CLIP + DINOv2 feature extraction
- PlatformDetector: Automatic hardware detection and optimization
- ConfigManager: Platform-specific configuration management

Usage:
    from unified_storage import create_unified_store
    
    # Create optimized storage system
    store = create_unified_store("data")
    
    # Store images with automatic feature extraction
    image_id = store.store_image("path/to/image.jpg")
    
    # Search for similar images
    results = store.search_similar("path/to/query.jpg", top_k=10)
    
    # Get system statistics
    stats = store.get_statistics()
    
    # Close when done
    store.close()
"""

from .platform_detector import PlatformDetector, PlatformConfig, detect_platform
from .config_manager import ConfigManager, UnifiedConfig, get_default_config
from .cross_platform_extractor import (
    CrossPlatformFeatureExtractor,
    ExtractionConfiguration,
    ExtractionStatistics,
    PlatformOptimizer,
    MemoryMonitor,
    create_cross_platform_extractor
)
from .unified_store import (
    UnifiedStore,
    StorageStatistics,
    SearchResult,
    create_unified_store,
    get_optimal_extraction_config
)
from .enhanced_unified_store import (
    EnhancedUnifiedStore,
    AugmentationConfig,
    ProcessingStatistics,
    create_enhanced_unified_store
)
from .enhanced_recognition_pipeline import (
    EnhancedRecognitionPipeline,
    RecognitionConfig,
    RecognitionResult,
    create_enhanced_recognition_pipeline
)
from .preprocessing.hybrid_db_indexer import (
    HybridDatabaseIndexer,
    EnhancedIndexConfig,
    SearchResult as IndexerSearchResult,
    create_hybrid_database_indexer
)
from .enhanced_unified_store_with_recognition import (
    EnhancedUnifiedStoreWithRecognition,
    EnhancedRecognitionStatistics,
    create_enhanced_unified_store_with_recognition
)

__all__ = [
    # Core unified interface
    'UnifiedStore',
    'create_unified_store',
    
    # Enhanced unified interface (with proven approach)
    'EnhancedUnifiedStore',
    'create_enhanced_unified_store',
    'AugmentationConfig',
    'ProcessingStatistics',
    
    # Complete recognition system (recommended)
    'EnhancedUnifiedStoreWithRecognition',
    'EnhancedRecognitionStatistics',
    'create_enhanced_unified_store_with_recognition',
    
    # Enhanced recognition pipeline (state-of-the-art)
    'EnhancedRecognitionPipeline',
    'create_enhanced_recognition_pipeline',
    'RecognitionConfig',
    'RecognitionResult',
    
    # Hybrid database indexer (high-performance)
    'HybridDatabaseIndexer',
    'create_hybrid_database_indexer',
    'EnhancedIndexConfig',
    'IndexerSearchResult',
    
    # Feature extraction system
    'CrossPlatformFeatureExtractor',
    'create_cross_platform_extractor',
    'ExtractionConfiguration',
    'ExtractionStatistics',
    'PlatformOptimizer',
    'MemoryMonitor',
    'get_optimal_extraction_config',
    
    # Platform detection and configuration
    'PlatformDetector',
    'PlatformConfig',
    'detect_platform',
    'ConfigManager',
    'UnifiedConfig',
    'get_default_config',
    
    # Data structures
    'StorageStatistics',
    'SearchResult',
]

__version__ = '2.0.0'