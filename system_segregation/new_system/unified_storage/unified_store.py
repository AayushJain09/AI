"""
Unified Storage Interface

Single entry point for all AI recognition system operations with automatic 
platform optimization and comprehensive error handling.

OVERVIEW:
This module provides a high-level unified interface that orchestrates all
storage, search, and feature extraction operations. It automatically optimizes
performance based on detected platform capabilities while providing a simple,
consistent API for all system components.

ARCHITECTURE:
- CrossPlatformFeatureExtractor: Optimized feature extraction (CLIP + DINOv2)
- SQLiteStore: Vector storage and metadata management  
- AnalyticsStore: DuckDB-based analytics and reporting
- HybridIndexer: High-performance SQLite + FAISS hybrid indexing (43x faster)
- ConfigManager: Platform-specific optimization

DESIGN PHILOSOPHY:
1. **Platform Transparency**: Automatically detects and optimizes for hardware
2. **Single Interface**: One class handles all storage operations
3. **Error Resilience**: Graceful degradation and comprehensive error handling
4. **Performance Focus**: Automatic optimization for recognition speed
5. **Memory Conscious**: Intelligent memory management and cleanup
6. **Monitoring Ready**: Built-in performance metrics and health monitoring

The unified store abstracts away platform complexity while maximizing
performance for the critical recognition pipeline.
"""

import os
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import numpy as np
from PIL import Image

# Import unified storage components
from .config_manager import ConfigManager, UnifiedConfig
from .platform_detector import PlatformDetector, PlatformConfig
from .cross_platform_extractor import (
    CrossPlatformFeatureExtractor,
    ExtractionConfiguration,
    ExtractionStatistics,
    create_cross_platform_extractor
)
from .preprocessing.hybrid_db_indexer import (
    HybridDatabaseIndexer,
    EnhancedIndexConfig,
    create_hybrid_database_indexer
)

# Configure logging for unified store
logger = logging.getLogger(__name__)


@dataclass
class StorageStatistics:
    """
    Comprehensive storage performance statistics.
    
    Tracks all aspects of unified storage performance for monitoring
    and optimization purposes.
    """
    # Operation counts
    total_images_stored: int = 0
    total_searches_performed: int = 0
    total_features_extracted: int = 0
    
    # Performance metrics
    avg_storage_time_ms: float = 0.0
    avg_search_time_ms: float = 0.0
    avg_extraction_time_ms: float = 0.0
    
    # Resource utilization
    peak_memory_usage_mb: float = 0.0
    current_cache_size_mb: float = 0.0
    database_size_mb: float = 0.0
    
    # Platform information
    platform_type: str = "unknown"
    device_acceleration: str = "cpu"
    optimization_level: str = "standard"
    
    # Health metrics
    error_count: int = 0
    success_rate: float = 1.0
    uptime_seconds: float = 0.0


@dataclass 
class SearchResult:
    """
    Search result with similarity score and metadata.
    
    Represents a single search result from the unified storage system
    with all necessary information for recognition decisions.
    """
    image_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    feature_vector: Optional[np.ndarray] = None
    extraction_time_ms: Optional[float] = None


class UnifiedStore:
    """
    Unified storage interface for AI recognition system.
    
    Provides a single, high-level interface for all storage operations
    with automatic platform optimization and comprehensive error handling.
    
    Key Features:
    - Automatic platform detection and optimization
    - Cross-platform feature extraction (CLIP + DINOv2)
    - Vector similarity search with hybrid SQLite + FAISS indexer
    - SQLite-based metadata and vector storage
    - DuckDB analytics and reporting
    - Comprehensive error handling and logging
    - Performance monitoring and statistics
    - Memory management and cleanup
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 config_path: Optional[str] = None,
                 enable_analytics: bool = True,
                 extraction_config: Optional[ExtractionConfiguration] = None):
        """
        Initialize unified storage system.
        
        Args:
            data_dir: Base directory for all data storage
            config_path: Optional path to user configuration file
            enable_analytics: Whether to enable DuckDB analytics
            extraction_config: Optional feature extraction configuration
        """
        # Initialize core components
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_analytics = enable_analytics
        self._initialized = False
        self._lock = threading.RLock()  # Thread-safe operations
        self._start_time = time.time()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize configuration management
        logger.info("🚀 Initializing Unified Storage System...")
        self.config_manager = ConfigManager(config_path, str(self.data_dir))
        self.config = self.config_manager.get_config()
        
        # Log platform information
        platform = self.config.platform
        logger.info(f"📊 Platform: {platform.platform_type}")
        logger.info(f"⚡ Device: {platform.device_type}")
        logger.info(f"🧠 Memory: {platform.memory_gb:.1f}GB")
        logger.info(f"🔧 Optimization: {platform.faiss_mode} mode for hybrid indexer")
        
        # Initialize statistics tracking
        self.statistics = StorageStatistics(
            platform_type=platform.platform_type,
            device_acceleration=platform.device_type,
            optimization_level=platform.faiss_mode
        )
        
        # Initialize core components
        self._initialize_feature_extractor(extraction_config)
        self._initialize_storage_backends()
        self._initialize_search_engine()
        
        # Mark as initialized
        self._initialized = True
        logger.info("✅ Unified Storage System initialized successfully")
    
    def _setup_logging(self):
        """
        Setup structured logging for unified storage operations.
        
        Creates separate log files for different operation types to enable
        targeted debugging and monitoring.
        """
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create formatters for different log types
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Setup unified storage logger
        self.logger = logging.getLogger(f"{__name__}.UnifiedStore")
        
        # Add file handler for unified storage operations
        storage_handler = logging.FileHandler(log_dir / "unified_storage.log")
        storage_handler.setFormatter(detailed_formatter)
        storage_handler.setLevel(logging.INFO)
        self.logger.addHandler(storage_handler)
        
        self.logger.info("Logging initialized for unified storage")
    
    def _initialize_feature_extractor(self, extraction_config: Optional[ExtractionConfiguration]):
        """
        Initialize cross-platform feature extractor.
        
        Sets up the feature extraction system with platform-specific optimizations
        for maximum performance on the detected hardware.
        """
        logger.info("🔧 Initializing cross-platform feature extractor...")
        
        try:
            # Use provided config or create optimal config based on platform
            if extraction_config is None:
                # Create optimal extraction configuration based on platform
                platform = self.config.platform
                extraction_config = ExtractionConfiguration(
                    batch_size=platform.optimal_batch_size,
                    max_memory_usage_gb=min(platform.memory_gb * 0.6, 8.0),
                    compile_models=True,  # Enable compilation for performance
                    use_mixed_precision=platform.device_type in ['cuda', 'mps'],
                    enable_memory_mapping=True,
                    cache_size_mb=min(platform.cache_size_mb, 512.0)
                )
                logger.info(f"📈 Optimal extraction config: batch_size={extraction_config.batch_size}, "
                           f"memory_limit={extraction_config.max_memory_usage_gb:.1f}GB")
            
            # Initialize feature extractor
            self.feature_extractor = CrossPlatformFeatureExtractor(
                self.config_manager,
                extraction_config
            )
            
            logger.info("✅ Feature extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize feature extractor: {e}")
            raise RuntimeError(f"Feature extractor initialization failed: {e}")
    
    def _initialize_storage_backends(self):
        """
        Initialize SQLite and DuckDB storage backends.
        
        Sets up the database connections with platform-optimized settings
        for maximum I/O performance.
        """
        logger.info("💾 Initializing storage backends...")
        
        try:
            # Initialize SQLite for vector storage and metadata
            self._initialize_sqlite_storage()
            
            # Initialize DuckDB for analytics (if enabled)
            if self.enable_analytics:
                self._initialize_analytics_storage()
            
            logger.info("✅ Storage backends initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize storage backends: {e}")
            raise RuntimeError(f"Storage backend initialization failed: {e}")
    
    def _initialize_sqlite_storage(self):
        """
        Initialize SQLite storage with platform optimizations.
        
        SQLite configuration is critical for vector retrieval performance:
        - Cache size based on available memory
        - WAL mode for concurrent access
        - Memory mapping for large datasets
        - Pragma optimizations for vector operations
        """
        db_config = self.config.database
        self.sqlite_path = db_config.sqlite_path
        
        # Create database directory if needed
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite connection with optimizations
        self.sqlite_connection = sqlite3.connect(
            self.sqlite_path,
            timeout=db_config.connection_timeout,
            check_same_thread=False  # Allow multi-thread access
        )
        
        # Apply platform-optimized SQLite pragmas
        cursor = self.sqlite_connection.cursor()
        
        # Memory and performance optimizations based on platform detection
        optimizations = [
            f"PRAGMA cache_size = {db_config.sqlite_cache_size}",  # Negative = KB
            f"PRAGMA mmap_size = {db_config.sqlite_memory_map_size}",  # Memory mapping
            f"PRAGMA journal_mode = {'WAL' if db_config.sqlite_wal_mode else 'DELETE'}",
            f"PRAGMA synchronous = {db_config.sqlite_synchronous}",
            "PRAGMA temp_store = MEMORY",  # Store temp tables in memory
            "PRAGMA page_size = 4096",  # Optimal page size for vectors
            "PRAGMA foreign_keys = ON",  # Enable foreign key constraints
        ]
        
        for pragma in optimizations:
            cursor.execute(pragma)
            logger.debug(f"Applied SQLite optimization: {pragma}")
        
        # Create tables if they don't exist
        self._create_sqlite_tables(cursor)
        
        self.sqlite_connection.commit()
        logger.info(f"📊 SQLite initialized: {self.sqlite_path}")
        logger.info(f"🔧 Cache: {abs(db_config.sqlite_cache_size)}KB, "
                   f"Memory mapping: {db_config.sqlite_memory_map_size // 1024 // 1024}MB")
    
    def _create_sqlite_tables(self, cursor):
        """
        Create SQLite tables for vector storage and metadata.
        
        Table design optimized for vector similarity search:
        - Primary vector storage with BLOB for efficient storage
        - Metadata table with indexes for fast lookup
        - Statistics table for performance monitoring
        """
        # Vector storage table - stores 1536D feature vectors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                clip_features BLOB NOT NULL,
                dinov2_features BLOB NOT NULL,
                combined_features BLOB NOT NULL,
                feature_norm REAL NOT NULL,
                extraction_time_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Metadata table - stores image metadata and recognition results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                original_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                item_category TEXT,
                confidence_score REAL,
                recognition_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES vectors (image_id)
            )
        """)
        
        # Performance statistics table - tracks system performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                execution_time_ms REAL NOT NULL,
                memory_usage_mb REAL,
                batch_size INTEGER,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                platform_type TEXT,
                device_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for optimal query performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_vectors_image_id ON vectors (image_id)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_image_id ON metadata (image_id)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_category ON metadata (item_category)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_status ON metadata (recognition_status)",
            "CREATE INDEX IF NOT EXISTS idx_stats_operation ON performance_stats (operation_type)",
            "CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON performance_stats (timestamp)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
            logger.debug(f"Created index: {index_sql.split()[-1]}")
    
    def _initialize_analytics_storage(self):
        """
        Initialize DuckDB analytics storage.
        
        DuckDB is used for complex analytics queries and reporting
        that would be inefficient in SQLite.
        """
        try:
            import duckdb
            
            db_config = self.config.database
            self.duckdb_path = db_config.duckdb_path
            
            # Create DuckDB connection with optimizations
            self.duckdb_connection = duckdb.connect(
                self.duckdb_path,
                config={
                    'memory_limit': db_config.duckdb_memory_limit,
                    'threads': db_config.duckdb_threads,
                    'enable_progress_bar': False,
                    'enable_object_cache': True
                }
            )
            
            # Create analytics tables
            self._create_duckdb_tables()
            
            logger.info(f"📈 DuckDB analytics initialized: {self.duckdb_path}")
            logger.info(f"🔧 Memory limit: {db_config.duckdb_memory_limit}, "
                       f"Threads: {db_config.duckdb_threads}")
            
        except ImportError:
            logger.warning("⚠️ DuckDB not available - analytics disabled")
            self.duckdb_connection = None
            self.enable_analytics = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize DuckDB: {e}")
            self.duckdb_connection = None
            self.enable_analytics = False
    
    def _create_duckdb_tables(self):
        """Create DuckDB tables for analytics and reporting."""
        if self.duckdb_connection is None:
            return
        
        # Recognition analytics table
        self.duckdb_connection.execute("""
            CREATE TABLE IF NOT EXISTS recognition_analytics (
                id INTEGER PRIMARY KEY,
                image_id VARCHAR,
                recognition_time_ms REAL,
                similarity_score REAL,
                item_category VARCHAR,
                confidence_level VARCHAR,
                platform_type VARCHAR,
                device_type VARCHAR,
                timestamp TIMESTAMP
            )
        """)
        
        # Performance metrics table
        self.duckdb_connection.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                metric_name VARCHAR,
                metric_value REAL,
                metric_unit VARCHAR,
                platform_type VARCHAR,
                timestamp TIMESTAMP
            )
        """)
    
    def _initialize_search_engine(self):
        """
        Initialize hybrid SQLite + FAISS search engine with platform optimizations.
        
        Hybrid indexer configuration for state-of-the-art performance:
        - 43x faster than previous methods with intelligent FAISS method selection
        - GPU acceleration with CUDA/MPS support
        - SQLite database integration for metadata and persistence
        - Automatic optimization based on dataset size
        """
        logger.info("🔍 Initializing hybrid SQLite + FAISS search engine...")
        
        try:
            search_config = self.config.search
            platform = self.config.platform
            
            # 1536 dimensions (768 CLIP + 768 DINOv2) 
            self.vector_dimension = 1536
            
            # Initialize hybrid indexer with correct parameters
            self.hybrid_indexer = create_hybrid_database_indexer(
                database_path=str(self.data_dir / "recognition.db"),
                precision_mode="balanced",
                use_gpu=search_config.gpu_enabled and platform.device_type in ['cuda', 'mps'],
                max_memory_gb=platform.memory_gb * 0.3  # Use 30% of system memory
            )
            
            # Track index state
            self.index_size = 0
            self.image_id_mapping = {}  # Maps index position to image_id
            
            logger.info(f"✅ Hybrid search engine initialized ({self.vector_dimension}D vectors)")
            logger.info(f"📁 Database: {str(self.data_dir / 'recognition.db')}")
            logger.info(f"🚀 GPU acceleration: {search_config.gpu_enabled and platform.device_type in ['cuda', 'mps']}")
            
            # Track index state
            self.index_size = 0
            self.image_id_mapping = {}  # Maps index position to image_id
            
        except ImportError as e:
            logger.error(f"❌ Hybrid indexer not available - search functionality disabled: {e}")
            self.hybrid_indexer = None
            raise RuntimeError("Hybrid indexer is required for search functionality")
        except Exception as e:
            logger.error(f"❌ Failed to initialize hybrid search engine: {e}")
            raise RuntimeError(f"Hybrid indexer initialization failed: {e}")
    
    @contextmanager
    def _database_transaction(self):
        """
        Context manager for database transactions with error handling.
        
        Ensures proper transaction management and cleanup even if errors occur.
        """
        if not self._initialized:
            raise RuntimeError("UnifiedStore not initialized")
        
        cursor = None
        try:
            cursor = self.sqlite_connection.cursor()
            yield cursor
            self.sqlite_connection.commit()
        except Exception as e:
            if cursor:
                self.sqlite_connection.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def store_image(self, 
                   image_path: str, 
                   image_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store image with feature extraction and indexing.
        
        This is the primary method for adding new images to the system.
        It performs feature extraction, stores vectors, and updates search index.
        
        Args:
            image_path: Path to the image file
            image_id: Optional unique identifier (auto-generated if not provided)
            metadata: Optional metadata dictionary
            
        Returns:
            String image_id for the stored image
            
        Raises:
            RuntimeError: If storage operation fails
        """
        if not self._initialized:
            raise RuntimeError("UnifiedStore not initialized")
        
        start_time = time.time()
        
        with self._lock:
            try:
                # Generate image_id if not provided
                if image_id is None:
                    image_id = f"img_{int(time.time() * 1000)}_{len(self.image_id_mapping)}"
                
                # Validate image exists
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                logger.info(f"📥 Storing image: {image_id} from {image_path}")
                
                # Extract features using cross-platform extractor
                features = self._extract_image_features(image_path)
                if features is None:
                    raise RuntimeError("Feature extraction failed")
                
                # Store in database and search index
                self._store_vectors_and_metadata(image_id, image_path, features, metadata)
                self._update_search_index(image_id, features)
                
                # Update statistics
                storage_time = (time.time() - start_time) * 1000
                self._update_statistics('store_image', storage_time, True)
                
                logger.info(f"✅ Image stored successfully: {image_id} ({storage_time:.1f}ms)")
                return image_id
                
            except Exception as e:
                storage_time = (time.time() - start_time) * 1000
                self._update_statistics('store_image', storage_time, False, str(e))
                logger.error(f"❌ Failed to store image {image_id}: {e}")
                raise RuntimeError(f"Image storage failed: {e}")
    
    def _extract_image_features(self, image_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract features from image using cross-platform extractor.
        
        Uses the optimized CLIP + DINOv2 pipeline for 1536-dimensional features.
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Extract features using cross-platform extractor
            features = self.feature_extractor.extract_features_single(image)
            
            if features is None:
                logger.error(f"Feature extraction returned None for {image_path}")
                return None
            
            # Split combined features back into CLIP and DINOv2 components
            # The cross-platform extractor returns 1536D combined features
            clip_features = features[:768]  # First 768 dimensions are CLIP
            dinov2_features = features[768:]  # Last 768 dimensions are DINOv2
            
            return {
                'clip': clip_features,
                'dinov2': dinov2_features,
                'combined': features
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None
    
    def _store_vectors_and_metadata(self, 
                                   image_id: str,
                                   image_path: str, 
                                   features: Dict[str, np.ndarray],
                                   metadata: Optional[Dict[str, Any]]):
        """
        Store vectors and metadata in SQLite database.
        
        Efficiently stores feature vectors as BLOBs and metadata in separate table.
        """
        with self._database_transaction() as cursor:
            # Get image file information
            file_stats = os.stat(image_path)
            
            # Try to get image dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                width, height = None, None
            
            # Serialize feature vectors for storage
            clip_blob = features['clip'].astype(np.float32).tobytes()
            dinov2_blob = features['dinov2'].astype(np.float32).tobytes()
            combined_blob = features['combined'].astype(np.float32).tobytes()
            
            # Calculate feature norm for normalization validation
            feature_norm = np.linalg.norm(features['combined'])
            
            # Store vectors
            cursor.execute("""
                INSERT OR REPLACE INTO vectors 
                (image_id, clip_features, dinov2_features, combined_features, 
                 feature_norm, extraction_time_ms) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                clip_blob,
                dinov2_blob, 
                combined_blob,
                float(feature_norm),
                None  # Extraction time tracked separately
            ))
            
            # Store metadata
            cursor.execute("""
                INSERT OR REPLACE INTO metadata 
                (image_id, original_path, file_size_bytes, image_width, image_height,
                 item_category, confidence_score, recognition_status) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                image_path,
                file_stats.st_size,
                width,
                height,
                metadata.get('category') if metadata else None,
                metadata.get('confidence') if metadata else None,
                'stored'
            ))
            
            logger.debug(f"Stored vectors and metadata for {image_id}")
    
    def _update_search_index(self, image_id: str, features: Dict[str, np.ndarray]):
        """
        Update hybrid search index with new feature vector.
        
        Adds the combined 1536D feature vector to the hybrid SQLite + FAISS index
        and maintains the ID mapping.
        """
        if self.hybrid_indexer is None:
            logger.warning("Hybrid indexer not available - skipping index update")
            return
        
        try:
            # Use combined features for search (1536 dimensions)
            vector = features['combined'].astype(np.float32).reshape(1, -1)
            
            # Add to hybrid index incrementally
            result = self.hybrid_indexer.add_vectors_incremental(
                vectors=vector,
                item_ids=[image_id],
                metadata_list=[{'source': 'unified_store', 'extraction_method': 'CLIP+DINOv2'}]
            )
            
            if result['success']:
                # Update ID mapping
                self.image_id_mapping[self.index_size] = image_id
                self.index_size += 1
                
                logger.debug(f"Updated hybrid index: {image_id} at position {self.index_size - 1}")
            else:
                logger.error(f"Failed to add vector to hybrid index: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Failed to update hybrid index for {image_id}: {e}")
            # Don't raise here - storage can continue without search indexing
    
    def search_similar(self, 
                      query_image_path: str,
                      top_k: int = 10,
                      similarity_threshold: float = 0.0) -> List[SearchResult]:
        """
        Search for similar images using vector similarity.
        
        Performs feature extraction on query image and searches for
        most similar vectors in the database using hybrid SQLite + FAISS indexer.
        
        Args:
            query_image_path: Path to query image
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        if not self._initialized:
            raise RuntimeError("UnifiedStore not initialized")
        
        if self.hybrid_indexer is None:
            raise RuntimeError("Hybrid search index not available")
        
        start_time = time.time()
        
        with self._lock:
            try:
                logger.info(f"🔍 Searching for similar images to: {query_image_path}")
                
                # Extract features from query image
                query_features = self._extract_image_features(query_image_path)
                if query_features is None:
                    raise RuntimeError("Failed to extract features from query image")
                
                # Search in hybrid index
                query_vector = query_features['combined'].astype(np.float32)
                
                # Perform similarity search using hybrid indexer
                search_result = self.hybrid_indexer.search(
                    query_vector=query_vector,
                    k=min(top_k, self.index_size),
                    return_metadata=True
                )
                
                if not search_result['success']:
                    raise RuntimeError(f"Hybrid search failed: {search_result.get('error', 'Unknown error')}")
                
                # Build search results from hybrid indexer response
                results = []
                for result_item in search_result['results']:
                    similarity = result_item['similarity']
                    
                    if similarity < similarity_threshold:
                        continue
                    
                    image_id = result_item['item_id']
                    
                    # Get metadata from database
                    metadata = self._get_image_metadata(image_id)
                    
                    results.append(SearchResult(
                        image_id=image_id,
                        similarity_score=float(similarity),
                        metadata=metadata
                    ))
                
                # Sort by similarity (highest first)
                results.sort(key=lambda x: x.similarity_score, reverse=True)
                
                # Update statistics
                search_time = (time.time() - start_time) * 1000
                self._update_statistics('search_similar', search_time, True)
                
                logger.info(f"✅ Search completed: {len(results)} results in {search_time:.1f}ms")
                return results
                
            except Exception as e:
                search_time = (time.time() - start_time) * 1000
                self._update_statistics('search_similar', search_time, False, str(e))
                logger.error(f"❌ Search failed: {e}")
                raise RuntimeError(f"Search operation failed: {e}")
    
    def _get_image_metadata(self, image_id: str) -> Dict[str, Any]:
        """Get metadata for image from database."""
        try:
            cursor = self.sqlite_connection.cursor()
            cursor.execute("""
                SELECT original_path, file_size_bytes, image_width, image_height,
                       item_category, confidence_score, recognition_status,
                       created_at, updated_at
                FROM metadata 
                WHERE image_id = ?
            """, (image_id,))
            
            row = cursor.fetchone()
            if row is None:
                return {}
            
            return {
                'original_path': row[0],
                'file_size_bytes': row[1],
                'image_width': row[2],
                'image_height': row[3],
                'item_category': row[4],
                'confidence_score': row[5],
                'recognition_status': row[6],
                'created_at': row[7],
                'updated_at': row[8]
            }
            
        except Exception as e:
            logger.error(f"Failed to get metadata for {image_id}: {e}")
            return {}
    
    def batch_store_images(self, 
                          image_paths: List[str],
                          metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Store multiple images in batch for optimal performance.
        
        Uses the cross-platform extractor's batch processing capabilities
        for maximum throughput.
        
        Args:
            image_paths: List of image file paths
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of image_ids for stored images
        """
        if not self._initialized:
            raise RuntimeError("UnifiedStore not initialized")
        
        if not image_paths:
            return []
        
        start_time = time.time()
        
        with self._lock:
            try:
                logger.info(f"📦 Batch storing {len(image_paths)} images...")
                
                # Validate all images exist
                for path in image_paths:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Image not found: {path}")
                
                # Load images for batch processing
                images = []
                for path in image_paths:
                    try:
                        image = Image.open(path).convert('RGB')
                        images.append(image)
                    except Exception as e:
                        logger.error(f"Failed to load image {path}: {e}")
                        images.append(None)
                
                # Batch feature extraction
                features_list = self.feature_extractor.extract_features_batch(images)
                
                # Store each image with its features
                stored_ids = []
                for i, (path, features) in enumerate(zip(image_paths, features_list)):
                    if features is None:
                        logger.warning(f"Skipping {path} - feature extraction failed")
                        continue
                    
                    try:
                        # Generate image_id
                        image_id = f"batch_{int(time.time() * 1000)}_{i}"
                        
                        # Convert 1536D features to format expected by storage
                        feature_dict = {
                            'clip': features[:768],
                            'dinov2': features[768:],
                            'combined': features
                        }
                        
                        # Get metadata for this image
                        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
                        
                        # Store vectors and metadata
                        self._store_vectors_and_metadata(image_id, path, feature_dict, metadata)
                        self._update_search_index(image_id, feature_dict)
                        
                        stored_ids.append(image_id)
                        
                    except Exception as e:
                        logger.error(f"Failed to store {path}: {e}")
                        continue
                
                # Update statistics
                batch_time = (time.time() - start_time) * 1000
                self._update_statistics('batch_store', batch_time, True)
                
                logger.info(f"✅ Batch storage completed: {len(stored_ids)}/{len(image_paths)} "
                           f"images stored in {batch_time:.1f}ms")
                
                return stored_ids
                
            except Exception as e:
                batch_time = (time.time() - start_time) * 1000
                self._update_statistics('batch_store', batch_time, False, str(e))
                logger.error(f"❌ Batch storage failed: {e}")
                raise RuntimeError(f"Batch storage operation failed: {e}")
    
    def get_statistics(self) -> StorageStatistics:
        """
        Get comprehensive storage system statistics.
        
        Returns current performance metrics, resource utilization,
        and system health information.
        """
        if not self._initialized:
            raise RuntimeError("UnifiedStore not initialized")
        
        try:
            # Update database size
            if os.path.exists(self.sqlite_path):
                self.statistics.database_size_mb = os.path.getsize(self.sqlite_path) / (1024 * 1024)
            
            # Update uptime
            self.statistics.uptime_seconds = time.time() - self._start_time
            
            # Get feature extractor statistics
            if hasattr(self.feature_extractor, 'get_performance_statistics'):
                extractor_stats = self.feature_extractor.get_performance_statistics()
                self.statistics.avg_extraction_time_ms = extractor_stats.avg_extraction_time_ms
                self.statistics.total_features_extracted = extractor_stats.total_images_processed
                self.statistics.peak_memory_usage_mb = extractor_stats.peak_memory_usage_mb
            
            # Calculate success rate
            total_operations = (self.statistics.total_images_stored + 
                              self.statistics.total_searches_performed)
            if total_operations > 0:
                self.statistics.success_rate = 1.0 - (self.statistics.error_count / total_operations)
            
            return self.statistics
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return self.statistics
    
    def _update_statistics(self, 
                          operation: str, 
                          execution_time_ms: float, 
                          success: bool,
                          error_message: Optional[str] = None):
        """Update internal statistics tracking."""
        try:
            # Update counters
            if operation == 'store_image':
                if success:
                    self.statistics.total_images_stored += 1
                    # Update average storage time
                    current_avg = self.statistics.avg_storage_time_ms
                    total_ops = self.statistics.total_images_stored
                    self.statistics.avg_storage_time_ms = (
                        (current_avg * (total_ops - 1) + execution_time_ms) / total_ops
                    )
            elif operation == 'search_similar':
                if success:
                    self.statistics.total_searches_performed += 1
                    # Update average search time
                    current_avg = self.statistics.avg_search_time_ms
                    total_ops = self.statistics.total_searches_performed
                    self.statistics.avg_search_time_ms = (
                        (current_avg * (total_ops - 1) + execution_time_ms) / total_ops
                    )
            elif operation == 'batch_store':
                if success:
                    # Batch operations count as multiple store operations
                    pass  # Individual images are tracked separately
            
            if not success:
                self.statistics.error_count += 1
            
            # Store performance stats in database
            self._store_performance_stats(operation, execution_time_ms, success, error_message)
            
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
    
    def _store_performance_stats(self, 
                                operation: str,
                                execution_time_ms: float, 
                                success: bool,
                                error_message: Optional[str]):
        """Store performance statistics in database."""
        try:
            with self._database_transaction() as cursor:
                cursor.execute("""
                    INSERT INTO performance_stats 
                    (operation_type, execution_time_ms, memory_usage_mb, success, 
                     error_message, platform_type, device_type) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    operation,
                    execution_time_ms,
                    self.statistics.peak_memory_usage_mb,
                    success,
                    error_message,
                    self.statistics.platform_type,
                    self.statistics.device_acceleration
                ))
        except Exception as e:
            logger.error(f"Failed to store performance stats: {e}")
    
    def optimize_index(self) -> Dict[str, Any]:
        """
        Optimize hybrid search index for better performance.
        
        Rebuilds hybrid SQLite + FAISS index from database with optimal parameters.
        Automatically selects best FAISS method based on dataset size.
        """
        if not self._initialized:
            raise RuntimeError("UnifiedStore not initialized")
        
        if self.hybrid_indexer is None or self.index_size == 0:
            return {"status": "no_optimization_needed", "reason": "Empty index"}
        
        start_time = time.time()
        
        with self._lock:
            try:
                logger.info(f"🔧 Optimizing hybrid search index ({self.index_size} vectors)...")
                
                # Rebuild hybrid index from database
                # Hybrid indexer intelligently selects optimal FAISS method
                rebuild_result = self.hybrid_indexer.rebuild_index_from_database()
                
                optimization_time = (time.time() - start_time) * 1000
                
                if rebuild_result['success']:
                    result = {
                        "status": "optimized",
                        "index_size": rebuild_result['vector_count'],
                        "optimization_time_ms": optimization_time,
                        "index_type": rebuild_result.get('index_type', 'Hybrid-FAISS'),
                        "rebuild_time": rebuild_result.get('build_time', 0.0)
                    }
                    
                    logger.info(f"✅ Hybrid index optimization completed in {optimization_time:.1f}ms")
                    return result
                else:
                    return {
                        "status": "failed", 
                        "error": rebuild_result.get('error', 'Unknown optimization error')
                    }
                
            except Exception as e:
                logger.error(f"❌ Hybrid index optimization failed: {e}")
                return {"status": "failed", "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Validates all components and returns detailed health status.
        """
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "components": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check database connectivity
            try:
                cursor = self.sqlite_connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM vectors")
                vector_count = cursor.fetchone()[0]
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "vector_count": vector_count,
                    "database_size_mb": self.statistics.database_size_mb
                }
            except Exception as e:
                health_status["components"]["database"] = {"status": "error", "error": str(e)}
                health_status["errors"].append(f"Database error: {e}")
            
            # Check feature extractor
            try:
                extractor_stats = self.feature_extractor.get_performance_statistics()
                health_status["components"]["feature_extractor"] = {
                    "status": "healthy",
                    "platform": extractor_stats.platform_type,
                    "device": extractor_stats.device_used,
                    "total_processed": extractor_stats.total_images_processed
                }
            except Exception as e:
                health_status["components"]["feature_extractor"] = {"status": "error", "error": str(e)}
                health_status["errors"].append(f"Feature extractor error: {e}")
            
            # Check search index
            try:
                if self.hybrid_indexer is not None:
                    health_status["components"]["search_index"] = {
                        "status": "healthy",
                        "index_size": self.index_size,
                        "dimension": self.vector_dimension,
                        "index_type": "Hybrid-FAISS"
                    }
                else:
                    health_status["components"]["search_index"] = {"status": "unavailable"}
                    health_status["warnings"].append("Hybrid search index not available")
            except Exception as e:
                health_status["components"]["search_index"] = {"status": "error", "error": str(e)}
                health_status["errors"].append(f"Search index error: {e}")
            
            # Check analytics (if enabled)
            if self.enable_analytics and self.duckdb_connection is not None:
                try:
                    result = self.duckdb_connection.execute("SELECT 1").fetchone()
                    health_status["components"]["analytics"] = {"status": "healthy"}
                except Exception as e:
                    health_status["components"]["analytics"] = {"status": "error", "error": str(e)}
                    health_status["errors"].append(f"Analytics error: {e}")
            
            # Overall status determination
            if health_status["errors"]:
                health_status["overall_status"] = "degraded"
            elif health_status["warnings"]:
                health_status["overall_status"] = "warning"
            
            return health_status
            
        except Exception as e:
            return {
                "overall_status": "error",
                "timestamp": time.time(),
                "error": str(e)
            }
    
    def close(self):
        """
        Clean up resources and close connections.
        
        Ensures proper cleanup of all system resources including
        database connections, GPU memory, and temporary files.
        """
        if not self._initialized:
            return
        
        logger.info("🔒 Closing unified storage system...")
        
        try:
            # Close feature extractor
            if hasattr(self, 'feature_extractor'):
                self.feature_extractor.close()
                logger.info("✅ Feature extractor closed")
            
            # Close database connections
            if hasattr(self, 'sqlite_connection'):
                self.sqlite_connection.close()
                logger.info("✅ SQLite connection closed")
            
            if hasattr(self, 'duckdb_connection') and self.duckdb_connection:
                self.duckdb_connection.close()
                logger.info("✅ DuckDB connection closed")
            
            # Clear hybrid indexer from memory
            if hasattr(self, 'hybrid_indexer'):
                self.hybrid_indexer = None
                logger.info("✅ Hybrid indexer cleared")
            
            self._initialized = False
            logger.info("✅ Unified storage system closed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


# Convenience functions for easy access

def create_unified_store(data_dir: str = "data", 
                        config_path: Optional[str] = None,
                        enable_analytics: bool = True) -> UnifiedStore:
    """
    Create and initialize a unified storage system.
    
    Convenience function for quick setup with automatic platform optimization.
    """
    return UnifiedStore(
        data_dir=data_dir,
        config_path=config_path,
        enable_analytics=enable_analytics
    )


def get_optimal_extraction_config(platform_config: PlatformConfig) -> ExtractionConfiguration:
    """
    Get optimal extraction configuration for a specific platform.
    
    Convenience function for creating extraction configurations
    based on platform capabilities.
    """
    return ExtractionConfiguration(
        batch_size=platform_config.optimal_batch_size,
        max_memory_usage_gb=min(platform_config.memory_gb * 0.6, 8.0),
        compile_models=True,
        use_mixed_precision=platform_config.device_type in ['cuda', 'mps'],
        enable_memory_mapping=True,
        cache_size_mb=min(platform_config.cache_size_mb, 512.0)
    )


if __name__ == "__main__":
    # Example usage and testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Fix imports for standalone execution
    from unified_storage.config_manager import ConfigManager, UnifiedConfig
    from unified_storage.platform_detector import PlatformDetector, PlatformConfig
    from unified_storage.cross_platform_extractor import (
        CrossPlatformFeatureExtractor,
        ExtractionConfiguration,
        ExtractionStatistics,
        create_cross_platform_extractor
    )
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("🚀 Testing Unified Storage System...")
        
        # Create unified store
        store = create_unified_store("test_data")
        
        # Print system information
        stats = store.get_statistics()
        print("\n=== Unified Storage System ===")
        print(f"Platform: {stats.platform_type}")
        print(f"Device: {stats.device_acceleration}")
        print(f"Optimization: {stats.optimization_level}")
        print(f"Uptime: {stats.uptime_seconds:.1f}s")
        
        # Health check
        health = store.health_check()
        print(f"\n=== Health Check ===")
        print(f"Overall Status: {health['overall_status']}")
        for component, status in health['components'].items():
            print(f"{component}: {status['status']}")
        
        if health.get('warnings'):
            print(f"Warnings: {health['warnings']}")
        if health.get('errors'):
            print(f"Errors: {health['errors']}")
        
        # Close system
        store.close()
        print("\n✅ Test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)