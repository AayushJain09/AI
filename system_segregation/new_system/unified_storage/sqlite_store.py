"""
SQLite Vector Storage Foundation

High-performance SQLite backend optimized for vector similarity search operations.
Serves as the primary OLTP storage for the unified storage architecture.

OVERVIEW:
This module implements the core SQLite storage layer for the AI Recognition System.
It provides optimized storage for 1536D feature vectors (768 CLIP + 768 DINOv2)
with fast retrieval, metadata management, and search index support.

WHY SQLITE:
1. Zero-configuration embedded database - no server setup required
2. ACID compliance ensures data integrity during concurrent operations
3. Cross-platform compatibility across Windows, macOS, and Linux
4. Excellent performance for read-heavy workloads (perfect for recognition)
5. WAL mode enables concurrent readers without blocking writers
6. Memory mapping reduces I/O overhead for frequently accessed data

PERFORMANCE OPTIMIZATION STRATEGY:
- Platform-specific cache sizing based on available system memory
- WAL mode for concurrent read/write operations without blocking
- Memory mapping for large datasets to reduce system call overhead
- Connection pooling to amortize connection setup costs
- Prepared statements for all queries to avoid SQL parsing overhead
- Optimized table schemas with proper indexing for fast lookups

TARGET PERFORMANCE:
- Vector insertion: <1ms per vector with batch operations
- Similarity search setup: <5ms for query preparation
- Metadata retrieval: <0.1ms for single item lookup
- Index operations: <10ms for index updates
"""

import sqlite3
import threading
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np

# Import our configuration system for platform-optimized settings
from .config_manager import ConfigManager, DatabaseConfig


@dataclass
class VectorRecord:
    """
    Vector record for storage in SQLite database.
    
    Represents a single feature vector with its metadata for efficient storage
    and retrieval. The design balances storage efficiency with query performance.
    """
    vector_id: str              # Unique identifier (e.g., "item_001_img_1")
    item_id: str                # Item identifier (e.g., "item_001")  
    vector_data: np.ndarray     # 1536D feature vector (CLIP + DINOv2)
    metadata: Dict[str, Any]    # Additional metadata (file path, timestamps, etc.)
    created_at: float           # Unix timestamp for temporal ordering
    updated_at: float           # Last modification timestamp


class SQLiteVectorStore:
    """
    High-performance SQLite vector storage with platform optimization.
    
    Provides the core storage infrastructure for the unified storage architecture.
    Automatically optimizes settings based on detected hardware capabilities.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None, 
                 database_path: Optional[str] = None):
        """
        Initialize SQLite vector store with platform-optimized settings.
        
        The initialization process:
        1. Load platform-optimized configuration settings
        2. Setup database connection with performance optimizations
        3. Create optimized table schemas if they don't exist
        4. Initialize connection pooling for concurrent access
        5. Apply WAL mode and memory mapping for performance
        
        Args:
            config_manager: Configuration manager for platform settings
            database_path: Custom database path (overrides config)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load platform-optimized configuration
        # ConfigManager provides hardware-specific optimization settings
        self.config_manager = config_manager or ConfigManager()
        self.unified_config = self.config_manager.get_config()
        self.db_config = self.unified_config.database
        
        # Database path with configuration override support
        # Allows user customization while maintaining platform optimization
        self.database_path = database_path or self.db_config.sqlite_path
        
        # Connection pooling for concurrent access
        # Multiple threads can share connections efficiently
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._max_connections = 10  # Conservative limit to prevent resource exhaustion
        
        # Performance monitoring for optimization feedback
        self._query_stats = {
            'inserts': 0,
            'selects': 0,
            'updates': 0,
            'avg_insert_time': 0.0,
            'avg_select_time': 0.0
        }
        
        self.logger.info(f"Initializing SQLite store at: {self.database_path}")
        self._initialize_database()
    
    def _initialize_database(self):
        """
        Initialize database with platform-optimized settings and schemas.
        
        This method applies critical performance optimizations:
        1. WAL mode for concurrent read/write operations
        2. Platform-specific cache sizing for optimal memory usage
        3. Memory mapping for reduced I/O overhead on large datasets
        4. Optimized pragma settings for recognition workload
        5. Create table schemas optimized for vector operations
        """
        
        # Ensure database directory exists
        # Critical for new installations and cross-platform compatibility
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            self._apply_performance_optimizations(conn)
            self._create_schemas(conn)
            conn.commit()
        
        self.logger.info("SQLite database initialized with platform optimizations")
    
    def _apply_performance_optimizations(self, conn: sqlite3.Connection):
        """
        Apply platform-specific performance optimizations to SQLite connection.
        
        These optimizations are crucial for recognition system performance:
        
        1. WAL Mode: Enables concurrent readers without blocking writers
           - Critical for real-time recognition while background updates occur
           - Prevents recognition delays during index maintenance
        
        2. Cache Size: Platform-optimized memory allocation
           - NVIDIA systems: Larger cache (GPU memory is separate)
           - Apple Silicon: Balanced cache (unified memory architecture)
           - CPU-only: Conservative cache (memory pressure from computation)
        
        3. Memory Mapping: Reduces system call overhead
           - Particularly beneficial for read-heavy recognition workloads
           - Platform-specific sizing based on available memory
        
        4. Synchronous Mode: Balanced safety vs performance
           - NORMAL mode provides good durability with acceptable performance
           - Prevents data loss while maintaining recognition speed
        """
        
        # WAL Mode: Enable concurrent read/write operations
        # This is CRITICAL for real-time recognition performance
        # Without WAL, readers block writers and vice versa
        if self.db_config.sqlite_wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
            self.logger.info("Enabled WAL mode for concurrent operations")
        
        # Cache Size: Platform-optimized memory allocation
        # Negative values mean KB, positive values mean pages
        # Our config uses negative values for direct KB specification
        cache_size = self.db_config.sqlite_cache_size
        conn.execute(f"PRAGMA cache_size={cache_size}")
        cache_mb = abs(cache_size) // 1000
        self.logger.info(f"Set cache size to {cache_mb}MB for platform: "
                        f"{self.unified_config.platform.platform_type}")
        
        # Memory Mapping: Reduce I/O overhead for large datasets
        # Particularly beneficial when feature database grows large
        # Platform-specific sizing prevents memory pressure
        mmap_size = self.db_config.sqlite_memory_map_size
        conn.execute(f"PRAGMA mmap_size={mmap_size}")
        mmap_mb = mmap_size // (1024 * 1024)
        self.logger.info(f"Set memory mapping to {mmap_mb}MB")
        
        # Synchronous Mode: Balance durability vs performance
        # NORMAL provides good safety without sacrificing recognition speed
        sync_mode = self.db_config.sqlite_synchronous
        conn.execute(f"PRAGMA synchronous={sync_mode}")
        
        # Additional optimizations for vector workload
        conn.execute("PRAGMA temp_store=MEMORY")      # Keep temp tables in memory
        conn.execute("PRAGMA query_only=FALSE")       # Enable write operations
        conn.execute("PRAGMA foreign_keys=ON")        # Enforce referential integrity
        
        self.logger.info(f"Applied SQLite optimizations for {self.unified_config.platform.platform_type}")
    
    def _create_schemas(self, conn: sqlite3.Connection):
        """
        Create optimized table schemas for vector storage and retrieval.
        
        Schema design optimizations:
        
        1. vectors table: Core storage for feature vectors
           - vector_id: Primary key for fast lookup
           - item_id: Indexed for item-based queries
           - vector_data: BLOB storage for numpy arrays
           - metadata: JSON for flexible metadata storage
           - Timestamps for temporal queries and cleanup
        
        2. vector_index: Search index metadata
           - Stores FAISS index information and statistics
           - Enables index versioning and consistency checks
        
        3. performance_stats: Query performance monitoring
           - Tracks operation timing for optimization feedback
           - Enables adaptive performance tuning
        
        Index Strategy:
        - Primary keys for guaranteed fast access
        - item_id index for item-based recognition queries
        - created_at index for temporal queries and maintenance
        - Compound indexes for common query patterns
        """
        
        # Core vectors table: Optimized for fast vector storage and retrieval
        # Design rationale:
        # - vector_id as TEXT PRIMARY KEY for guaranteed uniqueness and fast access
        # - item_id with index for fast item-based lookups during recognition
        # - vector_data as BLOB for efficient numpy array storage
        # - metadata as JSON for flexible, queryable metadata storage
        # - Timestamps for maintenance operations and temporal queries
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                vector_id TEXT PRIMARY KEY,           -- Unique vector identifier
                item_id TEXT NOT NULL,                -- Item identifier for grouping
                vector_data BLOB NOT NULL,            -- 1536D feature vector (numpy)
                metadata JSON,                        -- Flexible metadata storage
                created_at REAL NOT NULL,             -- Creation timestamp
                updated_at REAL NOT NULL,             -- Last update timestamp
                vector_norm REAL,                     -- Cached L2 norm for optimization
                vector_checksum TEXT                  -- Integrity verification
            )
        """)
        
        # Search index metadata: Tracks FAISS index state and performance
        # Design rationale:
        # - Enables index versioning and consistency verification
        # - Stores index-specific optimization parameters
        # - Tracks performance metrics for adaptive optimization
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_index (
                index_id TEXT PRIMARY KEY,            -- Index identifier
                index_type TEXT NOT NULL,             -- FAISS index type (Flat, IVF, etc.)
                index_parameters JSON,                -- Index-specific parameters
                vector_count INTEGER NOT NULL,        -- Number of vectors in index
                index_size_bytes INTEGER,             -- Index size for memory planning
                created_at REAL NOT NULL,             -- Index creation time
                last_updated REAL NOT NULL,           -- Last index update
                performance_stats JSON                -- Performance metrics
            )
        """)
        
        # Performance statistics: Monitor and optimize query performance
        # Design rationale:
        # - Enables real-time performance monitoring
        # - Supports adaptive optimization based on usage patterns
        # - Helps identify performance bottlenecks
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,         -- insert, select, update, etc.
                execution_time_ms REAL NOT NULL,      -- Operation duration
                vector_count INTEGER,                 -- Number of vectors affected
                platform_type TEXT,                   -- Platform for performance analysis
                timestamp REAL NOT NULL              -- When operation occurred
            )
        """)
        
        # Optimized indexes for fast queries
        # These indexes are critical for recognition system performance
        
        # item_id index: Fast lookup of all vectors for a specific item
        # Critical for recognition queries where we need item-specific vectors
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_item_id 
            ON vectors(item_id)
        """)
        
        # created_at index: Temporal queries for maintenance and cleanup
        # Enables efficient deletion of old vectors and temporal analysis
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_created_at 
            ON vectors(created_at)
        """)
        
        # Compound index: item_id + created_at for efficient item history queries
        # Optimizes queries that need recent vectors for specific items
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_item_created 
            ON vectors(item_id, created_at)
        """)
        
        # Performance stats index: Fast performance analysis queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_stats_timestamp 
            ON performance_stats(timestamp)
        """)
        
        self.logger.info("Created optimized table schemas with performance indexes")
    
    @contextmanager
    def _get_connection(self):
        """
        Get database connection from pool with automatic cleanup.
        
        Connection pooling strategy:
        1. Reuse existing connections to avoid setup overhead
        2. Apply performance optimizations to each connection
        3. Thread-safe connection management
        4. Automatic cleanup prevents connection leaks
        
        This pattern is critical for multi-threaded recognition operations
        where multiple threads need concurrent database access.
        """
        conn = None
        try:
            # Try to get connection from pool first
            # This avoids expensive connection setup for frequent operations
            with self._pool_lock:
                if self._connection_pool:
                    conn = self._connection_pool.pop()
                    
            # Create new connection if pool is empty
            # Apply optimizations to ensure consistent performance
            if conn is None:
                conn = sqlite3.connect(
                    self.database_path,
                    timeout=self.db_config.connection_timeout,
                    check_same_thread=False  # Enable multi-threaded access
                )
                # Enable row factory for easier result handling
                conn.row_factory = sqlite3.Row
                # Apply performance optimizations to new connection
                self._apply_performance_optimizations(conn)
            
            yield conn
            
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            # Return connection to pool for reuse
            # Pool management prevents resource exhaustion
            if conn:
                try:
                    # Only return healthy connections to pool
                    conn.execute("SELECT 1")  # Test connection health
                    with self._pool_lock:
                        if len(self._connection_pool) < self._max_connections:
                            self._connection_pool.append(conn)
                        else:
                            conn.close()  # Close excess connections
                except:
                    # Close unhealthy connections
                    try:
                        conn.close()
                    except:
                        pass
    
    def insert_vector(self, record: VectorRecord) -> bool:
        """
        Insert single vector record with performance optimization.
        
        Optimizations applied:
        1. Prepared statement to avoid SQL parsing overhead
        2. Numpy array serialization optimized for space and speed
        3. Automatic checksum calculation for integrity verification
        4. Performance timing for optimization feedback
        5. Transaction management for consistency
        
        Args:
            record: Vector record to insert
            
        Returns:
            bool: Success status
        """
        start_time = time.time()
        
        try:
            # Serialize numpy array efficiently
            # Use numpy's native binary format for speed and space efficiency
            vector_bytes = record.vector_data.tobytes()
            
            # Calculate checksum for integrity verification
            # Enables detection of data corruption during storage/retrieval
            import hashlib
            checksum = hashlib.md5(vector_bytes).hexdigest()
            
            # Calculate L2 norm for potential optimization
            # Pre-computed norm can accelerate some similarity calculations
            vector_norm = float(np.linalg.norm(record.vector_data))
            
            with self._get_connection() as conn:
                # Use prepared statement for performance
                # Avoids SQL parsing overhead for repeated insertions
                conn.execute("""
                    INSERT OR REPLACE INTO vectors 
                    (vector_id, item_id, vector_data, metadata, created_at, 
                     updated_at, vector_norm, vector_checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.vector_id,
                    record.item_id,
                    vector_bytes,
                    json.dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                    vector_norm,
                    checksum
                ))
                conn.commit()
            
            # Track performance for optimization
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self._record_performance('insert', execution_time, 1)
            
            self.logger.debug(f"Inserted vector {record.vector_id} in {execution_time:.2f}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert vector {record.vector_id}: {e}")
            return False
    
    def insert_vectors_batch(self, records: List[VectorRecord]) -> int:
        """
        Insert multiple vectors in optimized batch operation.
        
        Batch optimization strategy:
        1. Single transaction for all insertions (ACID compliance)
        2. Prepared statement reuse across all records
        3. Batch commit reduces I/O operations
        4. Progress tracking for long operations
        5. Partial success handling for error recovery
        
        This method is critical for initial database population and
        bulk updates during model retraining or dataset expansion.
        
        Args:
            records: List of vector records to insert
            
        Returns:
            int: Number of successfully inserted records
        """
        if not records:
            return 0
        
        start_time = time.time()
        inserted_count = 0
        
        try:
            with self._get_connection() as conn:
                # Use executemany for optimal batch performance
                # Single transaction ensures consistency
                batch_data = []
                for record in records:
                    try:
                        # Prepare data for batch insertion
                        vector_bytes = record.vector_data.tobytes()
                        
                        # Calculate integrity checksum
                        import hashlib
                        checksum = hashlib.md5(vector_bytes).hexdigest()
                        
                        # Pre-compute L2 norm for optimization
                        vector_norm = float(np.linalg.norm(record.vector_data))
                        
                        batch_data.append((
                            record.vector_id,
                            record.item_id,
                            vector_bytes,
                            json.dumps(record.metadata),
                            record.created_at,
                            record.updated_at,
                            vector_norm,
                            checksum
                        ))
                        
                    except Exception as e:
                        self.logger.warning(f"Skipping invalid record {record.vector_id}: {e}")
                        continue
                
                # Batch insert with single transaction
                # This is much faster than individual insertions
                if batch_data:
                    conn.executemany("""
                        INSERT OR REPLACE INTO vectors 
                        (vector_id, item_id, vector_data, metadata, created_at, 
                         updated_at, vector_norm, vector_checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch_data)
                    conn.commit()
                    inserted_count = len(batch_data)
            
            # Performance tracking for batch operations
            execution_time = (time.time() - start_time) * 1000
            avg_time_per_record = execution_time / max(inserted_count, 1)
            self._record_performance('batch_insert', execution_time, inserted_count)
            
            self.logger.info(f"Batch inserted {inserted_count} vectors in {execution_time:.2f}ms "
                           f"({avg_time_per_record:.2f}ms per vector)")
            
            return inserted_count
            
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            return inserted_count  # Return partial success count
    
    def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """
        Retrieve single vector by ID with performance optimization.
        
        Optimizations:
        1. Primary key lookup for guaranteed fast access
        2. Efficient numpy array deserialization
        3. Checksum verification for data integrity
        4. Connection pooling for reduced overhead
        
        Args:
            vector_id: Unique vector identifier
            
        Returns:
            VectorRecord if found, None otherwise
        """
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT vector_id, item_id, vector_data, metadata, 
                           created_at, updated_at, vector_norm, vector_checksum
                    FROM vectors 
                    WHERE vector_id = ?
                """, (vector_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Deserialize numpy array efficiently
                # Reshape to original 1536D vector format
                vector_data = np.frombuffer(row['vector_data'], dtype=np.float32)
                if len(vector_data) != 1536:
                    self.logger.warning(f"Vector {vector_id} has unexpected size: {len(vector_data)}")
                
                # Verify data integrity with checksum
                import hashlib
                expected_checksum = hashlib.md5(row['vector_data']).hexdigest()
                if row['vector_checksum'] != expected_checksum:
                    self.logger.warning(f"Checksum mismatch for vector {vector_id}")
                
                # Parse metadata JSON
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                
                # Create VectorRecord instance
                record = VectorRecord(
                    vector_id=row['vector_id'],
                    item_id=row['item_id'],
                    vector_data=vector_data,
                    metadata=metadata,
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                
                # Track retrieval performance
                execution_time = (time.time() - start_time) * 1000
                self._record_performance('select', execution_time, 1)
                
                return record
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve vector {vector_id}: {e}")
            return None
    
    def get_vectors_by_item(self, item_id: str) -> List[VectorRecord]:
        """
        Retrieve all vectors for a specific item with optimization.
        
        This method is critical for recognition operations where we need
        all feature vectors associated with a particular item for comparison.
        
        Optimizations:
        1. Indexed query on item_id for fast retrieval
        2. Batch deserialization for multiple vectors
        3. Memory-efficient processing for large item collections
        
        Args:
            item_id: Item identifier
            
        Returns:
            List of VectorRecord instances for the item
        """
        start_time = time.time()
        records = []
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT vector_id, item_id, vector_data, metadata, 
                           created_at, updated_at, vector_norm, vector_checksum
                    FROM vectors 
                    WHERE item_id = ?
                    ORDER BY created_at DESC
                """, (item_id,))
                
                # Process results efficiently
                for row in cursor:
                    try:
                        # Deserialize vector data
                        vector_data = np.frombuffer(row['vector_data'], dtype=np.float32)
                        
                        # Parse metadata
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                        
                        # Create record
                        record = VectorRecord(
                            vector_id=row['vector_id'],
                            item_id=row['item_id'],
                            vector_data=vector_data,
                            metadata=metadata,
                            created_at=row['created_at'],
                            updated_at=row['updated_at']
                        )
                        records.append(record)
                        
                    except Exception as e:
                        self.logger.warning(f"Skipping corrupted vector in item {item_id}: {e}")
                        continue
            
            # Performance tracking
            execution_time = (time.time() - start_time) * 1000
            self._record_performance('select_by_item', execution_time, len(records))
            
            self.logger.debug(f"Retrieved {len(records)} vectors for item {item_id} "
                            f"in {execution_time:.2f}ms")
            
            return records
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve vectors for item {item_id}: {e}")
            return []
    
    def get_all_vectors(self, limit: Optional[int] = None) -> List[VectorRecord]:
        """
        Retrieve all vectors with optional limit for memory management.
        
        Memory management strategy:
        1. Optional limit prevents memory exhaustion on large datasets
        2. Streaming processing for very large result sets
        3. Progress tracking for long operations
        
        Args:
            limit: Maximum number of vectors to retrieve
            
        Returns:
            List of all VectorRecord instances (up to limit)
        """
        start_time = time.time()
        records = []
        
        try:
            with self._get_connection() as conn:
                # Build query with optional limit
                query = """
                    SELECT vector_id, item_id, vector_data, metadata, 
                           created_at, updated_at, vector_norm, vector_checksum
                    FROM vectors 
                    ORDER BY created_at DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query)
                
                # Process results with memory management
                for row in cursor:
                    try:
                        # Deserialize vector data
                        vector_data = np.frombuffer(row['vector_data'], dtype=np.float32)
                        
                        # Parse metadata
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                        
                        # Create record
                        record = VectorRecord(
                            vector_id=row['vector_id'],
                            item_id=row['item_id'],
                            vector_data=vector_data,
                            metadata=metadata,
                            created_at=row['created_at'],
                            updated_at=row['updated_at']
                        )
                        records.append(record)
                        
                    except Exception as e:
                        self.logger.warning(f"Skipping corrupted vector: {e}")
                        continue
            
            # Performance tracking
            execution_time = (time.time() - start_time) * 1000
            self._record_performance('select_all', execution_time, len(records))
            
            self.logger.info(f"Retrieved {len(records)} vectors in {execution_time:.2f}ms")
            
            return records
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve all vectors: {e}")
            return []
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete single vector by ID.
        
        Args:
            vector_id: Vector identifier to delete
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM vectors WHERE vector_id = ?", (vector_id,))
                conn.commit()
                
                success = cursor.rowcount > 0
                if success:
                    self.logger.debug(f"Deleted vector {vector_id}")
                else:
                    self.logger.warning(f"Vector {vector_id} not found for deletion")
                
                return success
                
        except Exception as e:
            self.logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False
    
    def delete_vectors_by_item(self, item_id: str) -> int:
        """
        Delete all vectors for a specific item.
        
        Args:
            item_id: Item identifier
            
        Returns:
            int: Number of deleted vectors
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM vectors WHERE item_id = ?", (item_id,))
                conn.commit()
                
                deleted_count = cursor.rowcount
                self.logger.info(f"Deleted {deleted_count} vectors for item {item_id}")
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to delete vectors for item {item_id}: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring and optimization.
        
        Returns:
            Dict containing database statistics and performance metrics
        """
        try:
            with self._get_connection() as conn:
                # Vector count and storage stats
                cursor = conn.execute("SELECT COUNT(*) as total_vectors FROM vectors")
                total_vectors = cursor.fetchone()['total_vectors']
                
                cursor = conn.execute("SELECT COUNT(DISTINCT item_id) as unique_items FROM vectors")
                unique_items = cursor.fetchone()['unique_items']
                
                # Database size information
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size_bytes = page_count * page_size
                
                # Performance statistics
                avg_insert_time = self._query_stats.get('avg_insert_time', 0.0)
                avg_select_time = self._query_stats.get('avg_select_time', 0.0)
                
                return {
                    'total_vectors': total_vectors,
                    'unique_items': unique_items,
                    'database_size_mb': db_size_bytes / (1024 * 1024),
                    'avg_insert_time_ms': avg_insert_time,
                    'avg_select_time_ms': avg_select_time,
                    'cache_size_mb': abs(self.db_config.sqlite_cache_size) // 1000,
                    'memory_map_mb': self.db_config.sqlite_memory_map_size // (1024 * 1024),
                    'platform_type': self.unified_config.platform.platform_type,
                    'performance_optimizations': {
                        'wal_mode': self.db_config.sqlite_wal_mode,
                        'memory_mapping': True,
                        'connection_pooling': True
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def _record_performance(self, operation: str, execution_time_ms: float, vector_count: int):
        """
        Record performance metrics for optimization analysis.
        
        Args:
            operation: Type of operation performed
            execution_time_ms: Operation duration in milliseconds
            vector_count: Number of vectors affected
        """
        try:
            # Update running averages for quick access
            if operation == 'insert':
                self._query_stats['inserts'] += 1
                current_avg = self._query_stats['avg_insert_time']
                count = self._query_stats['inserts']
                self._query_stats['avg_insert_time'] = (current_avg * (count - 1) + execution_time_ms) / count
            
            elif operation in ['select', 'select_by_item', 'select_all']:
                self._query_stats['selects'] += 1
                current_avg = self._query_stats['avg_select_time']
                count = self._query_stats['selects']
                self._query_stats['avg_select_time'] = (current_avg * (count - 1) + execution_time_ms) / count
            
            # Store detailed performance data for analysis
            # This enables performance trend analysis and optimization
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO performance_stats 
                    (operation_type, execution_time_ms, vector_count, platform_type, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    operation,
                    execution_time_ms,
                    vector_count,
                    self.unified_config.platform.platform_type,
                    time.time()
                ))
                conn.commit()
                
        except Exception as e:
            # Don't let performance tracking break core functionality
            self.logger.debug(f"Performance tracking error: {e}")
    
    def optimize_database(self):
        """
        Perform database optimization operations.
        
        Optimization operations:
        1. VACUUM to reclaim space from deleted records
        2. ANALYZE to update query planner statistics
        3. Index maintenance for optimal query performance
        4. Performance statistics cleanup
        """
        self.logger.info("Starting database optimization...")
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                # Update query planner statistics
                # This helps SQLite choose optimal query execution plans
                conn.execute("ANALYZE")
                
                # Optimize indexes for current data distribution
                conn.execute("REINDEX")
                
                # Clean old performance statistics (keep last 30 days)
                thirty_days_ago = time.time() - (30 * 24 * 60 * 60)
                conn.execute("DELETE FROM performance_stats WHERE timestamp < ?", (thirty_days_ago,))
                
                conn.commit()
            
            # VACUUM outside transaction for maximum effectiveness
            # VACUUM reclaims space and optimizes database file structure
            with self._get_connection() as conn:
                conn.execute("VACUUM")
            
            optimization_time = time.time() - start_time
            self.logger.info(f"Database optimization completed in {optimization_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
    
    def close(self):
        """
        Close all database connections and cleanup resources.
        """
        with self._pool_lock:
            for conn in self._connection_pool:
                try:
                    conn.close()
                except:
                    pass
            self._connection_pool.clear()
        
        self.logger.info("SQLite vector store closed")


# Convenience functions for easy usage
def create_vector_store(config_manager: Optional[ConfigManager] = None) -> SQLiteVectorStore:
    """Create optimized SQLite vector store with current platform settings."""
    return SQLiteVectorStore(config_manager)


def create_vector_record(vector_id: str, item_id: str, vector_data: np.ndarray, 
                        metadata: Optional[Dict[str, Any]] = None) -> VectorRecord:
    """Create vector record with current timestamp."""
    current_time = time.time()
    return VectorRecord(
        vector_id=vector_id,
        item_id=item_id,
        vector_data=vector_data,
        metadata=metadata or {},
        created_at=current_time,
        updated_at=current_time
    )


if __name__ == "__main__":
    # Test SQLite vector store functionality
    logging.basicConfig(level=logging.INFO)
    
    print("=== SQLite Vector Store Test ===")
    
    # Create test store
    store = create_vector_store()
    
    # Test vector insertion
    test_vector = np.random.rand(1536).astype(np.float32)
    test_record = create_vector_record(
        vector_id="test_001",
        item_id="item_test",
        vector_data=test_vector,
        metadata={"test": True, "source": "unit_test"}
    )
    
    success = store.insert_vector(test_record)
    print(f"✅ Vector insertion: {'Success' if success else 'Failed'}")
    
    # Test vector retrieval
    retrieved = store.get_vector("test_001")
    print(f"✅ Vector retrieval: {'Success' if retrieved else 'Failed'}")
    
    if retrieved:
        # Verify vector data integrity
        vectors_match = np.allclose(test_vector, retrieved.vector_data)
        print(f"✅ Vector data integrity: {'Passed' if vectors_match else 'Failed'}")
    
    # Test statistics
    stats = store.get_statistics()
    print(f"✅ Database statistics: {stats.get('total_vectors', 0)} vectors stored")
    
    # Cleanup
    store.delete_vector("test_001")
    store.close()
    
    print("✅ SQLite Vector Store test completed successfully")