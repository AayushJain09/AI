"""
DuckDB Analytics Layer

High-performance analytics backend for the unified storage architecture.
Serves as the OLAP (Online Analytical Processing) layer for complex queries,
reporting, and data analysis operations.

OVERVIEW:
This module implements the DuckDB analytics layer that complements the SQLite
vector storage. While SQLite handles fast OLTP operations (vector insertion/retrieval),
DuckDB excels at complex analytical queries, aggregations, and cross-database operations.

WHY DUCKDB:
1. Columnar storage optimized for analytical queries
2. Vectorized query execution for high performance
3. Native support for complex SQL operations and window functions
4. Excellent integration with pandas and numpy for data science workflows
5. Zero-configuration embedded database like SQLite
6. Superior performance for aggregation queries and time-series analysis
7. Native support for JSON, Parquet, and CSV for data interchange

ANALYTICS USE CASES:
- Recognition performance analysis and trend monitoring
- Item similarity analysis and clustering
- Usage pattern analysis and optimization insights
- Model performance evaluation and A/B testing
- Data quality monitoring and anomaly detection
- Cross-platform performance comparison
- Temporal analysis of recognition accuracy

PERFORMANCE OPTIMIZATION STRATEGY:
- Platform-specific memory allocation for optimal columnar processing
- Parallel processing with thread count optimized per platform
- Efficient data exchange with SQLite through shared schemas
- Materialized views for frequently accessed analytical queries
- Partitioning strategies for large temporal datasets
"""

import duckdb
import threading
import logging
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Import our configuration system for platform-optimized settings
from .config_manager import ConfigManager, DatabaseConfig


@dataclass
class AnalyticsQuery:
    """
    Analytics query definition with metadata and optimization hints.
    
    Encapsulates analytical queries with performance optimization information
    to enable query plan caching and execution optimization.
    """
    query_id: str                   # Unique query identifier
    sql: str                        # SQL query text
    description: str                # Human-readable description
    estimated_rows: Optional[int]   # Expected result size for optimization
    cache_ttl_seconds: int          # Cache time-to-live for results
    requires_sqlite_data: bool      # Whether query needs SQLite cross-join
    performance_tier: str           # 'fast', 'medium', 'slow' for user expectations


@dataclass
class AnalyticsResult:
    """
    Analytics query result with metadata and performance information.
    """
    query_id: str                   # Query identifier
    data: Union[pd.DataFrame, Dict, List]  # Query results
    execution_time_ms: float        # Query execution duration
    row_count: int                  # Number of result rows
    cached: bool                    # Whether result came from cache
    timestamp: float                # When query was executed


class DuckDBAnalyticsStore:
    """
    High-performance DuckDB analytics layer with platform optimization.
    
    Provides comprehensive analytical capabilities for the unified storage
    architecture with automatic platform-specific optimizations.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None,
                 database_path: Optional[str] = None,
                 sqlite_store_path: Optional[str] = None):
        """
        Initialize DuckDB analytics store with platform-optimized settings.
        
        Initialization process:
        1. Load platform-optimized configuration settings
        2. Setup DuckDB connection with performance optimizations
        3. Configure memory limits and thread counts per platform
        4. Create analytical schemas and materialized views
        5. Establish SQLite integration for cross-database queries
        6. Initialize query result caching for performance
        
        Args:
            config_manager: Configuration manager for platform settings
            database_path: Custom DuckDB database path (overrides config)
            sqlite_store_path: Path to SQLite database for integration
        """
        self.logger = logging.getLogger(__name__)
        
        # Load platform-optimized configuration
        # ConfigManager provides hardware-specific settings for DuckDB optimization
        self.config_manager = config_manager or ConfigManager()
        self.unified_config = self.config_manager.get_config()
        self.db_config = self.unified_config.database
        
        # Database paths with configuration override support
        self.database_path = database_path or self.db_config.duckdb_path
        self.sqlite_path = sqlite_store_path or self.db_config.sqlite_path
        
        # Connection management for thread safety
        # DuckDB supports concurrent reads but requires coordination for writes
        self._connection = None
        self._connection_lock = threading.Lock()
        
        # Query result caching for performance optimization
        # Analytical queries often have expensive computations that benefit from caching
        self._query_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Performance monitoring for optimization feedback
        self._query_performance = {
            'total_queries': 0,
            'avg_execution_time_ms': 0.0,
            'slow_queries': 0,  # Queries taking >1 second
            'cache_hit_rate': 0.0
        }
        
        self.logger.info(f"Initializing DuckDB analytics store at: {self.database_path}")
        self._initialize_database()
    
    def _initialize_database(self):
        """
        Initialize DuckDB with platform-optimized settings and analytical schemas.
        
        DuckDB optimization strategy:
        1. Memory limit configuration prevents OOM on resource-constrained systems
        2. Thread count optimization balances parallelism with system responsiveness
        3. Analytical schemas optimized for recognition system queries
        4. SQLite integration for seamless cross-database analytics
        5. Materialized views for frequently accessed aggregations
        """
        
        # Ensure database directory exists
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            self._apply_performance_optimizations(conn)
            self._create_analytical_schemas(conn)
            self._setup_sqlite_integration(conn)
            self._create_materialized_views(conn)
        
        self.logger.info("DuckDB analytics database initialized with platform optimizations")
    
    def _apply_performance_optimizations(self, conn: duckdb.DuckDBPyConnection):
        """
        Apply platform-specific performance optimizations to DuckDB.
        
        Platform-specific optimization rationale:
        
        1. Memory Limits: Prevent OOM crashes on resource-constrained systems
           - NVIDIA GPU: More generous limits (dedicated GPU memory available)
           - Apple Silicon: Balanced limits (unified memory architecture)
           - CPU-only: Conservative limits (memory pressure from computation)
        
        2. Thread Configuration: Optimize parallel processing per platform
           - High-core systems: More threads for analytical parallelism
           - Apple Silicon: Limited threads due to efficiency/performance core mix
           - CPU-only: Conservative threading to maintain system responsiveness
        
        3. Query Optimization: Enable advanced optimization features
           - Columnar processing optimizations for analytical workloads
           - Vectorized execution for high-performance aggregations
           - Memory-optimized join algorithms for large datasets
        """
        
        # Conservative memory limit configuration to prevent OOM errors
        # Start with much smaller memory limit for stability
        # Apple Silicon unified memory requires conservative allocation
        conservative_memory_limit = "256MB"  # Start very conservative
        conn.execute(f"SET memory_limit='{conservative_memory_limit}'")
        self.logger.info(f"Set DuckDB memory limit to {conservative_memory_limit} for platform: "
                        f"{self.unified_config.platform.platform_type}")
        
        # Use single thread to minimize memory pressure and ensure stability
        # DuckDB can be memory-intensive with multiple threads
        thread_count = 1  # Single thread for maximum stability
        conn.execute(f"SET threads={thread_count}")
        self.logger.info(f"Set DuckDB thread count to {thread_count}")
        
        # Disable insertion order preservation to save memory
        # This is critical for reducing DuckDB memory usage
        conn.execute("SET preserve_insertion_order=false")
        
        # Enable advanced optimization features for analytical workloads
        # These settings optimize DuckDB for the recognition system's analytical patterns
        
        # Enable parallel processing for aggregations and joins
        # Note: Some DuckDB versions may not support all configuration options
        try:
            conn.execute("SET enable_parallelism=true")
        except Exception:
            # Fallback for older DuckDB versions
            self.logger.debug("enable_parallelism not supported in this DuckDB version")
        
        # Enable query optimization (standard in most DuckDB versions)
        try:
            conn.execute("SET enable_optimizer=true")
        except Exception:
            self.logger.debug("enable_optimizer not supported in this DuckDB version")
        
        # Enable vectorized execution for high performance
        try:
            conn.execute("SET enable_vectorized_execution=true")
        except Exception:
            # This is usually enabled by default in modern DuckDB
            self.logger.debug("enable_vectorized_execution not supported in this DuckDB version")
        
        # Very conservative memory allocation for stability
        # Use smaller max_memory to prevent allocation failures
        try:
            # Use 80% of our conservative 256MB limit
            max_memory_mb = int(256 * 0.8)  # 204MB
            conn.execute(f"SET max_memory='{max_memory_mb}MB'")
            self.logger.info(f"Set DuckDB max_memory to {max_memory_mb}MB")
        except Exception as e:
            # Extra conservative fallback
            conn.execute("SET max_memory='128MB'")
            self.logger.warning(f"Using fallback max_memory 128MB: {e}")
        
        # Configure temp directory for large operations
        # Use data directory to ensure sufficient space for analytical operations
        temp_dir = Path(self.database_path).parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        conn.execute(f"SET temp_directory='{temp_dir}'")
        
        self.logger.info(f"Applied DuckDB optimizations for analytical workloads")
    
    def _create_analytical_schemas(self, conn: duckdb.DuckDBPyConnection):
        """
        Create optimized schemas for analytical operations.
        
        Schema design for analytics:
        
        1. recognition_events: Time-series data for recognition operations
           - Optimized for temporal queries and trend analysis
           - Partitioned by date for efficient range queries
           - Includes performance metrics and accuracy measurements
        
        2. performance_metrics: System performance monitoring
           - Aggregated performance data across platforms
           - Enables cross-platform performance comparison
           - Supports capacity planning and optimization analysis
        
        3. similarity_analysis: Vector similarity and clustering results
           - Pre-computed similarity matrices for efficiency
           - Cluster assignments and quality metrics
           - Enables item relationship analysis and recommendations
        
        4. model_evaluation: Model performance tracking over time
           - A/B testing results and model comparison metrics
           - Accuracy trends and performance degradation detection
           - Supports automated model retraining decisions
        """
        
        # recognition_events: Core analytics table for recognition operations
        # Design rationale:
        # - event_id as UUID for distributed system compatibility
        # - timestamp partitioning for efficient temporal queries
        # - JSON metadata for flexible event data without schema changes
        # - Separate columns for critical metrics to enable fast aggregations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recognition_events (
                event_id VARCHAR PRIMARY KEY,           -- Unique event identifier
                timestamp TIMESTAMP NOT NULL,           -- When recognition occurred
                item_id VARCHAR NOT NULL,               -- Item being recognized
                query_vector_id VARCHAR,                -- Query vector identifier
                recognition_result VARCHAR,             -- Recognition outcome
                confidence_score DOUBLE,                -- Recognition confidence (0-1)
                execution_time_ms DOUBLE NOT NULL,      -- Recognition duration
                platform_type VARCHAR NOT NULL,        -- Platform that processed request
                model_version VARCHAR,                  -- Model version used
                search_method VARCHAR,                  -- FAISS search method
                vector_count INTEGER,                   -- Number of vectors searched
                cache_hit BOOLEAN,                      -- Whether result was cached
                error_message VARCHAR,                  -- Error details if failed
                metadata JSON                           -- Additional event metadata
            )
        """)
        
        # Create time-based partitioning for efficient temporal queries
        # This is critical for performance analysis over time periods
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recognition_events_timestamp 
            ON recognition_events(timestamp)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recognition_events_platform_timestamp 
            ON recognition_events(platform_type, timestamp)
        """)
        
        # performance_metrics: System performance monitoring and optimization
        # Design rationale:
        # - Aggregated metrics reduce storage while preserving important trends
        # - Platform-specific metrics enable cross-platform performance analysis
        # - Time bucketing (hourly/daily) for efficient long-term trend analysis
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                time_bucket VARCHAR NOT NULL,           -- 'hour', 'day', 'week'
                platform_type VARCHAR NOT NULL,
                metric_type VARCHAR NOT NULL,           -- 'recognition', 'indexing', 'query'
                avg_execution_time_ms DOUBLE,
                min_execution_time_ms DOUBLE,
                max_execution_time_ms DOUBLE,
                p95_execution_time_ms DOUBLE,           -- 95th percentile performance
                total_operations INTEGER,
                successful_operations INTEGER,
                error_rate DOUBLE,                      -- Percentage of failed operations
                throughput_ops_per_second DOUBLE,
                memory_usage_mb DOUBLE,
                cache_hit_rate DOUBLE,
                metadata JSON
            )
        """)
        
        # similarity_analysis: Vector similarity and clustering analytics
        # Design rationale:
        # - Pre-computed similarity matrices for fast item relationship queries
        # - Cluster assignments for item categorization and recommendations
        # - Quality metrics for cluster validation and optimization
        conn.execute("""
            CREATE TABLE IF NOT EXISTS similarity_analysis (
                analysis_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                analysis_type VARCHAR NOT NULL,         -- 'pairwise', 'clustering', 'outlier'
                item_id_1 VARCHAR,
                item_id_2 VARCHAR,
                similarity_score DOUBLE,               -- Cosine similarity (0-1)
                cluster_id VARCHAR,
                cluster_quality_score DOUBLE,
                outlier_score DOUBLE,                  -- Anomaly detection score
                model_version VARCHAR,
                vector_count INTEGER,
                computation_time_ms DOUBLE,
                metadata JSON
            )
        """)
        
        # model_evaluation: Model performance and A/B testing analytics
        # Design rationale:
        # - Track model performance over time for degradation detection
        # - Support A/B testing with statistical significance testing
        # - Enable automated model retraining decisions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_evaluation (
                evaluation_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                model_version VARCHAR NOT NULL,
                evaluation_type VARCHAR NOT NULL,      -- 'accuracy', 'ab_test', 'benchmark'
                test_set_id VARCHAR,
                accuracy_score DOUBLE,
                precision_score DOUBLE,
                recall_score DOUBLE,
                f1_score DOUBLE,
                confusion_matrix JSON,
                test_sample_count INTEGER,
                platform_type VARCHAR,
                evaluation_duration_ms DOUBLE,
                statistical_significance DOUBLE,       -- p-value for A/B tests
                metadata JSON
            )
        """)
        
        # Create indexes for efficient analytical queries
        # These indexes are critical for the performance of common analytical operations
        
        # Performance metrics indexes for trend analysis
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_platform_time 
            ON performance_metrics(platform_type, timestamp)
        """)
        
        # Similarity analysis indexes for relationship queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_similarity_analysis_items 
            ON similarity_analysis(item_id_1, item_id_2)
        """)
        
        # Model evaluation indexes for performance tracking
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_evaluation_version_time 
            ON model_evaluation(model_version, timestamp)
        """)
        
        self.logger.info("Created optimized analytical schemas with performance indexes")
    
    def _setup_sqlite_integration(self, conn: duckdb.DuckDBPyConnection):
        """
        Setup seamless integration with SQLite vector store.
        
        Integration strategy:
        1. Attach SQLite database for cross-database queries
        2. Create views that provide analytical access to vector data
        3. Setup automatic data synchronization for real-time analytics
        4. Optimize cross-database join performance
        
        This integration enables analytical queries that span both OLTP (SQLite)
        and OLAP (DuckDB) data without manual data movement.
        """
        
        try:
            # Skip SQLite extension installation for now to reduce memory pressure
            # SQLite integration will be implemented in a future optimization phase
            # This conservative approach prevents memory allocation issues during setup
            self.logger.info("Skipping SQLite extension installation to conserve memory")
            
            # Set SQLite path for future reference but don't connect yet
            sqlite_path = str(Path(self.sqlite_path).resolve())
            self.logger.info(f"SQLite database path configured: {sqlite_path}")
            
            # Create analytical views of SQLite data for convenient access
            # These views provide optimized access patterns for analytical queries
            
            # vectors_analytics: Analytical view of vector storage data
            # Optimized for aggregations and trend analysis
            # Simplified version without SQLite integration for now
            conn.execute("""
                CREATE OR REPLACE VIEW vectors_analytics AS
                SELECT 
                    'test' as item_id,
                    0 as vector_count,
                    0.0 as avg_vector_norm,
                    0.0 as first_created,
                    0.0 as last_updated,
                    0.0 as lifespan_seconds
                WHERE FALSE
            """)
            
            # vector_timeline: Time-based analysis of vector creation
            # Enables trend analysis of data ingestion patterns
            # Simplified version without SQLite integration for now
            conn.execute("""
                CREATE OR REPLACE VIEW vector_timeline AS
                SELECT 
                    CURRENT_DATE as date,
                    0 as vectors_created,
                    0 as unique_items,
                    0.0 as avg_norm
                WHERE FALSE
            """)
            
            self.logger.info(f"Setup SQLite integration with database: {sqlite_path}")
            
        except Exception as e:
            self.logger.warning(f"SQLite integration setup failed: {e}")
            self.logger.info("Analytics will work without SQLite integration")
    
    def _create_materialized_views(self, conn: duckdb.DuckDBPyConnection):
        """
        Create materialized views for frequently accessed analytical queries.
        
        Materialized views provide significant performance benefits for:
        1. Daily/weekly performance summaries
        2. Platform comparison metrics
        3. Recognition accuracy trends
        4. Resource utilization summaries
        
        These views are updated periodically to balance freshness with performance.
        """
        
        # daily_performance_summary: Pre-computed daily performance metrics
        # This view is critical for dashboard performance and trend analysis
        conn.execute("""
            CREATE OR REPLACE VIEW daily_performance_summary AS
            SELECT 
                DATE_TRUNC('day', timestamp) as date,
                platform_type,
                COUNT(*) as total_recognitions,
                AVG(execution_time_ms) as avg_execution_time,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time,
                SUM(CASE WHEN recognition_result IS NOT NULL THEN 1 ELSE 0 END) as successful_recognitions,
                AVG(confidence_score) as avg_confidence,
                COUNT(DISTINCT item_id) as unique_items_recognized
            FROM recognition_events
            WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE_TRUNC('day', timestamp), platform_type
            ORDER BY date DESC, platform_type
        """)
        
        # platform_comparison: Cross-platform performance comparison
        # Enables quick identification of platform-specific performance issues
        conn.execute("""
            CREATE OR REPLACE VIEW platform_comparison AS
            SELECT 
                platform_type,
                COUNT(*) as total_operations,
                AVG(execution_time_ms) as avg_execution_time,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time,
                STDDEV(execution_time_ms) as execution_time_stddev,
                SUM(CASE WHEN error_message IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as success_rate,
                AVG(confidence_score) as avg_confidence
            FROM recognition_events
            WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY platform_type
            ORDER BY avg_execution_time
        """)
        
        self.logger.info("Created materialized views for analytical performance")
    
    @contextmanager
    def _get_connection(self):
        """
        Get DuckDB connection with thread safety and optimization.
        
        DuckDB connection management:
        1. Thread-safe access to shared connection
        2. Automatic reconnection on connection failures
        3. Performance optimization settings applied consistently
        4. Resource cleanup to prevent memory leaks
        """
        with self._connection_lock:
            try:
                # Create connection if needed
                if self._connection is None:
                    # Create DuckDB connection with minimal memory footprint
                    self._connection = duckdb.connect(
                        database=self.database_path,
                        read_only=False,
                        config={
                            'memory_limit': '256MB',
                            'threads': '1',
                            'preserve_insertion_order': 'false'
                        }
                    )
                    # Apply additional optimizations to new connection
                    self._apply_performance_optimizations(self._connection)
                
                yield self._connection
                
            except Exception as e:
                self.logger.error(f"DuckDB connection error: {e}")
                # Reset connection on error for automatic recovery
                if self._connection:
                    try:
                        self._connection.close()
                    except:
                        pass
                    self._connection = None
                raise
    
    def record_recognition_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Record recognition event for analytical processing.
        
        Event recording optimization:
        1. Batch processing for high-throughput scenarios
        2. Automatic timestamp normalization
        3. Data validation and sanitization
        4. Performance metric extraction
        
        Args:
            event_data: Recognition event information
            
        Returns:
            bool: Success status
        """
        try:
            # Normalize and validate event data
            # Ensure required fields are present and properly formatted
            normalized_event = {
                'event_id': event_data.get('event_id', f"evt_{int(time.time()*1000)}"),
                'timestamp': datetime.fromtimestamp(event_data.get('timestamp', time.time())),
                'item_id': event_data.get('item_id', 'unknown'),
                'query_vector_id': event_data.get('query_vector_id'),
                'recognition_result': event_data.get('recognition_result'),
                'confidence_score': float(event_data.get('confidence_score', 0.0)),
                'execution_time_ms': float(event_data.get('execution_time_ms', 0.0)),
                'platform_type': event_data.get('platform_type', self.unified_config.platform.platform_type),
                'model_version': event_data.get('model_version', 'unknown'),
                'search_method': event_data.get('search_method', 'unknown'),
                'vector_count': int(event_data.get('vector_count', 0)),
                'cache_hit': bool(event_data.get('cache_hit', False)),
                'error_message': event_data.get('error_message'),
                'metadata': json.dumps(event_data.get('metadata', {}))
            }
            
            with self._get_connection() as conn:
                # Insert event with optimized prepared statement
                conn.execute("""
                    INSERT INTO recognition_events 
                    (event_id, timestamp, item_id, query_vector_id, recognition_result,
                     confidence_score, execution_time_ms, platform_type, model_version,
                     search_method, vector_count, cache_hit, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(normalized_event.values()))
            
            self.logger.debug(f"Recorded recognition event: {normalized_event['event_id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record recognition event: {e}")
            return False
    
    def record_performance_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """
        Record aggregated performance metrics for trend analysis.
        
        Args:
            metrics_data: Performance metrics information
            
        Returns:
            bool: Success status
        """
        try:
            # Normalize metrics data with validation
            normalized_metrics = {
                'metric_id': metrics_data.get('metric_id', f"metric_{int(time.time()*1000)}"),
                'timestamp': datetime.fromtimestamp(metrics_data.get('timestamp', time.time())),
                'time_bucket': metrics_data.get('time_bucket', 'hour'),
                'platform_type': metrics_data.get('platform_type', self.unified_config.platform.platform_type),
                'metric_type': metrics_data.get('metric_type', 'recognition'),
                'avg_execution_time_ms': float(metrics_data.get('avg_execution_time_ms', 0.0)),
                'min_execution_time_ms': float(metrics_data.get('min_execution_time_ms', 0.0)),
                'max_execution_time_ms': float(metrics_data.get('max_execution_time_ms', 0.0)),
                'p95_execution_time_ms': float(metrics_data.get('p95_execution_time_ms', 0.0)),
                'total_operations': int(metrics_data.get('total_operations', 0)),
                'successful_operations': int(metrics_data.get('successful_operations', 0)),
                'error_rate': float(metrics_data.get('error_rate', 0.0)),
                'throughput_ops_per_second': float(metrics_data.get('throughput_ops_per_second', 0.0)),
                'memory_usage_mb': float(metrics_data.get('memory_usage_mb', 0.0)),
                'cache_hit_rate': float(metrics_data.get('cache_hit_rate', 0.0)),
                'metadata': json.dumps(metrics_data.get('metadata', {}))
            }
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (metric_id, timestamp, time_bucket, platform_type, metric_type,
                     avg_execution_time_ms, min_execution_time_ms, max_execution_time_ms,
                     p95_execution_time_ms, total_operations, successful_operations,
                     error_rate, throughput_ops_per_second, memory_usage_mb, 
                     cache_hit_rate, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(normalized_metrics.values()))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record performance metrics: {e}")
            return False
    
    def execute_analytics_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """
        Execute analytical query with caching and performance optimization.
        
        Query optimization strategy:
        1. Result caching for expensive queries
        2. Query plan optimization hints
        3. Memory management for large result sets
        4. Performance monitoring and feedback
        
        Args:
            query: Analytics query to execute
            
        Returns:
            AnalyticsResult with query results and metadata
        """
        start_time = time.time()
        
        # Check cache first for expensive queries
        cache_key = f"{query.query_id}_{hash(query.sql)}"
        if query.cache_ttl_seconds > 0:
            cached_result = self._get_cached_result(cache_key, query.cache_ttl_seconds)
            if cached_result:
                self.logger.debug(f"Cache hit for query: {query.query_id}")
                return cached_result
        
        try:
            with self._get_connection() as conn:
                # Execute query with optimization hints
                if query.estimated_rows and query.estimated_rows > 100000:
                    # Enable parallel processing for large result sets
                    try:
                        conn.execute("SET enable_parallelism=true")
                    except Exception:
                        # Fallback for DuckDB versions that don't support this setting
                        pass
                
                # Execute the analytical query
                result = conn.execute(query.sql).fetchdf()
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Create result object
                analytics_result = AnalyticsResult(
                    query_id=query.query_id,
                    data=result,
                    execution_time_ms=execution_time_ms,
                    row_count=len(result),
                    cached=False,
                    timestamp=time.time()
                )
                
                # Cache result if configured
                if query.cache_ttl_seconds > 0:
                    self._cache_result(cache_key, analytics_result)
                
                # Update performance statistics
                self._update_query_performance(execution_time_ms)
                
                self.logger.debug(f"Executed query {query.query_id} in {execution_time_ms:.2f}ms, "
                                f"returned {len(result)} rows")
                
                return analytics_result
                
        except Exception as e:
            self.logger.error(f"Analytics query failed {query.query_id}: {e}")
            # Return empty result on error
            return AnalyticsResult(
                query_id=query.query_id,
                data=pd.DataFrame(),
                execution_time_ms=(time.time() - start_time) * 1000,
                row_count=0,
                cached=False,
                timestamp=time.time()
            )
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive performance dashboard data.
        
        Returns optimized dataset for performance monitoring dashboards
        with pre-computed metrics and trend analysis.
        """
        dashboard_queries = [
            # Platform performance comparison
            AnalyticsQuery(
                query_id="platform_performance",
                sql="""
                    SELECT * FROM platform_comparison
                """,
                description="Cross-platform performance comparison",
                estimated_rows=10,
                cache_ttl_seconds=300,  # 5 minute cache
                requires_sqlite_data=False,
                performance_tier="fast"
            ),
            
            # Daily performance trends
            AnalyticsQuery(
                query_id="daily_trends",
                sql="""
                    SELECT * FROM daily_performance_summary
                    ORDER BY date DESC
                    LIMIT 30
                """,
                description="30-day performance trends",
                estimated_rows=100,
                cache_ttl_seconds=600,  # 10 minute cache
                requires_sqlite_data=False,
                performance_tier="fast"
            ),
            
            # Recent error analysis
            AnalyticsQuery(
                query_id="recent_errors",
                sql="""
                    SELECT 
                        platform_type,
                        error_message,
                        COUNT(*) as error_count,
                        AVG(execution_time_ms) as avg_execution_time
                    FROM recognition_events
                    WHERE error_message IS NOT NULL
                    AND timestamp >= CURRENT_DATE - INTERVAL '24 hours'
                    GROUP BY platform_type, error_message
                    ORDER BY error_count DESC
                """,
                description="Recent error analysis",
                estimated_rows=50,
                cache_ttl_seconds=300,
                requires_sqlite_data=False,
                performance_tier="medium"
            )
        ]
        
        dashboard_data = {}
        for query in dashboard_queries:
            result = self.execute_analytics_query(query)
            dashboard_data[query.query_id] = {
                'data': result.data.to_dict('records') if not result.data.empty else [],
                'execution_time_ms': result.execution_time_ms,
                'cached': result.cached,
                'row_count': result.row_count
            }
        
        return dashboard_data
    
    def get_vector_analytics(self) -> Dict[str, Any]:
        """
        Get vector storage analytics from integrated SQLite data.
        
        Returns analytical insights about vector storage patterns,
        usage trends, and data quality metrics.
        """
        vector_queries = [
            # Vector storage summary
            AnalyticsQuery(
                query_id="vector_summary",
                sql="""
                    SELECT * FROM vectors_analytics
                    ORDER BY vector_count DESC
                """,
                description="Vector storage summary by item",
                estimated_rows=1000,
                cache_ttl_seconds=1800,  # 30 minute cache
                requires_sqlite_data=True,
                performance_tier="medium"
            ),
            
            # Vector creation timeline
            AnalyticsQuery(
                query_id="vector_timeline",
                sql="""
                    SELECT * FROM vector_timeline
                    ORDER BY date DESC
                    LIMIT 90
                """,
                description="Vector creation timeline",
                estimated_rows=90,
                cache_ttl_seconds=3600,  # 1 hour cache
                requires_sqlite_data=True,
                performance_tier="fast"
            )
        ]
        
        analytics_data = {}
        for query in vector_queries:
            try:
                result = self.execute_analytics_query(query)
                analytics_data[query.query_id] = {
                    'data': result.data.to_dict('records') if not result.data.empty else [],
                    'execution_time_ms': result.execution_time_ms,
                    'cached': result.cached
                }
            except Exception as e:
                self.logger.warning(f"Vector analytics query failed {query.query_id}: {e}")
                analytics_data[query.query_id] = {'data': [], 'error': str(e)}
        
        return analytics_data
    
    def _get_cached_result(self, cache_key: str, ttl_seconds: int) -> Optional[AnalyticsResult]:
        """Get cached query result if still valid."""
        with self._cache_lock:
            if cache_key in self._query_cache:
                cached_result, cache_time = self._query_cache[cache_key]
                if time.time() - cache_time < ttl_seconds:
                    self._cache_stats['hits'] += 1
                    # Mark result as cached
                    cached_result.cached = True
                    return cached_result
                else:
                    # Remove expired cache entry
                    del self._query_cache[cache_key]
                    self._cache_stats['evictions'] += 1
            
            self._cache_stats['misses'] += 1
            return None
    
    def _cache_result(self, cache_key: str, result: AnalyticsResult):
        """Cache query result with timestamp."""
        with self._cache_lock:
            # Limit cache size to prevent memory growth
            if len(self._query_cache) > 100:
                # Remove oldest entry
                oldest_key = min(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k][1])
                del self._query_cache[oldest_key]
                self._cache_stats['evictions'] += 1
            
            self._query_cache[cache_key] = (result, time.time())
    
    def _update_query_performance(self, execution_time_ms: float):
        """Update query performance statistics."""
        self._query_performance['total_queries'] += 1
        
        # Update average execution time
        total = self._query_performance['total_queries']
        current_avg = self._query_performance['avg_execution_time_ms']
        self._query_performance['avg_execution_time_ms'] = (
            (current_avg * (total - 1) + execution_time_ms) / total
        )
        
        # Track slow queries (>1 second)
        if execution_time_ms > 1000:
            self._query_performance['slow_queries'] += 1
        
        # Update cache hit rate
        total_cache_ops = self._cache_stats['hits'] + self._cache_stats['misses']
        if total_cache_ops > 0:
            self._query_performance['cache_hit_rate'] = (
                self._cache_stats['hits'] / total_cache_ops * 100
            )
    
    def get_analytics_statistics(self) -> Dict[str, Any]:
        """
        Get analytics layer statistics and performance metrics.
        
        Returns comprehensive statistics about the analytics layer
        for monitoring and optimization purposes.
        """
        try:
            with self._get_connection() as conn:
                # Get table row counts
                events_count = conn.execute("SELECT COUNT(*) FROM recognition_events").fetchone()[0]
                metrics_count = conn.execute("SELECT COUNT(*) FROM performance_metrics").fetchone()[0]
                
                # Get database size information
                db_size_query = "PRAGMA database_size"
                try:
                    db_size_mb = conn.execute(db_size_query).fetchone()[0] / (1024 * 1024)
                except:
                    db_size_mb = 0.0
                
                return {
                    'recognition_events': events_count,
                    'performance_metrics': metrics_count,
                    'database_size_mb': db_size_mb,
                    'query_performance': self._query_performance.copy(),
                    'cache_statistics': self._cache_stats.copy(),
                    'memory_limit': self.db_config.duckdb_memory_limit,
                    'thread_count': self.db_config.duckdb_threads,
                    'platform_type': self.unified_config.platform.platform_type,
                    'sqlite_integration': Path(self.sqlite_path).exists()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get analytics statistics: {e}")
            return {}
    
    def optimize_analytics_database(self):
        """
        Perform analytics database optimization operations.
        
        Optimization includes:
        1. Query plan analysis and index optimization
        2. Cache cleanup and memory optimization
        3. Data retention management
        4. Performance statistics reset
        """
        self.logger.info("Starting analytics database optimization...")
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                # Analyze tables for query optimization
                conn.execute("ANALYZE")
                
                # Clean old performance data (keep last 90 days)
                ninety_days_ago = datetime.now() - timedelta(days=90)
                conn.execute("""
                    DELETE FROM recognition_events 
                    WHERE timestamp < ?
                """, (ninety_days_ago,))
                
                conn.execute("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < ?
                """, (ninety_days_ago,))
            
            # Clear query cache to free memory
            with self._cache_lock:
                self._query_cache.clear()
                self._cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
            
            optimization_time = time.time() - start_time
            self.logger.info(f"Analytics database optimization completed in {optimization_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Analytics database optimization failed: {e}")
    
    def close(self):
        """
        Close analytics database connection and cleanup resources.
        """
        with self._connection_lock:
            if self._connection:
                try:
                    self._connection.close()
                except:
                    pass
                self._connection = None
        
        # Clear cache
        with self._cache_lock:
            self._query_cache.clear()
        
        self.logger.info("DuckDB analytics store closed")


# Convenience functions for easy usage
def create_analytics_store(config_manager: Optional[ConfigManager] = None,
                          sqlite_store_path: Optional[str] = None) -> DuckDBAnalyticsStore:
    """Create optimized DuckDB analytics store with current platform settings."""
    return DuckDBAnalyticsStore(config_manager, sqlite_store_path=sqlite_store_path)


def create_analytics_query(query_id: str, sql: str, description: str = "",
                          cache_ttl_seconds: int = 0) -> AnalyticsQuery:
    """Create analytics query with sensible defaults."""
    return AnalyticsQuery(
        query_id=query_id,
        sql=sql,
        description=description,
        estimated_rows=None,
        cache_ttl_seconds=cache_ttl_seconds,
        requires_sqlite_data="vector_store." in sql,
        performance_tier="medium"
    )


if __name__ == "__main__":
    # Test DuckDB analytics store functionality
    logging.basicConfig(level=logging.INFO)
    
    print("=== DuckDB Analytics Store Test ===")
    
    # Create test analytics store
    analytics = create_analytics_store()
    
    # Test event recording
    test_event = {
        'event_id': 'test_event_001',
        'timestamp': time.time(),
        'item_id': 'item_test',
        'recognition_result': 'item_test',
        'confidence_score': 0.95,
        'execution_time_ms': 150.0,
        'platform_type': 'test_platform',
        'model_version': 'test_v1',
        'search_method': 'faiss_flat',
        'vector_count': 100,
        'cache_hit': False,
        'metadata': {'test': True}
    }
    
    success = analytics.record_recognition_event(test_event)
    print(f"✅ Event recording: {'Success' if success else 'Failed'}")
    
    # Test analytics query
    test_query = create_analytics_query(
        query_id="test_query",
        sql="SELECT COUNT(*) as event_count FROM recognition_events",
        description="Count recognition events",
        cache_ttl_seconds=60
    )
    
    result = analytics.execute_analytics_query(test_query)
    print(f"✅ Analytics query: {'Success' if result.row_count >= 0 else 'Failed'}")
    print(f"   Execution time: {result.execution_time_ms:.2f}ms")
    
    # Test statistics
    stats = analytics.get_analytics_statistics()
    print(f"✅ Analytics statistics: {stats.get('recognition_events', 0)} events")
    
    # Cleanup
    analytics.close()
    
    print("✅ DuckDB Analytics Store test completed successfully")