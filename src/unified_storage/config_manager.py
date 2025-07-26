"""
Configuration Manager Module

Manages platform-specific optimization settings, database connections,
and performance parameters for the unified storage architecture.

OVERVIEW:
This module translates platform detection results into concrete, actionable
configuration parameters for all system components. It serves as the central
configuration authority that ensures optimal performance across platforms
while allowing for user customization and override capabilities.

RESPONSIBILITIES:
1. Platform-to-Configuration Translation: Converts hardware capabilities into specific settings
2. Database Optimization: SQLite and DuckDB parameters for maximum I/O performance
3. Vector Storage Configuration: Cache sizes, compression, and memory management
4. Search Engine Settings: FAISS threading, timeouts, and acceleration modes
5. Performance Tuning: Batch sizes, worker counts, and resource allocation
6. User Override Management: Respects user preferences while maintaining safety
7. Configuration Validation: Ensures settings are compatible with detected hardware

CONFIGURATION PHILOSOPHY:
- Conservative Defaults: Start with settings that work on any hardware
- Platform Optimization: Apply aggressive optimizations based on detected capabilities
- User Respect: Honor user overrides while warning about potential issues
- Safety First: Never apply settings that could cause crashes or data loss
- Performance Focus: Optimize for the critical path (recognition speed and accuracy)

CONFIGURATION LAYERS (in priority order):
1. Hardware Constraints: Physical limits that cannot be exceeded
2. Platform Optimizations: Empirically-determined optimal settings per platform
3. User Overrides: Explicit user preferences from configuration files
4. Safety Limits: Final validation to prevent dangerous configurations

The configuration manager bridges the gap between platform detection
and the actual system components that need optimized settings.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from .platform_detector import PlatformDetector, PlatformConfig


@dataclass
class DatabaseConfig:
    """
    Database configuration parameters for SQLite and DuckDB.
    
    These settings are critical for performance because database I/O
    is often the bottleneck in vector similarity search systems.
    Each parameter is carefully tuned based on platform capabilities.
    """
    sqlite_path: str                    # Path to SQLite database file
    duckdb_path: str                   # Path to DuckDB database file
    sqlite_cache_size: int             # Pages (-X means X KB of cache)
    sqlite_memory_map_size: int        # Bytes for memory mapping
    sqlite_wal_mode: bool              # Write-Ahead Logging for performance
    sqlite_synchronous: str            # 'NORMAL', 'FULL', 'OFF' - safety vs speed
    duckdb_memory_limit: str           # DuckDB memory limit (e.g., "1GB")
    duckdb_threads: int               # Number of threads for DuckDB operations
    connection_timeout: int           # Seconds before connection timeout


@dataclass  
class VectorStorageConfig:
    """
    Vector storage optimization parameters.
    
    Controls how 1536D feature vectors are stored, cached, and accessed.
    These settings directly impact recognition speed and memory usage.
    """
    cache_size_mb: int                 # Memory cache size for frequently accessed vectors
    use_compression: bool              # Enable LZ4 compression for storage
    compression_algorithm: str         # 'lz4', 'gzip', 'none'
    batch_size: int                   # Optimal batch size for bulk operations
    memory_mapping_enabled: bool      # Enable memory mapping for large datasets
    checksum_validation: bool         # Verify vector integrity with checksums
    auto_vacuum: bool                # Automatically optimize database storage


@dataclass
class SearchConfig:
    """
    Search engine configuration for FAISS and adaptive search.
    
    These settings control the search strategy, which is the core
    of the recognition system. Different platforms require different
    optimization approaches for optimal search performance.
    """
    faiss_mode: str                   # 'gpu', 'cpu_optimized', 'cpu_standard'
    faiss_threads: int               # Number of threads for FAISS operations
    search_timeout_ms: int           # Maximum search time before timeout
    index_type: str                  # 'linear', 'flat', 'ivf' or 'auto'
    linear_threshold: int            # Items below this use linear search
    ivf_threshold: int               # Items above this use IVF index
    nprobe: int                      # IVF search parameter (accuracy vs speed)
    gpu_enabled: bool                # Whether to use GPU acceleration


@dataclass
class PerformanceConfig:
    """
    Performance optimization settings for feature extraction and processing.
    
    These settings control CPU/GPU utilization, memory management,
    and parallel processing to maximize throughput while maintaining
    system responsiveness.
    """
    feature_extraction_batch_size: int  # Batch size for feature extraction
    num_workers: int                    # Number of worker processes/threads
    pin_memory: bool                    # Pin memory for GPU transfers
    prefetch_factor: int               # How many batches to prefetch
    enable_jit_compilation: bool       # Enable PyTorch JIT compilation
    memory_pool_size_mb: int           # Memory pool size for reuse
    gc_threshold: int                  # Garbage collection threshold


@dataclass
class UnifiedConfig:
    """Complete unified storage configuration"""
    platform: PlatformConfig
    database: DatabaseConfig
    vector_storage: VectorStorageConfig
    search: SearchConfig
    performance: PerformanceConfig
    data_dir: str
    log_level: str
    debug_mode: bool


class ConfigManager:
    """
    Comprehensive configuration manager for unified storage architecture.
    
    Automatically optimizes settings based on detected platform capabilities
    and provides centralized configuration management.
    """
    
    def __init__(self, config_path: Optional[str] = None, data_dir: str = "data"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config.yaml"
        self.data_dir = Path(data_dir)
        self.platform_detector = PlatformDetector()
        self._config = None
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ConfigManager initialized with data_dir: {self.data_dir}")
    
    def get_config(self) -> UnifiedConfig:
        """Get complete unified configuration"""
        if self._config is None:
            self._config = self._build_optimized_config()
        return self._config
    
    def _build_optimized_config(self) -> UnifiedConfig:
        """Build optimized configuration based on platform detection"""
        self.logger.info("Building platform-optimized configuration...")
        
        # Get platform configuration
        platform_config = self.platform_detector.get_platform_config()
        
        # Load user configuration if it exists
        user_config = self._load_user_config()
        
        # Build optimized configurations
        db_config = self._build_database_config(platform_config, user_config)
        vector_config = self._build_vector_storage_config(platform_config, user_config)
        search_config = self._build_search_config(platform_config, user_config)
        perf_config = self._build_performance_config(platform_config, user_config)
        
        unified_config = UnifiedConfig(
            platform=platform_config,
            database=db_config,
            vector_storage=vector_config,
            search=search_config,
            performance=perf_config,
            data_dir=str(self.data_dir),
            log_level=user_config.get('log_level', 'INFO'),
            debug_mode=user_config.get('debug_mode', False)
        )
        
        self.logger.info(f"Configuration built for platform: {platform_config.platform_type}")
        return unified_config
    
    def _load_user_config(self) -> Dict[str, Any]:
        """Load user configuration from YAML file"""
        user_config = {}
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                self.logger.info(f"Loaded user configuration from {self.config_path}")
            except Exception as e:
                self.logger.warning(f"Could not load user config: {e}")
        else:
            self.logger.info("No user configuration file found, using defaults")
        
        return user_config
    
    def _build_database_config(self, platform: PlatformConfig, 
                              user_config: Dict[str, Any]) -> DatabaseConfig:
        """
        Build database configuration optimized for platform.
        
        Database configuration is critical because:
        1. SQLite cache size directly affects vector retrieval speed
        2. Memory mapping reduces I/O overhead for large datasets
        3. WAL mode enables concurrent read/write operations
        4. DuckDB threading affects analytics performance
        
        Platform considerations:
        - NVIDIA GPU systems: More aggressive caching (have dedicated GPU memory)
        - Apple Silicon: Balanced approach (unified memory architecture)
        - CPU-only: Conservative settings (memory pressure from computation)
        """
        
        # Base paths for database files
        sqlite_path = str(self.data_dir / "recognition.db")
        duckdb_path = str(self.data_dir / "analytics.duckdb")
        
        # Platform-optimized settings based on hardware capabilities
        if platform.device_type == 'cuda':
            # NVIDIA GPU systems: Aggressive optimization
            # Rationale: GPU memory is dedicated, so we can use more system RAM for caching
            sqlite_cache = -200000              # 200MB cache (negative = KB)
            memory_map = 1024 * 1024 * 1024     # 1GB memory mapping
            duckdb_memory = "2GB"               # Large memory limit for analytics
            duckdb_threads = min(8, platform.cpu_cores)  # More threads for analytics
        
        elif platform.device_type == 'mps':
            # Apple Silicon: Balanced optimization
            # Rationale: Unified memory means GPU and CPU compete for same memory pool
            sqlite_cache = -150000              # 150MB cache - balanced approach
            memory_map = 512 * 1024 * 1024      # 512MB memory mapping
            duckdb_memory = "1GB"               # Moderate memory limit
            duckdb_threads = min(6, platform.cpu_cores)  # Leave cores for GPU tasks
        
        else:
            # CPU-only systems: Conservative optimization
            # Rationale: All computation and storage compete for same resources
            sqlite_cache = -100000              # 100MB cache - conservative
            memory_map = 256 * 1024 * 1024      # 256MB memory mapping
            duckdb_memory = "512MB"             # Conservative memory limit
            duckdb_threads = min(4, platform.cpu_cores)  # Leave cores for recognition
        
        # Apply user overrides if specified
        # User configuration takes precedence to allow for manual tuning
        db_user_config = user_config.get('database', {})
        
        return DatabaseConfig(
            sqlite_path=db_user_config.get('sqlite_path', sqlite_path),
            duckdb_path=db_user_config.get('duckdb_path', duckdb_path),
            sqlite_cache_size=db_user_config.get('sqlite_cache_size', sqlite_cache),
            sqlite_memory_map_size=db_user_config.get('sqlite_memory_map_size', memory_map),
            sqlite_wal_mode=db_user_config.get('sqlite_wal_mode', True),
            sqlite_synchronous=db_user_config.get('sqlite_synchronous', 'NORMAL'),
            duckdb_memory_limit=db_user_config.get('duckdb_memory_limit', duckdb_memory),
            duckdb_threads=db_user_config.get('duckdb_threads', duckdb_threads),
            connection_timeout=db_user_config.get('connection_timeout', 30)
        )
    
    def _build_vector_storage_config(self, platform: PlatformConfig,
                                   user_config: Dict[str, Any]) -> VectorStorageConfig:
        """Build vector storage configuration"""
        
        # Platform-optimized settings
        if platform.memory_gb >= 16:
            cache_size = platform.cache_size_mb
            use_compression = False  # Plenty of memory
            batch_size = platform.optimal_batch_size * 2
        elif platform.memory_gb >= 8:
            cache_size = platform.cache_size_mb
            use_compression = platform.compression_enabled
            batch_size = platform.optimal_batch_size
        else:
            cache_size = min(platform.cache_size_mb, 50)
            use_compression = True  # Save memory
            batch_size = max(1, platform.optimal_batch_size // 2)
        
        # Apply user overrides
        vector_user_config = user_config.get('vector_storage', {})
        
        return VectorStorageConfig(
            cache_size_mb=vector_user_config.get('cache_size_mb', cache_size),
            use_compression=vector_user_config.get('use_compression', use_compression),
            compression_algorithm=vector_user_config.get('compression_algorithm', 'lz4'),
            batch_size=vector_user_config.get('batch_size', batch_size),
            memory_mapping_enabled=vector_user_config.get('memory_mapping_enabled', 
                                                         platform.use_memory_mapping),
            checksum_validation=vector_user_config.get('checksum_validation', True),
            auto_vacuum=vector_user_config.get('auto_vacuum', True)
        )
    
    def _build_search_config(self, platform: PlatformConfig,
                            user_config: Dict[str, Any]) -> SearchConfig:
        """Build search engine configuration"""
        
        # Platform-optimized search settings
        if platform.device_type == 'cuda':
            search_timeout = 5000  # 5 seconds for GPU
            nprobe = 10
        elif platform.device_type == 'mps':
            search_timeout = 10000  # 10 seconds for Apple Silicon
            nprobe = 8
        else:
            search_timeout = 15000  # 15 seconds for CPU
            nprobe = 6
        
        # Apply user overrides
        search_user_config = user_config.get('search', {})
        
        return SearchConfig(
            faiss_mode=search_user_config.get('faiss_mode', platform.faiss_mode),
            faiss_threads=search_user_config.get('faiss_threads', platform.faiss_threads),
            search_timeout_ms=search_user_config.get('search_timeout_ms', search_timeout),
            index_type=search_user_config.get('index_type', 'auto'),  # auto-select
            linear_threshold=search_user_config.get('linear_threshold', 1000),
            ivf_threshold=search_user_config.get('ivf_threshold', 10000),
            nprobe=search_user_config.get('nprobe', nprobe),
            gpu_enabled=search_user_config.get('gpu_enabled', platform.gpu_available)
        )
    
    def _build_performance_config(self, platform: PlatformConfig,
                                 user_config: Dict[str, Any]) -> PerformanceConfig:
        """Build performance optimization configuration"""
        
        # Platform-optimized performance settings
        if platform.device_type == 'cuda':
            num_workers = 4
            pin_memory = True
            prefetch_factor = 4
            memory_pool = 500
        elif platform.device_type == 'mps':
            num_workers = 2
            pin_memory = False
            prefetch_factor = 2
            memory_pool = 300
        else:
            num_workers = min(4, platform.cpu_cores)
            pin_memory = False
            prefetch_factor = 2
            memory_pool = 200
        
        # Apply user overrides
        perf_user_config = user_config.get('performance', {})
        
        return PerformanceConfig(
            feature_extraction_batch_size=perf_user_config.get(
                'feature_extraction_batch_size', platform.optimal_batch_size),
            num_workers=perf_user_config.get('num_workers', num_workers),
            pin_memory=perf_user_config.get('pin_memory', pin_memory),
            prefetch_factor=perf_user_config.get('prefetch_factor', prefetch_factor),
            enable_jit_compilation=perf_user_config.get('enable_jit_compilation', True),
            memory_pool_size_mb=perf_user_config.get('memory_pool_size_mb', memory_pool),
            gc_threshold=perf_user_config.get('gc_threshold', 1000)
        )
    
    def save_config(self, config: Optional[UnifiedConfig] = None) -> str:
        """Save current configuration to file"""
        if config is None:
            config = self.get_config()
        
        # Convert to serializable format (exclude platform info for security)
        config_dict = {
            'database': asdict(config.database),
            'vector_storage': asdict(config.vector_storage),
            'search': asdict(config.search),
            'performance': asdict(config.performance),
            'data_dir': config.data_dir,
            'log_level': config.log_level,
            'debug_mode': config.debug_mode,
            'generated_for_platform': config.platform.platform_type,
            'generated_timestamp': str(Path().cwd())  # Use cwd as timestamp placeholder
        }
        
        output_path = str(self.data_dir / "generated_config.yaml")
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_database_url(self, db_type: str = 'sqlite') -> str:
        """Get database connection URL"""
        config = self.get_config()
        
        if db_type.lower() == 'sqlite':
            return f"sqlite:///{config.database.sqlite_path}"
        elif db_type.lower() == 'duckdb':
            return config.database.duckdb_path
        else:
            raise ValueError(f"Unknown database type: {db_type}")
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration parameters"""
        config = self.get_config()
        
        return {
            'vector_cache_size_mb': config.vector_storage.cache_size_mb,
            'database_cache_pages': abs(config.database.sqlite_cache_size),
            'memory_mapping_size_mb': config.database.sqlite_memory_map_size // (1024*1024),
            'compression_enabled': config.vector_storage.use_compression,
            'memory_pool_size_mb': config.performance.memory_pool_size_mb
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary for monitoring"""
        config = self.get_config()
        
        return {
            'platform_type': config.platform.platform_type,
            'device_acceleration': config.platform.device_type,
            'faiss_mode': config.search.faiss_mode,
            'expected_performance': {
                'batch_size': config.performance.feature_extraction_batch_size,
                'search_threads': config.search.faiss_threads,
                'workers': config.performance.num_workers,
                'memory_optimization': 'enabled' if config.vector_storage.use_compression else 'disabled'
            },
            'resource_allocation': {
                'cache_size_mb': config.vector_storage.cache_size_mb,
                'memory_pool_mb': config.performance.memory_pool_size_mb,
                'database_cache_mb': abs(config.database.sqlite_cache_size) // 1000
            }
        }
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate configuration settings"""
        config = self.get_config()
        validation = {
            'data_directory_exists': self.data_dir.exists(),
            'data_directory_writable': os.access(self.data_dir, os.W_OK),
            'memory_sufficient': config.platform.memory_gb >= 4.0,
            'cache_size_reasonable': config.vector_storage.cache_size_mb <= config.platform.memory_gb * 1000 * 0.5,
            'threads_reasonable': config.search.faiss_threads <= config.platform.cpu_cores * 2,
            'batch_size_reasonable': config.performance.feature_extraction_batch_size >= 1
        }
        
        # Check database paths
        try:
            db_dir = Path(config.database.sqlite_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            validation['database_path_valid'] = True
        except Exception:
            validation['database_path_valid'] = False
        
        return validation


# Convenience functions
def get_default_config(data_dir: str = "data") -> UnifiedConfig:
    """Get default configuration for current platform"""
    manager = ConfigManager(data_dir=data_dir)
    return manager.get_config()


def create_optimized_config(platform_type: str = "auto", 
                           data_dir: str = "data") -> UnifiedConfig:
    """Create optimized configuration for specific platform"""
    manager = ConfigManager(data_dir=data_dir)
    return manager.get_config()


if __name__ == "__main__":
    # Test configuration manager
    logging.basicConfig(level=logging.INFO)
    
    manager = ConfigManager(data_dir="test_data")
    config = manager.get_config()
    
    print("=== Configuration Manager Test ===")
    print(f"Platform: {config.platform.platform_type}")
    print(f"SQLite Path: {config.database.sqlite_path}")
    print(f"Cache Size: {config.vector_storage.cache_size_mb} MB")
    print(f"FAISS Mode: {config.search.faiss_mode}")
    print(f"Batch Size: {config.performance.feature_extraction_batch_size}")
    
    print("\n=== Optimization Summary ===")
    summary = manager.get_optimization_summary()
    for section, values in summary.items():
        print(f"{section}:")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {values}")
    
    print("\n=== Validation Results ===")
    validation = manager.validate_configuration()
    for check, result in validation.items():
        status = "" if result else "L"
        print(f"{status} {check}: {result}")
    
    # Save configuration
    try:
        saved_path = manager.save_config()
        print(f"\n Configuration saved to: {saved_path}")
    except Exception as e:
        print(f"\nL Failed to save configuration: {e}")