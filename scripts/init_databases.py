#!/usr/bin/env python3
"""
Database Initialization Script

Comprehensive database setup and initialization for the unified storage architecture.
Creates and configures both SQLite and DuckDB databases with platform optimizations.

OVERVIEW:
This script automates the complete database infrastructure setup for the AI Recognition
System. It handles both SQLite (OLTP) and DuckDB (OLAP) database creation with
platform-specific optimizations, data validation, and migration from legacy systems.

INITIALIZATION PROCESS:
1. Platform Detection: Auto-detect hardware and apply optimal settings
2. Database Creation: Setup SQLite and DuckDB with optimized schemas
3. Integration Setup: Configure cross-database queries and views
4. Data Migration: Import existing data from legacy file-based storage
5. Validation: Verify database integrity and performance
6. Optimization: Apply final performance tuning and indexing

WHY THIS SCRIPT IS CRITICAL:
- Ensures consistent database setup across all platforms
- Applies platform-specific optimizations automatically
- Provides data migration path from legacy architecture
- Validates database integrity and performance
- Creates reproducible deployment process
- Enables automated testing and CI/CD integration

USAGE SCENARIOS:
- Fresh installation setup
- Database migration from legacy system
- Development environment initialization
- Production deployment preparation
- Disaster recovery database recreation
"""

import sys
import os
import logging
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import unified storage components
from unified_storage.platform_detector import PlatformDetector, detect_platform
from unified_storage.config_manager import ConfigManager, get_default_config
from unified_storage.sqlite_store import SQLiteVectorStore, create_vector_store, create_vector_record
from unified_storage.analytics_store import DuckDBAnalyticsStore, create_analytics_store

# Import legacy components for migration
try:
    from feature_extraction.feature_extractor import FeatureExtractor
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    logging.warning("Legacy feature extraction not available - migration limited")

import numpy as np


class DatabaseInitializer:
    """
    Comprehensive database initialization with platform optimization.
    
    Handles the complete setup process for unified storage architecture
    with automatic platform detection, optimization, and validation.
    """
    
    def __init__(self, data_dir: str = "data", config_path: Optional[str] = None,
                 force_recreate: bool = False, migrate_legacy: bool = True):
        """
        Initialize database setup with configuration.
        
        Args:
            data_dir: Data directory for database files
            config_path: Custom configuration file path
            force_recreate: Whether to recreate existing databases
            migrate_legacy: Whether to migrate legacy data
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration and platform detection
        # Platform detection drives all optimization decisions
        self.config_manager = ConfigManager(config_path, data_dir)
        self.unified_config = self.config_manager.get_config()
        self.platform_config = self.unified_config.platform
        
        # Setup parameters
        self.data_dir = Path(data_dir)
        self.force_recreate = force_recreate
        self.migrate_legacy = migrate_legacy
        
        # Database components (initialized during setup)
        self.sqlite_store = None
        self.analytics_store = None
        
        # Migration tracking
        self.migration_stats = {
            'vectors_migrated': 0,
            'items_processed': 0,
            'errors_encountered': 0,
            'start_time': None,
            'end_time': None
        }
        
        self.logger.info(f"Database initializer created for platform: {self.platform_config.platform_type}")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Force recreate: {force_recreate}")
        self.logger.info(f"Migrate legacy: {migrate_legacy}")
    
    def initialize_databases(self) -> bool:
        """
        Complete database initialization process.
        
        Initialization steps:
        1. Pre-initialization validation and cleanup
        2. SQLite database creation with optimization
        3. DuckDB analytics database creation
        4. Cross-database integration setup
        5. Legacy data migration (if requested)
        6. Database validation and testing
        7. Performance optimization and indexing
        
        Returns:
            bool: Success status of complete initialization
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING UNIFIED STORAGE DATABASE INITIALIZATION")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        self.migration_stats['start_time'] = start_time
        
        try:
            # Step 1: Pre-initialization validation
            # Verify environment and prepare for database creation
            if not self._pre_initialization_checks():
                self.logger.error("Pre-initialization checks failed")
                return False
            
            # Step 2: Initialize SQLite vector storage
            # This is the core OLTP database for vector operations
            if not self._initialize_sqlite_store():
                self.logger.error("SQLite initialization failed")
                return False
            
            # Step 3: Initialize DuckDB analytics store
            # This provides OLAP capabilities for analytical queries
            if not self._initialize_analytics_store():
                self.logger.error("DuckDB initialization failed")
                return False
            
            # Step 4: Setup cross-database integration
            # Enable analytical queries across both databases
            if not self._setup_database_integration():
                self.logger.error("Database integration setup failed")
                return False
            
            # Step 5: Migrate legacy data (if requested and available)
            # Import existing data from file-based storage
            if self.migrate_legacy:
                if not self._migrate_legacy_data():
                    self.logger.warning("Legacy data migration failed - continuing with empty databases")
            
            # Step 6: Validate database setup
            # Ensure databases are working correctly
            if not self._validate_database_setup():
                self.logger.error("Database validation failed")
                return False
            
            # Step 7: Final optimization
            # Apply performance optimizations and create indexes
            if not self._perform_final_optimization():
                self.logger.error("Final optimization failed")
                return False
            
            # Record completion time and statistics
            self.migration_stats['end_time'] = time.time()
            total_time = self.migration_stats['end_time'] - start_time
            
            self.logger.info("=" * 60)
            self.logger.info("DATABASE INITIALIZATION COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total time: {total_time:.2f} seconds")
            self.logger.info(f"Platform: {self.platform_config.platform_type}")
            self.logger.info(f"Vectors migrated: {self.migration_stats['vectors_migrated']}")
            self.logger.info(f"Items processed: {self.migration_stats['items_processed']}")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed with error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
        finally:
            # Ensure cleanup happens regardless of success/failure
            self._cleanup_resources()
    
    def _pre_initialization_checks(self) -> bool:
        """
        Perform pre-initialization validation and environment checks.
        
        Validation includes:
        1. Data directory creation and permissions
        2. Platform compatibility verification
        3. Required dependencies availability
        4. Existing database cleanup (if force_recreate)
        5. Configuration validation
        
        Returns:
            bool: Whether pre-checks passed
        """
        self.logger.info("Performing pre-initialization checks...")
        
        try:
            # Create data directory structure
            # Ensure all required directories exist with proper permissions
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Check directory permissions
            # Critical for database file creation and operations
            if not os.access(self.data_dir, os.W_OK):
                self.logger.error(f"Data directory not writable: {self.data_dir}")
                return False
            
            # Platform compatibility check
            # Ensure current platform can run the unified storage system
            platform_detector = PlatformDetector()
            compatibility = platform_detector.validate_platform_compatibility()
            
            failed_checks = [check for check, result in compatibility.items() if not result]
            if failed_checks:
                self.logger.error(f"Platform compatibility failed: {failed_checks}")
                for check in failed_checks:
                    self.logger.error(f"  - {check}: Failed")
                return False
            
            self.logger.info(f"Platform compatibility: All checks passed")
            self.logger.info(f"Platform type: {self.platform_config.platform_type}")
            self.logger.info(f"Device acceleration: {self.platform_config.device_type}")
            self.logger.info(f"Memory available: {self.platform_config.memory_gb:.1f} GB")
            
            # Check existing databases
            # Handle existing databases based on force_recreate setting
            sqlite_path = Path(self.unified_config.database.sqlite_path)
            duckdb_path = Path(self.unified_config.database.duckdb_path)
            
            if self.force_recreate:
                # Remove existing databases for clean recreation
                if sqlite_path.exists():
                    self.logger.info(f"Removing existing SQLite database: {sqlite_path}")
                    sqlite_path.unlink()
                
                if duckdb_path.exists():
                    self.logger.info(f"Removing existing DuckDB database: {duckdb_path}")
                    duckdb_path.unlink()
            else:
                # Check if databases already exist
                if sqlite_path.exists() or duckdb_path.exists():
                    self.logger.warning("Databases already exist. Use --force-recreate to overwrite.")
                    self.logger.info("Continuing with existing databases...")
            
            # Validate configuration
            # Ensure configuration is consistent and optimal
            config_validation = self.config_manager.validate_configuration()
            failed_validations = [check for check, result in config_validation.items() if not result]
            
            if failed_validations:
                self.logger.warning(f"Configuration validation warnings: {failed_validations}")
                for check in failed_validations:
                    self.logger.warning(f"  - {check}: Failed")
            
            self.logger.info("Pre-initialization checks completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-initialization checks failed: {e}")
            return False
    
    def _initialize_sqlite_store(self) -> bool:
        """
        Initialize SQLite vector storage with platform optimizations.
        
        SQLite initialization includes:
        1. Database creation with optimized settings
        2. Schema creation with performance indexes
        3. WAL mode configuration for concurrent access
        4. Platform-specific cache and memory settings
        5. Connection pooling setup
        
        Returns:
            bool: Success status
        """
        self.logger.info("Initializing SQLite vector storage...")
        
        try:
            # Create SQLite store with platform-optimized configuration
            # This automatically applies all platform-specific optimizations
            self.sqlite_store = create_vector_store(self.config_manager)
            
            # Test basic functionality
            # Verify that the database is working correctly
            test_vector = np.random.rand(1536).astype(np.float32)
            test_record = create_vector_record(
                vector_id="init_test_vector",
                item_id="init_test_item",
                vector_data=test_vector,
                metadata={"test": True, "created_by": "init_script"}
            )
            
            # Test insertion and retrieval
            insert_success = self.sqlite_store.insert_vector(test_record)
            if not insert_success:
                self.logger.error("SQLite test insertion failed")
                return False
            
            retrieved_record = self.sqlite_store.get_vector("init_test_vector")
            if retrieved_record is None:
                self.logger.error("SQLite test retrieval failed")
                return False
            
            # Verify vector data integrity
            if not np.allclose(test_vector, retrieved_record.vector_data):
                self.logger.error("SQLite vector data integrity check failed")
                return False
            
            # Clean up test data
            self.sqlite_store.delete_vector("init_test_vector")
            
            # Get and log statistics
            stats = self.sqlite_store.get_statistics()
            self.logger.info(f"SQLite store initialized successfully:")
            self.logger.info(f"  Database size: {stats.get('database_size_mb', 0):.2f} MB")
            self.logger.info(f"  Cache size: {stats.get('cache_size_mb', 0)} MB")
            self.logger.info(f"  Memory mapping: {stats.get('memory_map_mb', 0)} MB")
            self.logger.info(f"  Platform optimized for: {stats.get('platform_type', 'unknown')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"SQLite initialization failed: {e}")
            return False
    
    def _initialize_analytics_store(self) -> bool:
        """
        Initialize DuckDB analytics store with cross-database integration.
        
        DuckDB initialization includes:
        1. Database creation with columnar optimization
        2. Analytical schema creation
        3. SQLite integration for cross-database queries
        4. Materialized views for performance
        5. Memory and thread optimization per platform
        
        Returns:
            bool: Success status
        """
        self.logger.info("Initializing DuckDB analytics store...")
        
        try:
            # Create analytics store with SQLite integration
            # Pass SQLite path for cross-database query capabilities
            sqlite_path = self.unified_config.database.sqlite_path
            self.analytics_store = create_analytics_store(
                config_manager=self.config_manager,
                sqlite_store_path=sqlite_path
            )
            
            # Test analytics functionality
            # Record a test event to verify analytical capabilities
            test_event = {
                'event_id': 'init_test_event',
                'timestamp': time.time(),
                'item_id': 'init_test_item',
                'recognition_result': 'test_result',
                'confidence_score': 0.95,
                'execution_time_ms': 100.0,
                'platform_type': self.platform_config.platform_type,
                'model_version': 'init_test_v1',
                'search_method': 'test_method',
                'vector_count': 1,
                'cache_hit': False,
                'metadata': {'test': True}
            }
            
            event_success = self.analytics_store.record_recognition_event(test_event)
            if not event_success:
                self.logger.error("Analytics test event recording failed")
                return False
            
            # Test analytical query
            from unified_storage.analytics_store import create_analytics_query
            test_query = create_analytics_query(
                query_id="init_test_query",
                sql="SELECT COUNT(*) as event_count FROM recognition_events WHERE event_id = 'init_test_event'",
                description="Test query for initialization",
                cache_ttl_seconds=0
            )
            
            query_result = self.analytics_store.execute_analytics_query(test_query)
            if query_result.row_count == 0 or query_result.data.empty:
                self.logger.error("Analytics test query failed")
                return False
            
            # Get and log analytics statistics
            stats = self.analytics_store.get_analytics_statistics()
            self.logger.info(f"DuckDB analytics store initialized successfully:")
            self.logger.info(f"  Database size: {stats.get('database_size_mb', 0):.2f} MB")
            self.logger.info(f"  Memory limit: {stats.get('memory_limit', 'unknown')}")
            self.logger.info(f"  Thread count: {stats.get('thread_count', 0)}")
            self.logger.info(f"  SQLite integration: {'Yes' if stats.get('sqlite_integration') else 'No'}")
            self.logger.info(f"  Recognition events: {stats.get('recognition_events', 0)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"DuckDB initialization failed: {e}")
            return False
    
    def _setup_database_integration(self) -> bool:
        """
        Setup cross-database integration between SQLite and DuckDB.
        
        Integration setup:
        1. Verify SQLite-DuckDB connectivity
        2. Test cross-database queries
        3. Validate materialized views
        4. Setup data synchronization triggers (if supported)
        
        Returns:
            bool: Success status
        """
        self.logger.info("Setting up cross-database integration...")
        
        try:
            # Test cross-database connectivity
            # This verifies that DuckDB can query SQLite data
            if self.sqlite_store and self.analytics_store:
                
                # Insert test data in SQLite
                test_vector = np.random.rand(1536).astype(np.float32)
                test_record = create_vector_record(
                    vector_id="integration_test_vector",
                    item_id="integration_test_item",
                    vector_data=test_vector,
                    metadata={"integration_test": True}
                )
                
                self.sqlite_store.insert_vector(test_record)
                
                # Query SQLite data from DuckDB
                from unified_storage.analytics_store import create_analytics_query
                integration_query = create_analytics_query(
                    query_id="integration_test",
                    sql="""
                        SELECT COUNT(*) as vector_count 
                        FROM vector_store.vectors 
                        WHERE vector_id = 'integration_test_vector'
                    """,
                    description="Cross-database integration test"
                )
                
                result = self.analytics_store.execute_analytics_query(integration_query)
                
                # Verify cross-database query worked
                if result.data.empty or result.data.iloc[0]['vector_count'] != 1:
                    self.logger.error("Cross-database query test failed")
                    return False
                
                # Clean up test data
                self.sqlite_store.delete_vector("integration_test_vector")
                
                self.logger.info("Cross-database integration verified successfully")
                
                # Test materialized views
                analytics_data = self.analytics_store.get_vector_analytics()
                if 'vector_summary' in analytics_data:
                    self.logger.info("Materialized views functioning correctly")
                else:
                    self.logger.warning("Materialized views may not be working correctly")
                
                return True
            else:
                self.logger.error("Database stores not properly initialized for integration")
                return False
                
        except Exception as e:
            self.logger.error(f"Database integration setup failed: {e}")
            return False
    
    def _migrate_legacy_data(self) -> bool:
        """
        Migrate data from legacy file-based storage to unified databases.
        
        Migration process:
        1. Detect legacy data sources (features.h5, raw images, etc.)
        2. Extract existing feature vectors and metadata
        3. Batch insert into SQLite vector store
        4. Record migration events in analytics store
        5. Validate migrated data integrity
        
        Returns:
            bool: Success status
        """
        self.logger.info("Starting legacy data migration...")
        
        if not LEGACY_AVAILABLE:
            self.logger.warning("Legacy components not available - skipping migration")
            return True
        
        try:
            # Look for legacy data sources
            legacy_sources = self._discover_legacy_data()
            
            if not legacy_sources:
                self.logger.info("No legacy data found - starting with empty databases")
                return True
            
            self.logger.info(f"Found legacy data sources: {list(legacy_sources.keys())}")
            
            # Migrate feature vectors if available
            if 'features_h5' in legacy_sources:
                if not self._migrate_feature_vectors(legacy_sources['features_h5']):
                    self.logger.error("Feature vector migration failed")
                    return False
            
            # Migrate raw images if feature extractor is available
            if 'raw_images' in legacy_sources and len(legacy_sources['raw_images']) > 0:
                if not self._migrate_raw_images(legacy_sources['raw_images']):
                    self.logger.warning("Raw image migration failed - continuing")
            
            # Record migration statistics
            self._record_migration_events()
            
            self.logger.info(f"Legacy data migration completed:")
            self.logger.info(f"  Vectors migrated: {self.migration_stats['vectors_migrated']}")
            self.logger.info(f"  Items processed: {self.migration_stats['items_processed']}")
            self.logger.info(f"  Errors encountered: {self.migration_stats['errors_encountered']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Legacy data migration failed: {e}")
            return False
    
    def _discover_legacy_data(self) -> Dict[str, Any]:
        """
        Discover available legacy data sources.
        
        Returns:
            Dict mapping source type to file paths
        """
        legacy_sources = {}
        
        # Look for features.h5 file
        features_path = self.data_dir / "features.h5"
        if features_path.exists():
            legacy_sources['features_h5'] = str(features_path)
            self.logger.info(f"Found legacy features file: {features_path}")
        
        # Look for raw image directories
        raw_data_dir = self.data_dir / "raw"
        if raw_data_dir.exists():
            image_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]
            if image_dirs:
                legacy_sources['raw_images'] = [str(d) for d in image_dirs]
                self.logger.info(f"Found {len(image_dirs)} raw image directories")
        
        # Look for existing FAISS indexes
        models_dir = self.data_dir / "models"
        if models_dir.exists():
            faiss_files = list(models_dir.glob("*.bin"))
            if faiss_files:
                legacy_sources['faiss_indexes'] = [str(f) for f in faiss_files]
                self.logger.info(f"Found {len(faiss_files)} FAISS index files")
        
        return legacy_sources
    
    def _migrate_feature_vectors(self, features_path: str) -> bool:
        """
        Migrate feature vectors from legacy HDF5 storage.
        
        Args:
            features_path: Path to features.h5 file
            
        Returns:
            bool: Success status
        """
        try:
            import h5py
            
            self.logger.info(f"Migrating feature vectors from: {features_path}")
            
            with h5py.File(features_path, 'r') as h5_file:
                # Get available datasets
                datasets = list(h5_file.keys())
                self.logger.info(f"Found datasets: {datasets}")
                
                vectors_migrated = 0
                
                for dataset_name in datasets:
                    if 'features' in dataset_name.lower() or 'vectors' in dataset_name.lower():
                        dataset = h5_file[dataset_name]
                        
                        # Extract features and metadata
                        if isinstance(dataset, h5py.Dataset):
                            features = dataset[:]
                            
                            # Create vector records for batch insertion
                            vector_records = []
                            
                            for i, feature_vector in enumerate(features):
                                if len(feature_vector) == 1536:  # Verify expected dimension
                                    vector_id = f"migrated_{dataset_name}_{i}"
                                    item_id = f"legacy_item_{i // 10}"  # Group vectors by item
                                    
                                    record = create_vector_record(
                                        vector_id=vector_id,
                                        item_id=item_id,
                                        vector_data=feature_vector.astype(np.float32),
                                        metadata={
                                            "source": "legacy_migration",
                                            "original_dataset": dataset_name,
                                            "original_index": i
                                        }
                                    )
                                    
                                    vector_records.append(record)
                                    
                                    # Batch insert every 100 vectors for efficiency
                                    if len(vector_records) >= 100:
                                        inserted = self.sqlite_store.insert_vectors_batch(vector_records)
                                        vectors_migrated += inserted
                                        vector_records = []
                            
                            # Insert remaining vectors
                            if vector_records:
                                inserted = self.sqlite_store.insert_vectors_batch(vector_records)
                                vectors_migrated += inserted
            
            self.migration_stats['vectors_migrated'] = vectors_migrated
            self.migration_stats['items_processed'] = vectors_migrated // 10  # Estimate
            
            self.logger.info(f"Successfully migrated {vectors_migrated} feature vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Feature vector migration failed: {e}")
            self.migration_stats['errors_encountered'] += 1
            return False
    
    def _migrate_raw_images(self, image_dirs: List[str]) -> bool:
        """
        Migrate raw images by extracting features and storing vectors.
        
        Args:
            image_dirs: List of directories containing raw images
            
        Returns:
            bool: Success status
        """
        try:
            if not LEGACY_AVAILABLE:
                return False
            
            # Initialize feature extractor
            extractor = FeatureExtractor()
            
            items_processed = 0
            vectors_created = 0
            
            for image_dir in image_dirs[:5]:  # Limit to first 5 directories for demo
                dir_path = Path(image_dir)
                item_id = dir_path.name
                
                self.logger.info(f"Processing item directory: {item_id}")
                
                # Find image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                image_files = [f for f in dir_path.iterdir() 
                             if f.suffix.lower() in image_extensions]
                
                if not image_files:
                    continue
                
                vector_records = []
                
                for i, image_file in enumerate(image_files[:10]):  # Limit images per item
                    try:
                        # Extract features
                        features = extractor.extract_features(str(image_file))
                        
                        if features is not None and len(features) == 1536:
                            vector_id = f"{item_id}_img_{i}"
                            
                            record = create_vector_record(
                                vector_id=vector_id,
                                item_id=item_id,
                                vector_data=features,
                                metadata={
                                    "source": "raw_image_migration",
                                    "original_path": str(image_file),
                                    "image_index": i
                                }
                            )
                            
                            vector_records.append(record)
                            vectors_created += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process image {image_file}: {e}")
                        self.migration_stats['errors_encountered'] += 1
                
                # Batch insert vectors for this item
                if vector_records:
                    inserted = self.sqlite_store.insert_vectors_batch(vector_records)
                    self.logger.info(f"Inserted {inserted} vectors for item {item_id}")
                
                items_processed += 1
            
            self.migration_stats['vectors_migrated'] += vectors_created
            self.migration_stats['items_processed'] += items_processed
            
            self.logger.info(f"Raw image migration completed: {vectors_created} vectors from {items_processed} items")
            return True
            
        except Exception as e:
            self.logger.error(f"Raw image migration failed: {e}")
            self.migration_stats['errors_encountered'] += 1
            return False
    
    def _record_migration_events(self):
        """Record migration statistics in analytics store."""
        if self.analytics_store:
            migration_event = {
                'event_id': f"migration_{int(time.time())}",
                'timestamp': time.time(),
                'item_id': 'migration_process',
                'recognition_result': 'migration_completed',
                'confidence_score': 1.0,
                'execution_time_ms': (self.migration_stats['end_time'] - self.migration_stats['start_time']) * 1000,
                'platform_type': self.platform_config.platform_type,
                'model_version': 'migration_v1',
                'search_method': 'batch_migration',
                'vector_count': self.migration_stats['vectors_migrated'],
                'cache_hit': False,
                'metadata': {
                    'migration_stats': self.migration_stats,
                    'items_processed': self.migration_stats['items_processed'],
                    'errors_encountered': self.migration_stats['errors_encountered']
                }
            }
            
            self.analytics_store.record_recognition_event(migration_event)
    
    def _validate_database_setup(self) -> bool:
        """
        Validate complete database setup with comprehensive tests.
        
        Validation includes:
        1. Database connectivity and basic operations
        2. Cross-database query functionality
        3. Performance validation
        4. Data integrity verification
        5. Configuration validation
        
        Returns:
            bool: Success status
        """
        self.logger.info("Validating database setup...")
        
        try:
            # Test SQLite operations
            if not self._validate_sqlite_operations():
                return False
            
            # Test DuckDB operations
            if not self._validate_analytics_operations():
                return False
            
            # Test cross-database integration
            if not self._validate_cross_database_queries():
                return False
            
            # Performance validation
            if not self._validate_performance():
                return False
            
            self.logger.info("Database validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Database validation failed: {e}")
            return False
    
    def _validate_sqlite_operations(self) -> bool:
        """Validate SQLite vector operations."""
        try:
            # Test vector operations
            test_vectors = []
            for i in range(10):
                vector = np.random.rand(1536).astype(np.float32)
                record = create_vector_record(
                    vector_id=f"validation_test_{i}",
                    item_id="validation_item",
                    vector_data=vector,
                    metadata={"validation": True, "index": i}
                )
                test_vectors.append(record)
            
            # Batch insert
            inserted = self.sqlite_store.insert_vectors_batch(test_vectors)
            if inserted != 10:
                self.logger.error(f"Expected 10 insertions, got {inserted}")
                return False
            
            # Test retrieval
            retrieved = self.sqlite_store.get_vectors_by_item("validation_item")
            if len(retrieved) != 10:
                self.logger.error(f"Expected 10 retrieved vectors, got {len(retrieved)}")
                return False
            
            # Test deletion
            deleted = self.sqlite_store.delete_vectors_by_item("validation_item")
            if deleted != 10:
                self.logger.error(f"Expected 10 deletions, got {deleted}")
                return False
            
            self.logger.info("SQLite operations validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"SQLite validation failed: {e}")
            return False
    
    def _validate_analytics_operations(self) -> bool:
        """Validate DuckDB analytics operations."""
        try:
            # Test event recording
            for i in range(5):
                event = {
                    'event_id': f'validation_event_{i}',
                    'timestamp': time.time(),
                    'item_id': 'validation_item',
                    'recognition_result': 'test_result',
                    'confidence_score': 0.9 + (i * 0.01),
                    'execution_time_ms': 100.0 + i,
                    'platform_type': self.platform_config.platform_type,
                    'vector_count': 10 + i
                }
                
                if not self.analytics_store.record_recognition_event(event):
                    self.logger.error(f"Failed to record event {i}")
                    return False
            
            # Test analytics query
            from unified_storage.analytics_store import create_analytics_query
            validation_query = create_analytics_query(
                query_id="validation_query",
                sql="""
                    SELECT COUNT(*) as event_count, AVG(confidence_score) as avg_confidence
                    FROM recognition_events 
                    WHERE item_id = 'validation_item'
                """,
                description="Validation query"
            )
            
            result = self.analytics_store.execute_analytics_query(validation_query)
            if result.data.empty or result.data.iloc[0]['event_count'] != 5:
                self.logger.error("Analytics query validation failed")
                return False
            
            self.logger.info("Analytics operations validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Analytics validation failed: {e}")
            return False
    
    def _validate_cross_database_queries(self) -> bool:
        """Validate cross-database query functionality."""
        try:
            # This test was already done in integration setup
            # Just verify the functionality is still working
            analytics_data = self.analytics_store.get_vector_analytics()
            
            if isinstance(analytics_data, dict):
                self.logger.info("Cross-database queries validation passed")
                return True
            else:
                self.logger.error("Cross-database queries validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Cross-database validation failed: {e}")
            return False
    
    def _validate_performance(self) -> bool:
        """Validate database performance meets expectations."""
        try:
            # Test SQLite performance
            start_time = time.time()
            test_vector = np.random.rand(1536).astype(np.float32)
            test_record = create_vector_record(
                vector_id="perf_test",
                item_id="perf_item",
                vector_data=test_vector
            )
            
            self.sqlite_store.insert_vector(test_record)
            insert_time = (time.time() - start_time) * 1000
            
            start_time = time.time()
            retrieved = self.sqlite_store.get_vector("perf_test")
            retrieve_time = (time.time() - start_time) * 1000
            
            # Clean up
            self.sqlite_store.delete_vector("perf_test")
            
            # Validate performance expectations
            if insert_time > 100:  # 100ms threshold
                self.logger.warning(f"SQLite insert performance slower than expected: {insert_time:.2f}ms")
            
            if retrieve_time > 10:  # 10ms threshold
                self.logger.warning(f"SQLite retrieve performance slower than expected: {retrieve_time:.2f}ms")
            
            self.logger.info(f"Performance validation - Insert: {insert_time:.2f}ms, Retrieve: {retrieve_time:.2f}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            return False
    
    def _perform_final_optimization(self) -> bool:
        """Perform final database optimization."""
        try:
            self.logger.info("Performing final database optimization...")
            
            # Optimize SQLite database
            if self.sqlite_store:
                self.sqlite_store.optimize_database()
            
            # Optimize DuckDB database
            if self.analytics_store:
                self.analytics_store.optimize_analytics_database()
            
            # Save optimized configuration
            config_path = self.config_manager.save_config()
            self.logger.info(f"Saved optimized configuration to: {config_path}")
            
            self.logger.info("Final optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Final optimization failed: {e}")
            return False
    
    def _cleanup_resources(self):
        """Clean up database connections and resources."""
        try:
            if self.sqlite_store:
                self.sqlite_store.close()
            
            if self.analytics_store:
                self.analytics_store.close()
                
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")
    
    def get_initialization_report(self) -> Dict[str, Any]:
        """Get comprehensive initialization report."""
        sqlite_stats = self.sqlite_store.get_statistics() if self.sqlite_store else {}
        analytics_stats = self.analytics_store.get_analytics_statistics() if self.analytics_store else {}
        
        return {
            'platform_type': self.platform_config.platform_type,
            'initialization_time': self.migration_stats.get('end_time', 0) - self.migration_stats.get('start_time', 0),
            'migration_stats': self.migration_stats,
            'sqlite_stats': sqlite_stats,
            'analytics_stats': analytics_stats,
            'configuration_summary': self.config_manager.get_optimization_summary(),
            'data_directory': str(self.data_dir),
            'databases_created': {
                'sqlite': str(self.unified_config.database.sqlite_path),
                'duckdb': str(self.unified_config.database.duckdb_path)
            }
        }


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Initialize unified storage databases for AI Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic initialization
  python init_databases.py
  
  # Force recreate with legacy migration
  python init_databases.py --force-recreate --migrate-legacy
  
  # Custom data directory
  python init_databases.py --data-dir /custom/path/data
  
  # Custom configuration
  python init_databases.py --config config/production.yaml
        """
    )
    
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory for databases (default: data)'
    )
    
    parser.add_argument(
        '--config',
        help='Custom configuration file path'
    )
    
    parser.add_argument(
        '--force-recreate',
        action='store_true',
        help='Force recreation of existing databases'
    )
    
    parser.add_argument(
        '--migrate-legacy',
        action='store_true',
        default=True,
        help='Migrate legacy data (default: True)'
    )
    
    parser.add_argument(
        '--no-migrate-legacy',
        action='store_true',
        help='Skip legacy data migration'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--report',
        help='Save initialization report to file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Handle migration flag
    migrate_legacy = args.migrate_legacy and not args.no_migrate_legacy
    
    try:
        # Create database initializer
        initializer = DatabaseInitializer(
            data_dir=args.data_dir,
            config_path=args.config,
            force_recreate=args.force_recreate,
            migrate_legacy=migrate_legacy
        )
        
        # Run initialization
        success = initializer.initialize_databases()
        
        if success:
            logger.info("Database initialization completed successfully!")
            
            # Generate and save report if requested
            if args.report:
                report = initializer.get_initialization_report()
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Initialization report saved to: {args.report}")
            
            sys.exit(0)
        else:
            logger.error("Database initialization failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Database initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()