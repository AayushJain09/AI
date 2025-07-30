#!/usr/bin/env python3
"""
Migration Orchestrator

Coordinates the complete migration from legacy scattered file architecture
to the unified storage system with comprehensive validation and reporting.

MIGRATION WORKFLOW:
1. Pre-migration Analysis: Analyze legacy data structure and compatibility
2. Data Preparation: Prepare unified storage system
3. Feature Migration: Migrate features from HDF5 to SQLite
4. Index Migration: Consolidate FAISS indices
5. Metadata Migration: Extract and store metadata
6. Post-migration Validation: Comprehensive validation and reporting
7. Performance Optimization: Final system optimization
8. Migration Report: Generate comprehensive migration report

ORCHESTRATION FEATURES:
- Automated workflow execution
- Error handling and recovery
- Progress tracking and reporting
- Rollback capabilities
- Performance monitoring
- Comprehensive logging

The orchestrator ensures a smooth, reliable migration with full
data integrity validation and performance optimization.
"""

import os
import sys
import json
import logging
import time
import shutil
from pathlib import Path

# Fix OpenMP duplicate library issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import argparse

# Add src to path for unified storage imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unified_storage import create_unified_store

# Import migration modules
try:
    from .migrate_features import FeatureMigrator, FeatureMigrationStats
    from .migrate_faiss_indices import FAISSIndexMigrator, IndexMigrationStats
    from .migrate_metadata import MetadataMigrator, MetadataMigrationStats
    from .validate_migration import MigrationValidator, ValidationResults
except ImportError:
    # Handle direct execution
    pass

# Always use direct imports for execution
from migrate_features import FeatureMigrator, FeatureMigrationStats
from migrate_faiss_indices import FAISSIndexMigrator, IndexMigrationStats
from migrate_metadata import MetadataMigrator, MetadataMigrationStats
from validate_migration import MigrationValidator, ValidationResults


@dataclass
class MigrationPlan:
    """Migration execution plan with all parameters."""
    # Source paths
    legacy_features_path: Optional[str] = None
    legacy_indices_dir: Optional[str] = None
    legacy_data_dir: Optional[str] = None
    
    # Target configuration
    target_data_dir: str = "data"
    backup_dir: Optional[str] = None
    
    # Migration options
    migrate_features: bool = True
    migrate_indices: bool = True
    migrate_metadata: bool = True
    run_validation: bool = True
    create_backup: bool = True
    
    # Processing options
    batch_size: int = 100
    validation_sample_size: int = 100
    enable_content_analysis: bool = True
    rebuild_optimized_indices: bool = True
    
    # Output options
    generate_report: bool = True
    report_output_path: Optional[str] = None
    verbose_logging: bool = False


@dataclass
class MigrationResults:
    """Comprehensive migration results."""
    # Overall status
    migration_successful: bool = False
    migration_start_time: float = field(default_factory=time.time)
    migration_end_time: float = 0.0
    total_migration_time: float = 0.0
    
    # Component results
    feature_migration_stats: Optional[FeatureMigrationStats] = None
    index_migration_stats: Optional[IndexMigrationStats] = None
    metadata_migration_stats: Optional[MetadataMigrationStats] = None
    validation_results: Optional[ValidationResults] = None
    
    # Error tracking
    errors_encountered: List[str] = field(default_factory=list)
    warnings_generated: List[str] = field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Migration summary
    summary: Dict[str, Any] = field(default_factory=dict)


class MigrationOrchestrator:
    """
    Orchestrates the complete migration workflow.
    
    Coordinates all migration components, handles errors, tracks progress,
    and provides comprehensive reporting.
    """
    
    def __init__(self, migration_plan: MigrationPlan):
        """
        Initialize migration orchestrator.
        
        Args:
            migration_plan: Complete migration plan with all parameters
        """
        self.plan = migration_plan
        self.results = MigrationResults()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Validate migration plan
        self._validate_migration_plan()
        
        # Initialize unified store (will be created during migration)
        self.unified_store = None
    
    def _setup_logging(self):
        """Setup comprehensive logging for migration orchestration."""
        # Create logs directory
        log_dir = Path(self.plan.target_data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging configuration
        log_level = logging.DEBUG if self.plan.verbose_logging else logging.INFO
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
        
        # File handler for migration log
        migration_log_path = log_dir / f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(migration_log_path)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        print(f"📝 Migration logging initialized: {migration_log_path}")
    
    def _validate_migration_plan(self):
        """Validate migration plan parameters."""
        issues = []
        
        # Check source paths
        if self.plan.migrate_features and self.plan.legacy_features_path:
            if not Path(self.plan.legacy_features_path).exists():
                issues.append(f"Legacy features file not found: {self.plan.legacy_features_path}")
        
        if self.plan.migrate_indices and self.plan.legacy_indices_dir:
            if not Path(self.plan.legacy_indices_dir).exists():
                issues.append(f"Legacy indices directory not found: {self.plan.legacy_indices_dir}")
        
        if self.plan.migrate_metadata and self.plan.legacy_data_dir:
            if not Path(self.plan.legacy_data_dir).exists():
                issues.append(f"Legacy data directory not found: {self.plan.legacy_data_dir}")
        
        # Check target directory
        target_dir = Path(self.plan.target_data_dir)
        if target_dir.exists() and any(target_dir.iterdir()):
            self.results.warnings_generated.append(
                f"Target directory not empty: {self.plan.target_data_dir}"
            )
        
        # Check backup directory
        if self.plan.create_backup and not self.plan.backup_dir:
            self.plan.backup_dir = f"{self.plan.target_data_dir}_backup_{int(time.time())}"
        
        if issues:
            raise ValueError(f"Migration plan validation failed: {issues}")
        
        print("✅ Migration plan validated successfully")
    
    def create_backup(self) -> bool:
        """Create backup of existing data before migration."""
        if not self.plan.create_backup:
            return True
        
        try:
            self.logger.info("💾 Creating data backup...")
            
            target_path = Path(self.plan.target_data_dir)
            backup_path = Path(self.plan.backup_dir)
            
            if target_path.exists():
                # Create backup directory
                backup_path.mkdir(parents=True, exist_ok=True)
                
                # Copy existing data
                for item in target_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, backup_path / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, backup_path / item.name, dirs_exist_ok=True)
                
                backup_size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
                backup_size_mb = backup_size / (1024 * 1024)
                
                self.logger.info(f"✅ Backup created: {backup_path} ({backup_size_mb:.1f}MB)")
                return True
            else:
                self.logger.info("ℹ️ No existing data to backup")
                return True
        
        except Exception as e:
            self.logger.error(f"❌ Backup creation failed: {e}")
            self.results.errors_encountered.append(f"Backup creation error: {e}")
            return False
    
    def analyze_legacy_data(self) -> Dict[str, Any]:
        """Analyze legacy data to plan migration strategy."""
        self.logger.info("🔍 Analyzing legacy data structure...")
        
        analysis = {
            'features_analysis': None,
            'indices_analysis': None,
            'metadata_analysis': None,
            'migration_strategy': {},
            'estimated_time': 0,
            'compatibility_issues': []
        }
        
        try:
            # Analyze features if available
            if self.plan.legacy_features_path and Path(self.plan.legacy_features_path).exists():
                try:
                    from migrate_features import FeatureMigrator
                    temp_store = create_unified_store(self.plan.target_data_dir)
                    migrator = FeatureMigrator(self.plan.legacy_features_path, temp_store)
                    analysis['features_analysis'] = migrator.analyze_legacy_features()
                    temp_store.close()
                except Exception as e:
                    analysis['compatibility_issues'].append(f"Features analysis error: {e}")
            
            # Analyze indices if available
            if self.plan.legacy_indices_dir and Path(self.plan.legacy_indices_dir).exists():
                try:
                    from migrate_faiss_indices import FAISSIndexMigrator
                    temp_store = create_unified_store(self.plan.target_data_dir)
                    migrator = FAISSIndexMigrator(self.plan.legacy_indices_dir, temp_store)
                    analysis['indices_analysis'] = migrator.discover_legacy_indices()
                    temp_store.close()
                except Exception as e:
                    analysis['compatibility_issues'].append(f"Indices analysis error: {e}")
            
            # Analyze metadata if available
            if self.plan.legacy_data_dir and Path(self.plan.legacy_data_dir).exists():
                try:
                    from migrate_metadata import MetadataMigrator
                    from unified_storage.sqlite_store import create_vector_store
                    temp_vector_store = create_vector_store(":memory:")  # Temporary in-memory store
                    migrator = MetadataMigrator(self.plan.legacy_data_dir, temp_vector_store)
                    analysis['metadata_analysis'] = migrator.discover_directory_structure()
                    temp_vector_store.close()
                except Exception as e:
                    analysis['compatibility_issues'].append(f"Metadata analysis error: {e}")
            
            # Determine migration strategy
            if analysis['features_analysis']:
                total_images = analysis['features_analysis'].get('total_images', 0)
                analysis['migration_strategy']['batch_processing'] = total_images > 1000
                analysis['estimated_time'] += total_images * 0.1  # ~0.1s per image
            
            if analysis['indices_analysis']:
                total_indices = len(analysis['indices_analysis'])
                analysis['estimated_time'] += total_indices * 30  # ~30s per index
            
            if analysis['metadata_analysis']:
                total_files = analysis['metadata_analysis'].get('total_files', 0)
                analysis['estimated_time'] += total_files * 0.05  # ~0.05s per file
            
            self.logger.info(f"📊 Legacy Data Analysis:")
            if analysis['features_analysis']:
                self.logger.info(f"   Features: {analysis['features_analysis'].get('total_images', 0)} images")
            if analysis['indices_analysis']:
                self.logger.info(f"   Indices: {len(analysis['indices_analysis'])} files")
            if analysis['metadata_analysis']:
                self.logger.info(f"   Metadata: {analysis['metadata_analysis'].get('total_files', 0)} files")
            self.logger.info(f"   Estimated time: {analysis['estimated_time']:.1f}s")
            
            if analysis['compatibility_issues']:
                self.logger.warning("⚠️ Compatibility issues found:")
                for issue in analysis['compatibility_issues']:
                    self.logger.warning(f"   - {issue}")
        
        except Exception as e:
            self.logger.error(f"❌ Legacy data analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def migrate_features(self) -> bool:
        """Execute feature migration."""
        if not self.plan.migrate_features or not self.plan.legacy_features_path:
            self.logger.info("⏭️ Skipping feature migration (not requested or no source)")
            return True
        
        try:
            self.logger.info("🔄 Starting feature migration...")
            
            from migrate_features import FeatureMigrator
            migrator = FeatureMigrator(
                legacy_features_path=self.plan.legacy_features_path,
                unified_store=self.unified_store,
                batch_size=self.plan.batch_size
            )
            
            # Run migration
            stats = migrator.migrate_all_features()
            self.results.feature_migration_stats = stats
            
            # Verify migration
            verification = migrator.verify_migration()
            
            if not verification.get('migration_complete', False):
                self.results.errors_encountered.append("Feature migration verification failed")
                return False
            
            self.logger.info(f"✅ Feature migration completed: {stats.images_migrated} images")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Feature migration failed: {e}")
            self.results.errors_encountered.append(f"Feature migration error: {e}")
            return False
    
    def migrate_indices(self) -> bool:
        """Execute FAISS index migration."""
        if not self.plan.migrate_indices or not self.plan.legacy_indices_dir:
            self.logger.info("⏭️ Skipping index migration (not requested or no source)")
            return True
        
        try:
            self.logger.info("🔄 Starting index migration...")
            
            from migrate_faiss_indices import FAISSIndexMigrator
            migrator = FAISSIndexMigrator(
                legacy_indices_dir=self.plan.legacy_indices_dir,
                unified_store=self.unified_store,
                rebuild_optimized=self.plan.rebuild_optimized_indices
            )
            
            # Run migration
            stats = migrator.migrate_all_indices()
            self.results.index_migration_stats = stats
            
            # Verify migration
            verification = migrator.verify_migration()
            
            if not verification.get('migration_complete', False):
                self.results.errors_encountered.append("Index migration verification failed")
                return False
            
            self.logger.info(f"✅ Index migration completed: {stats.total_vectors_migrated} vectors")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Index migration failed: {e}")
            self.results.errors_encountered.append(f"Index migration error: {e}")
            return False
    
    def migrate_metadata(self) -> bool:
        """Execute metadata migration."""
        if not self.plan.migrate_metadata or not self.plan.legacy_data_dir:
            self.logger.info("⏭️ Skipping metadata migration (not requested or no source)")
            return True
        
        try:
            self.logger.info("🔄 Starting metadata migration...")
            
            from migrate_metadata import MetadataMigrator
            from unified_storage.sqlite_store import create_vector_store
            vector_store = create_vector_store(str(Path(self.plan.target_data_dir) / "recognition.db"))
            migrator = MetadataMigrator(
                legacy_data_dir=self.plan.legacy_data_dir,
                vector_store=vector_store,
                enable_content_analysis=self.plan.enable_content_analysis
            )
            
            # Run migration
            stats = migrator.migrate_all_metadata()
            self.results.metadata_migration_stats = stats
            
            # Verify migration
            verification = migrator.verify_metadata_migration()
            
            if not verification.get('migration_complete', False):
                self.results.errors_encountered.append("Metadata migration verification failed")
                return False
            
            self.logger.info(f"✅ Metadata migration completed: {stats.metadata_records_created} records")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Metadata migration failed: {e}")
            self.results.errors_encountered.append(f"Metadata migration error: {e}")
            return False
    
    def validate_migration(self) -> bool:
        """Execute comprehensive migration validation."""
        if not self.plan.run_validation:
            self.logger.info("⏭️ Skipping migration validation (not requested)")
            return True
        
        try:
            self.logger.info("🔍 Starting migration validation...")
            
            from validate_migration import MigrationValidator
            validator = MigrationValidator(
                unified_store=self.unified_store,
                legacy_features_path=self.plan.legacy_features_path,
                legacy_indices_dir=self.plan.legacy_indices_dir,
                legacy_data_dir=self.plan.legacy_data_dir,
                validation_sample_size=self.plan.validation_sample_size
            )
            
            # Run comprehensive validation
            results = validator.run_comprehensive_validation()
            self.results.validation_results = results
            
            if not results.validation_passed:
                self.logger.error("❌ Migration validation failed")
                self.results.errors_encountered.extend(results.critical_issues)
                return False
            
            self.logger.info("✅ Migration validation passed")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Migration validation failed: {e}")
            self.results.errors_encountered.append(f"Validation error: {e}")
            return False
    
    def optimize_system(self) -> bool:
        """Perform final system optimization."""
        try:
            self.logger.info("⚡ Optimizing unified storage system...")
            
            # Optimize search index
            optimization_result = self.unified_store.optimize_index()
            if optimization_result.get('status') == 'failed':
                self.results.warnings_generated.append("Index optimization failed")
            
            # Collect performance metrics
            stats = self.unified_store.get_statistics()
            self.results.performance_metrics = {
                'total_images_stored': stats.total_images_stored,
                'total_searches_performed': stats.total_searches_performed,
                'avg_extraction_time_ms': stats.avg_extraction_time_ms,
                'peak_memory_usage_mb': stats.peak_memory_usage_mb,
                'database_size_mb': stats.database_size_mb,
                'platform_type': stats.platform_type,
                'device_acceleration': stats.device_acceleration
            }
            
            self.logger.info("✅ System optimization completed")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ System optimization failed: {e}")
            self.results.warnings_generated.append(f"Optimization error: {e}")
            return True  # Non-critical error
    
    def generate_migration_report(self) -> str:
        """Generate comprehensive migration report."""
        if not self.plan.generate_report:
            return ""
        
        try:
            # Determine report path
            if self.plan.report_output_path:
                report_path = self.plan.report_output_path
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_path = str(Path(self.plan.target_data_dir) / f"migration_report_{timestamp}.json")
            
            # Compile comprehensive report
            report_data = {
                'migration_summary': {
                    'successful': self.results.migration_successful,
                    'start_time': datetime.fromtimestamp(self.results.migration_start_time).isoformat(),
                    'end_time': datetime.fromtimestamp(self.results.migration_end_time).isoformat(),
                    'total_time_seconds': self.results.total_migration_time,
                    'errors_count': len(self.results.errors_encountered),
                    'warnings_count': len(self.results.warnings_generated)
                },
                'migration_plan': {
                    'legacy_features_path': self.plan.legacy_features_path,
                    'legacy_indices_dir': self.plan.legacy_indices_dir,
                    'legacy_data_dir': self.plan.legacy_data_dir,
                    'target_data_dir': self.plan.target_data_dir,
                    'components_migrated': {
                        'features': self.plan.migrate_features,
                        'indices': self.plan.migrate_indices,
                        'metadata': self.plan.migrate_metadata
                    }
                },
                'component_results': {},
                'performance_metrics': self.results.performance_metrics,
                'issues': {
                    'errors': self.results.errors_encountered,
                    'warnings': self.results.warnings_generated
                }
            }
            
            # Add component-specific results
            if self.results.feature_migration_stats:
                report_data['component_results']['features'] = {
                    'images_migrated': self.results.feature_migration_stats.images_migrated,
                    'images_failed': self.results.feature_migration_stats.images_failed,
                    'migration_time_seconds': self.results.feature_migration_stats.migration_time_seconds,
                    'data_size_mb': self.results.feature_migration_stats.data_size_mb
                }
            
            if self.results.index_migration_stats:
                report_data['component_results']['indices'] = {
                    'indices_processed': self.results.index_migration_stats.indices_processed,
                    'vectors_migrated': self.results.index_migration_stats.total_vectors_migrated,
                    'migration_time_seconds': self.results.index_migration_stats.migration_time_seconds,
                    'search_performance_ms': self.results.index_migration_stats.search_performance_ms
                }
            
            if self.results.metadata_migration_stats:
                report_data['component_results']['metadata'] = {
                    'files_discovered': self.results.metadata_migration_stats.files_discovered,
                    'images_processed': self.results.metadata_migration_stats.images_processed,
                    'metadata_records_created': self.results.metadata_migration_stats.metadata_records_created,
                    'migration_time_seconds': self.results.metadata_migration_stats.migration_time_seconds
                }
            
            if self.results.validation_results:
                report_data['validation_results'] = {
                    'validation_passed': self.results.validation_results.validation_passed,
                    'data_integrity_score': self.results.validation_results.data_integrity_score,
                    'feature_accuracy_score': self.results.validation_results.feature_accuracy_score,
                    'search_functionality_score': self.results.validation_results.search_functionality_score,
                    'metadata_completeness_score': self.results.validation_results.metadata_completeness_score,
                    'system_health_score': self.results.validation_results.system_health_score
                }
            
            # Write report
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"📋 Migration report generated: {report_path}")
            return report_path
        
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return ""
    
    def execute_migration(self) -> MigrationResults:
        """Execute the complete migration workflow."""
        self.logger.info("🚀 Starting migration orchestration...")
        self.results.migration_start_time = time.time()
        
        try:
            # Phase 1: Pre-migration analysis
            self.logger.info("📊 Phase 1: Pre-migration Analysis")
            analysis = self.analyze_legacy_data()
            
            # Phase 2: Backup creation
            self.logger.info("💾 Phase 2: Backup Creation")
            if not self.create_backup():
                raise RuntimeError("Backup creation failed")
            
            # Phase 3: Initialize unified storage
            self.logger.info("🔧 Phase 3: Initialize Unified Storage")
            self.unified_store = create_unified_store(self.plan.target_data_dir)
            
            # Phase 4: Data migration
            migration_success = True
            
            self.logger.info("📥 Phase 4: Data Migration")
            
            # Step 4a: Migrate features
            if not self.migrate_features():
                migration_success = False
                if not self.plan.migrate_indices and not self.plan.migrate_metadata:
                    raise RuntimeError("Feature migration failed and no other sources available")
            
            # Step 4b: Migrate indices
            if not self.migrate_indices():
                migration_success = False
                self.results.warnings_generated.append("Index migration failed")
            
            # Step 4c: Migrate metadata
            if not self.migrate_metadata():
                migration_success = False
                self.results.warnings_generated.append("Metadata migration failed")
            
            if not migration_success and len(self.results.errors_encountered) > 0:
                raise RuntimeError("Critical migration failures occurred")
            
            # Phase 5: System optimization
            self.logger.info("⚡ Phase 5: System Optimization")
            self.optimize_system()
            
            # Phase 6: Migration validation
            self.logger.info("🔍 Phase 6: Migration Validation")
            if not self.validate_migration():
                raise RuntimeError("Migration validation failed")
            
            # Phase 7: Final reporting
            self.logger.info("📋 Phase 7: Generate Migration Report")
            report_path = self.generate_migration_report()
            
            # Migration completed successfully
            self.results.migration_successful = True
            self.logger.info("✅ Migration orchestration completed successfully!")
        
        except Exception as e:
            self.logger.error(f"❌ Migration orchestration failed: {e}")
            self.results.migration_successful = False
            self.results.errors_encountered.append(f"Orchestration error: {e}")
        
        finally:
            # Cleanup
            if self.unified_store:
                self.unified_store.close()
            
            # Calculate total time
            self.results.migration_end_time = time.time()
            self.results.total_migration_time = (
                self.results.migration_end_time - self.results.migration_start_time
            )
            
            # Log final summary
            self._log_final_summary()
        
        return self.results
    
    def _log_final_summary(self):
        """Log final migration summary."""
        self.logger.info("=" * 60)
        self.logger.info("MIGRATION ORCHESTRATION SUMMARY")
        self.logger.info("=" * 60)
        
        status = "✅ SUCCESS" if self.results.migration_successful else "❌ FAILED"
        self.logger.info(f"Status: {status}")
        self.logger.info(f"Total Time: {self.results.total_migration_time:.1f}s")
        
        # Component summaries
        if self.results.feature_migration_stats:
            stats = self.results.feature_migration_stats
            self.logger.info(f"Features: {stats.images_migrated} migrated, {stats.images_failed} failed")
        
        if self.results.index_migration_stats:
            stats = self.results.index_migration_stats
            self.logger.info(f"Indices: {stats.total_vectors_migrated} vectors migrated")
        
        if self.results.metadata_migration_stats:
            stats = self.results.metadata_migration_stats
            self.logger.info(f"Metadata: {stats.metadata_records_created} records created")
        
        if self.results.validation_results:
            results = self.results.validation_results
            self.logger.info(f"Validation: {'PASSED' if results.validation_passed else 'FAILED'}")
        
        # Issues summary
        if self.results.errors_encountered:
            self.logger.error(f"Errors ({len(self.results.errors_encountered)}):")
            for error in self.results.errors_encountered:
                self.logger.error(f"  - {error}")
        
        if self.results.warnings_generated:
            self.logger.warning(f"Warnings ({len(self.results.warnings_generated)}):")
            for warning in self.results.warnings_generated[:5]:  # Show first 5
                self.logger.warning(f"  - {warning}")
        
        self.logger.info("=" * 60)


def create_migration_plan_from_args(args) -> MigrationPlan:
    """Create migration plan from command line arguments."""
    return MigrationPlan(
        legacy_features_path=args.legacy_features,
        legacy_indices_dir=args.legacy_indices,
        legacy_data_dir=args.legacy_data,
        target_data_dir=args.target_dir,
        backup_dir=args.backup_dir,
        migrate_features=not args.no_features,
        migrate_indices=not args.no_indices,
        migrate_metadata=not args.no_metadata,
        run_validation=not args.no_validation,
        create_backup=not args.no_backup,
        batch_size=args.batch_size,
        validation_sample_size=args.validation_sample_size,
        enable_content_analysis=not args.no_content_analysis,
        rebuild_optimized_indices=not args.no_rebuild_indices,
        generate_report=not args.no_report,
        report_output_path=args.report_output,
        verbose_logging=args.verbose
    )


def main():
    """Main entry point for migration orchestrator."""
    parser = argparse.ArgumentParser(
        description='Orchestrate complete migration to unified storage system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Migration Workflow:
  1. Pre-migration analysis of legacy data
  2. Backup creation (optional)
  3. Unified storage system initialization
  4. Data migration (features, indices, metadata)
  5. System optimization and indexing
  6. Comprehensive validation
  7. Migration report generation

Examples:
  # Full migration with all components
  python orchestrator.py --legacy-features data/features.h5 --legacy-indices data/models --legacy-data data/raw
  
  # Migration with specific components only
  python orchestrator.py --legacy-features data/features.h5 --no-indices --no-metadata
  
  # Migration with custom settings
  python orchestrator.py --legacy-data data/raw --batch-size 50 --no-backup --verbose
        '''
    )
    
    # Source data arguments
    parser.add_argument('--legacy-features', type=str,
                       help='Path to legacy features.h5 file')
    parser.add_argument('--legacy-indices', type=str,
                       help='Path to legacy FAISS indices directory')
    parser.add_argument('--legacy-data', type=str,
                       help='Path to legacy data directory')
    
    # Target configuration
    parser.add_argument('--target-dir', type=str, default='data',
                       help='Target directory for unified storage (default: data)')
    parser.add_argument('--backup-dir', type=str,
                       help='Backup directory path (auto-generated if not specified)')
    
    # Migration options
    parser.add_argument('--no-features', action='store_true',
                       help='Skip feature migration')
    parser.add_argument('--no-indices', action='store_true',
                       help='Skip index migration')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Skip metadata migration')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip migration validation')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backup creation')
    
    # Processing options
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing (default: 100)')
    parser.add_argument('--validation-sample-size', type=int, default=100,
                       help='Sample size for validation tests (default: 100)')
    parser.add_argument('--no-content-analysis', action='store_true',
                       help='Disable image content analysis')
    parser.add_argument('--no-rebuild-indices', action='store_true',
                       help='Skip index rebuilding/optimization')
    
    # Output options
    parser.add_argument('--no-report', action='store_true',
                       help='Skip migration report generation')
    parser.add_argument('--report-output', type=str,
                       help='Migration report output path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Analysis only mode
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze legacy data without migrating')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.legacy_features, args.legacy_indices, args.legacy_data]):
        parser.error("At least one legacy data source must be specified")
    
    try:
        if args.analyze_only:
            # Analysis-only mode
            print("🔍 Analyzing legacy data structure...")
            
            # Create temporary migration plan for analysis
            plan = create_migration_plan_from_args(args)
            orchestrator = MigrationOrchestrator(plan)
            analysis = orchestrator.analyze_legacy_data()
            
            # Print analysis results
            print("\n" + "="*60)
            print("LEGACY DATA ANALYSIS")
            print("="*60)
            
            if analysis.get('features_analysis'):
                features = analysis['features_analysis']
                print(f"Features File: {features.get('total_images', 0)} images")
                print(f"  File size: {features.get('file_size_mb', 0):.1f}MB")
                print(f"  Data integrity: {'✅' if features.get('data_integrity') else '❌'}")
            
            if analysis.get('indices_analysis'):
                indices = analysis['indices_analysis']
                loadable = sum(1 for idx in indices if idx.get('loadable', False))
                print(f"FAISS Indices: {len(indices)} files ({loadable} loadable)")
                total_vectors = sum(idx.get('vector_count', 0) for idx in indices if idx.get('loadable'))
                print(f"  Total vectors: {total_vectors}")
            
            if analysis.get('metadata_analysis'):
                metadata = analysis['metadata_analysis']
                print(f"Directory Structure: {metadata.get('total_files', 0)} files")
                print(f"  Image files: {metadata.get('image_files', 0)}")
                print(f"  Item directories: {len(metadata.get('item_directories', []))}")
            
            print(f"\nEstimated Migration Time: {analysis.get('estimated_time', 0):.1f} seconds")
            
            if analysis.get('compatibility_issues'):
                print(f"\nCompatibility Issues:")
                for issue in analysis['compatibility_issues']:
                    print(f"  - {issue}")
            
            return 0
        
        else:
            # Full migration mode
            print("🚀 Starting Migration Orchestration...")
            
            # Create migration plan
            plan = create_migration_plan_from_args(args)
            
            # Execute migration
            orchestrator = MigrationOrchestrator(plan)
            results = orchestrator.execute_migration()
            
            # Print final results
            print("\n" + "="*60)
            print("MIGRATION ORCHESTRATION RESULTS")
            print("="*60)
            
            status = "✅ SUCCESS" if results.migration_successful else "❌ FAILED"
            print(f"Migration Status: {status}")
            print(f"Total Time: {results.total_migration_time:.1f} seconds")
            
            if results.feature_migration_stats:
                print(f"Features Migrated: {results.feature_migration_stats.images_migrated}")
            
            if results.index_migration_stats:
                print(f"Vectors Migrated: {results.index_migration_stats.total_vectors_migrated}")
            
            if results.metadata_migration_stats:
                print(f"Metadata Records: {results.metadata_migration_stats.metadata_records_created}")
            
            if results.validation_results:
                val_status = "PASSED" if results.validation_results.validation_passed else "FAILED"
                print(f"Validation: {val_status}")
            
            if results.errors_encountered:
                print(f"\nErrors ({len(results.errors_encountered)}):")
                for error in results.errors_encountered:
                    print(f"  - {error}")
            
            return 0 if results.migration_successful else 1
    
    except KeyboardInterrupt:
        print("\n🛑 Migration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Migration orchestration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())