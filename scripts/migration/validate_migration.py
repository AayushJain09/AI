#!/usr/bin/env python3
"""
Migration Validation Script

Performs comprehensive validation of the migration from legacy scattered 
file architecture to the unified storage system.

VALIDATION SCOPE:
1. Data Integrity: Verify all data migrated without corruption
2. Feature Accuracy: Compare feature vectors before/after migration
3. Search Functionality: Test search performance and accuracy
4. Metadata Completeness: Validate all metadata was preserved
5. Performance Benchmarks: Ensure performance meets targets
6. System Health: Verify unified storage system is functioning correctly

VALIDATION PROCESS:
- Cross-reference legacy and unified data
- Sample-based integrity checks
- Performance benchmarking
- Functionality testing
- Comprehensive reporting

The validation ensures the migration maintains 100% data integrity
while improving performance and providing enhanced functionality.
"""

import os
import sys
import json
import logging
import sqlite3
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

# Add src to path for unified storage imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unified_storage import UnifiedStore, create_unified_store

# Import for legacy data access
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    h5py = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@dataclass
class ValidationResults:
    """Comprehensive validation results."""
    # Overall status
    validation_passed: bool = True
    validation_timestamp: float = field(default_factory=time.time)
    
    # Data integrity
    total_legacy_items: int = 0
    total_migrated_items: int = 0
    data_integrity_score: float = 0.0
    missing_items: List[str] = field(default_factory=list)
    corrupted_items: List[str] = field(default_factory=list)
    
    # Feature validation
    feature_accuracy_score: float = 0.0
    feature_comparisons_performed: int = 0
    dimension_mismatches: int = 0
    feature_corruption_count: int = 0
    
    # Search functionality
    search_functionality_score: float = 0.0
    search_tests_performed: int = 0
    search_failures: int = 0
    average_search_time_ms: float = 0.0
    
    # Metadata validation
    metadata_completeness_score: float = 0.0
    metadata_records_validated: int = 0
    metadata_issues: List[str] = field(default_factory=list)
    
    # Performance benchmarks
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    performance_targets_met: bool = True
    
    # System health
    system_health_score: float = 0.0
    health_issues: List[str] = field(default_factory=list)
    
    # Detailed issues
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MigrationValidator:
    """
    Comprehensive migration validation system.
    
    Validates all aspects of the migration from legacy architecture
    to unified storage with detailed reporting and recommendations.
    """
    
    def __init__(self,
                 unified_store: UnifiedStore,
                 legacy_features_path: Optional[str] = None,
                 legacy_indices_dir: Optional[str] = None,
                 legacy_data_dir: Optional[str] = None,
                 validation_sample_size: int = 100):
        """
        Initialize migration validator.
        
        Args:
            unified_store: Unified storage system to validate
            legacy_features_path: Path to legacy features.h5 file
            legacy_indices_dir: Directory with legacy FAISS indices
            legacy_data_dir: Legacy data directory
            validation_sample_size: Number of items to sample for testing
        """
        self.unified_store = unified_store
        self.legacy_features_path = Path(legacy_features_path) if legacy_features_path else None
        self.legacy_indices_dir = Path(legacy_indices_dir) if legacy_indices_dir else None
        self.legacy_data_dir = Path(legacy_data_dir) if legacy_data_dir else None
        self.validation_sample_size = validation_sample_size
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.results = ValidationResults()
        
        # Validate prerequisites
        self._validate_prerequisites()
    
    def _validate_prerequisites(self):
        """Validate that necessary components are available for validation."""
        issues = []
        
        # Check unified store
        if not self.unified_store._initialized:
            issues.append("Unified store is not initialized")
        
        # Check legacy data availability
        if self.legacy_features_path and not self.legacy_features_path.exists():
            issues.append(f"Legacy features file not found: {self.legacy_features_path}")
        
        if self.legacy_indices_dir and not self.legacy_indices_dir.exists():
            issues.append(f"Legacy indices directory not found: {self.legacy_indices_dir}")
        
        if self.legacy_data_dir and not self.legacy_data_dir.exists():
            issues.append(f"Legacy data directory not found: {self.legacy_data_dir}")
        
        # Check required libraries
        if self.legacy_features_path and not H5PY_AVAILABLE:
            issues.append("h5py not available for features validation")
        
        if self.legacy_indices_dir and not FAISS_AVAILABLE:
            issues.append("FAISS not available for index validation")
        
        if issues:
            self.logger.warning(f"⚠️ Validation prerequisites issues: {issues}")
            self.results.warnings.extend(issues)
        
        self.logger.info("✅ Validation prerequisites checked")
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate data integrity by comparing legacy and migrated data.
        
        Checks that all data was migrated without loss or corruption.
        """
        self.logger.info("🔍 Validating data integrity...")
        
        integrity_results = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'issues_found': []
        }
        
        try:
            # Get migrated data count
            cursor = self.unified_store.sqlite_connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM vectors")
            migrated_count = cursor.fetchone()[0]
            self.results.total_migrated_items = migrated_count
            
            # Compare with legacy features if available
            if self.legacy_features_path and H5PY_AVAILABLE:
                try:
                    with h5py.File(self.legacy_features_path, 'r') as f:
                        legacy_count = len([k for k in f.keys() if k.startswith('image_')])
                        self.results.total_legacy_items = legacy_count
                        
                        # Check counts match
                        if migrated_count != legacy_count:
                            issue = f"Count mismatch: {migrated_count} migrated vs {legacy_count} legacy"
                            integrity_results['issues_found'].append(issue)
                            self.results.missing_items.append(f"Count difference: {abs(migrated_count - legacy_count)}")
                        else:
                            integrity_results['passed_checks'] += 1
                        
                        integrity_results['total_checks'] += 1
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to validate legacy features: {e}")
                    integrity_results['issues_found'].append(f"Legacy features validation error: {e}")
            
            # Sample integrity validation
            sample_size = min(self.validation_sample_size, migrated_count)
            if sample_size > 0:
                self.logger.info(f"🔍 Sampling {sample_size} items for integrity validation...")
                
                # Get random sample of migrated items
                cursor.execute("SELECT image_id FROM vectors ORDER BY RANDOM() LIMIT ?", (sample_size,))
                sample_ids = [row[0] for row in cursor.fetchall()]
                
                for image_id in tqdm(sample_ids, desc="Validating integrity"):
                    try:
                        # Check vector data integrity
                        cursor.execute("SELECT combined_features, feature_norm FROM vectors WHERE image_id = ?", (image_id,))
                        result = cursor.fetchone()
                        
                        if result:
                            combined_blob, stored_norm = result
                            features = np.frombuffer(combined_blob, dtype=np.float32)
                            
                            # Verify dimensions
                            if len(features) != 1536:
                                integrity_results['issues_found'].append(
                                    f"Dimension error for {image_id}: {len(features)} != 1536"
                                )
                                self.results.corrupted_items.append(image_id)
                            else:
                                integrity_results['passed_checks'] += 1
                            
                            # Verify norm calculation
                            calculated_norm = float(np.linalg.norm(features))
                            if abs(calculated_norm - stored_norm) > 0.01:
                                integrity_results['issues_found'].append(
                                    f"Norm mismatch for {image_id}: {calculated_norm:.6f} vs {stored_norm:.6f}"
                                )
                        else:
                            integrity_results['issues_found'].append(f"Missing vector data for {image_id}")
                        
                        integrity_results['total_checks'] += 1
                        
                    except Exception as e:
                        integrity_results['issues_found'].append(f"Integrity check error for {image_id}: {e}")
                        integrity_results['failed_checks'] += 1
            
            # Calculate integrity score
            if integrity_results['total_checks'] > 0:
                self.results.data_integrity_score = (
                    integrity_results['passed_checks'] / integrity_results['total_checks']
                )
            
            # Log results
            self.logger.info(f"📊 Data Integrity Results:")
            self.logger.info(f"   Total checks: {integrity_results['total_checks']}")
            self.logger.info(f"   Passed: {integrity_results['passed_checks']}")
            self.logger.info(f"   Failed: {integrity_results['failed_checks']}")
            self.logger.info(f"   Integrity score: {self.results.data_integrity_score:.3f}")
            
            if integrity_results['issues_found']:
                self.logger.warning(f"⚠️ Integrity issues found:")
                for issue in integrity_results['issues_found'][:10]:  # Show first 10
                    self.logger.warning(f"   - {issue}")
        
        except Exception as e:
            self.logger.error(f"❌ Data integrity validation failed: {e}")
            integrity_results['error'] = str(e)
            self.results.critical_issues.append(f"Data integrity validation error: {e}")
        
        return integrity_results
    
    def validate_feature_accuracy(self) -> Dict[str, Any]:
        """
        Validate feature accuracy by comparing legacy and migrated features.
        
        Ensures feature vectors maintain accuracy after migration.
        """
        self.logger.info("🔍 Validating feature accuracy...")
        
        accuracy_results = {
            'comparisons_performed': 0,
            'high_similarity_count': 0,
            'medium_similarity_count': 0,
            'low_similarity_count': 0,
            'similarity_scores': [],
            'dimension_issues': 0,
            'issues_found': []
        }
        
        if not self.legacy_features_path or not H5PY_AVAILABLE:
            self.logger.warning("⚠️ Legacy features not available for accuracy validation")
            return accuracy_results
        
        try:
            with h5py.File(self.legacy_features_path, 'r') as f:
                # Get sample of items to compare
                legacy_items = [k for k in f.keys() if k.startswith('image_')]
                sample_size = min(self.validation_sample_size, len(legacy_items))
                sample_items = random.sample(legacy_items, sample_size)
                
                self.logger.info(f"🔍 Comparing {sample_size} feature vectors...")
                
                cursor = self.unified_store.sqlite_connection.cursor()
                
                for item_key in tqdm(sample_items, desc="Comparing features"):
                    try:
                        # Get legacy features
                        legacy_group = f[item_key]
                        legacy_clip = legacy_group['clip'][:]
                        legacy_dinov2 = legacy_group['dinov2'][:]
                        legacy_combined = np.concatenate([legacy_clip, legacy_dinov2])
                        legacy_combined = legacy_combined / np.linalg.norm(legacy_combined)
                        
                        # Get image ID
                        image_id = legacy_group.attrs.get('item_id', item_key)
                        
                        # Get migrated features
                        cursor.execute("SELECT combined_features FROM vectors WHERE image_id = ?", (str(image_id),))
                        result = cursor.fetchone()
                        
                        if result:
                            migrated_features = np.frombuffer(result[0], dtype=np.float32)
                            
                            # Check dimensions
                            if len(legacy_combined) != len(migrated_features):
                                accuracy_results['dimension_issues'] += 1
                                accuracy_results['issues_found'].append(
                                    f"Dimension mismatch for {image_id}: "
                                    f"{len(legacy_combined)} vs {len(migrated_features)}"
                                )
                                continue
                            
                            # Calculate similarity
                            similarity = float(np.dot(legacy_combined, migrated_features))
                            accuracy_results['similarity_scores'].append(similarity)
                            
                            # Categorize similarity
                            if similarity > 0.999:
                                accuracy_results['high_similarity_count'] += 1
                            elif similarity > 0.99:
                                accuracy_results['medium_similarity_count'] += 1
                            else:
                                accuracy_results['low_similarity_count'] += 1
                                accuracy_results['issues_found'].append(
                                    f"Low similarity for {image_id}: {similarity:.6f}"
                                )
                            
                            accuracy_results['comparisons_performed'] += 1
                        
                        else:
                            accuracy_results['issues_found'].append(f"Missing migrated features for {image_id}")
                    
                    except Exception as e:
                        accuracy_results['issues_found'].append(f"Feature comparison error for {item_key}: {e}")
                
                # Calculate accuracy metrics
                self.results.feature_comparisons_performed = accuracy_results['comparisons_performed']
                self.results.dimension_mismatches = accuracy_results['dimension_issues']
                
                if accuracy_results['similarity_scores']:
                    avg_similarity = np.mean(accuracy_results['similarity_scores'])
                    self.results.feature_accuracy_score = avg_similarity
                
                # Log results
                self.logger.info(f"📊 Feature Accuracy Results:")
                self.logger.info(f"   Comparisons: {accuracy_results['comparisons_performed']}")
                self.logger.info(f"   High similarity (>0.999): {accuracy_results['high_similarity_count']}")
                self.logger.info(f"   Medium similarity (>0.99): {accuracy_results['medium_similarity_count']}")
                self.logger.info(f"   Low similarity (<0.99): {accuracy_results['low_similarity_count']}")
                self.logger.info(f"   Average similarity: {self.results.feature_accuracy_score:.6f}")
                self.logger.info(f"   Dimension issues: {accuracy_results['dimension_issues']}")
        
        except Exception as e:
            self.logger.error(f"❌ Feature accuracy validation failed: {e}")
            accuracy_results['error'] = str(e)
            self.results.critical_issues.append(f"Feature accuracy validation error: {e}")
        
        return accuracy_results
    
    def validate_search_functionality(self) -> Dict[str, Any]:
        """
        Validate search functionality and performance.
        
        Tests search accuracy, performance, and reliability.
        """
        self.logger.info("🔍 Validating search functionality...")
        
        search_results = {
            'tests_performed': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'search_times': [],
            'accuracy_tests': [],
            'issues_found': []
        }
        
        try:
            # Check if search index is available
            if self.unified_store.faiss_index is None:
                search_results['issues_found'].append("No FAISS index available")
                self.results.critical_issues.append("Search functionality unavailable - no FAISS index")
                return search_results
            
            index_size = self.unified_store.index_size
            if index_size == 0:
                search_results['issues_found'].append("FAISS index is empty")
                self.results.critical_issues.append("Search functionality unavailable - empty index")
                return search_results
            
            # Get sample of vectors for testing
            cursor = self.unified_store.sqlite_connection.cursor()
            cursor.execute("SELECT image_id, combined_features FROM vectors ORDER BY RANDOM() LIMIT ?", 
                          (min(20, index_size),))
            test_vectors = cursor.fetchall()
            
            self.logger.info(f"🔍 Testing search with {len(test_vectors)} queries...")
            
            for image_id, feature_blob in tqdm(test_vectors, desc="Testing search"):
                try:
                    # Prepare query vector
                    query_vector = np.frombuffer(feature_blob, dtype=np.float32).reshape(1, -1)
                    
                    # Perform search
                    start_time = time.time()
                    distances, indices = self.unified_store.faiss_index.search(query_vector, min(10, index_size))
                    search_time = (time.time() - start_time) * 1000
                    
                    search_results['search_times'].append(search_time)
                    search_results['tests_performed'] += 1
                    
                    # Validate search results
                    if len(indices[0]) > 0 and indices[0][0] != -1:
                        search_results['successful_searches'] += 1
                        
                        # Check if the query image is the top result (should be for exact match)
                        top_result_id = self.unified_store.image_id_mapping.get(indices[0][0])
                        if top_result_id == image_id:
                            search_results['accuracy_tests'].append(1.0)  # Perfect match
                        else:
                            # Check if it's in top results
                            found_in_results = False
                            for i, idx in enumerate(indices[0]):
                                if idx != -1:
                                    result_id = self.unified_store.image_id_mapping.get(idx)
                                    if result_id == image_id:
                                        # Accuracy decreases with position
                                        search_results['accuracy_tests'].append(1.0 / (i + 1))
                                        found_in_results = True
                                        break
                            
                            if not found_in_results:
                                search_results['accuracy_tests'].append(0.0)
                                search_results['issues_found'].append(
                                    f"Query image {image_id} not found in its own search results"
                                )
                    else:
                        search_results['failed_searches'] += 1
                        search_results['issues_found'].append(f"Search returned no results for {image_id}")
                
                except Exception as e:
                    search_results['failed_searches'] += 1
                    search_results['issues_found'].append(f"Search error for {image_id}: {e}")
            
            # Calculate metrics
            if search_results['tests_performed'] > 0:
                success_rate = search_results['successful_searches'] / search_results['tests_performed']
                self.results.search_functionality_score = success_rate
                
                if search_results['search_times']:
                    self.results.average_search_time_ms = np.mean(search_results['search_times'])
                
                if search_results['accuracy_tests']:
                    avg_accuracy = np.mean(search_results['accuracy_tests'])
                    search_results['average_accuracy'] = avg_accuracy
            
            self.results.search_tests_performed = search_results['tests_performed']
            self.results.search_failures = search_results['failed_searches']
            
            # Performance benchmarking
            if search_results['search_times']:
                self.results.performance_benchmarks['search_time_ms'] = self.results.average_search_time_ms
                
                # Check against performance targets
                device_type = self.unified_store.config.platform.device_type
                target_times = {'cuda': 50, 'mps': 100, 'cpu': 200}  # ms
                target_time = target_times.get(device_type, 200)
                
                if self.results.average_search_time_ms > target_time:
                    self.results.performance_targets_met = False
                    self.results.warnings.append(
                        f"Search performance below target: {self.results.average_search_time_ms:.1f}ms > {target_time}ms"
                    )
            
            # Log results
            self.logger.info(f"📊 Search Functionality Results:")
            self.logger.info(f"   Tests performed: {search_results['tests_performed']}")
            self.logger.info(f"   Successful searches: {search_results['successful_searches']}")
            self.logger.info(f"   Failed searches: {search_results['failed_searches']}")
            self.logger.info(f"   Average search time: {self.results.average_search_time_ms:.1f}ms")
            self.logger.info(f"   Success rate: {self.results.search_functionality_score:.3f}")
            
            if 'average_accuracy' in search_results:
                self.logger.info(f"   Average accuracy: {search_results['average_accuracy']:.3f}")
        
        except Exception as e:
            self.logger.error(f"❌ Search functionality validation failed: {e}")
            search_results['error'] = str(e)
            self.results.critical_issues.append(f"Search functionality validation error: {e}")
        
        return search_results
    
    def validate_metadata_completeness(self) -> Dict[str, Any]:
        """
        Validate metadata completeness and integrity.
        
        Ensures all metadata was migrated and is accessible.
        """
        self.logger.info("🔍 Validating metadata completeness...")
        
        metadata_results = {
            'total_records': 0,
            'complete_records': 0,
            'incomplete_records': 0,
            'extended_metadata_count': 0,
            'issues_found': []
        }
        
        try:
            cursor = self.unified_store.sqlite_connection.cursor()
            
            # Check basic metadata table
            cursor.execute("SELECT COUNT(*) FROM metadata")
            total_metadata = cursor.fetchone()[0]
            metadata_results['total_records'] = total_metadata
            
            # Check extended metadata table
            cursor.execute("SELECT COUNT(*) FROM extended_metadata")
            extended_metadata = cursor.fetchone()[0]
            metadata_results['extended_metadata_count'] = extended_metadata
            
            if total_metadata != extended_metadata:
                metadata_results['issues_found'].append(
                    f"Metadata count mismatch: {total_metadata} basic vs {extended_metadata} extended"
                )
            
            # Sample metadata validation
            sample_size = min(50, total_metadata)
            cursor.execute("SELECT image_id, original_path, file_size_bytes FROM metadata ORDER BY RANDOM() LIMIT ?", 
                          (sample_size,))
            sample_records = cursor.fetchall()
            
            for image_id, original_path, file_size in sample_records:
                record_complete = True
                
                # Check required fields
                if not image_id:
                    metadata_results['issues_found'].append("Missing image_id")
                    record_complete = False
                
                # Check extended metadata
                cursor.execute("SELECT metadata_json FROM extended_metadata WHERE image_id = ?", (image_id,))
                extended_result = cursor.fetchone()
                
                if extended_result:
                    try:
                        extended_data = json.loads(extended_result[0])
                        if not extended_data.get('file_metadata'):
                            metadata_results['issues_found'].append(f"Missing file_metadata for {image_id}")
                            record_complete = False
                    except json.JSONDecodeError:
                        metadata_results['issues_found'].append(f"Invalid extended metadata JSON for {image_id}")
                        record_complete = False
                else:
                    metadata_results['issues_found'].append(f"Missing extended metadata for {image_id}")
                    record_complete = False
                
                if record_complete:
                    metadata_results['complete_records'] += 1
                else:
                    metadata_results['incomplete_records'] += 1
            
            # Calculate completeness score
            self.results.metadata_records_validated = len(sample_records)
            if len(sample_records) > 0:
                self.results.metadata_completeness_score = (
                    metadata_results['complete_records'] / len(sample_records)
                )
            
            # Log results
            self.logger.info(f"📊 Metadata Completeness Results:")
            self.logger.info(f"   Total records: {metadata_results['total_records']}")
            self.logger.info(f"   Extended records: {metadata_results['extended_metadata_count']}")
            self.logger.info(f"   Sample validated: {len(sample_records)}")
            self.logger.info(f"   Complete records: {metadata_results['complete_records']}")
            self.logger.info(f"   Incomplete records: {metadata_results['incomplete_records']}")
            self.logger.info(f"   Completeness score: {self.results.metadata_completeness_score:.3f}")
        
        except Exception as e:
            self.logger.error(f"❌ Metadata validation failed: {e}")
            metadata_results['error'] = str(e)
            self.results.critical_issues.append(f"Metadata validation error: {e}")
        
        return metadata_results
    
    def validate_system_health(self) -> Dict[str, Any]:
        """
        Validate overall system health and functionality.
        
        Performs comprehensive health checks on the unified storage system.
        """
        self.logger.info("🔍 Validating system health...")
        
        try:
            # Use unified store's health check
            health_status = self.unified_store.health_check()
            
            # Calculate health score
            healthy_components = sum(1 for comp in health_status['components'].values() 
                                   if comp.get('status') == 'healthy')
            total_components = len(health_status['components'])
            
            if total_components > 0:
                self.results.system_health_score = healthy_components / total_components
            
            # Collect health issues
            if health_status.get('errors'):
                self.results.health_issues.extend(health_status['errors'])
                self.results.critical_issues.extend(health_status['errors'])
            
            if health_status.get('warnings'):
                self.results.health_issues.extend(health_status['warnings'])
                self.results.warnings.extend(health_status['warnings'])
            
            # Log results
            self.logger.info(f"📊 System Health Results:")
            self.logger.info(f"   Overall status: {health_status['overall_status']}")
            self.logger.info(f"   Health score: {self.results.system_health_score:.3f}")
            self.logger.info(f"   Healthy components: {healthy_components}/{total_components}")
            
            for component, status in health_status['components'].items():
                component_status = status.get('status', 'unknown')
                self.logger.info(f"   {component}: {component_status}")
            
            return health_status
        
        except Exception as e:
            self.logger.error(f"❌ System health validation failed: {e}")
            self.results.critical_issues.append(f"System health validation error: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_validation(self) -> ValidationResults:
        """
        Run all validation tests and compile comprehensive results.
        
        Returns complete validation results with recommendations.
        """
        self.logger.info("🚀 Starting comprehensive migration validation...")
        start_time = time.time()
        
        try:
            # Run all validation tests
            self.logger.info("1/5 Validating data integrity...")
            data_integrity = self.validate_data_integrity()
            
            self.logger.info("2/5 Validating feature accuracy...")
            feature_accuracy = self.validate_feature_accuracy()
            
            self.logger.info("3/5 Validating search functionality...")
            search_functionality = self.validate_search_functionality()
            
            self.logger.info("4/5 Validating metadata completeness...")
            metadata_completeness = self.validate_metadata_completeness()
            
            self.logger.info("5/5 Validating system health...")
            system_health = self.validate_system_health()
            
            # Calculate overall validation score
            scores = [
                self.results.data_integrity_score,
                self.results.feature_accuracy_score,
                self.results.search_functionality_score,
                self.results.metadata_completeness_score,
                self.results.system_health_score
            ]
            
            # Filter out zero scores (tests that didn't run)
            valid_scores = [s for s in scores if s > 0]
            overall_score = np.mean(valid_scores) if valid_scores else 0.0
            
            # Determine if validation passed
            self.results.validation_passed = (
                overall_score >= 0.95 and  # 95% overall score
                len(self.results.critical_issues) == 0 and  # No critical issues
                self.results.data_integrity_score >= 0.99  # 99% data integrity
            )
            
            # Generate recommendations
            self._generate_recommendations()
            
            # Calculate validation time
            validation_time = time.time() - start_time
            
            # Log comprehensive results
            self.logger.info("✅ Comprehensive validation completed!")
            self.logger.info(f"📊 Validation Summary:")
            self.logger.info(f"   Overall Score: {overall_score:.3f}")
            self.logger.info(f"   Validation Passed: {'✅ Yes' if self.results.validation_passed else '❌ No'}")
            self.logger.info(f"   Data Integrity: {self.results.data_integrity_score:.3f}")
            self.logger.info(f"   Feature Accuracy: {self.results.feature_accuracy_score:.3f}")
            self.logger.info(f"   Search Functionality: {self.results.search_functionality_score:.3f}")
            self.logger.info(f"   Metadata Completeness: {self.results.metadata_completeness_score:.3f}")
            self.logger.info(f"   System Health: {self.results.system_health_score:.3f}")
            self.logger.info(f"   Validation Time: {validation_time:.1f}s")
            
            if self.results.critical_issues:
                self.logger.error(f"❌ Critical Issues ({len(self.results.critical_issues)}):")
                for issue in self.results.critical_issues:
                    self.logger.error(f"   - {issue}")
            
            if self.results.warnings:
                self.logger.warning(f"⚠️ Warnings ({len(self.results.warnings)}):")
                for warning in self.results.warnings[:10]:  # Show first 10
                    self.logger.warning(f"   - {warning}")
            
            if self.results.recommendations:
                self.logger.info(f"💡 Recommendations ({len(self.results.recommendations)}):")
                for recommendation in self.results.recommendations:
                    self.logger.info(f"   - {recommendation}")
        
        except Exception as e:
            self.logger.error(f"❌ Comprehensive validation failed: {e}")
            self.results.validation_passed = False
            self.results.critical_issues.append(f"Validation process error: {e}")
        
        return self.results
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        # Data integrity recommendations
        if self.results.data_integrity_score < 0.99:
            self.results.recommendations.append(
                "Consider re-running migration - data integrity score below 99%"
            )
        
        if self.results.missing_items:
            self.results.recommendations.append(
                f"Investigate {len(self.results.missing_items)} missing items"
            )
        
        # Feature accuracy recommendations
        if self.results.feature_accuracy_score < 0.999:
            self.results.recommendations.append(
                "Feature accuracy below expected threshold - verify migration process"
            )
        
        # Search performance recommendations
        if self.results.average_search_time_ms > 200:
            self.results.recommendations.append(
                "Consider index optimization for better search performance"
            )
        
        if self.results.search_functionality_score < 0.95:
            self.results.recommendations.append(
                "Search functionality issues detected - verify FAISS index integrity"
            )
        
        # Metadata recommendations
        if self.results.metadata_completeness_score < 0.95:
            self.results.recommendations.append(
                "Metadata completeness below threshold - verify metadata migration"
            )
        
        # System health recommendations
        if self.results.system_health_score < 0.9:
            self.results.recommendations.append(
                "System health issues detected - check component status"
            )
        
        # Performance recommendations
        if not self.results.performance_targets_met:
            self.results.recommendations.append(
                "Performance targets not met - consider platform-specific optimizations"
            )
    
    def export_validation_report(self, output_path: str) -> str:
        """Export comprehensive validation report."""
        report_data = {
            'validation_summary': {
                'passed': self.results.validation_passed,
                'timestamp': self.results.validation_timestamp,
                'overall_score': np.mean([
                    self.results.data_integrity_score,
                    self.results.feature_accuracy_score,
                    self.results.search_functionality_score,
                    self.results.metadata_completeness_score,
                    self.results.system_health_score
                ])
            },
            'detailed_results': {
                'data_integrity': {
                    'score': self.results.data_integrity_score,
                    'total_legacy_items': self.results.total_legacy_items,
                    'total_migrated_items': self.results.total_migrated_items,
                    'missing_items_count': len(self.results.missing_items),
                    'corrupted_items_count': len(self.results.corrupted_items)
                },
                'feature_accuracy': {
                    'score': self.results.feature_accuracy_score,
                    'comparisons_performed': self.results.feature_comparisons_performed,
                    'dimension_mismatches': self.results.dimension_mismatches
                },
                'search_functionality': {
                    'score': self.results.search_functionality_score,
                    'tests_performed': self.results.search_tests_performed,
                    'failures': self.results.search_failures,
                    'average_search_time_ms': self.results.average_search_time_ms
                },
                'metadata_completeness': {
                    'score': self.results.metadata_completeness_score,
                    'records_validated': self.results.metadata_records_validated
                },
                'system_health': {
                    'score': self.results.system_health_score
                }
            },
            'performance_benchmarks': self.results.performance_benchmarks,
            'issues': {
                'critical_issues': self.results.critical_issues,
                'warnings': self.results.warnings,
                'recommendations': self.results.recommendations
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"📋 Validation report exported to: {output_path}")
        return output_path


def main():
    """Main entry point for migration validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive migration validation for unified storage system'
    )
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Unified storage data directory')
    parser.add_argument('--legacy-features', type=str,
                       help='Path to legacy features.h5 file')
    parser.add_argument('--legacy-indices', type=str,
                       help='Path to legacy FAISS indices directory')
    parser.add_argument('--legacy-data', type=str,
                       help='Path to legacy data directory')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Sample size for validation tests')
    parser.add_argument('--output-report', type=str,
                       help='Path to export validation report (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create unified store
        logger.info("🚀 Initializing unified storage system...")
        store = create_unified_store(args.data_dir)
        
        # Create validator
        validator = MigrationValidator(
            unified_store=store,
            legacy_features_path=args.legacy_features,
            legacy_indices_dir=args.legacy_indices,
            legacy_data_dir=args.legacy_data,
            validation_sample_size=args.sample_size
        )
        
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Export report if requested
        if args.output_report:
            validator.export_validation_report(args.output_report)
        
        # Print summary
        print("\n" + "="*60)
        print("MIGRATION VALIDATION SUMMARY")
        print("="*60)
        print(f"Validation Result: {'✅ PASSED' if results.validation_passed else '❌ FAILED'}")
        print(f"Data Integrity: {results.data_integrity_score:.3f}")
        print(f"Feature Accuracy: {results.feature_accuracy_score:.3f}")
        print(f"Search Functionality: {results.search_functionality_score:.3f}")
        print(f"Metadata Completeness: {results.metadata_completeness_score:.3f}")
        print(f"System Health: {results.system_health_score:.3f}")
        
        if results.critical_issues:
            print(f"\n❌ Critical Issues ({len(results.critical_issues)}):")
            for issue in results.critical_issues:
                print(f"  - {issue}")
        
        if results.warnings:
            print(f"\n⚠️ Warnings ({len(results.warnings)}):")
            for warning in results.warnings[:5]:  # Show first 5
                print(f"  - {warning}")
        
        if results.recommendations:
            print(f"\n💡 Recommendations ({len(results.recommendations)}):")
            for recommendation in results.recommendations:
                print(f"  - {recommendation}")
        
        # Clean up
        store.close()
        logger.info("✅ Migration validation completed")
        
        return 0 if results.validation_passed else 1
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())