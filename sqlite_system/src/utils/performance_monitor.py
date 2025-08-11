"""
Performance Monitoring System for SQLite-based Recognition
Tracks and analyzes system performance with SQLite storage integration
Preserves monitoring capabilities from original system while adding SQLite-specific metrics
"""

import time
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import sqlite3
from collections import defaultdict, deque
import threading
import yaml

# Import SQLite storage and platform detection
from ..storage.sqlite_store import SQLiteVectorStore
from ..utils.platform_detector import get_platform_config
from ..inference.recognition_pipeline import RecognitionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    metric_id: str
    timestamp: float
    metric_type: str  # 'recognition', 'search', 'extraction', 'storage'
    value: float
    unit: str
    metadata: Dict
    

@dataclass  
class RecognitionPerformanceRecord:
    """Detailed performance record for a recognition operation"""
    record_id: str
    timestamp: float
    image_id: str
    item_id: str
    confidence: float
    recognition_time_ms: float
    feature_extraction_time_ms: float
    search_time_ms: float
    refinement_time_ms: Optional[float]
    recognition_method: str
    platform_type: str
    stage_results: Dict
    success: bool
    

class SQLitePerformanceMonitor:
    """
    Comprehensive performance monitoring system for SQLite-based recognition
    Tracks system performance, identifies bottlenecks, and provides optimization insights
    """
    
    def __init__(self, vector_store: SQLiteVectorStore, config: Optional[Dict] = None):
        self.vector_store = vector_store
        self.config = config or {}
        
        # Platform information
        self.platform_config = get_platform_config()
        
        # Performance tracking configuration
        self.enable_detailed_tracking = self.config.get('enable_performance_tracking', True)
        self.metrics_retention_days = self.config.get('metrics_retention_days', 30)
        self.real_time_monitoring = self.config.get('real_time_monitoring', True)
        
        # Performance targets from configuration
        self.target_recognition_time_ms = self.platform_config.get('recognition_target_ms', 250)
        self.target_accuracy_percent = self.config.get('target_accuracy_percent', 99.0)
        
        # Real-time metrics storage (in-memory for fast access)
        self.recent_metrics = {
            'recognition_times': deque(maxlen=1000),
            'confidence_scores': deque(maxlen=1000),
            'success_count': deque(maxlen=1000),
            'error_count': deque(maxlen=100)
        }
        
        # Performance statistics
        self.performance_stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'avg_recognition_time_ms': 0.0,
            'avg_confidence_score': 0.0,
            'p95_recognition_time_ms': 0.0,
            'p99_recognition_time_ms': 0.0,
            'accuracy_rate': 0.0,
            'cache_hit_rate': 0.0,
            'sqlite_performance': {},
            'system_health_score': 0.0
        }
        
        # Thread safety for concurrent access
        self.stats_lock = threading.Lock()
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        logger.info("📊 SQLite Performance Monitor initialized")
        logger.info(f"   Platform: {self.platform_config['platform_type']}")
        logger.info(f"   Target recognition time: {self.target_recognition_time_ms}ms")
        logger.info(f"   Detailed tracking: {'✅ Enabled' if self.enable_detailed_tracking else '❌ Disabled'}")
    
    def _initialize_monitoring(self):
        """Initialize performance monitoring infrastructure"""
        try:
            # Create performance monitoring tables in SQLite
            self._create_monitoring_schema()
            
            # Start background cleanup task if configured
            if self.metrics_retention_days > 0:
                self._start_cleanup_task()
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring: {e}")
    
    def _create_monitoring_schema(self):
        """Create performance monitoring tables in SQLite"""
        cursor = self.vector_store.connection.cursor()
        try:
            # Detailed recognition performance table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_performance (
                record_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                image_id TEXT,
                query_item_id TEXT,
                result_item_id TEXT,
                confidence_score REAL,
                recognition_time_ms REAL,
                feature_extraction_time_ms REAL,
                search_time_ms REAL,
                refinement_time_ms REAL,
                recognition_method TEXT,
                platform_type TEXT,
                stage_results JSON,
                success INTEGER,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # System metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                metric_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_unit TEXT,
                platform_type TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Performance aggregates table (for faster queries)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_aggregates (
                aggregate_id TEXT PRIMARY KEY,
                time_bucket TEXT NOT NULL,  -- 'hour', 'day', 'week'
                bucket_start REAL NOT NULL,
                total_recognitions INTEGER,
                successful_recognitions INTEGER,
                avg_recognition_time_ms REAL,
                p95_recognition_time_ms REAL,
                p99_recognition_time_ms REAL,
                avg_confidence_score REAL,
                accuracy_rate REAL,
                platform_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recognition_perf_timestamp ON recognition_performance(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recognition_perf_success ON recognition_performance(success)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_type ON system_metrics(metric_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_aggregates_bucket ON performance_aggregates(time_bucket, bucket_start)')
            
            self.vector_store.connection.commit()
            logger.debug("✅ Performance monitoring schema created")
        finally:
            cursor.close()
    
    def record_recognition_performance(self, result: RecognitionResult, 
                                     feature_extraction_time_ms: Optional[float] = None,
                                     search_time_ms: Optional[float] = None,
                                     refinement_time_ms: Optional[float] = None,
                                     image_id: Optional[str] = None,
                                     error_message: Optional[str] = None) -> str:
        """
        Record detailed performance metrics for a recognition operation
        
        Args:
            result: Recognition result with performance data
            feature_extraction_time_ms: Time spent on feature extraction
            search_time_ms: Time spent on vector search
            refinement_time_ms: Time spent on refinement (if used)
            image_id: Identifier for the input image
            error_message: Error message if recognition failed
            
        Returns:
            Record ID for the stored performance data
        """
        record_id = f"perf_{uuid.uuid4().hex}"
        timestamp = time.time()
        
        try:
            # Create performance record
            perf_record = RecognitionPerformanceRecord(
                record_id=record_id,
                timestamp=timestamp,
                image_id=image_id or f"img_{uuid.uuid4().hex[:8]}",
                item_id=result.item_id,
                confidence=result.confidence,
                recognition_time_ms=result.inference_time * 1000,
                feature_extraction_time_ms=feature_extraction_time_ms or 0.0,
                search_time_ms=search_time_ms or 0.0,
                refinement_time_ms=refinement_time_ms,
                recognition_method=result.recognition_method,
                platform_type=self.platform_config['platform_type'],
                stage_results=result.stage_results,
                success=result.item_id != "unknown"
            )
            
            # Store in SQLite if detailed tracking is enabled
            if self.enable_detailed_tracking:
                self._store_performance_record(perf_record, error_message)
            
            # Update real-time metrics
            with self.stats_lock:
                self._update_realtime_metrics(perf_record)
            
            # Update aggregated statistics
            self._update_performance_stats(perf_record)
            
            logger.debug(f"📊 Recorded performance: {result.item_id} in {result.inference_time*1000:.1f}ms")
            
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Failed to record performance: {e}")
            return ""
    
    def _store_performance_record(self, record: RecognitionPerformanceRecord, error_message: Optional[str] = None):
        """Store detailed performance record in SQLite"""
        try:
            cursor = self.vector_store.connection.cursor()
            
            cursor.execute('''
            INSERT INTO recognition_performance 
            (record_id, timestamp, image_id, query_item_id, result_item_id, 
             confidence_score, recognition_time_ms, feature_extraction_time_ms, 
             search_time_ms, refinement_time_ms, recognition_method, platform_type, 
             stage_results, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.record_id, record.timestamp, record.image_id, record.image_id,
                record.item_id, record.confidence, record.recognition_time_ms,
                record.feature_extraction_time_ms, record.search_time_ms,
                record.refinement_time_ms, record.recognition_method, 
                record.platform_type, json.dumps(record.stage_results),
                1 if record.success else 0, error_message
            ))
            
            self.vector_store.connection.commit()
            
        except Exception as e:
            logger.error(f"❌ Failed to store performance record: {e}")
    
    def _update_realtime_metrics(self, record: RecognitionPerformanceRecord):
        """Update real-time metrics for monitoring dashboard"""
        self.recent_metrics['recognition_times'].append(record.recognition_time_ms)
        self.recent_metrics['confidence_scores'].append(record.confidence)
        self.recent_metrics['success_count'].append(1 if record.success else 0)
        
        if not record.success:
            self.recent_metrics['error_count'].append({
                'timestamp': record.timestamp,
                'item_id': record.item_id,
                'method': record.recognition_method
            })
    
    def _update_performance_stats(self, record: RecognitionPerformanceRecord):
        """Update aggregated performance statistics"""
        with self.stats_lock:
            self.performance_stats['total_recognitions'] += 1
            
            if record.success:
                self.performance_stats['successful_recognitions'] += 1
            else:
                self.performance_stats['failed_recognitions'] += 1
            
            # Update running averages
            total = self.performance_stats['total_recognitions']
            
            # Average recognition time
            current_avg_time = self.performance_stats['avg_recognition_time_ms']
            self.performance_stats['avg_recognition_time_ms'] = (
                (current_avg_time * (total - 1) + record.recognition_time_ms) / total
            )
            
            # Average confidence score
            current_avg_conf = self.performance_stats['avg_confidence_score']
            self.performance_stats['avg_confidence_score'] = (
                (current_avg_conf * (total - 1) + record.confidence) / total
            )
            
            # Accuracy rate
            self.performance_stats['accuracy_rate'] = (
                self.performance_stats['successful_recognitions'] / total * 100
            )
            
            # Update percentiles from recent data
            if len(self.recent_metrics['recognition_times']) >= 10:
                times = list(self.recent_metrics['recognition_times'])
                self.performance_stats['p95_recognition_time_ms'] = np.percentile(times, 95)
                self.performance_stats['p99_recognition_time_ms'] = np.percentile(times, 99)
            
            # Calculate system health score
            self._calculate_system_health_score()
    
    def _calculate_system_health_score(self):
        """Calculate overall system health score (0-100)"""
        scores = []
        
        # Performance score (based on recognition time vs target)
        avg_time = self.performance_stats['avg_recognition_time_ms']
        if avg_time > 0:
            time_score = max(0, 100 - (avg_time / self.target_recognition_time_ms - 1) * 50)
            scores.append(time_score)
        
        # Accuracy score
        accuracy = self.performance_stats['accuracy_rate']
        accuracy_score = max(0, min(100, accuracy))
        scores.append(accuracy_score)
        
        # Error rate score
        total = self.performance_stats['total_recognitions']
        if total > 0:
            error_rate = self.performance_stats['failed_recognitions'] / total
            error_score = max(0, 100 - error_rate * 200)  # Penalize errors heavily
            scores.append(error_score)
        
        # Overall health score
        self.performance_stats['system_health_score'] = np.mean(scores) if scores else 0
    
    def record_system_metric(self, metric_type: str, metric_name: str, value: float, 
                           unit: str = "", metadata: Optional[Dict] = None):
        """
        Record a custom system metric
        
        Args:
            metric_type: Type of metric ('sqlite', 'platform', 'custom')
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            metadata: Additional metadata
        """
        try:
            metric_id = f"metric_{uuid.uuid4().hex}"
            timestamp = time.time()
            
            if self.enable_detailed_tracking:
                cursor = self.vector_store.connection.cursor()
                
                cursor.execute('''
                INSERT INTO system_metrics 
                (metric_id, timestamp, metric_type, metric_name, metric_value, 
                 metric_unit, platform_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric_id, timestamp, metric_type, metric_name, value,
                    unit, self.platform_config['platform_type'], 
                    json.dumps(metadata or {})
                ))
                
                self.vector_store.connection.commit()
            
            logger.debug(f"📊 Recorded metric: {metric_name} = {value} {unit}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record system metric: {e}")
    
    def get_performance_report(self, time_range_hours: int = 24) -> Dict:
        """
        Generate comprehensive performance report
        
        Args:
            time_range_hours: Time range for the report in hours
            
        Returns:
            Detailed performance report
        """
        try:
            end_time = time.time()
            start_time = end_time - (time_range_hours * 3600)
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'time_range_hours': time_range_hours,
                    'platform_type': self.platform_config['platform_type'],
                    'monitoring_enabled': self.enable_detailed_tracking
                },
                'overall_performance': dict(self.performance_stats),
                'detailed_metrics': self._get_detailed_metrics(start_time, end_time),
                'sqlite_performance': self._get_sqlite_performance_metrics(),
                'platform_optimization': self._get_platform_optimization_report(),
                'recommendations': self._generate_performance_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    def _get_detailed_metrics(self, start_time: float, end_time: float) -> Dict:
        """Get detailed metrics for specified time range"""
        try:
            cursor = self.vector_store.connection.cursor()
            
            # Query detailed performance data
            cursor.execute('''
            SELECT recognition_method, success, recognition_time_ms, confidence_score,
                   feature_extraction_time_ms, search_time_ms, refinement_time_ms
            FROM recognition_performance 
            WHERE timestamp BETWEEN ? AND ?
            ''', (start_time, end_time))
            
            records = cursor.fetchall()
            
            # Analyze the data
            method_stats = defaultdict(lambda: {
                'count': 0,
                'success_count': 0,
                'avg_time_ms': 0,
                'avg_confidence': 0,
                'times': []
            })
            
            for record in records:
                method, success, rec_time, confidence, feat_time, search_time, ref_time = record
                
                method_stats[method]['count'] += 1
                if success:
                    method_stats[method]['success_count'] += 1
                method_stats[method]['times'].append(rec_time)
                
                # Update running averages
                count = method_stats[method]['count']
                method_stats[method]['avg_time_ms'] = (
                    (method_stats[method]['avg_time_ms'] * (count - 1) + rec_time) / count
                )
                method_stats[method]['avg_confidence'] = (
                    (method_stats[method]['avg_confidence'] * (count - 1) + confidence) / count
                )
            
            # Calculate percentiles and success rates
            for method, stats in method_stats.items():
                if stats['times']:
                    stats['p95_time_ms'] = np.percentile(stats['times'], 95)
                    stats['p99_time_ms'] = np.percentile(stats['times'], 99)
                    stats['success_rate'] = stats['success_count'] / stats['count'] * 100
                    del stats['times']  # Remove raw data for cleaner output
            
            return {
                'time_range': {
                    'start_time': datetime.fromtimestamp(start_time).isoformat(),
                    'end_time': datetime.fromtimestamp(end_time).isoformat(),
                    'total_records': len(records)
                },
                'method_breakdown': dict(method_stats)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get detailed metrics: {e}")
            return {}
    
    def _get_sqlite_performance_metrics(self) -> Dict:
        """Get SQLite-specific performance metrics"""
        try:
            cursor = self.vector_store.connection.cursor()
            
            # Database statistics
            cursor.execute("SELECT page_count * page_size as db_size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
            
            cursor.execute("PRAGMA cache_size")
            cache_size = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
            
            # Query performance statistics
            vector_store_stats = self.vector_store.get_statistics()
            
            sqlite_metrics = {
                'database_size_mb': db_size / (1024 * 1024),
                'cache_size_kb': abs(cache_size) if cache_size < 0 else cache_size * 1024,
                'total_features': vector_store_stats.get('total_features', 0),
                'total_vectors': vector_store_stats.get('total_vectors', 0),
                'total_items': vector_store_stats.get('total_items', 0),
                'avg_search_time_ms': vector_store_stats.get('performance', {}).get('avg_search_time_ms', 0),
                'total_searches': vector_store_stats.get('performance', {}).get('search_count', 0)
            }
            
            return sqlite_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get SQLite metrics: {e}")
            return {}
    
    def _get_platform_optimization_report(self) -> Dict:
        """Get platform-specific optimization analysis"""
        platform_report = {
            'current_platform': self.platform_config['platform_type'],
            'optimization_tier': self.platform_config['optimization_tier'],
            'performance_targets': {
                'recognition_time_ms': self.target_recognition_time_ms,
                'batch_size': self.platform_config.get('batch_size', 8),
                'memory_limit_mb': self.platform_config.get('memory_limit_mb', 1024)
            },
            'actual_performance': {
                'avg_recognition_time_ms': self.performance_stats['avg_recognition_time_ms'],
                'p95_recognition_time_ms': self.performance_stats['p95_recognition_time_ms'],
                'system_health_score': self.performance_stats['system_health_score']
            }
        }
        
        # Performance comparison
        target_time = self.target_recognition_time_ms
        actual_time = self.performance_stats['avg_recognition_time_ms']
        
        if actual_time > 0:
            platform_report['performance_ratio'] = actual_time / target_time
            platform_report['meets_target'] = actual_time <= target_time
        
        return platform_report
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Recognition time analysis
        avg_time = self.performance_stats['avg_recognition_time_ms']
        target_time = self.target_recognition_time_ms
        
        if avg_time > target_time * 1.2:
            recommendations.append(
                f"Recognition time ({avg_time:.1f}ms) exceeds target ({target_time}ms) by 20%. "
                "Consider enabling refinement caching or upgrading hardware."
            )
        
        # Accuracy analysis
        accuracy = self.performance_stats['accuracy_rate']
        target_accuracy = self.target_accuracy_percent
        
        if accuracy < target_accuracy:
            recommendations.append(
                f"Accuracy ({accuracy:.1f}%) below target ({target_accuracy}%). "
                "Check feature quality and consider refiner model training."
            )
        
        # SQLite optimization
        sqlite_metrics = self._get_sqlite_performance_metrics()
        db_size_mb = sqlite_metrics.get('database_size_mb', 0)
        
        if db_size_mb > 1000:  # 1GB
            recommendations.append(
                "Database size exceeds 1GB. Consider implementing data archiving or "
                "increasing SQLite cache size for better performance."
            )
        
        # Platform-specific recommendations
        platform_type = self.platform_config['platform_type']
        
        if platform_type == 'CPU_Only' and avg_time > target_time:
            recommendations.append(
                "CPU-only processing detected with slow performance. "
                "Consider upgrading to GPU-accelerated hardware for 3-5x speedup."
            )
        
        if not recommendations:
            recommendations.append("System performance is meeting all targets. No optimizations needed.")
        
        return recommendations
    
    def _start_cleanup_task(self):
        """Start background task to clean up old metrics"""
        # Placeholder for cleanup task
        # In full implementation, this would start a background thread
        # to periodically remove old metrics based on retention policy
        pass
    
    def get_real_time_metrics(self) -> Dict:
        """Get current real-time metrics for monitoring dashboard"""
        with self.stats_lock:
            recent_times = list(self.recent_metrics['recognition_times'])
            recent_confidences = list(self.recent_metrics['confidence_scores'])
            recent_successes = list(self.recent_metrics['success_count'])
            
            metrics = {
                'current_stats': dict(self.performance_stats),
                'recent_performance': {
                    'recognition_times_ms': recent_times[-10:] if recent_times else [],
                    'confidence_scores': recent_confidences[-10:] if recent_confidences else [],
                    'success_rate_last_100': np.mean(recent_successes[-100:]) * 100 if recent_successes else 0,
                    'recent_errors': list(self.recent_metrics['error_count'])[-5:]
                },
                'system_status': {
                    'health_score': self.performance_stats['system_health_score'],
                    'status': self._get_system_status(),
                    'platform': self.platform_config['platform_type'],
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            return metrics
    
    def _get_system_status(self) -> str:
        """Get current system status based on health score"""
        health_score = self.performance_stats['system_health_score']
        
        if health_score >= 95:
            return "excellent"
        elif health_score >= 85:
            return "good"
        elif health_score >= 70:
            return "fair"
        elif health_score >= 50:
            return "poor"
        else:
            return "critical"
    
    def export_performance_data(self, output_path: str, time_range_hours: int = 168) -> bool:
        """
        Export performance data to JSON file
        
        Args:
            output_path: Path to save the exported data
            time_range_hours: Time range in hours (default: 1 week)
            
        Returns:
            Success status
        """
        try:
            report = self.get_performance_report(time_range_hours)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Performance data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export performance data: {e}")
            return False


def create_performance_monitor(vector_store: SQLiteVectorStore, config_path: Optional[str] = None) -> SQLitePerformanceMonitor:
    """
    Factory function to create performance monitor
    """
    config = {}
    if config_path:
        try:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                config = full_config.get('monitoring', {})
        except Exception as e:
            logger.warning(f"⚠️ Failed to load monitoring config: {e}")
    
    return SQLitePerformanceMonitor(vector_store, config)


def main():
    """Main entry point for performance monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SQLite Recognition Performance Monitor')
    parser.add_argument('--config', type=str, help='Configuration YAML file')
    parser.add_argument('--database', type=str, required=True, help='SQLite database path')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--export', type=str, help='Export performance data to JSON file')
    parser.add_argument('--time-range', type=int, default=24, help='Time range in hours for report')
    
    args = parser.parse_args()
    
    # Create vector store and performance monitor
    from ..storage.sqlite_store import create_vector_store
    vector_store = create_vector_store(args.database)
    monitor = create_performance_monitor(vector_store, args.config)
    
    if args.report:
        # Generate performance report
        report = monitor.get_performance_report(args.time_range)
        
        print(f"\n📊 SQLite Recognition Performance Report")
        print(f"   Time Range: {args.time_range} hours")
        print(f"   Platform: {report['report_metadata']['platform_type']}")
        
        overall = report['overall_performance']
        print(f"\n⚡ Overall Performance:")
        print(f"   Total Recognitions: {overall['total_recognitions']}")
        print(f"   Success Rate: {overall['accuracy_rate']:.1f}%")
        print(f"   Avg Recognition Time: {overall['avg_recognition_time_ms']:.1f}ms")
        print(f"   P95 Recognition Time: {overall['p95_recognition_time_ms']:.1f}ms")
        print(f"   System Health Score: {overall['system_health_score']:.1f}/100")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    if args.export:
        # Export performance data
        success = monitor.export_performance_data(args.export, args.time_range)
        if success:
            print(f"✅ Performance data exported to {args.export}")
        else:
            print("❌ Failed to export performance data")
    
    if not args.report and not args.export:
        # Show real-time metrics
        metrics = monitor.get_real_time_metrics()
        
        print(f"\n📊 Real-time SQLite Recognition Metrics")
        print(f"   System Status: {metrics['system_status']['status'].upper()}")
        print(f"   Health Score: {metrics['system_status']['health_score']:.1f}/100")
        print(f"   Total Recognitions: {metrics['current_stats']['total_recognitions']}")
        print(f"   Current Success Rate: {metrics['recent_performance']['success_rate_last_100']:.1f}%")
        print(f"   Platform: {metrics['system_status']['platform']}")


if __name__ == "__main__":
    main()