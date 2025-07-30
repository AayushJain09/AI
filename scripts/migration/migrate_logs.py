#!/usr/bin/env python3
"""
Log Migration Script

Migrates existing log files from legacy locations to unified storage system
and structures them for analytics processing in DuckDB.

MIGRATION PROCESS:
1. Scan all log directories for existing log files
2. Parse log entries and extract structured data
3. Categorize logs by type (system, recognition, training, api)
4. Store structured log data in DuckDB analytics database
5. Create searchable indices for efficient log queries
6. Preserve original log files as backup

LOG SOURCES:
- System logs: logs/system.log, frontend/logs/system.log
- Recognition logs: logs/recognition.log, frontend/logs/recognition.log  
- Training logs: logs/training.log, frontend/logs/training.log
- API logs: logs/api.log, backend/logs/api.log, frontend/backend/logs/api.log
- Migration logs: scripts/migration/*/logs/*.log
- Unified storage logs: data/logs/unified_storage.log

The migration creates a unified log analytics system with platform optimization
and cross-platform compatibility.
"""

import os
import sys
import re
import json
import logging
import sqlite3
import duckdb
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import glob
from collections import defaultdict

# Add src to path for unified storage imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unified_storage.sqlite_store import SQLiteVectorStore, create_vector_store
from unified_storage.config_manager import ConfigManager


@dataclass
class LogMigrationStats:
    """Statistics for log migration process."""
    log_files_discovered: int = 0
    log_entries_parsed: int = 0
    log_entries_migrated: int = 0
    log_categories_found: int = 0
    migration_time_seconds: float = 0.0
    total_log_size_mb: float = 0.0
    parsing_errors: int = 0
    duplicate_entries: int = 0
    error_messages: List[str] = field(default_factory=list)
    category_distribution: Dict[str, int] = field(default_factory=dict)


class LogMigrator:
    """
    Migrates logs from legacy locations to unified analytics system.
    
    Parses various log formats, extracts structured data, and stores
    in DuckDB for analytics and performance monitoring.
    """
    
    def __init__(self, 
                 project_root: str,
                 target_dir: str,
                 preserve_originals: bool = True):
        """
        Initialize log migrator.
        
        Args:
            project_root: Root directory of the project
            target_dir: Target directory for unified storage
            preserve_originals: Whether to keep original log files
        """
        self.project_root = Path(project_root)
        self.target_dir = Path(target_dir)
        self.preserve_originals = preserve_originals
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.stats = LogMigrationStats()
        
        # Initialize log parsing patterns
        self._setup_log_patterns()
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate migration inputs and prerequisites."""
        if not self.project_root.exists():
            raise FileNotFoundError(f"Project root not found: {self.project_root}")
        
        if not self.project_root.is_dir():
            raise ValueError(f"Path is not a directory: {self.project_root}")
        
        # Create target directories if needed
        self.target_dir.mkdir(parents=True, exist_ok=True)
        (self.target_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"✅ Validation passed for log migration")
    
    def _setup_log_patterns(self):
        """Setup regex patterns for parsing different log formats."""
        # Common log patterns for various systems
        self.patterns = {
            # Standard Python logging format: 2025-07-30 01:00:03,636 - module - LEVEL - message
            'python_standard': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s*-\s*'
                r'(?P<module>[^\s]+)\s*-\s*(?P<level>\w+)\s*-\s*(?P<message>.*)'
            ),
            
            # Extended Python format with function info: 2025-07-30 01:00:03,636 - module.Class - LEVEL - [function:line] - message
            'python_extended': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s*-\s*'
                r'(?P<module>[^\s]+)\s*-\s*(?P<level>\w+)\s*-\s*'
                r'\[(?P<function>[^:]+):(?P<line>\d+)\]\s*-\s*(?P<message>.*)'
            ),
            
            # Simple timestamp format: [2025-07-30 01:00:03] LEVEL: message
            'simple_timestamp': re.compile(
                r'\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*'
                r'(?P<level>\w+):\s*(?P<message>.*)'
            ),
            
            # API log format: 127.0.0.1 - - [30/Jul/2025:01:00:03 +0000] "GET /api/recognize" 200 1234
            'api_access': re.compile(
                r'(?P<ip>[\d\.]+)\s+-\s+-\s+\[(?P<timestamp>[^\]]+)\]\s+'
                r'"(?P<method>\w+)\s+(?P<endpoint>[^"]*?)"\s+(?P<status>\d+)\s+(?P<size>\d+)'
            ),
            
            # Performance log format: PERF: operation_name took 123.45ms (details)
            'performance': re.compile(
                r'PERF:\s*(?P<operation>\w+)\s+took\s+(?P<duration>[\d\.]+)(?P<unit>ms|s)\s*'
                r'(?:\((?P<details>.*?)\))?'
            ),
            
            # Error traceback start
            'traceback_start': re.compile(r'Traceback \(most recent call last\):'),
            
            # Migration log format: specific to our migration scripts
            'migration': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s*-\s*'
                r'(?P<script>[^\s]+)\s*-\s*(?P<level>\w+)\s*-\s*(?P<message>.*)'
            )
        }
    
    def discover_log_files(self) -> Dict[str, List[Path]]:
        """
        Discover all log files in the project structure.
        
        Returns dictionary organized by log category.
        """
        self.logger.info("🔍 Discovering log files...")
        
        log_discovery = {
            'system_logs': [],
            'recognition_logs': [],
            'training_logs': [],
            'api_logs': [],
            'migration_logs': [],
            'unified_storage_logs': [],
            'other_logs': []
        }
        
        # Define search patterns for different log types
        search_patterns = [
            # Main log directories
            'logs/*.log',
            'frontend/logs/*.log',
            'backend/logs/*.log',
            'frontend/backend/logs/*.log',
            
            # Data directory logs
            'data/logs/*.log',
            
            # Migration logs
            'scripts/migration/*/logs/*.log',
            'scripts/migration/data*/logs/*.log',
            
            # Any other .log files
            '**/*.log'
        ]
        
        discovered_files = set()  # Use set to avoid duplicates
        
        # Search for log files using glob patterns
        for pattern in search_patterns:
            full_pattern = str(self.project_root / pattern)
            
            for log_file in glob.glob(full_pattern, recursive=True):
                log_path = Path(log_file)
                
                # Skip if already found or if it's empty
                if log_path in discovered_files:
                    continue
                
                discovered_files.add(log_path)
                
                # Categorize by filename and path
                file_name = log_path.name.lower()
                path_str = str(log_path).lower()
                
                if 'system' in file_name:
                    log_discovery['system_logs'].append(log_path)
                elif 'recognition' in file_name or 'recognize' in file_name:
                    log_discovery['recognition_logs'].append(log_path)
                elif 'training' in file_name or 'train' in file_name:
                    log_discovery['training_logs'].append(log_path)
                elif 'api' in file_name:
                    log_discovery['api_logs'].append(log_path)
                elif 'migration' in path_str or 'migrate' in file_name:
                    log_discovery['migration_logs'].append(log_path)
                elif 'unified_storage' in file_name:
                    log_discovery['unified_storage_logs'].append(log_path)
                else:
                    log_discovery['other_logs'].append(log_path)
        
        # Update statistics
        total_files = sum(len(files) for files in log_discovery.values())
        self.stats.log_files_discovered = total_files
        
        # Log discovery results
        self.logger.info(f"📊 Log Discovery Results:")
        for category, files in log_discovery.items():
            if files:
                self.logger.info(f"   {category}: {len(files)} files")
                for file_path in files[:3]:  # Show first 3 files
                    self.logger.info(f"     - {file_path}")
                if len(files) > 3:
                    self.logger.info(f"     ... and {len(files) - 3} more")
        
        return log_discovery
    
    def parse_log_file(self, log_file: Path) -> List[Dict[str, Any]]:
        """
        Parse a single log file and extract structured entries.
        
        Args:
            log_file: Path to log file to parse
            
        Returns:
            List of parsed log entries as dictionaries
        """
        entries = []
        current_traceback = []
        in_traceback = False
        
        try:
            # Check file size
            file_size = log_file.stat().st_size
            self.stats.total_log_size_mb += file_size / (1024 * 1024)
            
            if file_size == 0:
                self.logger.debug(f"Skipping empty log file: {log_file}")
                return entries
                
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this is a traceback
                    if self.patterns['traceback_start'].match(line):
                        in_traceback = True
                        current_traceback = [line]
                        continue
                    
                    # If we're in a traceback, collect lines
                    if in_traceback:
                        current_traceback.append(line)
                        # End traceback when we see a line that doesn't start with spaces
                        if not line.startswith(' ') and not line.startswith('\t'):
                            # Process the complete traceback
                            traceback_entry = {
                                'timestamp': datetime.now().isoformat(),
                                'level': 'ERROR',
                                'message': '\n'.join(current_traceback),
                                'log_type': 'traceback',
                                'source_file': str(log_file),
                                'line_number': line_num - len(current_traceback) + 1,
                                'raw_line': line
                            }
                            entries.append(traceback_entry)
                            in_traceback = False
                            current_traceback = []
                        continue
                    
                    # Try to parse with different patterns
                    parsed_entry = None
                    
                    for pattern_name, pattern in self.patterns.items():
                        if pattern_name in ['traceback_start']:
                            continue
                            
                        match = pattern.match(line)
                        if match:
                            parsed_entry = match.groupdict()
                            parsed_entry['pattern_type'] = pattern_name
                            break
                    
                    # If no pattern matched, create a generic entry
                    if not parsed_entry:
                        parsed_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'level': 'INFO',
                            'message': line,
                            'pattern_type': 'unparsed'
                        }
                        
                    # Add metadata
                    parsed_entry.update({
                        'source_file': str(log_file),
                        'line_number': line_num,
                        'log_category': self._categorize_log_file(log_file),
                        'raw_line': line
                    })
                    
                    # Process timestamp if present
                    if 'timestamp' in parsed_entry and parsed_entry['timestamp']:
                        parsed_entry['parsed_timestamp'] = self._parse_timestamp(
                            parsed_entry['timestamp']
                        )
                    
                    entries.append(parsed_entry)
                    self.stats.log_entries_parsed += 1
        
        except Exception as e:
            self.logger.error(f"❌ Failed to parse log file {log_file}: {e}")
            self.stats.parsing_errors += 1
            self.stats.error_messages.append(f"Parse error in {log_file}: {e}")
        
        return entries
    
    def _categorize_log_file(self, log_file: Path) -> str:
        """Categorize log file based on name and path."""
        file_name = log_file.name.lower()
        path_str = str(log_file).lower()
        
        if 'system' in file_name:
            return 'system'
        elif 'recognition' in file_name or 'recognize' in file_name:
            return 'recognition'
        elif 'training' in file_name or 'train' in file_name:
            return 'training'
        elif 'api' in file_name:
            return 'api'
        elif 'migration' in path_str or 'migrate' in file_name:
            return 'migration'
        elif 'unified_storage' in file_name:
            return 'unified_storage'
        else:
            return 'other'
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[str]:
        """Parse various timestamp formats to ISO format."""
        timestamp_formats = [
            '%Y-%m-%d %H:%M:%S,%f',  # Python logging format
            '%Y-%m-%d %H:%M:%S',     # Standard format
            '%d/%b/%Y:%H:%M:%S %z',  # Apache log format
            '%Y-%m-%dT%H:%M:%S',     # ISO format
        ]
        
        for fmt in timestamp_formats:
            try:
                dt = datetime.strptime(timestamp_str.strip(), fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        # If no format matched, return None
        return None
    
    def setup_analytics_database(self) -> duckdb.DuckDBPyConnection:
        """
        Setup DuckDB analytics database for log storage.
        
        Creates optimized schema for log analytics and performance monitoring.
        """
        self.logger.info("🗄️ Setting up analytics database...")
        
        # Create analytics database path with timestamp to avoid locks
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analytics_db_path = self.target_dir / f"logs_analytics_{timestamp}.duckdb"
        
        # Connect to DuckDB
        conn = duckdb.connect(str(analytics_db_path))
        
        # Create optimized log table schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id BIGINT PRIMARY KEY,
                timestamp TIMESTAMP,
                parsed_timestamp TIMESTAMP,
                level VARCHAR(20),
                message TEXT,
                log_category VARCHAR(50),
                pattern_type VARCHAR(50),
                source_file VARCHAR(500),
                line_number INTEGER,
                module VARCHAR(200),
                function VARCHAR(200),
                
                -- API specific fields
                ip_address VARCHAR(50),
                http_method VARCHAR(10),
                endpoint VARCHAR(500),
                status_code INTEGER,
                response_size INTEGER,
                
                -- Performance specific fields
                operation VARCHAR(200),
                duration_ms FLOAT,
                details TEXT,
                
                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                migration_batch VARCHAR(100),
                raw_line TEXT
            )
        """)
        
        # Create indices for efficient querying
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)",
            "CREATE INDEX IF NOT EXISTS idx_logs_category ON logs(log_category)",
            "CREATE INDEX IF NOT EXISTS idx_logs_source ON logs(source_file)",
            "CREATE INDEX IF NOT EXISTS idx_logs_created ON logs(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_logs_operation ON logs(operation)",
        ]
        
        for index_sql in indices:
            conn.execute(index_sql)
        
        # Create performance monitoring views
        conn.execute("""
            CREATE VIEW IF NOT EXISTS performance_summary AS
            SELECT 
                log_category,
                operation,
                COUNT(*) as operation_count,
                AVG(duration_ms) as avg_duration_ms,
                MIN(duration_ms) as min_duration_ms,
                MAX(duration_ms) as max_duration_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration_ms
            FROM logs 
            WHERE duration_ms IS NOT NULL
            GROUP BY log_category, operation
        """)
        
        # Create error analysis view
        conn.execute("""
            CREATE VIEW IF NOT EXISTS error_analysis AS
            SELECT 
                log_category,
                level,
                DATE_TRUNC('hour', timestamp) as hour,
                COUNT(*) as error_count,
                COUNT(DISTINCT source_file) as affected_files
            FROM logs 
            WHERE level IN ('ERROR', 'CRITICAL', 'FATAL')
            GROUP BY log_category, level, DATE_TRUNC('hour', timestamp)
            ORDER BY hour DESC
        """)
        
        self.logger.info(f"✅ Analytics database setup complete: {analytics_db_path}")
        return conn
    
    def migrate_logs_to_analytics(self, log_entries: List[Dict[str, Any]], 
                                conn: duckdb.DuckDBPyConnection,
                                batch_id: str) -> int:
        """
        Migrate parsed log entries to DuckDB analytics database.
        
        Args:
            log_entries: List of parsed log entry dictionaries
            conn: DuckDB connection
            batch_id: Identifier for this migration batch
            
        Returns:
            Number of entries successfully migrated
        """
        if not log_entries:
            return 0
        
        migrated_count = 0
        batch_size = 1000  # Process in batches for efficiency
        
        for i in range(0, len(log_entries), batch_size):
            batch = log_entries[i:i + batch_size]
            
            try:
                # Prepare batch data for insertion
                batch_data = []
                
                for entry in batch:
                    # Extract and clean data for insertion
                    row_data = {
                        'id': int(time.time() * 1000000) + i + len(batch_data),  # Unique ID
                        'timestamp': entry.get('parsed_timestamp'),
                        'parsed_timestamp': entry.get('parsed_timestamp'),
                        'level': entry.get('level', 'INFO'),
                        'message': entry.get('message', '')[:10000],  # Limit message length
                        'log_category': entry.get('log_category', 'other'),
                        'pattern_type': entry.get('pattern_type', 'unknown'),
                        'source_file': entry.get('source_file', ''),
                        'line_number': entry.get('line_number'),
                        'module': entry.get('module'),
                        'function': entry.get('function'),
                        
                        # API fields
                        'ip_address': entry.get('ip'),
                        'http_method': entry.get('method'),
                        'endpoint': entry.get('endpoint'),
                        'status_code': int(entry.get('status', 0)) if entry.get('status') else None,
                        'response_size': int(entry.get('size', 0)) if entry.get('size') else None,
                        
                        # Performance fields
                        'operation': entry.get('operation'),
                        'duration_ms': float(entry.get('duration', 0)) if entry.get('duration') else None,
                        'details': entry.get('details'),
                        
                        # Metadata
                        'migration_batch': batch_id,
                        'raw_line': entry.get('raw_line', '')[:5000]  # Limit raw line length
                    }
                    
                    batch_data.append(row_data)
                
                # Insert batch using DuckDB's efficient batch insert
                conn.executemany("""
                    INSERT INTO logs (
                        id, timestamp, parsed_timestamp, level, message, log_category,
                        pattern_type, source_file, line_number, module, function,
                        ip_address, http_method, endpoint, status_code, response_size,
                        operation, duration_ms, details, migration_batch, raw_line
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, [
                    (
                        row['id'], row['timestamp'], row['parsed_timestamp'], 
                        row['level'], row['message'], row['log_category'],
                        row['pattern_type'], row['source_file'], row['line_number'],
                        row['module'], row['function'], row['ip_address'],
                        row['http_method'], row['endpoint'], row['status_code'],
                        row['response_size'], row['operation'], row['duration_ms'],
                        row['details'], row['migration_batch'], row['raw_line']
                    ) for row in batch_data
                ])
                
                migrated_count += len(batch_data)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to migrate batch {i//batch_size}: {e}")
                self.stats.error_messages.append(f"Batch migration error: {e}")
        
        return migrated_count
    
    def migrate_all_logs(self) -> LogMigrationStats:
        """
        Migrate all discovered logs to analytics database.
        
        Performs complete log migration with progress tracking.
        """
        self.logger.info("🚀 Starting log migration...")
        start_time = time.time()
        
        try:
            # Discover all log files
            log_discovery = self.discover_log_files()
            
            if not any(log_discovery.values()):
                self.logger.info("ℹ️ No log files found to migrate")
                return self.stats
            
            # Setup analytics database
            analytics_conn = self.setup_analytics_database()
            
            # Create batch identifier
            batch_id = f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Process each category of logs
            all_entries = []
            
            for category, log_files in log_discovery.items():
                if not log_files:
                    continue
                    
                self.logger.info(f"📄 Processing {category}: {len(log_files)} files")
                
                category_entries = 0
                for log_file in log_files:
                    self.logger.debug(f"Parsing: {log_file}")
                    entries = self.parse_log_file(log_file)
                    all_entries.extend(entries)
                    category_entries += len(entries)
                
                # Update category statistics
                self.stats.category_distribution[category] = category_entries
                self.stats.log_categories_found += 1 if category_entries > 0 else 0
                
                self.logger.info(f"   Parsed {category_entries} entries from {category}")
            
            # Migrate all entries to analytics database
            if all_entries:
                self.logger.info(f"💾 Migrating {len(all_entries)} log entries to analytics database...")
                migrated_count = self.migrate_logs_to_analytics(all_entries, analytics_conn, batch_id)
                self.stats.log_entries_migrated = migrated_count
            
            # Close analytics connection
            analytics_conn.close()
            
            # Calculate final statistics
            end_time = time.time()
            self.stats.migration_time_seconds = end_time - start_time
            
            # Log final results
            self.logger.info("✅ Log migration completed!")
            self.logger.info(f"📊 Migration Statistics:")
            self.logger.info(f"   Log files discovered: {self.stats.log_files_discovered}")
            self.logger.info(f"   Log entries parsed: {self.stats.log_entries_parsed}")
            self.logger.info(f"   Log entries migrated: {self.stats.log_entries_migrated}")
            self.logger.info(f"   Log categories: {self.stats.log_categories_found}")
            self.logger.info(f"   Total log size: {self.stats.total_log_size_mb:.1f}MB")
            self.logger.info(f"   Migration time: {self.stats.migration_time_seconds:.1f}s")
            self.logger.info(f"   Parsing errors: {self.stats.parsing_errors}")
            
            if self.stats.category_distribution:
                self.logger.info(f"   Category distribution:")
                for category, count in sorted(self.stats.category_distribution.items()):
                    self.logger.info(f"     {category}: {count} entries")
            
            if self.stats.error_messages:
                self.logger.warning(f"⚠️ {len(self.stats.error_messages)} errors occurred")
                for error in self.stats.error_messages[:5]:  # Show first 5 errors
                    self.logger.warning(f"   - {error}")
        
        except Exception as e:
            self.logger.error(f"❌ Migration failed: {e}")
            self.stats.error_messages.append(f"Migration error: {e}")
        
        return self.stats


def main():
    """Main entry point for log migration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Migrate logs from legacy locations to unified analytics storage'
    )
    parser.add_argument('--project-root', type=str, default='.',
                       help='Project root directory')
    parser.add_argument('--target-dir', type=str, default='data',
                       help='Target data directory for unified storage')
    parser.add_argument('--preserve-originals', action='store_true', default=True,
                       help='Preserve original log files')
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
        # Create log migrator
        migrator = LogMigrator(
            project_root=args.project_root,
            target_dir=args.target_dir,
            preserve_originals=args.preserve_originals
        )
        
        # Perform migration
        logger.info("📊 Starting log migration process...")
        stats = migrator.migrate_all_logs()
        
        # Print final summary
        print("\n" + "="*50)
        print("LOG MIGRATION SUMMARY")
        print("="*50)
        print(f"Log Files Discovered: {stats.log_files_discovered}")
        print(f"Log Entries Parsed: {stats.log_entries_parsed}")
        print(f"Log Entries Migrated: {stats.log_entries_migrated}")
        print(f"Log Categories: {stats.log_categories_found}")
        print(f"Total Log Size: {stats.total_log_size_mb:.1f}MB")
        print(f"Migration Time: {stats.migration_time_seconds:.1f}s")
        print(f"Parsing Errors: {stats.parsing_errors}")
        
        if stats.category_distribution:
            print(f"\nCategory Distribution:")
            for category, count in sorted(stats.category_distribution.items()):
                print(f"  {category}: {count}")
        
        if stats.error_messages:
            print(f"\nErrors ({len(stats.error_messages)}):")
            for error in stats.error_messages[:10]:
                print(f"  - {error}")
        
        logger.info("✅ Log migration process completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())