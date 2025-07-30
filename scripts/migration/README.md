# Migration Framework

Comprehensive migration framework for transitioning from legacy scattered file architecture to the unified storage system.

## Overview

The migration framework provides a complete solution for migrating data from the legacy AI recognition system to the new unified storage architecture with:

- **Data Integrity**: 100% data preservation with validation
- **Performance Optimization**: Platform-specific optimizations 
- **Error Handling**: Comprehensive error recovery and reporting
- **Validation**: Extensive testing and verification
- **Reporting**: Detailed migration reports and analytics

## Components

### Core Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `orchestrator.py` | Main migration coordinator | Automated workflow, error handling, reporting |
| `migrate_features.py` | Feature vector migration | HDF5 → SQLite, batch processing, validation |
| `migrate_faiss_indices.py` | Search index consolidation | FAISS optimization, platform-aware indexing |
| `migrate_metadata.py` | Metadata extraction | Directory analysis, EXIF data, content analysis |
| `validate_migration.py` | Comprehensive validation | Data integrity, search testing, performance |

### Architecture

```
Legacy Architecture          →        Unified Storage
├── features.h5             →        ├── recognition.db (SQLite)
├── data/models/*.index     →        ├── Optimized FAISS index  
├── data/raw/item_*/        →        └── Comprehensive metadata
└── Scattered files         →            Single interface
```

## Quick Start

### 1. Basic Migration

Migrate all components with default settings:

```bash
cd scripts/migration
python orchestrator.py \
  --legacy-features ../../data/features.h5 \
  --legacy-indices ../../data/models \
  --legacy-data ../../data/raw
```

### 2. Selective Migration

Migrate only specific components:

```bash
# Features and metadata only
python orchestrator.py \
  --legacy-features ../../data/features.h5 \
  --legacy-data ../../data/raw \
  --no-indices

# Metadata only with content analysis
python orchestrator.py \
  --legacy-data ../../data/raw \
  --no-features --no-indices
```

### 3. Analysis Mode

Analyze legacy data without migrating:

```bash
python orchestrator.py \
  --legacy-features ../../data/features.h5 \
  --legacy-indices ../../data/models \
  --legacy-data ../../data/raw \
  --analyze-only
```

## Detailed Usage

### Migration Orchestrator

The orchestrator coordinates the complete migration workflow:

```bash
python orchestrator.py [options]
```

**Source Data Options:**
- `--legacy-features PATH`: Legacy features.h5 file
- `--legacy-indices DIR`: Directory with FAISS indices  
- `--legacy-data DIR`: Legacy data directory structure

**Migration Control:**
- `--no-features`: Skip feature migration
- `--no-indices`: Skip index migration
- `--no-metadata`: Skip metadata migration
- `--no-validation`: Skip validation phase
- `--no-backup`: Skip backup creation

**Processing Options:**
- `--batch-size N`: Batch size for processing (default: 100)
- `--validation-sample-size N`: Validation sample size (default: 100)
- `--no-content-analysis`: Disable image content analysis
- `--no-rebuild-indices`: Skip index optimization

**Output Options:**
- `--target-dir DIR`: Target directory (default: data)
- `--backup-dir DIR`: Backup directory path
- `--report-output PATH`: Migration report path
- `--no-report`: Skip report generation
- `--verbose`: Enable detailed logging

### Individual Component Scripts

Each component can be run independently for targeted migration:

#### Feature Migration

```bash
python migrate_features.py \
  --features-file ../../data/features.h5 \
  --data-dir ../../data \
  --batch-size 100 \
  --verbose
```

Options:
- `--analyze-only`: Analyze features file only
- `--verify-only`: Verify existing migration
- `--batch-size N`: Processing batch size

#### Index Migration

```bash
python migrate_faiss_indices.py \
  --indices-dir ../../data/models \
  --data-dir ../../data \
  --verbose
```

Options:
- `--discover-only`: Discover indices only
- `--verify-only`: Verify existing migration
- `--no-rebuild`: Skip index optimization

#### Metadata Migration

```bash
python migrate_metadata.py \
  --data-dir ../../data/raw \
  --target-dir ../../data \
  --verbose
```

Options:
- `--analyze-only`: Analyze directory structure only
- `--verify-only`: Verify existing migration
- `--no-content-analysis`: Disable content analysis

#### Migration Validation

```bash
python validate_migration.py \
  --data-dir ../../data \
  --legacy-features ../../data/features.h5 \
  --legacy-indices ../../data/models \
  --legacy-data ../../data/raw \
  --output-report validation_report.json \
  --verbose
```

## Migration Workflow

### Phase 1: Pre-migration Analysis
- Analyze legacy data structure
- Validate compatibility
- Estimate migration time
- Generate migration strategy

### Phase 2: Backup Creation
- Create backup of existing data
- Verify backup integrity
- Prepare rollback mechanism

### Phase 3: Data Migration
- **Features**: HDF5 → SQLite vector storage
- **Indices**: FAISS consolidation and optimization
- **Metadata**: Directory structure analysis and extraction

### Phase 4: System Optimization
- Platform-specific FAISS optimization
- Memory and cache tuning
- Performance benchmarking

### Phase 5: Validation
- Data integrity verification
- Feature accuracy testing
- Search functionality validation
- Performance validation

### Phase 6: Reporting
- Comprehensive migration report
- Performance metrics
- Issue identification
- Recommendations

## Expected Performance

### Migration Times (Estimates)

| Component | Scale | Time |
|-----------|-------|------|
| Features | 1,000 images | ~2 minutes |
| Features | 10,000 images | ~15 minutes |
| Indices | 5 index files | ~3 minutes |
| Metadata | 10,000 files | ~8 minutes |

### Platform Performance

| Platform | Recognition Time | Search Time | Batch Size |
|----------|-----------------|-------------|------------|
| NVIDIA GPU | 0.15s | <50ms | 32 |
| Apple Silicon | 0.25s | <100ms | 8 |
| CPU-only | 0.35s | <200ms | 4 |

## Migration Reports

The migration framework generates comprehensive reports including:

### Summary Metrics
- Total migration time
- Components migrated
- Success/failure rates
- Performance benchmarks

### Validation Results
- Data integrity scores
- Feature accuracy metrics
- Search functionality tests
- System health checks

### Issue Tracking
- Critical errors
- Warnings and recommendations
- Performance bottlenecks
- Optimization suggestions

## Troubleshooting

### Common Issues

**"No legacy data sources specified"**
- Ensure at least one source is provided
- Check file/directory paths exist
- Verify read permissions

**"FAISS not available"**
- Install faiss-cpu: `pip install faiss-cpu`
- For GPU: `pip install faiss-gpu`

**"Migration validation failed"**
- Check data integrity scores
- Verify all components migrated
- Review validation report details

**"Performance targets not met"**
- Check platform detection results
- Verify hardware acceleration
- Consider system optimization

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
python orchestrator.py --verbose [other options]
```

This provides:
- Detailed progress information
- Error stack traces
- Performance timings
- Component-specific logs

### Recovery Procedures

**Migration Failure Recovery:**
1. Check error logs in `data/logs/`
2. Restore from backup if needed
3. Run individual components separately
4. Use `--verify-only` modes to check status

**Partial Migration:**
- Use component-specific scripts
- Skip completed components with `--no-*` flags
- Validate each component separately

## Integration

### With Existing Systems

The migration framework integrates with:
- **Unified Storage System**: Automatic initialization
- **Platform Detection**: Hardware optimization
- **Configuration Management**: Settings optimization
- **Cross-platform Extraction**: Feature processing

### Post-Migration

After successful migration:
1. Update application code to use unified storage
2. Archive legacy data files
3. Update documentation and workflows
4. Monitor system performance

## Best Practices

### Before Migration
- **Backup**: Always create backups
- **Analysis**: Run analysis mode first
- **Dependencies**: Verify all requirements installed
- **Space**: Ensure sufficient disk space

### During Migration
- **Monitoring**: Watch logs for issues
- **Resources**: Avoid heavy system load
- **Interruption**: Use Ctrl+C for clean shutdown

### After Migration
- **Validation**: Review validation reports
- **Testing**: Test application functionality
- **Performance**: Monitor system performance
- **Cleanup**: Archive legacy data safely

## Support

For issues or questions:
1. Check troubleshooting section
2. Review migration logs
3. Run validation tests
4. Check system requirements

The migration framework is designed to be robust and handle most scenarios automatically, but detailed logging and validation ensure any issues can be quickly identified and resolved.