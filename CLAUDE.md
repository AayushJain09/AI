# AI Recognition System - Claude Code Instructions

## Project Overview

This is a **cross-platform AI recognition system** being upgraded from scattered file architecture to a unified storage system with hybrid SQLite + DuckDB backend. The system provides visual object recognition with 0.15s-0.35s recognition times across Windows (NVIDIA), Apple Silicon, and Intel Mac platforms.

## Core Architecture

### Current Implementation Status
- ✅ **Platform Detection**: Auto-detects hardware and optimizes settings (`src/unified_storage/platform_detector.py`)
- ✅ **Configuration Management**: Platform-specific optimization (`src/unified_storage/config_manager.py`)
- ✅ **Test Suite**: Comprehensive platform validation (`tests/test_platform_detection.py`)
- ✅ **Legacy Recognition**: Existing pipeline working (`src/inference/recognize.py`)
- ✅ **Feature Extraction**: CLIP + DINOv2 pipeline (`src/feature_extraction/feature_extractor.py`)
- ✅ **FAISS Indexing**: Basic indexing system (`src/indexing/faiss_indexer.py`)
- ✅ **Frontend Interface**: GUI with recognition widgets (`frontend/`)
- 🟡 **Database Layer**: In progress - SQLite + DuckDB hybrid storage
- 🔴 **Vector Storage**: Pending - SQLite-based vector storage (`src/unified_storage/sqlite_store.py`)
- 🔴 **Analytics Layer**: Pending - DuckDB analytics (`src/unified_storage/analytics_store.py`)
- 🔴 **Migration Tools**: Pending - Legacy to unified storage migration

### Performance Targets
- **NVIDIA GPU**: 0.15s recognition, GPU FAISS, 32 batch size
- **Apple Silicon**: 0.25s recognition, CPU FAISS, 8 batch size  
- **CPU-only**: 0.35s recognition, CPU FAISS, 4 batch size

## Development Standards

### Code Style & Quality
- **Python 3.8+** required
- **Type hints mandatory** for all function signatures
- **Docstrings required** for all classes and functions
- **Inline documentation MANDATORY**: Add comprehensive inline comments explaining:
  - **WHY** each optimization decision was made
  - **Platform-specific** reasoning (e.g., "Apple Silicon has unified memory")
  - **Performance implications** of each setting
  - **Rationale** behind thresholds, limits, and calculations
  - **Hardware considerations** that drive the implementation
- **Error handling**: Graceful fallbacks, never crash
- **Logging**: Use structured logging with appropriate levels

### Architecture Principles
1. **Cross-platform first**: Must work on Windows, macOS, Linux
2. **Performance optimization**: Platform-specific settings automatically applied
3. **Graceful degradation**: System works even without optimal hardware
4. **Configuration-driven**: User overrides always respected
5. **Memory conscious**: Optimize for available system resources

### File Organization
```
Project Root/
├── CLAUDE.md                   # This instruction file - ALWAYS reference
├── QUICK_START.md             # User quick start guide
├── config.yaml                # User configuration overrides
├── requirements.txt           # Package dependencies
├── main.py                    # Legacy main entry point
├── start_system.py            # New system entry point
├── evaluation_results.json    # Performance evaluation data
│
├── src/                       # Core source code
│   ├── unified_storage/       # NEW: Core storage architecture
│   │   ├── __init__.py        # Module exports
│   │   ├── platform_detector.py  # Hardware detection & optimization
│   │   ├── config_manager.py     # Configuration management
│   │   ├── sqlite_store.py       # SQLite vector storage (PENDING)
│   │   └── analytics_store.py    # DuckDB analytics layer (PENDING)
│   ├── inference/
│   │   ├── __init__.py
│   │   └── recognize.py       # Main recognition pipeline
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   └── feature_extractor.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── faiss_indexer.py
│   └── data_preparation/
│       ├── __init__.py
│       └── prepare.py
│
├── data/                      # Data storage
│   ├── raw/                   # Original item images (item_001/ to item_026/)
│   ├── augmented/             # Augmented training data
│   ├── models/                # FAISS indices and metadata
│   └── features.h5            # Extracted feature vectors
│
├── tests/                     # Test suites
│   ├── test_platform_detection.py  # Platform detection tests
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── system/                # System-level tests
│
├── docs/                      # Documentation
│   ├── IMPLEMENTATION_PLAN.md # 4-week implementation timeline
│   ├── PERFORMANCE_OPTIMIZED_ARCHITECTURE.md
│   ├── README.md              # Main project documentation
│   ├── TROUBLESHOOTING_GUIDE.md
│   └── VALIDATION_PROCEDURES.md
│
├── frontend/                  # GUI Interface
│   ├── main.py               # Frontend entry point
│   └── widgets/              # UI components
│       ├── recognition.py
│       ├── training.py
│       ├── items.py
│       ├── evaluation.py
│       ├── settings.py
│       └── logs.py
│
├── backend/                   # API Backend
│   └── main.py               # API server
│
├── logs/                      # System logs
│   ├── system.log
│   ├── recognition.log
│   ├── training.log
│   └── api.log
│
├── scripts/                   # Utility scripts
├── checkpoints/               # Model checkpoints
└── benchmark_results/         # Performance benchmarks
```

### Database Architecture
- **SQLite**: Primary storage for vectors, metadata, and search indices
- **DuckDB**: Analytics, reporting, and complex queries
- **FAISS**: Vector similarity search with GPU/CPU optimization
- **Hybrid approach**: SQLite for OLTP, DuckDB for OLAP

### Testing Requirements
- **Comprehensive test coverage** for all platform combinations
- **Performance benchmarks** for optimization validation
- **Compatibility validation** before any major changes
- **Test files location**: `tests/test_*.py`

## Platform Detection Logic

### Hardware Tiers
1. **NVIDIA GPU** (Best): CUDA acceleration, GPU FAISS, large batches
2. **Apple Silicon** (Good): MPS + optimized CPU FAISS, unified memory
3. **CPU-only** (Functional): Standard CPU FAISS, conservative settings

### Automatic Optimizations
- **Cache sizing**: Based on available memory (20-50% allocation)
- **Thread optimization**: Platform-specific threading (16 max on Apple Silicon)
- **Batch sizes**: GPU memory-based calculation (2 images per GB GPU memory)
- **FAISS modes**: `gpu` > `cpu_optimized` > `cpu_standard`

## Configuration Management

### Configuration Hierarchy (priority order)
1. **Hardware constraints** (cannot be exceeded)
2. **Platform optimizations** (empirically determined)
3. **User overrides** (config.yaml)
4. **Safety limits** (prevent crashes)

### Key Configuration Files
- `config.yaml`: User customization (optional)
- `data/generated_config.yaml`: Auto-generated optimal settings
- **Environment variables**: Override any setting with `AI_RECOGNITION_*`

## Implementation Guidelines

### When Working on This Project

1. **Always check platform compatibility** before making changes
2. **Run tests** after any modifications: `python tests/test_platform_detection.py`
3. **Validate performance** doesn't degrade on any platform
4. **Follow the todo list** - use TodoWrite tool for task management
5. **Test on actual hardware** when possible
6. **ADD COMPREHENSIVE INLINE DOCUMENTATION** - explain every optimization decision
7. **Reference CLAUDE.md** - always follow these standards and project knowledge
8. **Use existing platform detection** - never recreate, always build upon existing infrastructure

### Error Handling Strategy
- **Graceful fallbacks**: Never crash, always provide CPU-only mode
- **Informative logging**: Help users understand what's happening
- **User-friendly messages**: Avoid technical jargon in user-facing errors
- **Recovery mechanisms**: Auto-retry with simpler settings on failure

### Performance Optimization Rules
- **Memory first**: Respect system memory limits
- **Platform-specific**: Don't use one-size-fits-all settings
- **User choice**: Allow manual override of any optimization
- **Conservative defaults**: Start safe, optimize based on detected capabilities

## Development Workflow

### Day-by-Day Implementation Plan
Following the 4-week implementation timeline in `docs/IMPLEMENTATION_PLAN.md`:

**Week 1**: Database infrastructure and vector storage
**Week 2**: Search engine and FAISS integration  
**Week 3**: Migration tools and system integration
**Week 4**: Testing, optimization, and deployment

### Commit Standards
- **Descriptive commits**: Explain why, not just what
- **Test before commit**: Ensure tests pass
- **Performance validation**: No degradation on any platform
- **Documentation updates**: Keep docs in sync with code

### Required Dependencies
Core packages that must be available:
```python
torch>=1.9.0          # PyTorch for feature extraction
faiss-cpu>=1.7.0       # Vector similarity search
sqlite3               # Built into Python
duckdb>=0.8.0         # Analytics database
psutil>=5.8.0         # System information
pyyaml>=6.0           # Configuration files
numpy>=1.21.0         # Numerical operations
```

## Common Tasks & Patterns

### Adding New Platform Support
1. Update `_detect_os_info()` in platform_detector.py
2. Add optimization settings in `_determine_optimization_settings()`
3. Update tests to validate new platform
4. Document performance characteristics

### Modifying Configuration
1. Update relevant dataclass in config_manager.py
2. Add platform-specific logic if needed
3. Update validation in `validate_configuration()`
4. Test across all platforms

### Performance Optimization
1. Profile on target platform first
2. Implement optimization in platform detector
3. Add configuration option for user override
4. Validate improvement with benchmarks

## Troubleshooting Guide

### Common Issues
1. **Import errors**: Check if required packages are installed
2. **GPU not detected**: Verify PyTorch CUDA/MPS installation
3. **Memory errors**: Reduce cache sizes and batch sizes
4. **Permission errors**: Check data directory write permissions

### Debug Commands
```bash
# Test platform detection
python src/unified_storage/platform_detector.py

# Test configuration management  
python src/unified_storage/config_manager.py

# Run full test suite
python tests/test_platform_detection.py
```

## Current Project State

The system is currently in **Week 1 Day 1 Afternoon** of implementation:
- ✅ Platform detection and configuration management complete
- 🟡 Next: Database infrastructure (SQLite + DuckDB foundations)
- 📋 Active todos: SQLite store and DuckDB analytics implementation

## Critical Documentation Requirements

### Inline Documentation Standards
**EVERY new function and class MUST include:**

1. **Comprehensive docstrings** with purpose, parameters, and return values
2. **Inline comments explaining WHY** (not what) for every optimization:
   ```python
   # OPTIMIZATION TIER 1: NVIDIA GPU (Best Performance)
   # CUDA GPUs provide the best performance characteristics:
   # - GPU-accelerated FAISS for 3-5x search speedup
   # - Large GPU memory enables bigger batch sizes
   # - Dedicated GPU memory reduces system memory pressure
   if gpu_info['cuda_available']:
   ```

3. **Platform-specific reasoning**:
   ```python
   # Apple Silicon specific optimizations based on empirical testing
   # Cap FAISS threads at 16 - beyond this, performance degrades due to 
   # contention between efficiency and performance cores
   settings['faiss_threads'] = min(16, settings['faiss_threads'])
   ```

4. **Performance implications** for every setting:
   ```python
   # Batch size based on GPU memory: 2 images per GB is conservative
   # but prevents OOM errors while maintaining good throughput
   'optimal_batch_size': min(32, int(gpu_info['gpu_memory_gb'] * 2))
   ```

5. **Hardware considerations** that drive implementation decisions

### Documentation Philosophy
- **Explain the WHY, not the WHAT** - code shows what, comments explain why
- **Include performance rationale** - why this setting over alternatives
- **Document platform differences** - why Apple Silicon differs from NVIDIA
- **Explain optimization trade-offs** - what we gain vs what we sacrifice

## Memory for Claude Code

When working on this project:
1. **Always prioritize cross-platform compatibility**
2. **Use the existing platform detection system** - don't recreate it
3. **Follow the established configuration patterns**
4. **Test changes across platform types**
5. **Maintain the performance optimization philosophy**
6. **Keep the todo list updated** with TodoWrite tool
7. **Reference this file** for project context and standards
8. **ADD MANDATORY INLINE DOCUMENTATION** following the standards above
9. **Explain every optimization decision** with comprehensive reasoning

---

*This instruction set ensures consistent development practices and automatic adherence to project standards. Update this file when architectural decisions change.*