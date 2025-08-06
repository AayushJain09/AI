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

### Universal Professional Standards
- **Follow industry-standard software engineering practices**
- **Implement clean code principles** (readable, maintainable, testable)
- **Apply SOLID principles** for object-oriented design
- **Use design patterns appropriately** (Factory, Strategy, Observer, etc.)
- **Implement comprehensive error handling and logging**
- **Follow semantic versioning and proper release management**
- **Maintain backward compatibility** where feasible
- **Document APIs and interfaces thoroughly**

### Modular Architecture Requirements
- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Loose Coupling**: Modules interact through well-defined interfaces
- **High Cohesion**: Related functionality grouped logically within modules
- **Dependency Injection**: Use dependency injection for testability and flexibility
- **Interface Segregation**: Create focused, role-specific interfaces
- **Plugin Architecture**: Support extensibility through plugin systems
- **Layer Separation**: Clear separation between presentation, business logic, and data layers

### Code Style & Quality
- **Python 3.8+** required
- **Type hints mandatory** for all function signatures
- **Docstrings required** for all classes and functions
- **PEP 8 compliance** with modern tooling (black, isort, flake8)
- **Unit test coverage** minimum 80% for critical components
- **Integration tests** for all major workflows
- **Inline documentation MANDATORY**: Add comprehensive inline comments explaining:
  - **WHY** each optimization decision was made
  - **Platform-specific** reasoning (e.g., "Apple Silicon has unified memory")
  - **Performance implications** of each setting
  - **Rationale** behind thresholds, limits, and calculations
  - **Hardware considerations** that drive the implementation
- **Error handling**: Graceful fallbacks, never crash
- **Logging**: Use structured logging with appropriate levels
- **Code review process**: All changes require review before merge

### Architecture Principles
1. **Cross-platform first**: Must work on Windows, macOS, Linux
2. **Performance optimization**: Platform-specific settings automatically applied
3. **Graceful degradation**: System works even without optimal hardware
4. **Configuration-driven**: User overrides always respected
5. **Memory conscious**: Optimize for available system resources
6. **Modular design**: Components can be independently developed, tested, and deployed
7. **Scalable architecture**: System can handle growth in data and users
8. **Security by design**: Security considerations built into every layer
9. **Observability**: Comprehensive monitoring, logging, and tracing
10. **Testability**: All components designed for easy testing

### File Organization (Current Enhanced System)
```
Project Root/
├── CLAUDE.md                   # ⭐ This instruction file - ALWAYS reference
├── .gitignore                  # Project-specific Git exclusions
├── GITIGNORE_GUIDE.md         # Git configuration documentation
├── requirements.txt           # Package dependencies
├── checkpoints/               # Model checkpoints directory
│   └── .gitkeep              # (lightweight_refiner.pth expected here)
│
├── new_system/               # ⭐ CURRENT ACTIVE SYSTEM
│   ├── README.md             # ⭐ Complete project documentation
│   ├── run_system.py         # ⭐ Primary entry point (GUI/CLI/test)
│   ├── launch_unified_gui.py # Alternative GUI launcher
│   │
│   ├── unified_storage/      # ⭐ CORE SYSTEM ARCHITECTURE
│   │   ├── __init__.py       # Module exports and factory functions
│   │   ├── enhanced_unified_store.py              # ✅ Core storage with proven approach
│   │   ├── enhanced_unified_store_with_recognition.py  # ✅ Complete recognition system (BEST)
│   │   ├── enhanced_recognition_pipeline.py       # ✅ State-of-the-art recognition pipeline
│   │   ├── unified_store.py                       # ✅ Base unified storage interface
│   │   ├── platform_detector.py                  # ✅ Hardware detection & optimization
│   │   ├── config_manager.py                     # ✅ Platform-specific configuration
│   │   ├── cross_platform_extractor.py           # ✅ CLIP + DINOv2 feature extraction
│   │   │
│   │   ├── gui/              # Modern GUI system
│   │   │   ├── __init__.py
│   │   │   └── unified_gui.py  # ⭐ Modern GUI with real-time processing
│   │   │
│   │   ├── preprocessing/    # ⭐ ENHANCED PREPROCESSING (ALL ACTIVE)
│   │   │   ├── __init__.py
│   │   │   ├── hybrid_db_indexer.py      # ⭐ High-performance SQLite + FAISS (43x faster)
│   │   │   ├── input_manager.py          # ✅ Unified image input management
│   │   │   ├── data_persistence.py       # ✅ Guaranteed data persistence
│   │   │   ├── faiss_indexer.py         # ✅ FAISS vector indexing
│   │   │   ├── feature_extractor.py     # ✅ Feature extraction pipeline
│   │   │   └── augmentation_adapter.py  # ✅ Augmentation pipeline adapter
│   │   │
│   │   └── [UNUSED FILES]    # Alternative implementations (not active)
│   │       ├── search_engine.py         # ❌ Standalone search (replaced)
│   │       ├── feature_storage.py       # ❌ Feature storage (replaced)
│   │       ├── sqlite_store.py         # ❌ SQLite-only storage (replaced)
│   │       ├── analytics_store.py      # ❌ DuckDB analytics (optional)
│   │       └── vector_store.py         # ❌ Generic storage (replaced)
│   │
│   ├── data/                 # ⭐ SYSTEM DATA (auto-created)
│   │   ├── .gitkeep          # Preserve directory structure
│   │   ├── recognition.db    # ⭐ SQLite database (vectors + metadata + images)
│   │   ├── models/           # Model files and indices
│   │   │   ├── hybrid_faiss_index.bin  # FAISS search index
│   │   │   └── hybrid_metadata.pkl     # Index metadata
│   │   ├── logs/             # System logs
│   │   │   └── unified_storage.log     # Main system log
│   │   └── temp_processing/  # Temporary processing (auto-cleaned)
│   │
│   └── documents/           # ⭐ PROJECT DOCUMENTATION (12 files)
│       ├── file_mapping.md                    # ⭐ Complete system reference
│       ├── ARCHITECTURE.md                   # System architecture overview
│       ├── IMAGE_STORAGE_FIX_SUMMARY.md     # Image storage implementation
│       ├── RECOGNITION_ACCURACY_FIX_SUMMARY.md  # Recognition pipeline fixes
│       ├── GENERALIZATION_FIX_SUMMARY.md    # Generalization improvements
│       ├── GUI_INDEX_METHOD_FIX_SUMMARY.md  # GUI fixes documentation
│       ├── SEARCH_FIX_SUMMARY.md           # Search functionality fixes
│       ├── CHROMADB_REMOVAL_SUMMARY.md     # ChromaDB migration guide
│       ├── INDEXING_SYSTEM_ANALYSIS.md     # Indexing system details
│       ├── RECOGNITION_SYSTEM_ANALYSIS.md  # Recognition analysis
│       ├── DETAILED_OLD_SYSTEM_ANALYSIS.md # Legacy system analysis
│       └── INCREMENTAL_UPDATES_GUIDE.md    # Update procedures
│
└── env/                     # Python virtual environment (gitignored)
```

### Database Architecture (Current Implementation)
- **SQLite**: Primary unified storage (`recognition.db`)
  - **vectors**: Feature vectors (CLIP, DINOv2, combined) as BLOBs
  - **metadata**: Image metadata and recognition results
  - **original_images**: Original image data as BLOBs with integrity checksums
  - **augmented_images**: Processed augmentations as BLOBs with parameters
  - **performance_stats**: System performance monitoring
- **FAISS**: High-performance vector similarity search (43x faster than ChromaDB)
  - **Hybrid Integration**: SQLite persistence + FAISS performance
  - **Index Types**: Auto-selected (Flat/IVF/IVF-PQ/HNSW) based on dataset size
  - **GPU Acceleration**: CUDA/MPS support with CPU fallback
- **Storage Efficiency**: Single unified database file (93% size reduction)

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

The system has **COMPLETED** integration of the original proven approach into unified storage:
- ✅ **System Segregation Complete**: Old and new systems cleanly separated
- ✅ **Enhanced Unified Storage**: Original proven approach integrated
- ✅ **Proven Accuracy Preserved**: Same 99%+ accuracy with 93% storage reduction
- ✅ **Complete Implementation**: All components functional and tested
- 📋 **Reference Location**: See `new_system/documents/file_mapping.md` for complete system reference

### **🎯 Current Implementation Status (2025 Update)**

**✅ FULLY IMPLEMENTED & ACTIVE:**
- **Entry Points**: `run_system.py` (primary), `launch_unified_gui.py` (alternative)
- **Core Storage**: `EnhancedUnifiedStore` with proven 99%+ accuracy augmentation pipeline
- **Complete Recognition**: `EnhancedUnifiedStoreWithRecognition` (most comprehensive implementation)
- **Modern GUI**: `unified_gui.py` with real-time processing and progress tracking
- **Cross-Platform**: Full NVIDIA GPU/Apple Silicon/CPU optimization with auto-detection
- **Hybrid Indexing**: SQLite + FAISS with 43x performance boost over ChromaDB
- **Image Storage**: Original and augmented images stored as BLOBs in database
- **Feature Extraction**: CLIP + DINOv2 with cross-platform optimization (1536D vectors)
- **Data Persistence**: Guaranteed SQLite storage with integrity checking
- **Error Handling**: Comprehensive recovery mechanisms and graceful fallbacks
- **Documentation**: 12 technical documents with complete implementation guides

**✅ PROVEN ACCURACY PRESERVED:**
- **Strategy Weights**: Geometric (30%), Perspective (25%), Lighting (25%), Noise/Blur (15%), Effects (5%)
- **Augmentation Pipeline**: 50 augmentations per image (empirically validated optimal)
- **Background Removal**: rembg integration with synthetic background generation
- **Performance**: 0.15s-0.35s recognition times (platform dependent)
- **Storage Efficiency**: 93% reduction (166MB vs 2.4GB original system)

**🟡 PARTIALLY IMPLEMENTED:**
- **Lightweight Refiner Model**: Module structure exists, model file missing (`checkpoints/lightweight_refiner.pth`)
  - Impact: System works with raw features, missing final accuracy boost for ambiguous cases
  - Current: Falls back gracefully, 99%+ accuracy still maintained
- **Geometric Verification**: SIFT-based spatial consistency checking (framework ready)
- **CLI Interface**: Basic commands implemented, full functionality pending

**⚠️ CRITICAL IMPLEMENTATION GAPS:**
- **GUI Architecture Issue**: Currently uses `EnhancedUnifiedStore` instead of complete `EnhancedUnifiedStoreWithRecognition`
  - Problem: GUI gets basic similarity search instead of full recognition pipeline
  - Solution: Update GUI to use `create_enhanced_unified_store_with_recognition()`
- **Recognition Method**: GUI calls `search_similar()` instead of `recognize_item()`
  - Impact: Missing confidence scoring, ensemble weighting, and advanced features

**📊 CODE UTILIZATION ANALYSIS:**
- **Active Files**: 18 of 23 files (78% utilization) - excellent efficiency
- **All Preprocessing Active**: 100% of preprocessing modules in execution chain
- **Unused Files**: 5 alternative implementations (candidates for cleanup)
- **Architecture Quality**: Clean separation, minimal dead code, well-structured

## System Segregation Reference

**IMPORTANT**: This project now has TWO complete systems:

### 🔄 OLD SYSTEM (Original Proven)
- **Location**: `system_segregation/old_system/`
- **Size**: 2.4GB (12,000+ augmented images)
- **Entry**: `python main.py`
- **Core**: `data_preparation/prepare.py` (your proven 99%+ pipeline)
- **Status**: Fully preserved and functional

### ⚡ NEW SYSTEM (Enhanced Unified)
- **Location**: `system_segregation/new_system/`
- **Size**: 166MB (93% storage reduction)
- **Entry**: `python launch_unified_gui.py`
- **Core**: `enhanced_unified_store.py` (proven approach enhanced)
- **Status**: Complete integration with same 99%+ accuracy

**📄 Complete Reference**: `/system_segregation/file_mapping.md` contains detailed mapping of all files, usage instructions, and system comparisons.

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

### **🔍 Pre-Implementation File Discovery Protocol**
**ALWAYS perform these checks BEFORE starting any work:**

1. **📁 Check for Existing Files FIRST**:
   - Use `Glob` tool to search for related files: `**/*keyword*.py`, `**/*feature*.py`
   - Use `Grep` tool to find existing implementations: `pattern="class.*Keyword|def.*function"`  
   - Read existing files to understand current implementation
   - **NEVER create duplicate files** - always extend or modify existing ones

2. **🏗️ Architecture Discovery**:
   - Check `system_segregation/file_mapping.md` for system structure
   - Review `README.md` project structure section
   - Identify which unified store component to use (`EnhancedUnifiedStore` vs `EnhancedUnifiedStoreWithRecognition`)
   - Understand the execution path: Entry Point → GUI → Core Storage → Processing Pipeline

3. **📊 Current Implementation Analysis**:
   - Trace actual usage patterns in `run_system.py` and `launch_unified_gui.py`
   - Identify active vs unused files in the codebase
   - Check import chains and dependencies
   - Understand the current limitation or missing component

### **🎯 State-of-the-Art Implementation Standards**

4. **🚀 Use Best Practices and Modern Approaches**:
   - **AI/ML**: Implement latest techniques (attention mechanisms, ensemble methods, adaptive learning)
   - **Architecture**: Follow modern software patterns (Factory, Strategy, Observer, Dependency Injection)
   - **Performance**: Use state-of-the-art optimization (GPU acceleration, memory mapping, async processing)
   - **Error Handling**: Implement resilient patterns (Circuit Breaker, Retry with backoff, Graceful degradation)
   - **Code Quality**: Apply clean code principles (SOLID, DRY, KISS) with comprehensive type hints

5. **🔬 Research-Driven Decisions**:
   - **Benchmark against industry standards** (compare with SOTA recognition systems)
   - **Use proven algorithms** (FAISS for vectors, SQLite for persistence, hybrid approaches)
   - **Apply empirically validated techniques** (confidence scoring, ensemble weighting, geometric verification)
   - **Implement adaptive systems** (platform-specific optimization, incremental learning)

### **🏆 State-of-the-Art Techniques to Implement**

**Recognition & AI:**
- **Multi-modal Fusion**: CLIP + DINOv2 + geometric verification (current: ✅ CLIP+DINOv2, 🔄 geometric)
- **Ensemble Methods**: Confidence-based weighting, bootstrap aggregation, stacking
- **Attention Mechanisms**: Self-attention for feature refinement, cross-attention for matching
- **Adaptive Thresholds**: Dynamic confidence thresholds based on dataset characteristics
- **Incremental Learning**: Online learning with catastrophic forgetting prevention
- **Meta-Learning**: Few-shot learning for new categories with minimal examples

**System Architecture:**
- **Microservices Pattern**: Loosely coupled components with well-defined APIs
- **Event-Driven Architecture**: Asynchronous processing with message queues
- **Circuit Breaker Pattern**: Fault tolerance for external dependencies (GPU, file systems)
- **CQRS Pattern**: Separate read/write models for optimal performance
- **Repository Pattern**: Abstracted data access with multiple backends
- **Factory Pattern**: Dynamic component creation based on platform capabilities

**Performance Optimization:**
- **Memory Mapping**: Large file access without loading into memory
- **Async/Await**: Non-blocking I/O operations for GUI responsiveness
- **Connection Pooling**: Efficient database connection management
- **Lazy Loading**: Load data only when needed to reduce memory footprint
- **Batch Processing**: Group operations for better throughput
- **Caching Strategies**: Multi-level caching (L1: memory, L2: disk, L3: distributed)

**Data Engineering:**
- **Apache Arrow**: Columnar data format for fast analytics
- **Parquet Files**: Efficient storage for large feature datasets
- **Vector Quantization**: Compressed vector storage for large-scale systems
- **Bloom Filters**: Probabilistic data structures for fast existence checks
- **LSM Trees**: Log-structured storage for write-heavy workloads

6. **📈 Performance-First Mindset**:
   - **Cross-platform optimization** with automatic hardware detection
   - **Memory-efficient algorithms** with intelligent caching strategies  
   - **Scalable architecture** that handles growth in data and users
   - **Sub-100ms response times** through optimized data structures and algorithms

### **🛠️ Development Workflow**

7. **Always prioritize cross-platform compatibility**
8. **Use the existing platform detection system** - don't recreate it
9. **Follow the established configuration patterns**
10. **Test changes across platform types**
11. **Maintain the performance optimization philosophy**
12. **Keep the todo list updated** with TodoWrite tool
13. **Reference this file** for project context and standards
14. **ADD MANDATORY INLINE DOCUMENTATION** following the standards above
15. **Explain every optimization decision** with comprehensive reasoning

### **🔧 File Management Rules**

16. **NO DUPLICATE FILES**: Always check for existing implementations before creating new files
17. **EXTEND, DON'T RECREATE**: Modify existing files rather than creating similar ones
18. **USE BEST AVAILABLE**: Always use the most complete implementation (`EnhancedUnifiedStoreWithRecognition` over basic versions)
19. **FOLLOW NAMING CONVENTIONS**: Match existing file naming patterns in the project
20. **PRESERVE ARCHITECTURE**: Maintain the established modular structure

### **🔄 Example Workflow: Before Starting Any Task**

```bash
# Step 1: Search for existing files
Glob: pattern="**/*recognition*.py"
Grep: pattern="class.*Recognition|def.*recognize"

# Step 2: Read existing implementations  
Read: file_path="path/to/existing/file.py"

# Step 3: Understand current architecture
Read: file_path="new_system/README.md" (project structure)
Read: file_path="new_system/documents/file_mapping.md" (system reference)

# Step 4: Identify the best implementation to use/extend
# Current best: EnhancedUnifiedStoreWithRecognition
# Current GUI issue: Uses incomplete EnhancedUnifiedStore

# Step 5: Implement using state-of-the-art techniques
# Apply: Modern patterns, performance optimization, error handling
# Include: Comprehensive documentation and reasoning
```

### **⚡ Quick Reference: Current System Architecture (2025)**

**🎯 EXECUTION CHAIN:**
1. **Entry**: `new_system/run_system.py` (primary) or `launch_unified_gui.py`
2. **GUI**: `unified_storage/gui/unified_gui.py` (modern interface)
3. **Storage**: `enhanced_unified_store.py` (current) → should use `enhanced_unified_store_with_recognition.py`
4. **Indexing**: `preprocessing/hybrid_db_indexer.py` (SQLite + FAISS, 43x performance)
5. **Features**: `cross_platform_extractor.py` (CLIP + DINOv2, 1536D)
6. **Platform**: `platform_detector.py` + `config_manager.py` (auto-optimization)

**🔧 CRITICAL FILES (ACTIVE):**
- **Best Implementation**: `EnhancedUnifiedStoreWithRecognition` (most complete, but unused by GUI)
- **Current GUI Issue**: Uses incomplete `EnhancedUnifiedStore` instead of full recognition system
- **Recognition Method**: Should call `recognize_item()` not `search_similar()`
- **Database**: `new_system/data/recognition.db` (unified SQLite with BLOBs)
- **Documentation**: `new_system/documents/` (12 comprehensive guides)

**🚨 PRIORITY FIXES:**
1. Update GUI to use `create_enhanced_unified_store_with_recognition()`
2. Implement missing `lightweight_refiner.pth` model
3. Add geometric verification for ambiguous cases

---

*This instruction set ensures consistent development practices and automatic adherence to project standards. Update this file when architectural decisions change.*