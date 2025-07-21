# Codebase Organization - AI Recognition System

## Overview

The AI Recognition System codebase has been completely reorganized for clarity, maintainability, and proper separation of concerns.

## Directory Structure

```
ai-recognition-system/
├── docs/                        # 📚 Complete documentation suite
│   ├── README.md               # Documentation overview
│   ├── system-overview.md      # Current system architecture  
│   ├── setup-guide.md          # Installation and setup
│   ├── user-guide.md           # GUI usage guide
│   ├── technical-architecture.md # Deep technical details
│   ├── troubleshooting.md      # Problem solving
│   ├── configuration.md        # Config reference
│   ├── api-reference.md        # REST API docs
│   ├── testing.md              # Testing guide
│   └── codebase-organization.md # This file
│
├── src/                         # 🔧 Core system components
│   ├── data_preparation/       # Image preprocessing & augmentation
│   │   └── prepare.py
│   ├── feature_extraction/     # CLIP + DINOv2 extractors
│   │   └── feature_extractor.py
│   ├── training/               # Model training (disabled)
│   │   └── modletraining.py
│   ├── indexing/               # FAISS index management
│   │   └── faiss_indexer.py
│   ├── inference/              # Recognition pipeline
│   │   └── recognize.py
│   └── evaluation/             # Performance testing
│       ├── test.py
│       └── metrics.py
│
├── tests/                       # 🧪 Organized test suite
│   ├── README.md               # Test documentation
│   ├── run_tests.py            # Test runner
│   ├── unit/                   # Component isolation tests
│   │   └── test_feature_extraction.py
│   ├── integration/            # Component interaction tests
│   │   └── test_pipeline_integration.py
│   └── system/                 # End-to-end tests
│       ├── test_recognition_system.py
│       └── test_recognition_final.py
│
├── frontend/                    # 🖥️ PyQt6 GUI application
│   ├── main.py                 # Main application
│   └── widgets/                # GUI components
│       ├── items.py            # Item management
│       ├── recognition.py      # Recognition interface
│       ├── training.py         # Training monitor
│       ├── evaluation.py       # Performance dashboard
│       ├── settings.py         # Configuration
│       └── logs.py             # Log viewer
│
├── backend/                     # 🚀 FastAPI server
│   └── main.py                 # REST API endpoints
│
├── data/                        # 📊 Data storage
│   ├── raw/                    # Original training images
│   ├── augmented/              # Generated training data
│   ├── models/                 # FAISS index and metadata
│   ├── features.h5             # Extracted features
│   └── features_1536.h5        # Optimized features
│
├── config.yaml                 # ⚙️ System configuration
├── requirements.txt            # 📦 Dependencies
├── start_system.py             # 🚀 System launcher
├── run_tests.py                # 🧪 Test runner
└── CLAUDE.md                   # 🤖 AI development guide
```

## Key Organizational Principles

### 1. Separation of Concerns
- **src/**: Core algorithmic components
- **frontend/**: User interface components  
- **backend/**: API server components
- **tests/**: Testing components
- **docs/**: Documentation components

### 2. Clear Hierarchies
- **tests/unit/**: Fast, isolated component tests
- **tests/integration/**: Component interaction tests
- **tests/system/**: End-to-end functionality tests

### 3. Logical Grouping
- **Data flow**: `data_preparation/ → feature_extraction/ → indexing/ → inference/`
- **User interaction**: `frontend/widgets/` organized by functionality
- **Documentation**: Topic-based organization in `docs/`

## Component Responsibilities

### Core System (src/)

#### data_preparation/
- **Purpose**: Image preprocessing and augmentation
- **Key File**: `prepare.py` - Generates 50x augmented training data
- **Input**: Raw images in `data/raw/`
- **Output**: Augmented images in `data/augmented/`

#### feature_extraction/
- **Purpose**: Convert images to feature vectors
- **Key File**: `feature_extractor.py` - CLIP+DINOv2 pipeline
- **Architecture**: 1536D features (768 CLIP + 768 DINOv2)
- **Output**: Features stored in HDF5 format

#### indexing/
- **Purpose**: Create and manage FAISS search index
- **Key File**: `faiss_indexer.py` - Optimized index creation
- **Index Type**: Flat (exact search) for maximum accuracy
- **Storage**: `data/models/faiss_index.bin` + metadata

#### inference/
- **Purpose**: Real-time item recognition
- **Key File**: `recognize.py` - Production recognition pipeline
- **Architecture**: Raw features → FAISS search → Confidence filtering
- **Performance**: ~300ms per recognition

### User Interface (frontend/)

#### main.py
- **Purpose**: Main application entry point
- **Framework**: PyQt6 desktop application
- **Features**: Tab-based interface, real-time updates

#### widgets/
- **items.py**: Item management, image upload, CRUD operations
- **recognition.py**: Real-time recognition, camera integration
- **training.py**: Training progress monitoring (legacy)
- **evaluation.py**: Performance metrics and testing
- **settings.py**: System configuration interface
- **logs.py**: Real-time log viewing and filtering

### API Server (backend/)

#### main.py
- **Purpose**: REST API for system integration
- **Framework**: FastAPI with async support
- **Endpoints**: Recognition, item management, system status
- **Port**: 8000 (configurable)

### Testing (tests/)

#### Unit Tests
- **Purpose**: Test individual components in isolation
- **Coverage**: Feature extraction, configuration, error handling
- **Dependencies**: Minimal (uses mocks)
- **Runtime**: ~30 seconds

#### Integration Tests  
- **Purpose**: Test component interactions
- **Coverage**: Pipeline flow, file formats, API integration
- **Dependencies**: Model files and configuration
- **Runtime**: ~2 minutes

#### System Tests
- **Purpose**: End-to-end functionality validation
- **Coverage**: Complete user workflows, performance
- **Dependencies**: Full system setup
- **Runtime**: ~5 minutes

## File Naming Conventions

### Python Files
- **Components**: `snake_case.py` (e.g., `feature_extractor.py`)
- **Tests**: `test_<component>.py` (e.g., `test_feature_extraction.py`)
- **Main files**: `main.py` for entry points

### Configuration Files
- **Main config**: `config.yaml`
- **Test config**: `test_config.yaml`
- **Requirements**: `requirements.txt`

### Data Files
- **Images**: Preserve original names
- **Features**: `features_<dimensions>.h5`
- **Models**: Descriptive names (e.g., `faiss_index.bin`)

## Import Path Structure

### Relative Imports (within src/)
```python
# From within src/
from feature_extraction.feature_extractor import MultiModalFeatureExtractor
from indexing.faiss_indexer import AdvancedFAISSIndexer
```

### Absolute Imports (from project root)
```python
# From tests/ or external files
import sys
sys.path.append('src')
from inference.recognize import create_pipeline
```

## Configuration Management

### Centralized Configuration
- **Main**: `config.yaml` - Production settings
- **Testing**: `test_config.yaml` - Test-specific overrides
- **Development**: Environment variables for overrides

### Configuration Sections
- **features**: Model selection and parameters
- **training**: Training hyperparameters (legacy)
- **recognition**: Confidence thresholds and pipeline settings
- **indexing**: FAISS configuration
- **api**: Server settings
- **logging**: Log levels and output

## Data Flow

### Training Pipeline (Setup Phase)
```
data/raw/ → data_preparation/ → data/augmented/ → feature_extraction/ 
→ data/features.h5 → indexing/ → data/models/faiss_index.bin
```

### Recognition Pipeline (Runtime)
```
Input Image → feature_extraction/ → inference/recognize.py → FAISS Search 
→ Confidence Filtering → Recognition Result
```

### User Interface Flow
```
frontend/main.py → widgets/ → backend/main.py → src/inference/ → Result Display
```

## Quality Assurance

### Code Organization
- ✅ Single responsibility per module
- ✅ Clear dependency hierarchy  
- ✅ Consistent naming conventions
- ✅ Proper separation of concerns

### Documentation
- ✅ Complete user guides
- ✅ Technical architecture documentation
- ✅ API reference with examples
- ✅ Troubleshooting guides

### Testing
- ✅ Multi-level test coverage
- ✅ Organized test structure
- ✅ Clear test documentation
- ✅ Production validation scripts

## Migration Notes

### What Was Removed
- Multiple scattered test files in project root
- Redundant documentation files
- Outdated implementation files
- Duplicate configuration files

### What Was Improved
- Centralized documentation in `docs/`
- Organized test suite in `tests/`
- Clear component separation
- Proper import path handling

## Development Workflow

### Adding New Features
1. **Plan**: Update documentation first
2. **Implement**: Add to appropriate `src/` subdirectory
3. **Test**: Add tests in appropriate `tests/` subdirectory
4. **Document**: Update relevant documentation
5. **Integrate**: Update main launchers if needed

### Making Changes
1. **Read**: Check existing documentation
2. **Test**: Run existing tests before changes
3. **Modify**: Make targeted changes
4. **Validate**: Run tests after changes
5. **Document**: Update documentation if needed

## Maintenance

### Regular Tasks
- **Documentation**: Keep docs/ in sync with code changes
- **Dependencies**: Update requirements.txt as needed
- **Testing**: Add tests for new functionality
- **Configuration**: Validate config.yaml settings

### Code Quality
- **Imports**: Keep import paths consistent
- **Structure**: Maintain separation of concerns
- **Documentation**: Document all public interfaces
- **Testing**: Maintain test coverage

This organized structure provides a solid foundation for the AI Recognition System, making it easy to understand, maintain, and extend.