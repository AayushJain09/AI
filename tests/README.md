# AI Recognition System - Test Suite

## Overview

This directory contains the complete test suite for the AI Recognition System, organized by test type and scope.

## Test Structure

```
tests/
├── unit/                    # Unit tests - isolated component testing
│   └── test_feature_extraction.py
├── integration/             # Integration tests - component interaction
│   └── test_pipeline_integration.py
├── system/                  # System tests - end-to-end testing
│   └── test_recognition_system.py
└── README.md               # This file
```

## Test Types

### Unit Tests (`tests/unit/`)
Test individual components in isolation with mocked dependencies.

- **test_feature_extraction.py**: Tests CLIP and DINOv2 feature extraction components
- Focus: Component behavior, error handling, configuration validation
- Runtime: Fast (~30 seconds)
- Dependencies: Minimal (uses mocks)

### Integration Tests (`tests/integration/`)
Test interaction between multiple components.

- **test_pipeline_integration.py**: Tests feature extraction → indexing → search pipeline
- Focus: Component compatibility, data flow, file formats
- Runtime: Medium (~2 minutes)
- Dependencies: Requires model files and test data

### System Tests (`tests/system/`)
Test the complete system end-to-end as a user would experience it.

- **test_recognition_system.py**: Tests complete recognition pipeline with real data
- Focus: User workflows, performance, accuracy
- Runtime: Slow (~5 minutes)
- Dependencies: Complete system setup required

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-cov

# Ensure system is set up
python3 test_recognition_final.py  # Should pass
```

### Run All Tests
```bash
# From project root
python -m pytest tests/ -v
```

### Run Specific Test Types
```bash
# Unit tests only (fast)
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# System tests only (requires full setup)
python -m pytest tests/system/ -v
```

### Run Individual Test Files
```bash
# Run specific test file
python -m pytest tests/system/test_recognition_system.py -v

# Run specific test method
python -m pytest tests/system/test_recognition_system.py::TestRecognitionSystem::test_known_item_recognition -v
```

### Run Tests with Coverage
```bash
# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

## Test Requirements

### For Unit Tests
- Python environment with dependencies installed
- No model files or training data required
- Uses mocks for external dependencies

### For Integration Tests
- Model files in `data/models/`:
  - `faiss_index.bin`
  - `index_metadata.pkl`
- Feature files (optional):
  - `data/features.h5` or `data/features_1536.h5`
- Configuration file: `config.yaml`

### For System Tests
- Complete system setup (all of the above plus):
- Training data in `data/raw/`:
  - `item_001/Copy of IMG_8388.JPG`
  - `item_002/Copy of IMG_8403.JPG`
  - `item_003/Copy of IMG_8428.JPG`
  - `item_004/1752950980813_IMG_8416.JPG`
- Unknown test image (for rejection testing)

## Test Configuration

### Environment Variables
```bash
# Optional: Set custom test configuration
export TEST_CONFIG_PATH="/path/to/test_config.yaml"
export TEST_DATA_PATH="/path/to/test/data"
```

### Test Configuration File
Create `test_config.yaml` for test-specific settings:
```yaml
# Test-specific configuration
confidence_threshold: 0.8  # More lenient for testing
logging:
  level: "WARNING"  # Reduce log noise during tests
features:
  batch_size: 8  # Smaller batches for test speed
```

## Expected Test Results

### Current System Performance
Based on the production system:

**System Tests:**
- ✅ All known items: 100% accuracy (confidence ~1.414)
- ✅ Unknown item rejection: 100% accuracy (confidence <0.7)
- ✅ Processing time: <500ms per image
- ✅ Index: 11 vectors, 1536 dimensions

**Integration Tests:**
- ✅ Feature extraction: 1536D (768 CLIP + 768 DINOv2)
- ✅ Index loading: Successful with correct metadata
- ✅ Search functionality: Returns valid results
- ✅ End-to-end pipeline: Complete flow works

**Unit Tests:**
- ✅ Configuration validation: Proper error handling
- ✅ Model loading: Correct model initialization
- ✅ Feature dimensions: Correct output sizes
- ✅ Error handling: Graceful failure modes

## Troubleshooting Tests

### Common Issues

#### Test Failures Due to Missing Files
```bash
# Check if required files exist
ls -la data/models/faiss_index.bin
ls -la data/models/index_metadata.pkl
ls -la data/raw/item_*/

# If missing, regenerate:
python3 src/indexing/faiss_indexer.py --features data/features.h5 --output data/models
```

#### Import Errors
```bash
# Check Python path
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Or run from project root
cd /path/to/ai-recognition-system
python -m pytest tests/
```

#### GPU/Device Issues
```bash
# Force CPU-only testing
export CUDA_VISIBLE_DEVICES=""
python -m pytest tests/
```

#### Memory Issues
```bash
# Reduce batch size in tests
# Edit test files to use smaller batch_size values
# Or run tests individually
```

### Test Debugging
```bash
# Run with debug output
python -m pytest tests/ -v -s --tb=long

# Run single test with debugging
python -m pytest tests/system/test_recognition_system.py::TestRecognitionSystem::test_known_item_recognition -v -s

# Check test logs
# Tests will output detailed information about failures
```

## Performance Benchmarks

### Expected Runtimes
- **Unit Tests**: ~30 seconds
- **Integration Tests**: ~2 minutes  
- **System Tests**: ~5 minutes
- **All Tests**: ~7 minutes

### Performance Assertions
Tests include performance assertions:
- Recognition time: <1 second per image
- Index loading: <30 seconds
- Feature extraction: <10 seconds per image
- Memory usage: <4GB

## Contributing

### Adding New Tests
1. **Unit Tests**: Add to `tests/unit/` for new components
2. **Integration Tests**: Add to `tests/integration/` for component interactions
3. **System Tests**: Add to `tests/system/` for new workflows

### Test Naming Convention
- Test files: `test_<component>.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<specific_behavior>`

### Test Documentation
Each test should include:
- Clear docstring explaining what is being tested
- Assertions with descriptive messages
- Setup/teardown as needed
- Performance expectations where relevant

## Continuous Integration

These tests are designed to run in CI/CD environments:
- No GUI dependencies
- Configurable timeouts
- Clear pass/fail criteria
- Detailed error reporting

### CI Configuration Example
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt pytest
    - name: Run tests
      run: python -m pytest tests/ -v
```