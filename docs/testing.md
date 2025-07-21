# Testing Guide - AI Recognition System

## Overview

The AI Recognition System includes a comprehensive test suite organized by scope and purpose. This guide explains how to run tests and interpret results.

## Quick Start

### Run All Tests
```bash
# Complete test suite with organized output
python run_tests.py

# Alternative: Run with pytest directly
python -m pytest tests/ -v
```

### Run Production Test Only
```bash
# Quick production verification (run from project root)
python3 test_recognition_final.py

# Or use the organized test structure
python3 tests/run_tests.py
```

Expected output:
```
🚀 AI Recognition System - Production Test
✅ item_001: 1.414 (0.399s)
✅ item_002: 1.414 (0.371s) 
✅ item_003: 1.414 (0.363s)
✅ item_004: 1.414 (0.369s)
✅ Unknown item correctly rejected (0.092s)
📊 System accuracy: PASS
```

## Test Organization

### Test Directory Structure
```
tests/
├── unit/                    # Fast, isolated component tests
│   └── test_feature_extraction.py
├── integration/             # Component interaction tests  
│   └── test_pipeline_integration.py
├── system/                  # End-to-end system tests
│   ├── test_recognition_system.py
│   └── test_recognition_final.py
└── README.md               # Detailed test documentation
```

## Test Categories

### 1. Unit Tests (30 seconds)
**Purpose**: Test individual components in isolation
**Location**: `tests/unit/`
**Dependencies**: Minimal (uses mocks)

```bash
# Run unit tests only
python -m pytest tests/unit/ -v
```

**What's Tested**:
- Feature extraction component validation
- Configuration handling
- Error handling and edge cases
- Device selection logic

### 2. Integration Tests (2 minutes)
**Purpose**: Test component interactions
**Location**: `tests/integration/`
**Dependencies**: Model files and configuration

```bash
# Run integration tests only
python -m pytest tests/integration/ -v
```

**What's Tested**:
- Feature extraction → FAISS indexing pipeline
- Index loading and searching
- File format compatibility
- End-to-end data flow

### 3. System Tests (5 minutes)  
**Purpose**: Test complete system functionality
**Location**: `tests/system/`
**Dependencies**: Complete system setup

```bash
# Run system tests only
python -m pytest tests/system/ -v

# Or run production test directly
python tests/system/test_recognition_final.py
```

**What's Tested**:
- Complete recognition pipeline
- Real-world accuracy and performance
- Unknown item rejection
- System-level requirements

## Current Test Results

### Production System Performance
Based on the latest system state:

**Recognition Accuracy**: 100%
- ✅ item_001: 1.414 confidence
- ✅ item_002: 1.414 confidence  
- ✅ item_003: 1.414 confidence
- ✅ item_004: 1.414 confidence

**Unknown Item Rejection**: 100%
- ✅ Unknown items properly rejected (confidence <0.7)
- ✅ No false positives

**Performance Metrics**:
- Average recognition time: ~300ms
- Index: 11 vectors, 1536 dimensions
- Memory usage: <2GB

## Test Requirements

### Minimal Requirements (Unit Tests)
- Python environment with dependencies
- `pytest` installed
- No data files required

### Full Requirements (All Tests)
- Complete system setup
- Model files in `data/models/`:
  - `faiss_index.bin`
  - `index_metadata.pkl`
- Training data in `data/raw/`:
  - `item_001/Copy of IMG_8388.JPG`
  - `item_002/Copy of IMG_8403.JPG`
  - `item_003/Copy of IMG_8428.JPG`
  - `item_004/1752950980813_IMG_8416.JPG`
- Configuration: `config.yaml`

### Installing Test Dependencies
```bash
pip install pytest pytest-cov
```

## Running Specific Tests

### Run Individual Test Files
```bash
# System test with detailed output
python -m pytest tests/system/test_recognition_system.py -v -s

# Integration test only
python -m pytest tests/integration/test_pipeline_integration.py -v

# Unit test with coverage
python -m pytest tests/unit/test_feature_extraction.py --cov=src
```

### Run Specific Test Methods
```bash
# Test only known item recognition
python -m pytest tests/system/test_recognition_system.py::TestRecognitionSystem::test_known_item_recognition -v

# Test only unknown item rejection
python -m pytest tests/system/test_recognition_system.py::TestRecognitionSystem::test_unknown_item_rejection -v
```

### Run with Different Verbosity
```bash
# Minimal output
python -m pytest tests/ -q

# Standard output
python -m pytest tests/ -v

# Maximum detail with stdout
python -m pytest tests/ -v -s --tb=long
```

## Test Configuration

### Environment Variables
```bash
# Run tests with specific configuration
export TEST_CONFIG_PATH="test_config.yaml"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Force CPU-only testing
export CUDA_VISIBLE_DEVICES=""

# Run tests
python -m pytest tests/
```

### Custom Test Configuration
Create `test_config.yaml`:
```yaml
# Test-specific settings
confidence_threshold: 0.8  # More lenient for testing
logging:
  level: "WARNING"  # Reduce log noise
features:
  batch_size: 8  # Faster testing
```

## Performance Testing

### Benchmark Recognition Speed
```bash
# Performance test with timing
python -c "
import time
from src.inference.recognize import create_pipeline

pipeline = create_pipeline('config.yaml')
test_image = 'data/raw/item_001/Copy of IMG_8388.JPG'

# Warmup
pipeline.recognize(test_image)

# Benchmark
times = []
for i in range(10):
    start = time.time()
    result = pipeline.recognize(test_image)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
print(f'Average recognition time: {avg_time:.3f}s')
print(f'Throughput: {1/avg_time:.1f} images/second')
"
```

### Memory Usage Testing
```bash
# Monitor memory during tests
python -c "
import psutil
import subprocess
import time

process = psutil.Process()
print(f'Memory before tests: {process.memory_info().rss / 1024 / 1024:.1f} MB')

# Run tests and monitor
result = subprocess.run(['python', '-m', 'pytest', 'tests/system/', '-v'], 
                       capture_output=True, text=True)

print(f'Memory after tests: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'Test result: {\"PASS\" if result.returncode == 0 else \"FAIL\"}')
"
```

## Test Coverage

### Generate Coverage Report
```bash
# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Coverage Targets
- **Overall**: >80% line coverage
- **Critical components**: >90% coverage
  - Feature extraction
  - Recognition pipeline
  - Index management

## Troubleshooting Tests

### Common Test Failures

#### Import Errors
```bash
# Fix Python path
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Or run from project root
cd /path/to/ai-recognition-system
python -m pytest tests/
```

#### Missing Files
```bash
# Check required files
ls -la data/models/faiss_index.bin
ls -la data/models/index_metadata.pkl

# Regenerate if missing
python src/indexing/faiss_indexer.py --features data/features.h5 --output data/models
```

#### GPU/Memory Issues
```bash
# Force CPU-only testing
export CUDA_VISIBLE_DEVICES=""

# Reduce memory usage
# Edit test files to use smaller batch sizes
```

#### Test Data Issues
```bash
# Verify test images exist
find data/raw -name "*.JPG" | head -5

# Check image accessibility
python -c "
from PIL import Image
img = Image.open('data/raw/item_001/Copy of IMG_8388.JPG')
print(f'Image size: {img.size}')
"
```

### Debug Individual Tests
```bash
# Run single test with maximum detail
python -m pytest tests/system/test_recognition_system.py::TestRecognitionSystem::test_known_item_recognition -v -s --tb=long

# Add print statements for debugging
python -c "
import sys
sys.path.append('src')
from inference.recognize import create_pipeline

print('Loading pipeline...')
pipeline = create_pipeline('config.yaml')
print(f'Pipeline loaded: {pipeline.index.ntotal} vectors')

print('Testing recognition...')
result = pipeline.recognize('data/raw/item_001/Copy of IMG_8388.JPG')
print(f'Result: {result.item_id} (confidence: {result.confidence})')
"
```

## Continuous Integration

### CI/CD Configuration
Tests are designed for automated environments:

```yaml
# .github/workflows/test.yml
name: AI Recognition Tests
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
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: python run_tests.py
```

### Test Automation
```bash
# Pre-commit hook
#!/bin/bash
# .git/hooks/pre-commit
python -m pytest tests/unit/ tests/integration/ -v
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

## Test Development

### Adding New Tests
1. **Determine test type**: Unit, integration, or system
2. **Choose appropriate directory**: `tests/unit/`, `tests/integration/`, or `tests/system/`
3. **Follow naming convention**: `test_<component>.py`
4. **Include proper documentation**: Clear docstrings and assertions

### Test Template
```python
#!/usr/bin/env python3
"""
Test Description: What this test validates
"""

import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from your_component import YourClass

class TestYourComponent:
    """Test class for YourComponent"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.component = YourClass()
    
    def test_basic_functionality(self):
        """Test basic component functionality"""
        result = self.component.do_something()
        assert result is not None, "Component should return a result"
    
    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            self.component.invalid_operation()

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

This testing framework ensures system reliability and provides confidence in production deployment.