# Setup Guide - AI Recognition System

## Prerequisites

### System Requirements
- **Python**: 3.8+ (3.9-3.11 recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for models and data
- **GPU**: Optional but recommended (Apple MPS or NVIDIA CUDA)

### Supported Platforms
- **macOS**: Apple Silicon (M1/M2) with MPS acceleration
- **Linux**: Ubuntu 20.04+ with CUDA support
- **Windows**: Windows 10/11 with WSL2 recommended

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-recognition-system
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import faiss; print(f'FAISS: {faiss.__version__}')"
```

### 4. Verify GPU Support (Optional)
```bash
# Check available acceleration
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')
"
```

## Initial Setup

### 1. Create Required Directories
```bash
mkdir -p data/raw data/augmented data/models logs checkpoints
```

### 2. Verify System
```bash
# Quick system check
python3 -c "
import sys
sys.path.append('src')
from feature_extraction.feature_extractor import MultiModalFeatureExtractor
print('✅ System setup complete')
"
```

### 3. Test Recognition System
```bash
# Run production test (should work with existing data)
python3 test_recognition_final.py
```

Expected output:
```
🚀 AI Recognition System - Production Test
✅ Pipeline loaded in X.XXs
📈 Index: 11 vectors, 1536 dimensions
✅ All tests PASS
```

## Configuration

### Default Configuration (config.yaml)
The system comes with an optimized configuration:

```yaml
# Current working configuration
confidence_threshold: 0.90
min_stage1_confidence: 0.7
model_path: "checkpoints/best_model_DISABLED.pth"

features:
  clip_model: "ViT-L/14"
  clip_dimensions: 768
  dinov2_dimensions: 768
  total_dimensions: 1536
```

### Memory Optimization
For systems with limited memory:

```yaml
# config_low_memory.yaml
features:
  batch_size: 8  # Reduce from default 32
  
training:
  batch_size: 8  # Reduce from default 24
```

## Starting the System

### Option 1: Complete System (Recommended)
```bash
# Start both backend and frontend
python3 start_system.py
```

This will:
- ✅ Activate virtual environment
- ✅ Check system status
- ✅ Test recognition system
- ✅ Start FastAPI backend (port 8000)
- ✅ Launch PyQt6 frontend GUI

### Option 2: Manual Startup
```bash
# Terminal 1: Start backend
python3 backend/main.py

# Terminal 2: Start frontend
python3 frontend/main.py
```

### Option 3: Test Only
```bash
# Test recognition without GUI
python3 test_recognition_final.py
```

## Adding Your First Items

### 1. Prepare Images
- **Format**: JPG, PNG supported
- **Quality**: High resolution (1080p+ recommended)
- **Quantity**: 8+ images per item minimum
- **Variety**: Different angles, lighting conditions

### 2. Directory Structure
```
data/raw/
├── item_001/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (8+ images)
├── item_002/
│   └── ... (8+ images)
```

### 3. Data Preparation
```bash
# Generate augmented training data
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --augmentations 50
```

### 4. Feature Extraction
```bash
# Extract CLIP+DINOv2 features
python3 src/feature_extraction/feature_extractor.py \
    --input data/augmented \
    --output data/features.h5
```

### 5. Create Recognition Index
```bash
# Build FAISS index
python3 src/indexing/faiss_indexer.py \
    --features data/features.h5 \
    --output data/models
```

### 6. Test Recognition
```bash
# Test your new system
python3 test_recognition_final.py
```

## Verification

### System Health Check
```bash
# Check all required files exist
ls -la data/models/faiss_index.bin
ls -la data/models/index_metadata.pkl
ls -la config.yaml
```

### API Health Check
```bash
# Test backend API (while backend is running)
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### Recognition Test
```bash
# Test recognition with sample image
python3 -c "
import sys
sys.path.append('src')
from inference.recognize import create_pipeline

pipeline = create_pipeline('config.yaml')
result = pipeline.recognize('data/raw/item_001/Copy of IMG_8388.JPG')
print(f'Result: {result.item_id} (confidence: {result.confidence:.3f})')
"
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Fix missing dependencies
pip install -r requirements.txt --upgrade
```

#### 2. Memory Issues
```bash
# Reduce batch sizes in config.yaml
features:
  batch_size: 8
training:
  batch_size: 8
```

#### 3. GPU Issues
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python3 your_command.py
```

#### 4. Missing Files
```bash
# Check if index exists
if [ ! -f "data/models/faiss_index.bin" ]; then
    echo "Index missing - run feature extraction and indexing"
fi
```

### Performance Optimization

#### For Apple Silicon (M1/M2)
```bash
# Verify MPS is working
python3 -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
"
```

#### For NVIDIA GPUs
```bash
# Check CUDA
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

## Next Steps

Once setup is complete:

1. **[User Guide](user-guide.md)** - Learn to use the GUI
2. **[System Overview](system-overview.md)** - Understand the architecture
3. **[Troubleshooting](troubleshooting.md)** - Solve common issues

## Support

- Check logs in `logs/` directory for detailed error information
- Use `python3 script.py --help` for command-specific help
- See troubleshooting guide for common solutions