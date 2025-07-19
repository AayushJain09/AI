# AI Recognition System - Complete Command Reference

This document provides a comprehensive reference for all commands used in the AI Recognition System pipeline.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Feature Extraction](#feature-extraction)
- [Training](#training)
- [Indexing](#indexing)
- [Evaluation & Testing](#evaluation--testing)
- [Recognition/Inference](#recognitioninference)
- [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"
```

### GPU/MPS Check
```bash
# Check available acceleration
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

---

## Data Preparation

### Basic Data Augmentation
```bash
# Quick augmentation with default settings
python3 src/data_preparation/prepare.py --input data/raw --output data/augmented

# Custom augmentation with specific parameters
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --augmentations 50 \
    --image-size 768 768 \
    --quality 90
```

### Preset-Based Augmentation
```bash
# Fast training (25 augmentations)
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --preset minimal

# Recommended balance (50 augmentations)
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --preset balanced

# Maximum generalization (100 augmentations)
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --preset aggressive

# Production optimized (75 augmentations)
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --preset production
```

### Advanced Augmentation Configuration
```bash
# Fine-tuned augmentation for optimal generalization
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --augmentations 75 \
    --intensity 0.7 \
    --diversity 0.9 \
    --geometric-weight 0.35 \
    --lighting-weight 0.30 \
    --backgrounds 30 \
    --workers 8
```

---

## Feature Extraction

### Optimized Feature Extraction (1536D)
```bash
# Extract features with optimized CLIP + DINOv2 architecture
python3 src/feature_extraction/feature_extractor.py \
    --input data/augmented \
    --output data/features_1536.h5 \
    --clip-model ViT-L/14 \
    --batch-size 32

# Alternative with different batch size for memory constraints
python3 src/feature_extraction/feature_extractor.py \
    --input data/augmented \
    --output data/features_1536.h5 \
    --clip-model ViT-L/14 \
    --batch-size 16
```

### Legacy Feature Extraction (1280D)
```bash
# Extract features with older architecture (for comparison)
python3 src/feature_extraction/feature_extractor.py \
    --input data/augmented \
    --output data/features_legacy.h5 \
    --clip-model ViT-B/32 \
    --batch-size 32
```

### Feature Inspection
```bash
# Check feature dimensions and content
python3 -c "
import h5py
with h5py.File('data/features_1536.h5', 'r') as f:
    print(f'Groups: {len(list(f.keys()))}')
    g = f[list(f.keys())[0]]
    print(f'CLIP dims: {g[\"clip\"].shape}')
    print(f'DINOv2 dims: {g[\"dinov2\"].shape}')
    print(f'Total dims: {g[\"clip\"].shape[0] + g[\"dinov2\"].shape[0]}')
"
```

---

## Training

### Basic Training
```bash
# Train with default configuration
python3 src/training/modletraining.py --features data/features_1536.h5

# Train with custom parameters
python3 src/training/modletraining.py \
    --features data/features_1536.h5 \
    --epochs 40 \
    --batch-size 24 \
    --lr 3e-4
```

### Advanced Training Configuration
```bash
# State-of-the-art training with optimized parameters
python3 src/training/modletraining.py \
    --features data/features_1536.h5 \
    --epochs 40 \
    --batch-size 24 \
    --lr 3e-4 \
    --num-classes 100 \
    --checkpoint-dir checkpoints \
    --use-wandb
```

### Training with Different Feature Files
```bash
# Train with legacy features (comparison)
python3 src/training/modletraining.py \
    --features data/features_legacy.h5 \
    --epochs 25 \
    --batch-size 16 \
    --lr 1e-4

# Train with optimized features
python3 src/training/modletraining.py \
    --features data/features_optimized.h5 \
    --epochs 40 \
    --batch-size 24 \
    --lr 3e-4
```

---

## Indexing

### Optimized FAISS Indexing
```bash
# Create optimized index with automatic type selection
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models

# Create index with GPU acceleration (if available)
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models \
    --gpu

# Create index with specific precision mode
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models \
    --precision accurate \
    --gpu
```

### Index Type Selection
```bash
# Force specific index types
# Exact search (best accuracy, slower for large datasets)
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models \
    --index-type flat

# IVF index (good balance)
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models \
    --index-type ivf

# IVF+PQ index (compressed, large datasets)
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models \
    --index-type ivf_pq

# HNSW index (fastest search)
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models \
    --index-type hnsw
```

### Index Configuration from Config File
```bash
# Use configuration from config.yaml
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models \
    --config config.yaml
```

---

## Evaluation & Testing

### Pipeline Integration Test
```bash
# Test complete pipeline
python3 test_complete_pipeline.py
```

### Indexing System Test
```bash
# Test FAISS indexing and search
python3 test_indexing.py
```

### Model Evaluation
```bash
# Evaluate trained model
python3 src/evaluation/test.py \
    --model checkpoints/best_model.pth \
    --features data/features_1536.h5

# Detailed evaluation with metrics
python3 src/evaluation/test_suite.py \
    --model checkpoints/best_model.pth \
    --index data/models/faiss_index.bin \
    --metadata data/models/index_metadata.pkl
```

### Performance Benchmarking
```bash
# Benchmark search performance
python3 -c "
from src.indexing.faiss_indexer import AdvancedFAISSIndexer, IndexConfig
import numpy as np
import time

config = IndexConfig()
indexer = AdvancedFAISSIndexer(config)
indexer.load_index('data/models/faiss_index.bin', 'data/models/index_metadata.pkl')

# Benchmark
n_queries = 100
start_time = time.time()
for i in range(n_queries):
    query = np.random.randn(1536).astype('float32')
    results = indexer.search(query, k=10)
end_time = time.time()

avg_time = (end_time - start_time) * 1000 / n_queries
print(f'Average search time: {avg_time:.2f}ms')
print(f'Throughput: {1000/avg_time:.0f} queries/second')
"
```

---

## Recognition/Inference

### Single Image Recognition
```bash
# Recognize single image
python3 src/inference/recognize.py \
    --image path/to/test/image.jpg \
    --model checkpoints/best_model.pth \
    --index data/models/faiss_index.bin \
    --metadata data/models/index_metadata.pkl

# With custom confidence threshold
python3 src/inference/recognize.py \
    --image path/to/test/image.jpg \
    --model checkpoints/best_model.pth \
    --index data/models/faiss_index.bin \
    --metadata data/models/index_metadata.pkl \
    --threshold 0.85
```

### Batch Recognition
```bash
# Recognize multiple images
python3 src/inference/recognize.py \
    --input-dir path/to/test/images/ \
    --output results.json \
    --model checkpoints/best_model.pth \
    --index data/models/faiss_index.bin \
    --metadata data/models/index_metadata.pkl \
    --batch-size 16
```

---

## Complete Pipeline Commands

### End-to-End Pipeline
```bash
# 1. Data preparation
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --preset balanced

# 2. Feature extraction
python3 src/feature_extraction/feature_extractor.py \
    --input data/augmented \
    --output data/features_1536.h5 \
    --clip-model ViT-L/14

# 3. Training
python3 src/training/modletraining.py \
    --features data/features_1536.h5 \
    --epochs 40 \
    --batch-size 24 \
    --lr 3e-4

# 4. Indexing
python3 src/indexing/faiss_indexer.py \
    --features data/features_1536.h5 \
    --output data/models \
    --precision accurate

# 5. Test pipeline
python3 test_complete_pipeline.py
```

### Quick Setup for New Dataset
```bash
# Minimal setup for testing
python3 src/data_preparation/prepare.py --input data/raw --output data/augmented --preset minimal
python3 src/feature_extraction/feature_extractor.py --input data/augmented --output data/features.h5
python3 src/training/modletraining.py --features data/features.h5 --epochs 10
python3 src/indexing/faiss_indexer.py --features data/features.h5 --output data/models
```

---

## File Management Commands

### Check System Status
```bash
# Check if all required files exist
ls -la data/features_1536.h5
ls -la checkpoints/best_model.pth
ls -la data/models/faiss_index.bin
ls -la data/models/index_metadata.pkl
```

### Clean Up Commands
```bash
# Remove old features
rm -f data/features.h5 data/features_optimized.h5

# Clean checkpoints (keep only best)
find checkpoints/ -name "*.pth" ! -name "best_model.pth" -delete

# Clean augmented data
rm -rf data/augmented/*

# Full cleanup (be careful!)
rm -rf data/augmented/ checkpoints/ data/models/
```

### Backup Commands
```bash
# Backup trained models
cp -r checkpoints/ backup/checkpoints_$(date +%Y%m%d)/

# Backup index
cp -r data/models/ backup/models_$(date +%Y%m%d)/

# Backup features
cp data/features_1536.h5 backup/features_1536_$(date +%Y%m%d).h5
```

---

## Troubleshooting Commands

### Memory Issues
```bash
# Check memory usage
python3 -c "
import psutil
import os
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'Current memory usage: {memory_mb:.1f} MB')
"

# Reduce batch size for memory constraints
python3 src/feature_extraction/feature_extractor.py \
    --input data/augmented \
    --output data/features_1536.h5 \
    --batch-size 8  # Reduced from default 32

python3 src/training/modletraining.py \
    --features data/features_1536.h5 \
    --batch-size 8  # Reduced from default 24
```

### GPU/MPS Issues
```bash
# Force CPU-only mode
CUDA_VISIBLE_DEVICES="" python3 src/training/modletraining.py --features data/features_1536.h5

# Check GPU memory
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon) is available')
else:
    print('No GPU acceleration available')
"
```

### Debug Mode
```bash
# Run with debug logging
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
" && python3 src/training/modletraining.py --features data/features_1536.h5

# Check feature extraction with single image
python3 -c "
from src.feature_extraction.feature_extractor import MultiModalFeatureExtractor
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

extractor = MultiModalFeatureExtractor(config['features'])
features = extractor.extract_all_features('data/raw/item_001/Copy of IMG_8388.JPG')
print(f'CLIP: {len(features[\"clip\"])}, DINOv2: {len(features[\"dinov2\"])}')
"
```

### Validation Commands
```bash
# Validate feature dimensions
python3 -c "
import h5py
with h5py.File('data/features_1536.h5', 'r') as f:
    sample = f[list(f.keys())[0]]
    clip_dim = sample['clip'].shape[0]
    dino_dim = sample['dinov2'].shape[0]
    total = clip_dim + dino_dim
    print(f'Dimensions: CLIP={clip_dim}, DINOv2={dino_dim}, Total={total}')
    assert total == 1536, f'Expected 1536, got {total}'
    print('✅ Feature dimensions are correct')
"

# Validate model checkpoint
python3 -c "
import torch
try:
    checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
    print(f'✅ Model checkpoint is valid')
    print(f'Keys: {list(checkpoint.keys())}')
except Exception as e:
    print(f'❌ Model checkpoint error: {e}')
"

# Validate index
python3 -c "
import faiss
try:
    index = faiss.read_index('data/models/faiss_index.bin')
    print(f'✅ Index is valid: {index.ntotal} vectors, {index.d} dimensions')
except Exception as e:
    print(f'❌ Index error: {e}')
"
```

---

## Environment Variables

### Performance Tuning
```bash
# Set number of threads for OpenMP
export OMP_NUM_THREADS=4

# Set MKL threads (Intel CPUs)
export MKL_NUM_THREADS=4

# Disable CUDA if needed
export CUDA_VISIBLE_DEVICES=""

# Set cache directories
export TORCH_HOME=/path/to/torch/cache
export HF_HOME=/path/to/huggingface/cache
```

### Memory Management
```bash
# Limit PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set Python memory debugging
export PYTHONMALLOC=debug
```

---

## Configuration Templates

### Quick Config for Small Dataset
```yaml
# config_small.yaml
training:
  epochs: 20
  batch_size: 8
  learning_rate: 1e-4

augmentation:
  augmentations_per_image: 25

indexing:
  index_type: "flat"
  precision_mode: "fast"
```

### Production Config
```yaml
# config_production.yaml
training:
  epochs: 40
  batch_size: 24
  learning_rate: 3e-4

augmentation:
  augmentations_per_image: 50

indexing:
  index_type: "auto"
  precision_mode: "accurate"
  use_gpu: true
```

---

## Common Command Combinations

### Development Workflow
```bash
# Quick iteration cycle
python3 src/data_preparation/prepare.py --input data/raw --output data/augmented --preset minimal
python3 src/feature_extraction/feature_extractor.py --input data/augmented --output data/features.h5 --batch-size 16
python3 src/training/modletraining.py --features data/features.h5 --epochs 10 --batch-size 8
python3 test_complete_pipeline.py
```

### Production Deployment
```bash
# Full production pipeline
python3 src/data_preparation/prepare.py --input data/raw --output data/augmented --preset production
python3 src/feature_extraction/feature_extractor.py --input data/augmented --output data/features_1536.h5
python3 src/training/modletraining.py --features data/features_1536.h5 --epochs 40 --batch-size 24 --lr 3e-4
python3 src/indexing/faiss_indexer.py --features data/features_1536.h5 --output data/models --precision accurate --gpu
python3 test_complete_pipeline.py
```

### Performance Optimization
```bash
# GPU-optimized pipeline
CUDA_VISIBLE_DEVICES=0 python3 src/feature_extraction/feature_extractor.py --input data/augmented --output data/features_1536.h5 --batch-size 64
CUDA_VISIBLE_DEVICES=0 python3 src/training/modletraining.py --features data/features_1536.h5 --epochs 40 --batch-size 32
python3 src/indexing/faiss_indexer.py --features data/features_1536.h5 --output data/models --gpu --index-type hnsw
```

---

This command reference covers all major operations in the AI Recognition System. For additional help with any specific command, use the `--help` flag:

```bash
python3 src/feature_extraction/feature_extractor.py --help
python3 src/training/modletraining.py --help
python3 src/indexing/faiss_indexer.py --help
```