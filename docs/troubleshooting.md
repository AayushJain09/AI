# Troubleshooting Guide - AI Recognition System

## Common Issues and Solutions

### 1. Recognition Problems

#### False Positives (Unknown Items Recognized as Known Items)
**Symptoms**: Unknown items being incorrectly matched to existing items

**Root Cause**: Low confidence thresholds or similar embeddings

**Solutions**:
```bash
# 1. Increase confidence thresholds in config.yaml
confidence_threshold: 0.95  # Increase from 0.90
min_stage1_confidence: 0.8  # Increase from 0.7

# 2. Test with higher thresholds
python3 -c "
from src.inference.recognize import create_pipeline
pipeline = create_pipeline('config.yaml')
# Temporarily override thresholds
pipeline.config['min_stage1_confidence'] = 0.8
result = pipeline.recognize('unknown_item.jpg')
print(f'Result: {result.item_id} (confidence: {result.confidence})')
"

# 3. Verify index quality
python3 -c "
import faiss
index = faiss.read_index('data/models/faiss_index.bin')
print(f'Index vectors: {index.ntotal}, dimensions: {index.d}')
"
```

#### Low Recognition Accuracy
**Symptoms**: Known items not being recognized correctly

**Causes & Solutions**:

1. **Insufficient Training Data**:
```bash
# Check training data quantity
find data/augmented -name "*.jpg" | wc -l
# Should be: (number_of_items × 8 × 50) = 1600+ for 4 items

# If low, re-run augmentation:
python3 src/data_preparation/prepare.py \
    --input data/raw \
    --output data/augmented \
    --augmentations 50
```

2. **Poor Image Quality**:
```bash
# Check image resolution and quality
identify data/raw/item_001/*.jpg
# Look for: resolution >1080p, file size >500KB

# Re-capture with better quality if needed
```

3. **Model Issues**:
```bash
# Verify model loading
python3 -c "
import sys
sys.path.append('src')
from feature_extraction.feature_extractor import MultiModalFeatureExtractor
extractor = MultiModalFeatureExtractor({'clip_model': 'ViT-L/14'})
print('✅ Models loaded successfully')
"
```

#### Slow Recognition Performance
**Symptoms**: Recognition taking >1 second per image

**Solutions**:

1. **Enable GPU Acceleration**:
```bash
# Check GPU availability
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
"

# For Apple Silicon, ensure MPS is working:
python3 -c "
import torch
if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(1, 3, 224, 224).to(device)
    print('✅ MPS acceleration working')
"
```

2. **Optimize Batch Processing**:
```python
# Modify config.yaml for better performance
features:
  batch_size: 16  # Reduce if memory issues, increase if more memory available
```

3. **Check System Resources**:
```bash
# Monitor memory usage during recognition
python3 -c "
import psutil
import time
from src.inference.recognize import create_pipeline

pipeline = create_pipeline('config.yaml')
process = psutil.Process()

print(f'Memory before: {process.memory_info().rss / 1024 / 1024:.1f} MB')
start = time.time()
result = pipeline.recognize('data/raw/item_001/Copy of IMG_8388.JPG')
end = time.time()
print(f'Memory after: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'Recognition time: {(end-start)*1000:.1f}ms')
"
```

### 2. System Startup Issues

#### Virtual Environment Problems
**Symptoms**: Import errors, missing packages

**Solutions**:
```bash
# 1. Verify virtual environment is activated
which python3
# Should show: /path/to/ai-recognition-system/venv/bin/python3

# 2. If not activated:
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Reinstall dependencies if needed:
pip install -r requirements.txt --upgrade

# 4. Check critical packages:
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
import faiss; print(f'FAISS: {faiss.__version__}')  
import PyQt6; print(f'PyQt6: {PyQt6.__version__}')
"
```

#### Missing Files Error
**Symptoms**: FileNotFoundError for index files, models, or config

**Solutions**:
```bash
# 1. Check required files exist:
required_files=(
    "config.yaml"
    "data/models/faiss_index.bin" 
    "data/models/index_metadata.pkl"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

# 2. If index files missing, recreate them:
# (Only if you have training data)
python3 src/indexing/faiss_indexer.py \
    --features data/features.h5 \
    --output data/models

# 3. If config missing, copy from template:
cp config_template.yaml config.yaml
```

#### Port Already in Use (Backend)
**Symptoms**: "Address already in use" when starting backend

**Solutions**:
```bash
# 1. Check what's using port 8000:
lsof -i :8000
# or
netstat -tulpn | grep :8000

# 2. Kill the process:
kill -9 <PID>

# 3. Or use different port:
# Modify backend/main.py:
# uvicorn.run(app, host="0.0.0.0", port=8001)  # Change to 8001
```

### 3. GUI Issues

#### PyQt6 Import Errors
**Symptoms**: "No module named 'PyQt6'" or similar

**Solutions**:
```bash
# 1. Install PyQt6:
pip install PyQt6

# 2. If on Linux, may need additional packages:
sudo apt-get install python3-pyqt6  # Ubuntu/Debian
# or
pip install PyQt6-Qt6

# 3. Test PyQt6 installation:
python3 -c "
from PyQt6.QtWidgets import QApplication
print('✅ PyQt6 working')
"
```

#### GUI Freezing or Unresponsive
**Symptoms**: Interface becomes unresponsive during operations

**Causes & Solutions**:

1. **Long-running operations blocking GUI**:
```python
# Check if operations are running in background threads
# Look for QThread usage in widgets
```

2. **Memory issues**:
```bash
# Monitor memory during GUI operations
python3 -c "
import psutil
import time

def monitor_memory():
    process = psutil.Process()
    while True:
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f'Memory usage: {memory_mb:.1f} MB')
        time.sleep(5)

monitor_memory()
"
```

#### Image Upload Failures
**Symptoms**: "read of closed file" or similar errors during image upload

**Solutions**:
```bash
# 1. Check file permissions:
ls -la your_image_file.jpg

# 2. Test file reading:
python3 -c "
with open('your_image_file.jpg', 'rb') as f:
    data = f.read()
    print(f'✅ File readable, size: {len(data)} bytes')
"

# 3. Check supported formats:
# Supported: .jpg, .jpeg, .png, .bmp
```

### 4. Training Issues

#### Training Won't Start
**Symptoms**: Training button does nothing or immediate error

**Diagnostic Steps**:
```bash
# 1. Check training data exists:
find data/augmented -name "*.jpg" | head -10
# Should show augmented images

# 2. Check feature extraction worked:
ls -la data/features*.h5
# Should show feature files

# 3. Verify feature file integrity:
python3 -c "
import h5py
try:
    with h5py.File('data/features.h5', 'r') as f:
        print(f'Groups: {len(list(f.keys()))}')
        print('✅ Feature file is valid')
except Exception as e:
    print(f'❌ Feature file error: {e}')
"
```

#### Training Crashes with Memory Error
**Solutions**:
```bash
# 1. Reduce batch size in config.yaml:
training:
  batch_size: 8  # Reduce from 24

# 2. Monitor memory during training:
python3 -c "
import psutil
import matplotlib.pyplot as plt
import time

memory_usage = []
times = []

for i in range(60):  # Monitor for 1 minute
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    memory_usage.append(memory_mb)
    times.append(i)
    print(f'Memory: {memory_mb:.1f} MB')
    time.sleep(1)
"

# 3. Use swap if available:
# Ensure system has swap space configured
```

#### Training Accuracy Not Improving
**Symptoms**: Loss/accuracy plateauing, poor validation results

**Solutions**:
```bash
# 1. Check learning rate:
# May be too high or too low
training:
  learning_rate: 1e-4  # Try lower if oscillating
  # or 
  learning_rate: 1e-3  # Try higher if not learning

# 2. Verify data diversity:
python3 -c "
import os
from collections import Counter

# Count augmented images per item
item_counts = Counter()
for root, dirs, files in os.walk('data/augmented'):
    if files:
        item_name = os.path.basename(root)
        item_counts[item_name] = len([f for f in files if f.endswith('.jpg')])

for item, count in item_counts.items():
    print(f'{item}: {count} images')
    if count < 200:
        print(f'  ⚠️  Low count, consider more augmentation')
"

# 3. Check for overfitting:
# Monitor validation vs training loss
```

### 5. API/Backend Issues

#### Backend Won't Start
**Symptoms**: FastAPI server fails to start

**Diagnostic Steps**:
```bash
# 1. Test backend directly:
cd backend
python3 -c "
import sys
sys.path.append('../src')
try:
    from main import app
    print('✅ Backend imports successfully')
except Exception as e:
    print(f'❌ Backend import error: {e}')
"

# 2. Check dependencies:
python3 -c "
import fastapi
import uvicorn
print('✅ FastAPI dependencies available')
"

# 3. Test with debug mode:
cd backend
python3 main.py --debug
```

#### API Endpoints Not Responding
**Symptoms**: 404 errors, timeouts

**Solutions**:
```bash
# 1. Test backend health:
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# 2. List available endpoints:
curl http://localhost:8000/docs
# Opens interactive API documentation

# 3. Test recognition endpoint:
curl -X POST "http://localhost:8000/recognize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/raw/item_001/Copy of IMG_8388.JPG"
```

### 6. Performance Optimization

#### Memory Usage Too High
**Solutions**:
```bash
# 1. Monitor memory usage by component:
python3 -c "
import psutil
import torch
from src.feature_extraction.feature_extractor import MultiModalFeatureExtractor

process = psutil.Process()
print(f'Baseline memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')

# Load models
extractor = MultiModalFeatureExtractor({'clip_model': 'ViT-L/14'})
print(f'After model loading: {process.memory_info().rss / 1024 / 1024:.1f} MB')

# Clear cache if needed
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
if hasattr(torch.mps, 'empty_cache'):
    torch.mps.empty_cache()
"

# 2. Reduce batch sizes globally:
# Edit config.yaml
features:
  batch_size: 8
training:
  batch_size: 8
```

#### Disk Space Issues
**Solutions**:
```bash
# 1. Check disk usage:
du -sh data/
du -sh checkpoints/
du -sh logs/

# 2. Clean old files:
# Remove old checkpoints (keep best model)
find checkpoints/ -name "*.pth" ! -name "best_model.pth" -delete

# Remove old log files
find logs/ -name "*.log" -mtime +7 -delete

# Clean cache directories
rm -rf ~/.cache/torch/hub/checkpoints/
rm -rf ~/.cache/huggingface/
```

### 7. Development and Debugging

#### Enable Debug Logging
```bash
# 1. Set logging level in config.yaml:
logging:
  level: DEBUG
  
# 2. Or set environment variable:
export PYTHONPATH="$PWD/src"
export LOG_LEVEL=DEBUG
python3 your_script.py

# 3. Add debug prints to code:
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message here")
```

#### Profile Performance Bottlenecks
```bash
# 1. Use cProfile for detailed profiling:
python3 -m cProfile -o profile_output.prof your_script.py

# 2. Analyze results:
python3 -c "
import pstats
stats = pstats.Stats('profile_output.prof')
stats.sort_stats('cumulative').print_stats(20)
"

# 3. Memory profiling with memory_profiler:
pip install memory_profiler
python3 -m memory_profiler your_script.py
```

### 8. Emergency Recovery

#### Complete System Reset
```bash
# WARNING: This will delete all trained models and data
# Only use if system is completely broken

# 1. Backup important data:
mkdir -p backup/$(date +%Y%m%d)
cp -r data/raw backup/$(date +%Y%m%d)/
cp config.yaml backup/$(date +%Y%m%d)/

# 2. Clean everything:
rm -rf data/augmented data/features* data/models checkpoints logs

# 3. Recreate directories:
mkdir -p data/augmented data/models logs checkpoints

# 4. Start fresh:
python3 src/data_preparation/prepare.py --input data/raw --output data/augmented
python3 src/feature_extraction/feature_extractor.py --input data/augmented --output data/features.h5
python3 src/indexing/faiss_indexer.py --features data/features.h5 --output data/models
```

#### Restore from Backup
```bash
# If you have backups:
cp -r backup/20240101/data/raw ./data/
cp backup/20240101/config.yaml ./
# Then re-run the pipeline as above
```

### 9. Getting Help

#### Collect Diagnostic Information
```bash
# Run this script to collect system info for support:
python3 -c "
import sys, torch, platform, psutil
print('=== System Information ===')
print(f'Platform: {platform.platform()}')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Total RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB')
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')

print('\\n=== File Status ===')
import os
files_to_check = ['config.yaml', 'data/models/faiss_index.bin', 'data/models/index_metadata.pkl']
for file in files_to_check:
    status = '✅' if os.path.exists(file) else '❌'
    print(f'{status} {file}')
"
```

#### Log Analysis
```bash
# Check recent errors in logs:
tail -100 logs/api.log | grep ERROR
tail -100 logs/system.log | grep ERROR

# Find specific error patterns:
grep -i "memory" logs/*.log
grep -i "timeout" logs/*.log
grep -i "failed" logs/*.log
```

For additional support, include the diagnostic information output when reporting issues.