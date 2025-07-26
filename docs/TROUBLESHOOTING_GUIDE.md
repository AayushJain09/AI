# AI Recognition System - Troubleshooting Guide

## Quick Diagnosis

### System Status Check
```bash
# Quick system health check
python -c "
from src.inference.recognize import create_pipeline
pipeline = create_pipeline('config.yaml')
print(f'✅ Pipeline loaded: {pipeline.index.ntotal} vectors')
result = pipeline.recognize('data/raw/item_001/Copy of IMG_8388.JPG')
print(f'✅ Test recognition: {result.item_id} ({result.confidence:.3f})')
"
```

### Performance Quick Check
```bash
# Check current evaluation results
cat evaluation_results.json | python -m json.tool | head -20
```

## Common Issues and Solutions

### 1. New Item Recognition Failures

#### Symptom: New item not recognized correctly
```
Expected: item_NEW
Actual: unknown (confidence: 0.45)
```

#### Diagnosis Steps:
```python
# Check image quality
from src.validation import ImageQualityValidator
validator = ImageQualityValidator()

for img_path in training_images:
    metrics = validator.validate_single_image(img_path)
    print(f"{img_path}: Score {metrics.overall_score:.3f}")
    if metrics.issues:
        print(f"  Issues: {metrics.issues}")
```

#### Root Causes & Solutions:

**A. Poor Image Quality**
```python
# Check specific quality metrics
if metrics.sharpness_score < 0.15:
    print("❌ Images too blurry")
    print("Solution: Retake with better focus, use tripod")

if metrics.brightness_score < 0.1 or metrics.brightness_score > 0.95:
    print("❌ Poor lighting")
    print("Solution: Use natural lighting, avoid flash")

if min(metrics.resolution) < 768:
    print("❌ Resolution too low") 
    print("Solution: Use higher resolution camera settings")
```

**B. Insufficient Training Data**
```python
# Check diversity
results = validator.validate_image_set(training_images)
diversity = results['diversity_analysis']

if diversity.overall_diversity < 0.4:
    print("❌ Images too similar")
    print("Solution: Add different angles, lighting, distances")

if len(training_images) < 5:
    print("❌ Not enough training images")
    print("Solution: Add 5-10 diverse, high-quality images")
```

**C. Confusion with Existing Items**
```python
# Test against similar items
pipeline = create_pipeline('config.yaml')
test_result = pipeline.recognize(training_images[0])

if test_result.item_id != 'item_NEW' and test_result.confidence > 0.8:
    print(f"❌ Confused with {test_result.item_id}")
    print("Solution: Add more distinctive training examples")
    print("Solution: Check if items are actually different")
```

### 2. System Performance Issues

#### Symptom: Slow inference times (> 1 second)
```
Average inference time: 2.35s (target: <0.5s)
```

#### Diagnosis Steps:
```python
import time
import psutil

# Profile system performance
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
print(f"CPU usage: {process.cpu_percent()}%")

# Time individual components
start_time = time.time()
pipeline = create_pipeline('config.yaml')
load_time = time.time() - start_time
print(f"Pipeline load time: {load_time:.2f}s")

start_time = time.time()
result = pipeline.recognize('test_image.jpg')
inference_time = time.time() - start_time
print(f"Single inference time: {inference_time:.3f}s")
```

#### Solutions:

**A. Memory Issues**
```bash
# Check for memory leaks
free -h
top -p $(pgrep -f python)

# Solution: Restart system if memory > 4GB
sudo systemctl restart your-service
```

**B. Image Processing Bottleneck**
```python
# Check image preprocessing time
import cv2
start_time = time.time()
image = cv2.imread('large_image.jpg')
resized = cv2.resize(image, (768, 768))
process_time = time.time() - start_time
print(f"Image processing: {process_time:.3f}s")

# Solution: Resize images before recognition
if process_time > 0.1:
    print("Pre-resize images to 768x768 before recognition")
```

**C. Model Loading Issues**
```python
# Check if models are being reloaded
from src.feature_extraction.feature_extractor import MultiModalFeatureExtractor

# Time feature extraction
extractor = MultiModalFeatureExtractor({})
start_time = time.time()
features = extractor.extract_all_features('test_image.jpg')
extract_time = time.time() - start_time
print(f"Feature extraction: {extract_time:.3f}s")

# Solution: Enable model caching
config = {
    'features': {
        'cache_models': True,
        'use_mixed_precision': True
    }
}
```

### 3. Accuracy Degradation

#### Symptom: Previously working items now fail
```
item_001: Was 100%, now 85%
Overall accuracy dropped from 1.0 to 0.85
```

#### Diagnosis Steps:
```python
# Compare with previous results
import json
with open('evaluation_results.json') as f:
    current_results = json.load(f)

with open('previous_evaluation_results.json') as f:
    previous_results = json.load(f)

# Check per-item degradation
current_items = current_results['performance_report']['per_item_metrics']
previous_items = previous_results['performance_report']['per_item_metrics']

for item_id in current_items:
    if item_id in previous_items:
        current_acc = current_items[item_id]['accuracy']
        previous_acc = previous_items[item_id]['accuracy']
        
        if current_acc < previous_acc - 0.1:
            print(f"❌ {item_id}: {previous_acc:.3f} → {current_acc:.3f}")
```

#### Root Causes & Solutions:

**A. Configuration Changes**
```bash
# Check recent config changes
git log -p --since="1 week ago" config.yaml

# Solution: Revert problematic config changes
git checkout HEAD~1 config.yaml
```

**B. Model File Corruption**
```python
# Check model file integrity
import torch
from pathlib import Path

model_path = Path('checkpoints/best_model.pth')
if model_path.exists():
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loads successfully")
        print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        print(f"❌ Model corrupted: {e}")
        print("Solution: Restore from backup or retrain")
```

**C. Index File Issues**
```python
# Check FAISS index integrity
import faiss
try:
    index = faiss.read_index('data/models/faiss_index.bin')
    print(f"✅ Index loaded: {index.ntotal} vectors, {index.d} dimensions")
    
    # Test search
    import numpy as np
    test_vector = np.random.random((1, index.d)).astype(np.float32)
    similarities, indices = index.search(test_vector, 5)
    print(f"✅ Index search working")
    
except Exception as e:
    print(f"❌ Index corrupted: {e}")
    print("Solution: Rebuild index from features")
```

### 4. Unknown Item Detection Problems

#### Symptom: Unknown items incorrectly identified
```
Unknown image → item_003 (confidence: 0.92)
Should be: unknown (confidence: <0.85)
```

#### Diagnosis Steps:
```python
# Test unknown item detection
unknown_test_images = [
    '/path/to/clearly/different/image.jpg',
    '/path/to/noise/pattern.jpg'
]

false_positives = []
for img_path in unknown_test_images:
    result = pipeline.recognize(img_path)
    if result.item_id != "unknown":
        false_positives.append((img_path, result.item_id, result.confidence))

print(f"False positives: {len(false_positives)}")
for img, predicted, conf in false_positives:
    print(f"  {Path(img).name} → {predicted} ({conf:.3f})")
```

#### Solutions:

**A. Confidence Threshold Too Low**
```yaml
# Increase confidence threshold in config.yaml
recognition:
  confidence_threshold: 0.95  # Increase from 0.90
  min_stage1_confidence: 0.8  # Increase from 0.7
```

**B. Missing Negative Examples**
```python
# Add negative training examples to reduce false positives
# This requires model retraining with hard negative mining
print("Solution: Collect challenging unknown images for training")
print("Add to training dataset as negative examples")
```

### 5. Training/Integration Issues

#### Symptom: Cannot add new items to system
```python
pipeline.add_item_to_index('new_item', image_paths)
# Error: Feature extraction failed
```

#### Diagnosis Steps:
```python
# Test feature extraction pipeline
from src.feature_extraction.feature_extractor import MultiModalFeatureExtractor

extractor = MultiModalFeatureExtractor({
    'clip_variant': 'ViT-L/14',
    'dinov2_variant': 'dinov2_vitb14'
})

for img_path in image_paths:
    try:
        features = extractor.extract_all_features(img_path)
        if features is None:
            print(f"❌ Feature extraction failed: {img_path}")
        else:
            print(f"✅ Features extracted: {img_path}")
            print(f"   CLIP: {features['clip'].shape}")
            print(f"   DINOv2: {features.get('dinov2', 'None')}")
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
```

#### Solutions:

**A. Image Format Issues**
```python
# Check image format compatibility
from PIL import Image
import cv2

for img_path in image_paths:
    try:
        # Test PIL loading
        pil_img = Image.open(img_path)
        print(f"✅ PIL: {img_path} - {pil_img.size} {pil_img.mode}")
        
        # Test OpenCV loading
        cv_img = cv2.imread(img_path)
        if cv_img is not None:
            print(f"✅ OpenCV: {img_path} - {cv_img.shape}")
        else:
            print(f"❌ OpenCV failed: {img_path}")
            
    except Exception as e:
        print(f"❌ Cannot load {img_path}: {e}")
        print("Solution: Convert to standard JPEG format")
```

**B. Memory Issues During Training**
```python
# Check available memory
import psutil
memory = psutil.virtual_memory()
print(f"Available memory: {memory.available / 1024 / 1024:.1f} MB")

if memory.available < 2048 * 1024 * 1024:  # Less than 2GB
    print("❌ Insufficient memory")
    print("Solution: Close other applications or reduce batch size")
```

### 6. Configuration Issues

#### Symptom: System fails to start or behaves unexpectedly

#### Diagnosis Steps:
```python
# Validate configuration
import yaml
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Config file loads successfully")
    
    # Check required sections
    required_sections = ['data', 'model', 'recognition', 'features']
    for section in required_sections:
        if section in config:
            print(f"✅ {section} section present")
        else:
            print(f"❌ {section} section missing")
            
except Exception as e:
    print(f"❌ Config file error: {e}")
```

#### Solutions:

**A. Invalid YAML Syntax**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Or use online YAML validator
# Fix indentation and syntax errors
```

**B. Missing Required Files**
```python
# Check if configured files exist
import os
paths_to_check = [
    config['data']['raw_images_dir'],
    config['recognition']['index_path'],
    config['recognition']['metadata_path']
]

for path in paths_to_check:
    if os.path.exists(path):
        print(f"✅ {path}")
    else:
        print(f"❌ Missing: {path}")
        print(f"Solution: Create directory or file: {path}")
```

## System Recovery Procedures

### 1. Complete System Reset

#### When to Use:
- Multiple components failing
- Corrupted model files
- Inconsistent behavior across items

#### Procedure:
```bash
# 1. Backup current data
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp -r data/models backups/$(date +%Y%m%d_%H%M%S)/
cp evaluation_results.json backups/$(date +%Y%m%d_%H%M%S)/

# 2. Clean corrupted files
rm -f data/models/faiss_index.bin
rm -f data/models/index_metadata.pkl
rm -f data/features.h5

# 3. Rebuild from scratch
python src/data_preparation/prepare.py
python src/indexing/faiss_indexer.py
python tests/system/test_recognition_final.py

# 4. Verify system works
python -c "
from src.inference.recognize import create_pipeline
pipeline = create_pipeline('config.yaml')
print('System restored successfully')
"
```

### 2. Selective Recovery

#### When to Use:
- Specific item failures
- Partial corruption
- Need to preserve most data

#### Procedure:
```python
# Identify and remove problematic items
from src.inference.recognize import create_pipeline
import json

pipeline = create_pipeline('config.yaml')

# Test each item
with open('evaluation_results.json') as f:
    results = json.load(f)

problematic_items = []
for item_id, metrics in results['performance_report']['per_item_metrics'].items():
    if metrics['accuracy'] < 0.9:
        problematic_items.append(item_id)
        print(f"❌ Problematic item: {item_id} (accuracy: {metrics['accuracy']:.3f})")

# Remove from index and retrain
for item_id in problematic_items:
    print(f"Retraining {item_id}...")
    # Find training images
    import glob
    training_images = glob.glob(f'data/raw/{item_id}/*.jpg')
    
    # Re-add to index
    pipeline.add_item_to_index(item_id, training_images)
```

### 3. Performance Recovery

#### When to Use:
- Slow inference times
- Memory issues
- Resource exhaustion

#### Procedure:
```bash
# 1. Clear system caches
sync
echo 3 > /proc/sys/vm/drop_caches  # Linux only

# 2. Restart Python processes
pkill -f python
sleep 5

# 3. Check system resources
free -h
df -h
nvidia-smi  # If using GPU

# 4. Optimize configuration
# Edit config.yaml:
#   features.batch_size: 8  # Reduce from 16
#   features.use_mixed_precision: true
#   recognition.cache_size: 500  # Reduce from 1000

# 5. Test performance
python scripts/benchmark_hybrid_system.py
```

## Monitoring and Prevention

### 1. Regular Health Checks

```bash
# Create monitoring script
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
import json
import time
from src.inference.recognize import create_pipeline

def health_check():
    print("🔍 AI Recognition System Health Check")
    print("=" * 40)
    
    try:
        # Load system
        start_time = time.time()
        pipeline = create_pipeline('config.yaml')
        load_time = time.time() - start_time
        print(f"✅ System load: {load_time:.2f}s")
        
        # Test recognition
        test_image = 'data/raw/item_001/Copy of IMG_8388.JPG'
        start_time = time.time()
        result = pipeline.recognize(test_image)
        inference_time = time.time() - start_time
        
        print(f"✅ Test recognition: {result.item_id} ({result.confidence:.3f})")
        print(f"✅ Inference time: {inference_time:.3f}s")
        
        # Check evaluation results
        with open('evaluation_results.json') as f:
            eval_results = json.load(f)
        
        accuracy = eval_results['evaluation_summary']['overall_accuracy']
        print(f"✅ System accuracy: {accuracy:.3f}")
        
        # Overall status
        if accuracy >= 0.95 and inference_time <= 1.0:
            print("🎉 System status: HEALTHY")
            return 0
        else:
            print("⚠️  System status: DEGRADED")
            return 1
            
    except Exception as e:
        print(f"❌ System status: FAILED - {e}")
        return 2

if __name__ == "__main__":
    exit(health_check())
EOF

chmod +x health_check.py

# Run daily via cron
# crontab -e
# 0 9 * * * /path/to/health_check.py >> /var/log/ai_health.log 2>&1
```

### 2. Performance Monitoring

```python
# Set up continuous monitoring
from src.validation import PerformanceMonitor

monitor = PerformanceMonitor(config)

# Add alert callback
def alert_handler(alert):
    print(f"🚨 ALERT: {alert.description}")
    if alert.severity == 'critical':
        # Send email/notification
        send_alert_notification(alert)

monitor.add_alert_callback(alert_handler)
monitor.start_monitoring()

# Generate weekly reports
import schedule
def weekly_report():
    report = monitor.generate_performance_report(hours=168)  # 1 week
    print(f"Weekly accuracy: {report.avg_accuracy:.3f}")
    if report.avg_accuracy < 0.95:
        print("⚠️  Performance degradation detected")

schedule.every().monday.at("09:00").do(weekly_report)
```

### 3. Backup and Recovery

```bash
# Automated backup script
cat > backup_system.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/ai_recognition/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup critical files
cp -r data/models "$BACKUP_DIR/"
cp -r checkpoints "$BACKUP_DIR/"
cp config.yaml "$BACKUP_DIR/"
cp evaluation_results.json "$BACKUP_DIR/"

# Create system snapshot
python -c "
from src.inference.recognize import create_pipeline
pipeline = create_pipeline('config.yaml')
print(f'Backup created: {pipeline.index.ntotal} vectors')
" > "$BACKUP_DIR/system_info.txt"

echo "Backup completed: $BACKUP_DIR"

# Clean old backups (keep 30 days)
find /backups/ai_recognition -type d -mtime +30 -exec rm -rf {} +
EOF

chmod +x backup_system.sh

# Schedule daily backups
# crontab -e  
# 0 2 * * * /path/to/backup_system.sh
```

## Emergency Contacts and Escalation

### Level 1: Self-Resolution (< 1 hour)
- Follow troubleshooting steps above
- Check system logs
- Restart services
- Run health check

### Level 2: System Recovery (< 4 hours)
- Full system reset procedure
- Restore from backup
- Rebuild indexes
- Comprehensive testing

### Level 3: Development Support (< 24 hours)
- Contact system developers
- Provide diagnostic information:
  ```bash
  # Collect diagnostic info
  python health_check.py > diagnostic_report.txt
  cat evaluation_results.json >> diagnostic_report.txt
  cat config.yaml >> diagnostic_report.txt
  ```

### Critical System Failure
- Immediately switch to backup system
- Notify all stakeholders
- Begin emergency recovery procedures
- Document incident for post-mortem

## Frequently Asked Questions

### Q: Why is my new item always recognized as "unknown"?
**A:** Check confidence thresholds and image quality. Run validation:
```python
from src.validation import ValidationProtocols
protocols = ValidationProtocols('config.yaml')
report = protocols.validate_new_item('your_item', image_paths)
print(report.recommendations)
```

### Q: System was working fine yesterday, now accuracy is 80%
**A:** Likely configuration or file corruption. Check:
1. Recent changes: `git log --oneline --since="yesterday"`
2. File integrity: Run system health check
3. Restore from backup if needed

### Q: Recognition is very slow (>2 seconds per image)
**A:** Performance issue. Check:
1. System resources: `free -h` and `top`
2. Image sizes: Resize to 768x768 before recognition
3. Enable caching: Set `cache_models: true` in config

### Q: Getting "CUDA out of memory" errors
**A:** Memory management issue:
```bash
# Check GPU memory
nvidia-smi

# Solutions:
# 1. Reduce batch size in config
# 2. Use CPU instead of GPU
# 3. Restart Python processes
```

### Q: Items are being confused with each other
**A:** Similarity issue:
1. Add more distinctive training images
2. Check if items are actually different
3. Increase confidence threshold
4. Run diversity analysis on training data

This troubleshooting guide covers the most common issues and provides systematic approaches to diagnose and resolve problems. Keep this guide accessible and update it as new issues are discovered.