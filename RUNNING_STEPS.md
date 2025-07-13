# AI Recognition System - Complete Running Guide

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: 10GB free space for models and data
- **GPU**: Optional but recommended (CUDA-compatible)

### Software Dependencies
- Git for version control
- Python virtual environment support
- PyQt6 system dependencies (varies by OS)

## Installation Guide

### Step 1: Clone and Setup Project
```bash
# Clone the repository
git clone <repository-url>
cd ai-recognition-system

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support (optional, requires CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

### Step 3: Verify Installation
```bash
# Test if all dependencies are installed correctly
python -c "import torch, torchvision, clip, faiss, cv2; print('All dependencies installed successfully')"
```

## Quick Start (5-minute setup)

### Option A: One-Command Startup
```bash
# Start the complete system with dependency checking
python start_system.py
```

This command will:
- Check all dependencies
- Create necessary directories
- Start the FastAPI backend
- Launch the PyQt6 frontend
- Verify system connectivity

### Option B: Manual Startup
```bash
# Terminal 1: Start backend API
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start frontend GUI
python frontend/main.py
```

## Complete System Workflow

### 1. Initial Setup and Configuration

#### A. System Configuration
1. Open the application using `python start_system.py`
2. Navigate to **Settings** tab
3. Configure system parameters:
   - **Recognition Threshold**: 0.85 (default)
   - **Batch Size**: 32 for training
   - **Image Size**: 224x224 pixels
   - **Augmentation Factor**: 50x per image

#### B. Directory Structure Verification
The system automatically creates:
```
data/
├── raw/           # Place your 8 images per item here
├── augmented/     # Auto-generated training data
├── processed/     # Preprocessed images
├── embeddings/    # Feature vectors
└── models/        # Trained models and indices
```

### 2. Adding Items and Training Data

#### A. Add New Items
1. Click **Items** in the navigation sidebar
2. Click **Add New Item** button
3. Fill in item details:
   - **Item Name**: Unique identifier
   - **Category**: Product category
   - **Description**: Optional details
4. Click **Upload Images** and select 8+ high-quality images
5. Save the item

#### B. Image Quality Guidelines
- **Resolution**: Minimum 1024x1024, Recommended 4K
- **Lighting**: Consistent, well-lit conditions
- **Angles**: Multiple viewpoints (front, back, sides, top)
- **Background**: Clean, minimal distractions
- **Format**: JPG, PNG, JPEG supported

### 3. Training the Recognition Model

#### A. Start Training Process
1. Navigate to **Training** tab
2. Review training configuration:
   - **Items to Train**: Shows available items
   - **Augmentation**: 50x per image (400+ total per item)
   - **Epochs**: 100 (with early stopping)
3. Click **Start Training**

#### B. Monitor Training Progress
- **Real-time Metrics**: Loss, accuracy, validation scores
- **Progress Bar**: Current epoch and estimated completion
- **Live Logs**: Detailed training information
- **Early Stopping**: Automatic halt when performance plateaus

#### C. Training Timeline
- **Small Dataset** (1-10 items): 10-30 minutes
- **Medium Dataset** (10-100 items): 1-2 hours
- **Large Dataset** (100+ items): 2-4 hours

### 4. System Evaluation

#### A. Run Comprehensive Evaluation
1. Navigate to **Evaluation** tab
2. Click **Run Evaluation**
3. The system will:
   - Test recognition accuracy on validation images
   - Measure inference speed
   - Calculate confidence distributions
   - Generate performance reports

#### B. Evaluation Metrics
- **Overall Accuracy**: Target 95%+
- **Per-Item Performance**: Individual item accuracy
- **Speed Analysis**: Average recognition time
- **Confidence Analysis**: Score distributions
- **Target Achievement**: Pass/fail against thresholds

#### C. Performance Optimization
Based on evaluation results:
- **Low Accuracy** (<90%): Add more training images, check image quality
- **Slow Recognition** (>500ms): Enable GPU, optimize batch size
- **High False Positives**: Increase recognition threshold

### 5. Real-time Recognition

#### A. Camera Recognition
1. Navigate to **Recognition** tab
2. Click **Start Camera**
3. Point camera at item
4. View real-time recognition results:
   - **Item Name**: Recognized item
   - **Confidence Score**: Recognition certainty
   - **Processing Time**: Speed metrics

#### B. File Recognition
1. Click **Upload Image** in Recognition tab
2. Select image file to recognize
3. View detailed results:
   - **Top Matches**: Ranked recognition results
   - **Confidence Scores**: For each match
   - **Feature Visualization**: Similar regions highlighted

### 6. System Monitoring

#### A. Real-time Status
- **Status Bar**: Shows system health, API connectivity
- **Dashboard**: Overview of system statistics
- **Activity Log**: Recent operations and results

#### B. Log Analysis
1. Navigate to **Logs** tab
2. Filter logs by:
   - **Level**: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - **Time Range**: Last hour, day, week
   - **Component**: Backend, Training, Recognition
   - **Search Terms**: Specific keywords
3. Export logs for analysis

## Advanced Usage

### Batch Processing
```bash
# Process multiple images at once
python scripts/batch_recognize.py --input_dir /path/to/images --output_dir /path/to/results
```

### API Usage
```python
import requests

# Recognize an image via API
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/api/recognition/recognize',
        files={'image': f}
    )
    result = response.json()
    print(f"Recognized: {result['item_name']} (confidence: {result['confidence']})")
```

### Custom Configuration
```yaml
# config.yaml
recognition:
  threshold: 0.85
  max_candidates: 5
  use_geometric_verification: true

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10

system:
  gpu_enabled: true
  cache_size: 1000
  log_level: INFO
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Backend Connection Failed
**Symptoms**: "Backend Connection Failed" error on startup

**Solutions**:
```bash
# Check if port 8000 is available
netstat -an | grep 8000

# Kill any process using port 8000
pkill -f "uvicorn"

# Restart backend manually
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

#### 2. Low Recognition Accuracy
**Symptoms**: Recognition accuracy below 90%

**Solutions**:
1. **Check Image Quality**:
   - Ensure high resolution (1024x1024+)
   - Good lighting conditions
   - Multiple angles per item

2. **Increase Training Data**:
   - Add more images per item (8+ recommended)
   - Verify augmentation is working (check data/augmented/)

3. **Adjust Parameters**:
   - Lower recognition threshold (0.75-0.80)
   - Increase training epochs
   - Add more augmentation techniques

#### 3. Slow Recognition Performance
**Symptoms**: Recognition takes >1 second per image

**Solutions**:
```bash
# Enable GPU acceleration (if CUDA available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu

# Optimize batch processing
# Edit config.yaml:
system:
  batch_size: 64
  use_optimization: true
```

#### 4. Memory Issues
**Symptoms**: Out of memory errors during training

**Solutions**:
1. **Reduce Batch Size**:
```yaml
# config.yaml
training:
  batch_size: 16  # Reduce from 32
```

2. **Enable Incremental Learning**:
```yaml
training:
  incremental_mode: true
  checkpoint_frequency: 5
```

#### 5. GUI Not Starting
**Symptoms**: PyQt6 errors or GUI doesn't appear

**Solutions**:
```bash
# On Linux: Install Qt6 system dependencies
sudo apt-get install qt6-base-dev

# On macOS: Install via Homebrew
brew install qt6

# On Windows: Ensure Visual C++ Redistributable is installed
```

### Performance Optimization

#### GPU Acceleration Setup
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Install GPU-optimized packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

#### Memory Optimization
```yaml
# config.yaml - Memory-optimized settings
system:
  cache_size: 500         # Reduce cache size
  batch_size: 16          # Smaller batches
  precision: mixed        # Use mixed precision

training:
  gradient_accumulation: 2 # Accumulate gradients
  checkpoint_every: 5     # More frequent checkpoints
```

## Production Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t ai-recognition-system .

# Run container
docker run -p 8000:8000 -v ./data:/app/data ai-recognition-system
```

### System Service Setup
```bash
# Create systemd service (Linux)
sudo cp scripts/ai-recognition.service /etc/systemd/system/
sudo systemctl enable ai-recognition
sudo systemctl start ai-recognition
```

## Maintenance and Updates

### Regular Maintenance Tasks
1. **Weekly**: Review system logs for errors
2. **Monthly**: Backup trained models and data
3. **Quarterly**: Retrain models with new data
4. **Yearly**: Update dependencies and system

### Backup Strategy
```bash
# Backup trained models
cp -r data/models/ backup/models_$(date +%Y%m%d)/

# Backup configuration
cp config.yaml backup/config_$(date +%Y%m%d).yaml

# Backup database
cp -r data/embeddings/ backup/embeddings_$(date +%Y%m%d)/
```

### Update Process
```bash
# Update application
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart system
python start_system.py
```

## Support and Resources

### Log Files Locations
- **Backend Logs**: `backend/logs/api.log`
- **Training Logs**: `logs/training.log`
- **Recognition Logs**: `logs/recognition.log`
- **System Logs**: `logs/system.log`

### Configuration Files
- **Main Config**: `config.yaml`
- **Training Config**: `configs/training.yaml`
- **Recognition Config**: `configs/recognition.yaml`

### Getting Help
1. **Check Logs**: Review relevant log files for error messages
2. **Documentation**: Refer to API_REFERENCE.md for technical details
3. **GitHub Issues**: Report bugs and request features
4. **Performance Issues**: Use SYSTEM_EXPLANATION.md for optimization

This guide provides comprehensive instructions for successfully running and maintaining the AI Recognition System.