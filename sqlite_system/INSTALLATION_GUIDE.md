# Installation Guide - SQLite AI Recognition System

## 🚀 **Quick Installation Commands**

### **Method 1: Install All Dependencies (Recommended)**
```bash
# Navigate to your project directory
cd /mnt/d/Workspace/AI\ Desktop\ App/AI/sqlite_system

# Install all requirements
pip install -r requirements.txt

# If OpenCV installation fails, try:
pip install opencv-python-headless
```

### **Method 2: Individual Package Installation**
```bash
# Core packages
pip install torch torchvision transformers
pip install opencv-python pillow numpy
pip install albumentations rembg onnxruntime
pip install PySide6 PySide6-Addons
pip install psutil pyyaml tqdm sqlite-utils

# CLIP model (from GitHub)
pip install git+https://github.com/openai/CLIP.git

# Optional performance packages
pip install numba timm scikit-learn scipy
```

---

## 🔧 **OpenCV Installation Troubleshooting**

### **Common OpenCV Issues & Solutions:**

#### **Issue 1: `ModuleNotFoundError: No module named 'cv2'`**
```bash
# Solution 1: Standard installation
pip install opencv-python

# Solution 2: Headless version (no GUI dependencies)
pip install opencv-python-headless

# Solution 3: Force reinstall
pip uninstall opencv-python
pip install --no-cache-dir opencv-python

# Solution 4: Use conda if pip fails
conda install -c conda-forge opencv
```

#### **Issue 2: OpenCV Import Error on Linux**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Then reinstall OpenCV
pip install opencv-python
```

#### **Issue 3: OpenCV Import Error on macOS**
```bash
# Install via Homebrew first
brew install opencv

# Then install Python package
pip install opencv-python

# Or use conda
conda install -c conda-forge opencv
```

#### **Issue 4: Windows OpenCV Issues**
```bash
# Try different OpenCV versions
pip install opencv-python==4.8.1.78
pip install opencv-contrib-python

# If still failing, use conda
conda install -c conda-forge opencv
```

---

## 🖥️ **Platform-Specific Installation**

### **Windows with NVIDIA GPU:**
```bash
# 1. Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **macOS Apple Silicon:**
```bash
# 1. Install PyTorch with MPS support
pip install torch torchvision

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Verify MPS installation
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### **Linux with NVIDIA GPU:**
```bash
# 1. Install NVIDIA drivers and CUDA toolkit
# Follow NVIDIA's official installation guide

# 2. Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install system dependencies
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# 4. Install Python dependencies
pip install -r requirements.txt
```

---

## 🧪 **Verify Installation**

### **Test Core Dependencies:**
```bash
cd /mnt/d/Workspace/AI\ Desktop\ App/AI/sqlite_system

# Test OpenCV
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Test PyTorch with GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test Albumentations
python -c "import albumentations; print('Albumentations: OK')"

# Test PySide6 (GUI)
python -c "import PySide6; print('PySide6: OK')"

# Test CLIP
python -c "import clip; print('CLIP: OK')"
```

### **Run System Test:**
```bash
# Test platform detection
python src/utils/platform_detector.py

# Test GUI launch
python gui_main.py

# Test optimized DataLoader
python src/utils/optimized_dataloader.py
```

---

## 📦 **Virtual Environment Setup (Recommended)**

### **Create Isolated Environment:**
```bash
# Navigate to project directory
cd /mnt/d/Workspace/AI\ Desktop\ App/AI

# Create virtual environment
python -m venv sqlite_env

# Activate environment
# Windows:
sqlite_env\Scripts\activate
# Linux/macOS:
source sqlite_env/bin/activate

# Install dependencies
cd sqlite_system
pip install -r requirements.txt

# Verify installation
python gui_main.py
```

---

## 🔍 **Common Installation Issues & Fixes**

### **Issue: PySide6 Installation Fails**
```bash
# Solution: Install Qt6 dependencies first
# Ubuntu/Debian:
sudo apt-get install qt6-base-dev

# macOS:
brew install qt6

# Then install PySide6
pip install PySide6 PySide6-Addons
```

### **Issue: PyTorch CUDA Not Working**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Issue: Memory Errors During Installation**
```bash
# Install packages individually with no cache
pip install --no-cache-dir torch
pip install --no-cache-dir torchvision
pip install --no-cache-dir opencv-python
# ... continue for other large packages
```

### **Issue: rembg Background Removal Fails**
```bash
# Install ONNX runtime separately
pip install onnxruntime

# For GPU acceleration (optional):
pip install onnxruntime-gpu

# Test background removal
python -c "from rembg import remove; print('rembg: OK')"
```

---

## ⚡ **Performance Optimization After Installation**

### **GPU Memory Optimization:**
```bash
# Set environment variables for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# Add to your ~/.bashrc or ~/.zshrc for permanent effect
```

### **CPU Optimization:**
```bash
# Set OpenMP threads (adjust based on your CPU cores)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

---

## 🎯 **Installation Verification Script**

Save this as `verify_installation.py` and run it:

```python
#!/usr/bin/env python3
"""
Installation verification script for SQLite AI Recognition System
"""

def check_installation():
    print("🔍 Checking SQLite AI Recognition System Installation...\n")
    
    # Core dependencies
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            print(f"   MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("❌ PyTorch not installed")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed - Run: pip install opencv-python")
    
    try:
        import albumentations
        print(f"✅ Albumentations: {albumentations.__version__}")
    except ImportError:
        print("❌ Albumentations not installed")
    
    try:
        import PySide6
        print(f"✅ PySide6: {PySide6.__version__}")
    except ImportError:
        print("❌ PySide6 not installed")
    
    try:
        import clip
        print("✅ CLIP: Available")
    except ImportError:
        print("❌ CLIP not installed")
    
    try:
        from rembg import remove
        print("✅ rembg: Background removal available")
    except ImportError:
        print("⚠️  rembg not installed (background removal unavailable)")
    
    print("\n🚀 Installation check complete!")

if __name__ == "__main__":
    check_installation()
```

Run with: `python verify_installation.py`

---

## 📞 **Getting Help**

If you encounter installation issues:

1. **Check Python version**: `python --version` (requires Python 3.8+)
2. **Update pip**: `pip install --upgrade pip`
3. **Clear pip cache**: `pip cache purge`
4. **Use virtual environment**: Isolates dependencies
5. **Check system requirements**: Ensure sufficient disk space and memory

**Common Installation Commands Summary:**
```bash
# Quick fix for most issues:
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# OpenCV specific fix:
pip install opencv-python-headless

# GPU PyTorch fix:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```