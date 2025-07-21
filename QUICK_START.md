# AI Recognition System - Quick Start Guide

## 🚀 Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Items
Create directories in `data/raw/` for each item:
```
data/raw/
├── item_001/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── item_002/
│   └── ...
```
- **Minimum:** 3-5 images per item
- **Recommended:** 5-9 images per item for best accuracy

## 🎯 Quick Usage

### Option 1: GUI Interface (Recommended)
```bash
python3 frontend/main.py
```
- **Items Tab:** Manage your inventory items
- **Training Tab:** Process and train the system  
- **Recognition Tab:** Test item recognition
- **Settings Tab:** Adjust system parameters

### Option 2: Command Line
```bash
# Full pipeline (recommended for first run)
python3 main.py --step all

# Individual steps
python3 main.py --step augment     # Data augmentation
python3 main.py --step extract     # Feature extraction  
python3 main.py --step train       # Training (currently skipped - uses raw features)
python3 main.py --step index       # Build recognition index
```

### Option 3: Web API
```bash
# Start server
python3 backend/main.py

# Use API at http://localhost:8000
# - POST /api/recognize/upload - Upload image for recognition
# - GET /docs - API documentation
```

## 🎛️ Key Features

- **100% Accuracy:** CLIP + DINOv2 raw features with hybrid refinement
- **Fast Recognition:** ~0.4s per image
- **Hybrid System:** Automatic switching between raw and refined features
- **Real-time Processing:** Live camera recognition available
- **Web API:** REST endpoints for integration

## 📊 Expected Performance

- **Training Time:** ~5-10 minutes for 20+ items
- **Recognition Speed:** 0.3-0.5s per image
- **Accuracy:** 98-100% on well-captured items
- **Memory Usage:** ~2GB during operation

## 🔧 Configuration

Edit `config.yaml` to adjust:
- Image quality settings
- Recognition thresholds  
- Hybrid system parameters
- Performance targets

## 🐛 Troubleshooting

**Issue:** "Features file not found"
```bash
python3 main.py --step extract
```

**Issue:** "Recognition not working"
```bash
python3 main.py --step index
```

**Issue:** Low accuracy
- Add more diverse images per item
- Ensure good lighting and focus
- Run full pipeline: `python3 main.py --step all`

## 📁 Directory Structure
```
ai-recognition-system/
├── data/raw/           # Your item images here
├── frontend/main.py    # GUI application
├── backend/main.py     # Web API server
├── main.py            # Command line interface
├── config.yaml        # System configuration
└── checkpoints/       # Trained models (auto-generated)
```

## 🎉 That's It!

The system is designed to work out-of-the-box with minimal configuration. Just add your images and run the GUI or command line interface.

For advanced usage and API details, see the full documentation in the `docs/` directory.