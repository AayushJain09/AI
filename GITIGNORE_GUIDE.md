# .gitignore Configuration Guide

## 📁 **What's Excluded from Git**

### **🚫 Large Data Files (Excluded)**
- **Database files**: `*.db`, `*.sqlite` (recognition.db, etc.)
- **Model files**: `*.pth`, `*.pkl`, `*.bin` (FAISS indices, neural networks)
- **Image datasets**: `*.jpg`, `*.png` (raw images, augmented datasets)
- **Cache files**: `__pycache__/`, `.cache/`, processing caches

### **🚫 Sensitive/Environment Files**
- **Virtual environments**: `env/`, `venv/`, `.env/`
- **Configuration**: `config_local.yaml`, `secrets.yaml`, `.env`
- **API keys**: `*.key`, `api_keys.txt`
- **User-specific settings**: `user_config.json`

### **🚫 System-Generated Files**
- **Logs**: `*.log`, `logs/`, system output
- **Temporary files**: `temp_processing/`, `*.tmp`
- **OS files**: `.DS_Store` (macOS), `Thumbs.db` (Windows)
- **IDE files**: `.vscode/`, `.idea/`, editor configs

### **🚫 Build Artifacts**
- **Python builds**: `dist/`, `build/`, `*.egg-info/`
- **Documentation**: `docs/_build/`, generated docs
- **Test results**: `.coverage`, `htmlcov/`, test artifacts

## ✅ **What's Included in Git**

### **📜 Essential Code & Config**
```
✅ Source code (.py files)
✅ Configuration templates (config.yaml.example)
✅ Documentation (.md files)
✅ Requirements (requirements.txt)
✅ Project structure (README.md, CLAUDE.md)
```

### **📁 Directory Structure**
```
✅ Empty directories with .gitkeep files
✅ Project architecture
✅ Module organization
✅ Examples and demos
```

## 🎯 **Project-Specific Exclusions**

### **AI/ML Specific**
- **Model checkpoints**: `checkpoints/` (except .gitkeep)
- **Training logs**: `runs/`, `tensorboard_logs/`, `wandb/`
- **Vector indices**: `*faiss_index*.bin`, FAISS files
- **Feature caches**: `features_cache/`, extraction artifacts

### **Recognition System Specific**
- **Database storage**: `data/recognition.db`, SQLite files
- **Processed images**: `data/augmented/`, temporary processing
- **Background cache**: `synthetic_backgrounds/`, cached backgrounds
- **Performance results**: `benchmark_*.json`, evaluation data

## 📋 **Directory Structure Maintained**

### **Tracked Empty Directories**
```
data/.gitkeep                 # Main data directory
checkpoints/.gitkeep          # Model checkpoints directory
logs/.gitkeep                # System logs directory (if created)
models/.gitkeep              # Model files directory (if created)
```

### **Why These Directories Matter**
- **data/**: Required for database and file storage
- **checkpoints/**: Expected location for `lightweight_refiner.pth`
- **logs/**: System logging output location
- **models/**: FAISS indices and model artifacts

## 🔧 **Usage Guidelines**

### **Adding New Files**
```bash
# Check what's ignored
git status

# Force add important files that might be ignored
git add -f important_config.yaml

# Add exception to .gitignore if needed
echo "!important_config.yaml" >> .gitignore
```

### **Managing Large Files**
```bash
# For large model files, consider Git LFS
git lfs track "*.pth"
git lfs track "checkpoints/*"

# Or use external storage and document in README
```

### **Local Development**
```bash
# Create local config (automatically ignored)
cp config.yaml.example config_local.yaml

# Edit local settings without affecting repository
```

## 🚨 **Important Notes**

### **Security**
- **Never commit**: API keys, passwords, database files with user data
- **Always ignore**: Local configurations, personal settings
- **Use examples**: Provide `.example` templates for configurations

### **Performance**
- **Large files excluded**: Prevents repository bloat
- **Cache files ignored**: Avoids unnecessary commits
- **Build artifacts ignored**: Keeps history clean

### **Collaboration**
- **Platform independence**: Ignores OS-specific files
- **IDE independence**: Ignores editor-specific files
- **Environment independence**: Ignores local Python environments

## 🎯 **Customization**

### **Add Project-Specific Exclusions**
```bash
# Add to .gitignore
echo "custom_data_folder/" >> .gitignore
echo "*.custom_extension" >> .gitignore
```

### **Include Exceptions**
```bash
# Include specific files that match ignore patterns
echo "!important_model.pth" >> .gitignore
echo "!examples/*.jpg" >> .gitignore
```

---

**Status**: ✅ **Complete .gitignore** configured for AI Recognition System with proper exclusions and security considerations.