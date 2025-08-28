"""
Simple GPU Detection Test Script
Run this to diagnose PyTorch CUDA issues
"""

print("=== GPU Detection Test ===")

# Test 1: Check if torch is available
try:
    import torch
    print(f"✅ PyTorch imported successfully: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    exit(1)

# Test 2: Check CUDA compilation
try:
    cuda_version = torch.version.cuda
    print(f"🔧 PyTorch compiled with CUDA: {cuda_version}")
    if cuda_version is None:
        print("❌ PyTorch was compiled WITHOUT CUDA support!")
        print("💡 You need to install PyTorch with CUDA support")
        print("💡 Visit: https://pytorch.org/get-started/locally/")
        exit(1)
except Exception as e:
    print(f"❌ Error checking CUDA compilation: {e}")

# Test 3: Check CUDA availability
try:
    cuda_available = torch.cuda.is_available()
    print(f"🎯 CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available. Possible reasons:")
        print("   1. NVIDIA GPU drivers not installed")
        print("   2. CUDA toolkit not installed")
        print("   3. PyTorch CUDA version mismatch with installed CUDA")
        print("   4. No compatible NVIDIA GPU found")
    else:
        # Test 4: Get GPU details
        gpu_count = torch.cuda.device_count()
        print(f"🖥️  GPU count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            print(f"🎮 GPU {i}: {gpu_name} ({gpu_memory_gb:.1f}GB)")
        
        # Test 5: Try creating a tensor on GPU
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✅ Successfully created tensor on GPU")
            print(f"📊 Tensor device: {test_tensor.device}")
            
            # Test 6: Try a simple GPU operation
            result = test_tensor @ test_tensor.T
            print("✅ GPU matrix multiplication successful")
            
            del test_tensor, result
            torch.cuda.empty_cache()
            print("✅ GPU memory cleaned up")
            
        except Exception as e:
            print(f"❌ GPU tensor operation failed: {e}")
            
except Exception as e:
    print(f"❌ Error checking CUDA availability: {e}")

# Test 7: Check environment variables
import os
cuda_path = os.environ.get('CUDA_PATH')
cuda_home = os.environ.get('CUDA_HOME')
print(f"🌍 CUDA_PATH: {cuda_path}")
print(f"🏠 CUDA_HOME: {cuda_home}")

print("\n=== Recommendation ===")
if torch.cuda.is_available():
    print("🎉 Your GPU setup looks good!")
    print("🐛 The issue might be in the application code")
else:
    print("🔧 You need to fix your PyTorch CUDA installation:")
    print("   1. Check NVIDIA GPU drivers: nvidia-smi")
    print("   2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
    print("   3. Reinstall PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("   4. Or use conda: conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")