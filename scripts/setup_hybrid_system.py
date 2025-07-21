"""
Hybrid System Setup Script

This script helps set up the hybrid recognition system by:
1. Validating the current system configuration
2. Running benchmarks to identify performance baseline
3. Training the lightweight refiner if needed
4. Enabling hybrid mode
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridSystemSetup:
    """Setup and validation for hybrid recognition system"""
    
    def __init__(self):
        self.project_root = project_root
        self.config_path = self.project_root / "config.yaml"
        
    def validate_prerequisites(self) -> bool:
        """Validate system prerequisites"""
        logger.info("🔍 Validating system prerequisites...")
        
        # Check config file
        if not self.config_path.exists():
            logger.error("❌ config.yaml not found")
            return False
        
        # Check data directory
        data_dir = self.project_root / "data" / "raw"
        if not data_dir.exists():
            logger.error("❌ data/raw directory not found")
            return False
        
        # Count items
        item_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if len(item_dirs) < 2:
            logger.error("❌ Need at least 2 items in data/raw for training")
            return False
        
        # Check if features exist
        features_file = self.project_root / "data" / "features.h5"
        if not features_file.exists():
            logger.warning("⚠️  Features file not found - run feature extraction first")
        
        # Check if raw index exists
        index_file = self.project_root / "data" / "models" / "faiss_index.bin"
        if not index_file.exists():
            logger.warning("⚠️  FAISS index not found - run index building first")
        
        logger.info(f"✅ Found {len(item_dirs)} items in data directory")
        return True
    
    def run_benchmark(self) -> bool:
        """Run benchmark to establish baseline and identify hard cases"""
        logger.info("🧪 Running benchmark to establish baseline...")
        
        benchmark_script = self.project_root / "scripts" / "benchmark_hybrid_system.py"
        
        try:
            result = subprocess.run([
                sys.executable, str(benchmark_script)
            ], capture_output=True, text=True, cwd=str(self.project_root))
            
            if result.returncode == 0:
                logger.info("✅ Benchmark completed successfully")
                logger.info("📊 Check benchmark_results/ for detailed analysis")
                return True
            else:
                logger.error(f"❌ Benchmark failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to run benchmark: {e}")
            return False
    
    def train_lightweight_model(self) -> bool:
        """Train lightweight refiner on hard cases"""
        logger.info("🏋️  Training lightweight refiner model...")
        
        # Check if hard cases exist
        hard_cases_file = self.project_root / "benchmark_results" / "hard_cases_for_training.json"
        if not hard_cases_file.exists():
            logger.error("❌ Hard cases file not found - run benchmark first")
            return False
        
        training_script = self.project_root / "scripts" / "train_lightweight_refiner.py"
        
        try:
            result = subprocess.run([
                sys.executable, str(training_script)
            ], capture_output=True, text=True, cwd=str(self.project_root))
            
            if result.returncode == 0:
                logger.info("✅ Lightweight model training completed")
                return True
            else:
                logger.error(f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to run training: {e}")
            return False
    
    def enable_hybrid_mode(self) -> bool:
        """Enable hybrid mode in configuration"""
        logger.info("🔄 Enabling hybrid mode in configuration...")
        
        try:
            # Load current config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if hybrid configuration exists
            if 'recognition' not in config:
                config['recognition'] = {}
            
            # Enable hybrid mode
            config['recognition']['hybrid_mode'] = True
            
            # Verify paths exist
            lightweight_model_path = Path(config['recognition'].get('lightweight_model_path', 
                                                                  'checkpoints/lightweight_refiner.pth'))
            learned_index_path = Path(config['recognition'].get('learned_index_path',
                                                               'data/models/learned_faiss_index.bin'))
            
            if not lightweight_model_path.exists():
                logger.error(f"❌ Lightweight model not found: {lightweight_model_path}")
                return False
            
            if not learned_index_path.exists():
                logger.error(f"❌ Learned index not found: {learned_index_path}")
                return False
            
            # Save updated config
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("✅ Hybrid mode enabled successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enable hybrid mode: {e}")
            return False
    
    def test_hybrid_system(self) -> bool:
        """Test the hybrid system with a quick recognition"""
        logger.info("🧪 Testing hybrid system...")
        
        try:
            # Import and test recognition pipeline
            from src.inference.recognize import RecognitionPipeline
            
            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Initialize pipeline
            pipeline = RecognitionPipeline(config.get('recognition', {}))
            
            # Find a test image
            data_dir = self.project_root / "data" / "raw"
            test_image = None
            
            for item_dir in data_dir.iterdir():
                if item_dir.is_dir():
                    for img_file in item_dir.glob("*.jpg"):
                        test_image = str(img_file)
                        break
                    if test_image:
                        break
            
            if not test_image:
                logger.error("❌ No test image found")
                return False
            
            # Run recognition
            result = pipeline.recognize(test_image)
            
            logger.info(f"✅ Test recognition completed:")
            logger.info(f"  Predicted: {result.item_id}")
            logger.info(f"  Confidence: {result.confidence:.3f}")
            logger.info(f"  Hybrid mode: {pipeline.hybrid_mode}")
            
            # Get hybrid stats
            if hasattr(pipeline, 'get_hybrid_stats'):
                stats = pipeline.get_hybrid_stats()
                logger.info(f"  Hybrid stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Hybrid system test failed: {e}")
            return False
    
    def run_complete_setup(self):
        """Run complete hybrid system setup"""
        print("🚀 Starting Hybrid Recognition System Setup")
        print("="*50)
        
        steps = [
            ("Validate Prerequisites", self.validate_prerequisites),
            ("Run Baseline Benchmark", self.run_benchmark),
            ("Train Lightweight Model", self.train_lightweight_model),
            ("Enable Hybrid Mode", self.enable_hybrid_mode),
            ("Test Hybrid System", self.test_hybrid_system)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 Step: {step_name}")
            success = step_func()
            
            if success:
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
                print("🛑 Setup aborted - please fix the issue and try again")
                return False
        
        print("\n" + "="*50)
        print("🎉 HYBRID SYSTEM SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\n📋 Next Steps:")
        print("1. Run additional benchmarks to validate performance")
        print("2. Monitor hybrid system usage in production")
        print("3. Retrain lightweight model with more data as needed")
        print("\n🔧 Configuration:")
        print(f"  Config file: {self.config_path}")
        print(f"  Hybrid mode: ✅ Enabled")
        print(f"  Lightweight model: checkpoints/lightweight_refiner.pth")
        print(f"  Learned index: data/models/learned_faiss_index.bin")
        
        return True


def main():
    """Main setup script"""
    setup = HybridSystemSetup()
    
    # Check if user wants to run complete setup
    print("🔧 Hybrid Recognition System Setup")
    print("\nThis script will:")
    print("1. Validate your current system")
    print("2. Run benchmarks to identify hard cases")
    print("3. Train a lightweight refinement model")
    print("4. Enable hybrid mode")
    print("5. Test the complete system")
    
    response = input("\nDo you want to proceed with complete setup? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = setup.run_complete_setup()
        return 0 if success else 1
    else:
        print("Setup cancelled.")
        return 0


if __name__ == "__main__":
    sys.exit(main())