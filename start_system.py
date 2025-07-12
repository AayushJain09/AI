#!/usr/bin/env python3
"""
AI Recognition System - Startup Script
Launches both backend API and frontend GUI application
"""

import sys
import os
import time
import subprocess
import signal
import logging
from pathlib import Path
from typing import Optional, List
import threading
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemLauncher:
    """System launcher for backend and frontend"""
    
    def __init__(self):
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.backend_port = 8000
        self.backend_url = f"http://127.0.0.1:{self.backend_port}"
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_requested = True
        self.shutdown()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("Checking system dependencies...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'requests', 'PyQt6', 'torch', 
            'torchvision', 'opencv-python', 'pillow', 'numpy',
            'faiss-cpu', 'clip-by-openai', 'albumentations', 'h5py'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'clip-by-openai':
                    import clip
                elif package == 'opencv-python':
                    import cv2
                elif package == 'PyQt6':
                    from PyQt6.QtWidgets import QApplication
                elif package == 'faiss-cpu':
                    import faiss
                else:
                    __import__(package.replace('-', '_'))
                
                logger.debug(f"✓ {package} is available")
            
            except ImportError:
                missing_packages.append(package)
                logger.error(f"✗ {package} is missing")
        
        if missing_packages:
            logger.error(f"Missing dependencies: {', '.join(missing_packages)}")
            logger.error("Please install missing packages using:")
            logger.error(f"pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("✓ All dependencies are available")
        return True
    
    def check_configuration(self) -> bool:
        """Check if configuration files exist"""
        logger.info("Checking configuration files...")
        
        required_files = [
            'config.yaml',
            'backend/main.py',
            'frontend/main.py'
        ]
        
        missing_files = []
        
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                logger.error(f"✗ {file_path} not found")
            else:
                logger.debug(f"✓ {file_path} exists")
        
        if missing_files:
            logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        logger.info("✓ All configuration files are available")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'logs',
            'backend/logs',
            'backend/temp',
            'data/raw',
            'data/augmented',
            'data/models',
            'checkpoints'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def wait_for_backend(self, timeout: int = 30) -> bool:
        """Wait for backend to become available"""
        logger.info(f"Waiting for backend to start on {self.backend_url}...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.backend_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        logger.info("✓ Backend is healthy and ready")
                        return True
            
            except requests.exceptions.RequestException:
                pass
            
            if self.shutdown_requested:
                return False
            
            logger.debug("Backend not ready yet, waiting...")
            time.sleep(1)
        
        logger.error(f"✗ Backend did not start within {timeout} seconds")
        return False
    
    def start_backend(self) -> bool:
        """Start the backend API server"""
        logger.info("Starting backend API server...")
        
        try:
            # Change to project directory
            project_dir = Path(__file__).parent
            
            # Start backend process
            cmd = [
                sys.executable, "-m", "uvicorn",
                "backend.main:app",
                "--host", "127.0.0.1",
                "--port", str(self.backend_port),
                "--reload",
                "--log-level", "info"
            ]
            
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start log monitoring thread
            log_thread = threading.Thread(
                target=self.monitor_backend_logs,
                daemon=True
            )
            log_thread.start()
            
            # Wait for backend to be ready
            if self.wait_for_backend():
                logger.info("✓ Backend started successfully")
                return True
            else:
                logger.error("✗ Backend failed to start")
                self.stop_backend()
                return False
        
        except Exception as e:
            logger.error(f"✗ Failed to start backend: {e}")
            return False
    
    def monitor_backend_logs(self):
        """Monitor backend process logs"""
        if not self.backend_process:
            return
        
        try:
            for line in iter(self.backend_process.stdout.readline, ''):
                if self.shutdown_requested:
                    break
                
                line = line.strip()
                if line:
                    logger.info(f"[Backend] {line}")
        
        except Exception as e:
            logger.error(f"Error monitoring backend logs: {e}")
    
    def start_frontend(self) -> bool:
        """Start the frontend GUI application"""
        logger.info("Starting frontend GUI application...")
        
        try:
            # Change to project directory
            project_dir = Path(__file__).parent
            
            # Start frontend process
            cmd = [sys.executable, "frontend/main.py"]
            
            self.frontend_process = subprocess.Popen(
                cmd,
                cwd=project_dir
            )
            
            logger.info("✓ Frontend started successfully")
            return True
        
        except Exception as e:
            logger.error(f"✗ Failed to start frontend: {e}")
            return False
    
    def stop_backend(self):
        """Stop the backend process"""
        if self.backend_process:
            logger.info("Stopping backend...")
            self.backend_process.terminate()
            
            try:
                self.backend_process.wait(timeout=10)
                logger.info("✓ Backend stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Backend did not stop gracefully, forcing...")
                self.backend_process.kill()
                self.backend_process.wait()
            
            self.backend_process = None
    
    def stop_frontend(self):
        """Stop the frontend process"""
        if self.frontend_process:
            logger.info("Stopping frontend...")
            self.frontend_process.terminate()
            
            try:
                self.frontend_process.wait(timeout=10)
                logger.info("✓ Frontend stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Frontend did not stop gracefully, forcing...")
                self.frontend_process.kill()
                self.frontend_process.wait()
            
            self.frontend_process = None
    
    def shutdown(self):
        """Shutdown the entire system"""
        logger.info("Shutting down AI Recognition System...")
        
        self.stop_frontend()
        self.stop_backend()
        
        logger.info("✓ System shutdown complete")
    
    def run(self):
        """Main execution method"""
        logger.info("🚀 AI Recognition System - Starting up...")
        
        # Pre-flight checks
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed")
            return False
        
        if not self.check_configuration():
            logger.error("❌ Configuration check failed")
            return False
        
        # Create directories
        self.create_directories()
        
        # Start backend
        if not self.start_backend():
            logger.error("❌ Failed to start backend")
            return False
        
        # Start frontend
        if not self.start_frontend():
            logger.error("❌ Failed to start frontend")
            self.stop_backend()
            return False
        
        logger.info("✅ AI Recognition System started successfully!")
        logger.info(f"Backend API: {self.backend_url}")
        logger.info("Frontend GUI: Running")
        
        try:
            # Wait for frontend to exit
            if self.frontend_process:
                self.frontend_process.wait()
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
        finally:
            self.shutdown()
        
        return True

def main():
    """Main entry point"""
    print("=" * 60)
    print("AI Recognition System - Inventory Management")
    print("=" * 60)
    
    launcher = SystemLauncher()
    
    try:
        success = launcher.run()
        exit_code = 0 if success else 1
    
    except Exception as e:
        logger.error(f"Critical error: {e}")
        exit_code = 1
    
    finally:
        launcher.shutdown()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()