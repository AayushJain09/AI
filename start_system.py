#!/usr/bin/env python3
"""
AI Recognition System Startup Script
Launches both backend and frontend components
"""

import os
import sys
import time
import subprocess
import signal
import threading
from pathlib import Path
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemLauncher:
    """Manages startup and shutdown of the AI Recognition System"""
    
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = False
        
    def check_dependencies(self):
        """Check if required files exist"""
        required_files = [
            "config.yaml",
            "backend/main.py", 
            "frontend/main.py",
            "data/models/faiss_index_corrected.bin",
            "data/models/index_metadata_corrected.pkl",
            "checkpoints/best_model.pth"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error("Missing required files:")
            for file_path in missing_files:
                logger.error(f"  - {file_path}")
            return False
        
        logger.info("✅ All required files found")
        return True
    
    def start_backend(self):
        """Start the backend server"""
        logger.info("🚀 Starting backend server...")
        
        try:
            # Start backend in the background
            self.backend_process = subprocess.Popen([
                sys.executable, "backend/main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait for backend to start
            logger.info("⏳ Waiting for backend to start...")
            for attempt in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Backend server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
                
                # Check if process is still running
                if self.backend_process.poll() is not None:
                    stdout, stderr = self.backend_process.communicate()
                    logger.error(f"❌ Backend process terminated unexpectedly")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    return False
            
            logger.error("❌ Backend failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend application"""
        logger.info("🖥️  Starting frontend application...")
        
        try:
            # Start frontend
            self.frontend_process = subprocess.Popen([
                sys.executable, "frontend/main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            logger.info("✅ Frontend application started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor backend and frontend processes"""
        logger.info("👀 Monitoring system processes...")
        
        while self.running:
            # Check backend
            if self.backend_process and self.backend_process.poll() is not None:
                logger.error("❌ Backend process terminated unexpectedly")
                break
            
            # Check frontend  
            if self.frontend_process and self.frontend_process.poll() is not None:
                logger.info("🛑 Frontend application closed")
                break
            
            time.sleep(2)
        
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all processes"""
        logger.info("🛑 Shutting down AI Recognition System...")
        self.running = False
        
        # Terminate frontend
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                logger.info("✅ Frontend terminated")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Force killing frontend")
                self.frontend_process.kill()
            except Exception as e:
                logger.error(f"Error terminating frontend: {e}")
        
        # Terminate backend
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                logger.info("✅ Backend terminated")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Force killing backend")
                self.backend_process.kill()
            except Exception as e:
                logger.error(f"Error terminating backend: {e}")
        
        logger.info("🎉 System shutdown complete")
    
    def start_system(self):
        """Start the complete AI Recognition System"""
        logger.info("🚀 Starting AI Recognition System...")
        logger.info("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ System startup failed - missing dependencies")
            return False
        
        # Start backend
        if not self.start_backend():
            logger.error("❌ System startup failed - backend error")
            return False
        
        # Start frontend
        if not self.start_frontend():
            logger.error("❌ System startup failed - frontend error")
            self.shutdown()
            return False
        
        self.running = True
        
        logger.info("🎉 AI Recognition System started successfully!")
        logger.info("=" * 50)
        logger.info("📊 Backend API: http://127.0.0.1:8000")
        logger.info("🖥️  Frontend: Desktop application window")
        logger.info("📋 Status: All systems operational")
        logger.info("=" * 50)
        logger.info("Press Ctrl+C to shutdown the system")
        
        # Start monitoring
        monitor_thread = threading.Thread(target=self.monitor_processes)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("🛑 Shutdown signal received")
    if 'launcher' in globals():
        launcher.shutdown()
    sys.exit(0)

def main():
    """Main entry point"""
    global launcher
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start system
    launcher = SystemLauncher()
    
    try:
        if launcher.start_system():
            # Keep main thread alive
            while launcher.running:
                time.sleep(1)
        else:
            logger.error("❌ Failed to start system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        launcher.shutdown()

if __name__ == "__main__":
    main()