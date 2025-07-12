"""
Training Dashboard Widget
Interface for managing model training and monitoring progress
"""

import json
import time
from typing import Dict, Any, Optional, List
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QGroupBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QProgressBar, QTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSplitter, QComboBox, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon, QPalette, QColor

logger = logging.getLogger(__name__)

class TrainingMonitorThread(QThread):
    """Background thread to monitor training progress"""
    
    status_updated = pyqtSignal(dict)
    
    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.running = False
    
    def start_monitoring(self):
        """Start monitoring training status"""
        self.running = True
        self.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.wait()
    
    def run(self):
        """Monitor training status"""
        while self.running:
            try:
                response = self.api_client.get("/api/train/status")
                self.status_updated.emit(response)
                self.msleep(2000)  # Check every 2 seconds during training
            except Exception as e:
                logger.error(f"Training status check failed: {e}")
                self.msleep(5000)  # Wait longer on error

class TrainingConfigWidget(QFrame):
    """Widget for configuring training parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup training configuration UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Training Configuration")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(title)
        
        # Configuration form
        form_layout = QFormLayout()
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setToolTip("Number of training epochs")
        form_layout.addRow("Epochs:", self.epochs_spin)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(16)
        self.batch_size_spin.setToolTip("Training batch size")
        form_layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setToolTip("Learning rate for optimization")
        form_layout.addRow("Learning Rate:", self.lr_spin)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout(advanced_group)
        
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.01)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setValue(0.00001)
        self.weight_decay_spin.setSingleStep(0.00001)
        advanced_layout.addRow("Weight Decay:", self.weight_decay_spin)
        
        self.save_every_spin = QSpinBox()
        self.save_every_spin.setRange(1, 20)
        self.save_every_spin.setValue(5)
        advanced_layout.addRow("Save Every:", self.save_every_spin)
        
        self.use_wandb_check = QCheckBox("Enable W&B Logging")
        advanced_layout.addRow("", self.use_wandb_check)
        
        layout.addLayout(form_layout)
        layout.addWidget(advanced_group)
        
        # Style the form elements
        for widget in [self.epochs_spin, self.batch_size_spin, self.lr_spin, 
                      self.weight_decay_spin, self.save_every_spin]:
            widget.setStyleSheet("""
                QSpinBox, QDoubleSpinBox {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                    font-size: 13px;
                }
                QSpinBox:focus, QDoubleSpinBox:focus {
                    border-color: #2196F3;
                }
            """)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current training configuration"""
        return {
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "save_every": self.save_every_spin.value(),
            "use_wandb": self.use_wandb_check.isChecked()
        }

class TrainingProgressWidget(QFrame):
    """Widget to display training progress"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.reset_progress()
    
    def setup_ui(self):
        """Setup training progress UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Training Progress")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(title)
        
        # Status
        self.status_label = QLabel("Not started")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #666;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Overall progress
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Overall Progress:"))
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        progress_layout.addWidget(self.overall_progress)
        
        layout.addLayout(progress_layout)
        
        # Stage-specific progress
        stages_group = QGroupBox("Pipeline Stages")
        stages_layout = QVBoxLayout(stages_group)
        
        # Create progress bars for each stage
        self.stage_progress = {}
        stages = [
            ("data_prep", "Data Preparation"),
            ("feature_extraction", "Feature Extraction"),
            ("training", "Model Training"),
            ("index_building", "Index Building")
        ]
        
        for stage_id, stage_name in stages:
            stage_layout = QHBoxLayout()
            
            stage_label = QLabel(f"{stage_name}:")
            stage_label.setFixedWidth(150)
            stage_layout.addWidget(stage_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    text-align: center;
                    font-size: 11px;
                    height: 20px;
                }
                QProgressBar::chunk {
                    background-color: #2196F3;
                    border-radius: 3px;
                }
            """)
            stage_layout.addWidget(progress_bar)
            
            self.stage_progress[stage_id] = progress_bar
            stages_layout.addLayout(stage_layout)
        
        layout.addWidget(stages_group)
        
        # Training metrics
        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        self.current_epoch_label = QLabel("0 / 0")
        metrics_layout.addRow("Current Epoch:", self.current_epoch_label)
        
        self.accuracy_label = QLabel("N/A")
        metrics_layout.addRow("Validation Accuracy:", self.accuracy_label)
        
        self.loss_label = QLabel("N/A")
        metrics_layout.addRow("Training Loss:", self.loss_label)
        
        self.eta_label = QLabel("N/A")
        metrics_layout.addRow("ETA:", self.eta_label)
        
        layout.addWidget(metrics_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
    
    def reset_progress(self):
        """Reset all progress indicators"""
        self.status_label.setText("Not started")
        self.status_label.setStyleSheet("color: #666;")
        
        self.overall_progress.setValue(0)
        
        for progress_bar in self.stage_progress.values():
            progress_bar.setValue(0)
        
        self.current_epoch_label.setText("0 / 0")
        self.accuracy_label.setText("N/A")
        self.loss_label.setText("N/A")
        self.eta_label.setText("N/A")
        
        self.log_text.clear()
    
    def update_progress(self, status_data: Dict[str, Any]):
        """Update progress based on status data"""
        if not status_data.get("success", False):
            return
        
        data = status_data.get("data", {})
        training_in_progress = data.get("training_in_progress", False)
        
        if training_in_progress:
            self.status_label.setText("🚀 Training in progress...")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Update stage progress based on task status
            tasks = data.get("tasks", {})
            for task_id, task_info in tasks.items():
                if "training" in task_id:
                    progress = task_info.get("progress", 0)
                    status = task_info.get("status", "")
                    
                    self.overall_progress.setValue(progress)
                    
                    # Update stage-specific progress
                    if status == "preparing":
                        self.stage_progress["data_prep"].setValue(100)
                    elif status == "data_prepared":
                        self.stage_progress["data_prep"].setValue(100)
                        self.stage_progress["feature_extraction"].setValue(50)
                    elif status == "features_extracted":
                        self.stage_progress["data_prep"].setValue(100)
                        self.stage_progress["feature_extraction"].setValue(100)
                        self.stage_progress["training"].setValue(25)
                    elif status == "training_complete":
                        self.stage_progress["data_prep"].setValue(100)
                        self.stage_progress["feature_extraction"].setValue(100)
                        self.stage_progress["training"].setValue(100)
                        self.stage_progress["index_building"].setValue(50)
                    elif status == "completed":
                        for progress_bar in self.stage_progress.values():
                            progress_bar.setValue(100)
                    
                    # Add log entry
                    timestamp = time.strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] {status.replace('_', ' ').title()}: {progress}%"
                    self.log_text.append(log_entry)
        else:
            self.status_label.setText("Training not active")
            self.status_label.setStyleSheet("color: #666;")

class TrainingWidget(QWidget):
    """Main training dashboard widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.monitor_thread = TrainingMonitorThread(api_client)
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup training dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("Model Training")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Training controls
        self.start_btn = QPushButton("🚀 Start Training")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.start_btn.clicked.connect(self.start_training)
        header_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop Training")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        header_layout.addWidget(self.stop_btn)
        
        layout.addLayout(header_layout)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Configuration
        self.config_widget = TrainingConfigWidget()
        splitter.addWidget(self.config_widget)
        
        # Right side - Progress
        self.progress_widget = TrainingProgressWidget()
        splitter.addWidget(self.progress_widget)
        
        # Set proportions
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # Training information
        info_group = QGroupBox("Training Information")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel("""
        <b>Training Process:</b><br>
        1. <b>Data Preparation:</b> Augments raw images to create diverse training set<br>
        2. <b>Feature Extraction:</b> Extracts CLIP and DINOv2 features from augmented images<br>
        3. <b>Model Training:</b> Trains Siamese network for optimized embeddings<br>
        4. <b>Index Building:</b> Creates FAISS search index for fast recognition<br><br>
        
        <b>Tips:</b><br>
        • Ensure you have at least 8 images per item before training<br>
        • Higher epochs = better accuracy but longer training time<br>
        • Batch size affects memory usage and training stability<br>
        • Training typically takes 10-30 minutes depending on dataset size
        """)
        info_text.setWordWrap(True)
        info_text.setStyleSheet("""
            QLabel {
                color: #555;
                font-size: 12px;
                background-color: #f0f8ff;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #2196F3;
            }
        """)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.monitor_thread.status_updated.connect(self.update_training_status)
    
    def start_training(self):
        """Start the training process"""
        try:
            # Get training configuration
            config = self.config_widget.get_config()
            
            # Confirm start
            reply = QMessageBox.question(
                self,
                "Start Training",
                f"Start training with the following configuration?\n\n"
                f"Epochs: {config['epochs']}\n"
                f"Batch Size: {config['batch_size']}\n"
                f"Learning Rate: {config['learning_rate']}\n\n"
                f"This process may take 10-30 minutes to complete.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Start training via API
            response = self.api_client.post("/api/train", config)
            
            if response.get("success"):
                QMessageBox.information(
                    self,
                    "Training Started",
                    "Training has been started successfully! Monitor progress below."
                )
                
                # Update UI state
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.progress_widget.reset_progress()
                
                # Start monitoring
                self.monitor_thread.start_monitoring()
                
            else:
                QMessageBox.warning(
                    self,
                    "Training Failed",
                    f"Failed to start training: {response.get('error', 'Unknown error')}"
                )
        
        except Exception as e:
            logger.error(f"Training start error: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while starting training: {str(e)}"
            )
    
    def stop_training(self):
        """Stop the training process"""
        reply = QMessageBox.question(
            self,
            "Stop Training",
            "Are you sure you want to stop training?\n\nProgress will be lost and you'll need to restart from the beginning.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Stop monitoring
            self.monitor_thread.stop_monitoring()
            
            # Update UI state
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            QMessageBox.information(
                self,
                "Training Stopped",
                "Training has been stopped. You can start a new training session anytime."
            )
    
    def update_training_status(self, status_data: Dict[str, Any]):
        """Update training status from monitoring thread"""
        self.progress_widget.update_progress(status_data)
        
        # Check if training completed
        if status_data.get("success", False):
            data = status_data.get("data", {})
            if not data.get("training_in_progress", False):
                # Training finished, stop monitoring
                self.monitor_thread.stop_monitoring()
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                
                # Check if there are any completed tasks
                tasks = data.get("tasks", {})
                for task_id, task_info in tasks.items():
                    if "training" in task_id and task_info.get("status") == "completed":
                        QMessageBox.information(
                            self,
                            "Training Completed",
                            "🎉 Training completed successfully!\n\n"
                            "Your model is now ready for recognition tasks."
                        )
                        break
                    elif "training" in task_id and task_info.get("status") == "failed":
                        error = task_info.get("error", "Unknown error")
                        QMessageBox.warning(
                            self,
                            "Training Failed",
                            f"❌ Training failed with error:\n{error}\n\n"
                            "Please check the logs and try again."
                        )
                        break