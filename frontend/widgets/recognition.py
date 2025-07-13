"""
Recognition Interface Widget
Real-time image recognition with camera and file upload support
"""

import os
import cv2
import base64
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import time
import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QFileDialog,
    QMessageBox, QGroupBox, QTabWidget, QSlider, QSpinBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QComboBox, QCheckBox, QSplitter, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon, QImage, QPainter, QPen

logger = logging.getLogger(__name__)

class CameraThread(QThread):
    """Background thread for camera capture"""
    
    frame_ready = pyqtSignal(QImage)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.running = False
        self.camera_index = 0
    
    def start_camera(self, camera_index: int = 0):
        """Start camera capture"""
        self.camera_index = camera_index
        self.running = True
        self.start()
    
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        self.wait()
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def run(self):
        """Camera capture loop"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                self.error_occurred.emit("Failed to open camera")
                return
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            while self.running:
                ret, frame = self.camera.read()
                
                if ret:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to QImage
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    
                    self.frame_ready.emit(qt_image)
                
                self.msleep(30)  # ~30 FPS
        
        except Exception as e:
            self.error_occurred.emit(f"Camera error: {str(e)}")
        
        finally:
            if self.camera:
                self.camera.release()

class RecognitionResultWidget(QFrame):
    """Widget to display recognition results"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.clear_result()
    
    def setup_ui(self):
        """Setup result display UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 15px;
                color: #212529;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Recognition Result")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #212529;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Result info
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        
        # Confidence bar
        confidence_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_label.setStyleSheet("color: #495057; font-weight: 500;")
        confidence_layout.addWidget(conf_label)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        confidence_layout.addWidget(self.confidence_bar)
        
        layout.addLayout(confidence_layout)
        
        # Processing time
        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: #6c757d; font-size: 12px; font-weight: 400;")
        layout.addWidget(self.time_label)
        
        # Top matches
        self.matches_label = QLabel("Top Matches:")
        self.matches_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #212529;")
        layout.addWidget(self.matches_label)
        
        self.matches_list = QTextEdit()
        self.matches_list.setMaximumHeight(100)
        self.matches_list.setReadOnly(True)
        self.matches_list.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: #f8f9fa;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                color: #495057;
                padding: 8px;
            }
        """)
        layout.addWidget(self.matches_list)
    
    def update_result(self, result_data: Dict[str, Any]):
        """Update the recognition result display"""
        item_id = result_data.get("item_id", "unknown")
        confidence = result_data.get("confidence", 0.0)
        processing_time = result_data.get("processing_time", 0.0)
        top_matches = result_data.get("top_matches", [])
        
        # Update main result with better visibility
        if item_id == "unknown":
            self.result_label.setText("❌ No match found")
            self.result_label.setStyleSheet("color: #dc3545; font-size: 14px; font-weight: bold;")
        else:
            self.result_label.setText(f"✅ Recognized: {item_id}")
            self.result_label.setStyleSheet("color: #28a745; font-size: 14px; font-weight: bold;")
        
        # Update confidence
        confidence_percent = int(confidence * 100)
        self.confidence_bar.setValue(confidence_percent)
        
        # Color code confidence bar with better contrast
        if confidence_percent >= 85:
            color = "#28a745"  # Bootstrap success green
        elif confidence_percent >= 70:
            color = "#fd7e14"  # Bootstrap warning orange
        else:
            color = "#dc3545"  # Bootstrap danger red
        
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ced4da;
                border-radius: 4px;
                text-align: center;
                font-weight: bold;
                color: #212529;
                background-color: #e9ecef;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        
        # Update processing time
        self.time_label.setText(f"Processing time: {processing_time:.3f}s")
        
        # Update top matches
        if top_matches:
            matches_text = ""
            for i, (match_id, score) in enumerate(top_matches[:5], 1):
                matches_text += f"{i}. {match_id}: {score:.3f}\n"
            self.matches_list.setText(matches_text.strip())
        else:
            self.matches_list.setText("No matches available")
    
    def clear_result(self):
        """Clear the result display"""
        self.result_label.setText("No recognition performed yet")
        self.result_label.setStyleSheet("color: #6c757d; font-size: 14px; font-style: italic;")
        self.confidence_bar.setValue(0)
        self.time_label.setText("")
        self.matches_list.setText("")

class CameraWidget(QWidget):
    """Camera capture widget"""
    
    recognition_requested = pyqtSignal(QImage)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_thread = CameraThread()
        self.current_frame = None
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup camera widget UI"""
        layout = QVBoxLayout(self)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 16px;
                color: #495057;
                font-weight: 500;
            }
        """)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("📷 Camera Preview\nClick 'Start Camera' to begin")
        layout.addWidget(self.camera_label)
        
        # Camera controls
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("📷 Start Camera")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_btn.clicked.connect(self.start_camera)
        
        self.stop_btn = QPushButton("⏹️ Stop Camera")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        
        self.capture_btn = QPushButton("📸 Capture & Recognize")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.capture_btn.clicked.connect(self.capture_and_recognize)
        self.capture_btn.setEnabled(False)
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.capture_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.error_occurred.connect(self.handle_camera_error)
    
    def start_camera(self):
        """Start camera capture"""
        self.camera_thread.start_camera()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_thread.stop_camera()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        
        # Reset camera display
        self.camera_label.setText("📷 Camera Preview\nClick 'Start Camera' to begin")
        self.current_frame = None
    
    def update_frame(self, qt_image: QImage):
        """Update camera frame display"""
        self.current_frame = qt_image
        
        # Scale image to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
    
    def capture_and_recognize(self):
        """Capture current frame and request recognition"""
        if self.current_frame:
            self.recognition_requested.emit(self.current_frame)
    
    def handle_camera_error(self, error_msg: str):
        """Handle camera errors"""
        QMessageBox.critical(self, "Camera Error", error_msg)
        self.stop_camera()

class FileUploadWidget(QWidget):
    """File upload widget for image recognition"""
    
    recognition_requested = pyqtSignal(str)  # file path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup file upload UI"""
        layout = QVBoxLayout(self)
        
        # Drop area
        self.drop_area = QLabel()
        self.drop_area.setMinimumSize(640, 400)
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 3px dashed #ced4da;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 16px;
                color: #495057;
                font-weight: 500;
            }
        """)
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setText("🖼️ Drag & Drop Image Here\nor\nClick 'Browse Files' to select")
        layout.addWidget(self.drop_area)
        
        # File controls
        controls_layout = QHBoxLayout()
        
        browse_btn = QPushButton("📁 Browse Files")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        browse_btn.clicked.connect(self.browse_file)
        
        self.recognize_btn = QPushButton("🔍 Recognize Image")
        self.recognize_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.recognize_btn.clicked.connect(self.recognize_current_image)
        self.recognize_btn.setEnabled(False)
        
        controls_layout.addWidget(browse_btn)
        controls_layout.addWidget(self.recognize_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        self.current_file_path = None
    
    def browse_file(self):
        """Open file browser to select image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path: str):
        """Load and display selected image"""
        try:
            self.current_file_path = file_path
            
            # Load and display image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.drop_area.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.drop_area.setPixmap(scaled_pixmap)
                self.recognize_btn.setEnabled(True)
                
                # Update border style for loaded image
                self.drop_area.setStyleSheet("""
                    QLabel {
                        border: 3px solid #28a745;
                        border-radius: 8px;
                        background-color: #f8f9fa;
                    }
                """)
            else:
                QMessageBox.warning(self, "Invalid Image", "Could not load the selected image file.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def recognize_current_image(self):
        """Request recognition for current image"""
        if self.current_file_path:
            self.recognition_requested.emit(self.current_file_path)

class RecognitionWidget(QWidget):
    """Main recognition interface widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup recognition interface UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        title = QLabel("Image Recognition")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #212529;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Input methods
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        
        # Tab widget for input methods
        self.input_tabs = QTabWidget()
        self.input_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 10px 20px;
                margin: 2px;
                border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196F3;
            }
        """)
        
        # Camera tab
        self.camera_widget = CameraWidget()
        self.input_tabs.addTab(self.camera_widget, "📷 Camera")
        
        # File upload tab
        self.file_widget = FileUploadWidget()
        self.input_tabs.addTab(self.file_widget, "📁 File Upload")
        
        input_layout.addWidget(self.input_tabs)
        
        # Recognition settings
        settings_group = QGroupBox("Recognition Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Confidence Threshold:")
        threshold_label.setStyleSheet("color: #495057; font-weight: 500;")
        threshold_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 99)
        self.threshold_slider.setValue(85)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        
        self.threshold_label = QLabel("85%")
        self.threshold_label.setStyleSheet("color: #212529; font-weight: 600; min-width: 40px;")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v}%")
        )
        
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        settings_layout.addLayout(threshold_layout)
        
        input_layout.addWidget(settings_group)
        
        # Right side - Results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        # Result display
        self.result_widget = RecognitionResultWidget()
        results_layout.addWidget(self.result_widget)
        
        # Recognition history
        history_group = QGroupBox("Recognition History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Time", "Result", "Confidence", "Processing Time"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setMaximumHeight(200)
        self.history_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
                gridline-color: #dee2e6;
                color: #212529;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #dee2e6;
            }
            QTableWidget::item:selected {
                background-color: #007bff;
                color: white;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 10px 8px;
                border: none;
                border-bottom: 2px solid #dee2e6;
                font-weight: 600;
                color: #495057;
            }
        """)
        
        history_layout.addWidget(self.history_table)
        results_layout.addWidget(history_group)
        
        # Add widgets to splitter
        splitter.addWidget(input_widget)
        splitter.addWidget(results_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.camera_widget.recognition_requested.connect(self.recognize_image_from_qt)
        self.file_widget.recognition_requested.connect(self.recognize_image_from_file)
    
    def recognize_image_from_qt(self, qt_image: QImage):
        """Recognize image from Qt image object"""
        try:
            # Convert QImage to base64
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            qt_image.save(temp_file.name, 'JPEG')
            
            with open(temp_file.name, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            os.unlink(temp_file.name)
            
            self.perform_recognition(image_data)
        
        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to process image: {str(e)}")
    
    def recognize_image_from_file(self, file_path: str):
        """Recognize image from file path"""
        try:
            with open(file_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            self.perform_recognition(image_data)
        
        except Exception as e:
            logger.error(f"File recognition error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to read image file: {str(e)}")
    
    def perform_recognition(self, image_data: str):
        """Perform recognition request to backend"""
        try:
            # Prepare request
            request_data = {
                "image_data": image_data,
                "confidence_threshold": self.threshold_slider.value() / 100.0
            }
            
            # Show progress
            self.result_widget.clear_result()
            self.result_widget.result_label.setText("🔄 Processing...")
            self.result_widget.result_label.setStyleSheet("color: #fd7e14; font-size: 14px; font-weight: bold;")
            
            # Make API request
            start_time = time.time()
            response = self.api_client.post("/api/recognize", request_data)
            processing_time = time.time() - start_time
            
            if response.get("success"):
                result_data = response.get("data", {})
                result_data["processing_time"] = processing_time
                
                # Update result display
                self.result_widget.update_result(result_data)
                
                # Add to history
                self.add_to_history(result_data)
            else:
                error_msg = response.get("error", "Unknown error")
                self.result_widget.result_label.setText(f"❌ Recognition failed: {error_msg}")
                self.result_widget.result_label.setStyleSheet("color: #dc3545; font-size: 14px; font-weight: bold;")
        
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            self.result_widget.result_label.setText(f"❌ Error: {str(e)}")
            self.result_widget.result_label.setStyleSheet("color: #dc3545; font-size: 14px; font-weight: bold;")
    
    def add_to_history(self, result_data: Dict[str, Any]):
        """Add recognition result to history table"""
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        
        # Time
        timestamp = time.strftime("%H:%M:%S")
        self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
        
        # Result
        item_id = result_data.get("item_id", "unknown")
        self.history_table.setItem(row, 1, QTableWidgetItem(item_id))
        
        # Confidence
        confidence = result_data.get("confidence", 0.0)
        confidence_item = QTableWidgetItem(f"{confidence:.1%}")
        self.history_table.setItem(row, 2, confidence_item)
        
        # Processing time
        proc_time = result_data.get("processing_time", 0.0)
        time_item = QTableWidgetItem(f"{proc_time:.3f}s")
        self.history_table.setItem(row, 3, time_item)
        
        # Color code by confidence with better contrast
        if confidence >= 0.85:
            color = "#d4edda"  # Bootstrap success background
        elif confidence >= 0.70:
            color = "#fff3cd"  # Bootstrap warning background
        else:
            color = "#f8d7da"  # Bootstrap danger background
        
        for col in range(4):
            item = self.history_table.item(row, col)
            if item:
                item.setBackground(QColor(color))
        
        # Scroll to bottom
        self.history_table.scrollToBottom()
        
        # Limit history to 50 entries
        while self.history_table.rowCount() > 50:
            self.history_table.removeRow(0)