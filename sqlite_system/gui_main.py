"""
Modern PySide6 GUI for SQLite Recognition System
Attractive, intuitive interface with dark theme and modern design
"""

import sys
import os
from pathlib import Path
import json
import time
import logging
from typing import Optional, Dict, List
import threading
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QTextEdit, QProgressBar,
    QFileDialog, QGroupBox, QGridLayout, QSpacerItem, QSizePolicy,
    QScrollArea, QFrame, QSplitter, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
    QMessageBox, QDialog, QDialogButtonBox, QFormLayout
)
from PySide6.QtCore import (
    Qt, QThread, QObject, Signal, QTimer, QPropertyAnimation,
    QEasingCurve, QRect, QSize
)
from PySide6.QtGui import (
    QPixmap, QIcon, QFont, QColor, QPalette, QBrush, QLinearGradient,
    QAction, QKeySequence, QPainter, QPainterPath
)

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import SQLiteRecognitionSystem

# Modern Dark Theme Stylesheet
DARK_THEME = """
/* Modern Dark Theme */
QMainWindow {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #1e1e2e, stop: 1 #181825);
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 4px;
    background: #1e1e2e;
}

QTabBar::tab {
    background: #313244;
    color: #cdd6f4;
    padding: 12px 24px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #89b4fa, stop: 1 #74c7ec);
    color: #1e1e2e;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background: #45475a;
}

/* Buttons */
QPushButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #89b4fa, stop: 1 #74c7ec);
    color: #1e1e2e;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 14px;
}

QPushButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #94bffe, stop: 1 #7dd3fc);
}

QPushButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #7c9ffa, stop: 1 #6cbaea);
}

QPushButton:disabled {
    background: #45475a;
    color: #6c7086;
}

/* Secondary Button */
QPushButton[class="secondary"] {
    background: #313244;
    color: #cdd6f4;
    border: 2px solid #45475a;
}

QPushButton[class="secondary"]:hover {
    background: #45475a;
    border-color: #585b70;
}

/* Danger Button */
QPushButton[class="danger"] {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #f38ba8, stop: 1 #eba0ac);
}

QPushButton[class="danger"]:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #f599b9, stop: 1 #f0a6bd);
}

/* Group Boxes */
QGroupBox {
    background: rgba(49, 50, 68, 0.6);
    border: 2px solid #45475a;
    border-radius: 12px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
    color: #cdd6f4;
    font-size: 14px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 4px 12px;
    background: #89b4fa;
    color: #1e1e2e;
    border-radius: 6px;
    font-weight: 600;
}

/* Text Elements */
QLabel {
    color: #cdd6f4;
    font-size: 14px;
}

QTextEdit, QLineEdit {
    background: #313244;
    border: 2px solid #45475a;
    border-radius: 8px;
    padding: 8px;
    color: #cdd6f4;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 13px;
}

QTextEdit:focus, QLineEdit:focus {
    border-color: #89b4fa;
}

/* Progress Bar */
QProgressBar {
    background: #313244;
    border: 2px solid #45475a;
    border-radius: 8px;
    text-align: center;
    color: #cdd6f4;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #89b4fa, stop: 1 #74c7ec);
    border-radius: 6px;
}

/* Table Widget */
QTableWidget {
    background: #1e1e2e;
    border: 2px solid #45475a;
    border-radius: 8px;
    gridline-color: #45475a;
    color: #cdd6f4;
}

QTableWidget::item {
    padding: 8px;
    border: none;
}

QTableWidget::item:selected {
    background: #89b4fa;
    color: #1e1e2e;
}

QHeaderView::section {
    background: #313244;
    color: #cdd6f4;
    padding: 12px;
    border: none;
    border-right: 1px solid #45475a;
    font-weight: 600;
}

/* Tree Widget */
QTreeWidget {
    background: #1e1e2e;
    border: 2px solid #45475a;
    border-radius: 8px;
    color: #cdd6f4;
}

QTreeWidget::item {
    padding: 6px;
}

QTreeWidget::item:selected {
    background: #89b4fa;
    color: #1e1e2e;
}

/* Combo Box */
QComboBox {
    background: #313244;
    border: 2px solid #45475a;
    border-radius: 8px;
    padding: 8px 12px;
    color: #cdd6f4;
}

QComboBox:hover {
    border-color: #585b70;
}

QComboBox::drop-down {
    border: none;
    background: transparent;
}

QComboBox::down-arrow {
    image: url(none);
    border: none;
}

QComboBox QAbstractItemView {
    background: #313244;
    border: 2px solid #45475a;
    border-radius: 8px;
    color: #cdd6f4;
}

/* Spin Box */
QSpinBox, QDoubleSpinBox {
    background: #313244;
    border: 2px solid #45475a;
    border-radius: 8px;
    padding: 8px;
    color: #cdd6f4;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #89b4fa;
}

/* Check Box */
QCheckBox {
    color: #cdd6f4;
    font-size: 14px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #45475a;
    border-radius: 4px;
    background: #313244;
}

QCheckBox::indicator:checked {
    background: #89b4fa;
    border-color: #89b4fa;
}

/* Slider */
QSlider::groove:horizontal {
    height: 6px;
    background: #313244;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #89b4fa;
    border: none;
    width: 18px;
    height: 18px;
    border-radius: 9px;
    margin: -6px 0;
}

QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 3px;
}

/* Scroll Area */
QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #313244;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #585b70;
}

/* Splitter */
QSplitter::handle {
    background: #45475a;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* Frame */
QFrame {
    border: none;
}

/* Status Cards */
QLabel[class="status-card"] {
    background: rgba(137, 180, 250, 0.1);
    border: 2px solid #89b4fa;
    border-radius: 12px;
    padding: 16px;
    font-size: 16px;
    font-weight: 600;
}

QLabel[class="metric-value"] {
    color: #89b4fa;
    font-size: 28px;
    font-weight: 700;
}

QLabel[class="metric-label"] {
    color: #cdd6f4;
    font-size: 12px;
    font-weight: 500;
}
"""


class ImageRecognitionWorker(QObject):
    """Worker thread for image recognition tasks"""
    
    finished = Signal(dict)
    progress = Signal(int)
    status_update = Signal(str)
    error = Signal(str)
    
    def __init__(self, system: SQLiteRecognitionSystem):
        super().__init__()
        self.system = system
        self.is_cancelled = False
    
    def recognize_single_image(self, image_path: str):
        """Recognize a single image"""
        try:
            self.status_update.emit(f"Processing {Path(image_path).name}...")
            result = self.system.recognize_image(image_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def recognize_batch(self, directory_path: str):
        """Recognize batch of images"""
        try:
            self.status_update.emit(f"Starting batch processing...")
            result = self.system.batch_recognize(directory_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def cancel(self):
        """Cancel current operation"""
        self.is_cancelled = True


class StatusCard(QLabel):
    """Modern status card widget"""
    
    def __init__(self, title: str, value: str = "0", parent=None):
        super().__init__(parent)
        self.setProperty("class", "status-card")
        
        # Create layout
        layout = QVBoxLayout()
        
        # Value label
        self.value_label = QLabel(value)
        self.value_label.setProperty("class", "metric-value")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setProperty("class", "metric-label")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.value_label)
        layout.addWidget(self.title_label)
        layout.setSpacing(4)
        layout.setContentsMargins(16, 16, 16, 16)
        
        self.setLayout(layout)
        self.setMinimumHeight(100)  # Reduced from 120 for better screen fit
        self.setMaximumHeight(140)  # Add max height constraint
    
    def update_value(self, value: str):
        """Update the displayed value"""
        self.value_label.setText(value)


class RecognitionTab(QWidget):
    """Main recognition interface tab"""
    
    def __init__(self, system: SQLiteRecognitionSystem):
        super().__init__()
        self.system = system
        self.current_image_path = None
        self.current_recognized_item_id = None  # Store currently recognized item ID
        self.worker_thread = None
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the recognition interface"""
        layout = QHBoxLayout()
        
        # Left panel - Image display and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Image display area
        image_group = QGroupBox("Image Preview")
        image_layout = QVBoxLayout()
        
        self.image_label = QLabel("Drop image here or click 'Select Image'")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(300, 250)  # Reduced from 400x400 for smaller screens
        self.image_label.setMaximumHeight(400)  # Add max height constraint
        self.image_label.setStyleSheet("""
            QLabel {
                border: 3px dashed #45475a;
                border-radius: 12px;
                background: rgba(49, 50, 68, 0.3);
                color: #6c7086;
                font-size: 16px;
            }
        """)
        self.image_label.setAcceptDrops(True)
        
        image_layout.addWidget(self.image_label)
        
        # Image controls
        image_controls = QHBoxLayout()
        
        self.select_image_btn = QPushButton("📁 Select Image")
        self.select_image_btn.clicked.connect(self.select_image)
        
        self.clear_image_btn = QPushButton("🗑️ Clear")
        self.clear_image_btn.setProperty("class", "secondary")
        self.clear_image_btn.clicked.connect(self.clear_image)
        
        image_controls.addWidget(self.select_image_btn)
        image_controls.addWidget(self.clear_image_btn)
        
        image_layout.addLayout(image_controls)
        image_group.setLayout(image_layout)
        
        # Recognition controls
        recognition_group = QGroupBox("Recognition")
        recognition_layout = QVBoxLayout()
        
        self.recognize_btn = QPushButton("🔍 Recognize Image")
        self.recognize_btn.clicked.connect(self.recognize_image)
        self.recognize_btn.setEnabled(False)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        recognition_layout.addWidget(self.recognize_btn)
        recognition_layout.addWidget(self.progress_bar)
        recognition_layout.addWidget(self.status_label)
        
        recognition_group.setLayout(recognition_layout)
        
        left_layout.addWidget(image_group)
        left_layout.addWidget(recognition_group)
        left_layout.addStretch()
        
        left_panel.setLayout(left_layout)
        
        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Results display
        results_group = QGroupBox("Recognition Results")
        results_layout = QVBoxLayout()
        
        # Main result
        self.result_label = QLabel("No recognition performed")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                background: rgba(49, 50, 68, 0.6);
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 20px;
                font-size: 18px;
                font-weight: 600;
            }
        """)
        
        # Confidence and timing
        metrics_layout = QHBoxLayout()
        
        self.confidence_card = StatusCard("Confidence", "0%")
        self.timing_card = StatusCard("Processing Time", "0ms")
        
        metrics_layout.addWidget(self.confidence_card)
        metrics_layout.addWidget(self.timing_card)
        
        # Top matches table with action column
        self.matches_table = QTableWidget()
        self.matches_table.setColumnCount(4)
        self.matches_table.setHorizontalHeaderLabels(["Rank", "Item ID", "Confidence", "Action"])
        
        # Configure column sizing
        header = self.matches_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Rank
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)           # Item ID
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Confidence  
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)             # Action - fixed width
        header.resizeSection(3, 100)  # Wider Action column for better button visibility
        
        # Increase table height to accommodate buttons properly
        self.matches_table.setMaximumHeight(200)
        self.matches_table.setMinimumHeight(120)
        
        results_layout.addWidget(self.result_label)
        
        # View Details button (initially hidden)
        self.view_details_btn = QPushButton("📋 View Item Details")
        self.view_details_btn.setProperty("class", "secondary")
        self.view_details_btn.clicked.connect(self.view_item_details)
        self.view_details_btn.setVisible(False)  # Hidden until successful recognition
        results_layout.addWidget(self.view_details_btn)
        
        results_layout.addLayout(metrics_layout)
        results_layout.addWidget(QLabel("Top Matches:"))
        results_layout.addWidget(self.matches_table)
        
        results_group.setLayout(results_layout)
        
        # Recognition details
        details_group = QGroupBox("Recognition Details")
        details_layout = QVBoxLayout()
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(120)  # Reduced for better fit
        self.details_text.setReadOnly(True)
        
        details_layout.addWidget(self.details_text)
        details_group.setLayout(details_layout)
        
        right_layout.addWidget(results_group)
        right_layout.addWidget(details_group)
        
        right_panel.setLayout(right_layout)
        
        # Main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 500])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path: str):
        """Load and display an image"""
        self.current_image_path = file_path
        
        # Display image
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # Enable recognition button
            self.recognize_btn.setEnabled(True)
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
        else:
            QMessageBox.warning(self, "Error", "Could not load the selected image.")
    
    def clear_image(self):
        """Clear the current image"""
        self.current_image_path = None
        self.current_recognized_item_id = None
        self.image_label.clear()
        self.image_label.setText("Drop image here or click 'Select Image'")
        self.recognize_btn.setEnabled(False)
        self.status_label.setText("Ready")
        
        # Clear results
        self.result_label.setText("No recognition performed")
        self.view_details_btn.setVisible(False)  # Hide details button
        self.confidence_card.update_value("0%")
        self.timing_card.update_value("0ms")
        self.matches_table.setRowCount(0)
        self.details_text.clear()
    
    def recognize_image(self):
        """Start image recognition"""
        if not self.current_image_path:
            return
        
        # Disable controls
        self.recognize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Create worker thread
        self.worker_thread = QThread()
        self.worker = ImageRecognitionWorker(self.system)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.finished.connect(self.on_recognition_finished)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.error.connect(self.on_recognition_error)
        self.worker_thread.started.connect(
            lambda: self.worker.recognize_single_image(self.current_image_path)
        )
        
        # Start thread
        self.worker_thread.start()
    
    def on_recognition_finished(self, result: dict):
        """Handle recognition completion"""
        # Re-enable controls
        self.recognize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update results
        if result.get('success', False):
            item_id = result['item_id']
            confidence = result['confidence']
            processing_time = result['processing_time'] * 1000  # Convert to ms
            
            # Store recognized item ID and show details button
            self.current_recognized_item_id = item_id
            self.view_details_btn.setVisible(True)
            
            # Update main result
            self.result_label.setText(f"✅ Recognized: {item_id}")
            self.result_label.setStyleSheet("""
                QLabel {
                    background: rgba(166, 227, 161, 0.2);
                    border: 2px solid #a6e3a1;
                    border-radius: 8px;
                    padding: 20px;
                    font-size: 18px;
                    font-weight: 600;
                    color: #a6e3a1;
                }
            """)
            
            # Update metrics
            self.confidence_card.update_value(f"{confidence:.1%}")
            self.timing_card.update_value(f"{processing_time:.0f}ms")
            
            # Update matches table with View Details buttons
            top_matches = result.get('top_matches', [])
            self.matches_table.setRowCount(len(top_matches))
            
            for i, match in enumerate(top_matches):
                match_item_id = match.get('item_id', '')
                
                # Create table items
                rank_item = QTableWidgetItem(str(i + 1))
                item_id_item = QTableWidgetItem(match_item_id)
                confidence_item = QTableWidgetItem(f"{match.get('confidence', 0):.3f}")
                
                rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                confidence_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Create View Details button for this match with high contrast styling
                view_btn = QPushButton("Details")  # Changed to "Details" for better visibility
                view_btn.setMinimumSize(70, 30)  # Larger size for better visibility
                view_btn.setMaximumSize(120, 40)  # More space for text
                
                # High contrast styling to ensure maximum text visibility
                view_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #89b4fa;
                        color: #1e1e2e;
                        border: 2px solid #74c7ec;
                        border-radius: 6px;
                        padding: 6px 12px;
                        font-weight: 700;
                        font-size: 11px;
                        text-align: center;
                    }
                    QPushButton:hover {
                        background-color: #74c7ec;
                        color: #181825;
                        border-color: #89b4fa;
                        font-weight: 700;
                    }
                    QPushButton:pressed {
                        background-color: #6c9fff;
                        color: #181825;
                        border-color: #89b4fa;
                    }
                """)
                
                view_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                view_btn.clicked.connect(lambda checked, item_id=match_item_id: self.view_match_details(item_id))
                
                # Add items to table
                self.matches_table.setItem(i, 0, rank_item)
                self.matches_table.setItem(i, 1, item_id_item)
                self.matches_table.setItem(i, 2, confidence_item)
                self.matches_table.setCellWidget(i, 3, view_btn)  # Button in Action column
                
                # Ensure row height accommodates the larger button
                self.matches_table.setRowHeight(i, 45)  # Increased for better button visibility
            
            # Update details
            details = {
                'Recognition Method': result.get('recognition_method', 'Unknown'),
                'Processing Time': f"{processing_time:.2f}ms",
                'Platform': result.get('system_info', {}).get('platform_type', 'Unknown'),
                'Stage Results': result.get('stage_results', {})
            }
            
            details_text = json.dumps(details, indent=2)
            self.details_text.setText(details_text)
            
            self.status_label.setText("Recognition completed successfully")
            
        else:
            # Recognition failed
            error = result.get('error', 'Recognition failed')
            
            # Hide details button on failure
            self.current_recognized_item_id = None
            self.view_details_btn.setVisible(False)
            
            self.result_label.setText(f"❌ Recognition Failed")
            self.result_label.setStyleSheet("""
                QLabel {
                    background: rgba(243, 139, 168, 0.2);
                    border: 2px solid #f38ba8;
                    border-radius: 8px;
                    padding: 20px;
                    font-size: 18px;
                    font-weight: 600;
                    color: #f38ba8;
                }
            """)
            
            self.details_text.setText(f"Error: {error}")
            self.status_label.setText(f"Error: {error}")
        
        # Clean up thread
        self.worker_thread.quit()
        self.worker_thread.wait()
    
    def on_recognition_error(self, error: str):
        """Handle recognition error"""
        self.recognize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error}")
        
        QMessageBox.critical(self, "Recognition Error", f"Recognition failed:\n{error}")
        
        # Clean up thread
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
    
    def view_item_details(self):
        """Show detailed information about the recognized item"""
        if not self.current_recognized_item_id:
            return
        
        try:
            # First, debug what's actually in the database
            print(f"🔍 DEBUG: Retrieving details for item: {self.current_recognized_item_id}")
            debug_data = self.system.vector_store.debug_item_metadata(self.current_recognized_item_id)
            print(f"🔍 DEBUG: Raw database data: {debug_data}")
            
            # Fetch comprehensive item details from the database
            item_details = self.system.vector_store.get_item_details(self.current_recognized_item_id)
            print(f"🔍 DEBUG: Processed item details: {item_details}")
            
            if item_details:
                # Create and show the details dialog
                dialog = ItemDetailsDialog(self.current_recognized_item_id, item_details, self)
                dialog.exec()
            else:
                QMessageBox.warning(
                    self, "Item Not Found", 
                    f"Could not find details for item: {self.current_recognized_item_id}"
                )
                
        except Exception as e:
            print(f"❌ DEBUG: Error retrieving item details: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Error", 
                f"Error retrieving item details:\n{str(e)}"
            )
    
    def view_match_details(self, item_id: str):
        """Show detailed information about a matched item from the top matches table"""
        try:
            # First, debug what's actually in the database
            print(f"🔍 DEBUG: Retrieving match details for item: {item_id}")
            debug_data = self.system.vector_store.debug_item_metadata(item_id)
            print(f"🔍 DEBUG: Raw database data: {debug_data}")
            
            # Fetch comprehensive item details from the database
            item_details = self.system.vector_store.get_item_details(item_id)
            print(f"🔍 DEBUG: Processed item details: {item_details}")
            
            if item_details:
                # Create and show the details dialog
                dialog = ItemDetailsDialog(item_id, item_details, self)
                dialog.exec()
            else:
                QMessageBox.warning(
                    self, "Item Not Found", 
                    f"Could not find details for item: {item_id}"
                )
                
        except Exception as e:
            print(f"❌ DEBUG: Error retrieving match details: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Error", 
                f"Error retrieving item details:\n{str(e)}"
            )


class ItemDetailsDialog(QDialog):
    """Dialog showing comprehensive item details"""
    
    def __init__(self, item_id: str, item_details: dict, parent=None):
        super().__init__(parent)
        self.item_id = item_id
        self.item_details = item_details
        
        self.setWindowTitle(f"Item Details - {item_id}")
        self.setModal(True)
        self.resize(700, 600)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the details dialog UI"""
        layout = QVBoxLayout()
        
        # Header with item ID
        header = QLabel(f"📋 Item Details: {self.item_id}")
        header.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 700;
                color: #89b4fa;
                padding: 16px;
                background: rgba(137, 180, 250, 0.1);
                border: 2px solid #89b4fa;
                border-radius: 8px;
                margin-bottom: 8px;
            }
        """)
        layout.addWidget(header)
        
        # Scrollable content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        
        # Basic Information with improved spacing
        basic_info = QGroupBox("📝 Basic Information")
        basic_layout = QFormLayout()
        basic_layout.setSpacing(12)  # Increase spacing between rows
        basic_layout.setContentsMargins(20, 20, 20, 20)  # Add padding inside group
        
        # Enhanced labels with consistent styling
        item_id_label = QLabel(str(self.item_details.get('item_id', self.item_id)))
        item_id_label.setStyleSheet("font-weight: 500; color: #cdd6f4;")
        
        item_name_label = QLabel(str(self.item_details.get('item_name', 'N/A')))
        item_name_label.setStyleSheet("font-weight: 600; color: #a6e3a1; font-size: 14px;")
        
        category_label = QLabel(str(self.item_details.get('category', 'N/A')))
        category_label.setStyleSheet("font-weight: 500; color: #f9e2af;")
        
        basic_layout.addRow("Item ID:", item_id_label)
        basic_layout.addRow("Item Name:", item_name_label)
        basic_layout.addRow("Category:", category_label)
        
        # Description with enhanced multi-line formatting
        description_text = str(self.item_details.get('description', 'N/A'))
        
        # Use QTextEdit for better multi-line display instead of QLabel
        desc_widget = QTextEdit()
        desc_widget.setPlainText(description_text)
        desc_widget.setReadOnly(True)
        desc_widget.setMaximumHeight(120)  # Allow more height for multi-line content
        desc_widget.setMinimumHeight(60)   # Ensure minimum visibility
        
        # Enhanced styling for better readability
        desc_widget.setStyleSheet("""
            QTextEdit {
                background-color: rgba(49, 50, 68, 0.3);
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 12px;
                margin: 4px;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border-color: #89b4fa;
                background-color: rgba(49, 50, 68, 0.5);
            }
        """)
        
        basic_layout.addRow("Description:", desc_widget)
        
        # Show data source to help users understand where the item came from
        data_source = self.item_details.get('data_source', 'Unknown')
        source_label = QLabel(data_source)
        if data_source == 'System Generated':
            source_label.setStyleSheet("color: #f9e2af; font-weight: 600;")  # Yellow for system items
        else:
            source_label.setStyleSheet("color: #a6e3a1; font-weight: 600;")  # Green for user items
        basic_layout.addRow("Data Source:", source_label)
        
        # Timestamps
        created_at = self.item_details.get('created_at')
        if created_at:
            try:
                if isinstance(created_at, (int, float)):
                    created_date = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    created_date = str(created_at)
                basic_layout.addRow("Created:", QLabel(created_date))
            except:
                basic_layout.addRow("Created:", QLabel("Unknown"))
        
        basic_info.setLayout(basic_layout)
        content_layout.addWidget(basic_info)
        content_layout.addSpacing(15)  # Add spacing between sections
        
        # Processing Options with improved formatting
        processing_opts = self.item_details.get('processing_options', {})
        if processing_opts:
            processing_group = QGroupBox("⚙️ Processing Options")
            processing_layout = QFormLayout()
            processing_layout.setSpacing(10)
            processing_layout.setContentsMargins(20, 15, 20, 15)
            
            augmentation = processing_opts.get('augmentation', False)
            aug_label = QLabel("✅ Enabled" if augmentation else "❌ Disabled")
            aug_label.setStyleSheet("font-weight: 500; color: #a6e3a1;" if augmentation else "font-weight: 500; color: #f38ba8;")
            processing_layout.addRow("Augmentation:", aug_label)
            
            bg_removal = processing_opts.get('background_removal', False)
            bg_label = QLabel("✅ Enabled" if bg_removal else "❌ Disabled")
            bg_label.setStyleSheet("font-weight: 500; color: #a6e3a1;" if bg_removal else "font-weight: 500; color: #f38ba8;")
            processing_layout.addRow("Background Removal:", bg_label)
            
            processing_group.setLayout(processing_layout)
            content_layout.addWidget(processing_group)
            content_layout.addSpacing(15)  # Add spacing after section
        
        # Feature Information with better spacing
        feature_info = QGroupBox("🧠 Feature Extraction")
        feature_layout = QVBoxLayout()
        feature_layout.setContentsMargins(20, 15, 20, 15)
        feature_layout.setSpacing(10)
        
        feature_text = QTextEdit()
        feature_text.setMaximumHeight(130)
        feature_text.setMinimumHeight(100)
        feature_text.setReadOnly(True)
        
        # Enhanced styling for feature text
        feature_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(49, 50, 68, 0.3);
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 10px;
                color: #cdd6f4;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                line-height: 1.3;
            }
        """)
        
        feature_details = {
            'Feature Types': 'CLIP + DINOv2 (Multimodal)',
            'Vector Dimensions': f"{self.item_details.get('feature_dimensions', '1536')}D",
            'Extraction Method': self.item_details.get('extraction_method', 'Cross-platform optimized'),
            'Storage Format': 'SQLite BLOB (compressed)',
            'Indexing': 'Hybrid SQLite + FAISS'
        }
        
        feature_text.setText(json.dumps(feature_details, indent=2))
        feature_layout.addWidget(feature_text)
        feature_info.setLayout(feature_layout)
        content_layout.addWidget(feature_info)
        
        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)


class BatchProcessingTab(QWidget):
    """Batch processing interface tab"""
    
    def __init__(self, system: SQLiteRecognitionSystem):
        super().__init__()
        self.system = system
        self.current_directory = None
        self.worker_thread = None
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the batch processing interface"""
        # Create scrollable layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Content widget
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # Directory selection
        directory_group = QGroupBox("Directory Selection")
        directory_layout = QHBoxLayout()
        
        self.directory_path = QLineEdit()
        self.directory_path.setPlaceholderText("Select a directory containing images...")
        
        self.browse_btn = QPushButton("📁 Browse")
        self.browse_btn.clicked.connect(self.select_directory)
        
        directory_layout.addWidget(self.directory_path)
        directory_layout.addWidget(self.browse_btn)
        directory_group.setLayout(directory_layout)
        
        # Batch controls
        controls_group = QGroupBox("Batch Processing")
        controls_layout = QVBoxLayout()
        
        # Start/stop buttons
        button_layout = QHBoxLayout()
        
        self.start_batch_btn = QPushButton("🚀 Start Batch Processing")
        self.start_batch_btn.clicked.connect(self.start_batch_processing)
        self.start_batch_btn.setEnabled(False)
        
        self.stop_batch_btn = QPushButton("⏹️ Stop")
        self.stop_batch_btn.setProperty("class", "danger")
        self.stop_batch_btn.clicked.connect(self.stop_batch_processing)
        self.stop_batch_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_batch_btn)
        button_layout.addWidget(self.stop_batch_btn)
        
        # Progress and status
        self.batch_progress = QProgressBar()
        self.batch_status = QLabel("Ready to process images")
        
        controls_layout.addLayout(button_layout)
        controls_layout.addWidget(self.batch_progress)
        controls_layout.addWidget(self.batch_status)
        
        controls_group.setLayout(controls_layout)
        
        # Results summary
        summary_group = QGroupBox("Processing Summary")
        summary_layout = QGridLayout()
        
        self.total_images_card = StatusCard("Total Images", "0")
        self.successful_card = StatusCard("Successful", "0")
        self.failed_card = StatusCard("Failed", "0")
        self.avg_confidence_card = StatusCard("Avg Confidence", "0%")
        
        summary_layout.addWidget(self.total_images_card, 0, 0)
        summary_layout.addWidget(self.successful_card, 0, 1)
        summary_layout.addWidget(self.failed_card, 0, 2)
        summary_layout.addWidget(self.avg_confidence_card, 0, 3)
        
        summary_group.setLayout(summary_layout)
        
        # Results table
        results_group = QGroupBox("Detailed Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Filename", "Status", "Item ID", "Confidence", "Processing Time"
        ])
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        
        # Export button
        export_btn = QPushButton("💾 Export Results")
        export_btn.setProperty("class", "secondary")
        export_btn.clicked.connect(self.export_results)
        
        results_layout.addWidget(self.results_table)
        results_layout.addWidget(export_btn)
        
        results_group.setLayout(results_layout)
        
        # Add all groups to main layout
        layout.addWidget(directory_group)
        layout.addWidget(controls_group)
        layout.addWidget(summary_group)
        layout.addWidget(results_group)
        
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        
        # Set scroll area as main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
    
    def select_directory(self):
        """Select directory containing images"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory with Images"
        )
        
        if directory:
            self.current_directory = directory
            self.directory_path.setText(directory)
            self.start_batch_btn.setEnabled(True)
            
            # Count images in directory
            image_count = self.count_images_in_directory(directory)
            self.batch_status.setText(f"Ready to process {image_count} images")
            self.total_images_card.update_value(str(image_count))
    
    def count_images_in_directory(self, directory: str) -> int:
        """Count image files in directory"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        count = 0
        
        for file_path in Path(directory).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                count += 1
        
        return count
    
    def start_batch_processing(self):
        """Start batch processing"""
        if not self.current_directory:
            return
        
        # Update UI
        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        self.batch_progress.setRange(0, 0)  # Indeterminate
        self.results_table.setRowCount(0)
        
        # Create worker thread
        self.worker_thread = QThread()
        self.worker = ImageRecognitionWorker(self.system)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.finished.connect(self.on_batch_finished)
        self.worker.status_update.connect(self.batch_status.setText)
        self.worker.error.connect(self.on_batch_error)
        self.worker_thread.started.connect(
            lambda: self.worker.recognize_batch(self.current_directory)
        )
        
        # Start processing
        self.worker_thread.start()
    
    def stop_batch_processing(self):
        """Stop batch processing"""
        if self.worker:
            self.worker.cancel()
        
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_status.setText("Batch processing stopped")
        
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
    
    def on_batch_finished(self, result: dict):
        """Handle batch processing completion"""
        # Update UI
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(1)
        
        if result.get('success', False):
            summary = result['summary']
            
            # Update summary cards
            self.total_images_card.update_value(str(summary['total_images']))
            self.successful_card.update_value(str(summary['successful_recognitions']))
            self.failed_card.update_value(str(summary['failed_recognitions']))
            self.avg_confidence_card.update_value(f"{summary['average_confidence']:.1%}")
            
            # Populate results table
            detailed_results = result.get('detailed_results', [])
            self.results_table.setRowCount(len(detailed_results))
            
            for i, item_result in enumerate(detailed_results):
                filename = item_result['filename']
                success = "✅ Success" if item_result.get('success', False) else "❌ Failed"
                item_id = item_result.get('item_id', 'Unknown')
                confidence = f"{item_result.get('confidence', 0):.3f}"
                processing_time = f"{item_result.get('processing_time', 0)*1000:.0f}ms"
                
                self.results_table.setItem(i, 0, QTableWidgetItem(filename))
                self.results_table.setItem(i, 1, QTableWidgetItem(success))
                self.results_table.setItem(i, 2, QTableWidgetItem(item_id))
                self.results_table.setItem(i, 3, QTableWidgetItem(confidence))
                self.results_table.setItem(i, 4, QTableWidgetItem(processing_time))
            
            # Update status
            success_rate = summary['success_rate_percent']
            total_time = summary['total_processing_time']
            
            self.batch_status.setText(
                f"Completed! Success rate: {success_rate:.1f}% in {total_time:.1f}s"
            )
            
        else:
            error = result.get('error', 'Unknown error')
            self.batch_status.setText(f"Batch processing failed: {error}")
        
        # Clean up thread
        self.worker_thread.quit()
        self.worker_thread.wait()
    
    def on_batch_error(self, error: str):
        """Handle batch processing error"""
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_status.setText(f"Error: {error}")
        
        QMessageBox.critical(self, "Batch Processing Error", f"Batch processing failed:\n{error}")
        
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
    
    def export_results(self):
        """Export batch results to file"""
        if self.results_table.rowCount() == 0:
            QMessageBox.information(self, "No Data", "No results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            # Extract data from table
            results = []
            for row in range(self.results_table.rowCount()):
                result = {
                    'filename': self.results_table.item(row, 0).text(),
                    'status': self.results_table.item(row, 1).text(),
                    'item_id': self.results_table.item(row, 2).text(),
                    'confidence': self.results_table.item(row, 3).text(),
                    'processing_time': self.results_table.item(row, 4).text()
                }
                results.append(result)
            
            # Export to file
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'total_images': self.total_images_card.value_label.text(),
                'successful': self.successful_card.value_label.text(),
                'failed': self.failed_card.value_label.text(),
                'avg_confidence': self.avg_confidence_card.value_label.text(),
                'results': results
            }
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{e}")


class AddItemsTab(QWidget):
    """Tab for adding new items to the system"""
    
    def __init__(self, system: SQLiteRecognitionSystem):
        super().__init__()
        self.system = system
        self.selected_images = []
        self.worker_thread = None
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the add items interface"""
        # Create scrollable layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Content widget
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # Item Information Group
        item_info_group = QGroupBox("Item Information")
        item_info_layout = QFormLayout()
        
        self.item_id_input = QLineEdit()
        self.item_id_input.setPlaceholderText("Enter unique item identifier...")
        
        self.item_name_input = QLineEdit()
        self.item_name_input.setPlaceholderText("Enter item name...")
        
        self.category_input = QLineEdit()
        self.category_input.setPlaceholderText("Enter category (optional)...")
        
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Enter item description...")
        self.description_input.setMaximumHeight(80)
        
        item_info_layout.addRow("Item ID:", self.item_id_input)
        item_info_layout.addRow("Item Name:", self.item_name_input)
        item_info_layout.addRow("Category:", self.category_input)
        item_info_layout.addRow("Description:", self.description_input)
        
        item_info_group.setLayout(item_info_layout)
        
        # Image Selection Group
        image_group = QGroupBox("Images")
        image_layout = QVBoxLayout()
        
        # Image controls
        image_controls = QHBoxLayout()
        
        self.select_images_btn = QPushButton("📁 Select Images")
        self.select_images_btn.clicked.connect(self.select_images)
        
        self.clear_images_btn = QPushButton("🗑️ Clear All")
        self.clear_images_btn.setProperty("class", "secondary")
        self.clear_images_btn.clicked.connect(self.clear_images)
        
        self.images_count_label = QLabel("No images selected")
        
        image_controls.addWidget(self.select_images_btn)
        image_controls.addWidget(self.clear_images_btn)
        image_controls.addStretch()
        image_controls.addWidget(self.images_count_label)
        
        # Image list
        self.images_list = QTreeWidget()
        self.images_list.setHeaderLabels(["Filename", "Size", "Path"])
        self.images_list.setMaximumHeight(150)
        
        image_layout.addLayout(image_controls)
        image_layout.addWidget(self.images_list)
        
        image_group.setLayout(image_layout)
        
        # Processing Group
        processing_group = QGroupBox("Processing Options")
        processing_layout = QVBoxLayout()
        
        # Processing options
        options_layout = QHBoxLayout()
        
        self.enable_augmentation = QCheckBox("Enable Augmentation")
        self.enable_augmentation.setChecked(True)
        
        self.enable_background_removal = QCheckBox("Background Removal")
        self.enable_background_removal.setChecked(True)
        
        options_layout.addWidget(self.enable_augmentation)
        options_layout.addWidget(self.enable_background_removal)
        options_layout.addStretch()
        
        # Add Item button
        self.add_item_btn = QPushButton("🎯 Add Item to System")
        self.add_item_btn.clicked.connect(self.add_item)
        self.add_item_btn.setEnabled(False)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_label = QLabel("Fill in item information and select images")
        
        processing_layout.addLayout(options_layout)
        processing_layout.addWidget(self.add_item_btn)
        processing_layout.addWidget(self.progress_bar)
        processing_layout.addWidget(self.status_label)
        
        processing_group.setLayout(processing_layout)
        
        # Add all groups to layout
        layout.addWidget(item_info_group)
        layout.addWidget(image_group)
        layout.addWidget(processing_group)
        layout.addStretch()
        
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        
        # Set scroll area as main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
        # Connect input validation
        self.item_id_input.textChanged.connect(self.validate_inputs)
        self.item_name_input.textChanged.connect(self.validate_inputs)
    
    def validate_inputs(self):
        """Validate inputs and enable/disable add button"""
        has_item_id = bool(self.item_id_input.text().strip())
        has_item_name = bool(self.item_name_input.text().strip())
        has_images = len(self.selected_images) > 0
        
        self.add_item_btn.setEnabled(has_item_id and has_item_name and has_images)
        
        if has_item_id and has_item_name and has_images:
            self.status_label.setText("Ready to add item")
        else:
            missing = []
            if not has_item_id:
                missing.append("Item ID")
            if not has_item_name:
                missing.append("Item Name")
            if not has_images:
                missing.append("Images")
            self.status_label.setText(f"Missing: {', '.join(missing)}")
    
    def select_images(self):
        """Select images for the item"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
        )
        
        if file_paths:
            self.selected_images = file_paths
            self.update_images_list()
            self.validate_inputs()
    
    def clear_images(self):
        """Clear selected images"""
        self.selected_images = []
        self.update_images_list()
        self.validate_inputs()
    
    def update_images_list(self):
        """Update the images list display"""
        self.images_list.clear()
        
        if not self.selected_images:
            self.images_count_label.setText("No images selected")
            return
        
        for image_path in self.selected_images:
            path = Path(image_path)
            try:
                size = path.stat().st_size
                size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
            except:
                size_str = "Unknown"
            
            item = QTreeWidgetItem([path.name, size_str, str(path.parent)])
            self.images_list.addTopLevelItem(item)
        
        count = len(self.selected_images)
        self.images_count_label.setText(f"{count} image{'s' if count != 1 else ''} selected")
    
    def add_item(self):
        """Add the item to the system"""
        if not self.add_item_btn.isEnabled():
            return
        
        item_id = self.item_id_input.text().strip()
        item_name = self.item_name_input.text().strip()
        category = self.category_input.text().strip()
        description = self.description_input.toPlainText().strip()
        
        # Disable controls and setup progress tracking
        self.add_item_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        # Use determinate progress for better user experience
        total_steps = len(self.selected_images) * (8 if self.enable_augmentation.isChecked() else 1) + 2  # +2 for setup/cleanup
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting item processing...")
        
        # Create worker thread for item processing
        self.worker_thread = QThread()
        self.worker = AddItemWorker(
            self.system, item_id, item_name, category, description,
            self.selected_images, self.enable_augmentation.isChecked(),
            self.enable_background_removal.isChecked()
        )
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.finished.connect(self.on_add_finished)
        self.worker.progress.connect(self.on_add_progress)
        self.worker.progress_value.connect(self.progress_bar.setValue)  # NEW: Connect numeric progress
        self.worker.error.connect(self.on_add_error)
        
        self.worker_thread.started.connect(self.worker.process_item)
        
        # Start processing
        self.worker_thread.start()
    
    def on_add_progress(self, message: str):
        """Handle progress updates"""
        self.status_label.setText(message)
    
    def on_add_finished(self, result: dict):
        """Handle successful item addition"""
        self.progress_bar.setVisible(False)
        self.add_item_btn.setEnabled(True)
        
        if result.get('success', False):
            self.status_label.setText(f"✅ Item '{result['item_id']}' added successfully!")
            
            # Clear form
            self.item_id_input.clear()
            self.item_name_input.clear()
            self.category_input.clear()
            self.description_input.clear()
            self.clear_images()
            
            QMessageBox.information(
                self, "Success",
                f"Item '{result['item_id']}' has been added to the system.\n\n"
                f"Images processed: {result.get('images_processed', 0)}\n"
                f"Features extracted: {result.get('features_extracted', 0)}\n"
                f"Processing time: {result.get('processing_time', 0):.2f}s\n\n"
                f"Database Status:\n"
                f"Total items: {result.get('total_items_in_db', 0)}\n"
                f"Total images: {result.get('total_images_in_db', 0)} ({result.get('total_original_images_in_db', 0)} original + {result.get('total_augmented_images_in_db', 0)} augmented)\n"
                f"Total features: {result.get('total_features_in_db', 0)}"
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            self.status_label.setText(f"❌ Failed: {error_msg}")
            QMessageBox.critical(self, "Error", f"Failed to add item:\n{error_msg}")
        
        # Clean up thread
        self.worker_thread.quit()
        self.worker_thread.wait()
    
    def on_add_error(self, error: str):
        """Handle processing errors"""
        self.progress_bar.setVisible(False)
        self.add_item_btn.setEnabled(True)
        self.status_label.setText(f"❌ Error: {error}")
        
        QMessageBox.critical(self, "Processing Error", f"Error processing item:\n{error}")
        
        # Clean up thread
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()


class AddItemWorker(QObject):
    """Worker thread for adding items to the system"""
    
    finished = Signal(dict)
    progress = Signal(str)
    progress_value = Signal(int)  # NEW: Numeric progress signal
    error = Signal(str)
    
    def __init__(self, system, item_id, item_name, category, description, 
                 image_paths, enable_augmentation, enable_background_removal):
        super().__init__()
        self.system = system
        self.item_id = item_id
        self.item_name = item_name
        self.category = category or "uncategorized"
        self.description = description or ""
        self.image_paths = image_paths
        self.enable_augmentation = enable_augmentation
        self.enable_background_removal = enable_background_removal
        
        # Progress tracking
        self.total_steps = len(image_paths) * (8 if enable_augmentation else 1) + 2
        self.current_step = 0
    
    def update_progress(self, message: str):
        """Update progress with both message and numeric value"""
        self.current_step += 1
        self.progress.emit(message)
        self.progress_value.emit(self.current_step)
    
    def process_item(self):
        """Process and add the item using the full pipeline"""
        try:
            start_time = time.time()
            
            self.update_progress(f"Starting to process {len(self.image_paths)} images...")
            
            # Create temporary directory for processing
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                item_dir = Path(temp_dir) / self.item_id
                item_dir.mkdir()
                
                # Copy images to temporary directory
                self.update_progress("Copying images to processing directory...")
                for i, image_path in enumerate(self.image_paths):
                    src_path = Path(image_path)
                    dst_path = item_dir / f"image_{i:03d}{src_path.suffix}"
                    shutil.copy2(src_path, dst_path)
                
                # Add item to database first
                self.update_progress("Adding item metadata to database...")
                metadata = {
                    'item_name': self.item_name,
                    'category': self.category,
                    'description': self.description,
                    'created_at': time.time(),
                    'processing_options': {
                        'augmentation': self.enable_augmentation,
                        'background_removal': self.enable_background_removal
                    }
                }
                
                success = self.system.vector_store.add_item(self.item_id, metadata)
                if not success:
                    raise RuntimeError(f"Failed to add item {self.item_id} to database")
                
                # Process images with augmentation pipeline if enabled
                if self.enable_augmentation:
                    self.update_progress("Setting up high-speed augmentation pipeline...")
                    
                    # Create GUI-optimized augmentation config - FAST processing for interactive use
                    augmentation_config = {
                        'augmentations_per_image': 8,   # DRAMATICALLY reduced for GUI speed (was 30)
                        'background_removal': self.enable_background_removal,
                        'strategy_weights': {
                            'geometric': 0.40,          # Focus on most effective augmentations
                            'perspective': 0.30, 
                            'lighting': 0.20,
                            'noise_blur': 0.10,         # Reduce less effective augmentations
                            'effects': 0.00             # Disable slowest effects for GUI
                        },
                        # GUI-SPECIFIC optimizations for responsive interface
                        'batch_processing': True,       # Enable batch processing for GPU efficiency
                        'parallel_workers': 2,          # Reduce workers for GUI responsiveness (was 4)
                        'memory_efficient': True,       # Enable memory efficiency for GUI stability
                        'cache_backgrounds': True,      # Cache synthetic backgrounds
                        'use_hybrid_pipeline': True,    # Enable CPU-GPU hybrid processing
                        'early_crop_optimization': True, # Crop first for 16x speedup
                        'gpu_batch_normalization': True, # Batch normalize on GPU
                        'extract_colors': True,         # Enable color extraction
                        'color_extraction': {
                            'num_colors': 3,            # Reduced colors for faster processing (was 5)
                            'color_quality': 2,         # Lower quality for GUI speed (was 3)
                            'remove_background': self.enable_background_removal
                        },
                        # GUI threading optimizations
                        'fix_opencv_threading': True,   # Prevent DataLoader conflicts
                        'progress_callback': self.progress.emit,  # Enable progress updates
                        'gui_mode': True                # Enable GUI-specific optimizations
                    }
                    
                    from src.data_preparation.advanced_augmentation import AdvancedAugmentationPipeline
                    augmentation_pipeline = AdvancedAugmentationPipeline(
                        augmentation_config, self.system.vector_store
                    )
                    
                    # Process the item directory with progress updates
                    self.update_progress(f"Applying 8 fast augmentations to {len(self.image_paths)} images...")
                    augment_result = augmentation_pipeline.process_item_to_sqlite(item_dir, self.item_id)
                    
                    if augment_result.get('status') != 'success':
                        error_msg = augment_result.get('error', 'Augmentation processing failed')
                        self.progress.emit(f"Augmentation failed: {error_msg}")
                        raise RuntimeError(f"Augmentation processing failed: {error_msg}")
                    
                    images_processed = augment_result.get('total_images_processed', len(self.image_paths))
                    self.update_progress(f"✅ Processed {images_processed} images with fast augmentation")
                else:
                    # Process without augmentation - just extract features
                    self.update_progress("Extracting CLIP + DINOv2 features (no augmentation)...")
                    
                    feature_result = self.system.feature_extractor.process_item_directory_to_sqlite(item_dir, self.item_id)
                    
                    if feature_result.get('status') != 'success':
                        raise RuntimeError("Feature extraction failed")
                        
                    images_processed = feature_result.get('success_count', len(self.image_paths))
                    self.update_progress(f"✅ Extracted features from {images_processed} images")
                
                # Final processing time
                processing_time = time.time() - start_time
                self.update_progress(f"Finalizing database... ({processing_time:.1f}s total)")
                
                # Get final statistics
                stats = self.system.vector_store.get_statistics()
                
                result = {
                    'success': True,
                    'item_id': self.item_id,
                    'images_processed': images_processed,
                    'features_extracted': images_processed * 2,  # CLIP + DINOv2
                    'processing_time': processing_time,
                    'total_items_in_db': stats.get('total_items', 0),
                    'total_images_in_db': stats.get('total_images', 0),
                    'total_original_images_in_db': stats.get('total_original_images', 0),
                    'total_augmented_images_in_db': stats.get('total_augmented_images', 0),
                    'total_features_in_db': stats.get('total_features', 0)
                }
                
                self.progress.emit("Item processing completed successfully!")
                self.finished.emit(result)
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing item {self.item_id}: {e}")
            self.error.emit(str(e))


class SystemMonitorTab(QWidget):
    """System monitoring and status tab"""
    
    def __init__(self, system: SQLiteRecognitionSystem):
        super().__init__()
        self.system = system
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        
        self.init_ui()
        self.start_monitoring()
    
    def init_ui(self):
        """Initialize the monitoring interface"""
        # Create scrollable layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Content widget
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # System status overview
        status_group = QGroupBox("System Status")
        status_layout = QGridLayout()
        
        self.platform_card = StatusCard("Platform", "Unknown")
        self.health_card = StatusCard("Health Score", "0/100")
        self.uptime_card = StatusCard("Uptime", "0s")
        self.db_size_card = StatusCard("Database Size", "0 MB")
        
        status_layout.addWidget(self.platform_card, 0, 0)
        status_layout.addWidget(self.health_card, 0, 1)
        status_layout.addWidget(self.uptime_card, 0, 2)
        status_layout.addWidget(self.db_size_card, 0, 3)
        
        status_group.setLayout(status_layout)
        
        # Performance metrics
        performance_group = QGroupBox("Performance Metrics")
        performance_layout = QGridLayout()
        
        self.total_recognitions_card = StatusCard("Total Recognitions", "0")
        self.success_rate_card = StatusCard("Success Rate", "0%")
        self.avg_time_card = StatusCard("Avg Time", "0ms")
        self.items_card = StatusCard("Database Items", "0")
        
        performance_layout.addWidget(self.total_recognitions_card, 0, 0)
        performance_layout.addWidget(self.success_rate_card, 0, 1)
        performance_layout.addWidget(self.avg_time_card, 0, 2)
        performance_layout.addWidget(self.items_card, 0, 3)
        
        performance_group.setLayout(performance_layout)
        
        # System information
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(180)  # Reduced for better responsive fit
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setProperty("class", "secondary")
        refresh_btn.clicked.connect(self.refresh_info)
        
        info_layout.addWidget(self.info_text)
        info_layout.addWidget(refresh_btn)
        
        info_group.setLayout(info_layout)
        
        # Add all groups
        layout.addWidget(status_group)
        layout.addWidget(performance_group)
        layout.addWidget(info_group)
        layout.addStretch()
        
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        
        # Set scroll area as main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
    
    def start_monitoring(self):
        """Start automatic monitoring updates"""
        self.update_metrics()
        self.update_timer.start(5000)  # Update every 5 seconds
    
    def stop_monitoring(self):
        """Stop automatic monitoring updates"""
        self.update_timer.stop()
    
    def update_metrics(self):
        """Update all monitoring metrics"""
        try:
            status = self.system.get_system_status()
            
            # Update system status cards
            system_info = status.get('system_info', {})
            db_status = status.get('database_status', {})
            platform_info = status.get('platform_optimization', {})
            
            # Platform and health
            platform_type = platform_info.get('platform_type', 'Unknown')
            self.platform_card.update_value(platform_type)
            
            # Calculate uptime
            init_time = system_info.get('initialized_at', '')
            if init_time:
                try:
                    init_datetime = datetime.fromisoformat(init_time)
                    uptime = datetime.now() - init_datetime
                    uptime_str = f"{int(uptime.total_seconds())}s"
                    self.uptime_card.update_value(uptime_str)
                except:
                    self.uptime_card.update_value("Unknown")
            
            # Database size
            db_size_mb = db_status.get('database_size_mb', 0)
            self.db_size_card.update_value(f"{db_size_mb:.1f} MB")
            
            # Performance metrics
            total_recognitions = system_info.get('total_recognitions', 0)
            successful = system_info.get('successful_recognitions', 0)
            
            self.total_recognitions_card.update_value(str(total_recognitions))
            
            if total_recognitions > 0:
                success_rate = (successful / total_recognitions) * 100
                self.success_rate_card.update_value(f"{success_rate:.1f}%")
                
                avg_time = system_info.get('average_recognition_time', 0) * 1000
                self.avg_time_card.update_value(f"{avg_time:.0f}ms")
            
            # Database items
            total_items = db_status.get('total_items', 0)
            self.items_card.update_value(str(total_items))
            
            # Calculate health score (simplified)
            health_score = 100  # Start with perfect score
            
            # Penalize if average time exceeds target
            target_time = platform_info.get('target_recognition_ms', 250)
            if total_recognitions > 0:
                actual_time = system_info.get('average_recognition_time', 0) * 1000
                if actual_time > target_time:
                    health_score -= min(50, (actual_time / target_time - 1) * 100)
            
            # Penalize low success rate
            if total_recognitions > 0:
                if success_rate < 80:
                    health_score -= (80 - success_rate)
            
            health_score = max(0, min(100, health_score))
            self.health_card.update_value(f"{health_score:.0f}/100")
            
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def refresh_info(self):
        """Refresh detailed system information"""
        try:
            status = self.system.get_system_status()
            
            # Format status for display
            info_text = "=== SQLite Recognition System Status ===\n\n"
            
            # System info
            system_info = status.get('system_info', {})
            info_text += "System Information:\n"
            info_text += f"  Platform: {status.get('platform_optimization', {}).get('platform_type', 'Unknown')}\n"
            info_text += f"  Optimization Tier: {status.get('platform_optimization', {}).get('optimization_tier', 'Unknown')}\n"
            info_text += f"  Total Recognitions: {system_info.get('total_recognitions', 0)}\n"
            info_text += f"  Successful: {system_info.get('successful_recognitions', 0)}\n"
            info_text += f"  Database Path: {system_info.get('database_path', 'Unknown')}\n\n"
            
            # Database status
            db_status = status.get('database_status', {})
            info_text += "Database Status:\n"
            info_text += f"  Size: {db_status.get('database_size_mb', 0):.2f} MB\n"
            info_text += f"  Items: {db_status.get('total_items', 0)}\n"
            info_text += f"  Images: {db_status.get('total_images', 0)} ({db_status.get('total_original_images', 0)} original + {db_status.get('total_augmented_images', 0)} augmented)\n"
            info_text += f"  Features: {db_status.get('total_features', 0)}\n"
            info_text += f"  Vectors: {db_status.get('total_vectors', 0)}\n\n"
            
            # Performance
            perf_stats = status.get('pipeline_statistics', {})
            if perf_stats:
                info_text += "Performance Statistics:\n"
                info_text += f"  Total Queries: {perf_stats.get('total_queries', 0)}\n"
                info_text += f"  Average Time: {perf_stats.get('avg_recognition_time_ms', 0):.1f}ms\n"
                info_text += f"  System Health Score: {perf_stats.get('system_health_score', 0):.1f}/100\n"
            
            self.info_text.setText(info_text)
            
        except Exception as e:
            self.info_text.setText(f"Error loading system information:\n{e}")


class ModernRecognitionGUI(QMainWindow):
    """Main application window with modern design"""
    
    def __init__(self):
        super().__init__()
        self.system = None
        self.config_path = "config.yaml"
        self.database_path = "recognition.db"
        
        self.init_ui()
        self.init_system()
    
    def init_ui(self):
        """Initialize the main user interface"""
        self.setWindowTitle("SQLite AI Recognition System")
        
        # Get screen dimensions for responsive sizing
        app = QApplication.instance()
        screen = app.primaryScreen()
        screen_size = screen.availableSize()
        
        # Calculate responsive window size (80% of screen, with practical limits)
        window_width = max(1000, min(1400, int(screen_size.width() * 0.8)))
        window_height = max(600, min(800, int(screen_size.height() * 0.8)))
        
        # Center window on screen
        x = (screen_size.width() - window_width) // 2
        y = (screen_size.height() - window_height) // 2
        
        self.setGeometry(x, y, window_width, window_height)
        self.setMinimumSize(800, 500)  # Minimum window size
        
        # Set application icon (you can add an actual icon file)
        # self.setWindowIcon(QIcon("icon.png"))
        
        # Create scrollable main layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Main content widget
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # We'll add tabs after system initialization
        layout.addWidget(self.tab_widget)
        
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        
        # Set scroll area as central widget
        self.setCentralWidget(scroll_area)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Initializing system...")
        
        # Apply modern theme
        self.setStyleSheet(DARK_THEME)
    
    def create_header(self) -> QWidget:
        """Create modern header with title and system info"""
        header = QFrame()
        header.setMaximumHeight(80)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #89b4fa, stop: 1 #74c7ec);
                border-radius: 12px;
                margin: 8px;
            }
        """)
        
        layout = QHBoxLayout()
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        
        title = QLabel("🤖 SQLite AI Recognition System")
        title.setStyleSheet("""
            QLabel {
                color: #1e1e2e;
                font-size: 24px;
                font-weight: 700;
                margin: 0;
            }
        """)
        
        subtitle = QLabel("Modern interface for high-accuracy image recognition")
        subtitle.setStyleSheet("""
            QLabel {
                color: rgba(30, 30, 46, 0.8);
                font-size: 14px;
                font-weight: 500;
                margin: 0;
            }
        """)
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        
        # System status indicator
        self.system_status_label = QLabel("🔄 Initializing...")
        self.system_status_label.setStyleSheet("""
            QLabel {
                color: #1e1e2e;
                font-size: 16px;
                font-weight: 600;
                padding: 12px 24px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 8px;
            }
        """)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        layout.addWidget(self.system_status_label)
        
        header.setLayout(layout)
        return header
    
    def init_system(self):
        """Initialize the recognition system"""
        try:
            # Check if config exists
            if not Path(self.config_path).exists():
                self.create_default_config()
            
            # Initialize system
            self.system = SQLiteRecognitionSystem(self.config_path, self.database_path)
            
            if self.system.initialize():
                self.system_status_label.setText("✅ System Ready")
                self.status_bar.showMessage("System initialized successfully")
                
                # Create tabs
                self.create_tabs()
                
            else:
                self.system_status_label.setText("❌ System Error")
                self.status_bar.showMessage("System initialization failed")
                
                QMessageBox.critical(
                    self, "System Error",
                    "Failed to initialize the recognition system.\nPlease check the configuration and try again."
                )
                
        except Exception as e:
            self.system_status_label.setText("❌ System Error")
            self.status_bar.showMessage(f"Error: {e}")
            
            QMessageBox.critical(
                self, "Initialization Error",
                f"Failed to initialize system:\n{e}"
            )
    
    def create_default_config(self):
        """Create a default configuration file"""
        default_config = {
            'platform': {
                'auto_optimize': True,
                'force_cpu': False
            },
            'feature_extraction': {
                'clip_model': 'ViT-L/14',
                'dinov2_model': 'dinov2_vitb14',
                'device': 'auto',
                'cache_features': True,
                'batch_size': 8
            },
            'recognition': {
                'confidence_threshold': 0.98,
                'min_stage1_confidence': 0.85,
                'high_confidence_threshold': 0.95,
                'refinement_threshold': 0.82,
                'confidence_gap_threshold': 0.15,
                'max_candidate_score_gap': 0.1,
                'min_top_score_margin': 0.05,
                'initial_search_k': 50,
                'final_candidates_k': 10,
                'hybrid_mode': True,
                'cache_size': 1000
            },
            'monitoring': {
                'enable_performance_tracking': True,
                'real_time_monitoring': True,
                'metrics_retention_days': 7
            }
        }
        
        with open(self.config_path, 'w') as f:
            import yaml
            yaml.dump(default_config, f, default_flow_style=False)
    
    def create_tabs(self):
        """Create all application tabs"""
        if not self.system:
            return
        
        # Recognition tab
        recognition_tab = RecognitionTab(self.system)
        self.tab_widget.addTab(recognition_tab, "🔍 Recognition")
        
        # Add Items tab
        add_items_tab = AddItemsTab(self.system)
        self.tab_widget.addTab(add_items_tab, "➕ Add Items")
        
        # Batch processing tab
        batch_tab = BatchProcessingTab(self.system)
        self.tab_widget.addTab(batch_tab, "📁 Batch Processing")
        
        # System monitoring tab
        monitor_tab = SystemMonitorTab(self.system)
        self.tab_widget.addTab(monitor_tab, "📊 System Monitor")
        
        # Store references for cleanup
        self.monitor_tab = monitor_tab
    
    def closeEvent(self, event):
        """Handle application close event"""
        if hasattr(self, 'monitor_tab'):
            self.monitor_tab.stop_monitoring()
        
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("SQLite AI Recognition System")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AI Recognition Team")
    
    # Create and show main window
    window = ModernRecognitionGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()