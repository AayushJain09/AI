#!/usr/bin/env python3
"""
AI Recognition System - Frontend Application
Modern PyQt6 desktop application for inventory management
"""

import sys
import os
import asyncio
import json
import requests
import base64
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QFrame, QScrollArea,
    QGridLayout, QLineEdit, QTextEdit, QFileDialog, QMessageBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QSplitter, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QListWidget, QListWidgetItem, QSlider, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, QSize, QPropertyAnimation,
    QEasingCurve, QRect, QUrl
)
from PyQt6.QtGui import (
    QFont, QPixmap, QIcon, QPalette, QColor, QAction,
    QLinearGradient, QPainter, QPen, QBrush
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ApiClient:
    """Enhanced HTTP client for backend API communication with robust error handling.
    
    This class provides a standardized interface for communicating with the AI Recognition
    System backend API. It includes automatic retry logic, comprehensive error handling,
    and connection status monitoring.
    
    Attributes:
        base_url (str): Base URL of the backend API
        session (requests.Session): HTTP session for connection reuse
        timeout (int): Request timeout in seconds
        max_retries (int): Maximum number of retry attempts
        retry_delay (float): Delay between retries in seconds
        is_connected (bool): Current connection status
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        """Initialize the API client with configuration parameters.
        
        Args:
            base_url: The base URL of the backend API endpoint
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1.0
        self.is_connected = False
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'AI-Recognition-System-Frontend/1.0',
            'Accept': 'application/json'
        })
        
        logger.info(f"Initialized API client for {base_url}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic and comprehensive error handling.
        
        This method implements the core request logic with automatic retries,
        connection error handling, and response validation.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments passed to requests
            
        Returns:
            Dict containing the response data or error information
            
        Raises:
            ConnectionError: When unable to connect after max retries
            TimeoutError: When requests timeout after max retries
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = getattr(self.session, method.lower())(url, timeout=self.timeout, **kwargs)
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Update connection status
                self.is_connected = True
                
                # Parse JSON response
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"success": True, "data": response.text}
                
            except requests.exceptions.ConnectionError as e:
                self.is_connected = False
                logger.warning(f"Connection failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                raise ConnectionError(f"Could not connect to backend after {self.max_retries} attempts")
            
            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
                raise TimeoutError(f"Request timed out after {self.max_retries} attempts")
            
            except requests.exceptions.HTTPError as e:
                # Don't retry HTTP errors (4xx, 5xx)
                logger.error(f"HTTP error: {e}")
                response = e.response
                
                try:
                    error_data = response.json()
                    return {
                        "success": False,
                        "error": error_data.get("detail", str(e)),
                        "status_code": response.status_code
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "status_code": response.status_code
                    }
            
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                
                return {
                    "success": False,
                    "error": f"Unexpected error: {str(e)}"
                }
        
        # Should not reach here
        return {"success": False, "error": "Maximum retries exceeded"}
    
    def get(self, endpoint: str) -> Dict[str, Any]:
        """Perform a GET request to the specified API endpoint.
        
        Args:
            endpoint: The API endpoint path
            
        Returns:
            Dict containing the response data
        """
        logger.debug(f"GET request to {endpoint}")
        return self._make_request("GET", endpoint)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, files: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform a POST request to the specified API endpoint.
        
        Args:
            endpoint: The API endpoint path
            data: Optional data to send in the request body
            files: Optional files to upload
            
        Returns:
            Dict containing the response data
        """
        logger.debug(f"POST request to {endpoint}")
        if files:
            return self._make_request("POST", endpoint, data=data, files=files)
        else:
            return self._make_request("POST", endpoint, json=data)
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Perform a DELETE request to the specified API endpoint.
        
        Args:
            endpoint: The API endpoint path
            
        Returns:
            Dict containing the response data
        """
        logger.debug(f"DELETE request to {endpoint}")
        return self._make_request("DELETE", endpoint)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform a PUT request to the specified API endpoint.
        
        Args:
            endpoint: The API endpoint path
            data: Optional data to send in the request body
            
        Returns:
            Dict containing the response data
        """
        logger.debug(f"PUT request to {endpoint}")
        return self._make_request("PUT", endpoint, json=data)
    
    def check_connection(self) -> bool:
        """Check if the backend API is reachable and healthy.
        
        Returns:
            bool: True if backend is healthy, False otherwise
        """
        try:
            response = self.get("/health")
            is_healthy = response.get("status") == "healthy"
            self.is_connected = is_healthy
            return is_healthy
        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            self.is_connected = False
            return False

class StatusThread(QThread):
    """Background thread for monitoring system status.
    
    This thread runs continuously in the background to monitor the health
    and status of the AI recognition system backend. It emits status updates
    that the main UI can use to update indicators and display current state.
    
    Signals:
        status_updated: Emitted when new status data is available
    """
    
    status_updated = pyqtSignal(dict)
    
    def __init__(self, api_client: ApiClient) -> None:
        """Initialize the status monitoring thread.
        
        Args:
            api_client: The API client instance for backend communication
        """
        super().__init__()
        self.api_client = api_client
        self.running = True
        logger.info("Status monitoring thread initialized")
    
    def run(self) -> None:
        """Main monitoring loop that checks system status every 5 seconds.
        
        This method runs in a separate thread and continuously polls the backend
        for status updates. If an error occurs, it waits longer before retrying.
        """
        logger.info("Starting status monitoring loop")
        while self.running:
            try:
                response = self.api_client.get("/api/status")
                self.status_updated.emit(response)
                self.msleep(5000)  # 5 seconds
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                # Emit offline status
                offline_status = {
                    "initialized": False,
                    "system_ready": False,
                    "last_error": str(e)
                }
                self.status_updated.emit(offline_status)
                self.msleep(10000)  # Wait longer on error
    
    def stop(self) -> None:
        """Stop the status monitoring thread gracefully.
        
        This method signals the thread to stop and waits for it to finish.
        """
        logger.info("Stopping status monitoring thread")
        self.running = False
        self.wait(5000)  # Wait up to 5 seconds for thread to finish
        if self.isRunning():
            logger.warning("Status thread did not stop gracefully, terminating")
            self.terminate()

class ModernButton(QPushButton):
    """Custom styled button with modern design and hover effects.
    
    This class provides a consistent, modern button design with customizable
    colors and automatic hover/pressed state styling.
    
    Attributes:
        color (str): The primary color for the button background
    """
    
    def __init__(self, text: str, color: str = "#2196F3", parent=None) -> None:
        """Initialize a modern styled button.
        
        Args:
            text: The button text to display
            color: The primary background color (hex format)
            parent: The parent widget
        """
        super().__init__(text, parent)
        self.color = color
        self.setup_style()
    
    def setup_style(self) -> None:
        """Apply modern styling to the button with state-based colors.
        
        This method sets up the complete button styling including normal,
        hover, pressed, and disabled states with appropriate color variations.
        """
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                min-height: 20px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(self.color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(self.color, 0.3)};
            }}
            QPushButton:disabled {{
                background-color: #6c757d;
                color: #adb5bd;
            }}
        """)
    
    def _darken_color(self, color: str, factor: float = 0.2) -> str:
        """Darken a hexadecimal color by a specified factor.
        
        Args:
            color: Hex color string (e.g., '#2196F3')
            factor: Darkening factor (0.0 = no change, 1.0 = black)
            
        Returns:
            Darkened hex color string
        """
        try:
            color = color.lstrip('#')
            rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            darkened = tuple(max(0, int(c * (1 - factor))) for c in rgb)
            return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid color format '{color}': {e}")
            return "#666666"  # Default fallback color

class StatusBar(QWidget):
    """Custom status bar with system information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.system_status = {}
    
    def setup_ui(self):
        """Setup status bar UI with responsive design."""
        # Make status bar responsive
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(40)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)
        
        # System status indicator
        self.status_label = QLabel("🔴 Disconnected")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #212529;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 15px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
            }
        """)
        
        # API status
        self.api_label = QLabel("API: Offline")
        self.api_label.setStyleSheet("color: #495057; font-size: 12px; font-weight: 500;")
        
        # Recognition status
        self.recognition_label = QLabel("Recognition: Not Ready")
        self.recognition_label.setStyleSheet("color: #495057; font-size: 12px; font-weight: 500;")
        
        # Timestamp
        self.timestamp_label = QLabel()
        self.timestamp_label.setStyleSheet("color: #6c757d; font-size: 11px; font-weight: 400;")
        self.update_timestamp()
        
        # Timer for timestamp updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timestamp)
        self.timer.start(1000)
        
        layout.addWidget(self.status_label)
        separator1 = QLabel("|")
        separator1.setStyleSheet("color: #adb5bd; font-weight: 300;")
        layout.addWidget(separator1)
        layout.addWidget(self.api_label)
        separator2 = QLabel("|")
        separator2.setStyleSheet("color: #adb5bd; font-weight: 300;")
        layout.addWidget(separator2)
        layout.addWidget(self.recognition_label)
        layout.addStretch()
        layout.addWidget(self.timestamp_label)
    
    def update_status(self, status: Dict[str, Any]):
        """Update status indicators"""
        self.system_status = status
        
        if status.get("initialized", False):
            if status.get("system_ready", False):
                self.status_label.setText("🟢 Online")
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #155724;
                        font-weight: bold;
                        padding: 5px 10px;
                        border-radius: 15px;
                        background-color: #d4edda;
                        border: 1px solid #c3e6cb;
                    }
                """)
                self.recognition_label.setText("Recognition: Ready")
                self.recognition_label.setStyleSheet("color: #155724; font-size: 12px; font-weight: 600;")
            else:
                self.status_label.setText("🟡 Initializing")
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #856404;
                        font-weight: bold;
                        padding: 5px 10px;
                        border-radius: 15px;
                        background-color: #fff3cd;
                        border: 1px solid #ffeaa7;
                    }
                """)
                self.recognition_label.setText("Recognition: Initializing")
                self.recognition_label.setStyleSheet("color: #856404; font-size: 12px; font-weight: 600;")
        else:
            self.status_label.setText("🔴 Offline")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #721c24;
                    font-weight: bold;
                    padding: 5px 10px;
                    border-radius: 15px;
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                }
            """)
            self.recognition_label.setText("Recognition: Offline")
            self.recognition_label.setStyleSheet("color: #721c24; font-size: 12px; font-weight: 600;")
        
        self.api_label.setText("API: Online")
        self.api_label.setStyleSheet("color: #155724; font-size: 12px; font-weight: 600;")
    
    def update_timestamp(self):
        """Update timestamp display"""
        now = datetime.now().strftime("%H:%M:%S")
        self.timestamp_label.setText(f"Last update: {now}")

class NavigationSidebar(QWidget):
    """Modern navigation sidebar"""
    
    page_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_page = "dashboard"
    
    def setup_ui(self):
        """Setup navigation sidebar UI with responsive design."""
        # Set responsive width constraints
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self.setFixedWidth(250)  # Default width
        
        # Set size policy for responsive behavior
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #212529;
                color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Ensure layout stretches to fill available space
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        
        # Logo/Title
        title_label = QLabel("AI Recognition\nSystem")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #007bff;
                padding: 20px;
                border-bottom: 1px solid #495057;
                background-color: #212529;
            }
        """)
        layout.addWidget(title_label)
        
        # Navigation buttons
        nav_items = [
            ("📊", "Dashboard", "dashboard"),
            ("📦", "Items", "items"),
            ("🔍", "Recognition", "recognition"),
            ("🏋️", "Training", "training"),
            ("📈", "Evaluation", "evaluation"),
            ("⚙️", "Settings", "settings"),
            ("📋", "Logs", "logs")
        ]
        
        for icon, label, page_id in nav_items:
            btn = self.create_nav_button(icon, label, page_id)
            layout.addWidget(btn)
        
        # Training indicator in sidebar
        self.sidebar_training_indicator = QLabel("⚠️ Training Required")
        self.sidebar_training_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_training_indicator.setStyleSheet("""
            QLabel {
                background-color: #ffc107;
                color: #212529;
                padding: 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                margin: 10px;
            }
        """)
        self.sidebar_training_indicator.hide()  # Initially hidden
        layout.addWidget(self.sidebar_training_indicator)
        
        layout.addStretch()
        
        # System info at bottom
        info_label = QLabel("v1.0.0\nPowered by PyQt6")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 10px;
                padding: 10px;
                border-top: 1px solid #495057;
                background-color: #212529;
            }
        """)
        layout.addWidget(info_label)
    
    def create_nav_button(self, icon: str, label: str, page_id: str) -> QPushButton:
        """Create a navigation button"""
        btn = QPushButton(f"{icon} {label}")
        btn.setObjectName(page_id)
        btn.clicked.connect(lambda: self.select_page(page_id))
        
        btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 15px 20px;
                border: none;
                background-color: transparent;
                color: #adb5bd;
                font-size: 14px;
                border-radius: 0;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #495057;
                color: #f8f9fa;
            }
            QPushButton[selected="true"] {
                background-color: #007bff;
                color: white;
                border-left: 4px solid #0056b3;
                font-weight: 600;
            }
        """)
        
        return btn
    
    def select_page(self, page_id: str):
        """Select a navigation page"""
        # Update button states
        for btn in self.findChildren(QPushButton):
            if btn.objectName():
                btn.setProperty("selected", btn.objectName() == page_id)
                btn.style().polish(btn)
        
        self.current_page = page_id
        self.page_changed.emit(page_id)
    
    def update_training_indicator(self, training_required: bool, interrupted: bool = False):
        """Update the sidebar training indicator"""
        if training_required or interrupted:
            self.sidebar_training_indicator.show()
            if interrupted:
                self.sidebar_training_indicator.setText("🚨 Training Interrupted")
                self.sidebar_training_indicator.setStyleSheet("""
                    QLabel {
                        background-color: #dc3545;
                        color: white;
                        padding: 8px;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: bold;
                        margin: 10px;
                    }
                """)
            else:
                self.sidebar_training_indicator.setText("⚠️ Training Required")
                self.sidebar_training_indicator.setStyleSheet("""
                    QLabel {
                        background-color: #ffc107;
                        color: #212529;
                        padding: 8px;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: bold;
                        margin: 10px;
                    }
                """)
        else:
            self.sidebar_training_indicator.hide()

class DashboardWidget(QWidget):
    """Main dashboard widget with real-time metrics and functional controls."""
    
    def __init__(self, api_client: ApiClient, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.metrics_timer = QTimer()
        self.setup_ui()
        self.setup_metrics_timer()
        self.load_initial_data()
    
    def setup_ui(self):
        """Setup dashboard UI with responsive design."""
        # Make dashboard responsive
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Main scroll area for responsive content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Content widget inside scroll area
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Set up scroll area
        scroll_area.setWidget(content_widget)
        
        # Main layout for the dashboard
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        
        # Page title with training indicator
        title_layout = QHBoxLayout()
        
        title = QLabel("Dashboard")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #212529;
                margin-bottom: 10px;
            }
        """)
        title_layout.addWidget(title)
        
        # Training required indicator
        self.training_indicator = QLabel("🚨 Training Required")
        self.training_indicator.setStyleSheet("""
            QLabel {
                background-color: #dc3545;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
                border: 2px solid #c82333;
            }
        """)
        self.training_indicator.hide()  # Initially hidden
        title_layout.addWidget(self.training_indicator)
        
        title_layout.addStretch()
        
        title_widget = QWidget()
        title_widget.setLayout(title_layout)
        layout.addWidget(title_widget)
        
        # Statistics cards with enhanced responsive grid
        stats_widget = QWidget()
        stats_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        stats_widget.setStyleSheet("""
            QWidget {
                background: transparent;
                margin: 10px 0;
            }
        """)
        stats_layout = QGridLayout(stats_widget)
        stats_layout.setSpacing(20)  # Increased spacing for better visual separation
        stats_layout.setContentsMargins(0, 10, 0, 20)  # Add top and bottom margins
        
        # Make grid responsive with equal column widths
        stats_layout.setColumnStretch(0, 1)
        stats_layout.setColumnStretch(1, 1)
        stats_layout.setColumnStretch(2, 1)
        stats_layout.setColumnStretch(3, 1)
        
        # Create stat cards
        self.items_card = self.create_stat_card("Total Items", "0", "#4CAF50", "📦")
        self.accuracy_card = self.create_stat_card("Accuracy", "0%", "#2196F3", "🎯")
        self.speed_card = self.create_stat_card("Avg Speed", "0ms", "#FF9800", "⚡")
        self.status_card = self.create_stat_card("System Status", "Offline", "#F44336", "🔴")
        
        stats_layout.addWidget(self.items_card, 0, 0)
        stats_layout.addWidget(self.accuracy_card, 0, 1)
        stats_layout.addWidget(self.speed_card, 0, 2)
        stats_layout.addWidget(self.status_card, 0, 3)
        
        layout.addWidget(stats_widget)
        
        # Recent activity section with enhanced design
        activity_group = QGroupBox("📊 Recent Activity")
        activity_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        activity_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #e8eaf6;
                border-radius: 12px;
                margin-top: 15px;
                padding-top: 20px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f8f9ff);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 15px;
                color: #3f51b5;
                background-color: #ffffff;
                border-radius: 6px;
                font-weight: 700;
            }
        """)
        
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.setContentsMargins(15, 10, 15, 15)
        
        self.activity_list = QListWidget()
        self.activity_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.activity_list.setMinimumHeight(160)
        self.activity_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e8eaf6;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #fafbff);
                border-radius: 8px;
                color: #2c3e50;
                padding: 5px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-bottom: 1px solid #e8eaf6;
                color: #34495e;
                border-radius: 4px;
                margin: 2px 0;
            }
            QListWidget::item:hover {
                background-color: #e8eaf6;
                color: #2c3e50;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3f51b5, stop:1 #5c6bc0);
                color: white;
                font-weight: 600;
            }
        """)
        activity_layout.addWidget(self.activity_list)
        
        layout.addWidget(activity_group)
        
        # Quick actions with responsive layout
        actions_group = QGroupBox("⚡ Quick Actions")
        actions_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #e8f5e8;
                border-radius: 12px;
                margin-top: 15px;
                padding-top: 20px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f8fff8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 15px;
                color: #4caf50;
                background-color: #ffffff;
                border-radius: 6px;
                font-weight: 700;
            }
        """)
        actions_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Use responsive layout that wraps on small screens
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setSpacing(15)
        actions_layout.setContentsMargins(15, 10, 15, 15)
        
        recognize_btn = ModernButton("🔍 Quick Recognition", "#2196F3")
        add_item_btn = ModernButton("📦 Add New Item", "#4CAF50")
        train_btn = ModernButton("🏋️ Start Training", "#FF9800")
        evaluate_btn = ModernButton("📈 Run Evaluation", "#9C27B0")
        
        # Connect buttons to functionality
        recognize_btn.clicked.connect(self.quick_recognition)
        add_item_btn.clicked.connect(self.add_new_item)
        train_btn.clicked.connect(self.start_training)
        evaluate_btn.clicked.connect(self.run_evaluation)
        
        # Make buttons responsive
        for btn in [recognize_btn, add_item_btn, train_btn, evaluate_btn]:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setMinimumHeight(50)
        
        actions_layout.addWidget(recognize_btn)
        actions_layout.addWidget(add_item_btn)
        actions_layout.addWidget(train_btn)
        actions_layout.addWidget(evaluate_btn)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()
    
    def create_stat_card(self, title: str, value: str, color: str, icon: str) -> QWidget:
        """
        Create a beautiful, high-contrast statistics card widget.
        
        Features:
        - Gradient background with theme colors
        - High contrast text for better visibility
        - Modern shadow effects
        - Responsive hover animations
        - Accessible color combinations
        """
        card = QFrame()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card.setMinimumHeight(140)
        card.setMaximumHeight(170)
        
        # Create gradient color variations for better contrast
        color_variants = {
            "#4CAF50": {"bg": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4CAF50, stop:1 #45a049)", "light": "#e8f5e8", "dark": "#2e7d32"},
            "#2196F3": {"bg": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2196F3, stop:1 #1976d2)", "light": "#e3f2fd", "dark": "#1565c0"},
            "#FF9800": {"bg": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF9800, stop:1 #f57c00)", "light": "#fff3e0", "dark": "#e65100"},
            "#F44336": {"bg": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F44336, stop:1 #d32f2f)", "light": "#ffebee", "dark": "#c62828"}
        }
        
        variant = color_variants.get(color, {"bg": color, "light": "#f8f9fa", "dark": color})
        
        card.setStyleSheet(f"""
            QFrame {{
                background: {variant['bg']};
                border: none;
                border-radius: 12px;
                padding: 20px;
                color: white;
                min-height: 140px;
                max-height: 170px;
                /* Add subtle shadow effect */
                border: 1px solid rgba(0, 0, 0, 0.1);
            }}
            QFrame:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {color}, stop:0.7 {variant['dark']}, stop:1 {color});
                transform: scale(1.02);
                border: 2px solid rgba(255, 255, 255, 0.3);
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Value with high contrast and larger size
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: white;
                margin-bottom: 8px;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.4);
                background: transparent;
            }
        """)
        
        # Title with icon integrated
        title_label = QLabel(f"{icon} {title}")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: rgba(255, 255, 255, 0.95);
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                background: transparent;
            }
        """)
        
        layout.addWidget(value_label)
        layout.addWidget(title_label)
        
        # Store references for updates
        card.value_label = value_label
        card.title_text = title
        card.color = color
        
        return card
    
    def setup_metrics_timer(self) -> None:
        """Setup timer for periodic metrics updates."""
        self.metrics_timer.timeout.connect(self.refresh_metrics)
        self.metrics_timer.start(10000)  # Update every 10 seconds
    
    def load_initial_data(self) -> None:
        """Load initial dashboard data."""
        self.refresh_metrics()
        self.load_recent_activity()
        self.check_training_status()
    
    def refresh_metrics(self) -> None:
        """Refresh dashboard metrics from API."""
        try:
            # Get items count
            items_response = self.api_client.get("/api/items")
            items_count = 0
            if items_response.get("success"):
                items_data = items_response.get("data", [])
                items_count = len(items_data)
            
            # Get evaluation results for accuracy and speed
            eval_response = self.api_client.get("/api/evaluate/results")
            accuracy = "N/A"
            avg_speed = "N/A"
            
            if eval_response.get("success"):
                eval_data = eval_response.get("data", {})
                summary = eval_data.get("evaluation_summary", {})
                accuracy = f"{summary.get('overall_accuracy', 0) * 100:.1f}%"
                avg_time = summary.get('avg_inference_time', 0)
                avg_speed = f"{avg_time * 1000:.0f}ms" if avg_time else "N/A"
            
            # Get system status
            status_response = self.api_client.get("/api/status")
            system_status = "Offline"
            if status_response.get("initialized", False):
                if status_response.get("system_ready", False):
                    system_status = "Online"
                else:
                    system_status = "Initializing"
            
            # Update stat cards
            self.update_stat_card(self.items_card, str(items_count))
            self.update_stat_card(self.accuracy_card, accuracy)
            self.update_stat_card(self.speed_card, avg_speed)
            self.update_stat_card(self.status_card, system_status)
            
            # Check training status periodically
            self.check_training_status()
            
        except Exception as e:
            logger.error(f"Failed to refresh metrics: {e}")
            # Show error state
            self.update_stat_card(self.items_card, "Error")
            self.update_stat_card(self.accuracy_card, "Error")
            self.update_stat_card(self.speed_card, "Error")
            self.update_stat_card(self.status_card, "Error")
    
    def update_stat_card(self, card: QWidget, value: str) -> None:
        """Update a statistics card with new value."""
        if hasattr(card, 'value_label'):
            card.value_label.setText(value)
    
    def check_training_status(self) -> None:
        """Check if training is required and update indicator."""
        try:
            response = self.api_client.get("/api/training/status")
            if response.get("success"):
                data = response.get("data", {})
                training_required = data.get("training_required", False)
                interrupted_training = data.get("interrupted_training", False)
                
                if training_required or interrupted_training:
                    self.training_indicator.show()
                    if interrupted_training:
                        self.training_indicator.setText("🚨 Training Interrupted - Action Required")
                        self.training_indicator.setStyleSheet("""
                            QLabel {
                                background-color: #dc3545;
                                color: white;
                                padding: 8px 12px;
                                border-radius: 6px;
                                font-size: 12px;
                                font-weight: bold;
                                border: 2px solid #c82333;
                                animation: blink 1s infinite;
                            }
                        """)
                    else:
                        self.training_indicator.setText("⚠️ Training Required")
                        self.training_indicator.setStyleSheet("""
                            QLabel {
                                background-color: #ffc107;
                                color: #212529;
                                padding: 8px 12px;
                                border-radius: 6px;
                                font-size: 12px;
                                font-weight: bold;
                                border: 2px solid #e0a800;
                            }
                        """)
                else:
                    self.training_indicator.hide()
                    
        except Exception as e:
            logger.error(f"Failed to check training status: {e}")
            # Hide indicator on error
            self.training_indicator.hide()
    
    def load_recent_activity(self) -> None:
        """Load recent system activity."""
        try:
            # Add some sample activities - in real app this would come from logs
            activities = [
                "🔍 Image recognition completed: item_003 (confidence: 100%)",
                "📦 New item added: item_003", 
                "🏋️ Training completed successfully",
                "📊 System evaluation completed (100% accuracy)",
                "🚀 System started successfully"
            ]
            
            self.activity_list.clear()
            for activity in activities:
                item = QListWidgetItem(activity)
                self.activity_list.addItem(item)
                
        except Exception as e:
            logger.error(f"Failed to load recent activity: {e}")
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update dashboard statistics from external source."""
        # This method can be called by parent to update stats
        self.refresh_metrics()
    
    def quick_recognition(self) -> None:
        """Navigate to recognition page for quick image recognition."""
        self._navigate_to_page('recognition')
    
    def add_new_item(self) -> None:
        """Navigate to items page to add a new item."""
        self._navigate_to_page('items')
    
    def start_training(self) -> None:
        """Navigate to training page to start model training."""
        self._navigate_to_page('training')
    
    def run_evaluation(self) -> None:
        """Navigate to evaluation page to run system evaluation."""
        self._navigate_to_page('evaluation')
    
    def _navigate_to_page(self, page_id: str) -> None:
        """Helper method to navigate to a specific page."""
        # Find the main window by traversing up the parent hierarchy
        widget = self
        while widget.parent():
            widget = widget.parent()
            if hasattr(widget, 'change_page'):
                widget.change_page(page_id)
                return
        
        logger.warning(f"Could not navigate to page: {page_id}")

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.api_client = ApiClient()
        self.status_thread = None
        self.setup_ui()
        self.setup_connections()
        self.start_status_monitoring()
        
        # Try to connect to backend
        self.check_backend_connection()
    
    def setup_ui(self):
        """Setup main window UI with responsive design."""
        self.setWindowTitle("AI Recognition System")
        
        # Make window fully resizable
        self.setMinimumSize(900, 600)  # Minimum usable size
        self.resize(1400, 900)  # Default size
        
        # Enable window resizing in all directions
        self.setMinimumWidth(900)
        self.setMinimumHeight(600)
        
        # Set size policy for responsive behavior
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set application icon
        self.setWindowIcon(QIcon("frontend/assets/icon.png"))
        
        # Central widget with responsive splitter
        central_widget = QWidget()
        central_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCentralWidget(central_widget)
        
        # Use splitter for responsive design
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Navigation sidebar with responsive design
        self.sidebar = NavigationSidebar()
        self.sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        # Main content area with responsive layout
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                color: #212529;
            }
        """)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Main content stack with responsive behavior
        self.content_stack = QStackedWidget()
        self.content_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create page widgets
        self.dashboard = DashboardWidget(self.api_client)
        
        # Import and create other widgets
        from widgets.items import ItemsWidget
        from widgets.recognition import RecognitionWidget
        from widgets.training import TrainingWidget
        from widgets.evaluation import EvaluationWidget
        from widgets.settings import SettingsWidget
        from widgets.logs import LogsWidget
        
        self.items_page = ItemsWidget(self.api_client)
        self.recognition_page = RecognitionWidget(self.api_client)
        self.training_page = TrainingWidget(self.api_client)
        self.evaluation_page = EvaluationWidget(self.api_client)
        self.settings_page = SettingsWidget(self.api_client)
        self.logs_page = LogsWidget(self.api_client)
        
        # Add all pages to stack
        self.content_stack.addWidget(self.dashboard)
        self.content_stack.addWidget(self.items_page)
        self.content_stack.addWidget(self.recognition_page)
        self.content_stack.addWidget(self.training_page)
        self.content_stack.addWidget(self.evaluation_page)
        self.content_stack.addWidget(self.settings_page)
        self.content_stack.addWidget(self.logs_page)
        
        # Create page mapping
        self.page_mapping = {
            "dashboard": 0,
            "items": 1,
            "recognition": 2,
            "training": 3,
            "evaluation": 4,
            "settings": 5,
            "logs": 6
        }
        
        content_layout.addWidget(self.content_stack)
        
        # Status bar
        self.status_bar = StatusBar()
        content_layout.addWidget(self.status_bar)
        
        # Store splitter reference for responsive behavior
        self.main_splitter = main_splitter
        
        # Add widgets to splitter for responsive behavior
        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(content_widget)
        
        # Set initial proportions (sidebar: 250px, content: rest)
        self.main_splitter.setSizes([250, 1150])
        self.main_splitter.setCollapsible(0, False)  # Don't collapse sidebar
        self.main_splitter.setCollapsible(1, False)  # Don't collapse content
        
        # Set minimum sizes for responsive behavior
        self.main_splitter.setStretchFactor(0, 0)  # Sidebar doesn't stretch
        self.main_splitter.setStretchFactor(1, 1)  # Content area stretches
        
        layout.addWidget(self.main_splitter)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.sidebar.page_changed.connect(self.change_page)
    
    def start_status_monitoring(self):
        """Start background status monitoring"""
        self.status_thread = StatusThread(self.api_client)
        self.status_thread.status_updated.connect(self.update_system_status)
        self.status_thread.start()
    
    def change_page(self, page_id: str):
        """Change the current page"""
        if page_id in self.page_mapping:
            page_index = self.page_mapping[page_id]
            self.content_stack.setCurrentIndex(page_index)
            logger.info(f"Changed to page: {page_id}")
        else:
            logger.warning(f"Unknown page: {page_id}")
    
    def update_system_status(self, status_response: Dict[str, Any]):
        """Update system status from API response"""
        try:
            if status_response.get("success", False):
                # This is from health check
                self.status_bar.update_status(status_response.get("system_status", {}))
            else:
                # This is direct status response
                self.status_bar.update_status(status_response)
            
            # Update training indicators
            self.update_training_indicators()
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def update_training_indicators(self):
        """Update training indicators in sidebar and dashboard"""
        try:
            response = self.api_client.get("/api/training/status")
            if response.get("success"):
                data = response.get("data", {})
                training_required = data.get("training_required", False)
                interrupted_training = data.get("interrupted_training", False)
                
                # Update sidebar indicator
                self.sidebar.update_training_indicator(training_required, interrupted_training)
                
                # Update dashboard indicator if dashboard is available
                if hasattr(self.dashboard, 'check_training_status'):
                    # Dashboard will update its own indicator through check_training_status
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to update training indicators: {e}")
    
    def check_backend_connection(self):
        """Check if backend is running"""
        try:
            response = self.api_client.get("/health")
            if response.get("success", True) and response.get("status") == "healthy":
                self.show_message("Backend Connected", "Successfully connected to AI Recognition System backend.")
                logger.info("Backend connection established successfully")
            else:
                error_msg = response.get("error", "Backend is running but not healthy")
                self.show_warning("Backend Issues", f"Backend connection issues: {error_msg}")
                logger.warning(f"Backend health check failed: {error_msg}")
        except ConnectionError as e:
            self.show_error("Backend Connection Failed", 
                          f"Could not connect to the backend server.\n\n"
                          f"Please ensure the backend is running on {self.api_client.base_url}\n\n"
                          f"Error: {str(e)}")
            logger.error(f"Backend connection failed: {e}")
        except Exception as e:
            self.show_error("Unexpected Error", 
                          f"An unexpected error occurred while connecting to the backend:\n\n{str(e)}")
            logger.error(f"Unexpected connection error: {e}")
    
    def show_message(self, title: str, message: str):
        """Show an information message"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()
    
    def show_warning(self, title: str, message: str):
        """Show a warning message"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()
    
    def show_error(self, title: str, message: str):
        """Show an error message"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()
    
    def resizeEvent(self, event) -> None:
        """Handle window resize events for responsive behavior."""
        super().resizeEvent(event)
        
        # Adjust sidebar visibility based on window size
        if hasattr(self, 'main_splitter') and hasattr(self, 'sidebar'):
            window_width = self.width()
            
            # On very small screens, consider collapsing sidebar or adjusting layout
            if window_width < 1000:
                # Reduce sidebar width on smaller screens
                current_sizes = self.main_splitter.sizes()
                if len(current_sizes) >= 2:
                    sidebar_width = min(200, current_sizes[0])
                    content_width = window_width - sidebar_width - 10
                    self.main_splitter.setSizes([sidebar_width, content_width])
            
            # Ensure minimum usable space for content
            current_sizes = self.main_splitter.sizes()
            if len(current_sizes) >= 2 and current_sizes[1] < 500:
                self.main_splitter.setSizes([180, window_width - 190])
    
    def closeEvent(self, event):
        """Handle application closing with proper cleanup."""
        logger.info("Shutting down application...")
        
        # Stop status monitoring thread
        if self.status_thread:
            logger.info("Stopping status monitoring thread...")
            self.status_thread.stop()
        
        # Clean up any other resources
        if hasattr(self, 'dashboard') and hasattr(self.dashboard, 'metrics_timer'):
            self.dashboard.metrics_timer.stop()
        
        logger.info("Application shutdown complete")
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI Recognition System")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AI Recognition Corp")
    
    # Apply modern styling with better text visibility
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f8f9fa;
            color: #212529;
        }
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #212529;
            background-color: transparent;
        }
        QLabel {
            color: #212529;
        }
        QScrollArea {
            border: none;
            background-color: white;
        }
        QGroupBox {
            background-color: white;
            color: #212529;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            font-weight: bold;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 10px;
            color: #495057;
            background-color: white;
        }
        QTableWidget {
            color: #212529;
            background-color: white;
            alternate-background-color: #f8f9fa;
            selection-background-color: #007bff;
            selection-color: white;
        }
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #dee2e6;
        }
        QTabWidget QWidget {
            color: #212529;
            background-color: white;
        }
        QTextEdit {
            color: #212529;
            background-color: white;
            border: 1px solid #ced4da;
        }
        QLineEdit {
            color: #212529;
            background-color: white;
            border: 1px solid #ced4da;
            padding: 8px;
            border-radius: 4px;
        }
        QComboBox {
            color: #212529;
            background-color: white;
            border: 1px solid #ced4da;
            padding: 8px;
            border-radius: 4px;
        }
    """)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()