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
    """Enhanced HTTP client for backend API communication with robust error handling"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1.0
        self.is_connected = False
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'AI-Recognition-System-Frontend/1.0',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic and error handling"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = getattr(self.session, method.lower())(url, **kwargs)
                
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
        """GET request to API"""
        return self._make_request("GET", endpoint)
    
    def post(self, endpoint: str, data: Dict[str, Any] = None, files: Dict = None) -> Dict[str, Any]:
        """POST request to API"""
        if files:
            return self._make_request("POST", endpoint, data=data, files=files)
        else:
            return self._make_request("POST", endpoint, json=data)
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """DELETE request to API"""
        return self._make_request("DELETE", endpoint)
    
    def put(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """PUT request to API"""
        return self._make_request("PUT", endpoint, json=data)
    
    def check_connection(self) -> bool:
        """Check if backend is reachable"""
        try:
            response = self.get("/health")
            return response.get("status") == "healthy"
        except:
            return False

class StatusThread(QThread):
    """Background thread for monitoring system status"""
    
    status_updated = pyqtSignal(dict)
    
    def __init__(self, api_client: ApiClient):
        super().__init__()
        self.api_client = api_client
        self.running = True
    
    def run(self):
        """Monitor system status every 5 seconds"""
        while self.running:
            try:
                response = self.api_client.get("/api/status")
                self.status_updated.emit(response)
                self.msleep(5000)  # 5 seconds
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                self.msleep(10000)  # Wait longer on error
    
    def stop(self):
        """Stop the status monitoring thread"""
        self.running = False
        self.wait()

class ModernButton(QPushButton):
    """Custom styled button with hover effects"""
    
    def __init__(self, text: str, color: str = "#2196F3", parent=None):
        super().__init__(text, parent)
        self.color = color
        self.setup_style()
    
    def setup_style(self):
        """Setup modern button styling"""
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
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(self.color)};
                transform: translateY(-2px);
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(self.color, 0.3)};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #888888;
            }}
        """)
    
    def _darken_color(self, color: str, factor: float = 0.2) -> str:
        """Darken a hex color by a factor"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(int(c * (1 - factor)) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"

class StatusBar(QWidget):
    """Custom status bar with system information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.system_status = {}
    
    def setup_ui(self):
        """Setup status bar UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # System status indicator
        self.status_label = QLabel("🔴 Disconnected")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 15px;
                background-color: #f0f0f0;
            }
        """)
        
        # API status
        self.api_label = QLabel("API: Offline")
        self.api_label.setStyleSheet("color: #666; font-size: 12px;")
        
        # Recognition status
        self.recognition_label = QLabel("Recognition: Not Ready")
        self.recognition_label.setStyleSheet("color: #666; font-size: 12px;")
        
        # Timestamp
        self.timestamp_label = QLabel()
        self.timestamp_label.setStyleSheet("color: #999; font-size: 11px;")
        self.update_timestamp()
        
        # Timer for timestamp updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timestamp)
        self.timer.start(1000)
        
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.api_label)
        layout.addWidget(QLabel("|"))
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
                        color: #4CAF50;
                        font-weight: bold;
                        padding: 5px 10px;
                        border-radius: 15px;
                        background-color: #E8F5E8;
                    }
                """)
                self.recognition_label.setText("Recognition: Ready")
                self.recognition_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
            else:
                self.status_label.setText("🟡 Initializing")
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #FF9800;
                        font-weight: bold;
                        padding: 5px 10px;
                        border-radius: 15px;
                        background-color: #FFF3E0;
                    }
                """)
                self.recognition_label.setText("Recognition: Initializing")
                self.recognition_label.setStyleSheet("color: #FF9800; font-size: 12px;")
        else:
            self.status_label.setText("🔴 Offline")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #F44336;
                    font-weight: bold;
                    padding: 5px 10px;
                    border-radius: 15px;
                    background-color: #FFEBEE;
                }
            """)
            self.recognition_label.setText("Recognition: Offline")
            self.recognition_label.setStyleSheet("color: #F44336; font-size: 12px;")
        
        self.api_label.setText("API: Online")
        self.api_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
    
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
        """Setup navigation sidebar UI"""
        self.setFixedWidth(250)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: white;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Logo/Title
        title_label = QLabel("AI Recognition\nSystem")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2196F3;
                padding: 20px;
                border-bottom: 1px solid #333;
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
        
        layout.addStretch()
        
        # System info at bottom
        info_label = QLabel("v1.0.0\nPowered by PyQt6")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 10px;
                padding: 10px;
                border-top: 1px solid #333;
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
                color: #ccc;
                font-size: 14px;
                border-radius: 0;
            }
            QPushButton:hover {
                background-color: #333;
                color: white;
            }
            QPushButton[selected="true"] {
                background-color: #2196F3;
                color: white;
                border-left: 4px solid #1976D2;
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

class DashboardWidget(QWidget):
    """Main dashboard widget"""
    
    def __init__(self, api_client: ApiClient, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.setup_ui()
    
    def setup_ui(self):
        """Setup dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Page title
        title = QLabel("Dashboard")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Statistics cards
        stats_layout = QGridLayout()
        stats_layout.setSpacing(15)
        
        # Create stat cards
        self.items_card = self.create_stat_card("Total Items", "0", "#4CAF50", "📦")
        self.accuracy_card = self.create_stat_card("Accuracy", "0%", "#2196F3", "🎯")
        self.speed_card = self.create_stat_card("Avg Speed", "0ms", "#FF9800", "⚡")
        self.status_card = self.create_stat_card("System Status", "Offline", "#F44336", "🔴")
        
        stats_layout.addWidget(self.items_card, 0, 0)
        stats_layout.addWidget(self.accuracy_card, 0, 1)
        stats_layout.addWidget(self.speed_card, 0, 2)
        stats_layout.addWidget(self.status_card, 0, 3)
        
        layout.addLayout(stats_layout)
        
        # Recent activity section
        activity_group = QGroupBox("Recent Activity")
        activity_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        
        activity_layout = QVBoxLayout(activity_group)
        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet("""
            QListWidget {
                border: none;
                background-color: #f9f9f9;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
        """)
        activity_layout.addWidget(self.activity_list)
        
        layout.addWidget(activity_group)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_group.setStyleSheet(activity_group.styleSheet())
        
        actions_layout = QHBoxLayout(actions_group)
        
        recognize_btn = ModernButton("🔍 Quick Recognition", "#2196F3")
        add_item_btn = ModernButton("📦 Add New Item", "#4CAF50")
        train_btn = ModernButton("🏋️ Start Training", "#FF9800")
        evaluate_btn = ModernButton("📈 Run Evaluation", "#9C27B0")
        
        actions_layout.addWidget(recognize_btn)
        actions_layout.addWidget(add_item_btn)
        actions_layout.addWidget(train_btn)
        actions_layout.addWidget(evaluate_btn)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()
    
    def create_stat_card(self, title: str, value: str, color: str, icon: str) -> QWidget:
        """Create a statistics card widget"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
            }}
            QFrame:hover {{
                border-color: {color};
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon and value
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 24px; margin-bottom: 5px;")
        
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                font-weight: bold;
                color: {color};
                margin-bottom: 5px;
            }}
        """)
        
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                font-weight: normal;
            }
        """)
        
        layout.addWidget(icon_label)
        layout.addWidget(value_label)
        layout.addWidget(title_label)
        
        # Store references for updates
        card.value_label = value_label
        card.title_text = title
        
        return card
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update dashboard statistics"""
        # Update stat cards based on received data
        pass

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
        """Setup main window UI"""
        self.setWindowTitle("AI Recognition System")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Set application icon
        self.setWindowIcon(QIcon("frontend/assets/icon.png"))
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Navigation sidebar
        self.sidebar = NavigationSidebar()
        
        # Main content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Main content stack
        self.content_stack = QStackedWidget()
        
        # Create page widgets
        self.dashboard = DashboardWidget(self.api_client)
        
        # Import and create other widgets
        from frontend.widgets.items import ItemsWidget
        from frontend.widgets.recognition import RecognitionWidget
        from frontend.widgets.training import TrainingWidget
        from frontend.widgets.evaluation import EvaluationWidget
        from frontend.widgets.settings import SettingsWidget
        from frontend.widgets.logs import LogsWidget
        
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
        
        layout.addWidget(self.sidebar)
        layout.addWidget(content_widget)
        
        # Set proportions
        layout.setStretch(0, 0)  # Sidebar fixed width
        layout.setStretch(1, 1)  # Content area flexible
    
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
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
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
    
    def closeEvent(self, event):
        """Handle application closing"""
        if self.status_thread:
            self.status_thread.stop()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI Recognition System")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AI Recognition Corp")
    
    # Apply modern styling
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QScrollArea {
            border: none;
        }
        QGroupBox {
            background-color: white;
        }
    """)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()