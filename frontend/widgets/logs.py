"""
Logs Viewer Widget
Real-time log monitoring and analysis interface
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QGroupBox,
    QTextEdit, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QTabWidget, QSplitter, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor

logger = logging.getLogger(__name__)

class LogMonitorThread(QThread):
    """Background thread for monitoring log files"""
    
    new_log_entry = pyqtSignal(str, str, str)  # timestamp, level, message
    
    def __init__(self, log_files: List[str]):
        super().__init__()
        self.log_files = log_files
        self.running = False
        self.file_positions = {}
        
        # Initialize file positions
        for log_file in log_files:
            if Path(log_file).exists():
                with open(log_file, 'r') as f:
                    f.seek(0, 2)  # Seek to end
                    self.file_positions[log_file] = f.tell()
            else:
                self.file_positions[log_file] = 0
    
    def start_monitoring(self):
        """Start monitoring log files"""
        self.running = True
        self.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.wait()
    
    def run(self):
        """Monitor log files for new entries"""
        while self.running:
            try:
                for log_file in self.log_files:
                    if not Path(log_file).exists():
                        continue
                    
                    with open(log_file, 'r') as f:
                        # Seek to last known position
                        current_pos = self.file_positions.get(log_file, 0)
                        f.seek(current_pos)
                        
                        # Read new lines
                        new_lines = f.readlines()
                        
                        # Update position
                        self.file_positions[log_file] = f.tell()
                        
                        # Process new log entries
                        for line in new_lines:
                            self.parse_log_line(line.strip(), log_file)
                
                self.msleep(1000)  # Check every second
            
            except Exception as e:
                logger.error(f"Log monitoring error: {e}")
                self.msleep(5000)
    
    def parse_log_line(self, line: str, source_file: str):
        """Parse a log line and extract components"""
        if not line.strip():
            return
        
        # Try to parse standard log format: timestamp - name - level - message
        log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.+?) - (\w+) - (.+)'
        match = re.match(log_pattern, line)
        
        if match:
            timestamp = match.group(1)
            level = match.group(3)
            message = match.group(4)
        else:
            # Fallback for non-standard format
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level = "INFO"
            message = line
        
        self.new_log_entry.emit(timestamp, level, message)

class LogFilterWidget(QFrame):
    """Widget for filtering log entries"""
    
    filter_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup log filter UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Log Filters")
        title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Filter controls
        form_layout = QFormLayout()
        
        # Log level filter
        self.level_combo = QComboBox()
        self.level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_combo.setCurrentText("ALL")
        self.level_combo.currentTextChanged.connect(self.emit_filter_change)
        form_layout.addRow("Level:", self.level_combo)
        
        # Search filter
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search in messages...")
        self.search_edit.textChanged.connect(self.emit_filter_change)
        form_layout.addRow("Search:", self.search_edit)
        
        # Time range filter
        self.time_combo = QComboBox()
        self.time_combo.addItems([
            "All Time", "Last Hour", "Last 6 Hours", 
            "Last 24 Hours", "Last 7 Days"
        ])
        self.time_combo.currentTextChanged.connect(self.emit_filter_change)
        form_layout.addRow("Time Range:", self.time_combo)
        
        # Component filter
        self.component_combo = QComboBox()
        self.component_combo.addItems([
            "All Components", "Backend", "Frontend", "Training", 
            "Recognition", "Evaluation"
        ])
        self.component_combo.currentTextChanged.connect(self.emit_filter_change)
        form_layout.addRow("Component:", self.component_combo)
        
        layout.addLayout(form_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        clear_btn = QPushButton("🔄 Clear Filters")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        clear_btn.clicked.connect(self.clear_filters)
        buttons_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("📁 Export Logs")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        export_btn.clicked.connect(self.export_logs)
        buttons_layout.addWidget(export_btn)
        
        layout.addLayout(buttons_layout)
    
    def emit_filter_change(self):
        """Emit filter change signal"""
        filters = {
            "level": self.level_combo.currentText(),
            "search": self.search_edit.text(),
            "time_range": self.time_combo.currentText(),
            "component": self.component_combo.currentText()
        }
        self.filter_changed.emit(filters)
    
    def clear_filters(self):
        """Clear all filters"""
        self.level_combo.setCurrentText("ALL")
        self.search_edit.clear()
        self.time_combo.setCurrentText("All Time")
        self.component_combo.setCurrentText("All Components")
    
    def export_logs(self):
        """Export filtered logs to file"""
        self.parent().export_logs()

class LogDisplayWidget(QFrame):
    """Widget for displaying log entries"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_entries = []
        self.filtered_entries = []
        self.current_filters = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup log display UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("System Logs")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Controls
        self.auto_scroll_check = QCheckBox("Auto Scroll")
        self.auto_scroll_check.setChecked(True)
        header_layout.addWidget(self.auto_scroll_check)
        
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        clear_btn.clicked.connect(self.clear_logs)
        header_layout.addWidget(clear_btn)
        
        layout.addLayout(header_layout)
        
        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        
        # Set up text formats for different log levels
        self.setup_log_formats()
        
        layout.addWidget(self.log_text)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.entries_label = QLabel("Entries: 0")
        self.entries_label.setStyleSheet("color: #666; font-size: 11px;")
        status_layout.addWidget(self.entries_label)
        
        status_layout.addStretch()
        
        self.last_update_label = QLabel("Last update: Never")
        self.last_update_label.setStyleSheet("color: #666; font-size: 11px;")
        status_layout.addWidget(self.last_update_label)
        
        layout.addLayout(status_layout)
    
    def setup_log_formats(self):
        """Setup text formats for different log levels"""
        self.log_formats = {
            "DEBUG": QTextCharFormat(),
            "INFO": QTextCharFormat(),
            "WARNING": QTextCharFormat(),
            "ERROR": QTextCharFormat(),
            "CRITICAL": QTextCharFormat()
        }
        
        # Set colors for each level
        self.log_formats["DEBUG"].setForeground(QColor("#888888"))
        self.log_formats["INFO"].setForeground(QColor("#ffffff"))
        self.log_formats["WARNING"].setForeground(QColor("#ff9800"))
        self.log_formats["ERROR"].setForeground(QColor("#f44336"))
        self.log_formats["CRITICAL"].setForeground(QColor("#ff0000"))
        self.log_formats["CRITICAL"].setFontWeight(QFont.Weight.Bold)
    
    def add_log_entry(self, timestamp: str, level: str, message: str):
        """Add a new log entry"""
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "raw_time": datetime.now()
        }
        
        self.log_entries.append(entry)
        
        # Limit entries to prevent memory issues
        if len(self.log_entries) > 10000:
            self.log_entries = self.log_entries[-5000:]  # Keep last 5000
        
        # Apply current filters
        self.apply_filters(self.current_filters)
        
        # Update last update time
        self.last_update_label.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    def apply_filters(self, filters: Dict[str, str]):
        """Apply filters to log entries"""
        self.current_filters = filters
        
        # Filter entries
        filtered = []
        
        for entry in self.log_entries:
            # Level filter
            if filters.get("level", "ALL") != "ALL":
                if entry["level"] != filters["level"]:
                    continue
            
            # Search filter
            search_term = filters.get("search", "").lower()
            if search_term and search_term not in entry["message"].lower():
                continue
            
            # Time range filter
            time_range = filters.get("time_range", "All Time")
            if time_range != "All Time":
                cutoff_time = self.get_time_cutoff(time_range)
                if entry["raw_time"] < cutoff_time:
                    continue
            
            # Component filter (simple keyword matching)
            component = filters.get("component", "All Components")
            if component != "All Components":
                component_keywords = {
                    "Backend": ["backend", "api", "fastapi"],
                    "Frontend": ["frontend", "pyqt", "gui"],
                    "Training": ["training", "train", "epoch"],
                    "Recognition": ["recognition", "recognize", "inference"],
                    "Evaluation": ["evaluation", "evaluate", "test"]
                }
                
                keywords = component_keywords.get(component, [])
                if not any(keyword in entry["message"].lower() for keyword in keywords):
                    continue
            
            filtered.append(entry)
        
        self.filtered_entries = filtered
        self.update_display()
    
    def get_time_cutoff(self, time_range: str) -> datetime:
        """Get cutoff time for time range filter"""
        now = datetime.now()
        
        if time_range == "Last Hour":
            return now - timedelta(hours=1)
        elif time_range == "Last 6 Hours":
            return now - timedelta(hours=6)
        elif time_range == "Last 24 Hours":
            return now - timedelta(days=1)
        elif time_range == "Last 7 Days":
            return now - timedelta(days=7)
        
        return datetime.min
    
    def update_display(self):
        """Update the log display"""
        self.log_text.clear()
        
        cursor = self.log_text.textCursor()
        
        for entry in self.filtered_entries[-1000:]:  # Show last 1000 entries
            # Format timestamp
            timestamp_part = f"[{entry['timestamp']}]"
            
            # Format level with color
            level_part = f" {entry['level']:<8}"
            
            # Message
            message_part = f" {entry['message']}\n"
            
            # Insert timestamp (white)
            cursor.insertText(timestamp_part)
            
            # Insert level with color
            format_obj = self.log_formats.get(entry['level'], self.log_formats['INFO'])
            cursor.insertText(level_part, format_obj)
            
            # Insert message (white)
            cursor.insertText(message_part)
        
        # Auto scroll to bottom if enabled
        if self.auto_scroll_check.isChecked():
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
        # Update entry count
        self.entries_label.setText(f"Entries: {len(self.filtered_entries)}")
    
    def clear_logs(self):
        """Clear all log entries"""
        self.log_entries.clear()
        self.filtered_entries.clear()
        self.log_text.clear()
        self.entries_label.setText("Entries: 0")
        self.last_update_label.setText("Last update: Never")

class LogsWidget(QWidget):
    """Main logs viewer widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.log_monitor = None
        self.setup_ui()
        self.setup_connections()
        self.start_monitoring()
    
    def setup_ui(self):
        """Setup logs viewer UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("System Logs")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Controls
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_logs)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Filters
        self.filter_widget = LogFilterWidget()
        splitter.addWidget(self.filter_widget)
        
        # Right side - Log display
        self.display_widget = LogDisplayWidget()
        splitter.addWidget(self.display_widget)
        
        # Set proportions
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])
        
        layout.addWidget(splitter)
        
        # Information panel
        info_group = QGroupBox("Log Information")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel("""
        <b>Real-time Log Monitoring:</b><br>
        • <b>Live Updates:</b> Automatically monitors backend and system logs<br>
        • <b>Filtering:</b> Filter by level, time range, component, or search terms<br>
        • <b>Export:</b> Save filtered logs to file for analysis<br>
        • <b>Color Coding:</b> Different colors for DEBUG, INFO, WARNING, ERROR, CRITICAL<br><br>
        
        <b>Log Sources:</b><br>
        • Backend API logs (FastAPI server)<br>
        • Training process logs<br>
        • Recognition pipeline logs<br>
        • System error logs
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
        self.filter_widget.filter_changed.connect(self.display_widget.apply_filters)
    
    def start_monitoring(self):
        """Start log file monitoring"""
        # Define log files to monitor
        log_files = [
            "backend/logs/api.log",
            "logs/training.log",
            "logs/recognition.log",
            "logs/system.log"
        ]
        
        # Create log files if they don't exist
        for log_file in log_files:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if not log_path.exists():
                log_path.touch()
        
        # Start monitoring thread
        self.log_monitor = LogMonitorThread(log_files)
        self.log_monitor.new_log_entry.connect(self.display_widget.add_log_entry)
        self.log_monitor.start_monitoring()
        
        # Add some initial log entries
        self.add_startup_logs()
    
    def add_startup_logs(self):
        """Add initial startup log entries"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        startup_logs = [
            (timestamp, "INFO", "AI Recognition System - Log viewer started"),
            (timestamp, "INFO", "Monitoring backend logs: backend/logs/api.log"),
            (timestamp, "INFO", "Real-time log monitoring active"),
            (timestamp, "DEBUG", "Log filters initialized with default settings")
        ]
        
        for ts, level, message in startup_logs:
            self.display_widget.add_log_entry(ts, level, message)
    
    def refresh_logs(self):
        """Refresh log display"""
        if self.log_monitor:
            self.log_monitor.stop_monitoring()
        
        self.start_monitoring()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.display_widget.add_log_entry(timestamp, "INFO", "Log monitoring refreshed")
    
    def export_logs(self):
        """Export current filtered logs to file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Logs",
                f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text Files (*.txt);;All Files (*)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write("AI Recognition System - Log Export\n")
                    f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for entry in self.display_widget.filtered_entries:
                        f.write(f"[{entry['timestamp']}] {entry['level']:<8} {entry['message']}\n")
                
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Logs exported successfully to:\n{filename}"
                )
        
        except Exception as e:
            logger.error(f"Export error: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export logs: {str(e)}"
            )
    
    def closeEvent(self, event):
        """Handle widget closing"""
        if self.log_monitor:
            self.log_monitor.stop_monitoring()
        event.accept()