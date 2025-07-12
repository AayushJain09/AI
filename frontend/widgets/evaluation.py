"""
Evaluation Dashboard Widget
Interface for running system evaluation and viewing performance metrics
"""

import json
import time
from typing import Dict, Any, Optional, List
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QGroupBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QSplitter, QTabWidget,
    QSizePolicy, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon, QPalette, QColor

logger = logging.getLogger(__name__)

class EvaluationMonitorThread(QThread):
    """Background thread to monitor evaluation progress"""
    
    status_updated = pyqtSignal(dict)
    
    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.running = False
    
    def start_monitoring(self):
        """Start monitoring evaluation status"""
        self.running = True
        self.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.wait()
    
    def run(self):
        """Monitor evaluation status"""
        while self.running:
            try:
                # Check system status for evaluation progress
                response = self.api_client.get("/api/status")
                self.status_updated.emit(response)
                self.msleep(2000)  # Check every 2 seconds
            except Exception as e:
                logger.error(f"Evaluation status check failed: {e}")
                self.msleep(5000)

class MetricsDisplayWidget(QFrame):
    """Widget to display evaluation metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.clear_metrics()
    
    def setup_ui(self):
        """Setup metrics display UI"""
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
        title = QLabel("Performance Metrics")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(title)
        
        # Metrics grid
        metrics_layout = QGridLayout()
        
        # Create metric cards
        self.accuracy_card = self.create_metric_card("Overall Accuracy", "0%", "#4CAF50", "🎯")
        self.confidence_card = self.create_metric_card("Avg Confidence", "0.00", "#2196F3", "📊")
        self.speed_card = self.create_metric_card("Avg Speed", "0.000s", "#FF9800", "⚡")
        self.tests_card = self.create_metric_card("Tests Passed", "0/0", "#9C27B0", "✅")
        
        metrics_layout.addWidget(self.accuracy_card, 0, 0)
        metrics_layout.addWidget(self.confidence_card, 0, 1)
        metrics_layout.addWidget(self.speed_card, 1, 0)
        metrics_layout.addWidget(self.tests_card, 1, 1)
        
        layout.addLayout(metrics_layout)
        
        # Target achievement
        targets_group = QGroupBox("Target Achievement")
        targets_layout = QVBoxLayout(targets_group)
        
        self.accuracy_target = self.create_target_indicator("Accuracy Target", "95%", False)
        self.speed_target = self.create_target_indicator("Speed Target", "0.5s", False)
        
        targets_layout.addWidget(self.accuracy_target)
        targets_layout.addWidget(self.speed_target)
        
        layout.addWidget(targets_group)
    
    def create_metric_card(self, title: str, value: str, color: str, icon: str) -> QFrame:
        """Create a metric display card"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 white, stop:1 {color}15);
                border: 2px solid {color};
                border-radius: 8px;
                padding: 15px;
                min-height: 80px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 24px; margin-bottom: 5px;")
        layout.addWidget(icon_label)
        
        # Value
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet(f"""
            QLabel {{
                font-size: 20px;
                font-weight: bold;
                color: {color};
                margin-bottom: 5px;
            }}
        """)
        layout.addWidget(value_label)
        
        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                font-weight: normal;
            }
        """)
        layout.addWidget(title_label)
        
        # Store references for updates
        card.value_label = value_label
        card.title_text = title
        
        return card
    
    def create_target_indicator(self, label: str, target: str, achieved: bool) -> QFrame:
        """Create a target achievement indicator"""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        # Status icon
        status_icon = QLabel("✅" if achieved else "❌")
        status_icon.setStyleSheet("font-size: 16px;")
        layout.addWidget(status_icon)
        
        # Label
        label_widget = QLabel(f"{label} ({target})")
        label_widget.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(label_widget)
        
        # Status text
        status_text = QLabel("Achieved" if achieved else "Not Achieved")
        status_color = "#4CAF50" if achieved else "#f44336"
        status_text.setStyleSheet(f"color: {status_color}; font-size: 12px;")
        layout.addWidget(status_text)
        
        layout.addStretch()
        
        # Store references for updates
        frame.status_icon = status_icon
        frame.status_text = status_text
        
        return frame
    
    def update_metrics(self, results: Dict[str, Any]):
        """Update metrics display with evaluation results"""
        try:
            summary = results.get("evaluation_summary", {})
            
            # Update metric cards
            accuracy = summary.get("overall_accuracy", 0.0)
            self.accuracy_card.value_label.setText(f"{accuracy:.1%}")
            
            confidence = summary.get("avg_confidence", 0.0)
            self.confidence_card.value_label.setText(f"{confidence:.3f}")
            
            speed = summary.get("avg_inference_time", 0.0)
            self.speed_card.value_label.setText(f"{speed:.3f}s")
            
            total_tests = summary.get("total_tests", 0)
            successful_tests = summary.get("successful_tests", 0)
            self.tests_card.value_label.setText(f"{successful_tests}/{total_tests}")
            
            # Update target achievement
            targets = summary.get("targets_achieved", {})
            
            accuracy_achieved = targets.get("accuracy", False)
            self.accuracy_target.status_icon.setText("✅" if accuracy_achieved else "❌")
            self.accuracy_target.status_text.setText("Achieved" if accuracy_achieved else "Not Achieved")
            self.accuracy_target.status_text.setStyleSheet(
                f"color: {'#4CAF50' if accuracy_achieved else '#f44336'}; font-size: 12px;"
            )
            
            speed_achieved = targets.get("speed", False)
            self.speed_target.status_icon.setText("✅" if speed_achieved else "❌")
            self.speed_target.status_text.setText("Achieved" if speed_achieved else "Not Achieved")
            self.speed_target.status_text.setStyleSheet(
                f"color: {'#4CAF50' if speed_achieved else '#f44336'}; font-size: 12px;"
            )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def clear_metrics(self):
        """Clear all metrics"""
        self.accuracy_card.value_label.setText("0%")
        self.confidence_card.value_label.setText("0.00")
        self.speed_card.value_label.setText("0.000s")
        self.tests_card.value_label.setText("0/0")
        
        for target in [self.accuracy_target, self.speed_target]:
            target.status_icon.setText("❌")
            target.status_text.setText("Not Achieved")
            target.status_text.setStyleSheet("color: #f44336; font-size: 12px;")

class PerItemResultsWidget(QFrame):
    """Widget to display per-item evaluation results"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup per-item results UI"""
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
        title = QLabel("Per-Item Performance")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(title)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Item ID", "Accuracy", "Tests", "Status"
        ])
        
        # Configure table
        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: white;
                gridline-color: #eee;
                selection-background-color: #e3f2fd;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 8px;
                border: none;
                font-weight: bold;
                border-bottom: 2px solid #ddd;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
        """)
        
        layout.addWidget(self.results_table)
        
        # Summary
        self.summary_label = QLabel("No evaluation results available")
        self.summary_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 13px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 4px;
                margin-top: 10px;
            }
        """)
        layout.addWidget(self.summary_label)
    
    def update_results(self, results: Dict[str, Any]):
        """Update per-item results display"""
        try:
            performance_report = results.get("performance_report", {})
            per_item_metrics = performance_report.get("per_item_metrics", {})
            
            # Clear and populate table
            self.results_table.setRowCount(len(per_item_metrics))
            
            excellent_count = 0
            good_count = 0
            poor_count = 0
            
            for row, (item_id, metrics) in enumerate(per_item_metrics.items()):
                accuracy = metrics.get("accuracy", 0.0)
                total_tests = metrics.get("total_tests", 0)
                correct = metrics.get("correct", 0)
                
                # Item ID
                self.results_table.setItem(row, 0, QTableWidgetItem(item_id))
                
                # Accuracy
                accuracy_item = QTableWidgetItem(f"{accuracy:.1%}")
                self.results_table.setItem(row, 1, accuracy_item)
                
                # Tests
                tests_item = QTableWidgetItem(f"{correct}/{total_tests}")
                self.results_table.setItem(row, 2, tests_item)
                
                # Status with color coding
                if accuracy >= 0.9:
                    status = "🟢 Excellent"
                    color = "#E8F5E8"
                    excellent_count += 1
                elif accuracy >= 0.7:
                    status = "🟡 Good"
                    color = "#FFF3E0"
                    good_count += 1
                else:
                    status = "🔴 Needs Work"
                    color = "#FFEBEE"
                    poor_count += 1
                
                status_item = QTableWidgetItem(status)
                self.results_table.setItem(row, 3, status_item)
                
                # Color code the entire row
                for col in range(4):
                    item = self.results_table.item(row, col)
                    if item:
                        item.setBackground(QColor(color))
            
            # Update summary
            total_items = len(per_item_metrics)
            if total_items > 0:
                summary_text = (
                    f"📊 Total Items: {total_items} | "
                    f"🟢 Excellent: {excellent_count} | "
                    f"🟡 Good: {good_count} | "
                    f"🔴 Needs Work: {poor_count}"
                )
                self.summary_label.setText(summary_text)
            
        except Exception as e:
            logger.error(f"Error updating per-item results: {e}")
            self.summary_label.setText(f"Error loading results: {str(e)}")

class EvaluationProgressWidget(QFrame):
    """Widget to show evaluation progress"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.reset_progress()
    
    def setup_ui(self):
        """Setup evaluation progress UI"""
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
        title = QLabel("Evaluation Progress")
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
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #9C27B0;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Progress log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
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
        layout.addWidget(self.log_text)
    
    def reset_progress(self):
        """Reset progress display"""
        self.status_label.setText("Not started")
        self.status_label.setStyleSheet("color: #666;")
        self.progress_bar.setValue(0)
        self.log_text.clear()
    
    def update_progress(self, status: str, progress: int = 0):
        """Update progress display"""
        self.status_label.setText(status)
        if "progress" in status.lower() or "running" in status.lower():
            self.status_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
        elif "complete" in status.lower():
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif "error" in status.lower() or "fail" in status.lower():
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        
        self.progress_bar.setValue(progress)
        
        # Add to log
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {status}"
        self.log_text.append(log_entry)

class EvaluationWidget(QWidget):
    """Main evaluation dashboard widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.monitor_thread = EvaluationMonitorThread(api_client)
        self.setup_ui()
        self.setup_connections()
        self.load_existing_results()
    
    def setup_ui(self):
        """Setup evaluation dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("System Evaluation")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Evaluation controls
        self.run_btn = QPushButton("📈 Run Evaluation")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.run_btn.clicked.connect(self.run_evaluation)
        header_layout.addWidget(self.run_btn)
        
        self.refresh_btn = QPushButton("🔄 Refresh Results")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.refresh_btn.clicked.connect(self.load_existing_results)
        header_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Main content tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 12px 20px;
                margin: 2px;
                border-radius: 4px 4px 0 0;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #9C27B0;
            }
        """)
        
        # Overview tab
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        
        # Progress widget
        self.progress_widget = EvaluationProgressWidget()
        overview_layout.addWidget(self.progress_widget)
        
        # Metrics widget
        self.metrics_widget = MetricsDisplayWidget()
        overview_layout.addWidget(self.metrics_widget)
        
        self.tabs.addTab(overview_widget, "📊 Overview")
        
        # Detailed results tab
        self.per_item_widget = PerItemResultsWidget()
        self.tabs.addTab(self.per_item_widget, "📋 Per-Item Results")
        
        # Raw data tab
        raw_data_widget = QWidget()
        raw_layout = QVBoxLayout(raw_data_widget)
        
        self.raw_data_text = QTextEdit()
        self.raw_data_text.setReadOnly(True)
        self.raw_data_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        raw_layout.addWidget(self.raw_data_text)
        
        self.tabs.addTab(raw_data_widget, "🔍 Raw Data")
        
        layout.addWidget(self.tabs)
        
        # Information panel
        info_group = QGroupBox("Evaluation Information")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel("""
        <b>System Evaluation Process:</b><br>
        • Tests recognition accuracy on original (non-augmented) images<br>
        • Measures processing speed and confidence scores<br>
        • Generates comprehensive performance reports<br>
        • Compares results against target thresholds (95% accuracy, 0.5s speed)<br><br>
        
        <b>Metrics Explained:</b><br>
        • <b>Overall Accuracy:</b> Percentage of correctly identified items<br>
        • <b>Average Confidence:</b> Mean confidence score across all predictions<br>
        • <b>Average Speed:</b> Mean processing time per image<br>
        • <b>Per-Item Performance:</b> Individual accuracy for each item type
        """)
        info_text.setWordWrap(True)
        info_text.setStyleSheet("""
            QLabel {
                color: #555;
                font-size: 12px;
                background-color: #f0f8ff;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #9C27B0;
            }
        """)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.monitor_thread.status_updated.connect(self.update_evaluation_status)
    
    def run_evaluation(self):
        """Start system evaluation"""
        try:
            reply = QMessageBox.question(
                self,
                "Run Evaluation",
                "Start system evaluation?\n\n"
                "This will test the recognition accuracy on all items "
                "and may take several minutes to complete.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Start evaluation via API
            response = self.api_client.post("/api/evaluate")
            
            if response.get("success"):
                QMessageBox.information(
                    self,
                    "Evaluation Started",
                    "Evaluation has been started! Monitor progress in the Overview tab."
                )
                
                # Reset and start progress monitoring
                self.progress_widget.reset_progress()
                self.progress_widget.update_progress("🚀 Evaluation started...", 10)
                
                self.run_btn.setEnabled(False)
                self.monitor_thread.start_monitoring()
                
            else:
                QMessageBox.warning(
                    self,
                    "Evaluation Failed",
                    f"Failed to start evaluation: {response.get('error', 'Unknown error')}"
                )
        
        except Exception as e:
            logger.error(f"Evaluation start error: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while starting evaluation: {str(e)}"
            )
    
    def load_existing_results(self):
        """Load existing evaluation results"""
        try:
            response = self.api_client.get("/api/evaluate/results")
            
            if response.get("success"):
                results = response.get("data", {})
                
                # Update displays
                self.metrics_widget.update_metrics(results)
                self.per_item_widget.update_results(results)
                
                # Update raw data
                formatted_json = json.dumps(results, indent=2)
                self.raw_data_text.setText(formatted_json)
                
                # Update progress
                summary = results.get("evaluation_summary", {})
                accuracy = summary.get("overall_accuracy", 0.0)
                timestamp = summary.get("timestamp", "Unknown")
                
                self.progress_widget.update_progress(
                    f"✅ Last evaluation completed at {timestamp} (Accuracy: {accuracy:.1%})",
                    100
                )
                
                logger.info("Loaded existing evaluation results")
            
            else:
                self.progress_widget.update_progress("No evaluation results found", 0)
                self.metrics_widget.clear_metrics()
        
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            self.progress_widget.update_progress(f"Error loading results: {str(e)}", 0)
    
    def update_evaluation_status(self, status_data: Dict[str, Any]):
        """Update evaluation status from monitoring thread"""
        try:
            # Check if evaluation is in progress
            if status_data.get("evaluation_in_progress", False):
                self.progress_widget.update_progress("🔍 Evaluation in progress...", 50)
            else:
                # Evaluation finished, stop monitoring and reload results
                self.monitor_thread.stop_monitoring()
                self.run_btn.setEnabled(True)
                
                # Load updated results
                self.load_existing_results()
        
        except Exception as e:
            logger.error(f"Error updating evaluation status: {e}")