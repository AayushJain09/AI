"""
Settings Widget
System configuration and settings management interface
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QGroupBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QTextEdit, QMessageBox, QTabWidget,
    QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)

class AutoTrainingWidget(QGroupBox):
    """Auto-training configuration widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__("🤖 Auto-Training", parent)
        self.api_client = api_client
        self.setup_ui()
        self.refresh_status()
    
    def setup_ui(self):
        """Setup auto-training UI"""
        layout = QVBoxLayout(self)
        
        # Status display
        self.status_label = QLabel("Loading auto-training status...")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
                border-left: 4px solid #2196F3;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Configuration controls
        config_frame = QFrame()
        config_layout = QFormLayout(config_frame)
        
        # Enable/disable auto-training
        self.enabled_checkbox = QCheckBox("Enable Auto-Training")
        self.enabled_checkbox.setStyleSheet("font-size: 13px; font-weight: bold;")
        config_layout.addRow(self.enabled_checkbox)
        
        # Minimum images threshold
        self.min_images_spinbox = QSpinBox()
        self.min_images_spinbox.setRange(1, 50)
        self.min_images_spinbox.setValue(5)
        self.min_images_spinbox.setSuffix(" images")
        config_layout.addRow("Minimum images to trigger:", self.min_images_spinbox)
        
        # Batch delay
        self.batch_delay_spinbox = QSpinBox()
        self.batch_delay_spinbox.setRange(5, 240)
        self.batch_delay_spinbox.setValue(30)
        self.batch_delay_spinbox.setSuffix(" minutes")
        config_layout.addRow("Batch delay:", self.batch_delay_spinbox)
        
        layout.addWidget(config_frame)
        
        # Pending items display
        self.pending_label = QLabel("No items pending training")
        self.pending_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                padding: 8px;
                background-color: #f0f0f0;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.pending_label)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        # Save config button
        save_btn = QPushButton("💾 Save Config")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        save_btn.clicked.connect(self.save_config)
        buttons_layout.addWidget(save_btn)
        
        # Force training button
        force_btn = QPushButton("🚀 Force Training Now")
        force_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #F57C00; }
        """)
        force_btn.clicked.connect(self.force_training)
        buttons_layout.addWidget(force_btn)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        refresh_btn.clicked.connect(self.refresh_status)
        buttons_layout.addWidget(refresh_btn)
        
        layout.addLayout(buttons_layout)
    
    def refresh_status(self):
        """Refresh auto-training status"""
        try:
            response = self.api_client.get("/api/auto-training/status")
            if response.get("success"):
                data = response.get("data", {})
                config = data.get("config", {})
                pending_items = data.get("pending_items", [])
                active_training = data.get("active_training", False)
                
                # Update controls
                self.enabled_checkbox.setChecked(config.get("enabled", True))
                self.min_images_spinbox.setValue(config.get("min_images_threshold", 5))
                self.batch_delay_spinbox.setValue(config.get("batch_delay_minutes", 30))
                
                # Update status
                status_text = "✅ Auto-training enabled" if config.get("enabled") else "⏸️ Auto-training disabled"
                if active_training:
                    status_text += " | 🔄 Training in progress"
                if config.get("last_training_trigger"):
                    status_text += f" | Last training: {config.get('last_training_trigger')}"
                
                self.status_label.setText(status_text)
                
                # Update pending items
                if pending_items:
                    self.pending_label.setText(f"📋 Pending items: {', '.join(pending_items)}")
                    self.pending_label.setStyleSheet("""
                        QLabel {
                            font-size: 12px;
                            color: #FF9800;
                            padding: 8px;
                            background-color: #FFF3E0;
                            border-radius: 4px;
                        }
                    """)
                else:
                    self.pending_label.setText("✅ No items pending training")
                    self.pending_label.setStyleSheet("""
                        QLabel {
                            font-size: 12px;
                            color: #4CAF50;
                            padding: 8px;
                            background-color: #F1F8E9;
                            border-radius: 4px;
                        }
                    """)
            else:
                self.status_label.setText("❌ Failed to load auto-training status")
        except Exception as e:
            logger.error(f"Failed to refresh auto-training status: {e}")
            self.status_label.setText(f"❌ Error: {str(e)}")
    
    def save_config(self):
        """Save auto-training configuration"""
        try:
            config_data = {
                "enabled": self.enabled_checkbox.isChecked(),
                "min_images_threshold": self.min_images_spinbox.value(),
                "batch_delay_minutes": self.batch_delay_spinbox.value()
            }
            
            response = self.api_client.post("/api/auto-training/config", config_data)
            if response.get("success"):
                QMessageBox.information(
                    self,
                    "Config Saved",
                    "Auto-training configuration saved successfully!"
                )
                self.refresh_status()
            else:
                QMessageBox.warning(
                    self,
                    "Save Failed",
                    f"Failed to save configuration: {response.get('error', 'Unknown error')}"
                )
        except Exception as e:
            logger.error(f"Failed to save auto-training config: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save configuration: {str(e)}"
            )
    
    def force_training(self):
        """Force immediate auto-training"""
        reply = QMessageBox.question(
            self,
            "Force Training",
            "This will immediately trigger training for all pending items. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                response = self.api_client.post("/api/auto-training/trigger")
                if response.get("success"):
                    data = response.get("data", {})
                    items_count = data.get("items_count", 0)
                    QMessageBox.information(
                        self,
                        "Training Triggered",
                        f"Training started for {items_count} items. Check the training tab for progress."
                    )
                    self.refresh_status()
                else:
                    QMessageBox.warning(
                        self,
                        "Training Failed",
                        f"Failed to trigger training: {response.get('message', 'Unknown error')}"
                    )
            except Exception as e:
                logger.error(f"Failed to force auto-training: {e}")
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to trigger training: {str(e)}"
                )

class ManualTrainingWidget(QGroupBox):
    """Manual training control widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__("🎯 Manual Training", parent)
        self.api_client = api_client
        self.setup_ui()
        self.refresh_items()
    
    def setup_ui(self):
        """Setup manual training UI"""
        layout = QVBoxLayout(self)
        
        # Description
        desc_label = QLabel("Train specific items manually whenever needed")
        desc_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                padding: 5px 0;
                font-style: italic;
            }
        """)
        layout.addWidget(desc_label)
        
        # Item selection
        selection_frame = QFrame()
        selection_layout = QFormLayout(selection_frame)
        
        # Available items dropdown
        self.items_combo = QComboBox()
        self.items_combo.setMinimumHeight(35)
        self.items_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
        """)
        selection_layout.addRow("Select Item:", self.items_combo)
        
        # Training options
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 100)
        self.epochs_spinbox.setValue(10)
        self.epochs_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 13px;
                min-height: 20px;
            }
        """)
        selection_layout.addRow("Epochs:", self.epochs_spinbox)
        
        layout.addWidget(selection_frame)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        # Train selected item
        train_btn = QPushButton("🚀 Train Selected Item")
        train_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        train_btn.clicked.connect(self.train_selected_item)
        buttons_layout.addWidget(train_btn)
        
        # Train all items
        train_all_btn = QPushButton("🔥 Train All Items")
        train_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c82333; }
        """)
        train_all_btn.clicked.connect(self.train_all_items)
        buttons_layout.addWidget(train_all_btn)
        
        # Refresh items
        refresh_items_btn = QPushButton("🔄")
        refresh_items_btn.setMaximumWidth(40)
        refresh_items_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #5a6268; }
        """)
        refresh_items_btn.clicked.connect(self.refresh_items)
        buttons_layout.addWidget(refresh_items_btn)
        
        layout.addLayout(buttons_layout)
    
    def refresh_items(self):
        """Refresh available items list"""
        try:
            response = self.api_client.get("/api/items")
            if response.get("success"):
                items = response.get("data", [])
                self.items_combo.clear()
                self.items_combo.addItem("All Items", "all")
                
                for item in items:
                    item_id = item.get("id", "")
                    item_name = item.get("name", item_id)
                    image_count = len(item.get("images", []))
                    display_text = f"{item_name} ({image_count} images)"
                    self.items_combo.addItem(display_text, item_id)
                    
            else:
                QMessageBox.warning(self, "Error", "Failed to load items list")
        except Exception as e:
            logger.error(f"Failed to refresh items: {e}")
            QMessageBox.warning(self, "Error", f"Failed to refresh items: {str(e)}")
    
    def train_selected_item(self):
        """Train the selected item"""
        current_data = self.items_combo.currentData()
        if not current_data:
            QMessageBox.warning(self, "No Selection", "Please select an item to train")
            return
        
        epochs = self.epochs_spinbox.value()
        
        if current_data == "all":
            self.train_all_items()
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Training",
            f"Train item '{self.items_combo.currentText()}' for {epochs} epochs?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                training_data = {
                    "item_id": current_data,
                    "epochs": epochs,
                    "manual": True
                }
                
                response = self.api_client.post("/api/training/manual", training_data)
                if response.get("success"):
                    QMessageBox.information(
                        self,
                        "Training Started",
                        f"Manual training started for '{self.items_combo.currentText()}'. "
                        f"Check the training tab for progress."
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Training Failed",
                        f"Failed to start training: {response.get('message', 'Unknown error')}"
                    )
            except Exception as e:
                logger.error(f"Failed to start manual training: {e}")
                QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")
    
    def train_all_items(self):
        """Train all available items"""
        reply = QMessageBox.question(
            self,
            "Confirm Batch Training",
            f"Train ALL items for {self.epochs_spinbox.value()} epochs each? This may take a long time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                training_data = {
                    "epochs": self.epochs_spinbox.value(),
                    "manual": True,
                    "batch": True
                }
                
                response = self.api_client.post("/api/training/manual", training_data)
                if response.get("success"):
                    items_count = response.get("data", {}).get("items_count", 0)
                    QMessageBox.information(
                        self,
                        "Batch Training Started",
                        f"Manual training started for {items_count} items. "
                        f"Check the training tab for progress."
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Training Failed",
                        f"Failed to start batch training: {response.get('message', 'Unknown error')}"
                    )
            except Exception as e:
                logger.error(f"Failed to start batch training: {e}")
                QMessageBox.critical(self, "Error", f"Failed to start batch training: {str(e)}")

class ConfigEditor(QFrame):
    """Configuration editor widget"""
    
    config_changed = pyqtSignal(dict)
    
    def __init__(self, title: str, config_section: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.title = title
        self.config_section = config_section
        self.controls = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup configuration editor UI"""
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
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)
        
        # Configuration form
        form_layout = QFormLayout()
        
        for key, value in self.config_section.items():
            if isinstance(value, dict):
                # Skip nested dictionaries for now
                continue
            
            control = self.create_control(key, value)
            if control:
                self.controls[key] = control
                form_layout.addRow(self.format_label(key), control)
        
        layout.addLayout(form_layout)
        
        # Save button
        save_btn = QPushButton("💾 Save Changes")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        save_btn.clicked.connect(self.save_config)
        layout.addWidget(save_btn)
    
    def create_control(self, key: str, value: Any) -> QWidget:
        """Create appropriate control for configuration value"""
        if isinstance(value, bool):
            control = QCheckBox()
            control.setChecked(value)
            return control
        
        elif isinstance(value, int):
            control = QSpinBox()
            control.setRange(-999999, 999999)
            control.setValue(value)
            return control
        
        elif isinstance(value, float):
            control = QDoubleSpinBox()
            control.setRange(-999999.0, 999999.0)
            control.setDecimals(6)
            control.setValue(value)
            return control
        
        elif isinstance(value, str):
            control = QLineEdit()
            control.setText(value)
            return control
        
        return None
    
    def format_label(self, key: str) -> str:
        """Format configuration key as readable label"""
        return key.replace('_', ' ').title() + ":"
    
    def save_config(self):
        """Save configuration changes"""
        try:
            new_config = {}
            
            for key, control in self.controls.items():
                if isinstance(control, QCheckBox):
                    new_config[key] = control.isChecked()
                elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                    new_config[key] = control.value()
                elif isinstance(control, QLineEdit):
                    new_config[key] = control.text()
            
            # Merge with original config
            updated_section = {**self.config_section, **new_config}
            self.config_changed.emit(updated_section)
            
            QMessageBox.information(
                self,
                "Settings Saved",
                f"{self.title} settings have been saved successfully!"
            )
        
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save settings: {str(e)}"
            )

class SystemInfoWidget(QFrame):
    """System information display widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.setup_ui()
        self.refresh_info()
    
    def setup_ui(self):
        """Setup system info UI"""
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
        title = QLabel("System Information")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Info display
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        self.info_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.info_text)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_info)
        layout.addWidget(refresh_btn)
    
    def refresh_info(self):
        """Refresh system information"""
        try:
            # Get system status
            status_response = self.api_client.get("/api/status")
            health_response = self.api_client.get("/health")
            
            info_lines = [
                "=== AI Recognition System Information ===",
                "",
                f"Backend Status: {'Online' if health_response.get('status') == 'healthy' else 'Offline'}",
                f"System Initialized: {status_response.get('initialized', False)}",
                f"Recognition Ready: {status_response.get('system_ready', False)}",
                f"Training in Progress: {status_response.get('training_in_progress', False)}",
                f"Evaluation in Progress: {status_response.get('evaluation_in_progress', False)}",
                "",
                "=== Backend Health ===",
                f"Timestamp: {health_response.get('timestamp', 'Unknown')}",
                "",
                "=== Last Error ===",
                f"{status_response.get('last_error', 'None')}"
            ]
            
            self.info_text.setText("\n".join(info_lines))
        
        except Exception as e:
            self.info_text.setText(f"Error retrieving system information:\n{str(e)}")

class SettingsWidget(QWidget):
    """Main settings widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.config_data = {}
        self.setup_ui()
        self.load_configuration()
    
    def setup_ui(self):
        """Setup settings UI with scrollable content"""
        # Main layout for the widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create scroll area for all content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8f9fa;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #f8f9fa;
            }
        """)
        
        # Content widget inside scroll area
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Content layout
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("System Settings")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Control buttons
        load_btn = QPushButton("📁 Load Config")
        load_btn.setStyleSheet("""
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
        load_btn.clicked.connect(self.load_configuration)
        header_layout.addWidget(load_btn)
        
        save_btn = QPushButton("💾 Save Config")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        save_btn.clicked.connect(self.save_configuration)
        header_layout.addWidget(save_btn)
        
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
                border-bottom: 3px solid #2196F3;
            }
        """)
        # Set reasonable minimum height for tabs
        self.tabs.setMinimumHeight(300)
        
        # Configuration tabs will be created when config is loaded
        layout.addWidget(self.tabs)
        
        # Auto-training controls
        self.auto_training_widget = AutoTrainingWidget(self.api_client)
        layout.addWidget(self.auto_training_widget)
        
        # Manual training controls
        self.manual_training_widget = ManualTrainingWidget(self.api_client)
        layout.addWidget(self.manual_training_widget)
        
        # System information
        self.system_info = SystemInfoWidget(self.api_client)
        layout.addWidget(self.system_info)
        
        # Instructions
        instructions_group = QGroupBox("Configuration Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        
        instructions_text = QLabel("""
        <b>Configuration Management:</b><br>
        • <b>Load Config:</b> Refresh configuration from backend<br>
        • <b>Save Config:</b> Save all changes to configuration file<br>
        • <b>Model Settings:</b> CLIP variant, embedding dimensions<br>
        • <b>Training Settings:</b> Epochs, batch size, learning rates<br>
        • <b>Recognition Settings:</b> Confidence thresholds, cache settings<br><br>
        
        <b>Important Notes:</b><br>
        • Changes require system restart to take effect<br>
        • Invalid configurations may cause system errors<br>
        • Backup your configuration before making major changes<br>
        • Some settings affect training time and memory usage
        """)
        instructions_text.setWordWrap(True)
        instructions_text.setStyleSheet("""
            QLabel {
                color: #555;
                font-size: 12px;
                background-color: #f0f8ff;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #2196F3;
            }
        """)
        instructions_layout.addWidget(instructions_text)
        
        layout.addWidget(instructions_group)
        
        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
    
    def load_configuration(self):
        """Load configuration from backend"""
        try:
            response = self.api_client.get("/api/config")
            
            if response.get("success"):
                self.config_data = response.get("data", {})
                self.create_config_tabs()
                
                QMessageBox.information(
                    self,
                    "Config Loaded",
                    "Configuration loaded successfully from backend!"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Load Failed",
                    f"Failed to load configuration: {response.get('error', 'Unknown error')}"
                )
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load configuration: {str(e)}"
            )
    
    def create_config_tabs(self):
        """Create configuration editor tabs"""
        # Clear existing tabs
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)
        
        # Create tabs for each config section
        config_sections = [
            ("Model", "model", "⚙️"),
            ("Training", "training", "🏋️"),
            ("Features", "features", "🔧"),
            ("Recognition", "recognition", "🔍"),
            ("Data Paths", "data", "📁"),
            ("Targets", "targets", "🎯")
        ]
        
        for section_name, section_key, icon in config_sections:
            if section_key in self.config_data:
                section_data = self.config_data[section_key]
                
                # Create editor widget
                editor = ConfigEditor(section_name, section_data)
                editor.config_changed.connect(
                    lambda new_config, key=section_key: self.update_config_section(key, new_config)
                )
                
                # Add to tabs
                self.tabs.addTab(editor, f"{icon} {section_name}")
        
        # Raw JSON editor tab
        raw_editor = QWidget()
        raw_layout = QVBoxLayout(raw_editor)
        
        self.raw_config_text = QTextEdit()
        self.raw_config_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        
        # Format and display config as JSON
        formatted_config = json.dumps(self.config_data, indent=2)
        self.raw_config_text.setText(formatted_config)
        
        raw_layout.addWidget(QLabel("Raw Configuration (JSON):"))
        raw_layout.addWidget(self.raw_config_text)
        
        update_raw_btn = QPushButton("Update from JSON")
        update_raw_btn.clicked.connect(self.update_from_raw_json)
        raw_layout.addWidget(update_raw_btn)
        
        self.tabs.addTab(raw_editor, "📝 Raw JSON")
    
    def update_config_section(self, section_key: str, new_section_data: Dict[str, Any]):
        """Update a configuration section"""
        self.config_data[section_key] = new_section_data
        
        # Update raw JSON display
        if hasattr(self, 'raw_config_text'):
            formatted_config = json.dumps(self.config_data, indent=2)
            self.raw_config_text.setText(formatted_config)
    
    def update_from_raw_json(self):
        """Update configuration from raw JSON text"""
        try:
            raw_text = self.raw_config_text.toPlainText()
            new_config = json.loads(raw_text)
            
            self.config_data = new_config
            self.create_config_tabs()
            
            QMessageBox.information(
                self,
                "JSON Updated",
                "Configuration updated from JSON successfully!"
            )
        
        except json.JSONDecodeError as e:
            QMessageBox.critical(
                self,
                "JSON Error",
                f"Invalid JSON format:\n{str(e)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Update Error",
                f"Failed to update from JSON: {str(e)}"
            )
    
    def save_configuration(self):
        """Save configuration to file"""
        try:
            # Save to local file
            config_file = "config.yaml"
            
            with open(config_file, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            
            QMessageBox.information(
                self,
                "Config Saved",
                f"Configuration saved to {config_file}!\n\n"
                "Note: Restart the system for changes to take effect."
            )
        
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save configuration: {str(e)}"
            )