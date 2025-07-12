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
        """Setup settings UI"""
        layout = QVBoxLayout(self)
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
        
        # Configuration tabs will be created when config is loaded
        layout.addWidget(self.tabs)
        
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