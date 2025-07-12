"""
Items Management Widget
Complete interface for managing inventory items
"""

import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QLineEdit, QTextEdit,
    QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QListWidget, QListWidgetItem, QSplitter, QTabWidget,
    QComboBox, QSpinBox, QCheckBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

logger = logging.getLogger(__name__)

class ImageUploadDialog(QDialog):
    """Dialog for uploading images to an item"""
    
    def __init__(self, item_id: str, api_client, parent=None):
        super().__init__(parent)
        self.item_id = item_id
        self.api_client = api_client
        self.selected_files = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup image upload dialog UI"""
        self.setWindowTitle(f"Upload Images - {self.item_id}")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Select images to upload for this item. Recommended: 8-20 high-quality images "
            "from different angles and lighting conditions."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 13px;
                padding: 10px;
                background-color: #f0f8ff;
                border-radius: 4px;
                border-left: 4px solid #2196F3;
            }
        """)
        layout.addWidget(instructions)
        
        # File selection area
        file_area = QGroupBox("Selected Images")
        file_layout = QVBoxLayout(file_area)
        
        # Select files button
        select_btn = QPushButton("📁 Select Images")
        select_btn.setStyleSheet("""
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
        select_btn.clicked.connect(self.select_files)
        file_layout.addWidget(select_btn)
        
        # Selected files list
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(200)
        self.files_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #fafafa;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
        """)
        file_layout.addWidget(self.files_list)
        
        layout.addWidget(file_area)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
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
        layout.addWidget(self.progress_bar)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.upload_images)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.upload_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.upload_btn.setText("Upload Images")
        self.upload_btn.setEnabled(False)
    
    def select_files(self):
        """Open file dialog to select images"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if files:
            self.selected_files = files
            self.update_files_list()
            self.upload_btn.setEnabled(True)
    
    def update_files_list(self):
        """Update the selected files list"""
        self.files_list.clear()
        for file_path in self.selected_files:
            item = QListWidgetItem(f"📷 {Path(file_path).name}")
            self.files_list.addItem(item)
    
    def upload_images(self):
        """Upload selected images to the backend"""
        if not self.selected_files:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.selected_files))
        self.progress_bar.setValue(0)
        
        # Disable buttons during upload
        self.upload_btn.setEnabled(False)
        
        try:
            # Prepare files for upload
            files = []
            for i, file_path in enumerate(self.selected_files):
                with open(file_path, 'rb') as f:
                    files.append(('files', (Path(file_path).name, f, 'image/jpeg')))
                
                self.progress_bar.setValue(i + 1)
                # Process events to update UI
                self.parent().app.processEvents()
            
            # Upload to backend
            response = self.api_client.post(f"/api/items/{self.item_id}/images", files=dict(files))
            
            if response.get("success"):
                QMessageBox.information(
                    self,
                    "Upload Successful",
                    f"Successfully uploaded {len(self.selected_files)} images!"
                )
                self.accept()
            else:
                QMessageBox.warning(
                    self,
                    "Upload Failed",
                    f"Failed to upload images: {response.get('error', 'Unknown error')}"
                )
        
        except Exception as e:
            logger.error(f"Upload error: {e}")
            QMessageBox.critical(
                self,
                "Upload Error",
                f"An error occurred during upload: {str(e)}"
            )
        
        finally:
            self.progress_bar.setVisible(False)
            self.upload_btn.setEnabled(True)

class AddItemDialog(QDialog):
    """Dialog for adding a new item"""
    
    item_created = pyqtSignal(dict)
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.setup_ui()
    
    def setup_ui(self):
        """Setup add item dialog UI"""
        self.setWindowTitle("Add New Item")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Form
        form_group = QGroupBox("Item Information")
        form_layout = QFormLayout(form_group)
        
        self.item_id_edit = QLineEdit()
        self.item_id_edit.setPlaceholderText("e.g., item_004")
        form_layout.addRow("Item ID*:", self.item_id_edit)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Blue Widget")
        form_layout.addRow("Name*:", self.name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Optional description...")
        form_layout.addRow("Description:", self.description_edit)
        
        self.category_edit = QLineEdit()
        self.category_edit.setPlaceholderText("e.g., Electronics")
        form_layout.addRow("Category:", self.category_edit)
        
        layout.addWidget(form_group)
        
        # Validation message
        self.validation_label = QLabel()
        self.validation_label.setStyleSheet("color: red; font-size: 12px;")
        self.validation_label.setVisible(False)
        layout.addWidget(self.validation_label)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.create_item)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.create_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.create_btn.setText("Create Item")
        
        # Connect validation
        self.item_id_edit.textChanged.connect(self.validate_form)
        self.name_edit.textChanged.connect(self.validate_form)
        
        # Style the form
        form_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #2196F3;
            }
        """)
    
    def validate_form(self):
        """Validate form inputs"""
        item_id = self.item_id_edit.text().strip()
        name = self.name_edit.text().strip()
        
        is_valid = True
        message = ""
        
        if not item_id:
            is_valid = False
            message = "Item ID is required"
        elif not item_id.replace("_", "").replace("-", "").isalnum():
            is_valid = False
            message = "Item ID can only contain letters, numbers, hyphens, and underscores"
        elif not name:
            is_valid = False
            message = "Name is required"
        
        self.validation_label.setText(message)
        self.validation_label.setVisible(not is_valid)
        self.create_btn.setEnabled(is_valid)
    
    def create_item(self):
        """Create the new item"""
        if not self.create_btn.isEnabled():
            return
        
        item_data = {
            "item_id": self.item_id_edit.text().strip(),
            "name": self.name_edit.text().strip(),
            "description": self.description_edit.toPlainText().strip(),
            "category": self.category_edit.text().strip()
        }
        
        try:
            response = self.api_client.post("/api/items", item_data)
            
            if response.get("success"):
                self.item_created.emit(item_data)
                QMessageBox.information(
                    self,
                    "Item Created",
                    f"Item '{item_data['item_id']}' created successfully!"
                )
                self.accept()
            else:
                QMessageBox.warning(
                    self,
                    "Creation Failed",
                    f"Failed to create item: {response.get('error', 'Unknown error')}"
                )
        
        except Exception as e:
            logger.error(f"Create item error: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred: {str(e)}"
            )

class ItemCard(QFrame):
    """Individual item card widget"""
    
    item_selected = pyqtSignal(str)
    upload_requested = pyqtSignal(str)
    delete_requested = pyqtSignal(str)
    
    def __init__(self, item_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.item_data = item_data
        self.setup_ui()
    
    def setup_ui(self):
        """Setup item card UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
            }
            QFrame:hover {
                border-color: #2196F3;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
        """)
        self.setFixedSize(280, 200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Item header
        header_layout = QHBoxLayout()
        
        # Item icon/image placeholder
        icon_label = QLabel("📦")
        icon_label.setStyleSheet("font-size: 24px;")
        header_layout.addWidget(icon_label)
        
        # Item info
        info_layout = QVBoxLayout()
        
        name_label = QLabel(self.item_data.get("name", self.item_data["item_id"]))
        name_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
        """)
        name_label.setWordWrap(True)
        info_layout.addWidget(name_label)
        
        id_label = QLabel(f"ID: {self.item_data['item_id']}")
        id_label.setStyleSheet("font-size: 11px; color: #666;")
        info_layout.addWidget(id_label)
        
        header_layout.addLayout(info_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Image count
        image_count = self.item_data.get("image_count", 0)
        count_label = QLabel(f"📷 {image_count} images")
        count_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                background-color: #f0f0f0;
                padding: 4px 8px;
                border-radius: 10px;
            }
        """)
        layout.addWidget(count_label)
        
        layout.addStretch()
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        upload_btn = QPushButton("📷 Upload")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        upload_btn.clicked.connect(lambda: self.upload_requested.emit(self.item_data["item_id"]))
        
        delete_btn = QPushButton("🗑️")
        delete_btn.setFixedSize(30, 30)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.item_data["item_id"]))
        
        buttons_layout.addWidget(upload_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(delete_btn)
        
        layout.addLayout(buttons_layout)

class ItemsWidget(QWidget):
    """Main items management widget"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.items_data = []
        self.setup_ui()
        self.refresh_items()
    
    def setup_ui(self):
        """Setup items management UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("Items Management")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Add item button
        add_btn = QPushButton("📦 Add New Item")
        add_btn.setStyleSheet("""
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
        """)
        add_btn.clicked.connect(self.add_item)
        header_layout.addWidget(add_btn)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
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
        refresh_btn.clicked.connect(self.refresh_items)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Search and filter
        filter_layout = QHBoxLayout()
        
        search_edit = QLineEdit()
        search_edit.setPlaceholderText("🔍 Search items...")
        search_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ddd;
                border-radius: 20px;
                padding: 10px 15px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
        """)
        filter_layout.addWidget(search_edit)
        
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Items grid container
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        self.items_container = QWidget()
        self.items_layout = QGridLayout(self.items_container)
        self.items_layout.setSpacing(15)
        self.items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll_area.setWidget(self.items_container)
        layout.addWidget(scroll_area)
        
        # Status message
        self.status_label = QLabel("Loading items...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 16px;
                padding: 40px;
            }
        """)
        layout.addWidget(self.status_label)
    
    def refresh_items(self):
        """Refresh items from backend"""
        try:
            self.status_label.setText("Loading items...")
            self.status_label.setVisible(True)
            
            response = self.api_client.get("/api/items")
            
            if response.get("success"):
                self.items_data = response.get("data", [])
                self.update_items_display()
                
                if not self.items_data:
                    self.status_label.setText("No items found. Click 'Add New Item' to get started!")
                else:
                    self.status_label.setVisible(False)
            else:
                self.status_label.setText(f"Error loading items: {response.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error refreshing items: {e}")
            self.status_label.setText(f"Failed to connect to backend: {str(e)}")
    
    def update_items_display(self):
        """Update the items grid display"""
        # Clear existing items
        for i in reversed(range(self.items_layout.count())):
            child = self.items_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Add item cards
        row, col = 0, 0
        cols_per_row = 4
        
        for item_data in self.items_data:
            card = ItemCard(item_data)
            card.upload_requested.connect(self.upload_images)
            card.delete_requested.connect(self.delete_item)
            
            self.items_layout.addWidget(card, row, col)
            
            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1
    
    def add_item(self):
        """Show add item dialog"""
        dialog = AddItemDialog(self.api_client, self)
        dialog.item_created.connect(self.on_item_created)
        dialog.exec()
    
    def on_item_created(self, item_data: Dict[str, Any]):
        """Handle new item creation"""
        self.refresh_items()
    
    def upload_images(self, item_id: str):
        """Show image upload dialog"""
        dialog = ImageUploadDialog(item_id, self.api_client, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_items()
    
    def delete_item(self, item_id: str):
        """Delete an item"""
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete item '{item_id}'?\n\nThis will permanently remove the item and all its images.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                response = self.api_client.delete(f"/api/items/{item_id}")
                
                if response.get("success"):
                    QMessageBox.information(
                        self,
                        "Item Deleted",
                        f"Item '{item_id}' has been deleted successfully."
                    )
                    self.refresh_items()
                else:
                    QMessageBox.warning(
                        self,
                        "Delete Failed",
                        f"Failed to delete item: {response.get('error', 'Unknown error')}"
                    )
            
            except Exception as e:
                logger.error(f"Delete error: {e}")
                QMessageBox.critical(
                    self,
                    "Error",
                    f"An error occurred while deleting the item: {str(e)}"
                )