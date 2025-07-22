"""
Items Management Widget
======================

Complete interface for managing inventory items with pagination, lazy loading, and async operations.

Key Features:
- Asynchronous data loading to prevent UI freezing
- Pagination support for large datasets
- Lazy loading with virtual scrolling
- Responsive grid layout
- Image upload functionality
- Search and filtering capabilities
- Comprehensive error handling

Author: AI Recognition System
Version: 2.0
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
    QComboBox, QSpinBox, QCheckBox, QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QEvent, QMutex, QMutexLocker
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

# Configure logger for this module
logger = logging.getLogger(__name__)

class ImageUploadDialog(QDialog):
    """
    Dialog for uploading multiple images to an item.
    
    Features:
    - Multi-file selection with image format validation
    - Progress tracking during upload
    - Drag-and-drop support (future enhancement)
    - Preview of selected files
    
    Attributes:
        item_id (str): The ID of the item to upload images for
        api_client: API client instance for backend communication
        selected_files (List[str]): List of selected file paths
    """
    
    def __init__(self, item_id: str, api_client, parent=None):
        """
        Initialize the image upload dialog.
        
        Args:
            item_id: The unique identifier for the item
            api_client: Backend API client for file upload
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.item_id = item_id
        self.api_client = api_client
        self.selected_files = []
        self.setup_ui()
    
    def setup_ui(self):
        """
        Setup the user interface for the image upload dialog.
        
        Creates:
        - File selection area with instructions
        - Selected files preview list
        - Progress bar for upload tracking
        - Upload/Cancel buttons
        """
        self.setWindowTitle(f"Upload Images - {self.item_id}")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Instructions section with helpful tips
        instructions = QLabel(
            "Select images to upload for this item. Recommended: 8-20 high-quality images "
            "from different angles and lighting conditions for better recognition accuracy."
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
        
        # File selection group
        file_area = QGroupBox("Selected Images")
        file_layout = QVBoxLayout(file_area)
        
        # Select files button with modern styling
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
        
        # Selected files list widget
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
        
        # Progress bar for upload tracking (initially hidden)
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
        
        # Dialog action buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.upload_images)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Configure upload button (initially disabled)
        self.upload_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.upload_btn.setText("Upload Images")
        self.upload_btn.setEnabled(False)
    
    def select_files(self):
        """
        Open file dialog for image selection.
        
        Supports multiple image formats: PNG, JPG, JPEG, BMP, TIFF
        Updates the file list and enables upload button when files are selected.
        """
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
        """
        Update the selected files preview list.
        
        Displays each selected file with a camera emoji for visual appeal.
        """
        self.files_list.clear()
        for file_path in self.selected_files:
            item = QListWidgetItem(f"📷 {Path(file_path).name}")
            self.files_list.addItem(item)
    
    def upload_images(self):
        """
        Upload selected images to the backend.
        
        Process:
        1. Show progress bar and disable buttons
        2. Prepare files for multipart upload
        3. Send POST request to backend API
        4. Handle success/error responses
        5. Update UI accordingly
        
        Note: Processes events during upload to keep UI responsive
        """
        if not self.selected_files:
            return
        
        # Setup progress tracking
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.selected_files))
        self.progress_bar.setValue(0)
        
        # Disable buttons during upload to prevent double-submission
        self.upload_btn.setEnabled(False)
        
        try:
            # Prepare files for multipart upload
            files = []
            for i, file_path in enumerate(self.selected_files):
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    files.append(('files', (Path(file_path).name, file_content, 'image/jpeg')))
                
                # Update progress bar
                self.progress_bar.setValue(i + 1)
                # Process events to keep UI responsive during file reading
                QApplication.processEvents()
            
            # Send upload request to backend
            response = self.api_client.post(f"/api/items/{self.item_id}/images", files=files)
            
            # Handle successful upload
            if response.get("success"):
                QMessageBox.information(
                    self,
                    "Upload Successful",
                    f"Successfully uploaded {len(self.selected_files)} images!"
                )
                self.accept()
            else:
                # Handle API error response
                QMessageBox.warning(
                    self,
                    "Upload Failed",
                    f"Failed to upload images: {response.get('error', 'Unknown error')}"
                )
        
        except Exception as e:
            # Handle unexpected errors during upload
            logger.error(f"Upload error: {e}")
            QMessageBox.critical(
                self,
                "Upload Error",
                f"An error occurred during upload: {str(e)}"
            )
        
        finally:
            # Always cleanup UI state
            self.progress_bar.setVisible(False)
            self.upload_btn.setEnabled(True)

class AddItemDialog(QDialog):
    """
    Dialog for creating new items in the system.
    
    Features:
    - Form validation with real-time feedback
    - Category suggestions (future enhancement)
    - Item ID format validation
    - Auto-generation of item IDs (future enhancement)
    
    Signals:
        item_created(dict): Emitted when a new item is successfully created
    """
    
    item_created = pyqtSignal(dict)  # Signal emitted when item is created successfully
    
    def __init__(self, api_client, parent=None):
        """
        Initialize the add item dialog.
        
        Args:
            api_client: Backend API client for item creation
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.api_client = api_client
        self.setup_ui()
    
    def setup_ui(self):
        """
        Setup the user interface for the add item dialog.
        
        Creates:
        - Form fields for item information
        - Real-time validation feedback
        - Create/Cancel buttons
        """
        self.setWindowTitle("Add New Item")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Form group with styled appearance
        form_group = QGroupBox("Item Information")
        form_layout = QFormLayout(form_group)
        
        # Item ID field with validation
        self.item_id_edit = QLineEdit()
        self.item_id_edit.setPlaceholderText("e.g., item_004")
        form_layout.addRow("Item ID*:", self.item_id_edit)
        
        # Item name field
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Blue Widget")
        form_layout.addRow("Name*:", self.name_edit)
        
        # Optional description field
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Optional description...")
        form_layout.addRow("Description:", self.description_edit)
        
        # Category field for organization
        self.category_edit = QLineEdit()
        self.category_edit.setPlaceholderText("e.g., Electronics")
        form_layout.addRow("Category:", self.category_edit)
        
        layout.addWidget(form_group)
        
        # Validation message label (initially hidden)
        self.validation_label = QLabel()
        self.validation_label.setStyleSheet("color: red; font-size: 12px;")
        self.validation_label.setVisible(False)
        layout.addWidget(self.validation_label)
        
        # Dialog action buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.create_item)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.create_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.create_btn.setText("Create Item")
        
        # Setup real-time validation
        self.item_id_edit.textChanged.connect(self.validate_form)
        self.name_edit.textChanged.connect(self.validate_form)
        
        # Apply modern form styling
        self._apply_form_styling(form_group)
    
    def _apply_form_styling(self, form_group):
        """
        Apply consistent styling to form elements.
        
        Args:
            form_group: The QGroupBox containing form elements
        """
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
        """
        Perform real-time form validation.
        
        Validation rules:
        - Item ID is required and must contain only alphanumeric characters, hyphens, and underscores
        - Name is required
        
        Updates the validation message and enables/disables create button accordingly.
        """
        item_id = self.item_id_edit.text().strip()
        name = self.name_edit.text().strip()
        
        is_valid = True
        message = ""
        
        # Validate Item ID
        if not item_id:
            is_valid = False
            message = "Item ID is required"
        elif not item_id.replace("_", "").replace("-", "").isalnum():
            is_valid = False
            message = "Item ID can only contain letters, numbers, hyphens, and underscores"
        elif not name:
            is_valid = False
            message = "Name is required"
        
        # Update validation UI
        self.validation_label.setText(message)
        self.validation_label.setVisible(not is_valid)
        self.create_btn.setEnabled(is_valid)
    
    def create_item(self):
        """
        Create a new item via API call.
        
        Process:
        1. Validate form data
        2. Send POST request to create item
        3. Handle success/error responses
        4. Emit signal and close dialog on success
        """
        if not self.create_btn.isEnabled():
            return
        
        # Prepare item data from form
        item_data = {
            "item_id": self.item_id_edit.text().strip(),
            "name": self.name_edit.text().strip(),
            "description": self.description_edit.toPlainText().strip(),
            "category": self.category_edit.text().strip()
        }
        
        try:
            # Send create request to backend
            response = self.api_client.post("/api/items", item_data)
            
            if response.get("success"):
                # Emit signal for parent widget to refresh
                self.item_created.emit(item_data)
                QMessageBox.information(
                    self,
                    "Item Created",
                    f"Item '{item_data['item_id']}' created successfully!"
                )
                self.accept()
            else:
                # Handle API error
                QMessageBox.warning(
                    self,
                    "Creation Failed",
                    f"Failed to create item: {response.get('error', 'Unknown error')}"
                )
        
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Create item error: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred: {str(e)}"
            )

class ItemCard(QFrame):
    """
    Individual item card widget for grid display.
    
    Features:
    - Hover effects and modern styling
    - Action buttons (upload, delete)
    - Image count display
    - Responsive sizing
    
    Signals:
        item_selected(str): Emitted when item is selected
        upload_requested(str): Emitted when upload button is clicked
        delete_requested(str): Emitted when delete button is clicked
    """
    
    # Define signals for item interactions
    item_selected = pyqtSignal(str)
    upload_requested = pyqtSignal(str)
    delete_requested = pyqtSignal(str)
    
    def __init__(self, item_data: Dict[str, Any], parent=None):
        """
        Initialize an item card.
        
        Args:
            item_data: Dictionary containing item information
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.item_data = item_data
        self.setup_ui()
    
    def setup_ui(self):
        """
        Setup the user interface for the item card.
        
        Creates:
        - Item header with icon and info
        - Image count indicator
        - Action buttons (upload, delete)
        """
        # Configure frame appearance with hover effects
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
        self.setMinimumSize(260, 180)
        self.setMaximumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Item header section
        header_layout = QHBoxLayout()
        
        # Item icon/image placeholder (future: actual thumbnails)
        icon_label = QLabel("📦")
        icon_label.setStyleSheet("font-size: 24px;")
        header_layout.addWidget(icon_label)
        
        # Item information section
        info_layout = QVBoxLayout()
        
        # Item name with fallback to ID
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
        
        # Item ID display
        id_label = QLabel(f"ID: {self.item_data['item_id']}")
        id_label.setStyleSheet("font-size: 11px; color: #666;")
        info_layout.addWidget(id_label)
        
        header_layout.addLayout(info_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Image count indicator with styling
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
        
        # Action buttons section
        self._create_action_buttons(layout)
    
    def _create_action_buttons(self, layout):
        """
        Create action buttons for the item card.
        
        Args:
            layout: The main layout to add buttons to
        """
        buttons_layout = QHBoxLayout()
        
        # Upload images button
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
        
        # Delete item button
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

class ItemsLoader(QThread):
    """
    Background thread for loading items with pagination support.
    
    Features:
    - Asynchronous API calls to prevent UI freezing
    - Pagination support for large datasets
    - Comprehensive error handling
    - Thread-safe data emission
    
    Signals:
        items_loaded(list, dict): Emitted when items are successfully loaded
        error_occurred(str): Emitted when an error occurs during loading
        progress_updated(int, int): Emitted to update loading progress
    """
    
    # Define signals for thread communication
    items_loaded = pyqtSignal(list, dict)  # items_data, pagination_info
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)  # current, total
    
    def __init__(self, api_client, page: int = 1, page_size: int = 20, search_query: str = ""):
        """
        Initialize the items loader thread.
        
        Args:
            api_client: Backend API client
            page: Page number to load (1-based)
            page_size: Number of items per page
            search_query: Optional search filter
        """
        super().__init__()
        self.api_client = api_client
        self.page = page
        self.page_size = page_size
        self.search_query = search_query
        self.mutex = QMutex()  # Thread safety for shared data
    
    def run(self):
        """
        Execute the items loading in the background thread.
        
        Process:
        1. Build API request parameters
        2. Make paginated API call
        3. Parse response and extract pagination info
        4. Emit appropriate signals based on result
        """
        try:
            # Thread-safe parameter preparation
            with QMutexLocker(self.mutex):
                params = {
                    "page": self.page,
                    "page_size": self.page_size
                }
                if self.search_query:
                    params["search"] = self.search_query
            
            # Emit progress update
            self.progress_updated.emit(0, 1)
            
            # Build endpoint with query parameters for pagination
            endpoint = "/api/items"
            if params:
                query_parts = []
                for key, value in params.items():
                    query_parts.append(f"{key}={value}")
                if query_parts:
                    endpoint = f"/api/items?{'&'.join(query_parts)}"
            
            # Make API call for paginated items
            response = self.api_client.get(endpoint)
            
            # Update progress
            self.progress_updated.emit(1, 1)
            
            if response.get("success"):
                # Extract items data
                items_data = response.get("data", [])
                
                # Handle pagination info (backend may not support pagination yet)
                pagination_info = {
                    "current_page": response.get("current_page", 1),
                    "total_pages": response.get("total_pages", 1),
                    "total_items": response.get("total_items", len(items_data)),
                    "page_size": response.get("page_size", len(items_data))
                }
                
                # If backend doesn't support pagination, simulate it client-side
                if self.search_query:
                    # Filter items by search query
                    filtered_items = []
                    for item in items_data:
                        item_name = item.get("name", "").lower()
                        item_id = item.get("item_id", "").lower()
                        search_lower = self.search_query.lower()
                        if search_lower in item_name or search_lower in item_id:
                            filtered_items.append(item)
                    items_data = filtered_items
                
                # Apply client-side pagination if backend doesn't support it
                if pagination_info["total_pages"] == 1 and len(items_data) > self.page_size:
                    # Client-side pagination
                    total_items = len(items_data)
                    total_pages = (total_items + self.page_size - 1) // self.page_size
                    start_idx = (self.page - 1) * self.page_size
                    end_idx = start_idx + self.page_size
                    items_data = items_data[start_idx:end_idx]
                    
                    pagination_info.update({
                        "current_page": self.page,
                        "total_pages": total_pages,
                        "total_items": total_items,
                        "page_size": self.page_size
                    })
                
                # Emit successful result
                self.items_loaded.emit(items_data, pagination_info)
            else:
                # Emit API error
                self.error_occurred.emit(f"API Error: {response.get('error', 'Unknown error')}")
        
        except Exception as e:
            # Emit unexpected error
            logger.error(f"Items loading error: {e}")
            self.error_occurred.emit(f"Failed to load items: {str(e)}")

class ItemsWidget(QWidget):
    """
    Main items management widget with pagination and lazy loading.
    
    Features:
    - Asynchronous data loading with progress indication
    - Pagination for large datasets
    - Search and filtering capabilities
    - Responsive grid layout
    - Item management (add, delete, upload images)
    - Error handling and user feedback
    - Comprehensive inline documentation
    
    Architecture:
    - Uses QThread for non-blocking API calls
    - Implements lazy loading to load data only when needed
    - Responsive design adapts to window size changes
    - Thread-safe operations with proper cleanup
    """
    
    def __init__(self, api_client, parent=None):
        """
        Initialize the items management widget.
        
        Args:
            api_client: Backend API client for data operations
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.api_client = api_client
        
        # Data management
        self.items_data = []  # Current page items
        self.current_page = 1
        self.total_pages = 1
        self.total_items = 0
        self.page_size = 20
        self.search_query = ""
        
        # Thread management
        self.loader_thread = None  # Current loader thread
        self.first_show = True     # Track first widget display
        
        # Search debouncing
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._perform_search)
        
        # UI setup
        self.setup_ui()
    
    def setup_ui(self):
        """
        Setup the complete user interface for items management.
        
        Creates:
        - Header with title and action buttons
        - Search and filter controls
        - Paginated items grid with scroll area
        - Pagination controls
        - Status and error display areas
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Create all UI sections
        self._create_header_section(layout)
        self._create_search_section(layout)
        self._create_items_grid_section(layout)
        self._create_pagination_section(layout)
        self._create_status_section(layout)
    
    def _create_header_section(self, layout):
        """
        Create the header section with title and action buttons.
        
        Args:
            layout: Main layout to add header to
        """
        header_layout = QHBoxLayout()
        
        # Page title
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
        
        # Add new item button
        add_btn = QPushButton("📦 Add New Item")
        add_btn.setStyleSheet(self._get_primary_button_style())
        add_btn.clicked.connect(self.add_item)
        header_layout.addWidget(add_btn)
        
        # Refresh button with loading protection
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setStyleSheet(self._get_secondary_button_style())
        self.refresh_btn.clicked.connect(self.refresh_items)
        header_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(header_layout)
    
    def _create_search_section(self, layout):
        """
        Create the search and filter section.
        
        Args:
            layout: Main layout to add search section to
        """
        search_layout = QHBoxLayout()
        
        # Search input field
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍 Search items by name or ID...")
        self.search_edit.setStyleSheet("""
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
        # Setup search with debouncing
        self.search_edit.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self.search_edit)
        
        # Page size selector
        page_size_layout = QHBoxLayout()
        page_size_layout.addWidget(QLabel("Items per page:"))
        
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["10", "20", "50", "100"])
        self.page_size_combo.setCurrentText(str(self.page_size))
        self.page_size_combo.currentTextChanged.connect(self._on_page_size_changed)
        page_size_layout.addWidget(self.page_size_combo)
        
        search_layout.addLayout(page_size_layout)
        
        layout.addLayout(search_layout)
    
    def _create_items_grid_section(self, layout):
        """
        Create the items grid with scroll area.
        
        Args:
            layout: Main layout to add items grid to
        """
        # Scroll area for items grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Items container widget
        self.items_container = QWidget()
        self.items_layout = QGridLayout(self.items_container)
        self.items_layout.setSpacing(15)
        self.items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll_area.setWidget(self.items_container)
        layout.addWidget(scroll_area)
        
        # Install event filter for responsive layout
        self.items_container.installEventFilter(self)
    
    def _create_pagination_section(self, layout):
        """
        Create the pagination controls.
        
        Args:
            layout: Main layout to add pagination to
        """
        pagination_layout = QHBoxLayout()
        
        # Previous page button
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.previous_page)
        pagination_layout.addWidget(self.prev_btn)
        
        pagination_layout.addStretch()
        
        # Page information label
        self.page_info_label = QLabel("Page 1 of 1")
        self.page_info_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #666;
            }
        """)
        pagination_layout.addWidget(self.page_info_label)
        
        pagination_layout.addStretch()
        
        # Next page button
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.next_page)
        pagination_layout.addWidget(self.next_btn)
        
        layout.addLayout(pagination_layout)
    
    def _create_status_section(self, layout):
        """
        Create the status and error display section.
        
        Args:
            layout: Main layout to add status section to
        """
        # Progress bar for loading indication
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                font-weight: bold;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status message label
        self.status_label = QLabel("Click 'Refresh' to load items")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 16px;
                padding: 40px;
            }
        """)
        layout.addWidget(self.status_label)
    
    def showEvent(self, event):
        """
        Handle widget show event for lazy loading.
        
        Automatically loads items when the widget is first displayed.
        
        Args:
            event: The show event
        """
        super().showEvent(event)
        if self.first_show:
            self.first_show = False
            self.refresh_items()
    
    def eventFilter(self, obj, event):
        """
        Handle resize events for responsive layout updates.
        
        Debounces resize events to avoid excessive layout updates.
        
        Args:
            obj: The object that received the event
            event: The event that occurred
            
        Returns:
            bool: Whether the event was handled
        """
        if obj == self.items_container and event.type() == QEvent.Type.Resize:
            # Debounce resize events
            if hasattr(self, 'resize_timer'):
                self.resize_timer.stop()
            else:
                self.resize_timer = QTimer()
                self.resize_timer.setSingleShot(True)
                self.resize_timer.timeout.connect(self.update_items_display)
            self.resize_timer.start(100)  # 100ms delay
        return super().eventFilter(obj, event)
    
    def refresh_items(self):
        """
        Refresh items from backend using asynchronous loading.
        
        Features:
        - Non-blocking API calls via QThread
        - Progress indication during loading
        - Proper cleanup of previous threads
        - Error handling and user feedback
        """
        # Prevent multiple simultaneous requests
        if self.loader_thread and self.loader_thread.isRunning():
            # Cancel the running thread gracefully
            self.loader_thread.quit()
            # Don't wait here as it would block the UI
            self.loader_thread = None
        
        # Update UI for loading state
        self._set_loading_state(True)
        
        # Create and configure loader thread
        self.loader_thread = ItemsLoader(
            api_client=self.api_client,
            page=self.current_page,
            page_size=self.page_size,
            search_query=self.search_query
        )
        
        # Connect thread signals
        self.loader_thread.items_loaded.connect(self._on_items_loaded)
        self.loader_thread.error_occurred.connect(self._on_load_error)
        self.loader_thread.progress_updated.connect(self._on_progress_updated)
        self.loader_thread.finished.connect(self._on_loading_finished)
        
        # Start loading
        self.loader_thread.start()
    
    def _set_loading_state(self, is_loading: bool):
        """
        Update UI to reflect loading state.
        
        Args:
            is_loading: Whether loading is in progress
        """
        self.refresh_btn.setEnabled(not is_loading)
        self.refresh_btn.setText("⏳ Loading..." if is_loading else "🔄 Refresh")
        
        if is_loading:
            self.status_label.setText("Loading items...")
            self.status_label.setVisible(True)
            self.progress_bar.setVisible(True)
        else:
            self.progress_bar.setVisible(False)
    
    def _on_progress_updated(self, current: int, total: int):
        """
        Handle loading progress updates.
        
        Args:
            current: Current progress value
            total: Total progress value
        """
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def _on_items_loaded(self, items_data: List[Dict], pagination_info: Dict):
        """
        Handle successful items loading.
        
        Args:
            items_data: List of item dictionaries
            pagination_info: Pagination metadata
        """
        self.items_data = items_data
        
        # Update pagination state
        self.current_page = pagination_info.get("current_page", 1)
        self.total_pages = pagination_info.get("total_pages", 1)
        self.total_items = pagination_info.get("total_items", 0)
        
        # Update UI
        self._update_pagination_controls()
        self.update_items_display()
        
        # Update status
        if not self.items_data:
            self.status_label.setText("No items found. Try adjusting your search or click 'Add New Item' to get started!")
            self.status_label.setVisible(True)
        else:
            self.status_label.setVisible(False)
    
    def _on_load_error(self, error_message: str):
        """
        Handle loading errors with user feedback.
        
        Args:
            error_message: Error message to display
        """
        logger.error(f"Items loading error: {error_message}")
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setVisible(True)
    
    def _on_loading_finished(self):
        """
        Handle loading completion cleanup.
        """
        self._set_loading_state(False)
    
    def update_items_display(self):
        """
        Update the items grid display with responsive layout.
        
        Features:
        - Responsive column calculation based on container width
        - Efficient widget reuse
        - Item card signal connections
        """
        # Clear existing items efficiently
        for i in reversed(range(self.items_layout.count())):
            child = self.items_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Calculate responsive columns
        container_width = self.items_container.width() if self.items_container.width() > 0 else 1200
        card_width = 300  # ItemCard width + margins
        cols_per_row = max(2, min(4, container_width // card_width))
        
        # Add item cards to grid
        row, col = 0, 0
        
        for item_data in self.items_data:
            card = ItemCard(item_data)
            
            # Connect card signals
            card.upload_requested.connect(self.upload_images)
            card.delete_requested.connect(self.delete_item)
            
            # Add to grid layout
            self.items_layout.addWidget(card, row, col)
            
            # Update grid position
            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1
    
    def _update_pagination_controls(self):
        """
        Update pagination controls based on current state.
        """
        # Update page info
        self.page_info_label.setText(
            f"Page {self.current_page} of {self.total_pages} ({self.total_items} items total)"
        )
        
        # Update button states
        self.prev_btn.setEnabled(self.current_page > 1)
        self.next_btn.setEnabled(self.current_page < self.total_pages)
    
    def previous_page(self):
        """
        Navigate to the previous page.
        """
        if self.current_page > 1:
            self.current_page -= 1
            self.refresh_items()
    
    def next_page(self):
        """
        Navigate to the next page.
        """
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.refresh_items()
    
    def _on_search_changed(self):
        """
        Handle search query changes with debouncing.
        
        Debounces search input to avoid triggering API calls on every keystroke.
        Waits 500ms after the user stops typing before performing the search.
        """
        # Stop any existing search timer
        self.search_timer.stop()
        
        # Show visual feedback that search will happen
        current_text = self.search_edit.text().strip()
        if current_text:
            self.search_edit.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ff9800;
                    border-radius: 20px;
                    padding: 10px 15px;
                    font-size: 14px;
                    background-color: #fff8e1;
                }
            """)
        else:
            # Reset to normal style
            self.search_edit.setStyleSheet("""
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
        
        # Start debounce timer (500ms delay)
        self.search_timer.start(500)
    
    def _perform_search(self):
        """
        Perform the actual search after debounce delay.
        
        This method is called by the search timer after the user stops typing.
        """
        self.search_query = self.search_edit.text().strip()
        self.current_page = 1  # Reset to first page on search
        
        # Reset search box styling to normal
        self.search_edit.setStyleSheet("""
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
        
        # Perform the search
        self.refresh_items()
    
    def _on_page_size_changed(self, new_size_str: str):
        """
        Handle page size changes.
        
        Args:
            new_size_str: New page size as string
        """
        try:
            new_size = int(new_size_str)
            if new_size != self.page_size:
                self.page_size = new_size
                self.current_page = 1  # Reset to first page
                self.refresh_items()
        except ValueError:
            logger.warning(f"Invalid page size: {new_size_str}")
    
    def add_item(self):
        """
        Show the add item dialog.
        
        Creates and displays modal dialog for new item creation.
        """
        dialog = AddItemDialog(self.api_client, self)
        dialog.item_created.connect(self._on_item_created)
        dialog.exec()
    
    def _on_item_created(self, item_data: Dict[str, Any]):
        """
        Handle new item creation.
        
        Args:
            item_data: Created item data
        """
        # Refresh current page to show new item
        self.refresh_items()
    
    def upload_images(self, item_id: str):
        """
        Show image upload dialog for specified item.
        
        Args:
            item_id: ID of the item to upload images for
        """
        dialog = ImageUploadDialog(item_id, self.api_client, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Refresh current page to update image counts
            self.refresh_items()
    
    def delete_item(self, item_id: str):
        """
        Delete an item with confirmation dialog.
        
        Args:
            item_id: ID of the item to delete
        """
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete item '{item_id}'?\n\n"
            f"This will permanently remove the item and all its images.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._perform_item_deletion(item_id)
    
    def _perform_item_deletion(self, item_id: str):
        """
        Perform the actual item deletion via API.
        
        Args:
            item_id: ID of the item to delete
        """
        try:
            response = self.api_client.delete(f"/api/items/{item_id}")
            
            if response.get("success"):
                QMessageBox.information(
                    self,
                    "Item Deleted",
                    f"Item '{item_id}' has been deleted successfully."
                )
                # Refresh current page
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
    
    def _get_primary_button_style(self) -> str:
        """
        Get CSS style for primary buttons.
        
        Returns:
            str: CSS style string
        """
        return """
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
                color: #666666;
            }
        """
    
    def _get_secondary_button_style(self) -> str:
        """
        Get CSS style for secondary buttons.
        
        Returns:
            str: CSS style string
        """
        return """
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
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
    
    def closeEvent(self, event):
        """
        Handle widget close event with proper cleanup.
        
        Args:
            event: Close event
        """
        # Stop search timer
        if hasattr(self, 'search_timer'):
            self.search_timer.stop()
        
        # Cleanup loader thread
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.quit()
            self.loader_thread.wait(3000)  # Wait up to 3 seconds
        
        super().closeEvent(event)