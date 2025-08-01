"""
Unified Preprocessing Module

This module provides the enhanced preprocessing pipeline with guaranteed data persistence
and 100% accuracy preservation of the existing augmentation system.
"""

from .input_manager import (
    ItemInformation,
    ImageSource, 
    ImageSourceType,
    ProcessedImageWithPersistence,
    UnifiedImageInputManager
)

from .data_persistence import (
    GuaranteedDataPersistence,
    ItemSaveResult,
    ProcessingMetadata
)

__all__ = [
    'ItemInformation',
    'ImageSource',
    'ImageSourceType', 
    'ProcessedImageWithPersistence',
    'UnifiedImageInputManager',
    'GuaranteedDataPersistence',
    'ItemSaveResult',
    'ProcessingMetadata'
]