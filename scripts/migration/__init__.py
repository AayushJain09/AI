"""
Migration Framework for AI Recognition System

Migrates from legacy scattered file architecture to unified storage system.

Components:
- migrate_features.py: Migrate feature vectors from HDF5 files
- migrate_faiss_indices.py: Consolidate FAISS indices
- migrate_metadata.py: Extract metadata from directory structure
- validate_migration.py: Comprehensive validation and verification
- orchestrator.py: Complete migration workflow

The migration framework ensures data integrity while transitioning to the
optimized unified storage architecture with minimal downtime.
"""

__version__ = "1.0.0"
__all__ = [
    "migrate_features",
    "migrate_faiss_indices", 
    "migrate_metadata",
    "validate_migration",
    "orchestrator"
]