"""
ChromaDB Vector Indexer for Scalable Recognition

This module replaces FAISS with ChromaDB to handle millions of vectors efficiently.
ChromaDB provides better scalability, persistent storage, and easier management.

FEATURES:
- Handles 2.8M+ vectors efficiently  
- Persistent storage in single directory
- Incremental updates without full rebuilds
- Native Python integration
- Better memory management than FAISS
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from datetime import datetime
import json

# ChromaDB imports
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class OptimalChromaDBIndexer:
    """
    ChromaDB-based vector indexer for scalable recognition system
    
    Replaces FAISS with ChromaDB for better scalability and management.
    Handles millions of vectors with persistent storage and fast search.
    """
    
    def __init__(self, 
                 dimension: int = 1536,
                 collection_name: str = "recognition_vectors",
                 persist_directory: str = "data/chromadb",
                 gpu_enabled: bool = False):
        """
        Initialize ChromaDB indexer
        
        Args:
            dimension: Feature vector dimension (1536 for CLIP+DINOv2)
            collection_name: ChromaDB collection name
            persist_directory: Directory to store ChromaDB data
            gpu_enabled: Not used (ChromaDB handles optimization internally)
        """
        self.dimension = dimension
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.gpu_enabled = gpu_enabled  # For compatibility
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False,  # Disable telemetry
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"dimension": dimension, "hnsw:space": "cosine"}
            )
            logger.info(f"Created new ChromaDB collection: {collection_name}")
        
        # Statistics tracking
        self.stats = {
            'indices_built': 0,
            'vectors_added': 0,
            'searches_performed': 0,
            'total_build_time': 0.0,
            'total_search_time': 0.0,
            'gpu_operations': 0
        }
        
        logger.info(f"ChromaDB indexer initialized (dim: {dimension}, collection: {collection_name})")
    
    def build_index(self, 
                   vectors: np.ndarray, 
                   item_ids: List[str],
                   index_method: Optional[str] = None) -> Dict[str, Any]:
        """
        Build/rebuild ChromaDB index with all vectors
        
        Args:
            vectors: Feature vectors (N x dimension)
            item_ids: Corresponding item IDs
            index_method: Ignored (ChromaDB optimizes automatically)
            
        Returns:
            Build result with metadata
        """
        start_time = time.time()
        
        try:
            vector_count = len(vectors)
            
            # Validate inputs
            if vector_count != len(item_ids):
                raise ValueError(f"Vector count ({vector_count}) != item ID count ({len(item_ids)})")
            
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Vector dimension ({vectors.shape[1]}) != expected ({self.dimension})")
            
            logger.info(f"Building ChromaDB index for {vector_count} vectors")
            
            # Clear existing data if rebuilding
            if self.collection.count() > 0:
                logger.info("Clearing existing vectors for rebuild")
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dimension": self.dimension, "hnsw:space": "cosine"}
                )
            
            # Prepare data for ChromaDB
            embeddings = vectors.astype(np.float32).tolist()
            ids = [f"vec_{i}_{item_id}" for i, item_id in enumerate(item_ids)]
            metadatas = [{"item_id": item_id, "vector_index": i} for i, item_id in enumerate(item_ids)]
            
            # Add vectors to ChromaDB (handles batching automatically)
            self.collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            build_time = time.time() - start_time
            
            # Update statistics
            self.stats['indices_built'] += 1
            self.stats['total_build_time'] += build_time
            self.stats['vectors_added'] += vector_count
            
            logger.info(f"ChromaDB index built successfully in {build_time:.2f}s ({vector_count} vectors)")
            
            return {
                'success': True,
                'method': 'ChromaDB-HNSW',
                'vector_count': vector_count,
                'vectors_added': vector_count,
                'build_time': build_time,
                'index_size': self.collection.count(),
                'gpu_enabled': False  # ChromaDB handles optimization
            }
            
        except Exception as e:
            logger.error(f"Failed to build ChromaDB index: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'ChromaDB-HNSW',
                'vector_count': len(vectors) if 'vectors' in locals() else 0
            }
    
    def add_vectors_incremental(self, 
                               new_vectors: np.ndarray, 
                               new_item_ids: List[str]) -> Dict[str, Any]:
        """
        Add new vectors to existing ChromaDB collection
        
        Args:
            new_vectors: New feature vectors to add
            new_item_ids: Corresponding item IDs
            
        Returns:
            Addition result
        """
        start_time = time.time()
        
        try:
            vector_count = len(new_vectors)
            if vector_count != len(new_item_ids):
                raise ValueError(f"Vector count ({vector_count}) != item ID count ({len(new_item_ids)})")
            
            # Get current count for unique IDs
            current_count = self.collection.count()
            
            # Prepare data for ChromaDB
            embeddings = new_vectors.astype(np.float32).tolist()
            ids = [f"vec_{current_count + i}_{item_id}" for i, item_id in enumerate(new_item_ids)]
            metadatas = [{"item_id": item_id, "vector_index": current_count + i} for i, item_id in enumerate(new_item_ids)]
            
            # Add vectors incrementally
            self.collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            add_time = time.time() - start_time
            
            # Update statistics
            self.stats['vectors_added'] += vector_count
            
            logger.info(f"Added {vector_count} vectors incrementally in {add_time:.2f}s")
            
            return {
                'success': True,
                'method': 'ChromaDB-HNSW',
                'vectors_added': vector_count,
                'add_time': add_time,
                'total_vectors': self.collection.count()
            }
            
        except Exception as e:
            logger.error(f"Failed to add vectors incrementally: {e}")
            return {
                'success': False,
                'error': str(e),
                'vectors_added': 0
            }
    
    def search(self, 
              query_vector: np.ndarray, 
              k: int = 10,
              return_distances: bool = True) -> Dict[str, Any]:
        """
        Search for similar vectors using ChromaDB
        
        Args:
            query_vector: Query feature vector
            k: Number of results to return
            return_distances: Include similarity scores
            
        Returns:
            Search results with item IDs and similarities
        """
        start_time = time.time()
        
        try:
            # Ensure query vector is correct shape and type
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            if query_vector.shape[1] != self.dimension:
                raise ValueError(f"Query dimension ({query_vector.shape[1]}) != expected ({self.dimension})")
            
            # Search using ChromaDB
            results = self.collection.query(
                query_embeddings=query_vector.astype(np.float32).tolist(),
                n_results=k,
                include=['metadatas', 'distances']
            )
            
            search_time = time.time() - start_time
            
            # Update statistics
            self.stats['searches_performed'] += 1
            self.stats['total_search_time'] += search_time
            
            # Format results for compatibility with existing system
            formatted_results = []
            
            if results['ids'] and len(results['ids'][0]) > 0:
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                distances = results['distances'][0] if results['distances'] else []
                
                for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
                    # Convert distance to similarity (ChromaDB returns distances, we want similarity)
                    similarity = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
                    
                    formatted_results.append({
                        'rank': i + 1,
                        'item_id': metadata.get('item_id', 'unknown'),
                        'similarity': similarity,
                        'distance': distance,
                        'metadata': metadata
                    })
            
            return {
                'success': True,
                'results': formatted_results,
                'search_time': search_time,
                'total_vectors': self.collection.count(),
                'index_method': 'ChromaDB-HNSW'
            }
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'search_time': time.time() - start_time
            }
    
    def rebuild_index_from_database(self, conn=None) -> Dict[str, Any]:
        """
        Rebuild ChromaDB index from all vectors in SQLite database
        
        Args:
            conn: SQLite connection (optional)
            
        Returns:
            Rebuild result
        """
        try:
            logger.info("Rebuilding ChromaDB index from database...")
            
            # Get all vectors from database  
            vectors, item_ids = self._get_all_vectors_from_database(conn)
            
            if len(vectors) == 0:
                return {
                    'success': False,
                    'error': 'No vectors found in database',
                    'vector_count': 0
                }
            
            # Build new index
            result = self.build_index(vectors, item_ids)
            
            if result['success']:
                logger.info(f"ChromaDB index rebuilt successfully from database ({len(vectors)} vectors)")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to rebuild ChromaDB index from database: {e}")
            return {
                'success': False,
                'error': str(e),
                'vector_count': 0
            }
    
    def _get_all_vectors_from_database(self, conn=None) -> Tuple[np.ndarray, List[str]]:
        """Get all vectors from database for index rebuilding"""
        try:
            import sqlite3
            
            # Use provided connection or create new one
            if conn is not None:
                cursor = conn.execute("""
                    SELECT fv.feature_vector, fv.item_id
                    FROM feature_vectors fv
                    WHERE fv.feature_dimension = ?
                    ORDER BY fv.extraction_timestamp
                """, (self.dimension,))
                
                vectors = []
                item_ids = []
                
                for row in cursor:
                    vector_blob, item_id = row
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    vectors.append(vector)
                    item_ids.append(item_id)
                
                if vectors:
                    vectors = np.array(vectors)
                    logger.info(f"Loaded {len(vectors)} vectors from database")
                    return vectors, item_ids
                else:
                    return np.empty((0, self.dimension), dtype=np.float32), []
            else:
                # This shouldn't happen in practice, but handle it
                logger.warning("No database connection provided for vector loading")
                return np.empty((0, self.dimension), dtype=np.float32), []
                    
        except Exception as e:
            logger.error(f"Failed to get vectors from database: {e}")
            return np.empty((0, self.dimension), dtype=np.float32), []
    
    def load_index_from_database(self) -> bool:
        """
        Load ChromaDB collection (automatically persistent)
        
        Returns:
            True if collection exists and is loaded
        """
        try:
            # ChromaDB is automatically persistent, just check if collection has data
            count = self.collection.count()
            
            if count > 0:
                logger.info(f"ChromaDB collection loaded successfully ({count} vectors)")
                return True
            else:
                logger.info("ChromaDB collection is empty")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load ChromaDB collection: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        try:
            stats = self.stats.copy()
            
            # Add current collection info
            vector_count = self.collection.count()
            stats['current_index'] = {
                'method': 'ChromaDB-HNSW',
                'vector_count': vector_count,
                'dimension': self.dimension,
                'gpu_enabled': False,  # ChromaDB handles optimization internally
                'collection_name': self.collection_name
            }
            
            # Calculate averages
            if self.stats['searches_performed'] > 0:
                stats['average_search_time'] = self.stats['total_search_time'] / self.stats['searches_performed']
            else:
                stats['average_search_time'] = 0.0
                
            if self.stats['indices_built'] > 0:
                stats['average_build_time'] = self.stats['total_build_time'] / self.stats['indices_built']
            else:
                stats['average_build_time'] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get ChromaDB statistics: {e}")
            return self.stats
    
    @property
    def current_index(self):
        """Compatibility property for existing code"""
        return self.collection
    
    @property
    def current_index_type(self):
        """Compatibility property for existing code"""
        return "ChromaDB-HNSW"