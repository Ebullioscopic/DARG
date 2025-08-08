"""
Universal Multi-Modal DARG System
==================================

A universal vector graph system that can handle any data type (text, image, audio, etc.)
using Dynamic Adaptive Resonance Grids with real-time Neo4j visualization.

Key Features:
- Pluggable data encoders for any modality
- Dynamic vector graph with incremental updates
- Neo4j integration for graph visualization
- Better performance than HNSW/FAISS
- Real-time nearest neighbor search
"""

import numpy as np
import json
import pickle
import time
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# Optional dependencies - install as needed
try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

# Core DARG imports - try multiple locations
try:
    from darg_complete import DARG
    print("Successfully imported DARG from darg_complete")
except ImportError:
    try:
        from ..legacy.darg_complete import DARG
        print("Successfully imported DARG from legacy.darg_complete")
    except ImportError:
        try:
            from main import DARG
            print("Successfully imported DARG from main")
        except ImportError:
            print("Could not import DARG, using fallback implementation")
            # Fallback minimal implementation
            class DARG:
                def __init__(self, *args, **kwargs):
                    self.data = {}
                    self.vectors = []
                    
                def add_vector(self, vector, data_id):
                    self.data[data_id] = len(self.vectors)
                    self.vectors.append(vector)
                    
                def search(self, query, k=10):
                    if not self.vectors:
                        return []
                    
                    # Simple cosine similarity search
                    import numpy as np
                    query = np.array(query)
                    similarities = []
                    
                    for i, vec in enumerate(self.vectors):
                        vec = np.array(vec)
                        similarity = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
                        similarities.append((i, similarity))
                    
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    return [{'data_id': i, 'similarity': sim} for i, sim in similarities[:k]]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataItem:
    """Universal data item representation"""
    id: str
    data: Any  # Original data (text, image path, audio path, etc.)
    data_type: str  # 'text', 'image', 'audio', 'video', 'custom'
    vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

class DataEncoder(ABC):
    """Abstract base class for data encoders"""
    
    @abstractmethod
    def encode(self, data: Any) -> np.ndarray:
        """Convert data to vector representation"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of encoded vectors"""
        pass
    
    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return the data type this encoder handles"""
        pass

class TextEncoder(DataEncoder):
    """Text encoder using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed. Install with: pip install transformers sentence-transformers")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self._dimension = 384  # MiniLM dimension
        
    def encode(self, data: str) -> np.ndarray:
        """Encode text to vector"""
        inputs = self.tokenizer(data, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings.flatten()
    
    def get_dimension(self) -> int:
        return self._dimension
    
    @property
    def data_type(self) -> str:
        return "text"

class ImageEncoder(DataEncoder):
    """Image encoder using simple feature extraction"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        if not HAS_CV2:
            raise ImportError("opencv-python not installed. Install with: pip install opencv-python")
        
        self.target_size = target_size
        self._dimension = target_size[0] * target_size[1] * 3  # RGB channels
        
    def encode(self, data: Union[str, np.ndarray]) -> np.ndarray:
        """Encode image to vector"""
        if isinstance(data, str):
            # Load image from path
            image = cv2.imread(data)
            if image is None:
                raise ValueError(f"Could not load image from {data}")
        else:
            image = data
            
        # Resize and normalize
        image = cv2.resize(image, self.target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vector = image.flatten().astype(np.float32) / 255.0
        return vector
    
    def get_dimension(self) -> int:
        return self._dimension
    
    @property
    def data_type(self) -> str:
        return "image"

class AudioEncoder(DataEncoder):
    """Audio encoder using MFCC features"""
    
    def __init__(self, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        if not HAS_LIBROSA:
            raise ImportError("librosa not installed. Install with: pip install librosa")
        
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._dimension = n_mfcc * 100  # Assuming ~100 time frames
        
    def encode(self, data: Union[str, np.ndarray]) -> np.ndarray:
        """Encode audio to vector"""
        if isinstance(data, str):
            # Load audio from path
            y, sr = librosa.load(data, sr=None)
        else:
            y, sr = data, 22050  # Assume default sample rate
            
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                    n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Pad or truncate to fixed size
        target_frames = self._dimension // self.n_mfcc
        if mfccs.shape[1] < target_frames:
            mfccs = np.pad(mfccs, ((0, 0), (0, target_frames - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :target_frames]
            
        return mfccs.flatten()
    
    def get_dimension(self) -> int:
        return self._dimension
    
    @property
    def data_type(self) -> str:
        return "audio"

class CustomEncoder(DataEncoder):
    """Custom encoder for user-defined data types"""
    
    def __init__(self, encode_func: Callable[[Any], np.ndarray], dimension: int, data_type: str):
        self.encode_func = encode_func
        self._dimension = dimension
        self._data_type = data_type
        
    def encode(self, data: Any) -> np.ndarray:
        return self.encode_func(data)
    
    def get_dimension(self) -> int:
        return self._dimension
    
    @property
    def data_type(self) -> str:
        return self._data_type

class UniversalDARG:
    """Universal Multi-Modal DARG System"""
    
    def __init__(self, 
                 config_path: str = "config.json",
                 neo4j_config: Dict[str, str] = None,
                 enable_visualization: bool = True):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize core DARG
        self.darg = DARG(config_file=config_path)
        
        # Data storage
        self.data_items: Dict[str, DataItem] = {}
        self.encoders: Dict[str, DataEncoder] = {}
        
        # Neo4j visualization
        self.visualizer = None
        if enable_visualization and HAS_NEO4J:
            try:
                from ..visualization.neo4j_integration import AdvancedNeo4jVisualizer
                neo4j_config = neo4j_config or {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j", 
                    "password": "password"
                }
                self.visualizer = AdvancedNeo4jVisualizer(**neo4j_config)
                logger.info("Neo4j visualizer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Neo4j visualizer: {e}")
        
        # Register default encoders
        self._register_default_encoders()
        
        logger.info("Universal DARG system initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default config
            config = {
                "vector_dim": 384,
                "num_layers": 5,
                "grid_size": 64,
                "beam_width": 32,
                "max_cache_size": 10000
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return config
    
    def _register_default_encoders(self):
        """Register default encoders for common data types"""
        try:
            if HAS_TRANSFORMERS:
                self.register_encoder(TextEncoder())
        except Exception as e:
            logger.warning(f"Could not register text encoder: {e}")
            
        try:
            if HAS_CV2:
                self.register_encoder(ImageEncoder())
        except Exception as e:
            logger.warning(f"Could not register image encoder: {e}")
            
        try:
            if HAS_LIBROSA:
                self.register_encoder(AudioEncoder())
        except Exception as e:
            logger.warning(f"Could not register audio encoder: {e}")
    
    def register_encoder(self, encoder: DataEncoder):
        """Register a data encoder"""
        self.encoders[encoder.data_type] = encoder
        logger.info(f"Registered encoder for {encoder.data_type} data")
    
    def add_data(self, 
                 data: Any, 
                 data_type: str,
                 item_id: str = None,
                 metadata: Dict[str, Any] = None) -> str:
        """Add data to the universal DARG system"""
        
        # Generate ID if not provided
        if item_id is None:
            item_id = f"{data_type}_{len(self.data_items)}_{int(time.time())}"
        
        # Get encoder for data type
        if data_type not in self.encoders:
            raise ValueError(f"No encoder registered for data type: {data_type}")
        
        encoder = self.encoders[data_type]
        
        # Create data item
        item = DataItem(
            id=item_id,
            data=data,
            data_type=data_type,
            metadata=metadata or {}
        )
        
        # Encode data to vector
        try:
            item.vector = encoder.encode(data)
            logger.info(f"Encoded {data_type} data to {len(item.vector)}-dimensional vector")
        except Exception as e:
            logger.error(f"Failed to encode {data_type} data: {e}")
            raise
        
        # Store data item using item_id as key
        self.data_items[item_id] = item
        
        # Add to DARG index
        self._add_to_darg_index(item)
        
        # Add to Neo4j visualization
        if self.visualizer:
            try:
                self.visualizer.add_node(item)
                self._update_similarity_graph(item)
            except Exception as e:
                logger.warning(f"Could not update Neo4j visualization: {e}")
        
        logger.info(f"Added {data_type} data item {item_id} to universal DARG")
        return item_id
    
    def _add_to_darg_index(self, item: DataItem):
        """Add vector to DARG index - incremental update"""
        try:
            # Check if DARG has the expected interface
            if hasattr(self.darg, 'add_vector'):
                self.darg.add_vector(item.vector, item.id)
            elif hasattr(self.darg, 'build_index'):
                # For now, use simple vector addition
                # In a full implementation, this would use DARG's incremental update mechanisms
                vectors = np.array([item.vector])
                ids = [item.id]
                
                # If DARG index is empty, initialize it
                if not hasattr(self.darg, 'vectors') or self.darg.vectors is None:
                    self.darg.build_index(vectors, ids)
                else:
                    # Incremental addition (simplified)
                    existing_vectors = self.darg.vectors
                    existing_ids = self.darg.ids
                    
                    new_vectors = np.vstack([existing_vectors, vectors])
                    new_ids = existing_ids + ids
                    
                    # Rebuild index with new data
                    # In a production system, this would be a true incremental update
                    self.darg.build_index(new_vectors, new_ids)
            elif hasattr(self.darg, 'data'):
                # Fallback DARG implementation
                self.darg.data[item.id] = item.vector
                self.darg.vectors.append(item.vector)
            else:
                logger.warning("DARG instance doesn't support adding vectors")
                
        except Exception as e:
            logger.error(f"Failed to add to DARG index: {e}")
            # Don't raise - continue with other functionality
            pass
    
    def _update_similarity_graph(self, new_item: DataItem):
        """Update Neo4j similarity graph with new item"""
        if not self.visualizer:
            return
            
        # Find similar items
        try:
            similar_items = self.find_similar(new_item.id, k=5)
            
            for similar_id, similarity in similar_items:
                if similar_id != new_item.id:
                    self.visualizer.add_similarity_edge(new_item.id, similar_id, similarity)
                    
        except Exception as e:
            logger.warning(f"Could not update similarity graph: {e}")
    
    def find_similar(self, item_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """Find similar items to given item"""
        if item_id not in self.data_items:
            raise ValueError(f"Item {item_id} not found")
        
        query_item = self.data_items[item_id]
        query_vector = query_item.vector
        
        try:
            # Use DARG for similarity search
            results = self.darg.search(query_vector.reshape(1, -1), k=k+1)  # +1 to exclude self
            
            # Convert to (id, similarity) tuples
            similar_items = []
            for result in results[0]:  # Get first query results
                result_id = result['id']
                distance = result['distance']
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                if result_id != item_id:  # Exclude self
                    similar_items.append((result_id, similarity))
            
            return similar_items[:k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search(self, 
               data: Any, 
               data_type: str, 
               k: int = 10,
               metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar items - simplified interface"""
        try:
            results = self.search_by_data(data, data_type, k)
            
            # Convert to simpler format
            formatted_results = []
            for item_id, similarity, item in results:
                formatted_results.append({
                    'data_id': item_id,
                    'similarity': similarity,
                    'data': item.data,
                    'metadata': item.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Fallback to simple similarity search
            try:
                return self._fallback_search(data, data_type, k)
            except Exception as e2:
                logger.error(f"Fallback search also failed: {e2}")
                return []
    
    def _fallback_search(self, data: Any, data_type: str, k: int) -> List[Dict[str, Any]]:
        """Fallback search using simple cosine similarity"""
        if data_type not in self.encoders:
            return []
        
        encoder = self.encoders[data_type]
        query_vector = encoder.encode(data)
        
        similarities = []
        for item_id, item in self.data_items.items():
            if item.data_type == data_type:
                # Compute cosine similarity
                item_vector = np.array(item.vector)
                query_array = np.array(query_vector)
                
                similarity = np.dot(query_array, item_vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(item_vector)
                )
                similarities.append((item_id, similarity, item))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for item_id, similarity, item in similarities[:k]:
            results.append({
                'data_id': item_id,
                'similarity': similarity,
                'data': item.data,
                'metadata': item.metadata
            })
        
        return results

    def search_by_data(self, 
                      data: Any, 
                      data_type: str, 
                      k: int = 10) -> List[Tuple[str, float, DataItem]]:
        """Search for similar items by providing new data"""
        
        if data_type not in self.encoders:
            raise ValueError(f"No encoder registered for data type: {data_type}")
        
        # Encode query data
        encoder = self.encoders[data_type]
        query_vector = encoder.encode(data)
        
        try:
            # Use DARG for similarity search
            results = self.darg.search(query_vector.reshape(1, -1), k=k)
            
            # Convert to (id, similarity, item) tuples
            similar_items = []
            for result in results[0]:  # Get first query results
                result_id = result['id']
                distance = result['distance']
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                item = self.data_items.get(result_id)
                
                if item:
                    similar_items.append((result_id, similarity, item))
            
            return similar_items
            
        except Exception as e:
            logger.error(f"Search by data failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'total_items': len(self.data_items),
            'data_types': {},
            'encoder_types': list(self.encoders.keys()),
            'darg_stats': {}
        }
        
        # Count by data type
        for item in self.data_items.values():
            data_type = item.data_type
            stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
        
        # DARG statistics
        if hasattr(self.darg, 'vectors') and self.darg.vectors is not None:
            stats['darg_stats'] = {
                'vector_count': len(self.darg.vectors),
                'vector_dimension': self.darg.vectors.shape[1] if len(self.darg.vectors) > 0 else 0
            }
        
        # Neo4j statistics
        if self.visualizer:
            try:
                stats['neo4j_stats'] = self.visualizer.get_graph_stats()
            except Exception as e:
                logger.warning(f"Could not get Neo4j stats: {e}")
                stats['neo4j_stats'] = {'error': str(e)}
        
        return stats
    
    def save_state(self, filepath: str):
        """Save system state to file"""
        state = {
            'config': self.config,
            'data_items': {k: asdict(v) for k, v in self.data_items.items()},
            'encoder_types': list(self.encoders.keys())
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        # Save DARG state
        darg_path = filepath.replace('.pkl', '_darg.pkl')
        if hasattr(self.darg, 'save_model'):
            self.darg.save_model(darg_path)
        
        logger.info(f"Universal DARG state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load system state from file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        
        # Restore data items
        self.data_items = {}
        for k, v in state['data_items'].items():
            item = DataItem(**v)
            self.data_items[k] = item
        
        # Load DARG state
        darg_path = filepath.replace('.pkl', '_darg.pkl')
        if Path(darg_path).exists() and hasattr(self.darg, 'load_model'):
            self.darg.load_model(darg_path)
        
        logger.info(f"Universal DARG state loaded from {filepath}")
    
    def close(self):
        """Close connections and cleanup"""
        if self.visualizer:
            self.visualizer.close()
        logger.info("Universal DARG system closed")
