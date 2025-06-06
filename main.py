import numpy as np
import json
import uuid
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
import warnings
import orjson
import joblib
import os
from functools import wraps
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Performance Timing Framework
# ============================================================================

class PerformanceTimer:
    """Efficient performance timing system - minimal overhead"""
    
    def __init__(self):
        self._operations = {}
        self._lock = threading.Lock()
    
    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation and return unique operation ID"""
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        start_time = time.perf_counter()
        return operation_id, start_time
    
    def end_operation(self, operation_name: str, start_time: float) -> float:
        """End timing an operation and return duration"""
        duration = time.perf_counter() - start_time
        
        with self._lock:
            if operation_name not in self._operations:
                self._operations[operation_name] = {
                    'total_time': 0.0,
                    'count': 0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'sum_squares': 0.0,
                    'durations': []  # Store individual durations for percentile calculation
                }
            
            stats = self._operations[operation_name]
            stats['total_time'] += duration
            stats['count'] += 1
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
            stats['sum_squares'] += duration * duration
            stats['durations'].append(duration)
            
            # Keep only last 1000 durations to prevent memory issues
            if len(stats['durations']) > 1000:
                stats['durations'] = stats['durations'][-1000:]
            
            return duration
    
    def _calculate_percentile(self, durations: list, percentile: float) -> float:
        """Calculate percentile from sorted durations"""
        if not durations:
            return 0.0
        
        sorted_durations = sorted(durations)
        index = (percentile / 100.0) * (len(sorted_durations) - 1)
        
        if index.is_integer():
            return sorted_durations[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            if upper_index >= len(sorted_durations):
                return sorted_durations[lower_index]
            
            # Linear interpolation
            weight = index - lower_index
            return sorted_durations[lower_index] * (1 - weight) + sorted_durations[upper_index] * weight
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get comprehensive timing statistics including percentiles"""
        with self._lock:
            result = {}
            for op_name, stats in self._operations.items():
                if stats['count'] > 0:
                    avg_time = stats['total_time'] / stats['count']
                    variance = (stats['sum_squares'] / stats['count']) - (avg_time * avg_time)
                    std_dev = (variance ** 0.5) if variance > 0 else 0.0
                    
                    # Calculate percentiles
                    p50_time = self._calculate_percentile(stats['durations'], 50)
                    p95_time = self._calculate_percentile(stats['durations'], 95)
                    p99_time = self._calculate_percentile(stats['durations'], 99)
                    
                    result[op_name] = {
                        'count': stats['count'],
                        'total_time': stats['total_time'],
                        'avg_time': avg_time,
                        'min_time': stats['min_time'],
                        'max_time': stats['max_time'],
                        'std_dev': std_dev,
                        'p50_time': p50_time,
                        'p95_time': p95_time,
                        'p99_time': p99_time
                    }
            return result
    
    def print_summary(self):
        """Print a formatted summary of all timing statistics"""
        stats = self.get_stats()
        if not stats:
            print("No timing data available")
            return
        
        print("\n" + "="*110)
        print("DARG v2.2 PERFORMANCE SUMMARY")
        print("="*110)
        
        # Sort by total time
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        print(f"{'Operation':<30} {'Count':<8} {'Total(s)':<10} {'Avg(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10}")
        print("-" * 110)
        
        for op_name, op_stats in sorted_ops:
            print(f"{op_name:<30} {op_stats['count']:<8} "
                  f"{op_stats['total_time']:<10.4f} "
                  f"{op_stats['avg_time']*1000:<10.3f} "
                  f"{op_stats['p50_time']*1000:<10.3f} "
                  f"{op_stats['p95_time']*1000:<10.3f} "
                  f"{op_stats['p99_time']*1000:<10.3f}")
        
        print("="*110)
    
    def reset_stats(self):
        """Reset all timing statistics"""
        with self._lock:
            self._operations.clear()

# Global performance timer instance
performance_timer = PerformanceTimer()

def time_operation(operation_name: str, include_args: bool = False):
    """Decorator to time function execution - NO individual logging for efficiency"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_id, start_time = performance_timer.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_timer.end_operation(operation_name, start_time)
        return wrapper
    return decorator

# ============================================================================
# Module 1: Global Configuration & Utilities
# ============================================================================

@dataclass
class GlobalConfig:
    """Global configuration for DARG v2.2"""
    # Grid parameters
    initial_grid_levels: int = 3
    base_max_pop_per_cell: int = 100
    min_pop_per_cell: int = 10
    max_grid_depth: int = 15
    
    # Search parameters
    beam_width_B: int = 5
    K_top_candidates: int = 50
    echo_trigger_threshold_S: float = 0.6
    echo_search_K_top_trigger: int = 10
    echo_search_N_echo: int = 3
    
    # PCA parameters
    pca_pop_trigger_for_split: int = 20
    proj_dimensions: int = 16
    pca_batch_size: int = 1000
    
    # LID parameters
    LID_influence_factor: float = 0.3
    LID_sample_size: int = 50
    Threshold_LID_Updates: int = 100
    
    # Linkage cache parameters
    max_linkage_cache_size: int = 20
    epsilon_exploration_linkage: float = 0.1
    min_resonance_hit_rate_linkage: float = 0.7
    linkage_activation_decay_factor: float = 0.95
    
    # Echo trigger weights
    echo_trigger_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.3, 0.2])
    
    # Maintenance parameters
    maintenance_interval_seconds: int = 300
    updates_threshold_for_pca: int = 50
    
    @classmethod
    @time_operation("config_load")
    def from_file(cls, filepath: str) -> 'GlobalConfig':
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            logger.warning(f"Config file {filepath} not found. Using defaults.")
            return cls()

class Utils:
    """Utility functions for DARG v2.2"""
    
    @staticmethod
    @time_operation("distance_calculation")
    def calculate_distance(vec1: np.ndarray, vec2: np.ndarray, metric: str = 'euclidean') -> float:
        """Calculate distance between two vectors"""
        if metric == 'euclidean':
            return np.linalg.norm(vec1 - vec2)
        elif metric == 'cosine':
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norms == 0:
                return 0.0
            return 1 - (dot_product / norms)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    @time_operation("vector_mean_calculation")
    def vector_mean(vectors: List[np.ndarray]) -> np.ndarray:
        """Calculate mean of vectors"""
        if not vectors:
            return np.array([])
        return np.mean(vectors, axis=0)
    
    @staticmethod
    @time_operation("batch_pca", include_args=True)
    def batch_pca(points: List[np.ndarray], n_components: int) -> Dict[str, Any]:
        """Perform batch PCA on points"""
        if len(points) < 2:
            return None
        
        points_array = np.array(points)
        if points_array.shape[1] <= n_components:
            n_components = min(n_components, points_array.shape[1] - 1)
            if n_components <= 0:
                return None
        
        pca = IncrementalPCA(n_components=n_components)
        pca.fit(points_array)
        
        return {
            'mean': pca.mean_,
            'components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'n_samples_seen': len(points)
        }
    
    @staticmethod
    @time_operation("incremental_pca_update", include_args=True)
    def incremental_pca_update(pca_model: Dict[str, Any], new_vector: np.ndarray) -> Dict[str, Any]:
        """Update PCA model incrementally"""
        if pca_model is None:
            return Utils.batch_pca([new_vector], len(new_vector))
        
        # Simple incremental update of mean
        n = pca_model['n_samples_seen']
        old_mean = pca_model['mean']
        new_mean = (old_mean * n + new_vector) / (n + 1)
        
        # For simplicity, we'll trigger batch PCA every N updates
        # In production, use proper incremental PCA algorithms
        pca_model['mean'] = new_mean
        pca_model['n_samples_seen'] = n + 1
        
        return pca_model
    
    @staticmethod
    @time_operation("project_vector")
    def project_vector(vector: np.ndarray, pca_model: Dict[str, Any]) -> np.ndarray:
        """Project vector using PCA model"""
        if pca_model is None:
            return vector
        
        centered = vector - pca_model['mean']
        return np.dot(centered, pca_model['components'].T)
    
    @staticmethod
    @time_operation("estimate_lid_two_nn", include_args=True)
    def estimate_lid_two_nn(points: List[np.ndarray]) -> float:
        """Estimate Local Intrinsic Dimension using Two-NN method"""
        if len(points) < 3:
            return 1.0
        
        points_array = np.array(points)
        nn = NearestNeighbors(n_neighbors=3)  # self + 2 neighbors
        nn.fit(points_array)
        
        mu_values = []
        for point in points_array:
            distances, _ = nn.kneighbors([point])
            distances = distances[0]
            
            if len(distances) >= 3:
                r1 = distances[1]  # nearest neighbor (excluding self)
                r2 = distances[2]  # second nearest neighbor
                
                if r1 > 0:
                    mu = r2 / r1
                    if mu > 1:  # mu should be >= 1
                        mu_values.append(np.log(mu))
        
        if not mu_values:
            return 1.0
        
        mean_log_mu = np.mean(mu_values)
        if mean_log_mu <= 0:
            return 1.0
        
        return 1.0 / mean_log_mu

@dataclass
class AABB:
    """Axis-Aligned Bounding Box"""
    min_coords: np.ndarray
    max_coords: np.ndarray
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside AABB"""
        return np.all(point >= self.min_coords) and np.all(point <= self.max_coords)
    
    def split(self, dimension: int, split_value: float) -> Tuple['AABB', 'AABB']:
        """Split AABB along dimension at split_value"""
        left_max = self.max_coords.copy()
        left_max[dimension] = split_value
        left_aabb = AABB(self.min_coords.copy(), left_max)
        
        right_min = self.min_coords.copy()
        right_min[dimension] = split_value
        right_aabb = AABB(right_min, self.max_coords.copy())
        
        return left_aabb, right_aabb
    
    def union(self, other: 'AABB') -> 'AABB':
        """Union of two AABBs"""
        min_coords = np.minimum(self.min_coords, other.min_coords)
        max_coords = np.maximum(self.max_coords, other.max_coords)
        return AABB(min_coords, max_coords)
    
    def centroid(self) -> np.ndarray:
        """Get centroid of AABB"""
        return (self.min_coords + self.max_coords) / 2

# ============================================================================
# Module 2: Point Database
# ============================================================================

class PointDB:
    """Point Database for storing original vectors and cell mappings"""
    
    def __init__(self):
        self.point_vectors_store: Dict[str, np.ndarray] = {}
        self.point_to_cell_map: Dict[str, str] = {}
        self._lock = threading.RLock()
    
    def __getstate__(self):
        """Custom serialization - exclude lock"""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        del state['_lock']
        return state
    
    def __setstate__(self, state):
        """Custom deserialization - recreate lock"""
        self.__dict__.update(state)
        # Recreate the lock
        self._lock = threading.RLock()
    
    @time_operation("store_point", include_args=True)
    def store_point(self, point_id: str, vector: np.ndarray, leaf_cell_id: str) -> bool:
        """Store point with its vector and cell assignment"""
        try:
            with self._lock:
                self.point_vectors_store[point_id] = vector.copy()
                self.point_to_cell_map[point_id] = leaf_cell_id
            return True
        except Exception as e:
            logger.error(f"Error storing point {point_id}: {e}")
            return False
    
    @time_operation("get_point_vector")
    def get_point_vector(self, point_id: str) -> Optional[np.ndarray]:
        """Get original vector for point"""
        with self._lock:
            return self.point_vectors_store.get(point_id)
    
    @time_operation("get_point_leaf_cell")
    def get_point_leaf_cell(self, point_id: str) -> Optional[str]:
        """Get leaf cell ID for point"""
        with self._lock:
            return self.point_to_cell_map.get(point_id)
    
    @time_operation("delete_point", include_args=True)
    def delete_point(self, point_id: str) -> bool:
        """Delete point from database"""
        try:
            with self._lock:
                self.point_vectors_store.pop(point_id, None)
                self.point_to_cell_map.pop(point_id, None)
            return True
        except Exception as e:
            logger.error(f"Error deleting point {point_id}: {e}")
            return False
    
    @time_operation("update_point_cell", include_args=True)
    def update_point_cell(self, point_id: str, new_cell_id: str) -> bool:
        """Update cell assignment for point"""
        try:
            with self._lock:
                if point_id in self.point_to_cell_map:
                    self.point_to_cell_map[point_id] = new_cell_id
                    return True
            return False
        except Exception as e:
            logger.error(f"Error updating point cell {point_id}: {e}")
            return False

# ============================================================================
# Module 3: Grid Manager
# ============================================================================

@dataclass
class LinkageEntry:
    """Entry in linkage cache"""
    target_cell_id: str
    target_cell_rep_proj: np.ndarray
    activation_score: float
    last_activation_timestamp: float

@dataclass
class QueryStats:
    """Query statistics for a cell"""
    total_queries_passed_through: int = 0
    queries_hopped_via_linkage: int = 0

@dataclass
class CellObject:
    """Cell object in the grid"""
    cell_id: str
    level: int
    boundary_box: AABB
    parent_cell_id: Optional[str] = None
    child_cell_ids: List[str] = field(default_factory=list)
    
    # Representatives
    representative_vector_orig: Optional[np.ndarray] = None
    local_pca_model: Optional[Dict[str, Any]] = None
    representative_vector_proj: Optional[np.ndarray] = None
    
    # Leaf cell data
    is_leaf: bool = True
    point_ids: List[str] = field(default_factory=list)
    point_count: int = 0
    local_LID_estimate: float = 1.0
    max_pop_local: int = 100
    updates_since_last_pca: int = 0
    updates_since_last_LID_calc: int = 0
    
    # Linkage cache
    linkage_cache: List[LinkageEntry] = field(default_factory=list)
    
    # Query statistics
    query_stats: QueryStats = field(default_factory=QueryStats)
    
    def __post_init__(self):
        self._lock = threading.RLock()
    
    def __getstate__(self):
        """Custom serialization - exclude lock"""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        if '_lock' in state:
            del state['_lock']
        return state
    
    def __setstate__(self, state):
        """Custom deserialization - recreate lock"""
        self.__dict__.update(state)
        # Recreate the lock
        self._lock = threading.RLock()


class GridManager:
    """Grid Manager for DARG v2.2"""
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.cells_store: Dict[str, CellObject] = {}
        self.root_cell_id: Optional[str] = None
        self.max_grid_depth_reached: int = 0
        self._lock = threading.RLock()
        self.global_pca_model: Optional[Dict[str, Any]] = None
    
    def __getstate__(self):
        """Custom serialization - exclude lock"""
        state = self.__dict__.copy()
        # Remove the unpicklable lock
        del state['_lock']
        return state
    
    def __setstate__(self, state):
        """Custom deserialization - recreate lock"""
        self.__dict__.update(state)
        # Recreate the lock
        self._lock = threading.RLock()

    @time_operation("initialize_grid", include_args=True)
    def initialize_grid(self, initial_data_sample: Optional[List[np.ndarray]] = None) -> str:
        """Initialize the grid structure"""
        with self._lock:
            # Create root cell
            if initial_data_sample and len(initial_data_sample) > 0:
                sample_array = np.array(initial_data_sample)
                min_coords = np.min(sample_array, axis=0)
                max_coords = np.max(sample_array, axis=0)
                
                # Add some padding
                padding = (max_coords - min_coords) * 0.1
                min_coords -= padding
                max_coords += padding
                
                # Initialize global PCA if we have enough data
                if len(initial_data_sample) >= self.config.pca_pop_trigger_for_split:
                    self.global_pca_model = Utils.batch_pca(
                        initial_data_sample[:self.config.pca_batch_size], 
                        self.config.proj_dimensions
                    )
            else:
                # Default large bounding box
                dimensions = 128  # Default dimensionality
                min_coords = np.full(dimensions, -1000.0)
                max_coords = np.full(dimensions, 1000.0)
            
            root_aabb = AABB(min_coords, max_coords)
            root_cell = CellObject(
                cell_id=str(uuid.uuid4()),
                level=0,
                boundary_box=root_aabb,
                max_pop_local=self.config.base_max_pop_per_cell
            )
            
            # Initialize root representative from sample data
            if initial_data_sample and len(initial_data_sample) > 0:
                root_cell.representative_vector_orig = Utils.vector_mean(initial_data_sample)
                root_cell.representative_vector_proj = self._project_vector_global(root_cell.representative_vector_orig)
            
            self.root_cell_id = root_cell.cell_id
            self.cells_store[root_cell.cell_id] = root_cell
            
            # DON'T create initial levels - let them be created dynamically as needed
            # This avoids creating empty cells that break routing
            
            return self.root_cell_id
    
    @time_operation("create_initial_levels", include_args=True)
    def _create_initial_levels(self, cell_id: str, levels_remaining: int):
        """Create initial grid levels"""
        if levels_remaining <= 0:
            return
        
        cell = self.get_cell(cell_id)
        if not cell or not cell.is_leaf:
            return
        
        # Split along longest dimension
        aabb = cell.boundary_box
        dimensions = aabb.max_coords - aabb.min_coords
        split_dim = np.argmax(dimensions)
        split_value = (aabb.min_coords[split_dim] + aabb.max_coords[split_dim]) / 2
        
        result = self.split_cell(cell_id, split_dim, split_value)
        if result:
            child1_id, child2_id = result
            self._create_initial_levels(child1_id, levels_remaining - 1)
            self._create_initial_levels(child2_id, levels_remaining - 1)
    
    @time_operation("get_cell")
    def get_cell(self, cell_id: str) -> Optional[CellObject]:
        """Get cell by ID"""
        with self._lock:
            return self.cells_store.get(cell_id)
    
    @time_operation("update_cell")
    def update_cell(self, cell: CellObject):
        """Update cell in store"""
        with self._lock:
            self.cells_store[cell.cell_id] = cell
    
    @time_operation("find_leaf_cell", include_args=True)
    def find_leaf_cell_for_vector(self, vector_orig: np.ndarray, vector_proj: Optional[np.ndarray] = None) -> Optional[CellObject]:
        """Find leaf cell for given vector"""
        if vector_proj is None:
            vector_proj = self._project_vector_global(vector_orig)
        
        current_cell_id = self.root_cell_id
        if not current_cell_id:
            return None
        
        while True:
            current_cell = self.get_cell(current_cell_id)
            if not current_cell:
                return None
            
            if current_cell.is_leaf:
                return current_cell
            
            if not current_cell.child_cell_ids:
                logger.error(f"Non-leaf cell has no children: {current_cell_id}")
                return current_cell
            
            # Find best child
            best_child_id = None
            min_dist = float('inf')
            
            for child_id in current_cell.child_cell_ids:
                child_cell = self.get_cell(child_id)
                if child_cell and child_cell.representative_vector_proj is not None:
                    dist = Utils.calculate_distance(vector_proj, child_cell.representative_vector_proj)
                    if dist < min_dist:
                        min_dist = dist
                        best_child_id = child_id
            
            if best_child_id is None:
                # Fallback: choose first child or treat current as leaf
                if current_cell.child_cell_ids:
                    best_child_id = current_cell.child_cell_ids[0]
                else:
                    return current_cell
            
            current_cell_id = best_child_id
    
    @time_operation("split_cell", include_args=True)
    def split_cell(self, cell_id: str, split_dim: Optional[int] = None, split_value: Optional[float] = None) -> Optional[Tuple[str, str]]:
        """Split a cell into two children"""
        with self._lock:
            cell = self.get_cell(cell_id)
            if not cell or not cell.is_leaf:
                return None
            
            # Determine split dimension and value
            if split_dim is None or split_value is None:
                aabb = cell.boundary_box
                dimensions = aabb.max_coords - aabb.min_coords
                split_dim = np.argmax(dimensions)
                split_value = (aabb.min_coords[split_dim] + aabb.max_coords[split_dim]) / 2
            
            # Create child AABBs
            child1_aabb, child2_aabb = cell.boundary_box.split(split_dim, split_value)
            
            # Create child cells
            child1 = CellObject(
                cell_id=str(uuid.uuid4()),
                level=cell.level + 1,
                boundary_box=child1_aabb,
                parent_cell_id=cell_id,
                max_pop_local=cell.max_pop_local
            )
            
            child2 = CellObject(
                cell_id=str(uuid.uuid4()),
                level=cell.level + 1,
                boundary_box=child2_aabb,
                parent_cell_id=cell_id,
                max_pop_local=cell.max_pop_local
            )
            
            # Reassign points from parent to children
            #from .point_db import PointDB  # Import here to avoid circular import
            
            point_db = getattr(self, '_point_db', None)
            
            for point_id in cell.point_ids:
                if point_db:
                    vector = point_db.get_point_vector(point_id)
                    if vector is not None:
                        if vector[split_dim] < split_value:
                            child1.point_ids.append(point_id)
                            if point_db:
                                point_db.update_point_cell(point_id, child1.cell_id)
                        else:
                            child2.point_ids.append(point_id)
                            if point_db:
                                point_db.update_point_cell(point_id, child2.cell_id)
            
            child1.point_count = len(child1.point_ids)
            child2.point_count = len(child2.point_ids)
            
            # Update parent cell
            cell.is_leaf = False
            cell.child_cell_ids = [child1.cell_id, child2.cell_id]
            cell.point_ids = []
            cell.point_count = child1.point_count + child2.point_count
            
            # Compute representatives for children
            self._compute_cell_representative(child1, point_db)
            self._compute_cell_representative(child2, point_db)
            
            # Initialize linkage caches
            self._initialize_linkage_cache(child1)
            self._initialize_linkage_cache(child2)
            
            # Store cells
            self.cells_store[child1.cell_id] = child1
            self.cells_store[child2.cell_id] = child2
            self.cells_store[cell_id] = cell
            
            # Update max depth
            self.max_grid_depth_reached = max(self.max_grid_depth_reached, child1.level)
            
            return child1.cell_id, child2.cell_id
    
    @time_operation("compute_cell_representative", include_args=True)
    def _compute_cell_representative(self, cell: CellObject, point_db):
        """Compute representative vector for cell"""
        if not cell.point_ids or not point_db:
            return
        
        vectors = []
        for point_id in cell.point_ids:
            vector = point_db.get_point_vector(point_id)
            if vector is not None:
                vectors.append(vector)
        
        if not vectors:
            return
        
        # Compute original representative
        cell.representative_vector_orig = Utils.vector_mean(vectors)
        
        # Compute PCA and projected representative if enough points
        if len(vectors) >= self.config.pca_pop_trigger_for_split:
            cell.local_pca_model = Utils.batch_pca(vectors, self.config.proj_dimensions)
            if cell.local_pca_model:
                cell.representative_vector_proj = Utils.project_vector(
                    cell.representative_vector_orig, cell.local_pca_model
                )
        else:
            # Use global PCA or original vector
            cell.representative_vector_proj = self._project_vector_global(cell.representative_vector_orig)
        
        # Estimate LID
        if len(vectors) >= 3:
            cell.local_LID_estimate = Utils.estimate_lid_two_nn(vectors[:self.config.LID_sample_size])
            # Adjust max population based on LID
            base_pop = self.config.base_max_pop_per_cell
            cell.max_pop_local = int(base_pop * (1 + self.config.LID_influence_factor * cell.local_LID_estimate))
    
    @time_operation("project_vector_global")
    def _project_vector_global(self, vector: np.ndarray) -> np.ndarray:
        """Project vector using global PCA model"""
        if self.global_pca_model:
            return Utils.project_vector(vector, self.global_pca_model)
        return vector
    
    @time_operation("initialize_linkage_cache")
    def _initialize_linkage_cache(self, cell: CellObject):
        """Initialize linkage cache for cell"""
        # Add parent link
        if cell.parent_cell_id:
            parent = self.get_cell(cell.parent_cell_id)
            if parent and parent.representative_vector_proj is not None:
                entry = LinkageEntry(
                    target_cell_id=cell.parent_cell_id,
                    target_cell_rep_proj=parent.representative_vector_proj.copy(),
                    activation_score=1.0,
                    last_activation_timestamp=time.time()
                )
                cell.linkage_cache.append(entry)
        
        # Add sibling links
        if cell.parent_cell_id:
            parent = self.get_cell(cell.parent_cell_id)
            if parent:
                for sibling_id in parent.child_cell_ids:
                    if sibling_id != cell.cell_id:
                        sibling = self.get_cell(sibling_id)
                        if sibling and sibling.representative_vector_proj is not None:
                            entry = LinkageEntry(
                                target_cell_id=sibling_id,
                                target_cell_rep_proj=sibling.representative_vector_proj.copy(),
                                activation_score=0.8,
                                last_activation_timestamp=time.time()
                            )
                            cell.linkage_cache.append(entry)
    
    def update_linkage_cache(self, cell_id: str, successful_hop_to_target_id: Optional[str] = None):
        """Update linkage cache with successful hop information"""
        cell = self.get_cell(cell_id)
        if not cell:
            return
        
        current_time = time.time()
        
        # Update activation score for successful hop
        if successful_hop_to_target_id:
            for entry in cell.linkage_cache:
                if entry.target_cell_id == successful_hop_to_target_id:
                    entry.activation_score = min(1.0, entry.activation_score + 0.1)
                    entry.last_activation_timestamp = current_time
                    break
        
        # Apply decay to all entries
        for entry in cell.linkage_cache:
            time_decay = max(0.1, 1.0 - (current_time - entry.last_activation_timestamp) / 3600)  # Decay over 1 hour
            entry.activation_score *= self.config.linkage_activation_decay_factor * time_decay
        
        # Prune low-score entries if cache is full
        if len(cell.linkage_cache) > self.config.max_linkage_cache_size:
            cell.linkage_cache.sort(key=lambda x: x.activation_score, reverse=True)
            cell.linkage_cache = cell.linkage_cache[:self.config.max_linkage_cache_size]
        
        self.update_cell(cell)

# ============================================================================
# Module 4: Update Manager
# ============================================================================

class UpdateManager:
    """Update Manager for handling insertions and deletions"""
    
    def __init__(self, grid_manager: GridManager, point_db: PointDB, config: GlobalConfig):
        self.grid_manager = grid_manager
        self.point_db = point_db
        self.config = config
        self._lock = threading.RLock()
        
        # Set point_db reference in grid_manager for splits
        self.grid_manager._point_db = point_db
    
    @time_operation("insert_point", include_args=True)
    def insert_point(self, point_id: str, vector_orig: np.ndarray) -> bool:
        """Insert a point into the index"""
        try:
            with self._lock:
                # Project vector
                vector_proj = self.grid_manager._project_vector_global(vector_orig)
                
                # Find target leaf cell
                target_leaf_cell = self.grid_manager.find_leaf_cell_for_vector(vector_orig, vector_proj)
                if not target_leaf_cell:
                    logger.error("Could not find target leaf cell")
                    return False
                
                # Store point in database
                if not self.point_db.store_point(point_id, vector_orig, target_leaf_cell.cell_id):
                    return False
                
                # Add point to cell
                target_leaf_cell.point_ids.append(point_id)
                target_leaf_cell.point_count += 1
                target_leaf_cell.updates_since_last_pca += 1
                
                # Update cell representative
                self._update_cell_representative(target_leaf_cell, vector_orig, is_addition=True)
                
                # Check if split is needed
                if target_leaf_cell.point_count > target_leaf_cell.max_pop_local:
                    self.grid_manager.split_cell(target_leaf_cell.cell_id)
                else:
                    self.grid_manager.update_cell(target_leaf_cell)
                
                # Bubble up point counts
                self._bubble_up_counts(target_leaf_cell.parent_cell_id, 1)
                
                return True
                
        except Exception as e:
            logger.error(f"Error inserting point {point_id}: {e}")
            return False
    
    @time_operation("delete_point_update", include_args=True)
    def delete_point(self, point_id: str) -> bool:
        """Delete a point from the index"""
        try:
            with self._lock:
                # Get point info
                leaf_cell_id = self.point_db.get_point_leaf_cell(point_id)
                if not leaf_cell_id:
                    return False
                
                vector_orig = self.point_db.get_point_vector(point_id)
                if vector_orig is None:
                    return False
                
                leaf_cell = self.grid_manager.get_cell(leaf_cell_id)
                if not leaf_cell:
                    return False
                
                # Remove point from cell
                if point_id in leaf_cell.point_ids:
                    leaf_cell.point_ids.remove(point_id)
                    leaf_cell.point_count -= 1
                    leaf_cell.updates_since_last_pca += 1
                
                # Delete from database
                self.point_db.delete_point(point_id)
                
                # Update cell representative
                self._update_cell_representative(leaf_cell, vector_orig, is_addition=False)
                
                # Check if merge is needed
                if leaf_cell.point_count < self.config.min_pop_per_cell:
                    # Simple merge logic - could be expanded
                    pass
                
                self.grid_manager.update_cell(leaf_cell)
                
                # Bubble up count changes
                self._bubble_up_counts(leaf_cell.parent_cell_id, -1)
                
                return True
                
        except Exception as e:
            logger.error(f"Error deleting point {point_id}: {e}")
            return False
    
    @time_operation("update_cell_representative", include_args=True)
    def _update_cell_representative(self, cell: CellObject, vector: np.ndarray, is_addition: bool):
        """Update cell representative incrementally"""
        if cell.representative_vector_orig is None:
            if is_addition:
                cell.representative_vector_orig = vector.copy()
                cell.representative_vector_proj = self.grid_manager._project_vector_global(vector)
            return
        
        # Simple incremental update of mean
        if is_addition:
            n = cell.point_count
            cell.representative_vector_orig = (cell.representative_vector_orig * (n - 1) + vector) / n
        else:
            n = cell.point_count + 1  # +1 because we already decremented point_count
            if n > 1:
                cell.representative_vector_orig = (cell.representative_vector_orig * n - vector) / (n - 1)
        
        # Update projected representative
        cell.representative_vector_proj = self.grid_manager._project_vector_global(cell.representative_vector_orig)
        
        # Update PCA model if needed
        if cell.local_pca_model and is_addition:
            cell.local_pca_model = Utils.incremental_pca_update(cell.local_pca_model, vector)
    
    @time_operation("bubble_up_counts", include_args=True)
    def _bubble_up_counts(self, parent_cell_id: Optional[str], count_delta: int):
        """Bubble up point count changes to parent cells"""
        while parent_cell_id:
            parent_cell = self.grid_manager.get_cell(parent_cell_id)
            if parent_cell:
                parent_cell.point_count += count_delta
                self.grid_manager.update_cell(parent_cell)
                parent_cell_id = parent_cell.parent_cell_id
            else:
                break

# ============================================================================
# Module 5: Search Orchestrator
# ============================================================================

@dataclass
class SearchResult:
    """Search result entry"""
    distance: float
    point_id: str

class SearchOrchestrator:
    """Search Orchestrator for k-NN queries"""
    
    def __init__(self, grid_manager: GridManager, point_db: PointDB, config: GlobalConfig):
        self.grid_manager = grid_manager
        self.point_db = point_db
        self.config = config
    
    @time_operation("search_k_nearest", include_args=True)
    def search_k_nearest(self, query_vector_orig: np.ndarray, k: int) -> List[SearchResult]:
        """Search for k nearest neighbors"""
        try:
            # Phase 1: Grid Resonance (Beam Search)
            query_vector_proj = self.grid_manager._project_vector_global(query_vector_orig)
            candidate_leaf_cells = self._phase1_grid_resonance(query_vector_proj)
            
            # Phase 2: Localized Refinement
            potential_nns = self._phase2_localized_refinement(query_vector_orig, candidate_leaf_cells)
            
            # Phase 3: Conditional Echo Search
            s_trigger = self._calculate_s_trigger(potential_nns, query_vector_orig, candidate_leaf_cells)
            if s_trigger > self.config.echo_trigger_threshold_S or np.random.random() < 0.1:  # 10% safety net
                echo_candidates = self._phase3_echo_search(query_vector_orig, potential_nns[:self.config.echo_search_K_top_trigger])
                potential_nns.extend(echo_candidates)
            
            # Phase 4: Final Ranking
            potential_nns.sort(key=lambda x: x.distance)
            return potential_nns[:k]
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    @time_operation("phase1_grid_resonance", include_args=True)
    def _phase1_grid_resonance(self, query_vector_proj: np.ndarray) -> List[str]:
        """Phase 1: Grid resonance with beam search"""
        if not self.grid_manager.root_cell_id:
            return []
        
        current_beam = [self.grid_manager.root_cell_id]
        candidate_leaf_cells = []
        
        for level in range(self.grid_manager.max_grid_depth_reached + 1):
            if not current_beam:
                break
            
            next_level_candidates = []
            
            for cell_id in current_beam:
                cell = self.grid_manager.get_cell(cell_id)
                if not cell:
                    continue
                
                # Update query stats
                cell.query_stats.total_queries_passed_through += 1
                
                if cell.is_leaf:
                    candidate_leaf_cells.append(cell_id)
                    continue
                
                # Add children
                next_level_candidates.extend(cell.child_cell_ids)
                
                # Explore linkage cache
                linkage_candidates = self._explore_linkage_cache(cell, query_vector_proj)
                next_level_candidates.extend(linkage_candidates)
            
            if not next_level_candidates:
                break
            
            # Rank candidates and select top beam_width
            ranked_candidates = []
            for candidate_id in next_level_candidates:
                candidate_cell = self.grid_manager.get_cell(candidate_id)
                if candidate_cell and candidate_cell.representative_vector_proj is not None:
                    dist = Utils.calculate_distance(query_vector_proj, candidate_cell.representative_vector_proj)
                    ranked_candidates.append((dist, candidate_id))
            
            ranked_candidates.sort(key=lambda x: x[0])
            current_beam = [cid for _, cid in ranked_candidates[:self.config.beam_width_B]]
        
        # Add remaining beam cells if they are leaves
        for cell_id in current_beam:
            cell = self.grid_manager.get_cell(cell_id)
            if cell and cell.is_leaf:
                candidate_leaf_cells.append(cell_id)
        
        return list(set(candidate_leaf_cells))  # Remove duplicates
    
    @time_operation("explore_linkage_cache", include_args=True)
    def _explore_linkage_cache(self, cell: CellObject, query_vector_proj: np.ndarray) -> List[str]:
        """Explore linkage cache for additional candidates"""
        if not cell.linkage_cache:
            return []
        
        # Calculate effective epsilon
        hit_rate = (cell.query_stats.queries_hopped_via_linkage / 
                   max(1, cell.query_stats.total_queries_passed_through))
        effective_epsilon = self.config.epsilon_exploration_linkage
        if hit_rate < self.config.min_resonance_hit_rate_linkage:
            effective_epsilon = min(1.0, effective_epsilon * 2)
        
        # Select links
        m_links = min(3, len(cell.linkage_cache))  # Select up to 3 links
        
        if np.random.random() < effective_epsilon:
            # Exploration: random or least activated
            selected_links = np.random.choice(len(cell.linkage_cache), 
                                            size=min(m_links, len(cell.linkage_cache)), 
                                            replace=False)
        else:
            # Exploitation: best scoring links close to query
            scored_links = []
            for i, link in enumerate(cell.linkage_cache):
                proximity_score = 1.0 / (1.0 + Utils.calculate_distance(query_vector_proj, link.target_cell_rep_proj))
                combined_score = link.activation_score * proximity_score
                scored_links.append((combined_score, i))
            
            scored_links.sort(key=lambda x: x[0], reverse=True)
            selected_links = [i for _, i in scored_links[:m_links]]
        
        # Update linkage cache with successful hops
        selected_cell_ids = []
        for link_idx in selected_links:
            if link_idx < len(cell.linkage_cache):
                target_id = cell.linkage_cache[link_idx].target_cell_id
                selected_cell_ids.append(target_id)
                self.grid_manager.update_linkage_cache(cell.cell_id, target_id)
                cell.query_stats.queries_hopped_via_linkage += 1
        
        return selected_cell_ids
    
    @time_operation("phase2_localized_refinement", include_args=True)
    def _phase2_localized_refinement(self, query_vector_orig: np.ndarray, candidate_leaf_cells: List[str]) -> List[SearchResult]:
        """Phase 2: Localized refinement in candidate cells"""
        potential_nns = []
        
        for leaf_cell_id in candidate_leaf_cells:
            leaf_cell = self.grid_manager.get_cell(leaf_cell_id)
            if not leaf_cell or not leaf_cell.point_ids:
                continue
            
            for point_id in leaf_cell.point_ids:
                vector = self.point_db.get_point_vector(point_id)
                if vector is not None:
                    distance = Utils.calculate_distance(query_vector_orig, vector)
                    potential_nns.append(SearchResult(distance=distance, point_id=point_id))
        
        # Sort and keep top N_intermediate
        potential_nns.sort(key=lambda x: x.distance)
        return potential_nns[:self.config.K_top_candidates]
    
    @time_operation("calculate_s_trigger")
    def _calculate_s_trigger(self, potential_nns: List[SearchResult], 
                           query_vector_orig: np.ndarray, 
                           candidate_cells: List[str]) -> float:
        """Calculate trigger score for echo search"""
        if len(potential_nns) < 2:
            return 1.0  # Trigger if too few candidates
        
        # Distance-based metrics
        distances = [result.distance for result in potential_nns[:10]]  # Top 10
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        spread = std_dist / (mean_dist + 1e-8)
        
        # Boundary proximity (simplified)
        boundary_prox = 0.5  # Placeholder
        
        # Beam diversity (simplified)
        beam_diversity = min(1.0, len(candidate_cells) / self.config.beam_width_B)
        
        # Combine metrics
        weights = self.config.echo_trigger_weights
        s_trigger = (weights[0] * min(1.0, mean_dist) +
                    weights[1] * min(1.0, spread) +
                    weights[2] * boundary_prox +
                    weights[3] * beam_diversity)
        
        return s_trigger
    
    @time_operation("phase3_echo_search", include_args=True)
    def _phase3_echo_search(self, query_vector_orig: np.ndarray, top_candidates: List[SearchResult]) -> List[SearchResult]:
        """Phase 3: Echo search in linked cells"""
        echo_results = []
        visited_cells = set()
        
        for result in top_candidates[:self.config.echo_search_K_top_trigger]:
            # Get origin cell of this point
            origin_cell_id = self.point_db.get_point_leaf_cell(result.point_id)
            if not origin_cell_id:
                continue
            
            origin_cell = self.grid_manager.get_cell(origin_cell_id)
            if not origin_cell or not origin_cell.linkage_cache:
                continue
            
            # Select echo cells from linkage cache
            echo_count = 0
            for link in origin_cell.linkage_cache:
                if echo_count >= self.config.echo_search_N_echo:
                    break
                
                if link.target_cell_id not in visited_cells:
                    visited_cells.add(link.target_cell_id)
                    target_cell = self.grid_manager.get_cell(link.target_cell_id)
                    
                    if target_cell and target_cell.is_leaf:
                        # Search in this echo cell
                        for point_id in target_cell.point_ids:
                            vector = self.point_db.get_point_vector(point_id)
                            if vector is not None:
                                distance = Utils.calculate_distance(query_vector_orig, vector)
                                echo_results.append(SearchResult(distance=distance, point_id=point_id))
                    
                    echo_count += 1
        
        return echo_results

# ============================================================================
# Module 6: Maintenance Scheduler
# ============================================================================

class MaintenanceScheduler:
    """Maintenance Scheduler for periodic optimization tasks"""
    
    def __init__(self, grid_manager: GridManager, config: GlobalConfig):
        self.grid_manager = grid_manager
        self.config = config
        self.running = False
        self.thread = None
    
    def start(self):
        """Start maintenance scheduler"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._maintenance_loop, daemon=True)
            self.thread.start()
            logger.info("Maintenance scheduler started")
    
    def stop(self):
        """Stop maintenance scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)  # 5 second timeout
        logger.info("Maintenance scheduler stopped")
    
    def _maintenance_loop(self):
        """Main maintenance loop"""
        while self.running:
            try:
                self._run_maintenance_tasks()
                # Use interruptible sleep
                for _ in range(self.config.maintenance_interval_seconds):
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                # Use interruptible sleep for error recovery too
                for _ in range(60):
                    if not self.running:
                        break
                    time.sleep(1)
    
    @time_operation("run_maintenance_tasks")
    def _run_maintenance_tasks(self):
        """Run all maintenance tasks"""
        logger.info("Running maintenance tasks")
        
        # Task 1: Linkage cache adaptation
        self._adapt_linkage_caches()
        
        # Task 2: LID re-estimation
        self._reestimate_lid_values()
        
        # Task 3: PCA model refresh
        self._refresh_stale_pca_models()
        
        logger.info("Maintenance tasks completed")
    
    @time_operation("adapt_linkage_caches")
    def _adapt_linkage_caches(self):
        """Adapt linkage caches across all cells"""
        current_time = time.time()
        
        for cell_id, cell in list(self.grid_manager.cells_store.items()):
            if not cell.linkage_cache:
                continue
            
            # Apply decay and prune
            for entry in cell.linkage_cache:
                entry.activation_score *= self.config.linkage_activation_decay_factor
            
            # Remove very low scoring entries
            cell.linkage_cache = [entry for entry in cell.linkage_cache 
                                if entry.activation_score > 0.1]
            
            # Add new geometric links if cache has space
            if len(cell.linkage_cache) < self.config.max_linkage_cache_size:
                self._add_geometric_links(cell)
            
            self.grid_manager.update_cell(cell)
    
    @time_operation("add_geometric_links", include_args=True)
    def _add_geometric_links(self, cell: CellObject):
        """Add new geometric links to cell's linkage cache"""
        if not cell.representative_vector_proj is not None:
            return
        
        # Find nearby cells at same level
        current_level_cells = [c for c in self.grid_manager.cells_store.values() 
                             if c.level == cell.level and c.cell_id != cell.cell_id 
                             and c.representative_vector_proj is not None]
        
        if not current_level_cells:
            return
        
        # Calculate distances and add closest cells
        candidates = []
        for other_cell in current_level_cells:
            distance = Utils.calculate_distance(cell.representative_vector_proj, 
                                              other_cell.representative_vector_proj)
            candidates.append((distance, other_cell))
        
        candidates.sort(key=lambda x: x[0])
        
        # Add top candidates not already in cache
        existing_targets = {entry.target_cell_id for entry in cell.linkage_cache}
        added = 0
        max_to_add = self.config.max_linkage_cache_size - len(cell.linkage_cache)
        
        for distance, other_cell in candidates:
            if added >= max_to_add:
                break
            
            if other_cell.cell_id not in existing_targets:
                entry = LinkageEntry(
                    target_cell_id=other_cell.cell_id,
                    target_cell_rep_proj=other_cell.representative_vector_proj.copy(),
                    activation_score=0.5,  # Initial moderate score
                    last_activation_timestamp=time.time()
                )
                cell.linkage_cache.append(entry)
                added += 1
    
    @time_operation("reestimate_lid_values")
    def _reestimate_lid_values(self):
        """Re-estimate LID values for cells with significant updates"""
        point_db = getattr(self.grid_manager, '_point_db', None)
        if not point_db:
            return
        
        for cell_id, cell in list(self.grid_manager.cells_store.items()):
            if (not cell.is_leaf or 
                cell.updates_since_last_pca < self.config.updates_threshold_for_pca or
                len(cell.point_ids) < 10):
                continue
            
            # Sample points for LID estimation
            sample_size = min(self.config.LID_sample_size, len(cell.point_ids))
            sampled_point_ids = np.random.choice(cell.point_ids, sample_size, replace=False)
            
            vectors = []
            for point_id in sampled_point_ids:
                vector = point_db.get_point_vector(point_id)
                if vector is not None:
                    vectors.append(vector)
            
            if len(vectors) >= 3:
                new_lid = Utils.estimate_lid_two_nn(vectors)
                cell.local_LID_estimate = new_lid
                
                # Adjust max population
                base_pop = self.config.base_max_pop_per_cell
                cell.max_pop_local = int(base_pop * (1 + self.config.LID_influence_factor * new_lid))
                cell.max_pop_local = max(self.config.min_pop_per_cell, cell.max_pop_local)
                
                # Reset update counter for LID-related updates
                cell.updates_since_last_pca = max(0, cell.updates_since_last_pca - self.config.updates_threshold_for_pca)
                
                self.grid_manager.update_cell(cell)
    
    @time_operation("refresh_stale_pca_models")
    def _refresh_stale_pca_models(self):
        """Refresh stale PCA models"""
        point_db = getattr(self.grid_manager, '_point_db', None)
        if not point_db:
            return
        
        for cell_id, cell in list(self.grid_manager.cells_store.items()):
            if (not cell.is_leaf or 
                not cell.local_pca_model or
                cell.updates_since_last_pca < self.config.updates_threshold_for_pca * 2):
                continue
            
            # Recompute PCA on current points
            vectors = []
            for point_id in cell.point_ids:
                vector = point_db.get_point_vector(point_id)
                if vector is not None:
                    vectors.append(vector)
            
            if len(vectors) >= self.config.pca_pop_trigger_for_split:
                new_pca_model = Utils.batch_pca(vectors, self.config.proj_dimensions)
                if new_pca_model:
                    cell.local_pca_model = new_pca_model
                    # Update representative projection
                    if cell.representative_vector_orig is not None:
                        cell.representative_vector_proj = Utils.project_vector(
                            cell.representative_vector_orig, cell.local_pca_model
                        )
                    
                    cell.updates_since_last_pca = 0
                    self.grid_manager.update_cell(cell)

# ============================================================================
# Main DARG v2.2 System
# ============================================================================

class DARGv22:
    """Main DARG v2.2 System"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            self.config = GlobalConfig.from_file(config_path)
        else:
            self.config = GlobalConfig()
        
        # Initialize components
        self.point_db = PointDB()
        self.grid_manager = GridManager(self.config)
        self.update_manager = UpdateManager(self.grid_manager, self.point_db, self.config)
        self.search_orchestrator = SearchOrchestrator(self.grid_manager, self.point_db, self.config)
        self.maintenance_scheduler = MaintenanceScheduler(self.grid_manager, self.config)
        
        # System state
        self.initialized = False
        logger.info("DARG v2.2 system created")
    
    @time_operation("darg_initialize", include_args=True)
    def initialize(self, initial_data: Optional[List[np.ndarray]] = None):
        """Initialize the DARG system"""
        if not self.initialized:
            self.grid_manager.initialize_grid(initial_data)
            self.maintenance_scheduler.start()
            self.initialized = True
            logger.info("DARG v2.2 system initialized")
        else:
            logger.warning("System already initialized")
    
    @time_operation("darg_insert", include_args=True)
    def insert(self, point_id: str, vector: np.ndarray) -> bool:
        """Insert a point into the index"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        return self.update_manager.insert_point(point_id, vector)
    
    @time_operation("darg_delete", include_args=True)
    def delete(self, point_id: str) -> bool:
        """Delete a point from the index"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        return self.update_manager.delete_point(point_id)
    
    @time_operation("darg_search", include_args=True)
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        results = self.search_orchestrator.search_k_nearest(query_vector, k)
        return [(result.point_id, result.distance) for result in results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.initialized:
            return {"error": "System not initialized"}
        
        total_cells = len(self.grid_manager.cells_store)
        leaf_cells = sum(1 for cell in self.grid_manager.cells_store.values() if cell.is_leaf)
        total_points = sum(len(cell.point_ids) for cell in self.grid_manager.cells_store.values() if cell.is_leaf)
        
        return {
            "total_cells": total_cells,
            "leaf_cells": leaf_cells,
            "total_points": total_points,
            "max_depth": self.grid_manager.max_grid_depth_reached,
            "root_cell_id": self.grid_manager.root_cell_id
        }
    
    def shutdown(self):
        """Shutdown the system"""
        self.maintenance_scheduler.stop()
        logger.info("DARG v2.2 system shutdown")
    
    @time_operation("darg_save_index", include_args=True)
    def save_index(self, filepath: str):
        """Save index to file using high-performance serialization"""
        # Stop maintenance scheduler temporarily
        was_running = self.maintenance_scheduler.running
        if was_running:
            self.maintenance_scheduler.stop()
        
        try:
            logger.info("Starting high-performance index serialization...")
            base_path = filepath.rsplit('.', 1)[0]  # Remove extension
            
            # Separate numpy arrays from serializable data
            numpy_data = {}
            serializable_data = {
                'config': self.config.__dict__,
                'root_cell_id': self.grid_manager.root_cell_id,
                'max_grid_depth_reached': self.grid_manager.max_grid_depth_reached
            }
            
            # Process cells data
            cells_data = {}
            for cell_id, cell in list(self.grid_manager.cells_store.items()):
                cell_state = cell.__getstate__() if hasattr(cell, '__getstate__') else cell.__dict__.copy()
                if '_lock' in cell_state:
                    del cell_state['_lock']
                
                # Extract numpy arrays
                cell_numpy = {}
                cell_serializable = {}
                
                for key, value in cell_state.items():
                    if isinstance(value, np.ndarray):
                        array_key = f"cell_{cell_id}_{key}"
                        numpy_data[array_key] = value
                        cell_serializable[key] = f"__NUMPY_REF__{array_key}"
                    elif key == 'boundary_box':
                        # Handle AABB specially
                        if hasattr(value, 'min_coords') and hasattr(value, 'max_coords'):
                            min_key = f"cell_{cell_id}_boundary_min"
                            max_key = f"cell_{cell_id}_boundary_max"
                            numpy_data[min_key] = value.min_coords
                            numpy_data[max_key] = value.max_coords
                            cell_serializable[key] = {
                                'min_coords': f"__NUMPY_REF__{min_key}",
                                'max_coords': f"__NUMPY_REF__{max_key}"
                            }
                        else:
                            cell_serializable[key] = value
                    elif key == 'local_pca_model' and value is not None:
                        # Handle PCA model
                        pca_data = {}
                        pca_numpy = {}
                        for pca_key, pca_value in value.items():
                            if isinstance(pca_value, np.ndarray):
                                pca_array_key = f"cell_{cell_id}_pca_{pca_key}"
                                numpy_data[pca_array_key] = pca_value
                                pca_data[pca_key] = f"__NUMPY_REF__{pca_array_key}"
                            else:
                                pca_data[pca_key] = pca_value
                        cell_serializable[key] = pca_data
                    elif key == 'linkage_cache' and value is not None:
                        # Handle LinkageEntry objects
                        linkage_data = []
                        for entry in value:
                            if hasattr(entry, 'target_cell_rep_proj') and isinstance(entry.target_cell_rep_proj, np.ndarray):
                                array_key = f"cell_{cell_id}_linkage_{len(linkage_data)}"
                                numpy_data[array_key] = entry.target_cell_rep_proj
                                entry_data = {
                                    'target_cell_id': entry.target_cell_id,
                                    'target_cell_rep_proj': f"__NUMPY_REF__{array_key}",
                                    'activation_score': entry.activation_score,
                                    'last_activation_timestamp': entry.last_activation_timestamp
                                }
                            else:
                                # Fallback for cases where entry might be a dict already
                                entry_data = entry.__dict__ if hasattr(entry, '__dict__') else entry
                            linkage_data.append(entry_data)
                        cell_serializable[key] = linkage_data
                    elif key == 'query_stats' and value is not None:
                        # Handle QueryStats dataclass
                        if hasattr(value, '__dict__'):
                            cell_serializable[key] = value.__dict__
                        else:
                            cell_serializable[key] = value
                    else:
                        cell_serializable[key] = value
                
                cells_data[cell_id] = cell_serializable
            
            serializable_data['cells_store'] = cells_data
            
            # Process global PCA model
            if self.grid_manager.global_pca_model:
                global_pca_data = {}
                for key, value in self.grid_manager.global_pca_model.items():
                    if isinstance(value, np.ndarray):
                        array_key = f"global_pca_{key}"
                        numpy_data[array_key] = value
                        global_pca_data[key] = f"__NUMPY_REF__{array_key}"
                    else:
                        global_pca_data[key] = value
                serializable_data['global_pca_model'] = global_pca_data
            else:
                serializable_data['global_pca_model'] = None
            
            # Process point database
            point_vectors = {}
            for point_id, vector in self.point_db.point_vectors_store.items():
                array_key = f"point_{point_id}"
                numpy_data[array_key] = vector
                point_vectors[point_id] = f"__NUMPY_REF__{array_key}"
            
            serializable_data['point_db_state'] = {
                'point_vectors_store': point_vectors,
                'point_to_cell_map': self.point_db.point_to_cell_map
            }
            
            # Save numpy arrays using joblib (very fast)
            numpy_path = f"{base_path}_arrays.joblib"
            joblib.dump(numpy_data, numpy_path, compress=3)
            logger.info(f"Saved {len(numpy_data)} numpy arrays to {numpy_path}")
            
            # Save serializable data using orjson (ultra-fast JSON)
            json_path = f"{base_path}_data.json"
            with open(json_path, 'wb') as f:
                f.write(orjson.dumps(serializable_data, option=orjson.OPT_SERIALIZE_NUMPY))
            logger.info(f"Saved serializable data to {json_path}")
            
            # Create manifest file
            manifest = {
                'version': '2.2',
                'format': 'split_fast',
                'arrays_file': os.path.basename(numpy_path),
                'data_file': os.path.basename(json_path),
                'array_count': len(numpy_data),
                'timestamp': time.time()
            }
            
            manifest_path = f"{base_path}.manifest"
            with open(manifest_path, 'wb') as f:
                f.write(orjson.dumps(manifest))
            
            logger.info(f"Index saved successfully to {base_path}.*")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
        finally:
            # Restart maintenance scheduler if it was running
            if was_running:
                self.maintenance_scheduler.start()

    @time_operation("darg_load_index", include_args=True)
    def load_index(self, filepath: str):
        """Load index from file supporting both old pickle and new fast format"""
        # Check if it's new format by looking for manifest
        base_path = filepath.rsplit('.', 1)[0]
        manifest_path = f"{base_path}.manifest"
        
        if os.path.exists(manifest_path):
            # New fast format
            self._load_fast_format(base_path)
        else:
            # Legacy pickle format
            self._load_pickle_format(filepath)
    
    def _load_fast_format(self, base_path: str):
        """Load index from fast format"""
        try:
            # Load manifest
            manifest_path = f"{base_path}.manifest"
            with open(manifest_path, 'rb') as f:
                manifest = orjson.loads(f.read())
            
            logger.info(f"Loading fast format index v{manifest['version']}")
            
            # Load numpy arrays
            arrays_path = os.path.join(os.path.dirname(base_path), manifest['arrays_file'])
            numpy_data = joblib.load(arrays_path)
            logger.info(f"Loaded {manifest['array_count']} numpy arrays")
            
            # Load serializable data
            data_path = os.path.join(os.path.dirname(base_path), manifest['data_file'])
            with open(data_path, 'rb') as f:
                serializable_data = orjson.loads(f.read())
            
            # Reconstruct data structures
            self._reconstruct_from_fast_format(serializable_data, numpy_data)
            
            logger.info(f"Fast format index loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fast format index: {e}")
            raise
    
    def _reconstruct_from_fast_format(self, serializable_data: dict, numpy_data: dict):
        """Reconstruct data structures from fast format"""
        # Restore configuration
        self.config = GlobalConfig(**serializable_data['config'])
        
        # Restore grid manager state
        self.grid_manager = GridManager(self.config)
        self.grid_manager.root_cell_id = serializable_data['root_cell_id']
        self.grid_manager.max_grid_depth_reached = serializable_data['max_grid_depth_reached']
        
        # Restore global PCA model
        if serializable_data['global_pca_model']:
            global_pca = {}
            for key, value in serializable_data['global_pca_model'].items():
                if isinstance(value, str) and value.startswith('__NUMPY_REF__'):
                    array_key = value.replace('__NUMPY_REF__', '')
                    global_pca[key] = numpy_data[array_key]
                else:
                    global_pca[key] = value
            self.grid_manager.global_pca_model = global_pca
        else:
            self.grid_manager.global_pca_model = None
        
        # Restore cells
        self.grid_manager.cells_store = {}
        for cell_id, cell_state in serializable_data['cells_store'].items():
            # Reconstruct cell data
            reconstructed_state = {}
            for key, value in cell_state.items():
                if isinstance(value, str) and value.startswith('__NUMPY_REF__'):
                    array_key = value.replace('__NUMPY_REF__', '')
                    reconstructed_state[key] = numpy_data[array_key]
                elif key == 'boundary_box' and isinstance(value, dict):
                    # Reconstruct AABB
                    min_coords = numpy_data[value['min_coords'].replace('__NUMPY_REF__', '')]
                    max_coords = numpy_data[value['max_coords'].replace('__NUMPY_REF__', '')]
                    reconstructed_state[key] = AABB(min_coords, max_coords)
                elif key == 'local_pca_model' and value is not None:
                    # Reconstruct PCA model
                    pca_model = {}
                    for pca_key, pca_value in value.items():
                        if isinstance(pca_value, str) and pca_value.startswith('__NUMPY_REF__'):
                            array_key = pca_value.replace('__NUMPY_REF__', '')
                            pca_model[pca_key] = numpy_data[array_key]
                        else:
                            pca_model[pca_key] = pca_value
                    reconstructed_state[key] = pca_model
                elif key == 'linkage_cache' and value is not None:
                    # Reconstruct LinkageEntry objects
                    linkage_cache = []
                    for entry_data in value:
                        # Reconstruct numpy array reference
                        if isinstance(entry_data.get('target_cell_rep_proj'), str) and entry_data['target_cell_rep_proj'].startswith('__NUMPY_REF__'):
                            array_key = entry_data['target_cell_rep_proj'].replace('__NUMPY_REF__', '')
                            target_cell_rep_proj = numpy_data[array_key]
                        else:
                            target_cell_rep_proj = entry_data.get('target_cell_rep_proj')
                        
                        # Create LinkageEntry object
                        entry = LinkageEntry(
                            target_cell_id=entry_data['target_cell_id'],
                            target_cell_rep_proj=target_cell_rep_proj,
                            activation_score=entry_data['activation_score'],
                            last_activation_timestamp=entry_data['last_activation_timestamp']
                        )
                        linkage_cache.append(entry)
                    reconstructed_state[key] = linkage_cache
                elif key == 'query_stats' and value is not None:
                    # Reconstruct QueryStats object
                    reconstructed_state[key] = QueryStats(
                        total_queries_passed_through=value.get('total_queries_passed_through', 0),
                        queries_hopped_via_linkage=value.get('queries_hopped_via_linkage', 0)
                    )
                else:
                    reconstructed_state[key] = value
            
            # Create cell object
            cell = CellObject(
                cell_id=reconstructed_state['cell_id'],
                level=reconstructed_state['level'],
                boundary_box=reconstructed_state['boundary_box']
            )
            cell.__setstate__(reconstructed_state)
            self.grid_manager.cells_store[cell_id] = cell
        
        # Restore point database
        self.point_db = PointDB()
        point_db_state = serializable_data.get('point_db_state', {})
        
        # Reconstruct point vectors
        point_vectors_refs = point_db_state.get('point_vectors_store', {})
        self.point_db.point_vectors_store = {}
        for point_id, ref in point_vectors_refs.items():
            if isinstance(ref, str) and ref.startswith('__NUMPY_REF__'):
                array_key = ref.replace('__NUMPY_REF__', '')
                self.point_db.point_vectors_store[point_id] = numpy_data[array_key]
        
        self.point_db.point_to_cell_map = point_db_state.get('point_to_cell_map', {})
        
        # Reinitialize components
        self.update_manager = UpdateManager(self.grid_manager, self.point_db, self.config)
        self.search_orchestrator = SearchOrchestrator(self.grid_manager, self.point_db, self.config)
        self.maintenance_scheduler = MaintenanceScheduler(self.grid_manager, self.config)
        
        # Set point_db reference in grid_manager
        self.grid_manager._point_db = self.point_db
        
        self.initialized = True
        self.maintenance_scheduler.start()
    
    def _load_pickle_format(self, filepath: str):
        """Load index from legacy pickle format"""
        logger.info("Loading legacy pickle format")
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        # Restore configuration
        self.config = GlobalConfig(**index_data['config'])
        
        # Restore grid manager state
        self.grid_manager = GridManager(self.config)
        self.grid_manager.root_cell_id = index_data['root_cell_id']
        self.grid_manager.max_grid_depth_reached = index_data['max_grid_depth_reached']
        self.grid_manager.global_pca_model = index_data.get('global_pca_model')
        
        # Restore cells
        self.grid_manager.cells_store = {}
        for cell_id, cell_state in index_data['cells_store'].items():
            # Recreate CellObject
            cell = CellObject(
                cell_id=cell_state['cell_id'],
                level=cell_state['level'],
                boundary_box=cell_state['boundary_box']
            )
            cell.__setstate__(cell_state)
            self.grid_manager.cells_store[cell_id] = cell
        
        # Restore point database
        self.point_db = PointDB()
        point_db_state = index_data.get('point_db_state', {})
        self.point_db.point_vectors_store = point_db_state.get('point_vectors_store', {})
        self.point_db.point_to_cell_map = point_db_state.get('point_to_cell_map', {})
        
        # Reinitialize components
        self.update_manager = UpdateManager(self.grid_manager, self.point_db, self.config)
        self.search_orchestrator = SearchOrchestrator(self.grid_manager, self.point_db, self.config)
        self.maintenance_scheduler = MaintenanceScheduler(self.grid_manager, self.config)
        
        # Set point_db reference in grid_manager
        self.grid_manager._point_db = self.point_db
        
        self.initialized = True
        self.maintenance_scheduler.start()
        
        logger.info(f"Legacy index loaded from {filepath}")


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_usage():
    """Example usage of DARG v2.2 with 1 BILLION data points"""
    
    print("DARG v2.2 - Testing with 1 BILLION Data Points")
    print("=" * 60)
    
    # Create system with heavily optimized configuration for 1B points
    darg = DARGv22()
    
    # Ultra-optimized configuration for 1B points
    darg.config.base_max_pop_per_cell = 2000  # Much larger cells
    darg.config.beam_width_B = 12  # Wider beam search
    darg.config.K_top_candidates = 200  # More candidates
    darg.config.pca_batch_size = 5000  # Larger PCA batches
    darg.config.maintenance_interval_seconds = 1800  # Much less frequent maintenance (30 min)
    darg.config.max_grid_depth = 25  # Allow deeper grid
    darg.config.proj_dimensions = 8  # Smaller projections for speed
    darg.config.LID_sample_size = 30  # Smaller LID samples
    darg.config.updates_threshold_for_pca = 200  # Less frequent PCA updates
    
    # Generate initial sample data for grid initialization
    print("Generating initial sample data...")
    np.random.seed(42)
    initial_sample = [np.random.randn(128) for _ in range(5000)]  # Larger initial sample
    
    # Initialize with sample data
    print("Initializing DARG system...")
    start_time = time.time()
    darg.initialize(initial_sample)
    init_time = time.time() - start_time
    print(f"System initialized in {init_time:.2f} seconds")
    
    # Generate and insert 1 BILLION data points
    print("\nGenerating and inserting 1,000,000,000 data points...")
    total_points = 1_000_000_000  # 1 billion
    batch_size = 100000  # Much larger batches for efficiency
    num_batches = total_points // batch_size
    
    print(f"Processing {num_batches:,} batches of {batch_size:,} points each")
    
    total_start_time = time.time()
    checkpoint_interval = 500  # Save checkpoint every 500 batches
    
    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        
        # Generate batch of data points with varied distributions for realism
        if batch_idx % 100 == 0:
            # Every 100th batch: clustered data
            centers = [np.random.randn(128) * 10 for _ in range(10)]
            batch_data = []
            for _ in range(batch_size):
                center = centers[np.random.randint(0, len(centers))]
                noise = np.random.randn(128) * 0.5
                batch_data.append(center + noise)
        else:
            # Regular random data
            batch_data = [np.random.randn(128) for _ in range(batch_size)]
        
        # Insert batch points
        successful_insertions = 0
        for i, vector in enumerate(batch_data):
            point_id = f"point_{batch_idx * batch_size + i}"
            success = darg.insert(point_id, vector)
            if success:
                successful_insertions += 1
        
        batch_time = time.time() - batch_start_time
        total_inserted = (batch_idx + 1) * batch_size
        
        # Progress update every 10 batches
        if (batch_idx + 1) % 10 == 0:
            elapsed_time = time.time() - total_start_time
            rate = total_inserted / elapsed_time
            percent_complete = (batch_idx + 1) / num_batches * 100
            eta_seconds = (num_batches - batch_idx - 1) * (elapsed_time / (batch_idx + 1))
            eta_hours = eta_seconds / 3600
            
            print(f"Batch {batch_idx + 1:4d}/{num_batches}: {successful_insertions:6d}/{batch_size} points "
                  f"in {batch_time:6.2f}s | Total: {total_inserted:11,} points | "
                  f"{percent_complete:5.2f}% | Rate: {rate:6.0f} pts/sec | ETA: {eta_hours:.1f}h")
        
        # Print detailed stats every 100 batches
        if (batch_idx + 1) % 100 == 0:
            stats = darg.get_stats()
            elapsed_time = time.time() - total_start_time
            rate = total_inserted / elapsed_time
            print(f"  *** Stats: {stats['total_cells']:,} cells, {stats['leaf_cells']:,} leaves, "
                  f"depth {stats['max_depth']} | Rate: {rate:.0f} points/sec ***")
        
        # Checkpoint saving every 500 batches
        if (batch_idx + 1) % checkpoint_interval == 0:
            checkpoint_start = time.time()
            checkpoint_file = f"darg_1b_checkpoint_{batch_idx + 1}.pkl"
            print(f"\n  Saving checkpoint at batch {batch_idx + 1}...")
            darg.save_index(checkpoint_file)
            checkpoint_time = time.time() - checkpoint_start
            print(f"  Checkpoint saved in {checkpoint_time:.2f} seconds\n")
        
        # Memory cleanup hint every 1000 batches
        if (batch_idx + 1) % 1000 == 0:
            import gc
            gc.collect()
    

    
    total_insertion_time = time.time() - total_start_time
    insertion_rate = total_points / total_insertion_time
    
    print(f"\nInsertion completed!")
    print(f"Total time: {total_insertion_time:.2f} seconds ({total_insertion_time/3600:.2f} hours)")
    print(f"Average rate: {insertion_rate:.0f} points/second")
    
    # Get final system statistics
    print("\n" + "=" * 60)
    print("FINAL SYSTEM STATISTICS")
    print("=" * 60)
    stats = darg.get_stats()
    for key, value in stats.items():
        if isinstance(value, int):
            print(f"{key:25}: {value:,}")
        else:
            print(f"{key:25}: {value}")
    
    # Test search performance with reduced query count for time
    print("\n" + "=" * 60)
    print("SEARCH PERFORMANCE TEST")
    print("=" * 60)
    
    # Test fewer queries but different k values
    test_queries = 20  # Reduced for 1B scale
    k_values = [1, 10, 50, 100, 500]  # Added k=500 for large scale
    
    for k in k_values:
        print(f"\nTesting {test_queries} queries with k={k}...")
        search_times = []
        
        for query_idx in range(test_queries):
            # Generate random query vector
            query_vector = np.random.randn(128)
            
            # Measure search time
            search_start = time.time()
            results = darg.search(query_vector, k=k)
            search_time = time.time() - search_start
            
            search_times.append(search_time)
        
        # Calculate statistics
        avg_time = sum(search_times) / len(search_times)
        min_time = min(search_times)
        max_time = max(search_times)
        
        print(f"k={k:3d}: avg={avg_time*1000:7.2f}ms, min={min_time*1000:7.2f}ms, max={max_time*1000:7.2f}ms")
        
        # Show sample results for k=10
        if k == 10:
            query_vector = np.random.randn(128)
            results = darg.search(query_vector, k=10)
            print(f"Sample search results (k=10):")
            for i, (point_id, distance) in enumerate(results[:5]):
                print(f"  {i+1:2d}. {point_id}: {distance:.4f}")
    
    # Test deletion performance (reduced scale)
    print("\n" + "=" * 60)
    print("DELETION PERFORMANCE TEST")
    print("=" * 60)
    
    # Delete fewer random points for 1B scale
    delete_count = 10000  # Increased but reasonable
    print(f"Testing deletion of {delete_count:,} random points...")
    
    # Select random points to delete
    delete_points = [f"point_{np.random.randint(0, total_points)}" for _ in range(delete_count)]
    
    delete_start_time = time.time()
    successful_deletions = 0
    
    for i, point_id in enumerate(delete_points):
        success = darg.delete(point_id)
        if success:
            successful_deletions += 1
        
        # Progress for deletions
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - delete_start_time
            rate = (i + 1) / elapsed
            print(f"  Deleted {i + 1:,}/{delete_count:,} points ({rate:.0f} deletions/sec)")
    
    delete_time = time.time() - delete_start_time
    delete_rate = successful_deletions / delete_time
    
    print(f"Deleted {successful_deletions:,}/{delete_count:,} points in {delete_time:.2f} seconds")
    print(f"Deletion rate: {delete_rate:.0f} points/second")
    
    # Final stats after deletions
    final_stats = darg.get_stats()
    print(f"Final point count: {final_stats['total_points']:,}")
    
    # Save final index
    print("\n" + "=" * 60)
    print("FINAL INDEX SERIALIZATION")
    print("=" * 60)
    
    print("Saving final 1B point index to disk...")
    save_start = time.time()
    darg.save_index("darg_1b_final_index.pkl")
    save_time = time.time() - save_start
    print(f"Final index saved in {save_time:.2f} seconds")
    
    # Check file sizes
    import os
    file_stats = []
    for ext in ['.manifest', '_arrays.joblib', '_data.json']:
        filepath = f"darg_1b_final_index{ext}"
        if os.path.exists(filepath):
            size_gb = os.path.getsize(filepath) / (1024 * 1024 * 1024)
            file_stats.append(f"{filepath}: {size_gb:.2f} GB")
    
    if file_stats:
        print("Final saved files:")
        for stat in file_stats:
            print(f"  {stat}")
    
    # Shutdown
    darg.shutdown()
    
    # Test loading (optional for 1B scale - might be time consuming)
    test_loading = input("\nTest loading the 1B index? (y/n): ").lower() == 'y'
    
    if test_loading:
        print("Testing 1B index loading...")
        darg2 = DARGv22()
        
        load_start = time.time()
        darg2.load_index("darg_1b_final_index.pkl")
        load_time = time.time() - load_start
        print(f"1B index loaded in {load_time:.2f} seconds")
        
        # Verify loaded system
        loaded_stats = darg2.get_stats()
        print(f"Loaded system has {loaded_stats['total_points']:,} points")
        
        # Test search in loaded system
        query_vector = np.random.randn(128)
        results2 = darg2.search(query_vector, k=5)
        print(f"\nLoaded system search results:")
        for i, (point_id, distance) in enumerate(results2):
            print(f"  {i+1}. {point_id}: {distance:.4f}")
        
        darg2.shutdown()
    
    # Print performance summary
    print("\n" + "=" * 60)
    print("1 BILLION POINT PERFORMANCE SUMMARY")
    print("=" * 60)
    performance_timer.print_summary()
    
    print(f"\nTotal points processed: {total_points:,}")
    print(f"Total processing time: {total_insertion_time/3600:.2f} hours")
    print(f"Average insertion rate: {insertion_rate:.0f} points/second")

if __name__ == "__main__":
    example_usage()
