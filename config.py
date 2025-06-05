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
warnings.filterwarnings('ignore')
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
    def vector_mean(vectors: List[np.ndarray]) -> np.ndarray:
        """Calculate mean of vectors"""
        if not vectors:
            return np.array([])
        return np.mean(vectors, axis=0)
    
    @staticmethod
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
    def project_vector(vector: np.ndarray, pca_model: Dict[str, Any]) -> np.ndarray:
        """Project vector using PCA model"""
        if pca_model is None:
            return vector
        
        centered = vector - pca_model['mean']
        return np.dot(centered, pca_model['components'].T)
    
    @staticmethod
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