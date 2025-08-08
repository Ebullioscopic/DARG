"""
Enhanced DARG Implementation with Dynamic Vector Graph
=====================================================

Advanced DARG implementation with:
- Dynamic vector graph construction
- Incremental updates without full rebuilds
- Enhanced performance optimizations
- Real-time similarity tracking
"""

import numpy as np
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import heapq
from concurrent.futures import ThreadPoolExecutor
import logging

# Import fallback if main doesn't work
try:
    from darg_complete import DARG
    print("Successfully imported DARG from darg_complete")
except ImportError:
    try:
        from main import DARG
        print("Successfully imported DARG from main")
    except ImportError:
        print("Could not import DARG, using minimal fallback")
        # Minimal fallback
        class DARG:
            def __init__(self, *args, **kwargs):
                pass
            def search(self, query, k=10):
                return []

logger = logging.getLogger(__name__)

@dataclass
class VectorNode:
    """Enhanced vector node with dynamic properties"""
    id: str
    vector: np.ndarray
    data_type: str
    timestamp: float
    metadata: Dict[str, Any]
    neighbors: List[str] = None
    local_id: float = None  # Local Intrinsic Dimensionality
    partition_id: int = None
    update_count: int = 0
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []
        if self.timestamp is None:
            self.timestamp = time.time()

class DynamicLinkageCache:
    """Dynamic cache for vector relationships with echo calibration"""
    
    def __init__(self, max_size: int = 1000, decay_factor: float = 0.95):
        self.max_size = max_size
        self.decay_factor = decay_factor
        self.cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        
    def update_link(self, node1_id: str, node2_id: str, similarity: float, success: bool = True):
        """Update link strength with echo calibration"""
        with self.lock:
            current_time = time.time()
            
            # Echo calibration formula: s(t+1) = α * s(t) + (1-α) * success_score
            success_score = similarity if success else similarity * 0.5
            
            if node1_id in self.cache and node2_id in self.cache[node1_id]:
                current_strength = self.cache[node1_id][node2_id]
                new_strength = self.decay_factor * current_strength + (1 - self.decay_factor) * success_score
            else:
                new_strength = success_score
            
            self.cache[node1_id][node2_id] = new_strength
            self.cache[node2_id][node1_id] = new_strength  # Symmetric
            
            self.access_times[f"{node1_id}-{node2_id}"] = current_time
            
            # Cleanup if cache is too large
            if len(self.cache) > self.max_size:
                self._cleanup_cache()
    
    def get_link_strength(self, node1_id: str, node2_id: str) -> float:
        """Get link strength between nodes"""
        with self.lock:
            return self.cache.get(node1_id, {}).get(node2_id, 0.0)
    
    def get_neighbors(self, node_id: str, min_strength: float = 0.1) -> List[Tuple[str, float]]:
        """Get neighbors above minimum strength threshold"""
        with self.lock:
            neighbors = []
            for neighbor_id, strength in self.cache.get(node_id, {}).items():
                if strength >= min_strength:
                    neighbors.append((neighbor_id, strength))
            return sorted(neighbors, key=lambda x: x[1], reverse=True)
    
    def _cleanup_cache(self):
        """Remove least recently used entries"""
        # Remove 10% of oldest entries
        current_time = time.time()
        entries_to_remove = int(len(self.cache) * 0.1)
        
        # Sort by access time
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for entry_key, _ in sorted_entries[:entries_to_remove]:
            node1_id, node2_id = entry_key.split('-')
            if node1_id in self.cache and node2_id in self.cache[node1_id]:
                del self.cache[node1_id][node2_id]
                if node2_id in self.cache and node1_id in self.cache[node2_id]:
                    del self.cache[node2_id][node1_id]
            if entry_key in self.access_times:
                del self.access_times[entry_key]

class AdaptiveBeamSearch:
    """Enhanced beam search with epsilon-greedy exploration"""
    
    def __init__(self, base_beam_width: int = 32, epsilon_base: float = 0.1):
        self.base_beam_width = base_beam_width
        self.epsilon_base = epsilon_base
        self.query_stats = defaultdict(int)
        self.miss_stats = defaultdict(int)
        
    def adaptive_epsilon(self, partition_id: int) -> float:
        """Calculate adaptive epsilon based on recent performance"""
        recent_queries = self.query_stats[partition_id]
        recent_misses = self.miss_stats[partition_id]
        
        if recent_queries == 0:
            return self.epsilon_base
        
        miss_rate = recent_misses / recent_queries
        beta = 0.5  # Adaptation factor
        
        return self.epsilon_base * (1 + beta * miss_rate)
    
    def update_stats(self, partition_id: int, success: bool):
        """Update query statistics"""
        self.query_stats[partition_id] += 1
        if not success:
            self.miss_stats[partition_id] += 1
    
    def beam_search(self, 
                   query_vector: np.ndarray,
                   root_nodes: List[VectorNode],
                   linkage_cache: DynamicLinkageCache,
                   k: int = 10) -> List[Tuple[str, float]]:
        """Perform adaptive beam search"""
        
        # Phase 1: Initial beam from root nodes
        beam = []
        for node in root_nodes:
            distance = np.linalg.norm(query_vector - node.vector)
            heapq.heappush(beam, (distance, node.id, node))
        
        # Keep only top beam_width candidates
        beam = heapq.nsmallest(self.base_beam_width, beam)
        
        visited = set()
        results = []
        
        # Phase 2: Expand beam using linkage cache
        while beam and len(results) < k:
            distance, node_id, node = heapq.heappop(beam)
            
            if node_id in visited:
                continue
                
            visited.add(node_id)
            results.append((node_id, distance))
            
            # Get neighbors from linkage cache
            neighbors = linkage_cache.get_neighbors(node_id)
            
            # Epsilon-greedy exploration
            epsilon = self.adaptive_epsilon(node.partition_id)
            
            for neighbor_id, link_strength in neighbors:
                if neighbor_id not in visited:
                    # Explore with probability based on link strength and epsilon
                    explore_prob = link_strength + epsilon
                    if np.random.random() < explore_prob:
                        # Add neighbor to beam (need to reconstruct neighbor node)
                        neighbor_distance = distance + (1.0 - link_strength)  # Approximate
                        heapq.heappush(beam, (neighbor_distance, neighbor_id, None))
        
        return results[:k]

class EnhancedDARG:
    """Enhanced DARG with dynamic vector graph capabilities"""
    
    def __init__(self, vector_dim: int = None, config: Dict[str, Any] = None):
        self.vector_dim = vector_dim or 128
        self.config = config or self._default_config()
        self.nodes: Dict[str, VectorNode] = {}
        self.partitions: Dict[int, List[str]] = defaultdict(list)
        self.linkage_cache = DynamicLinkageCache(
            max_size=self.config.get('max_cache_size', 10000)
        )
        self.beam_search = AdaptiveBeamSearch(
            base_beam_width=self.config.get('beam_width', 32)
        )
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'avg_latency': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Enhanced DARG initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'vector_dim': 384,
            'num_layers': 5,
            'grid_size': 64,
            'beam_width': 32,
            'max_cache_size': 10000,
            'lid_k': 10,  # K for LID estimation
            'echo_threshold': 0.75,
            'lazy_update_threshold': 100
        }
    
    def estimate_local_intrinsic_dimensionality(self, 
                                               node_id: str, 
                                               k: int = None) -> float:
        """Estimate Local Intrinsic Dimensionality using Two-NN estimator"""
        if k is None:
            k = self.config['lid_k']
        
        with self.lock:
            if node_id not in self.nodes:
                return 0.0
            
            query_node = self.nodes[node_id]
            query_vector = query_node.vector
            
            # Find k nearest neighbors
            distances = []
            for other_id, other_node in self.nodes.items():
                if other_id != node_id:
                    dist = np.linalg.norm(query_vector - other_node.vector)
                    distances.append(dist)
            
            if len(distances) < k:
                return float(len(distances))
            
            distances.sort()
            neighbors_distances = distances[:k]
            
            # LID estimation: LID(x) = -1/k * Σ(log(ri(x)/rk(x)))^-1
            r_k = neighbors_distances[-1]  # Distance to k-th neighbor
            
            if r_k == 0:
                return float(k)
            
            lid_sum = 0.0
            for r_i in neighbors_distances:
                if r_i > 0:
                    ratio = r_i / r_k
                    if ratio > 0:
                        lid_sum += np.log(ratio)
            
            if lid_sum == 0:
                return float(k)
            
            lid = -1.0 / k * (1.0 / lid_sum) if lid_sum != 0 else float(k)
            return max(1.0, min(float(k), lid))
    
    def add_vector(self, 
                   vector: np.ndarray, 
                   node_id: str,
                   data_type: str = "unknown",
                   metadata: Dict[str, Any] = None) -> bool:
        """Add vector to dynamic graph with incremental updates"""
        
        start_time = time.time()
        
        with self.lock:
            # Create node
            node = VectorNode(
                id=node_id,
                vector=vector,
                data_type=data_type,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # Estimate LID
            node.local_id = self.estimate_local_intrinsic_dimensionality(node_id)
            
            # Assign to partition (simplified grid-based)
            partition_id = self._assign_partition(vector)
            node.partition_id = partition_id
            
            # Store node
            self.nodes[node_id] = node
            self.partitions[partition_id].append(node_id)
            
            # Update linkage cache with similar nodes
            self._update_linkage_for_new_node(node)
            
            # Log performance
            latency = time.time() - start_time
            self._update_performance_stats(latency)
            
            logger.info(f"Added vector {node_id} to partition {partition_id} (LID: {node.local_id:.2f})")
            return True
    
    def _assign_partition(self, vector: np.ndarray) -> int:
        """Assign vector to partition using grid-based approach"""
        # Simplified grid assignment
        grid_size = self.config['grid_size']
        
        # Normalize vector to [0, 1] range for each dimension
        normalized = (vector - vector.min()) / (vector.max() - vector.min() + 1e-8)
        
        # Map to grid coordinates
        grid_coords = (normalized * grid_size).astype(int)
        grid_coords = np.clip(grid_coords, 0, grid_size - 1)
        
        # Convert to single partition ID
        partition_id = 0
        for i, coord in enumerate(grid_coords[:3]):  # Use first 3 dimensions
            partition_id += coord * (grid_size ** i)
        
        return partition_id % 10000  # Limit partition IDs
    
    def _update_linkage_for_new_node(self, new_node: VectorNode):
        """Update linkage cache for newly added node"""
        # Find similar nodes and update linkage cache
        similarities = []
        
        for other_id, other_node in self.nodes.items():
            if other_id != new_node.id:
                # Calculate similarity
                distance = np.linalg.norm(new_node.vector - other_node.vector)
                similarity = 1.0 / (1.0 + distance)
                
                similarities.append((other_id, similarity))
        
        # Sort by similarity and take top connections
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_connections = similarities[:20]  # Connect to top 20 similar nodes
        
        # Update linkage cache
        for other_id, similarity in top_connections:
            if similarity > 0.1:  # Minimum similarity threshold
                self.linkage_cache.update_link(new_node.id, other_id, similarity, True)
    
    def search(self, 
              query_vector: np.ndarray, 
              k: int = 10,
              use_beam_search: bool = True) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for k nearest neighbors"""
        
        start_time = time.time()
        
        with self.lock:
            self.performance_stats['total_queries'] += 1
            
            if not self.nodes:
                return []
            
            if use_beam_search:
                # Use enhanced beam search
                root_nodes = self._get_root_nodes(query_vector)
                beam_results = self.beam_search.beam_search(
                    query_vector, root_nodes, self.linkage_cache, k
                )
                
                # Convert to final format
                results = []
                for node_id, distance in beam_results:
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        results.append((node_id, distance, {
                            'data_type': node.data_type,
                            'timestamp': node.timestamp,
                            'metadata': node.metadata,
                            'local_id': node.local_id
                        }))
                
            else:
                # Brute force search for comparison
                results = []
                for node_id, node in self.nodes.items():
                    distance = np.linalg.norm(query_vector - node.vector)
                    results.append((node_id, distance, {
                        'data_type': node.data_type,
                        'timestamp': node.timestamp,
                        'metadata': node.metadata,
                        'local_id': node.local_id
                    }))
                
                results.sort(key=lambda x: x[1])
                results = results[:k]
            
            # Update performance stats
            latency = time.time() - start_time
            self._update_performance_stats(latency)
            
            return results
    
    def _get_root_nodes(self, query_vector: np.ndarray) -> List[VectorNode]:
        """Get root nodes for beam search"""
        # Find the partition for the query
        partition_id = self._assign_partition(query_vector)
        
        # Get nodes from the same partition and neighboring partitions
        root_node_ids = set()
        root_node_ids.update(self.partitions.get(partition_id, []))
        
        # Add neighboring partitions
        for neighbor_partition in range(max(0, partition_id - 1), partition_id + 2):
            root_node_ids.update(self.partitions.get(neighbor_partition, []))
        
        # Convert to VectorNode objects
        root_nodes = []
        for node_id in root_node_ids:
            if node_id in self.nodes:
                root_nodes.append(self.nodes[node_id])
        
        return root_nodes[:50]  # Limit root nodes
    
    def _update_performance_stats(self, latency: float):
        """Update performance statistics"""
        current_avg = self.performance_stats['avg_latency']
        total_queries = self.performance_stats['total_queries']
        
        # Running average (avoid division by zero)
        if total_queries == 0:
            new_avg = latency
        else:
            new_avg = (current_avg * (total_queries - 1) + latency) / total_queries
        self.performance_stats['avg_latency'] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = self.performance_stats.copy()
            stats['total_nodes'] = len(self.nodes)
            stats['total_partitions'] = len(self.partitions)
            stats['cache_size'] = len(self.linkage_cache.cache)
            
            # Calculate recall and other metrics
            if stats['total_queries'] > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            else:
                stats['cache_hit_rate'] = 0.0
            
            return stats
    
    def benchmark_against_baselines(self, 
                                   test_vectors: np.ndarray,
                                   test_ids: List[str],
                                   query_vectors: np.ndarray,
                                   k: int = 10) -> Dict[str, Any]:
        """Benchmark Enhanced DARG against baseline methods"""
        
        logger.info("Starting benchmark against baselines...")
        
        # Build index with test vectors
        for i, (vector, vector_id) in enumerate(zip(test_vectors, test_ids)):
            self.add_vector(vector, vector_id, f"test_type_{i % 3}")
        
        # Benchmark Enhanced DARG
        darg_times = []
        darg_results = []
        
        for query in query_vectors:
            start_time = time.time()
            results = self.search(query, k=k, use_beam_search=True)
            end_time = time.time()
            
            darg_times.append(end_time - start_time)
            darg_results.append(results)
        
        darg_avg_time = np.mean(darg_times)
        
        # Benchmark brute force (baseline)
        brute_times = []
        brute_results = []
        
        for query in query_vectors:
            start_time = time.time()
            results = self.search(query, k=k, use_beam_search=False)
            end_time = time.time()
            
            brute_times.append(end_time - start_time)
            brute_results.append(results)
        
        brute_avg_time = np.mean(brute_times)
        
        # Calculate metrics
        speedup = brute_avg_time / darg_avg_time
        
        # Calculate recall (simplified)
        total_recall = 0.0
        for i in range(len(query_vectors)):
            darg_ids = set([r[0] for r in darg_results[i]])
            brute_ids = set([r[0] for r in brute_results[i]])
            
            if len(brute_ids) > 0:
                recall = len(darg_ids.intersection(brute_ids)) / len(brute_ids)
                total_recall += recall
        
        avg_recall = total_recall / len(query_vectors) if len(query_vectors) > 0 else 0.0
        
        benchmark_results = {
            'enhanced_darg_latency_ms': darg_avg_time * 1000,
            'brute_force_latency_ms': brute_avg_time * 1000,
            'speedup': speedup,
            'recall_at_k': avg_recall,
            'total_nodes': len(self.nodes),
            'total_partitions': len(self.partitions),
            'cache_size': len(self.linkage_cache.cache),
            'performance_stats': self.get_performance_stats()
        }
        
        logger.info(f"Benchmark completed: {speedup:.2f}x speedup, {avg_recall:.3f} recall@{k}")
        
        return benchmark_results
    
    def save_graph(self, filepath: str):
        """Save dynamic graph state"""
        state = {
            'config': self.config,
            'nodes': {k: {
                'id': v.id,
                'vector': v.vector.tolist(),
                'data_type': v.data_type,
                'timestamp': v.timestamp,
                'metadata': v.metadata,
                'neighbors': v.neighbors,
                'local_id': v.local_id,
                'partition_id': v.partition_id,
                'update_count': v.update_count
            } for k, v in self.nodes.items()},
            'partitions': {k: v for k, v in self.partitions.items()},
            'performance_stats': self.performance_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Enhanced DARG graph saved to {filepath}")
    
    def load_graph(self, filepath: str):
        """Load dynamic graph state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = state['config']
        self.performance_stats = state.get('performance_stats', {})
        
        # Restore nodes
        self.nodes = {}
        for k, v in state['nodes'].items():
            node = VectorNode(
                id=v['id'],
                vector=np.array(v['vector']),
                data_type=v['data_type'],
                timestamp=v['timestamp'],
                metadata=v['metadata'],
                neighbors=v['neighbors'],
                local_id=v['local_id'],
                partition_id=v['partition_id'],
                update_count=v['update_count']
            )
            self.nodes[k] = node
        
        # Restore partitions
        self.partitions = defaultdict(list)
        for k, v in state['partitions'].items():
            self.partitions[int(k)] = v
        
        # Rebuild linkage cache
        self._rebuild_linkage_cache()
        
        logger.info(f"Enhanced DARG graph loaded from {filepath}")
    
    def _rebuild_linkage_cache(self):
        """Rebuild linkage cache from nodes"""
        logger.info("Rebuilding linkage cache...")
        
        for node_id, node in self.nodes.items():
            self._update_linkage_for_new_node(node)
        
        logger.info(f"Linkage cache rebuilt with {len(self.linkage_cache.cache)} entries")

def demo_enhanced_darg():
    """Demonstrate Enhanced DARG capabilities"""
    print("🚀 Enhanced DARG Demo")
    print("====================")
    
    # Initialize Enhanced DARG
    enhanced_darg = EnhancedDARG()
    
    # Generate test data
    np.random.seed(42)
    n_vectors = 1000
    vector_dim = 128
    test_vectors = np.random.randn(n_vectors, vector_dim).astype(np.float32)
    test_ids = [f"vec_{i}" for i in range(n_vectors)]
    
    print(f"\n📊 Adding {n_vectors} vectors to Enhanced DARG...")
    start_time = time.time()
    
    for i, (vector, vector_id) in enumerate(zip(test_vectors, test_ids)):
        enhanced_darg.add_vector(vector, vector_id, f"type_{i % 5}")
        
        if (i + 1) % 200 == 0:
            print(f"  Added {i + 1}/{n_vectors} vectors...")
    
    build_time = time.time() - start_time
    print(f"✅ Index built in {build_time:.2f} seconds")
    
    # Generate queries
    n_queries = 100
    query_vectors = np.random.randn(n_queries, vector_dim).astype(np.float32)
    
    print(f"\n🔍 Running {n_queries} search queries...")
    
    # Benchmark
    benchmark_results = enhanced_darg.benchmark_against_baselines(
        test_vectors[:500],  # Use subset for benchmarking
        test_ids[:500],
        query_vectors[:10],  # Use subset for queries
        k=10
    )
    
    print(f"\n📈 Benchmark Results:")
    for key, value in benchmark_results.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Demonstrate search
    print(f"\n🎯 Demonstrating search...")
    query = query_vectors[0]
    results = enhanced_darg.search(query, k=5)
    
    print(f"  Query vector shape: {query.shape}")
    print(f"  Found {len(results)} results:")
    for i, (node_id, distance, metadata) in enumerate(results):
        print(f"    {i+1}. {node_id}: distance={distance:.4f}, type={metadata['data_type']}")
    
    # Performance statistics
    print(f"\n📊 Performance Statistics:")
    stats = enhanced_darg.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✅ Enhanced DARG demo completed!")

if __name__ == "__main__":
    demo_enhanced_darg()
