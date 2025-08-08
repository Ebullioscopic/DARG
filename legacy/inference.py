#!/usr/bin/env python3
"""
DARG Inference Module
High-level inference classes and functions for DARG
"""

import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

from main import DARGv22, GlobalConfig
from platform_detection import get_platform_config, setup_acceleration
from dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

@dataclass
@dataclass
class InferenceConfig:
    """Configuration for DARG inference"""
    model_path: str
    batch_size: int = 1000
    k_default: int = 10
    use_acceleration: bool = True
    timeout_seconds: float = 30.0
    cache_size: int = 10000
    models_dir: str = "models"
    cache_dir: str = "cache"
    
    def __post_init__(self):
        """Create directories after initialization"""
        Path(self.models_dir).mkdir(exist_ok=True)
        Path(self.cache_dir).mkdir(exist_ok=True)

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    point_id: str
    distance: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class BatchSearchResult:
    """Result from batch search operation"""
    query_id: str
    results: List[SearchResult]
    latency_ms: float
    total_candidates: int
    
class DARGInferenceEngine:
    """High-level inference engine for DARG"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.darg_system = None
        self.platform_config = get_platform_config()
        self.acceleration_config = setup_acceleration()
        self.metadata_cache = {}
        self.stats = {
            'total_queries': 0,
            'total_latency': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"DARG Inference Engine initialized")
        logger.info(f"Platform: {self.platform_config.system}")
        logger.info(f"Acceleration: {self.acceleration_config['type']}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load DARG model from disk"""
        if model_path is None:
            model_path = self.config.model_path
        
        # Handle different file extensions - check for manifest or .pkl
        base_path = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
        manifest_path = f"{base_path}.manifest"
        
        # Check if new format or legacy format exists
        if Path(manifest_path).exists():
            # New fast format
            model_path = base_path
        elif Path(f"{base_path}.pkl").exists():
            # Legacy format
            model_path = f"{base_path}.pkl"
        elif Path(model_path).exists():
            # Direct path provided
            pass
        else:
            logger.error(f"Model file not found: {model_path} (also checked {manifest_path} and {base_path}.pkl)")
            return False
        
        try:
            logger.info(f"Loading DARG model from {model_path}")
            start_time = time.time()
            
            self.darg_system = DARGv22()
            self.darg_system.load_index(model_path)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Get model statistics
            stats = self.darg_system.get_stats()
            logger.info(f"Model contains {stats['total_points']:,} points in {stats['total_cells']:,} cells")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: Optional[int] = None, 
               timeout: Optional[float] = None) -> List[SearchResult]:
        """Search for nearest neighbors"""
        if self.darg_system is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if k is None:
            k = self.config.k_default
        
        if timeout is None:
            timeout = self.config.timeout_seconds
        
        try:
            start_time = time.time()
            
            # Perform search with timeout
            raw_results = self.darg_system.search(query_vector, k)
            
            search_time = time.time() - start_time
            
            # Check timeout
            if search_time > timeout:
                logger.warning(f"Search exceeded timeout: {search_time:.2f}s > {timeout}s")
            
            # Convert to SearchResult objects
            results = []
            for point_id, distance in raw_results:
                # Calculate confidence score (inverse of normalized distance)
                confidence = max(0.0, 1.0 - min(distance / 10.0, 1.0))
                
                # Get metadata if available
                metadata = self.metadata_cache.get(point_id, {})
                
                result = SearchResult(
                    point_id=point_id,
                    distance=distance,
                    confidence=confidence,
                    metadata=metadata
                )
                results.append(result)
            
            # Update statistics
            self.stats['total_queries'] += 1
            self.stats['total_latency'] += search_time
            
            logger.debug(f"Search completed: {len(results)} results in {search_time*1000:.2f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def batch_search(self, query_vectors: List[np.ndarray], 
                    k: Optional[int] = None,
                    query_ids: Optional[List[str]] = None) -> List[BatchSearchResult]:
        """Perform batch search operations"""
        if self.darg_system is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if k is None:
            k = self.config.k_default
        
        if query_ids is None:
            query_ids = [f"query_{i}" for i in range(len(query_vectors))]
        
        if len(query_ids) != len(query_vectors):
            raise ValueError("Number of query_ids must match number of query_vectors")
        
        results = []
        
        logger.info(f"Starting batch search: {len(query_vectors)} queries")
        
        for i, (query_id, query_vector) in enumerate(zip(query_ids, query_vectors)):
            start_time = time.time()
            
            try:
                search_results = self.search(query_vector, k)
                latency_ms = (time.time() - start_time) * 1000
                
                batch_result = BatchSearchResult(
                    query_id=query_id,
                    results=search_results,
                    latency_ms=latency_ms,
                    total_candidates=len(search_results)
                )
                
                results.append(batch_result)
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    avg_latency = sum(r.latency_ms for r in results) / len(results)
                    logger.info(f"Processed {i + 1}/{len(query_vectors)} queries, "
                              f"avg latency: {avg_latency:.2f}ms")
                
            except Exception as e:
                logger.error(f"Batch search failed for query {query_id}: {e}")
                # Add empty result for failed queries
                batch_result = BatchSearchResult(
                    query_id=query_id,
                    results=[],
                    latency_ms=0.0,
                    total_candidates=0
                )
                results.append(batch_result)
        
        logger.info(f"Batch search completed: {len(results)} results")
        return results
    
    def add_metadata(self, point_id: str, metadata: Dict[str, Any]) -> None:
        """Add metadata for a point"""
        self.metadata_cache[point_id] = metadata
        
        # Manage cache size
        if len(self.metadata_cache) > self.config.cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.metadata_cache.keys())[:-self.config.cache_size]
            for key in keys_to_remove:
                del self.metadata_cache[key]
    
    def bulk_add_metadata(self, metadata_dict: Dict[str, Dict[str, Any]]) -> None:
        """Add metadata for multiple points"""
        for point_id, metadata in metadata_dict.items():
            self.add_metadata(point_id, metadata)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference engine statistics"""
        model_stats = self.darg_system.get_stats() if self.darg_system else {}
        
        avg_latency = (self.stats['total_latency'] / self.stats['total_queries'] 
                      if self.stats['total_queries'] > 0 else 0.0)
        
        cache_hit_rate = (self.stats['cache_hits'] / 
                         (self.stats['cache_hits'] + self.stats['cache_misses'])
                         if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0)
        
        return {
            'inference_stats': {
                'total_queries': self.stats['total_queries'],
                'average_latency_ms': avg_latency * 1000,
                'cache_hit_rate': cache_hit_rate,
                'metadata_cache_size': len(self.metadata_cache)
            },
            'model_stats': model_stats,
            'platform_info': {
                'system': self.platform_config.system,
                'acceleration': self.acceleration_config['type'],
                'gpu_available': self.platform_config.gpu_available
            }
        }
    
    def benchmark(self, test_queries: List[np.ndarray], 
                  k_values: List[int] = None) -> Dict[str, Any]:
        """Run benchmark on test queries"""
        if k_values is None:
            k_values = [1, 5, 10, 20, 50]
        
        logger.info(f"Running benchmark with {len(test_queries)} queries")
        
        benchmark_results = {}
        
        for k in k_values:
            logger.info(f"Benchmarking k={k}")
            
            latencies = []
            result_counts = []
            
            for query_vector in test_queries:
                start_time = time.time()
                results = self.search(query_vector, k)
                latency = time.time() - start_time
                
                latencies.append(latency)
                result_counts.append(len(results))
            
            benchmark_results[f'k_{k}'] = {
                'avg_latency_ms': np.mean(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'avg_results': np.mean(result_counts),
                'queries_per_second': len(test_queries) / sum(latencies)
            }
        
        logger.info("Benchmark completed")
        return benchmark_results
    
    def export_results(self, results: List[BatchSearchResult], 
                      output_path: str, format: str = 'json') -> bool:
        """Export search results to file"""
        try:
            output_path = Path(output_path)
            
            if format == 'json':
                # Convert results to JSON-serializable format
                export_data = []
                for batch_result in results:
                    batch_data = {
                        'query_id': batch_result.query_id,
                        'latency_ms': batch_result.latency_ms,
                        'total_candidates': batch_result.total_candidates,
                        'results': [
                            {
                                'point_id': r.point_id,
                                'distance': r.distance,
                                'confidence': r.confidence,
                                'metadata': r.metadata
                            }
                            for r in batch_result.results
                        ]
                    }
                    export_data.append(batch_data)
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format == 'csv':
                # Export as CSV (flattened format)
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['query_id', 'rank', 'point_id', 'distance', 'confidence', 'latency_ms'])
                    
                    for batch_result in results:
                        for rank, result in enumerate(batch_result.results):
                            writer.writerow([
                                batch_result.query_id,
                                rank + 1,
                                result.point_id,
                                result.distance,
                                result.confidence,
                                batch_result.latency_ms
                            ])
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Results exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown inference engine"""
        if self.darg_system:
            self.darg_system.shutdown()
            self.darg_system = None
        
        logger.info("DARG Inference Engine shutdown")

class DARGInferenceFactory:
    """Factory for creating DARG inference engines"""
    
    @staticmethod
    def create_engine(model_path: str, **kwargs) -> DARGInferenceEngine:
        """Create a DARG inference engine"""
        config = InferenceConfig(model_path=model_path, **kwargs)
        engine = DARGInferenceEngine(config)
        
        if not engine.load_model():
            raise RuntimeError(f"Failed to load model from {model_path}")
        
        return engine
    
    @staticmethod
    def create_from_dataset(dataset_name: str, model_save_path: str = None, 
                          **kwargs) -> DARGInferenceEngine:
        """Create inference engine by training on dataset"""
        logger.info(f"Creating inference engine from dataset: {dataset_name}")
        
        # Load dataset
        dataset_manager = DatasetManager()
        vectors, queries = dataset_manager.load_dataset(dataset_name)
        
        if vectors is None:
            raise ValueError(f"Could not load dataset: {dataset_name}")
        
        # Create and train DARG system
        darg_system = DARGv22()
        
        # Use first 1000 vectors for initialization - convert to list
        init_sample_size = min(1000, len(vectors))
        init_sample = [vectors[i] for i in range(init_sample_size)]
        logger.info(f"Initializing with {len(init_sample)} vectors")
        darg_system.initialize(init_sample)
        
        # Insert all vectors
        logger.info(f"Inserting {len(vectors):,} vectors")
        for i, vector in enumerate(vectors):
            point_id = f"vec_{i}"
            try:
                success = darg_system.insert(point_id, vector)
                # Handle case where success might be an array or boolean
                if hasattr(success, 'all'):
                    success = success.all()
                elif hasattr(success, '__iter__') and not isinstance(success, (str, bytes)):
                    success = all(success)
                
                if not success:
                    logger.warning(f"Failed to insert vector {i}")
            except Exception as e:
                logger.warning(f"Error inserting vector {i}: {e}")
            
            if (i + 1) % 1000 == 0:  # More frequent updates for debugging
                logger.info(f"Inserted {i + 1:,}/{len(vectors):,} vectors")
        
        # Save model if path specified
        if model_save_path:
            logger.info(f"Saving model to {model_save_path}")
            darg_system.save_index(model_save_path)
        
        # Create inference engine
        config = InferenceConfig(model_path=model_save_path or "", **kwargs)
        engine = DARGInferenceEngine(config)
        engine.darg_system = darg_system  # Use the trained system directly
        
        logger.info("Inference engine created from dataset")
        return engine

# Convenience functions
def create_inference_engine(model_path: str, **kwargs) -> DARGInferenceEngine:
    """Create a DARG inference engine"""
    return DARGInferenceFactory.create_engine(model_path, **kwargs)

def train_and_create_engine(dataset_name: str, model_save_path: str = None, 
                           **kwargs) -> DARGInferenceEngine:
    """Train DARG on dataset and create inference engine"""
    return DARGInferenceFactory.create_from_dataset(dataset_name, model_save_path, **kwargs)

def quick_search(model_path: str, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
    """Quick search function"""
    engine = create_inference_engine(model_path)
    try:
        results = engine.search(query_vector, k)
        return results
    finally:
        engine.shutdown()

if __name__ == "__main__":
    # Example usage
    print("DARG Inference Module")
    print("=" * 40)
    
    # This would normally use a real model file
    print("Example: Creating inference engine from synthetic dataset")
    
    try:
        # Create engine from synthetic dataset
        engine = train_and_create_engine('synthetic_small', 'test_model.pkl')
        
        # Test search
        query = np.random.randn(128)
        results = engine.search(query, k=5)
        
        print(f"Search results for random query:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.point_id}: {result.distance:.4f} (conf: {result.confidence:.3f})")
        
        # Get statistics
        stats = engine.get_statistics()
        print(f"\nEngine statistics:")
        print(f"  Total queries: {stats['inference_stats']['total_queries']}")
        print(f"  Average latency: {stats['inference_stats']['average_latency_ms']:.2f}ms")
        
        engine.shutdown()
        print("✅ Inference example completed")
        
    except Exception as e:
        print(f"❌ Inference example failed: {e}")
        import traceback
        traceback.print_exc()
