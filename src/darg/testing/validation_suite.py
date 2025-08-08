"""
Comprehensive Testing and Validation Suite for Universal DARG
=============================================================

Complete testing framework for validating Universal DARG performance,
accuracy, and scalability against state-of-the-art techniques.
"""

import time
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
from pathlib import Path

# Import our implementations
try:
    from ..core.universal_darg import UniversalDARG, TextEncoder, ImageEncoder, AudioEncoder
    from ..core.enhanced_darg import EnhancedDARG
except ImportError:
    # Fallback imports for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    try:
        from src.darg.core.universal_darg import UniversalDARG, TextEncoder, ImageEncoder, AudioEncoder
        from src.darg.core.enhanced_darg import EnhancedDARG
    except ImportError:
        # If still failing, create dummy classes
        class UniversalDARG:
            pass
        class EnhancedDARG:
            pass
        class TextEncoder:
            pass
        class ImageEncoder:
            pass
        class AudioEncoder:
            pass
# Import DARG with fallback
try:
    from ...legacy.darg_complete import DARG  # Original implementation
    print("Successfully imported DARG from legacy.darg_complete")
except ImportError:
    try:
        from ...legacy.main import DARG
        print("Successfully imported DARG from legacy.main")
    except ImportError:
        print("Could not import original DARG, using Enhanced DARG only")
        DARG = None

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result structure"""
    method_name: str
    dataset_name: str
    dataset_size: int
    data_type: str
    vector_dimension: int
    
    # Performance metrics
    index_build_time: float
    query_time_per_item: float
    memory_usage_mb: float
    
    # Accuracy metrics
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    
    # Quality metrics
    mean_reciprocal_rank: float
    ndcg_at_10: float
    
    # Scalability metrics
    throughput_qps: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    # Resource metrics
    cpu_utilization: float
    peak_memory_mb: float
    disk_usage_mb: float
    
    # Additional metadata
    timestamp: float
    configuration: Dict[str, Any]
    notes: str = ""

class DatasetGenerator:
    """Advanced dataset generator for comprehensive testing"""
    
    @staticmethod
    def generate_synthetic_text_data(n_docs: int = 1000, 
                                   vocab_size: int = 5000,
                                   doc_length_range: Tuple[int, int] = (10, 100)) -> List[str]:
        """Generate synthetic text data with controlled properties"""
        
        # Create vocabulary
        words = [f"word_{i}" for i in range(vocab_size)]
        
        # Generate documents with semantic clusters
        documents = []
        cluster_size = n_docs // 10  # 10 clusters
        
        for cluster_id in range(10):
            # Each cluster focuses on specific word subsets
            cluster_words = words[cluster_id * 500:(cluster_id + 1) * 500]
            
            for _ in range(cluster_size):
                doc_length = np.random.randint(*doc_length_range)
                # 70% words from cluster, 30% random
                cluster_word_count = int(doc_length * 0.7)
                random_word_count = doc_length - cluster_word_count
                
                doc_words = (
                    np.random.choice(cluster_words, cluster_word_count).tolist() +
                    np.random.choice(words, random_word_count).tolist()
                )
                
                documents.append(" ".join(doc_words))
        
        # Fill remaining with random documents
        remaining = n_docs - len(documents)
        for _ in range(remaining):
            doc_length = np.random.randint(*doc_length_range)
            doc_words = np.random.choice(words, doc_length)
            documents.append(" ".join(doc_words))
        
        np.random.shuffle(documents)
        return documents
    
    @staticmethod
    def generate_synthetic_vectors(n_vectors: int = 1000,
                                 dimensions: int = 128,
                                 n_clusters: int = 10) -> np.ndarray:
        """Generate synthetic high-dimensional vectors with cluster structure"""
        
        vectors = []
        cluster_size = n_vectors // n_clusters
        
        # Generate cluster centers
        cluster_centers = np.random.randn(n_clusters, dimensions)
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        for i in range(n_clusters):
            center = cluster_centers[i]
            
            # Generate vectors around cluster center
            for _ in range(cluster_size):
                # Add controlled noise to cluster center
                noise = np.random.randn(dimensions) * 0.3
                vector = center + noise
                vector = vector / np.linalg.norm(vector)  # Normalize
                vectors.append(vector)
        
        # Fill remaining with random vectors
        remaining = n_vectors - len(vectors)
        for _ in range(remaining):
            vector = np.random.randn(dimensions)
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        
        return np.array(vectors)
    
    @staticmethod
    def create_test_datasets() -> Dict[str, Dict[str, Any]]:
        """Create comprehensive test datasets"""
        
        datasets = {}
        
        # Small datasets for quick testing
        datasets['small_text'] = {
            'data': DatasetGenerator.generate_synthetic_text_data(100, 1000, (5, 20)),
            'type': 'text',
            'size': 100,
            'description': 'Small synthetic text dataset'
        }
        
        datasets['small_vectors'] = {
            'data': DatasetGenerator.generate_synthetic_vectors(100, 64, 5),
            'type': 'vectors',
            'size': 100,
            'description': 'Small synthetic vector dataset'
        }
        
        # Medium datasets for standard benchmarking
        datasets['medium_text'] = {
            'data': DatasetGenerator.generate_synthetic_text_data(1000, 5000, (10, 50)),
            'type': 'text',
            'size': 1000,
            'description': 'Medium synthetic text dataset'
        }
        
        datasets['medium_vectors'] = {
            'data': DatasetGenerator.generate_synthetic_vectors(1000, 256, 10),
            'type': 'vectors',
            'size': 1000,
            'description': 'Medium synthetic vector dataset'
        }
        
        # Large datasets for scalability testing
        datasets['large_vectors'] = {
            'data': DatasetGenerator.generate_synthetic_vectors(5000, 512, 20),
            'type': 'vectors',
            'size': 5000,
            'description': 'Large synthetic vector dataset'
        }
        
        # High-dimensional datasets
        datasets['high_dim_vectors'] = {
            'data': DatasetGenerator.generate_synthetic_vectors(500, 1536, 10),
            'type': 'vectors',
            'size': 500,
            'description': 'High-dimensional synthetic vectors'
        }
        
        return datasets

class BaselineComparator:
    """Implement baseline comparison methods"""
    
    @staticmethod
    def brute_force_search(database: np.ndarray, 
                          queries: np.ndarray, 
                          k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Brute force exact nearest neighbor search"""
        
        start_time = time.time()
        
        # Compute all pairwise distances
        distances = np.cdist(queries, database, metric='cosine')
        
        # Get top-k neighbors
        indices = np.argsort(distances, axis=1)[:, :k]
        distances_sorted = np.sort(distances, axis=1)[:, :k]
        
        search_time = time.time() - start_time
        
        return indices, distances_sorted, search_time
    
    @staticmethod
    def random_search(database: np.ndarray,
                     queries: np.ndarray,
                     k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Random baseline for comparison"""
        
        start_time = time.time()
        
        n_db, n_queries = len(database), len(queries)
        indices = np.random.randint(0, n_db, (n_queries, k))
        
        # Compute actual distances for random selections
        distances = np.zeros((n_queries, k))
        for i in range(n_queries):
            for j in range(k):
                distances[i, j] = np.linalg.norm(
                    queries[i] - database[indices[i, j]]
                )
        
        search_time = time.time() - start_time
        
        return indices, distances, search_time

class PerformanceProfiler:
    """Comprehensive performance profiling"""
    
    def __init__(self):
        self.start_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self._monitoring = False
        self._monitor_thread = None
    
    def start_profiling(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self._monitoring = True
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and return metrics"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        metrics = {
            'total_time': total_time,
            'peak_memory_mb': max(self.memory_samples) if self.memory_samples else 0,
            'avg_memory_mb': np.mean(self.memory_samples) if self.memory_samples else 0,
            'avg_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0
        }
        
        return metrics
    
    def _monitor_resources(self):
        """Monitor system resources"""
        try:
            import psutil
            process = psutil.Process()
            
            while self._monitoring:
                # Memory usage in MB
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                
                # CPU usage percentage
                cpu_percent = process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms
                
        except ImportError:
            # psutil not available, use basic monitoring
            while self._monitoring:
                # Basic memory estimation (not accurate)
                self.memory_samples.append(100.0)  # Placeholder
                self.cpu_samples.append(50.0)      # Placeholder
                time.sleep(0.1)

class AccuracyEvaluator:
    """Comprehensive accuracy evaluation metrics"""
    
    @staticmethod
    def compute_recall_at_k(true_neighbors: np.ndarray,
                           predicted_neighbors: np.ndarray,
                           k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Compute recall@k for various k values"""
        
        recalls = {}
        n_queries = len(true_neighbors)
        
        for k in k_values:
            if k > predicted_neighbors.shape[1]:
                recalls[k] = 0.0
                continue
            
            total_recall = 0.0
            for i in range(n_queries):
                true_k = set(true_neighbors[i, :k])
                pred_k = set(predicted_neighbors[i, :k])
                
                if len(true_k) > 0:
                    recall = len(true_k.intersection(pred_k)) / len(true_k)
                    total_recall += recall
            
            recalls[k] = total_recall / n_queries
        
        return recalls
    
    @staticmethod
    def compute_precision_at_k(true_neighbors: np.ndarray,
                              predicted_neighbors: np.ndarray,
                              k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Compute precision@k for various k values"""
        
        precisions = {}
        n_queries = len(true_neighbors)
        
        for k in k_values:
            if k > predicted_neighbors.shape[1]:
                precisions[k] = 0.0
                continue
            
            total_precision = 0.0
            for i in range(n_queries):
                true_k = set(true_neighbors[i, :min(k, len(true_neighbors[i]))])
                pred_k = set(predicted_neighbors[i, :k])
                
                if len(pred_k) > 0:
                    precision = len(true_k.intersection(pred_k)) / len(pred_k)
                    total_precision += precision
            
            precisions[k] = total_precision / n_queries
        
        return precisions
    
    @staticmethod
    def compute_mrr(true_neighbors: np.ndarray,
                   predicted_neighbors: np.ndarray) -> float:
        """Compute Mean Reciprocal Rank"""
        
        mrr = 0.0
        n_queries = len(true_neighbors)
        
        for i in range(n_queries):
            true_first = true_neighbors[i, 0]  # First true neighbor
            
            # Find rank of first true neighbor in predictions
            pred_list = predicted_neighbors[i].tolist()
            try:
                rank = pred_list.index(true_first) + 1  # 1-based rank
                mrr += 1.0 / rank
            except ValueError:
                # True neighbor not found in predictions
                mrr += 0.0
        
        return mrr / n_queries
    
    @staticmethod
    def compute_ndcg_at_k(true_neighbors: np.ndarray,
                         predicted_neighbors: np.ndarray,
                         true_distances: np.ndarray,
                         k: int = 10) -> float:
        """Compute Normalized Discounted Cumulative Gain at k"""
        
        def dcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
            """Compute DCG@k"""
            k = min(k, len(relevance_scores))
            dcg = relevance_scores[0]
            for i in range(1, k):
                dcg += relevance_scores[i] / np.log2(i + 1)
            return dcg
        
        total_ndcg = 0.0
        n_queries = len(true_neighbors)
        
        for i in range(n_queries):
            # Create relevance scores (inverse of distance)
            max_distance = np.max(true_distances[i])
            true_relevance = max_distance - true_distances[i]
            
            # Get predicted relevance scores
            pred_relevance = []
            for pred_idx in predicted_neighbors[i, :k]:
                if pred_idx < len(true_relevance):
                    pred_relevance.append(true_relevance[pred_idx])
                else:
                    pred_relevance.append(0.0)
            
            pred_relevance = np.array(pred_relevance)
            
            # Compute DCG for predictions and ideal ordering
            dcg = dcg_at_k(pred_relevance, k)
            idcg = dcg_at_k(np.sort(true_relevance)[::-1], k)  # Ideal DCG
            
            if idcg > 0:
                total_ndcg += dcg / idcg
        
        return total_ndcg / n_queries

class ComprehensiveBenchmark:
    """Main benchmarking system"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.datasets = DatasetGenerator.create_test_datasets()
        self.results = []
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def benchmark_method(self,
                        method_name: str,
                        build_func: callable,
                        search_func: callable,
                        dataset_name: str,
                        query_fraction: float = 0.1,
                        k_values: List[int] = [1, 5, 10]) -> BenchmarkResult:
        """Comprehensive benchmark of a single method"""
        
        logger.info(f"Benchmarking {method_name} on {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        
        # Prepare data based on type
        if dataset['type'] == 'text':
            # For text data, we need to vectorize first
            encoder = TextEncoder()
            vectors = np.array([encoder.encode(text) for text in data])
        else:
            vectors = data
        
        # Split into database and queries
        n_total = len(vectors)
        n_queries = int(n_total * query_fraction)
        
        db_vectors = vectors[:-n_queries] if n_queries > 0 else vectors[:-10]
        query_vectors = vectors[-n_queries:] if n_queries > 0 else vectors[-10:]
        
        profiler = PerformanceProfiler()
        
        # Build index
        logger.info(f"Building index for {len(db_vectors)} vectors...")
        profiler.start_profiling()
        
        build_start = time.time()
        index = build_func(db_vectors)
        build_time = time.time() - build_start
        
        build_metrics = profiler.stop_profiling()
        
        # Get ground truth using brute force
        logger.info("Computing ground truth...")
        true_indices, true_distances, _ = BaselineComparator.brute_force_search(
            db_vectors, query_vectors, max(k_values)
        )
        
        # Perform queries
        logger.info(f"Performing {len(query_vectors)} queries...")
        profiler.start_profiling()
        
        query_start = time.time()
        pred_indices, pred_distances = search_func(index, query_vectors, max(k_values))
        total_query_time = time.time() - query_start
        
        query_metrics = profiler.stop_profiling()
        
        # Compute accuracy metrics
        evaluator = AccuracyEvaluator()
        
        recalls = evaluator.compute_recall_at_k(true_indices, pred_indices, k_values)
        precisions = evaluator.compute_precision_at_k(true_indices, pred_indices, k_values)
        mrr = evaluator.compute_mrr(true_indices, pred_indices)
        ndcg = evaluator.compute_ndcg_at_k(true_indices, pred_indices, true_distances, 10)
        
        # Compute performance metrics
        query_time_per_item = total_query_time / len(query_vectors)
        throughput_qps = len(query_vectors) / total_query_time
        
        # Create benchmark result
        result = BenchmarkResult(
            method_name=method_name,
            dataset_name=dataset_name,
            dataset_size=len(db_vectors),
            data_type=dataset['type'],
            vector_dimension=vectors.shape[1] if len(vectors.shape) > 1 else 1,
            
            # Performance metrics
            index_build_time=build_time,
            query_time_per_item=query_time_per_item,
            memory_usage_mb=query_metrics['avg_memory_mb'],
            
            # Accuracy metrics
            recall_at_1=recalls.get(1, 0.0),
            recall_at_5=recalls.get(5, 0.0),
            recall_at_10=recalls.get(10, 0.0),
            precision_at_1=precisions.get(1, 0.0),
            precision_at_5=precisions.get(5, 0.0),
            precision_at_10=precisions.get(10, 0.0),
            
            # Quality metrics
            mean_reciprocal_rank=mrr,
            ndcg_at_10=ndcg,
            
            # Scalability metrics
            throughput_qps=throughput_qps,
            latency_p95_ms=query_time_per_item * 1000,  # Approximation
            latency_p99_ms=query_time_per_item * 1000,  # Approximation
            
            # Resource metrics
            cpu_utilization=query_metrics['avg_cpu_percent'],
            peak_memory_mb=max(build_metrics['peak_memory_mb'], query_metrics['peak_memory_mb']),
            disk_usage_mb=0.0,  # Placeholder
            
            # Metadata
            timestamp=time.time(),
            configuration={
                'query_fraction': query_fraction,
                'k_values': k_values,
                'dataset_description': dataset['description']
            }
        )
        
        self.results.append(result)
        logger.info(f"Completed benchmark: {method_name} on {dataset_name}")
        
        return result
    
    def run_universal_darg_benchmark(self, dataset_name: str) -> BenchmarkResult:
        """Benchmark Universal DARG"""
        
        def build_func(vectors):
            darg = UniversalDARG()
            
            # Add vectors to DARG
            for i, vector in enumerate(vectors):
                # Convert to text format for universal handling
                data_item = f"vector_{i}"
                metadata = {"vector": vector.tolist()}
                darg.add_data(data_item, "text", metadata)
            
            return darg
        
        def search_func(darg, queries, k):
            results = []
            for query in queries:
                query_text = f"query_vector"
                query_metadata = {"vector": query.tolist()}
                
                # Perform search
                search_results = darg.search(query_text, "text", k, query_metadata)
                
                # Extract indices
                indices = [int(result['data_id'].split('_')[1]) for result in search_results]
                distances = [1.0 - result['similarity'] for result in search_results]
                
                results.append((indices, distances))
            
            # Convert to numpy arrays
            indices_array = np.array([r[0] for r in results])
            distances_array = np.array([r[1] for r in results])
            
            return indices_array, distances_array
        
        return self.benchmark_method("Universal DARG", build_func, search_func, dataset_name)
    
    def run_enhanced_darg_benchmark(self, dataset_name: str) -> BenchmarkResult:
        """Benchmark Enhanced DARG"""
        
        def build_func(vectors):
            darg = EnhancedDARG(vector_dim=vectors.shape[1])
            
            # Add vectors to DARG
            for i, vector in enumerate(vectors):
                darg.add_vector(vector, f"data_{i}")
            
            return darg
        
        def search_func(darg, queries, k):
            indices_list = []
            distances_list = []
            
            for query in queries:
                results = darg.search(query, k)
                
                indices = [r['node_id'] for r in results]
                # Convert node_id back to integer index
                indices = [int(idx.split('_')[1]) if '_' in idx else int(idx) for idx in indices]
                
                distances = [r['distance'] for r in results]
                
                indices_list.append(indices)
                distances_list.append(distances)
            
            return np.array(indices_list), np.array(distances_list)
        
        return self.benchmark_method("Enhanced DARG", build_func, search_func, dataset_name)
    
    def run_baseline_benchmarks(self, dataset_name: str) -> List[BenchmarkResult]:
        """Run baseline benchmarks"""
        
        results = []
        
        # Brute Force baseline
        def build_func_bf(vectors):
            return vectors  # No index needed
        
        def search_func_bf(database, queries, k):
            indices, distances, _ = BaselineComparator.brute_force_search(database, queries, k)
            return indices, distances
        
        results.append(
            self.benchmark_method("Brute Force", build_func_bf, search_func_bf, dataset_name)
        )
        
        # Random baseline
        def search_func_random(database, queries, k):
            indices, distances, _ = BaselineComparator.random_search(database, queries, k)
            return indices, distances
        
        results.append(
            self.benchmark_method("Random Search", build_func_bf, search_func_random, dataset_name)
        )
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across all datasets and methods"""
        
        logger.info("Starting comprehensive benchmark suite")
        
        for dataset_name in self.datasets.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Benchmarking dataset: {dataset_name}")
            logger.info(f"{'='*50}")
            
            try:
                # Run Universal DARG
                self.run_universal_darg_benchmark(dataset_name)
                
                # Run Enhanced DARG
                self.run_enhanced_darg_benchmark(dataset_name)
                
                # Run baselines
                self.run_baseline_benchmarks(dataset_name)
                
            except Exception as e:
                logger.error(f"Error benchmarking {dataset_name}: {e}")
                continue
        
        logger.info("Comprehensive benchmark completed")
    
    def save_results(self, filename: str = None):
        """Save benchmark results"""
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert results to serializable format
        serializable_results = [asdict(result) for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report"""
        
        if not self.results:
            return "No benchmark results available"
        
        report = []
        report.append("UNIVERSAL DARG BENCHMARK REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total benchmarks: {len(self.results)}\n")
        
        # Group results by dataset
        datasets = {}
        for result in self.results:
            if result.dataset_name not in datasets:
                datasets[result.dataset_name] = []
            datasets[result.dataset_name].append(result)
        
        # Generate report for each dataset
        for dataset_name, dataset_results in datasets.items():
            report.append(f"Dataset: {dataset_name}")
            report.append("-" * 30)
            
            # Create comparison table
            methods = [r.method_name for r in dataset_results]
            
            report.append(f"{'Method':<20} {'Build(s)':<10} {'Query(ms)':<12} {'R@1':<8} {'R@10':<8} {'P@1':<8} {'MRR':<8}")
            report.append("-" * 80)
            
            for result in dataset_results:
                report.append(
                    f"{result.method_name:<20} "
                    f"{result.index_build_time:<10.3f} "
                    f"{result.query_time_per_item*1000:<12.3f} "
                    f"{result.recall_at_1:<8.3f} "
                    f"{result.recall_at_10:<8.3f} "
                    f"{result.precision_at_1:<8.3f} "
                    f"{result.mean_reciprocal_rank:<8.3f}"
                )
            
            # Find best performing method
            best_accuracy = max(dataset_results, key=lambda x: x.recall_at_10)
            best_speed = min(dataset_results, key=lambda x: x.query_time_per_item)
            
            report.append(f"\nBest Accuracy: {best_accuracy.method_name} (R@10: {best_accuracy.recall_at_10:.3f})")
            report.append(f"Best Speed: {best_speed.method_name} ({best_speed.query_time_per_item*1000:.3f}ms per query)")
            report.append("")
        
        # Overall summary
        report.append("OVERALL SUMMARY")
        report.append("=" * 20)
        
        universal_results = [r for r in self.results if "Universal" in r.method_name]
        enhanced_results = [r for r in self.results if "Enhanced" in r.method_name]
        baseline_results = [r for r in self.results if r.method_name in ["Brute Force", "Random Search"]]
        
        if universal_results:
            avg_recall = np.mean([r.recall_at_10 for r in universal_results])
            avg_speed = np.mean([r.query_time_per_item for r in universal_results])
            report.append(f"Universal DARG - Avg R@10: {avg_recall:.3f}, Avg Query Time: {avg_speed*1000:.3f}ms")
        
        if enhanced_results:
            avg_recall = np.mean([r.recall_at_10 for r in enhanced_results])
            avg_speed = np.mean([r.query_time_per_item for r in enhanced_results])
            report.append(f"Enhanced DARG - Avg R@10: {avg_recall:.3f}, Avg Query Time: {avg_speed*1000:.3f}ms")
        
        return "\n".join(report)

def run_complete_validation():
    """Run complete validation and testing suite"""
    
    print("🚀 Universal DARG Complete Validation Suite")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = ComprehensiveBenchmark()
    
    # Run comprehensive benchmarks
    benchmark.run_comprehensive_benchmark()
    
    # Save results
    results_file = benchmark.save_results()
    
    # Generate and display report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save report
    report_file = benchmark.results_dir / "benchmark_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Validation completed!")
    print(f"📊 Results saved to: {results_file}")
    print(f"📋 Report saved to: {report_file}")

if __name__ == "__main__":
    run_complete_validation()
