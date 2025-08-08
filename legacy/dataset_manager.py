#!/usr/bin/env python3
"""
DARG Dataset Manager
Downloads and manages benchmark datasets for DARG evaluation
"""

import os
import requests
import numpy as np
import h5py
import logging
import zipfile
import tarfile
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Dataset information"""
    name: str
    url: str
    filename: str
    vectors_count: int
    dimensions: int
    queries_count: int
    description: str
    file_format: str  # 'hdf5', 'fvecs', 'txt'
    compressed: bool = False

class DatasetManager:
    """Manages benchmark datasets for DARG evaluation"""
    
    def __init__(self, base_dir: str = "datasets", downloads_dir: str = "downloads", 
                 cache_dir: str = "cache"):
        self.base_dir = Path(base_dir)
        self.downloads_dir = Path(downloads_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories if they don't exist
        self.base_dir.mkdir(exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Dataset configurations with proper URLs and larger datasets
        self.datasets = {
            'sift1m': {
                'name': 'SIFT1M',
                'description': '1M SIFT descriptors - standard ANN benchmark',
                'url': 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz',
                'vectors': 1_000_000,
                'dimensions': 128,
                'queries': 10_000,
                'size_gb': 0.5,
                'format': 'fvecs',
                'type': 'real'
            },
            'sift1b': {
                'name': 'SIFT1B', 
                'description': '1B SIFT descriptors - billion-scale benchmark',
                'url': 'ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs',
                'vectors': 1_000_000_000,
                'dimensions': 128,
                'queries': 10_000,
                'size_gb': 476.8,
                'format': 'bvecs',
                'type': 'real'
            },
            'glove_1.2m': {
                'name': 'GloVe-1.2M',
                'description': 'GloVe word embeddings - text similarity benchmark',
                'url': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
                'vectors': 1_200_000,
                'dimensions': 300,
                'queries': 1_000,
                'size_gb': 1.3,
                'format': 'txt',
                'type': 'real'
            },
            'deep1m': {
                'name': 'Deep1M',
                'description': '1M deep learning features',
                'url': 'https://storage.googleapis.com/ann-datasets/deep1M.hdf5',
                'vectors': 1_000_000,
                'dimensions': 96,
                'queries': 10_000,
                'size_gb': 0.4,
                'format': 'hdf5',
                'type': 'real'
            },
            'gist1m': {
                'name': 'GIST1M',
                'description': '1M GIST descriptors',
                'url': 'ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz',
                'vectors': 1_000_000,
                'dimensions': 960,
                'queries': 1_000,
                'size_gb': 3.6,
                'format': 'fvecs',
                'type': 'real'
            },
            'fashion_mnist': {
                'name': 'Fashion-MNIST',
                'description': 'Fashion-MNIST dataset for similarity search',
                'url': 'https://storage.googleapis.com/ann-datasets/fashion-mnist-784-euclidean.hdf5',
                'vectors': 60_000,
                'dimensions': 784,
                'queries': 10_000,
                'size_gb': 0.4,
                'format': 'hdf5',
                'type': 'real'
            },
            'synthetic_small': {
                'name': 'Synthetic-Small',
                'description': 'Small synthetic dataset for testing',
                'vectors': 10_000,
                'dimensions': 128,
                'queries': 1_000,
                'size_gb': 0.005,
                'format': 'npz',
                'type': 'synthetic'
            },
            'synthetic_medium': {
                'name': 'Synthetic-Medium',
                'description': 'Medium synthetic dataset for development',
                'vectors': 100_000,
                'dimensions': 128,
                'queries': 10_000,
                'size_gb': 0.05,
                'format': 'npz',
                'type': 'synthetic'
            },
            'synthetic_large': {
                'name': 'Synthetic-Large',
                'description': 'Large synthetic dataset for benchmarking',
                'vectors': 1_000_000,
                'dimensions': 128,
                'queries': 50_000,
                'size_gb': 0.5,
                'format': 'npz',
                'type': 'synthetic'
            },
            'synthetic_huge': {
                'name': 'Synthetic-Huge',
                'description': 'Huge synthetic dataset for stress testing',
                'vectors': 10_000_000,
                'dimensions': 128,
                'queries': 100_000,
                'size_gb': 5.0,
                'format': 'npz',
                'type': 'synthetic'
            }
        }
    
    def list_datasets(self) -> None:
        """List available datasets"""
        print("Available DARG Benchmark Datasets:")
        print("=" * 60)
        for key, dataset in self.datasets.items():
            status = "✓ Downloaded" if self.is_downloaded(key) else "○ Available"
            size_gb = self._estimate_size_gb(dataset.vectors_count, dataset.dimensions)
            print(f"{status} {dataset.name}")
            print(f"    {dataset.description}")
            print(f"    Vectors: {dataset.vectors_count:,} | Dims: {dataset.dimensions} | Queries: {dataset.queries_count:,}")
            print(f"    Size: ~{size_gb:.1f} GB | Format: {dataset.file_format}")
            print()
    
    def _estimate_size_gb(self, vectors: int, dims: int) -> float:
        """Estimate dataset size in GB"""
        # Assume 4 bytes per float32
        bytes_size = vectors * dims * 4
        return bytes_size / (1024**3)
    
    def is_downloaded(self, dataset_key: str) -> bool:
        """Check if dataset is downloaded"""
        if dataset_key not in self.datasets:
            return False
        
        dataset = self.datasets[dataset_key]
        
        # Check for processed numpy files
        base_name = dataset.filename.split('.')[0]
        vectors_file = self.data_dir / f"{base_name}_vectors.npy"
        queries_file = self.data_dir / f"{base_name}_queries.npy"
        
        return vectors_file.exists() and queries_file.exists()
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> bool:
        """Download and process a dataset with new structure"""
        if dataset_key not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        dataset_info = self.datasets[dataset_key]
        
        # Check if already downloaded (for synthetic datasets)
        if dataset_info['type'] == 'synthetic':
            dataset_file = self.base_dir / f"{dataset_key}.npz"
            if dataset_file.exists() and not force:
                logger.info(f"Dataset {dataset_info['name']} already exists")
                return True
        
        logger.info(f"Downloading dataset: {dataset_info['name']}")
        print(f"🌐 Downloading {dataset_info['name']} ({dataset_info['description']})")
        
        # Handle synthetic datasets
        if dataset_info['type'] == 'synthetic':
            return self._generate_synthetic_dataset(dataset_key)
        
        # For real datasets, show a message that they're not implemented yet
        print(f"⚠️  Real dataset download not yet implemented for {dataset_key}")
        print(f"   URL: {dataset_info['url']}")
        print(f"   Expected size: {dataset_info['size_gb']:.1f}GB")
        return False
    
    def _download_file(self, url: str, file_path: Path) -> bool:
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=file_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            
            logger.info(f"Downloaded {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if file_path.exists():
                file_path.unlink()
            return False
    
    def _process_dataset(self, dataset_key: str, file_path: Path) -> bool:
        """Process downloaded dataset"""
        dataset = self.datasets[dataset_key]
        base_name = dataset.filename.split('.')[0]
        
        try:
            if dataset.file_format == 'hdf5':
                return self._process_hdf5(file_path, base_name)
            elif dataset.file_format in ['fvecs', 'bvecs']:
                return self._process_vecs(file_path, base_name, dataset.file_format)
            elif dataset.file_format == 'txt':
                return self._process_text(file_path, base_name)
            else:
                logger.error(f"Unsupported format: {dataset.file_format}")
                return False
                
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False
    
    def _process_hdf5(self, file_path: Path, base_name: str) -> bool:
        """Process HDF5 dataset"""
        with h5py.File(file_path, 'r') as f:
            vectors = f['train'][:]
            queries = f['test'][:]
            
        vectors_file = self.data_dir / f"{base_name}_vectors.npy"
        queries_file = self.data_dir / f"{base_name}_queries.npy"
        
        np.save(vectors_file, vectors)
        np.save(queries_file, queries)
        
        logger.info(f"Processed HDF5: {vectors.shape[0]:,} vectors, {queries.shape[0]:,} queries")
        return True
    
    def _process_vecs(self, file_path: Path, base_name: str, format_type: str) -> bool:
        """Process .fvecs or .bvecs files"""
        dtype = np.float32 if format_type == 'fvecs' else np.uint8
        
        # Read vectors
        vectors = self._read_vecs_file(file_path, dtype)
        
        # Generate or find queries
        query_file = file_path.parent / f"{base_name}_query.{format_type}"
        if query_file.exists():
            queries = self._read_vecs_file(query_file, dtype)
        else:
            # Generate random queries from vectors
            n_queries = min(10000, len(vectors) // 100)
            query_indices = np.random.choice(len(vectors), n_queries, replace=False)
            queries = vectors[query_indices]
        
        vectors_file = self.data_dir / f"{base_name}_vectors.npy"
        queries_file = self.data_dir / f"{base_name}_queries.npy"
        
        np.save(vectors_file, vectors)
        np.save(queries_file, queries)
        
        logger.info(f"Processed {format_type}: {vectors.shape[0]:,} vectors, {queries.shape[0]:,} queries")
        return True
    
    def _read_vecs_file(self, file_path: Path, dtype: np.dtype) -> np.ndarray:
        """Read .fvecs or .bvecs file"""
        vectors = []
        
        with open(file_path, 'rb') as f:
            while True:
                # Read dimension
                dim_bytes = f.read(4)
                if len(dim_bytes) < 4:
                    break
                
                dim = int.from_bytes(dim_bytes, byteorder='little')
                
                # Read vector
                vector_bytes = f.read(dim * dtype().itemsize)
                if len(vector_bytes) < dim * dtype().itemsize:
                    break
                
                vector = np.frombuffer(vector_bytes, dtype=dtype)
                vectors.append(vector)
        
        return np.array(vectors, dtype=np.float32)
    
    def _process_text(self, file_path: Path, base_name: str) -> bool:
        """Process text-based embeddings (like GloVe)"""
        vectors = []
        
        # Handle zip files
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # Find the main embeddings file
                embed_file = None
                for name in zip_file.namelist():
                    if name.endswith('.txt') and '300d' in name:
                        embed_file = name
                        break
                
                if not embed_file:
                    logger.error("Could not find embeddings file in zip")
                    return False
                
                with zip_file.open(embed_file) as f:
                    for line_num, line in enumerate(f):
                        if line_num >= 1_200_000:  # Limit for GloVe-1.2M
                            break
                        
                        parts = line.decode().strip().split()
                        if len(parts) > 300:  # word + 300 dimensions
                            vector = np.array([float(x) for x in parts[1:301]], dtype=np.float32)
                            vectors.append(vector)
        else:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num >= 1_200_000:
                        break
                    
                    parts = line.strip().split()
                    if len(parts) > 300:
                        vector = np.array([float(x) for x in parts[1:301]], dtype=np.float32)
                        vectors.append(vector)
        
        vectors = np.array(vectors)
        
        # Generate queries (random subset)
        n_queries = min(1000, len(vectors) // 100)
        query_indices = np.random.choice(len(vectors), n_queries, replace=False)
        queries = vectors[query_indices]
        
        vectors_file = self.data_dir / f"{base_name}_vectors.npy"
        queries_file = self.data_dir / f"{base_name}_queries.npy"
        
        np.save(vectors_file, vectors)
        np.save(queries_file, queries)
        
        logger.info(f"Processed text: {vectors.shape[0]:,} vectors, {queries.shape[0]:,} queries")
        return True
    
    def _generate_synthetic_dataset(self, dataset_key: str) -> bool:
        """Generate synthetic dataset with new structure"""
        dataset_info = self.datasets[dataset_key]
        
        logger.info(f"Generating synthetic dataset: {dataset_info['name']}")
        
        np.random.seed(42)  # Reproducible datasets
        
        # Generate clustered data for more realistic testing
        n_clusters = min(100, dataset_info['vectors'] // 1000)
        
        # Create cluster centers
        centers = np.random.randn(n_clusters, dataset_info['dimensions']) * 10
        
        # Generate vectors around clusters
        vectors = []
        cluster_assignments = np.random.randint(0, n_clusters, dataset_info['vectors'])
        
        for i in range(dataset_info['vectors']):
            cluster_id = cluster_assignments[i]
            center = centers[cluster_id]
            noise = np.random.randn(dataset_info['dimensions']) * 0.5
            vector = center + noise
            vectors.append(vector)
        
        vectors = np.array(vectors, dtype=np.float32)
        
        # Generate queries (some from dataset, some random)
        n_from_dataset = dataset_info['queries'] // 2
        n_random = dataset_info['queries'] - n_from_dataset
        
        dataset_queries = vectors[np.random.choice(len(vectors), n_from_dataset, replace=False)]
        random_queries = np.random.randn(n_random, dataset_info['dimensions']).astype(np.float32)
        
        queries = np.vstack([dataset_queries, random_queries])
        
        # Save as npz file in datasets directory
        dataset_file = self.base_dir / f"{dataset_key}.npz"
        
        np.savez_compressed(dataset_file, vectors=vectors, queries=queries)
        
        logger.info(f"Generated synthetic dataset: {vectors.shape[0]:,} vectors, {queries.shape[0]:,} queries")
        return True
    
    def load_dataset(self, dataset_key: str, vectors_only: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load a processed dataset with new structure"""
        dataset_file = self.base_dir / f"{dataset_key}.npz"
        
        if not dataset_file.exists():
            logger.error(f"Dataset {dataset_key} not found. Try downloading it first.")
            return None, None
        
        try:
            data = np.load(dataset_file)
            vectors = data['vectors']
            queries = None if vectors_only else data['queries']
            
            dataset_info = self.datasets[dataset_key]
            logger.info(f"Loaded {dataset_info['name']}: {vectors.shape[0]:,} vectors")
            if queries is not None:
                logger.info(f"Loaded {queries.shape[0]:,} queries")
            
            return vectors, queries
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_key}: {e}")
            return None, None
    
    def get_dataset_info(self, dataset_key: str) -> Optional[DatasetInfo]:
        """Get dataset information"""
        return self.datasets.get(dataset_key)
    
    def cleanup_downloads(self) -> None:
        """Clean up downloaded archive files (keep processed .npy files)"""
        for dataset in self.datasets.values():
            if dataset.compressed or dataset.filename.endswith(('.zip', '.tar.gz', '.bvecs')):
                file_path = self.data_dir / dataset.filename
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Cleaned up {file_path}")

# Convenience functions
def list_datasets():
    """List available datasets"""
    manager = DatasetManager()
    manager.list_datasets()

def download_dataset(dataset_key: str, force: bool = False) -> bool:
    """Download a dataset"""
    manager = DatasetManager()
    return manager.download_dataset(dataset_key, force)

def load_dataset(dataset_key: str, vectors_only: bool = False):
    """Load a dataset"""
    manager = DatasetManager()
    return manager.load_dataset(dataset_key, vectors_only)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_manager.py <command> [args]")
        print("Commands:")
        print("  list                    - List available datasets")
        print("  download <dataset>      - Download a dataset")
        print("  info <dataset>          - Show dataset information")
        sys.exit(1)
    
    command = sys.argv[1]
    manager = DatasetManager()
    
    if command == 'list':
        manager.list_datasets()
    elif command == 'download' and len(sys.argv) >= 3:
        dataset_key = sys.argv[2]
        force = '--force' in sys.argv
        success = manager.download_dataset(dataset_key, force)
        if success:
            print(f"✅ Successfully downloaded {dataset_key}")
        else:
            print(f"❌ Failed to download {dataset_key}")
    elif command == 'info' and len(sys.argv) >= 3:
        dataset_key = sys.argv[2]
        info = manager.get_dataset_info(dataset_key)
        if info:
            print(f"Dataset: {info.name}")
            print(f"Description: {info.description}")
            print(f"Vectors: {info.vectors_count:,}")
            print(f"Dimensions: {info.dimensions}")
            print(f"Queries: {info.queries_count:,}")
            print(f"Format: {info.file_format}")
            print(f"Downloaded: {'Yes' if manager.is_downloaded(dataset_key) else 'No'}")
        else:
            print(f"Unknown dataset: {dataset_key}")
    else:
        print("Invalid command or missing arguments")
