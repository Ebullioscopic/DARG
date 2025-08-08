#!/usr/bin/env python3
"""
DARG Complete Implementation
Main entry point for the enhanced DARG system with all components
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all DARG components
from main import DARGv22, GlobalConfig
from platform_detection import print_platform_info, setup_acceleration
from dataset_manager import DatasetManager, list_datasets
from inference import create_inference_engine, train_and_create_engine
from visualization import create_visualizer
from test_suite import run_test_suite, run_specific_test
from cpp_acceleration import create_cpp_acceleration_library, get_cpp_accelerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_darg_cli():
    """Create command-line interface for DARG"""
    parser = argparse.ArgumentParser(
        description="DARG: Distributed Adaptive Routing Graph for High-Dimensional ANN Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show platform information
  python darg_complete.py platform
  
  # List available datasets
  python darg_complete.py datasets list
  
  # Download a dataset
  python darg_complete.py datasets download synthetic_small
  
  # Train DARG on a dataset
  python darg_complete.py train synthetic_small --model-path model.pkl
  
  # Run inference benchmark
  python darg_complete.py benchmark model.pkl --dataset synthetic_small
  
  # Run tests
  python darg_complete.py test
  
  # Create C++ acceleration library
  python darg_complete.py cpp-setup
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Platform command
    platform_parser = subparsers.add_parser('platform', help='Show platform information')
    
    # Dataset commands
    dataset_parser = subparsers.add_parser('datasets', help='Dataset management')
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_action')
    
    dataset_list_parser = dataset_subparsers.add_parser('list', help='List available datasets')
    
    dataset_download_parser = dataset_subparsers.add_parser('download', help='Download dataset')
    dataset_download_parser.add_argument('dataset_name', help='Name of dataset to download')
    dataset_download_parser.add_argument('--force', action='store_true', help='Force re-download')
    
    dataset_download_all_parser = dataset_subparsers.add_parser('download-all', help='Download all datasets')
    dataset_download_all_parser.add_argument('--include-large', action='store_true', 
                                           help='Include large datasets (>1GB)')
    dataset_download_all_parser.add_argument('--include-huge', action='store_true',
                                           help='Include huge datasets (>10GB)')
    dataset_download_all_parser.add_argument('--force', action='store_true', help='Force re-download')
    
    dataset_clean_parser = dataset_subparsers.add_parser('clean', help='Clean downloaded datasets')
    dataset_clean_parser.add_argument('--all', action='store_true', help='Clean all datasets')
    dataset_clean_parser.add_argument('--models', action='store_true', help='Clean model files')
    dataset_clean_parser.add_argument('--cache', action='store_true', help='Clean cache files')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='Model management')
    models_subparsers = models_parser.add_subparsers(dest='models_action')
    
    models_list_parser = models_subparsers.add_parser('list', help='List available models')
    models_info_parser = models_subparsers.add_parser('info', help='Show model information')
    models_info_parser.add_argument('model_name', help='Name of model to inspect')
    models_clean_parser = models_subparsers.add_parser('clean', help='Clean old models')
    models_clean_parser.add_argument('--older-than', type=int, default=7, help='Remove models older than N days')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train DARG model')
    train_parser.add_argument('dataset_name', help='Dataset to train on')
    train_parser.add_argument('--model-path', required=True, help='Path to save trained model')
    train_parser.add_argument('--config', help='Configuration file path')
    train_parser.add_argument('--subset-size', type=int, help='Use subset of dataset for training')
    
    # Benchmark command  
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark')
    benchmark_parser.add_argument('model_path', help='Path to trained model')
    benchmark_parser.add_argument('--dataset', help='Dataset for benchmark queries')
    benchmark_parser.add_argument('--num-queries', type=int, default=1000, help='Number of queries')
    benchmark_parser.add_argument('--k-values', nargs='+', type=int, default=[1, 5, 10, 20], help='K values to test')
    benchmark_parser.add_argument('--output', help='Output file for results')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Perform search')
    search_parser.add_argument('model_path', help='Path to trained model')
    search_parser.add_argument('--query-file', help='File containing query vectors (.npy)')
    search_parser.add_argument('--k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--output', help='Output file for results')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--test-type', choices=['all', 'core', 'platform', 'dataset', 'inference', 'performance', 'integration'], 
                           default='all', help='Type of test to run')
    
    # C++ setup command
    cpp_parser = subparsers.add_parser('cpp-setup', help='Setup C++ acceleration')
    cpp_parser.add_argument('--output-dir', default='./cpp', help='Output directory for C++ code')
    
    # Performance analysis command
    perf_parser = subparsers.add_parser('performance', help='Performance analysis')
    perf_parser.add_argument('--num-vectors', type=int, default=10000, help='Number of vectors')
    perf_parser.add_argument('--num-queries', type=int, default=1000, help='Number of queries')
    perf_parser.add_argument('--dimensions', type=int, default=128, help='Vector dimensions')
    
    return parser

def handle_platform_command(args):
    """Handle platform command"""
    print("🖥️  DARG Platform Information")
    print("=" * 50)
    print_platform_info()
    
    # Show acceleration info
    accel_config = setup_acceleration()
    print(f"\nAcceleration Configuration:")
    for key, value in accel_config.items():
        print(f"  {key}: {value}")
    
    # Show C++ acceleration status
    cpp_accelerator = get_cpp_accelerator()
    gpu_info = cpp_accelerator.get_gpu_info()
    print(f"\nC++ Acceleration:")
    print(f"  Library Available: {cpp_accelerator.is_available()}")
    print(f"  GPU Available: {gpu_info['available']}")
    print(f"  GPU Type: {gpu_info['type']}")

def handle_dataset_command(args):
    """Handle dataset commands"""
    dataset_manager = DatasetManager()
    
    if args.dataset_action == 'list':
        dataset_manager.list_datasets()
    
    elif args.dataset_action == 'download':
        print(f"📥 Downloading dataset: {args.dataset_name}")
        success = dataset_manager.download_dataset(args.dataset_name, force=args.force)
        if success:
            print(f"✅ Successfully downloaded {args.dataset_name}")
        else:
            print(f"❌ Failed to download {args.dataset_name}")
            return 1
    
    elif args.dataset_action == 'download-all':
        print("📥 Downloading all datasets...")
        
        # Get list of datasets to download
        all_datasets = list(dataset_manager.datasets.keys())
        to_download = []
        
        for dataset_name in all_datasets:
            dataset_info = dataset_manager.datasets[dataset_name]
            size_gb = dataset_info['size_gb']
            
            # Filter by size
            if size_gb > 10 and not args.include_huge:
                print(f"⏭️  Skipping {dataset_name} (huge dataset {size_gb:.1f}GB, use --include-huge)")
                continue
            elif size_gb > 1 and not args.include_large:
                print(f"⏭️  Skipping {dataset_name} (large dataset {size_gb:.1f}GB, use --include-large)")
                continue
            
            to_download.append(dataset_name)
        
        print(f"📋 Will download {len(to_download)} datasets: {', '.join(to_download)}")
        
        # Estimate total download size
        total_size = sum(dataset_manager.datasets[d]['size_gb'] for d in to_download)
        print(f"💾 Total download size: {total_size:.1f}GB")
        
        # Download each dataset
        successful = 0
        for dataset_name in to_download:
            print(f"\n📥 Downloading {dataset_name}...")
            success = dataset_manager.download_dataset(dataset_name, force=args.force)
            if success:
                successful += 1
                print(f"✅ {dataset_name} downloaded successfully")
            else:
                print(f"❌ Failed to download {dataset_name}")
        
        print(f"\n🎯 Downloaded {successful}/{len(to_download)} datasets successfully")
        return 0 if successful == len(to_download) else 1
    
    elif args.dataset_action == 'clean':
        print("🧹 Cleaning datasets and files...")
        
        if args.all or args.models:
            models_dir = Path("models")
            if models_dir.exists():
                import shutil
                shutil.rmtree(models_dir)
                models_dir.mkdir()
                print("🗑️  Cleaned models directory")
        
        if args.all or args.cache:
            for cache_dir in ["cache", "downloads", ".cache"]:
                cache_path = Path(cache_dir)
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path)
                    cache_path.mkdir()
                    print(f"🗑️  Cleaned {cache_dir} directory")
        
        if args.all:
            datasets_dir = Path("datasets")
            if datasets_dir.exists():
                import shutil
                shutil.rmtree(datasets_dir)
                datasets_dir.mkdir()
                print("🗑️  Cleaned datasets directory")
        
        print("✅ Cleanup completed")
    
    return 0

def handle_train_command(args):
    """Handle training command"""
    print(f"🏋️  Training DARG on dataset: {args.dataset_name}")
    
    try:
        # Organize model path in models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Create model path with timestamp for uniqueness
        if not args.model_path.startswith("models/"):
            model_filename = Path(args.model_path).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = models_dir / f"{model_filename}_{args.dataset_name}_{timestamp}"
        else:
            model_path = Path(args.model_path)
        
        print(f"📁 Model will be saved to: {model_path}")
        
        # Load configuration if provided
        config = None
        if args.config:
            config = GlobalConfig.from_file(args.config)
        
        # Create inference engine from dataset
        start_time = time.time()
        engine = train_and_create_engine(
            args.dataset_name, 
            str(model_path),
            batch_size=config.pca_batch_size if config else 1000
        )
        
        training_time = time.time() - start_time
        
        # Get model statistics
        stats = engine.get_statistics()
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Model Statistics:")
        print(f"   Total points: {stats['model_stats']['total_points']:,}")
        print(f"   Total cells: {stats['model_stats']['total_cells']:,}")
        print(f"   Max depth: {stats['model_stats']['max_depth']}")
        print(f"   Model saved to: {model_path}")
        
        # Show model files created
        model_files = list(model_path.parent.glob(f"{model_path.name}*"))
        print(f"📄 Model files created:")
        for file in model_files:
            size_mb = file.stat().st_size / (1024*1024)
            print(f"   {file.name} ({size_mb:.1f}MB)")
        
        engine.shutdown()
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training error: {e}")
        return 1

def handle_models_command(args):
    """Handle models management commands"""
    models_dir = Path("models")
    
    if args.models_action == 'list':
        print("🗃️  Available DARG Models")
        print("=" * 50)
        
        if not models_dir.exists():
            print("No models directory found")
            return 0
        
        # Find all model manifest files
        manifest_files = list(models_dir.glob("*.manifest"))
        
        if not manifest_files:
            print("No trained models found")
            return 0
        
        for manifest_file in sorted(manifest_files):
            model_name = manifest_file.stem
            
            # Get model size
            model_files = list(models_dir.glob(f"{model_name}*"))
            total_size = sum(f.stat().st_size for f in model_files) / (1024*1024)
            
            # Get creation time
            created = manifest_file.stat().st_mtime
            created_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(created))
            
            print(f"📦 {model_name}")
            print(f"   Created: {created_str}")
            print(f"   Size: {total_size:.1f}MB")
            print(f"   Files: {len(model_files)}")
            print()
    
    elif args.models_action == 'info':
        model_path = models_dir / args.model_name
        if not (models_dir / f"{args.model_name}.manifest").exists():
            print(f"❌ Model {args.model_name} not found")
            return 1
        
        print(f"📊 Model Information: {args.model_name}")
        print("=" * 50)
        
        try:
            # Load model to get statistics
            engine = create_inference_engine(str(model_path))
            stats = engine.get_statistics()
            
            print(f"Model Statistics:")
            for key, value in stats['model_stats'].items():
                print(f"   {key}: {value:,}" if isinstance(value, int) else f"   {key}: {value}")
            
            print(f"\nSearch Statistics:")
            for key, value in stats['search_stats'].items():
                print(f"   {key}: {value}")
            
            engine.shutdown()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return 1
    
    elif args.models_action == 'clean':
        print(f"🧹 Cleaning models older than {args.older_than} days...")
        
        if not models_dir.exists():
            print("No models directory found")
            return 0
        
        cutoff_time = time.time() - (args.older_than * 24 * 3600)
        removed_count = 0
        
        for manifest_file in models_dir.glob("*.manifest"):
            if manifest_file.stat().st_mtime < cutoff_time:
                model_name = manifest_file.stem
                model_files = list(models_dir.glob(f"{model_name}*"))
                
                for file in model_files:
                    file.unlink()
                    removed_count += 1
                
                print(f"🗑️  Removed {model_name} ({len(model_files)} files)")
        
        print(f"✅ Cleaned {removed_count} files")
    
    return 0
    """Handle training command"""
    print(f"🏋️  Training DARG on dataset: {args.dataset_name}")
    
    try:
        # Organize model path in models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Create model path with timestamp for uniqueness
        if not args.model_path.startswith("models/"):
            model_filename = Path(args.model_path).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = models_dir / f"{model_filename}_{args.dataset_name}_{timestamp}"
        else:
            model_path = Path(args.model_path)
        
        print(f"📁 Model will be saved to: {model_path}")
        
        # Load configuration if provided
        config = None
        if args.config:
            config = GlobalConfig.from_file(args.config)
        
        # Create inference engine from dataset
        start_time = time.time()
        engine = train_and_create_engine(
            args.dataset_name, 
            str(model_path),
            batch_size=config.pca_batch_size if config else 1000
        )
        
        training_time = time.time() - start_time
        
        # Get model statistics
        stats = engine.get_statistics()
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Model Statistics:")
        print(f"   Total points: {stats['model_stats']['total_points']:,}")
        print(f"   Total cells: {stats['model_stats']['total_cells']:,}")
        print(f"   Max depth: {stats['model_stats']['max_depth']}")
        print(f"   Model saved to: {model_path}")
        
        # Show model files created
        model_files = list(model_path.parent.glob(f"{model_path.name}*"))
        print(f"📄 Model files created:")
        for file in model_files:
            size_mb = file.stat().st_size / (1024*1024)
            print(f"   {file.name} ({size_mb:.1f}MB)")
        
        engine.shutdown()
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training error: {e}")
        return 1

def handle_benchmark_command(args):
    """Handle benchmark command"""
    print(f"🏃 Running benchmark on model: {args.model_path}")
    
    try:
        # Create inference engine
        engine = create_inference_engine(args.model_path)
        
        # Load test queries
        if args.dataset:
            dataset_manager = DatasetManager()
            vectors, queries = dataset_manager.load_dataset(args.dataset)
            if queries is None:
                print(f"❌ No queries available in dataset {args.dataset}")
                return 1
            test_queries = queries[:args.num_queries]
        else:
            # Generate random queries
            import numpy as np
            test_queries = [np.random.randn(128) for _ in range(args.num_queries)]
        
        print(f"📋 Benchmarking with {len(test_queries)} queries")
        print(f"🎯 Testing k values: {args.k_values}")
        
        # Run benchmark
        start_time = time.time()
        results = engine.benchmark(test_queries, args.k_values)
        benchmark_time = time.time() - start_time
        
        # Display results
        print(f"\n📊 Benchmark Results (completed in {benchmark_time:.2f}s):")
        print("=" * 60)
        
        for k_label, metrics in results.items():
            k = k_label.split('_')[1]
            print(f"k={k}:")
            print(f"  Average latency: {metrics['avg_latency_ms']:.2f}ms")
            print(f"  P95 latency: {metrics['p95_latency_ms']:.2f}ms")
            print(f"  P99 latency: {metrics['p99_latency_ms']:.2f}ms")
            print(f"  Queries per second: {metrics['queries_per_second']:.0f}")
            print()
        
        # Save results if output specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {args.output}")
        
        engine.shutdown()
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        logger.error(f"Benchmark error: {e}")
        return 1

def handle_search_command(args):
    """Handle search command"""
    print(f"🔍 Performing search with model: {args.model_path}")
    
    try:
        # Create inference engine
        engine = create_inference_engine(args.model_path)
        
        # Load query vectors
        if args.query_file:
            import numpy as np
            query_vectors = np.load(args.query_file)
            if query_vectors.ndim == 1:
                query_vectors = query_vectors.reshape(1, -1)
        else:
            # Use random query for demo
            import numpy as np
            query_vectors = np.random.randn(1, 128)
        
        print(f"🎯 Searching with {len(query_vectors)} queries, k={args.k}")
        
        # Perform batch search
        query_ids = [f"query_{i}" for i in range(len(query_vectors))]
        batch_results = engine.batch_search(query_vectors, k=args.k, query_ids=query_ids)
        
        # Display results
        for batch_result in batch_results:
            print(f"\n{batch_result.query_id} (latency: {batch_result.latency_ms:.2f}ms):")
            for i, result in enumerate(batch_result.results):
                print(f"  {i+1:2d}. {result.point_id}: {result.distance:.4f} (conf: {result.confidence:.3f})")
        
        # Save results if output specified
        if args.output:
            success = engine.export_results(batch_results, args.output, format='json')
            if success:
                print(f"💾 Results saved to: {args.output}")
            else:
                print(f"❌ Failed to save results")
        
        engine.shutdown()
        return 0
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        logger.error(f"Search error: {e}")
        return 1

def handle_test_command(args):
    """Handle test command"""
    print(f"🧪 Running DARG tests: {args.test_type}")
    
    if args.test_type == 'all':
        success = run_test_suite()
    else:
        success = run_specific_test(args.test_type)
    
    return 0 if success else 1

def handle_cpp_setup_command(args):
    """Handle C++ setup command"""
    print(f"🔧 Setting up C++ acceleration library")
    
    success = create_cpp_acceleration_library(args.output_dir)
    
    if success:
        print(f"✅ C++ acceleration library created in {args.output_dir}")
        print(f"📋 To build the library:")
        print(f"   cd {args.output_dir}")
        print(f"   ./build.sh")
        return 0
    else:
        print(f"❌ Failed to create C++ library")
        return 1

def handle_performance_command(args):
    """Handle performance analysis command"""
    print(f"⚡ Running performance analysis")
    print(f"   Vectors: {args.num_vectors:,}")
    print(f"   Queries: {args.num_queries:,}")
    print(f"   Dimensions: {args.dimensions}")
    
    try:
        import numpy as np
        from performance_test import DARGPerformanceTest
        
        # Create performance test instance
        test_runner = DARGPerformanceTest()
        
        # Run performance test
        test_runner.run_performance_test(args.num_vectors, args.num_queries)
        
        return 0
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        logger.error(f"Performance error: {e}")
        return 1

def main():
    """Main entry point"""
    parser = create_darg_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    print("🚀 DARG: Distributed Adaptive Routing Graph")
    print("=" * 50)
    
    try:
        if args.command == 'platform':
            return handle_platform_command(args)
        
        elif args.command == 'datasets':
            return handle_dataset_command(args)
        
        elif args.command == 'models':
            return handle_models_command(args)
        
        elif args.command == 'train':
            return handle_train_command(args)
        
        elif args.command == 'benchmark':
            return handle_benchmark_command(args)
        
        elif args.command == 'search':
            return handle_search_command(args)
        
        elif args.command == 'test':
            return handle_test_command(args)
        
        elif args.command == 'cpp-setup':
            return handle_cpp_setup_command(args)
        
        elif args.command == 'performance':
            return handle_performance_command(args)
        
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        return 1
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
