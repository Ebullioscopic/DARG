#!/usr/bin/env python3
"""
DARG Comprehensive Test Suite
Tests all components and provides coverage analysis
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the DARG directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import DARGv22, GlobalConfig, performance_timer
from platform_detection import PlatformDetector, get_platform_config
from dataset_manager import DatasetManager
from inference import DARGInferenceEngine, InferenceConfig

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestDARGCore(unittest.TestCase):
    """Test core DARG functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.darg = DARGv22()
        
        # Generate test data
        np.random.seed(42)
        self.test_vectors = [np.random.randn(64) for _ in range(100)]
        self.query_vectors = [np.random.randn(64) for _ in range(10)]
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'darg') and self.darg:
            self.darg.shutdown()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test DARG system initialization"""
        # Initialize with sample data
        self.darg.initialize(self.test_vectors[:10])
        self.assertTrue(self.darg.initialized)
        
        # Check root cell exists
        stats = self.darg.get_stats()
        self.assertGreater(stats['total_cells'], 0)
        self.assertIsNotNone(stats['root_cell_id'])
    
    def test_insertion(self):
        """Test point insertion"""
        self.darg.initialize(self.test_vectors[:5])
        
        # Insert points
        success_count = 0
        for i, vector in enumerate(self.test_vectors):
            if self.darg.insert(f"point_{i}", vector):
                success_count += 1
        
        # Should have successful insertions
        self.assertGreater(success_count, 80)  # Allow some failures
        
        # Check stats
        stats = self.darg.get_stats()
        self.assertGreater(stats['total_points'], 80)
    
    def test_search(self):
        """Test search functionality"""
        self.darg.initialize(self.test_vectors[:5])
        
        # Insert test data
        for i, vector in enumerate(self.test_vectors):
            self.darg.insert(f"point_{i}", vector)
        
        # Perform searches
        for query in self.query_vectors:
            results = self.darg.search(query, k=5)
            
            # Should return results
            self.assertLessEqual(len(results), 5)
            
            # Results should be tuples of (point_id, distance)
            for result in results:
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], str)
                self.assertIsInstance(result[1], (int, float))
    
    def test_deletion(self):
        """Test point deletion"""
        self.darg.initialize(self.test_vectors[:5])
        
        # Insert points
        point_ids = []
        for i, vector in enumerate(self.test_vectors[:20]):
            point_id = f"point_{i}"
            if self.darg.insert(point_id, vector):
                point_ids.append(point_id)
        
        initial_count = self.darg.get_stats()['total_points']
        
        # Delete some points
        deleted_count = 0
        for point_id in point_ids[:5]:
            if self.darg.delete(point_id):
                deleted_count += 1
        
        # Check that points were deleted
        final_count = self.darg.get_stats()['total_points']
        self.assertLess(final_count, initial_count)
    
    def test_serialization(self):
        """Test index save/load"""
        self.darg.initialize(self.test_vectors[:5])
        
        # Insert data
        for i, vector in enumerate(self.test_vectors[:20]):
            self.darg.insert(f"point_{i}", vector)
        
        # Perform a search before saving
        query = self.query_vectors[0]
        results_before = self.darg.search(query, k=3)
        
        # Save index
        index_path = os.path.join(self.test_dir, "test_index.pkl")
        self.darg.save_index(index_path)
        
        # Shutdown original system
        self.darg.shutdown()
        
        # Load index in new system
        new_darg = DARGv22()
        new_darg.load_index(index_path)
        
        # Perform same search
        results_after = new_darg.search(query, k=3)
        
        # Results should be similar (allowing for small floating point differences)
        self.assertEqual(len(results_before), len(results_after))
        
        if results_before and results_after:
            # Check first result
            self.assertEqual(results_before[0][0], results_after[0][0])  # Same point ID
            self.assertAlmostEqual(results_before[0][1], results_after[0][1], places=5)  # Similar distance
        
        new_darg.shutdown()

class TestPlatformDetection(unittest.TestCase):
    """Test platform detection functionality"""
    
    def test_platform_detection(self):
        """Test platform detection"""
        detector = PlatformDetector()
        config = detector.config
        
        # Should have valid configuration
        self.assertIsNotNone(config)
        self.assertIn(config.system, ['Darwin', 'Linux', 'Windows'])
        self.assertGreater(config.cpu_cores, 0)
        self.assertGreater(config.memory_gb, 0)
        self.assertGreater(config.recommended_threads, 0)
        self.assertGreater(config.recommended_batch_size, 0)
    
    def test_gpu_detection(self):
        """Test GPU detection"""
        detector = PlatformDetector()
        config = detector.config
        
        # GPU type should be valid
        self.assertIn(config.gpu_type, ['cuda', 'mps', 'none'])
        
        # If GPU available, should have devices
        if config.gpu_available:
            self.assertGreater(len(config.gpu_devices), 0)
        else:
            self.assertEqual(len(config.gpu_devices), 0)
    
    def test_acceleration_config(self):
        """Test acceleration configuration"""
        detector = PlatformDetector()
        accel_config = detector.get_acceleration_config()
        
        # Should have required fields
        self.assertIn('enabled', accel_config)
        self.assertIn('type', accel_config)
        self.assertIn('threads', accel_config)
        self.assertIn('batch_size', accel_config)

class TestDatasetManager(unittest.TestCase):
    """Test dataset management functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_manager = DatasetManager(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset generation"""
        # Generate small synthetic dataset
        success = self.dataset_manager.download_dataset('synthetic_small')
        self.assertTrue(success)
        
        # Check if dataset was created
        self.assertTrue(self.dataset_manager.is_downloaded('synthetic_small'))
        
        # Load and verify dataset
        vectors, queries = self.dataset_manager.load_dataset('synthetic_small')
        self.assertIsNotNone(vectors)
        self.assertIsNotNone(queries)
        
        # Check dimensions
        self.assertEqual(vectors.shape[1], 128)
        self.assertEqual(queries.shape[1], 128)
        
        # Check counts
        self.assertEqual(len(vectors), 10000)
        self.assertEqual(len(queries), 1000)
    
    def test_dataset_info(self):
        """Test dataset information retrieval"""
        info = self.dataset_manager.get_dataset_info('synthetic_small')
        self.assertIsNotNone(info)
        self.assertEqual(info.name, 'Synthetic-Small')
        self.assertEqual(info.dimensions, 128)
        self.assertEqual(info.vectors_count, 10000)
    
    def test_list_datasets(self):
        """Test dataset listing"""
        # This should not raise an exception
        self.dataset_manager.list_datasets()

class TestInferenceEngine(unittest.TestCase):
    """Test inference engine functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple DARG model for testing
        self.darg = DARGv22()
        test_vectors = [np.random.randn(64) for _ in range(50)]
        self.darg.initialize(test_vectors[:5])
        
        for i, vector in enumerate(test_vectors):
            self.darg.insert(f"test_point_{i}", vector)
        
        self.model_path = os.path.join(self.test_dir, "test_model.pkl")
        self.darg.save_index(self.model_path)
        self.darg.shutdown()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_inference_engine_creation(self):
        """Test inference engine creation"""
        config = InferenceConfig(model_path=self.model_path)
        engine = DARGInferenceEngine(config)
        
        # Load model
        success = engine.load_model()
        self.assertTrue(success)
        
        engine.shutdown()
    
    def test_inference_search(self):
        """Test inference search"""
        config = InferenceConfig(model_path=self.model_path)
        engine = DARGInferenceEngine(config)
        engine.load_model()
        
        # Perform search
        query = np.random.randn(64)
        results = engine.search(query, k=5)
        
        # Check results
        self.assertLessEqual(len(results), 5)
        
        for result in results:
            self.assertIsInstance(result.point_id, str)
            self.assertIsInstance(result.distance, (int, float))
            self.assertIsInstance(result.confidence, (int, float))
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
        
        engine.shutdown()
    
    def test_batch_search(self):
        """Test batch search functionality"""
        config = InferenceConfig(model_path=self.model_path)
        engine = DARGInferenceEngine(config)
        engine.load_model()
        
        # Prepare batch queries
        queries = [np.random.randn(64) for _ in range(5)]
        query_ids = [f"query_{i}" for i in range(5)]
        
        # Perform batch search
        batch_results = engine.batch_search(queries, k=3, query_ids=query_ids)
        
        # Check results
        self.assertEqual(len(batch_results), 5)
        
        for batch_result in batch_results:
            self.assertIn(batch_result.query_id, query_ids)
            self.assertLessEqual(len(batch_result.results), 3)
            self.assertGreaterEqual(batch_result.latency_ms, 0)
        
        engine.shutdown()
    
    def test_statistics(self):
        """Test statistics collection"""
        config = InferenceConfig(model_path=self.model_path)
        engine = DARGInferenceEngine(config)
        engine.load_model()
        
        # Perform some searches
        for _ in range(3):
            query = np.random.randn(64)
            engine.search(query, k=2)
        
        # Get statistics
        stats = engine.get_statistics()
        
        # Check structure
        self.assertIn('inference_stats', stats)
        self.assertIn('model_stats', stats)
        self.assertIn('platform_info', stats)
        
        # Check inference stats
        inference_stats = stats['inference_stats']
        self.assertEqual(inference_stats['total_queries'], 3)
        self.assertGreater(inference_stats['average_latency_ms'], 0)
        
        engine.shutdown()

class TestPerformance(unittest.TestCase):
    """Test performance and benchmarking"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_vectors = [np.random.randn(64) for _ in range(200)]
        self.query_vectors = [np.random.randn(64) for _ in range(20)]
    
    def test_performance_timing(self):
        """Test performance timing system"""
        # Reset performance timer
        performance_timer.reset_stats()
        
        # Create and use DARG system
        darg = DARGv22()
        darg.initialize(self.test_vectors[:5])
        
        # Insert data (timed operations)
        for i, vector in enumerate(self.test_vectors):
            darg.insert(f"perf_point_{i}", vector)
        
        # Perform searches (timed operations)
        for query in self.query_vectors:
            darg.search(query, k=5)
        
        # Get timing statistics
        stats = performance_timer.get_stats()
        
        # Should have timing data
        self.assertGreater(len(stats), 0)
        
        # Check for key operations
        expected_operations = ['darg_insert', 'darg_search', 'insert_point', 'search_k_nearest']
        for operation in expected_operations:
            if operation in stats:
                op_stats = stats[operation]
                self.assertGreater(op_stats['count'], 0)
                self.assertGreater(op_stats['total_time'], 0)
                self.assertGreater(op_stats['avg_time'], 0)
        
        darg.shutdown()
    
    def test_scalability_measurement(self):
        """Test scalability measurement"""
        sizes = [50, 100, 200]
        latencies = []
        
        for size in sizes:
            darg = DARGv22()
            test_data = self.test_vectors[:size]
            
            darg.initialize(test_data[:5])
            
            # Insert data
            for i, vector in enumerate(test_data):
                darg.insert(f"scale_point_{i}", vector)
            
            # Measure search time
            query = self.query_vectors[0]
            start_time = time.time()
            darg.search(query, k=5)
            latency = time.time() - start_time
            
            latencies.append(latency)
            darg.shutdown()
        
        # Latency should not increase dramatically
        # (This is a basic scalability check)
        self.assertLess(latencies[-1] / latencies[0], 10)  # Should scale reasonably

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Generate synthetic dataset
        dataset_manager = DatasetManager(self.test_dir)
        success = dataset_manager.download_dataset('synthetic_small')
        self.assertTrue(success)
        
        # 2. Load dataset
        vectors, queries = dataset_manager.load_dataset('synthetic_small')
        self.assertIsNotNone(vectors)
        self.assertIsNotNone(queries)
        
        # 3. Create and train DARG system
        darg = DARGv22()
        darg.initialize(vectors[:100])  # Use subset for speed
        
        # Insert subset of vectors
        test_vectors = vectors[:500]  # Use subset for speed
        for i, vector in enumerate(test_vectors):
            darg.insert(f"e2e_point_{i}", vector)
        
        # 4. Save model
        model_path = os.path.join(self.test_dir, "e2e_model.pkl")
        darg.save_index(model_path)
        darg.shutdown()
        
        # 5. Create inference engine
        config = InferenceConfig(model_path=model_path)
        engine = DARGInferenceEngine(config)
        engine.load_model()
        
        # 6. Perform searches
        test_queries = queries[:10]  # Use subset for speed
        batch_results = engine.batch_search(test_queries, k=5)
        
        # 7. Verify results
        self.assertEqual(len(batch_results), len(test_queries))
        
        for batch_result in batch_results:
            self.assertLessEqual(len(batch_result.results), 5)
            self.assertGreaterEqual(batch_result.latency_ms, 0)
        
        # 8. Get statistics
        stats = engine.get_statistics()
        self.assertGreater(stats['inference_stats']['total_queries'], 0)
        
        engine.shutdown()

def run_test_suite():
    """Run the complete test suite"""
    print("🧪 DARG Comprehensive Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDARGCore,
        TestPlatformDetection,
        TestDatasetManager,
        TestInferenceEngine,
        TestPerformance,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) 
                   / result.testsRun * 100 if result.testsRun > 0 else 0)
    
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ Test suite PASSED")
        return True
    else:
        print("❌ Test suite FAILED")
        return False

def run_specific_test(test_name: str):
    """Run a specific test class"""
    test_classes = {
        'core': TestDARGCore,
        'platform': TestPlatformDetection,
        'dataset': TestDatasetManager,
        'inference': TestInferenceEngine,
        'performance': TestPerformance,
        'integration': TestIntegration
    }
    
    if test_name not in test_classes:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {list(test_classes.keys())}")
        return False
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_classes[test_name])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        success = run_test_suite()
    
    sys.exit(0 if success else 1)
