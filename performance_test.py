#!/usr/bin/env python3
"""
DARG v2.2 Performance Timing Test Script

This script demonstrates the comprehensive performance timing system
implemented in DARG v2.2. It runs various operations and provides
detailed timing analysis for optimization purposes.
"""

import numpy as np
import time
import logging
from typing import List, Dict, Any
import sys
import os

# Add the DARG directory to path if needed
sys.path.append('/home/akhil/Downloads/temp/DARG')

from main import DARGv22, GlobalConfig, performance_timer

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DARGPerformanceTest:
    """Comprehensive performance testing for DARG v2.2"""
    
    def __init__(self):
        self.results = {}
        print("🚀 DARG v2.2 Performance Timing Test Suite")
        print("=" * 60)
    
    def generate_test_data(self, num_vectors: int = 1000, dimensions: int = 128) -> List[np.ndarray]:
        """Generate synthetic test data"""
        print(f"📊 Generating {num_vectors} test vectors with {dimensions} dimensions...")
        np.random.seed(42)  # For reproducible results
        
        # Create clustered data for more realistic testing
        centers = [np.random.randn(dimensions) * 10 for _ in range(10)]
        vectors = []
        
        for i in range(num_vectors):
            center = centers[i % len(centers)]
            noise = np.random.randn(dimensions) * 0.5
            vectors.append(center + noise)
        
        print(f"✅ Generated {len(vectors)} test vectors")
        return vectors
    
    def test_initialization(self, test_data: List[np.ndarray]) -> DARGv22:
        """Test DARG initialization timing"""
        print("\n🏗️  Testing DARG Initialization...")
        
        # Reset timing statistics
        performance_timer.reset_stats()
        
        # Create DARG instance
        darg = DARGv22()
        
        # Initialize with sample data
        sample_size = min(100, len(test_data))
        sample_data = test_data[:sample_size]
        
        start_time = time.perf_counter()
        darg.initialize(sample_data)
        init_time = time.perf_counter() - start_time
        
        print(f"✅ Initialization completed in {init_time:.4f}s")
        
        # Store results
        self.results['initialization'] = {
            'total_time': init_time,
            'sample_size': sample_size
        }
        
        return darg
    
    def test_bulk_insertions(self, darg: DARGv22, test_data: List[np.ndarray]) -> None:
        """Test bulk insertion performance"""
        print(f"\n📥 Testing Bulk Insertions ({len(test_data)} vectors)...")
        
        # Reset specific timing stats for insertions
        insert_start_time = time.perf_counter()
        successful_inserts = 0
        failed_inserts = 0
        
        for i, vector in enumerate(test_data):
            point_id = f"test_point_{i}"
            success = darg.insert(point_id, vector)
            
            if success:
                successful_inserts += 1
            else:
                failed_inserts += 1
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Inserted {i + 1}/{len(test_data)} vectors...")
        
        insert_total_time = time.perf_counter() - insert_start_time
        
        print(f"✅ Bulk insertion completed:")
        print(f"   Total time: {insert_total_time:.4f}s")
        print(f"   Successful: {successful_inserts}")
        print(f"   Failed: {failed_inserts}")
        print(f"   Rate: {successful_inserts/insert_total_time:.2f} inserts/sec")
        
        # Store results
        self.results['bulk_insertions'] = {
            'total_time': insert_total_time,
            'successful_inserts': successful_inserts,
            'failed_inserts': failed_inserts,
            'rate_per_second': successful_inserts / insert_total_time if insert_total_time > 0 else 0
        }
    
    def test_search_performance(self, darg: DARGv22, num_queries: int = 50, k: int = 10) -> None:
        """Test search performance with various query patterns"""
        print(f"\n🔍 Testing Search Performance ({num_queries} queries, k={k})...")
        
        # Generate query vectors
        np.random.seed(123)  # Different seed for queries
        query_vectors = [np.random.randn(128) for _ in range(num_queries)]
        
        search_times = []
        result_counts = []
        
        search_start_time = time.perf_counter()
        
        for i, query_vector in enumerate(query_vectors):
            query_start = time.perf_counter()
            results = darg.search(query_vector, k)
            query_time = time.perf_counter() - query_start
            
            search_times.append(query_time)
            result_counts.append(len(results))
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_queries} queries...")
        
        search_total_time = time.perf_counter() - search_start_time
        
        # Calculate statistics
        avg_search_time = np.mean(search_times)
        min_search_time = np.min(search_times)
        max_search_time = np.max(search_times)
        p95_search_time = np.percentile(search_times, 95)
        avg_results = np.mean(result_counts)
        
        print(f"✅ Search performance results:")
        print(f"   Total time: {search_total_time:.4f}s")
        print(f"   Average query time: {avg_search_time:.6f}s")
        print(f"   Min/Max query time: {min_search_time:.6f}s / {max_search_time:.6f}s")
        print(f"   P95 query time: {p95_search_time:.6f}s")
        print(f"   Average results returned: {avg_results:.1f}")
        print(f"   Query rate: {num_queries/search_total_time:.2f} queries/sec")
        
        # Store results
        self.results['search_performance'] = {
            'total_time': search_total_time,
            'num_queries': num_queries,
            'avg_query_time': avg_search_time,
            'min_query_time': min_search_time,
            'max_query_time': max_search_time,
            'p95_query_time': p95_search_time,
            'queries_per_second': num_queries / search_total_time if search_total_time > 0 else 0,
            'avg_results_returned': avg_results
        }
    
    def test_mixed_operations(self, darg: DARGv22) -> None:
        """Test mixed read/write operations"""
        print("\n🔄 Testing Mixed Operations (Insert/Search/Delete)...")
        
        np.random.seed(456)
        mixed_start_time = time.perf_counter()
        
        operations = {
            'inserts': 0,
            'searches': 0,
            'deletes': 0
        }
        
        # Perform mixed operations
        for i in range(100):
            operation = np.random.choice(['insert', 'search', 'delete'], p=[0.4, 0.5, 0.1])
            
            if operation == 'insert':
                vector = np.random.randn(128)
                point_id = f"mixed_point_{i}"
                darg.insert(point_id, vector)
                operations['inserts'] += 1
            
            elif operation == 'search':
                query_vector = np.random.randn(128)
                darg.search(query_vector, k=5)
                operations['searches'] += 1
            
            elif operation == 'delete':
                # Try to delete a random point
                point_id = f"test_point_{np.random.randint(0, 100)}"
                darg.delete(point_id)
                operations['deletes'] += 1
        
        mixed_total_time = time.perf_counter() - mixed_start_time
        
        print(f"✅ Mixed operations completed in {mixed_total_time:.4f}s:")
        print(f"   Inserts: {operations['inserts']}")
        print(f"   Searches: {operations['searches']}")
        print(f"   Deletes: {operations['deletes']}")
        
        self.results['mixed_operations'] = {
            'total_time': mixed_total_time,
            'operations': operations
        }
    
    def test_index_serialization(self, darg: DARGv22) -> None:
        """Test index save/load performance"""
        print("\n💾 Testing Index Serialization...")
        
        index_path = "/tmp/darg_perf_test_index"
        
        # Test save performance
        save_start = time.perf_counter()
        darg.save_index(index_path)
        save_time = time.perf_counter() - save_start
        
        # Test load performance
        new_darg = DARGv22()
        load_start = time.perf_counter()
        new_darg.load_index(index_path)
        load_time = time.perf_counter() - load_start
        
        # Verify loaded index works
        query_vector = np.random.randn(128)
        results = new_darg.search(query_vector, k=5)
        
        print(f"✅ Serialization performance:")
        print(f"   Save time: {save_time:.4f}s")
        print(f"   Load time: {load_time:.4f}s")
        print(f"   Verification: Found {len(results)} results in loaded index")
        
        # Cleanup
        new_darg.shutdown()
        
        self.results['serialization'] = {
            'save_time': save_time,
            'load_time': load_time,
            'verification_results': len(results)
        }
    
    def analyze_operation_breakdown(self) -> None:
        """Analyze timing breakdown by operation type"""
        print("\n📈 DETAILED OPERATION TIMING ANALYSIS")
        print("=" * 60)
        
        # Get comprehensive timing statistics
        all_stats = performance_timer.get_stats()
        
        if not all_stats:
            print("⚠️  No timing statistics available")
            return
        
        # Group operations by category
        categories = {
            'Configuration': ['config_load'],
            'Vector Operations': ['distance_calculation', 'vector_mean_calculation', 'project_vector'],
            'PCA Operations': ['batch_pca', 'incremental_pca_update'],
            'LID Operations': ['estimate_lid_two_nn'],
            'Database Operations': ['store_point', 'get_point_vector', 'get_point_leaf_cell', 'delete_point', 'update_point_cell'],
            'Grid Operations': ['get_cell', 'update_cell', 'find_leaf_cell', 'split_cell', 'initialize_grid', '_create_initial_levels'],
            'Cell Operations': ['compute_cell_representative', 'project_vector_global', 'initialize_linkage_cache'],
            'Update Operations': ['insert_point', 'delete_point_update', 'update_cell_representative', 'bubble_up_counts'],
            'Search Operations': ['search_k_nearest', 'phase1_grid_resonance', 'phase2_localized_refinement', 'phase3_echo_search', 'calculate_s_trigger', 'explore_linkage_cache'],
            'Maintenance Operations': ['_run_maintenance_tasks', '_adapt_linkage_caches', '_add_geometric_links', '_reestimate_lid_values', '_refresh_stale_pca_models'],
            'High-Level Operations': ['darg_initialize', 'darg_insert', 'darg_delete', 'darg_search', 'darg_save_index', 'darg_load_index']
        }
        
        for category, operations in categories.items():
            print(f"\n🏷️  {category}:")
            category_total_time = 0
            category_total_calls = 0
            
            for op_name in operations:
                if op_name in all_stats:
                    stats = all_stats[op_name]
                    category_total_time += stats['total_time']
                    category_total_calls += stats['count']
                    
                    print(f"   {op_name:35} | "
                          f"Count: {stats['count']:4d} | "
                          f"Total: {stats['total_time']:8.4f}s | "
                          f"Avg: {stats['avg_time']:8.6f}s | "
                          f"P95: {stats['p95_time']:8.6f}s")
            
            if category_total_calls > 0:
                print(f"   {'CATEGORY TOTAL':35} | "
                      f"Count: {category_total_calls:4d} | "
                      f"Total: {category_total_time:8.4f}s | "
                      f"Avg: {category_total_time/category_total_calls:8.6f}s")
    
    def generate_performance_report(self) -> None:
        """Generate comprehensive performance report"""
        print("\n📊 PERFORMANCE SUMMARY REPORT")
        print("=" * 60)
        
        # High-level metrics
        if 'bulk_insertions' in self.results:
            insert_stats = self.results['bulk_insertions']
            print(f"📥 Insertion Performance:")
            print(f"   Rate: {insert_stats['rate_per_second']:.2f} inserts/second")
            print(f"   Success rate: {insert_stats['successful_inserts']/(insert_stats['successful_inserts']+insert_stats['failed_inserts'])*100:.1f}%")
        
        if 'search_performance' in self.results:
            search_stats = self.results['search_performance']
            print(f"\n🔍 Search Performance:")
            print(f"   Rate: {search_stats['queries_per_second']:.2f} queries/second")
            print(f"   Average latency: {search_stats['avg_query_time']*1000:.2f}ms")
            print(f"   P95 latency: {search_stats['p95_query_time']*1000:.2f}ms")
            
            # Performance targets analysis
            target_latency_ms = 1000  # 1 second target
            if search_stats['p95_query_time'] * 1000 <= target_latency_ms:
                print(f"   ✅ Meeting sub-1s query target (P95: {search_stats['p95_query_time']*1000:.2f}ms)")
            else:
                print(f"   ⚠️  Above 1s target (P95: {search_stats['p95_query_time']*1000:.2f}ms)")
        
        if 'serialization' in self.results:
            serial_stats = self.results['serialization']
            print(f"\n💾 Serialization Performance:")
            print(f"   Save time: {serial_stats['save_time']:.4f}s")
            print(f"   Load time: {serial_stats['load_time']:.4f}s")
        
        # Identify bottlenecks
        print(f"\n🚨 POTENTIAL BOTTLENECKS:")
        all_stats = performance_timer.get_stats()
        
        # Find operations with highest total time
        total_times = [(name, stats['total_time']) for name, stats in all_stats.items()]
        total_times.sort(key=lambda x: x[1], reverse=True)
        
        print("   Top time-consuming operations:")
        for name, total_time in total_times[:5]:
            stats = all_stats[name]
            print(f"   {name:35} | Total: {total_time:8.4f}s | Calls: {stats['count']:4d}")
        
        # Find operations with highest average time
        avg_times = [(name, stats['avg_time']) for name, stats in all_stats.items() if stats['count'] > 1]
        avg_times.sort(key=lambda x: x[1], reverse=True)
        
        print("\n   Slowest average operations:")
        for name, avg_time in avg_times[:5]:
            stats = all_stats[name]
            print(f"   {name:35} | Avg: {avg_time:8.6f}s | Calls: {stats['count']:4d}")
    
    def run_comprehensive_test(self):
        """Run the complete performance test suite"""
        print("🏁 Starting Comprehensive Performance Test Suite...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(num_vectors=500, dimensions=128)
            
            # Test initialization
            darg = self.test_initialization(test_data)
            
            # Test bulk insertions
            self.test_bulk_insertions(darg, test_data)
            
            # Test search performance
            self.test_search_performance(darg, num_queries=25, k=10)
            
            # Test mixed operations
            self.test_mixed_operations(darg)
            
            # Test serialization
            self.test_index_serialization(darg)
            
            # Analyze timing breakdown
            self.analyze_operation_breakdown()
            
            # Generate report
            self.generate_performance_report()
            
            # Final cleanup
            darg.shutdown()
            
            print("\n🎉 Performance test suite completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    def run_quick_test(self):
        """Run a quick performance test with smaller data"""
        print("⚡ Running Quick Performance Test...")
        
        try:
            # Smaller test data
            test_data = self.generate_test_data(num_vectors=100, dimensions=64)
            
            # Initialize
            darg = self.test_initialization(test_data)
            
            # Quick insertion test
            print("\n📥 Quick insertion test...")
            for i, vector in enumerate(test_data[:50]):
                darg.insert(f"quick_point_{i}", vector)
            
            # Quick search test
            print("\n🔍 Quick search test...")
            query_vector = np.random.randn(64)
            results = darg.search(query_vector, k=5)
            print(f"Found {len(results)} results")
            
            # Show timing summary
            performance_timer.print_summary()
            
            darg.shutdown()
            print("\n✅ Quick test completed!")
            
        except Exception as e:
            print(f"\n❌ Quick test failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run performance tests"""
    
    # Check if comprehensive or quick test
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        test_mode = 'quick'
    else:
        test_mode = 'comprehensive'
    
    print(f"🔥 DARG v2.2 Performance Testing ({test_mode} mode)")
    print("=" * 60)
    
    # Create test instance
    test_runner = DARGPerformanceTest()
    
    if test_mode == 'quick':
        test_runner.run_quick_test()
    else:
        test_runner.run_comprehensive_test()
    
    print("\n🏆 Performance testing complete!")
    print("This timing data will help optimize DARG for billion-scale performance.")


if __name__ == "__main__":
    main()
