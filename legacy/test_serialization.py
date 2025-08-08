#!/usr/bin/env python3

import numpy as np
import time
from main import DARGv22

def test_serialization_fix():
    """Test the serialization fix with a smaller dataset"""
    print("Testing DARG serialization fix...")
    
    # Create system
    darg = DARGv22()
    
    # Generate smaller sample data
    np.random.seed(42)
    sample_data = [np.random.randn(128) for _ in range(1000)]
    
    # Initialize with sample data
    print("Initializing DARG system...")
    darg.initialize(sample_data)
    
    # Insert points
    print("Inserting 10,000 points...")
    for i in range(10000):
        vector = np.random.randn(128)
        success = darg.insert(f"point_{i}", vector)
        if not success:
            print(f"Failed to insert point_{i}")
    
    # Test search before save
    print("Testing search before save...")
    query_vector = np.random.randn(128)
    results_before = darg.search(query_vector, k=5)
    print("Search results before save:")
    for i, (point_id, distance) in enumerate(results_before):
        print(f"  {i+1}. {point_id}: {distance:.4f}")
    
    # Save index
    print("Saving index...")
    start_time = time.time()
    darg.save_index("test_index.pkl")
    save_time = time.time() - start_time
    print(f"Index saved in {save_time:.2f} seconds")
    
    # Shutdown original system
    darg.shutdown()
    
    # Load index in new system
    print("Loading index...")
    darg2 = DARGv22()
    start_time = time.time()
    darg2.load_index("test_index.pkl")
    load_time = time.time() - start_time
    print(f"Index loaded in {load_time:.2f} seconds")
    
    # Test search after load
    print("Testing search after load...")
    try:
        results_after = darg2.search(query_vector, k=5)
        print("Search results after load:")
        for i, (point_id, distance) in enumerate(results_after):
            print(f"  {i+1}. {point_id}: {distance:.4f}")
        
        # Compare results
        print("\nComparison:")
        for i in range(min(len(results_before), len(results_after))):
            before_id, before_dist = results_before[i]
            after_id, after_dist = results_after[i]
            match = "✓" if before_id == after_id and abs(before_dist - after_dist) < 1e-6 else "✗"
            print(f"  {i+1}. {match} {before_id} ({before_dist:.4f}) vs {after_id} ({after_dist:.4f})")
        
        print("\n✅ Serialization test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during search after load: {e}")
        import traceback
        traceback.print_exc()
    
    # Shutdown
    darg2.shutdown()

if __name__ == "__main__":
    test_serialization_fix()
