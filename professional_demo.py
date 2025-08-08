#!/usr/bin/env python3
"""
DARG Professional Demo
======================

Professional demonstration of the DARG system showing all core functionality
without requiring external dependencies.
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def simple_demo():
    """Simple demo without external dependencies"""
    
    print("🚀 DARG Professional Demo")
    print("=" * 40)
    
    # Test 1: Enhanced DARG with synthetic data
    print("\n1️⃣  Enhanced DARG - Vector Management")
    print("-" * 35)
    
    try:
        from darg.core.enhanced_darg import EnhancedDARG
        
        # Initialize
        edarg = EnhancedDARG(vector_dim=64)
        print("✅ Enhanced DARG initialized")
        
        # Add synthetic vectors
        print("📊 Adding synthetic vectors...")
        for i in range(20):
            vector = np.random.rand(64).astype(np.float32)
            vector_id = f"vector_{i:03d}"
            edarg.add_vector(vector, vector_id, metadata={"group": i % 4})
        
        print(f"✅ Added {len(edarg.nodes)} vectors to {len(edarg.partitions)} partitions")
        
        # Test search
        query = np.random.rand(64).astype(np.float32)
        results = edarg.search(query, k=5)
        print(f"✅ Search found {len(results)} similar vectors")
        
        print("📈 Enhanced DARG working correctly!")
        
    except Exception as e:
        print(f"❌ Enhanced DARG failed: {e}")
        return False
    
    # Test 2: Universal DARG with custom encoder
    print("\n2️⃣  Universal DARG - Custom Data Types")
    print("-" * 38)
    
    try:
        from darg.core.universal_darg import UniversalDARG, CustomEncoder
        
        # Create custom encoder for simple strings
        def string_to_vector(text):
            # Simple hash-based encoding (deterministic)
            hash_val = hash(text) % (2**32)
            np.random.seed(hash_val)
            return np.random.rand(32).astype(np.float32)
        
        custom_encoder = CustomEncoder(
            encode_func=string_to_vector,
            dimension=32,
            data_type="simple_text"
        )
        
        # Initialize Universal DARG
        udarg = UniversalDARG(enable_visualization=False)
        udarg.register_encoder(custom_encoder)
        print("✅ Universal DARG with custom encoder initialized")
        
        # Add custom data
        sample_texts = [
            "machine learning",
            "artificial intelligence", 
            "deep learning",
            "neural networks",
            "data science"
        ]
        
        item_ids = []
        for i, text in enumerate(sample_texts):
            item_id = udarg.add_data(text, "simple_text", metadata={"index": i})
            item_ids.append(item_id)
        
        print(f"✅ Added {len(item_ids)} custom data items")
        
        # Test search
        results = udarg.search("machine learning algorithms", "simple_text", k=3)
        print(f"✅ Search found {len(results)} similar items")
        
        # Show statistics
        stats = udarg.get_statistics()
        print(f"📊 System stats: {stats['total_items']} items, {len(stats['data_types'])} data types")
        
        print("🌐 Universal DARG working correctly!")
        udarg.close()
        
    except Exception as e:
        print(f"❌ Universal DARG failed: {e}")
        return False
    
    # Test 3: Neo4j Visualization (Mock)
    print("\n3️⃣  Neo4j Visualization - Mock Mode")
    print("-" * 35)
    
    try:
        from darg.visualization.neo4j_integration import AdvancedNeo4jVisualizer
        
        # Initialize mock visualizer
        visualizer = AdvancedNeo4jVisualizer(mock_mode=True)
        print("✅ Mock Neo4j visualizer initialized")
        
        # Create sample graph
        for i in range(10):
            node_data = {
                'id': f'node_{i}',
                'data_type': 'sample',
                'vector_dim': 64,
                'timestamp': time.time()
            }
            visualizer.add_node(node_data)
        
        # Add some edges
        for i in range(5):
            visualizer.add_similarity_edge(f'node_{i}', f'node_{i+1}', 0.8)
        
        # Get analytics
        analytics = visualizer.get_graph_analytics()
        print(f"📈 Graph analytics: {analytics['total_nodes']} nodes, {analytics['total_edges']} edges")
        
        print("📊 Neo4j visualization working correctly!")
        visualizer.close()
        
    except Exception as e:
        print(f"❌ Neo4j visualization failed: {e}")
        return False
    
    # Test 4: Import Validation
    print("\n4️⃣  System Integration - Import Validation")
    print("-" * 42)
    
    try:
        from darg.testing.validation_suite import ComprehensiveBenchmark, DatasetGenerator
        print("✅ Testing framework imports successful")
        
        # Quick generator test
        generator = DatasetGenerator()
        test_data = generator.generate_synthetic_vectors(100, 32)
        print(f"✅ Generated {len(test_data)} synthetic test vectors")
        
        print("🧪 Testing framework working correctly!")
        
    except Exception as e:
        print(f"❌ Testing framework failed: {e}")
        return False
    
    return True

def main():
    """Main demo function"""
    success = simple_demo()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ DARG System is professionally organized and functional")
        print("\n📋 Next Steps:")
        print("   • Install optional dependencies for full functionality:")
        print("     pip install transformers opencv-python librosa neo4j")
        print("   • Run comprehensive tests: python test_integration.py")
        print("   • Explore examples: python examples/complete_demo.py")
        print("   • Read documentation: README.md")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
