"""
Simple test for Universal DARG System
=====================================
"""

import numpy as np
import sys
import os
sys.path.append('/Users/hariharan/Python-Programs/DARG')

def test_universal_darg():
    """Test Universal DARG with simple data"""
    
    print("🧪 Testing Universal DARG System")
    print("=" * 40)
    
    try:
        from universal_darg import UniversalDARG, TextEncoder
        
        # Initialize system
        darg = UniversalDARG()
        
        # Add some text data
        sample_texts = [
            "Machine learning is awesome",
            "Deep learning uses neural networks",
            "Artificial intelligence transforms technology",
            "Data science combines statistics and programming",
            "Natural language processing handles text"
        ]
        
        print("📝 Adding text data to Universal DARG...")
        for i, text in enumerate(sample_texts):
            darg.add_data(text, "text", metadata={"id": i})
        
        # Perform search
        print("🔍 Performing similarity search...")
        query = "AI and machine learning"
        results = darg.search(query, "text", k=3)
        
        print(f"\nQuery: '{query}'")
        print("Results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['data']} (similarity: {result['similarity']:.3f})")
        
        print("\n✅ Universal DARG test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Universal DARG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_darg():
    """Test Enhanced DARG with vectors"""
    
    print("\n⚡ Testing Enhanced DARG System")
    print("=" * 40)
    
    try:
        from enhanced_darg import EnhancedDARG
        
        # Create sample vectors
        vectors = np.random.randn(10, 128).astype(np.float32)
        
        # Initialize Enhanced DARG
        darg = EnhancedDARG(vector_dim=128)
        
        print("📊 Adding vectors to Enhanced DARG...")
        for i, vector in enumerate(vectors):
            darg.add_vector(vector, f"vector_{i}")
        
        # Perform search
        print("🔍 Performing vector search...")
        query_vector = np.random.randn(128).astype(np.float32)
        results = darg.search(query_vector, k=3)
        
        print("Search Results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['node_id']} (distance: {result['distance']:.3f})")
        
        print("\n✅ Enhanced DARG test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced DARG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neo4j_mock():
    """Test Neo4j visualization with mock"""
    
    print("\n🎨 Testing Neo4j Visualization (Mock)")
    print("=" * 40)
    
    try:
        from neo4j_visualizer import AdvancedNeo4jVisualizer
        
        # Force use of mock by providing invalid connection details
        visualizer = AdvancedNeo4jVisualizer(uri="mock://localhost:7687")
        
        # Add sample nodes
        print("📊 Adding sample nodes...")
        visualizer.add_data_node(
            "node_1", "text", 384, time.time(), 
            {"content": "Test node"}, 5.2, 1
        )
        
        visualizer.add_data_node(
            "node_2", "text", 384, time.time(), 
            {"content": "Another test node"}, 4.8, 1
        )
        
        # Add relationship
        visualizer.add_similarity_relationship("node_1", "node_2", 0.85)
        
        # Get analytics
        print("📈 Getting graph analytics...")
        analytics = visualizer.get_graph_analytics()
        
        print("Analytics:")
        basic_stats = analytics.get('basic_stats', {})
        for key, value in basic_stats.items():
            print(f"  {key}: {value}")
        
        visualizer.close()
        
        print("\n✅ Neo4j visualization test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("🚀 Universal DARG Test Suite")
    print("=" * 50)
    
    # Import time after other imports
    import time
    
    tests_passed = 0
    total_tests = 3
    
    if test_universal_darg():
        tests_passed += 1
    
    if test_enhanced_darg():
        tests_passed += 1
    
    if test_neo4j_mock():
        tests_passed += 1
    
    print(f"\n🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Universal DARG system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    import time
    success = main()
    exit(0 if success else 1)
