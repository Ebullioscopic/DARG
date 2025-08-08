"""
Simple test just for the core systems working
"""

def test_basic_functionality():
    """Test basic functionality only"""
    
    print("🧪 Basic Functionality Test")
    print("=" * 30)
    
    try:
        from universal_darg import UniversalDARG
        
        # Test 1: Create system
        darg = UniversalDARG()
        print("✅ Universal DARG initialized")
        
        # Test 2: Add data
        darg.add_data("test document", "text", metadata={"id": 1})
        print("✅ Data addition works")
        
        # Test 3: Check data storage
        print(f"✅ Total items: {len(darg.data_items)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_enhanced_darg():
    """Test Enhanced DARG"""
    
    print("\n⚡ Enhanced DARG Test")
    print("=" * 25)
    
    try:
        from enhanced_darg import EnhancedDARG
        import numpy as np
        
        # Test 1: Create system
        darg = EnhancedDARG(vector_dim=64)
        print("✅ Enhanced DARG initialized")
        
        # Test 2: Add vectors
        vector = np.random.randn(64).astype(np.float32)
        darg.add_vector(vector, "test_doc")
        print("✅ Vector addition works")
        
        # Test 3: Get stats (this should handle division by zero)
        stats = darg.get_performance_stats()
        print(f"✅ Stats retrieved: {stats['total_nodes']} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Basic System Tests")
    print("=" * 25)
    
    success = 0
    total = 2
    
    if test_basic_functionality():
        success += 1
    
    if test_enhanced_darg():
        success += 1
    
    print(f"\n📊 Results: {success}/{total} tests passed")
    
    if success == total:
        print("🎉 All basic tests passed!")
    else:
        print("⚠️  Some tests failed")
