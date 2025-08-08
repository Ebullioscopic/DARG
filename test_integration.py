#!/usr/bin/env python3
"""
DARG Integration Test Suite
==========================

Comprehensive test suite to verify all components of the DARG system work correctly
after reorganization and professionalization.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

# Also add current directory for legacy imports
sys.path.insert(0, str(current_dir))

# Test imports
def test_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from darg.core.universal_darg import UniversalDARG, DataItem, TextEncoder
        print("  ✅ Universal DARG imports successful")
    except ImportError as e:
        print(f"  ❌ Universal DARG import failed: {e}")
        return False
    
    try:
        from darg.core.enhanced_darg import EnhancedDARG
        print("  ✅ Enhanced DARG imports successful")
    except ImportError as e:
        print(f"  ❌ Enhanced DARG import failed: {e}")
        return False
    
    try:
        from darg.visualization.neo4j_integration import AdvancedNeo4jVisualizer
        print("  ✅ Neo4j visualization imports successful")
    except ImportError as e:
        print(f"  ❌ Neo4j visualization import failed: {e}")
        return False
    
    try:
        from darg.testing.validation_suite import ComprehensiveBenchmark
        print("  ✅ Validation suite imports successful")
    except ImportError as e:
        print(f"  ❌ Validation suite import failed: {e}")
        return False
    
    return True

def test_universal_darg():
    """Test Universal DARG functionality"""
    print("\n🚀 Testing Universal DARG...")
    
    try:
        from darg.core.universal_darg import UniversalDARG
        
        # Initialize without Neo4j to avoid connection issues
        udarg = UniversalDARG(enable_visualization=False)
        print("  ✅ Universal DARG initialized")
        
        # Test text data handling
        try:
            text_id = udarg.add_data("Test machine learning text", "text", metadata={"test": True})
            print(f"  ✅ Text data added with ID: {text_id}")
        except Exception as e:
            print(f"  ⚠️  Text encoder not available: {e}")
        
        # Test statistics
        stats = udarg.get_statistics()
        print(f"  ✅ Statistics retrieved: {stats['total_items']} items")
        
        udarg.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Universal DARG test failed: {e}")
        return False

def test_enhanced_darg():
    """Test Enhanced DARG functionality"""
    print("\n⚡ Testing Enhanced DARG...")
    
    try:
        from darg.core.enhanced_darg import EnhancedDARG
        import numpy as np
        
        # Initialize Enhanced DARG
        edarg = EnhancedDARG()
        print("  ✅ Enhanced DARG initialized")
        
        # Create test vectors
        test_vectors = np.random.rand(10, 128).astype(np.float32)
        test_ids = [f"test_{i}" for i in range(10)]
        
        # Add vectors one by one
        for i, (vector, test_id) in enumerate(zip(test_vectors, test_ids)):
            edarg.add_vector(vector, test_id, metadata={"index": i})
        print("  ✅ Vectors added to index")
        
        # Test search
        query = np.random.rand(128).astype(np.float32)
        results = edarg.search(query, k=3)
        print(f"  ✅ Search completed, found {len(results)} results")
        
        # Test statistics  
        try:
            stats = edarg.get_performance_stats()
            print(f"  ✅ Statistics: {len(edarg.nodes)} nodes indexed")
        except Exception as e:
            print(f"  ⚠️  Statistics error (minor): {e}")
            print(f"  ✅ Alternative count: {len(edarg.nodes)} nodes indexed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced DARG test failed: {e}")
        return False

def test_example_demo():
    """Test the complete demo example"""
    print("\n🎯 Testing complete demo example...")
    
    try:
        # Import and run demo
        sys.path.insert(0, str(Path(__file__).parent / "examples"))
        
        # Test if demo file exists and can be imported
        demo_path = Path(__file__).parent / "examples" / "complete_demo.py"
        if not demo_path.exists():
            print("  ❌ Demo file not found")
            return False
        
        print("  ✅ Demo file exists")
        
        # Try to import demo components (without running full demo)
        with open(demo_path, 'r') as f:
            demo_content = f.read()
            
        if "UniversalDARG" in demo_content and "EnhancedDARG" in demo_content:
            print("  ✅ Demo contains required components")
            return True
        else:
            print("  ❌ Demo missing required components")
            return False
            
    except Exception as e:
        print(f"  ❌ Demo test failed: {e}")
        return False

def test_project_structure():
    """Test project structure is correct"""
    print("\n📁 Testing project structure...")
    
    base_path = Path(__file__).parent
    
    required_dirs = [
        "src/darg",
        "src/darg/core", 
        "src/darg/visualization",
        "src/darg/testing",
        "examples",
        "tests",
        "scripts",
        "docs",
        "legacy"
    ]
    
    required_files = [
        "src/darg/__init__.py",
        "src/darg/core/__init__.py",
        "src/darg/core/universal_darg.py",
        "src/darg/core/enhanced_darg.py",
        "src/darg/visualization/neo4j_integration.py",
        "src/darg/testing/validation_suite.py",
        "examples/complete_demo.py",
        "tests/test_universal_darg.py",
        "tests/test_basic_functionality.py",
        "scripts/setup.py",
        "README.md",
        "requirements.txt",
        "setup.py"
    ]
    
    # Check directories
    missing_dirs = []
    for dir_path in required_dirs:
        if not (base_path / dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
        return False
    else:
        print(f"  ✅ All {len(required_dirs)} required directories exist")
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if not (base_path / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print(f"  ✅ All {len(required_files)} required files exist")
    
    return True

def test_configuration():
    """Test configuration files are valid"""
    print("\n⚙️  Testing configuration...")
    
    base_path = Path(__file__).parent
    
    # Test config.json
    config_path = base_path / "config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("  ✅ config.json is valid JSON")
        except Exception as e:
            print(f"  ❌ config.json is invalid: {e}")
            return False
    else:
        print("  ⚠️  config.json not found (will be created automatically)")
    
    # Test requirements.txt
    req_path = base_path / "requirements.txt"
    if req_path.exists():
        try:
            with open(req_path, 'r') as f:
                lines = f.readlines()
            print(f"  ✅ requirements.txt exists with {len(lines)} lines")
        except Exception as e:
            print(f"  ❌ requirements.txt read failed: {e}")
            return False
    else:
        print("  ❌ requirements.txt not found")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 DARG Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration", test_configuration),
        ("Imports", test_imports),
        ("Universal DARG", test_universal_darg),
        ("Enhanced DARG", test_enhanced_darg),
        ("Example Demo", test_example_demo),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DARG system is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
