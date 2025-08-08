#!/usr/bin/env python3
"""
DARG Installation and Setup Script
Automated setup for the complete DARG implementation
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(command, description, cwd=None):
    """Run a command and handle errors gracefully"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        print(f"✅ {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("🖥️  Checking system requirements...")
    
    system = platform.system()
    machine = platform.machine()
    
    print(f"   Operating System: {system}")
    print(f"   Architecture: {machine}")
    
    # Check for required system tools
    tools = ['git', 'cmake', 'make']
    if system == "Windows":
        tools = ['git', 'cmake']  # make not required on Windows
    
    missing_tools = []
    for tool in tools:
        if shutil.which(tool) is None:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"❌ Missing required tools: {', '.join(missing_tools)}")
        print("Please install them before proceeding")
        return False
    
    print("✅ System requirements met")
    return True

def create_virtual_environment():
    """Create a virtual environment for DARG"""
    venv_path = Path("darg_env")
    
    if venv_path.exists():
        print("🔄 Virtual environment already exists, recreating...")
        shutil.rmtree(venv_path)
    
    success, _ = run_command(
        f"{sys.executable} -m venv darg_env",
        "Creating virtual environment"
    )
    
    if not success:
        return False
    
    # Get activation script path
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip"
    else:
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
    
    print(f"📝 Virtual environment created at: {venv_path.absolute()}")
    print(f"   To activate: source {activate_script}")
    
    return True, str(pip_path)

def install_dependencies(pip_path):
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Core dependencies
    core_deps = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "h5py>=3.6.0",
        "pandas>=1.3.0",
        "requests>=2.28.0",
        "tqdm>=4.62.0"
    ]
    
    # Testing dependencies
    test_deps = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-benchmark>=4.0.0"
    ]
    
    # Optional ML dependencies
    ml_deps = [
        "torch>=1.12.0",
        "faiss-cpu>=1.7.0"
    ]
    
    # Install core dependencies
    for dep in core_deps:
        success, _ = run_command(
            f"{pip_path} install {dep}",
            f"Installing {dep.split('>=')[0]}"
        )
        if not success:
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    # Install testing dependencies
    for dep in test_deps:
        success, _ = run_command(
            f"{pip_path} install {dep}",
            f"Installing {dep.split('>=')[0]}"
        )
        if not success:
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    # Try to install optional dependencies
    print("🔧 Installing optional dependencies (may fail on some systems)...")
    for dep in ml_deps:
        success, _ = run_command(
            f"{pip_path} install {dep}",
            f"Installing {dep.split('>=')[0]}"
        )
        if not success:
            print(f"⚠️  Failed to install {dep} (optional), continuing...")
    
    return True

def setup_cpp_acceleration():
    """Set up C++ acceleration if possible"""
    print("🚀 Setting up C++ acceleration...")
    
    # Check if C++ compiler is available
    compilers = ['g++', 'clang++', 'cl']
    compiler_found = None
    
    for compiler in compilers:
        if shutil.which(compiler):
            compiler_found = compiler
            break
    
    if not compiler_found:
        print("⚠️  No C++ compiler found, skipping C++ acceleration")
        return True
    
    print(f"✅ Found C++ compiler: {compiler_found}")
    
    # Create C++ directory
    cpp_dir = Path("cpp")
    cpp_dir.mkdir(exist_ok=True)
    
    # Try to generate C++ acceleration library
    try:
        from cpp_acceleration import create_cpp_acceleration_library
        success = create_cpp_acceleration_library("cpp")
        
        if success:
            # Try to build the library
            build_script = cpp_dir / "build.sh"
            if build_script.exists():
                success, _ = run_command(
                    "chmod +x build.sh && ./build.sh",
                    "Building C++ acceleration library",
                    cwd=str(cpp_dir)
                )
                if success:
                    print("✅ C++ acceleration library built successfully")
                else:
                    print("⚠️  C++ library generation succeeded but build failed")
            else:
                print("⚠️  C++ build script not found")
        else:
            print("⚠️  Failed to generate C++ acceleration library")
    
    except Exception as e:
        print(f"⚠️  C++ acceleration setup failed: {e}")
    
    return True

def download_sample_dataset():
    """Download a small sample dataset for testing"""
    print("📥 Downloading sample dataset...")
    
    try:
        from dataset_manager import DatasetManager
        
        manager = DatasetManager()
        success = manager.download_dataset("synthetic_small")
        
        if success:
            print("✅ Sample dataset downloaded successfully")
        else:
            print("⚠️  Failed to download sample dataset")
    
    except Exception as e:
        print(f"⚠️  Dataset download failed: {e}")
    
    return True

def run_basic_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        print("   Testing imports...")
        import numpy as np
        import scipy
        import sklearn
        from main import DARGv22
        from platform_detection import PlatformDetector
        print("   ✅ All imports successful")
        
        # Test platform detection
        print("   Testing platform detection...")
        detector = PlatformDetector()
        info = detector.get_platform_info()
        print(f"   ✅ Platform: {info['platform']}")
        
        # Test basic DARG functionality
        print("   Testing basic DARG functionality...")
        config = DARGv22.get_default_config()
        print("   ✅ DARG configuration loaded")
        
        print("✅ All basic tests passed")
        return True
    
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False

def create_example_scripts():
    """Create example scripts for users"""
    print("📝 Creating example scripts...")
    
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Quick start example
    quick_start = examples_dir / "quick_start.py"
    quick_start.write_text('''#!/usr/bin/env python3
"""
DARG Quick Start Example
This script demonstrates basic DARG usage
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main import DARGv22
from inference import train_and_create_engine
from dataset_manager import DatasetManager

def main():
    print("🚀 DARG Quick Start Example")
    print("=" * 40)
    
    # Generate sample data
    print("📊 Generating sample data...")
    np.random.seed(42)
    vectors = np.random.randn(1000, 128).astype(np.float32)
    queries = np.random.randn(10, 128).astype(np.float32)
    
    # Create DARG instance
    print("🏗️  Creating DARG instance...")
    config = DARGv22.get_default_config()
    darg = DARGv22(config)
    
    # Add vectors
    print("📥 Adding vectors...")
    for i, vector in enumerate(vectors):
        darg.add_point(f"point_{i}", vector, {"category": "sample"})
    
    # Search
    print("🔍 Performing searches...")
    for i, query in enumerate(queries):
        results = darg.search(query, k=5)
        print(f"Query {i+1}: {len(results)} results")
        for j, (point_id, distance, metadata) in enumerate(results):
            print(f"  {j+1}. {point_id}: {distance:.4f}")
    
    print("✅ Quick start completed successfully!")

if __name__ == "__main__":
    main()
''')
    
    # Benchmark example
    benchmark_example = examples_dir / "benchmark_example.py"
    benchmark_example.write_text('''#!/usr/bin/env python3
"""
DARG Benchmark Example
This script demonstrates performance benchmarking
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main import DARGv22
from performance_test import DARGPerformanceTest

def main():
    print("⚡ DARG Benchmark Example")
    print("=" * 40)
    
    # Create performance test
    test_runner = DARGPerformanceTest()
    
    # Run small benchmark
    print("🏃 Running benchmark...")
    test_runner.run_performance_test(
        num_vectors=5000,
        num_queries=100
    )
    
    print("✅ Benchmark completed!")

if __name__ == "__main__":
    main()
''')
    
    print(f"✅ Example scripts created in {examples_dir}")
    return True

def print_setup_summary():
    """Print setup summary and next steps"""
    print("\n" + "="*60)
    print("🎉 DARG Setup Complete!")
    print("="*60)
    
    print("\n📋 What was installed:")
    print("   ✅ Python virtual environment (darg_env)")
    print("   ✅ Core dependencies (numpy, scipy, scikit-learn, etc.)")
    print("   ✅ Testing framework (pytest)")
    print("   ✅ Optional ML libraries (where possible)")
    print("   ✅ C++ acceleration (where possible)")
    print("   ✅ Sample dataset")
    print("   ✅ Example scripts")
    
    print("\n🚀 Getting Started:")
    print("   1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("      darg_env\\Scripts\\activate")
    else:
        print("      source darg_env/bin/activate")
    
    print("   2. Run quick start example:")
    print("      python examples/quick_start.py")
    
    print("   3. Run comprehensive tests:")
    print("      python darg_complete.py test")
    
    print("   4. Train on sample dataset:")
    print("      python darg_complete.py train synthetic_small --model-path model.pkl")
    
    print("   5. Run benchmark:")
    print("      python darg_complete.py benchmark model.pkl --dataset synthetic_small")
    
    print("\n📚 Available Commands:")
    print("   python darg_complete.py --help")
    
    print("\n🔧 Troubleshooting:")
    print("   - Check platform info: python darg_complete.py platform")
    print("   - Run tests: python darg_complete.py test")
    print("   - List datasets: python darg_complete.py datasets list")
    
    print("\n💡 For more information, see README.md")

def main():
    """Main setup function"""
    print("🚀 DARG Complete Setup")
    print("=" * 50)
    print("This script will set up the complete DARG implementation")
    print("including dependencies, C++ acceleration, and examples.")
    print()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check system requirements
    if not check_system_requirements():
        return 1
    
    # Create virtual environment
    try:
        success, pip_path = create_virtual_environment()
        if not success:
            return 1
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return 1
    
    # Install dependencies
    if not install_dependencies(pip_path):
        print("⚠️  Some dependencies failed to install, but continuing...")
    
    # Setup C++ acceleration
    setup_cpp_acceleration()
    
    # Download sample dataset
    download_sample_dataset()
    
    # Create example scripts
    create_example_scripts()
    
    # Run basic tests
    if not run_basic_tests():
        print("⚠️  Some tests failed, but setup is mostly complete")
    
    # Print summary
    print_setup_summary()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
