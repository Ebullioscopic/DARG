#!/usr/bin/env python3
"""
DARG Setup Script
================

Professional setup script for the DARG (Dynamic Adaptive Resonance Grids) package.
This script handles installation, dependencies, and configuration.
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Check Python version
if sys.version_info < (3, 8):
    raise RuntimeError("DARG requires Python 3.8 or higher")

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Universal Multi-Modal Vector Search System with Dynamic Graph Visualization"

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    req_path = Path(__file__).parent / filename
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Core requirements
install_requires = [
    "numpy>=1.19.0",
    "scipy>=1.5.0",
    "scikit-learn>=0.24.0",
    "joblib>=1.0.0",
    "psutil>=5.7.0",
    "tqdm>=4.50.0",
]

# Optional requirements
extras_require = {
    "text": [
        "transformers>=4.20.0",
        "torch>=1.9.0",
        "sentence-transformers>=2.2.0",
    ],
    "image": [
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
    ],
    "audio": [
        "librosa>=0.8.0",
        "soundfile>=0.10.0",
    ],
    "visualization": [
        "neo4j>=5.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    "performance": [
        "faiss-cpu>=1.7.0",
        "numba>=0.56.0",
    ],
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "mypy>=0.812",
        "pre-commit>=2.15.0",
    ],
}

# All optional dependencies
extras_require["all"] = list(set().union(*extras_require.values()))

# Package metadata
setup(
    name="darg",
    version="1.0.0",
    author="DARG Research Team",
    author_email="contact@darg-research.org",
    description="Universal Multi-Modal Vector Search System with Dynamic Graph Visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ebullioscopic/DARG",
    project_urls={
        "Bug Reports": "https://github.com/Ebullioscopic/DARG/issues",
        "Source": "https://github.com/Ebullioscopic/DARG",
        "Documentation": "https://github.com/Ebullioscopic/DARG/wiki",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "darg-demo=darg.examples.complete_demo:main",
            "darg-test=darg.testing.validation_suite:main",
            "darg-setup=darg.scripts.setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "darg": [
            "config/*.json",
            "data/*.txt",
            "examples/*.py",
        ],
    },
    zip_safe=False,
    keywords=[
        "vector search",
        "similarity search",
        "nearest neighbors",
        "machine learning",
        "deep learning",
        "graph database",
        "multi-modal",
        "DARG",
        "Neo4j",
    ],
)
