"""
DARG - Dynamic Adaptive Resonance Grids
Universal Multi-Modal Vector Search System

Core modules for DARG implementation including:
- Universal DARG for multi-modal data handling
- Enhanced DARG with advanced algorithms
- Visualization and testing utilities
"""

__version__ = "1.0.0"
__author__ = "DARG Research Team"

try:
    from .core.universal_darg import UniversalDARG
    from .core.enhanced_darg import EnhancedDARG
    __all__ = ['UniversalDARG', 'EnhancedDARG']
except ImportError:
    # Graceful fallback for development
    __all__ = []
