"""
DARG v2.2 Configuration Module
Global configuration settings as per Blueprint.md specifications.
"""

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Module 1: Global Configuration
# ============================================================================

@dataclass
class GlobalConfig:
    """Global configuration for DARG v2.2"""
    # Grid parameters
    initial_grid_levels: int = 3
    base_max_pop_per_cell: int = 100
    min_pop_per_cell: int = 10
    max_grid_depth: int = 15
    
    # Search parameters
    beam_width_B: int = 5
    K_top_candidates: int = 50
    echo_trigger_threshold_S: float = 0.6
    echo_search_K_top_trigger: int = 10
    echo_search_N_echo: int = 3
    
    # PCA parameters
    pca_pop_trigger_for_split: int = 20
    proj_dimensions: int = 16
    pca_batch_size: int = 1000
    
    # LID parameters
    LID_influence_factor: float = 0.3
    LID_sample_size: int = 50
    Threshold_LID_Updates: int = 100  # As per Blueprint.md
    
    # Linkage cache parameters
    max_linkage_cache_size: int = 20
    epsilon_exploration_linkage: float = 0.1
    min_resonance_hit_rate_linkage: float = 0.7
    linkage_activation_decay_factor: float = 0.95
    
    # Echo trigger weights
    echo_trigger_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.3, 0.2])
    
    # Maintenance parameters
    maintenance_interval_seconds: int = 300
    updates_threshold_for_pca: int = 50
    Threshold_LID_Updates: int = 100
    
    @classmethod
    def from_file(cls, filepath: str) -> 'GlobalConfig':
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            logger.warning(f"Config file {filepath} not found. Using defaults.")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config file {filepath}: {e}")
            return cls()