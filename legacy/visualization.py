#!/usr/bin/env python3
"""
DARG Visualization Module
Creates performance visualizations and analysis plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class DARGVisualizer:
    """Visualization toolkit for DARG performance analysis"""
    
    def __init__(self, output_dir: str = "./plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib for high-quality output
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11
        })
    
    def plot_performance_comparison(self, results: Dict[str, Dict[str, float]], 
                                  save_name: str = "performance_comparison") -> None:
        """Plot performance comparison between different methods"""
        
        # Prepare data
        methods = list(results.keys())
        metrics = ['latency_ms', 'recall_10', 'memory_mb', 'qps']
        metric_labels = ['Latency (ms)', 'Recall@10 (%)', 'Memory (MB)', 'QPS']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            
            values = [results[method].get(metric, 0) for method in methods]
            colors = sns.color_palette("husl", len(methods))
            
            bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight DARG if present
            if 'DARG' in methods:
                darg_idx = methods.index('DARG')
                bars[darg_idx].set_color('red')
                bars[darg_idx].set_alpha(1.0)
        
        plt.tight_layout()
        self._save_plot(save_name)
    
    def plot_scalability_analysis(self, dataset_sizes: List[int], 
                                 latencies: List[float], 
                                 memory_usage: List[float],
                                 save_name: str = "scalability_analysis") -> None:
        """Plot scalability analysis"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency scaling
        ax1.loglog(dataset_sizes, latencies, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Dataset Size (Million Vectors)')
        ax1.set_ylabel('Query Latency (ms)')
        ax1.set_title('Latency Scaling', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(np.log(dataset_sizes), np.log(latencies), 1)
        p = np.poly1d(z)
        ax1.loglog(dataset_sizes, np.exp(p(np.log(dataset_sizes))), "--", alpha=0.8, color='red')
        ax1.text(0.05, 0.95, f'Slope: {z[0]:.2f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # Memory scaling
        ax2.loglog(dataset_sizes, memory_usage, 's-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Dataset Size (Million Vectors)')
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.set_title('Memory Scaling', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z2 = np.polyfit(np.log(dataset_sizes), np.log(memory_usage), 1)
        p2 = np.poly1d(z2)
        ax2.loglog(dataset_sizes, np.exp(p2(np.log(dataset_sizes))), "--", alpha=0.8, color='red')
        ax2.text(0.05, 0.95, f'Slope: {z2[0]:.2f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        self._save_plot(save_name)
    
    def plot_recall_vs_latency(self, methods_data: Dict[str, List[Tuple[float, float]]],
                              save_name: str = "recall_vs_latency") -> None:
        """Plot recall vs latency trade-off curves"""
        
        plt.figure(figsize=(12, 8))
        
        colors = sns.color_palette("husl", len(methods_data))
        
        for i, (method, data) in enumerate(methods_data.items()):
            latencies, recalls = zip(*data)
            plt.plot(latencies, recalls, 'o-', linewidth=2, markersize=6, 
                    label=method, color=colors[i])
        
        plt.xlabel('Query Latency (ms)')
        plt.ylabel('Recall@10 (%)')
        plt.title('Recall vs Latency Trade-off', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add target regions
        plt.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% Recall Target')
        plt.axvline(x=1000, color='orange', linestyle='--', alpha=0.5, label='1s Latency Target')
        
        # Highlight the "sweet spot"
        plt.fill_between([0, 1000], [90, 90], [100, 100], alpha=0.1, color='green', 
                        label='Target Region')
        
        plt.legend()
        self._save_plot(save_name)
    
    def plot_operation_timing_breakdown(self, timing_stats: Dict[str, Dict[str, float]],
                                       save_name: str = "timing_breakdown") -> None:
        """Plot detailed operation timing breakdown"""
        
        # Prepare data for hierarchical visualization
        categories = {
            'Search Operations': ['search_k_nearest', 'phase1_grid_resonance', 
                                'phase2_localized_refinement', 'phase3_echo_search'],
            'Update Operations': ['insert_point', 'delete_point_update', 'update_cell_representative'],
            'Grid Operations': ['find_leaf_cell', 'split_cell', 'compute_cell_representative'],
            'Vector Operations': ['distance_calculation', 'project_vector', 'estimate_lid_two_nn'],
            'Cache Operations': ['explore_linkage_cache', 'initialize_linkage_cache']
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (category, operations) in enumerate(categories.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Filter operations that exist in timing_stats
            valid_ops = [op for op in operations if op in timing_stats]
            if not valid_ops:
                ax.set_visible(False)
                continue
            
            # Get total times
            total_times = [timing_stats[op]['total_time'] for op in valid_ops]
            avg_times = [timing_stats[op]['avg_time'] for op in valid_ops]
            
            # Create pie chart for total time distribution
            ax.pie(total_times, labels=valid_ops, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{category}\nTotal Time Distribution', fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(categories), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self._save_plot(save_name)
    
    def plot_search_quality_analysis(self, search_results: List[Dict[str, Any]],
                                    save_name: str = "search_quality") -> None:
        """Plot search quality analysis"""
        
        # Extract metrics
        k_values = [r['k'] for r in search_results]
        recalls = [r['recall'] for r in search_results]
        precisions = [r.get('precision', 0) for r in search_results]
        latencies = [r['latency_ms'] for r in search_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Recall vs k
        ax1.plot(k_values, recalls, 'o-', linewidth=2, markersize=6, color='blue')
        ax1.set_xlabel('k (Number of Results)')
        ax1.set_ylabel('Recall (%)')
        ax1.set_title('Recall vs k', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Precision vs k
        ax2.plot(k_values, precisions, 's-', linewidth=2, markersize=6, color='green')
        ax2.set_xlabel('k (Number of Results)')
        ax2.set_ylabel('Precision (%)')
        ax2.set_title('Precision vs k', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Latency vs k
        ax3.plot(k_values, latencies, '^-', linewidth=2, markersize=6, color='red')
        ax3.set_xlabel('k (Number of Results)')
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Latency vs k', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Combined efficiency score
        efficiency_scores = [r * 1000 / l for r, l in zip(recalls, latencies)]
        ax4.plot(k_values, efficiency_scores, 'D-', linewidth=2, markersize=6, color='purple')
        ax4.set_xlabel('k (Number of Results)')
        ax4.set_ylabel('Efficiency Score (Recall/ms)')
        ax4.set_title('Search Efficiency', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(save_name)
    
    def plot_parameter_sensitivity(self, param_analysis: Dict[str, Dict[str, List[float]]],
                                  save_name: str = "parameter_sensitivity") -> None:
        """Plot parameter sensitivity analysis"""
        
        n_params = len(param_analysis)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 6))
        
        if n_params == 1:
            axes = [axes]
        
        for i, (param_name, data) in enumerate(param_analysis.items()):
            ax = axes[i]
            
            param_values = data['values']
            recalls = data['recalls']
            latencies = data['latencies']
            
            # Dual y-axis plot
            ax2 = ax.twinx()
            
            line1 = ax.plot(param_values, recalls, 'b-o', label='Recall@10', linewidth=2)
            line2 = ax2.plot(param_values, latencies, 'r-s', label='Latency (ms)', linewidth=2)
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Recall@10 (%)', color='b')
            ax2.set_ylabel('Latency (ms)', color='r')
            ax.set_title(f'{param_name} Sensitivity', fontweight='bold')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(save_name)
    
    def plot_dataset_characteristics(self, dataset_analysis: Dict[str, Any],
                                   save_name: str = "dataset_characteristics") -> None:
        """Plot dataset characteristics analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Dimension histogram
        if 'dimensions' in dataset_analysis:
            dims = dataset_analysis['dimensions']
            ax1.hist(dims, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Vector Dimensions')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Dimension Distribution', fontweight='bold')
        
        # Distance distribution
        if 'distances' in dataset_analysis:
            distances = dataset_analysis['distances']
            ax2.hist(distances, bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_xlabel('Pairwise Distances')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distance Distribution', fontweight='bold')
        
        # LID estimates
        if 'lid_estimates' in dataset_analysis:
            lid_est = dataset_analysis['lid_estimates']
            ax3.hist(lid_est, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_xlabel('Local Intrinsic Dimensionality')
            ax3.set_ylabel('Frequency')
            ax3.set_title('LID Distribution', fontweight='bold')
        
        # Cluster analysis
        if 'cluster_sizes' in dataset_analysis:
            cluster_sizes = dataset_analysis['cluster_sizes']
            ax4.bar(range(len(cluster_sizes)), sorted(cluster_sizes, reverse=True), 
                   color='plum', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Cluster Rank')
            ax4.set_ylabel('Cluster Size')
            ax4.set_title('Cluster Size Distribution', fontweight='bold')
        
        plt.tight_layout()
        self._save_plot(save_name)
    
    def create_performance_dashboard(self, comprehensive_results: Dict[str, Any],
                                   save_name: str = "performance_dashboard") -> None:
        """Create comprehensive performance dashboard"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Performance comparison (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'comparison' in comprehensive_results:
            methods = list(comprehensive_results['comparison'].keys())
            latencies = [comprehensive_results['comparison'][m]['latency_ms'] for m in methods]
            recalls = [comprehensive_results['comparison'][m]['recall_10'] for m in methods]
            
            ax1.scatter(latencies, recalls, s=100, alpha=0.7)
            for i, method in enumerate(methods):
                ax1.annotate(method, (latencies[i], recalls[i]), xytext=(5, 5), 
                           textcoords='offset points')
            ax1.set_xlabel('Latency (ms)')
            ax1.set_ylabel('Recall@10 (%)')
            ax1.set_title('Method Comparison', fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # 2. Scalability (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'scalability' in comprehensive_results:
            sizes = comprehensive_results['scalability']['sizes']
            latencies = comprehensive_results['scalability']['latencies']
            ax2.loglog(sizes, latencies, 'o-', linewidth=2)
            ax2.set_xlabel('Dataset Size')
            ax2.set_ylabel('Latency (ms)')
            ax2.set_title('Scalability Analysis', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. Timing breakdown (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'timing' in comprehensive_results:
            operations = list(comprehensive_results['timing'].keys())[:8]  # Top 8
            times = [comprehensive_results['timing'][op]['total_time'] for op in operations]
            ax3.barh(operations, times, color=sns.color_palette("viridis", len(operations)))
            ax3.set_xlabel('Total Time (s)')
            ax3.set_title('Operation Timing Breakdown', fontweight='bold')
        
        # 4. Memory usage (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'memory' in comprehensive_results:
            components = list(comprehensive_results['memory'].keys())
            usage = list(comprehensive_results['memory'].values())
            ax4.pie(usage, labels=components, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Memory Usage Distribution', fontweight='bold')
        
        # 5. Search quality (bottom-left)
        ax5 = fig.add_subplot(gs[2, :2])
        if 'search_quality' in comprehensive_results:
            k_values = comprehensive_results['search_quality']['k_values']
            recalls = comprehensive_results['search_quality']['recalls']
            ax5.plot(k_values, recalls, 'o-', linewidth=2, color='green')
            ax5.set_xlabel('k')
            ax5.set_ylabel('Recall (%)')
            ax5.set_title('Search Quality vs k', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. Parameter sensitivity (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2:])
        if 'parameter_sensitivity' in comprehensive_results:
            param_data = comprehensive_results['parameter_sensitivity']
            for param_name, data in param_data.items():
                ax6.plot(data['values'], data['recalls'], 'o-', label=param_name, linewidth=2)
            ax6.set_xlabel('Parameter Value (Normalized)')
            ax6.set_ylabel('Recall@10 (%)')
            ax6.set_title('Parameter Sensitivity', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. System metrics (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        if 'system_metrics' in comprehensive_results:
            metrics = comprehensive_results['system_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax7.bar(metric_names, metric_values, color=sns.color_palette("husl", len(metric_names)))
            ax7.set_ylabel('Value')
            ax7.set_title('System Performance Metrics', fontweight='bold')
            ax7.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.suptitle('DARG Performance Dashboard', fontsize=20, fontweight='bold', y=0.98)
        self._save_plot(save_name)
    
    def _save_plot(self, filename: str) -> None:
        """Save plot with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        full_path = self.output_dir / f"{filename}_{timestamp}.png"
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {full_path}")
        plt.show()

# Convenience functions
def create_visualizer(output_dir: str = "./plots") -> DARGVisualizer:
    """Create a DARG visualizer"""
    return DARGVisualizer(output_dir)

def quick_performance_plot(results: Dict[str, Dict[str, float]]) -> None:
    """Quick performance comparison plot"""
    viz = DARGVisualizer()
    viz.plot_performance_comparison(results, "quick_comparison")

if __name__ == "__main__":
    # Example usage
    print("DARG Visualization Module")
    print("=" * 40)
    
    # Create sample data for demonstration
    sample_results = {
        'DARG': {'latency_ms': 1.4, 'recall_10': 94.3, 'memory_mb': 260, 'qps': 1050},
        'HNSW': {'latency_ms': 1.5, 'recall_10': 94.2, 'memory_mb': 310, 'qps': 990},
        'IVF-PQ': {'latency_ms': 2.0, 'recall_10': 88.5, 'memory_mb': 180, 'qps': 830},
        'ScaNN': {'latency_ms': 1.7, 'recall_10': 93.4, 'memory_mb': 220, 'qps': 900}
    }
    
    # Create visualizer and demo plot
    viz = DARGVisualizer()
    viz.plot_performance_comparison(sample_results, "demo_comparison")
    
    print("Demo plot created in ./plots/ directory")
