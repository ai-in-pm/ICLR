import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.iclr_agent import ICLRAgent
from src.visualization.visualizer import ICLRVisualizer
from src.visualization.advanced_visualizer import AdvancedVisualizer
from src.metrics.performance_metrics import PerformanceMetrics
from src.analysis.phase_transition import PhaseTransitionAnalyzer
from src.analysis.semantic_priors import SemanticPriorAnalyzer
import matplotlib.pyplot as plt
import numpy as np

def run_demo():
    # Initialize components
    agent = ICLRAgent()
    visualizer = ICLRVisualizer()
    adv_visualizer = AdvancedVisualizer()
    metrics = PerformanceMetrics()
    phase_analyzer = PhaseTransitionAnalyzer()
    semantic_analyzer = SemanticPriorAnalyzer()
    
    print("Running ICLR Agent demonstration...")
    
    # Test different graph types with more context sizes
    graph_types = ['grid', 'ring']
    context_sizes = np.array([4, 9, 16, 25, 36, 49, 64, 81, 100])
    
    for graph_type in graph_types:
        print(f"\nAnalyzing {graph_type} graph structure:")
        
        order_params = []
        all_representations = []
        
        for context_size in context_sizes:
            print(f"  Processing context size: {context_size}")
            
            # Run graph tracing task
            results = agent.trace_graph(graph_type=graph_type, 
                                      context_size=context_size)
            
            # Collect data for phase transition analysis
            order_param = phase_analyzer.compute_order_parameter(
                results['representations'],
                results['graph']
            )
            order_params.append(order_param)
            all_representations.append(results['representations'])
            
            # Generate visualizations
            save_dir = os.path.join(os.path.dirname(__file__), 'outputs', graph_type)
            os.makedirs(save_dir, exist_ok=True)
            
            # Basic representation plot
            fig = visualizer.plot_representations(
                results['pca_visualization'],
                results['graph'],
                title=f"{graph_type.capitalize()} Graph (n={context_size})"
            )
            fig.savefig(os.path.join(save_dir, f'basic_vis_n{context_size}.png'))
            plt.close(fig)
            
            # Advanced visualizations
            fig = adv_visualizer.plot_representation_landscape(
                results['representations'],
                results['graph'],
                title=f"{graph_type.capitalize()} Landscape (n={context_size})"
            )
            fig.savefig(os.path.join(save_dir, f'landscape_n{context_size}.png'))
            plt.close(fig)
        
        # Phase transition analysis
        critical_analysis = phase_analyzer.analyze_critical_behavior(
            context_sizes, np.array(order_params)
        )
        
        fig = phase_analyzer.plot_phase_transition(
            context_sizes, np.array(order_params), critical_analysis
        )
        fig.savefig(os.path.join(save_dir, 'phase_transition.png'))
        plt.close(fig)
        
        print(f"\nPhase Transition Analysis for {graph_type}:")
        print(f"  Transition point: {critical_analysis['transition_point']}")
        print(f"  Pre-transition exponent: {critical_analysis['pre_transition_exponent']:.3f}")
        print(f"  Post-transition exponent: {critical_analysis['post_transition_exponent']:.3f}")
        
        # Print semantic analysis
        print("\nSemantic Analysis:")
        print(results['semantic_analysis'])
    
    # Semantic prior analysis
    print("\nAnalyzing semantic priors...")
    for prior_type in ['weekdays', 'months', 'numbers']:
        prior_embeddings, labels = semantic_analyzer.generate_prior_embeddings(prior_type)
        results = agent.trace_graph(graph_type='ring', context_size=len(prior_embeddings))
        
        influence = semantic_analyzer.compute_prior_influence(
            results['representations'], prior_embeddings
        )
        
        print(f"\n{prior_type.capitalize()} Prior Analysis:")
        print(f"  Correlation: {influence['correlation']:.3f}")
        print(f"  P-value: {influence['p_value']:.3f}")
        print("\nLLM Analysis:")
        if 'analysis' in results:
            print(results['analysis'])
        
        fig = semantic_analyzer.visualize_prior_comparison(
            results['representations'], prior_embeddings, labels
        )
        fig.savefig(os.path.join(os.path.dirname(__file__), 
                                'outputs', f'semantic_prior_{prior_type}.png'))
        plt.close(fig)

if __name__ == "__main__":
    run_demo()
