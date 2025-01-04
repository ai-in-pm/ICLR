import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns

class PhaseTransitionAnalyzer:
    def __init__(self):
        self.transition_points = {}
        self.order_parameters = {}
        
    def compute_order_parameter(self, representations, graph):
        """Compute order parameter based on mutual information between representations."""
        # Compute pairwise distances in representation space
        rep_distances = np.array([[np.linalg.norm(r1 - r2) for r2 in representations] 
                                for r1 in representations])
        
        # Normalize distances
        rep_distances = (rep_distances - rep_distances.min()) / (rep_distances.max() - rep_distances.min())
        
        # Convert graph to adjacency matrix
        adj_matrix = np.array([[1 if (i, j) in graph.edges() or (j, i) in graph.edges() else 0
                              for j in range(len(representations))]
                             for i in range(len(representations))])
        
        # Compute mutual information between representation distances and graph structure
        mi = mutual_info_score(rep_distances.flatten(), adj_matrix.flatten())
        
        return mi
        
    def detect_phase_transition(self, context_sizes, order_params):
        """Detect phase transition point using maximum derivative."""
        derivatives = np.gradient(order_params)
        transition_point = context_sizes[np.argmax(np.abs(derivatives))]
        transition_strength = np.max(np.abs(derivatives))
        
        return {
            'transition_point': transition_point,
            'transition_strength': transition_strength,
            'derivatives': derivatives
        }
        
    def analyze_critical_behavior(self, context_sizes, order_params):
        """Analyze critical behavior near the phase transition."""
        # Detect transition point
        transition_results = self.detect_phase_transition(context_sizes, order_params)
        transition_idx = np.where(context_sizes == transition_results['transition_point'])[0][0]
        
        # Compute critical exponents
        pre_transition = np.polyfit(
            np.log(context_sizes[:transition_idx+1]), 
            np.log(order_params[:transition_idx+1]), 
            1
        )
        post_transition = np.polyfit(
            np.log(context_sizes[transition_idx:]), 
            np.log(order_params[transition_idx:]), 
            1
        )
        
        return {
            'pre_transition_exponent': pre_transition[0],
            'post_transition_exponent': post_transition[0],
            'transition_point': transition_results['transition_point'],
            'transition_strength': transition_results['transition_strength']
        }
        
    def plot_phase_transition(self, context_sizes, order_params, critical_analysis):
        """Create comprehensive phase transition visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot order parameter
        sns.scatterplot(x=context_sizes, y=order_params, ax=ax1)
        ax1.axvline(x=critical_analysis['transition_point'], color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Context Size')
        ax1.set_ylabel('Order Parameter (MI)')
        ax1.set_title('Phase Transition in Representation Learning')
        
        # Plot critical behavior
        sns.lineplot(x=np.log(context_sizes), y=np.log(order_params), ax=ax2)
        ax2.set_xlabel('log(Context Size)')
        ax2.set_ylabel('log(Order Parameter)')
        ax2.set_title('Critical Scaling Behavior')
        
        plt.tight_layout()
        return fig
