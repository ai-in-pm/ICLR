import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

class ICLRVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
        sns.set_style("whitegrid")  # Using seaborn's whitegrid style
        
    def plot_representations(self, pca_result, graph, title="Representation Space"):
        """Plot PCA visualization of representations with graph structure."""
        plt.figure(figsize=self.fig_size)
        
        # Plot points
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', alpha=0.6)
        
        # Plot edges
        pos = {i: pca_result[i] for i in range(len(pca_result))}
        nx.draw_networkx_edges(graph, pos, alpha=0.3)
        
        plt.title(title)
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        
        return plt.gcf()
    
    def plot_energy_evolution(self, energies, context_sizes):
        """Plot evolution of Dirichlet energy with context size."""
        plt.figure(figsize=self.fig_size)
        plt.plot(context_sizes, energies, marker='o')
        plt.title("Dirichlet Energy vs Context Size")
        plt.xlabel("Context Size")
        plt.ylabel("Dirichlet Energy")
        
        return plt.gcf()
    
    def plot_phase_transition(self, accuracies, context_sizes):
        """Visualize phase transition behavior in representation learning."""
        plt.figure(figsize=self.fig_size)
        plt.plot(context_sizes, accuracies, marker='o')
        
        # Add phase transition annotation
        transition_point = context_sizes[np.argmax(np.diff(accuracies))]
        plt.axvline(x=transition_point, color='r', linestyle='--', alpha=0.5)
        plt.text(transition_point, plt.ylim()[0], 'Phase Transition', rotation=90)
        
        plt.title("Phase Transition in Representation Learning")
        plt.xlabel("Context Size")
        plt.ylabel("Task Accuracy")
        
        return plt.gcf()
    
    def create_educational_visualization(self, agent_results):
        """Create comprehensive visualization with educational tooltips."""
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplot grid
        gs = plt.GridSpec(2, 2)
        
        # Representation space
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_representations(agent_results['pca_visualization'], 
                                agent_results['graph'])
        ax1.set_title("Semantic Representation Space")
        
        # Energy plot
        ax2 = fig.add_subplot(gs[0, 1])
        if 'energy_history' in agent_results:
            self.plot_energy_evolution(agent_results['energy_history'],
                                     range(len(agent_results['energy_history'])))
        
        # Add educational annotations
        ax1.annotate("Emergent Structure", 
                    xy=(0.5, 0.5),
                    xytext=(0.7, 0.7),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.tight_layout()
        return fig
