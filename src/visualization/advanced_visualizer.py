import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

class AdvancedVisualizer:
    def __init__(self):
        self.fig_size = (15, 10)
        sns.set_style("whitegrid")
        
    def compute_pca(self, data, n_components=3):
        """Simple PCA implementation using numpy."""
        # Center the data
        centered_data = data - np.mean(data, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.dot(centered_data.T, centered_data) / (data.shape[0] - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project data onto principal components
        return np.dot(centered_data, eigenvectors[:, :n_components])
        
    def plot_representation_landscape(self, representations, graph, title="Representation Landscape"):
        """Create a landscape visualization of the representation space."""
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Use PCA for 3D visualization
        coords_3d = self.compute_pca(representations, n_components=3)
        
        # Create surface plot
        x = coords_3d[:, 0]
        y = coords_3d[:, 1]
        z = coords_3d[:, 2]
        
        # Plot surface
        surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Plot edges
        for edge in graph.edges():
            i, j = edge
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k-', alpha=0.3)
        
        ax.set_title(title)
        plt.colorbar(surf)
        
        return fig
    
    def plot_energy_landscape(self, representations, graph, title="Energy Landscape"):
        """Visualize the energy landscape of representations."""
        fig = plt.figure(figsize=self.fig_size)
        
        # Use PCA for 2D visualization
        coords_2d = self.compute_pca(representations, n_components=2)
        
        # Create grid for energy landscape
        x = np.linspace(coords_2d[:, 0].min(), coords_2d[:, 0].max(), 100)
        y = np.linspace(coords_2d[:, 1].min(), coords_2d[:, 1].max(), 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute energy at each point (simplified)
        Z = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(Y)):
                point = np.array([X[i,j], Y[i,j]])
                energy = 0
                for edge in graph.edges():
                    p1, p2 = coords_2d[edge[0]], coords_2d[edge[1]]
                    energy += np.sum((point - p1)**2) + np.sum((point - p2)**2)
                Z[i,j] = energy
        
        plt.contour(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='Energy')
        plt.title(title)
        
        return fig
    
    def plot_representation_flow(self, initial_reps, final_reps, title="Representation Flow"):
        """Visualize how representations evolve during learning."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Use PCA for 2D visualization
        initial_2d = self.compute_pca(initial_reps, n_components=2)
        final_2d = self.compute_pca(final_reps, n_components=2)
        
        # Plot points and arrows
        ax.scatter(initial_2d[:, 0], initial_2d[:, 1], c='blue', alpha=0.5, label='Initial')
        ax.scatter(final_2d[:, 0], final_2d[:, 1], c='red', alpha=0.5, label='Final')
        
        # Draw arrows showing evolution
        for i in range(len(initial_2d)):
            ax.arrow(initial_2d[i,0], initial_2d[i,1],
                    final_2d[i,0] - initial_2d[i,0],
                    final_2d[i,1] - initial_2d[i,1],
                    head_width=0.05, head_length=0.1, fc='k', ec='k', alpha=0.3)
        
        plt.title(title)
        plt.legend()
        
        return fig
    
    def plot_attention_patterns(self, attention_weights, labels=None, title="Attention Patterns"):
        """Visualize attention patterns between representations."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        sns.heatmap(attention_weights, annot=True, cmap='viridis', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        
        return fig
