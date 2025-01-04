import numpy as np
from sklearn.manifold import MDS
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

class SemanticPriorAnalyzer:
    def __init__(self):
        self.prior_types = {
            'weekdays': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'months': ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December'],
            'numbers': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        }
        
    def generate_prior_embeddings(self, prior_type, embedding_dim=256):
        """Generate structured embeddings for different semantic priors."""
        items = self.prior_types.get(prior_type, [])
        n_items = len(items)
        
        if prior_type in ['weekdays', 'months']:
            # Circular structure
            angle = np.linspace(0, 2*np.pi, n_items, endpoint=False)
            base_embedding = np.column_stack([np.cos(angle), np.sin(angle)])
        elif prior_type == 'numbers':
            # Linear structure with logarithmic scaling
            base_embedding = np.column_stack([
                np.log(1 + np.arange(n_items)),
                np.zeros(n_items)
            ])
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
            
        # Pad to full embedding dimension
        embeddings = np.pad(base_embedding, ((0, 0), (0, embedding_dim-2)))
        
        # Add structured noise to remaining dimensions
        noise = np.random.randn(n_items, embedding_dim-2) * 0.1
        embeddings[:, 2:] = noise
        
        return embeddings, items
        
    def compute_prior_influence(self, learned_representations, prior_embeddings):
        """Compute the influence of semantic priors on learned representations."""
        # Compute correlation between distance matrices
        learned_distances = np.array([[np.linalg.norm(r1 - r2) for r2 in learned_representations] 
                                    for r1 in learned_representations])
        prior_distances = np.array([[np.linalg.norm(p1 - p2) for p2 in prior_embeddings] 
                                  for p1 in prior_embeddings])
        
        correlation, p_value = spearmanr(learned_distances.flatten(), prior_distances.flatten())
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'learned_distances': learned_distances,
            'prior_distances': prior_distances
        }
        
    def visualize_prior_comparison(self, learned_representations, prior_embeddings, labels):
        """Create visualization comparing learned representations with prior structure."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MDS for dimensionality reduction
        mds = MDS(n_components=2, random_state=42)
        learned_2d = mds.fit_transform(learned_representations)
        prior_2d = mds.fit_transform(prior_embeddings)
        
        # Plot learned representations
        sns.scatterplot(x=learned_2d[:, 0], y=learned_2d[:, 1], ax=ax1)
        for i, label in enumerate(labels):
            ax1.annotate(label, (learned_2d[i, 0], learned_2d[i, 1]))
        ax1.set_title('Learned Representations')
        
        # Plot prior structure
        sns.scatterplot(x=prior_2d[:, 0], y=prior_2d[:, 1], ax=ax2)
        for i, label in enumerate(labels):
            ax2.annotate(label, (prior_2d[i, 0], prior_2d[i, 1]))
        ax2.set_title('Prior Structure')
        
        plt.tight_layout()
        return fig
