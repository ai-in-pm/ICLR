import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

class LLMInterface:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4-1106-preview"  # Using the latest GPT-4 model
        
    def get_embedding(self, text):
        """Get embedding for a given text using OpenAI's embedding model."""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def analyze_representation(self, representation, context):
        """Analyze a representation using GPT-4."""
        # Convert representation to a more readable format
        rep_summary = {
            'dimensions': len(representation),
            'mean': float(np.mean(representation)),
            'std': float(np.std(representation)),
            'principal_components': self._get_principal_components(representation)
        }
        
        prompt = f"""Analyze this neural representation in the context of in-context learning:
Context: {context}
Representation Statistics: {rep_summary}

Please provide insights on:
1. How this representation aligns with the given context
2. What patterns or structures are evident
3. Potential implications for in-context learning

Analysis:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def generate_semantic_description(self, representations, graph_type, context_size):
        """Generate a semantic description of the learned representations."""
        stats = {
            'mean_norm': float(np.mean([np.linalg.norm(r) for r in representations])),
            'std_norm': float(np.std([np.linalg.norm(r) for r in representations])),
            'dimensionality': representations.shape[1],
            'num_points': len(representations)
        }
        
        prompt = f"""Analyze these learned representations from an in-context learning task:
Graph Type: {graph_type}
Context Size: {context_size}
Statistics: {stats}

Please describe:
1. The emergent structure in the representations
2. How the graph type might influence the organization
3. Any phase transition indicators at this context size

Analysis:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _get_principal_components(self, representation, n_components=3):
        """Get the first few principal components of a representation."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, len(representation)))
        if len(representation.shape) == 1:
            representation = representation.reshape(1, -1)
        transformed = pca.fit_transform(representation)
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'components': transformed.tolist()
        }
