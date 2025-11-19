"""
Semantic retrieval using sentence embeddings.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

class SemanticRetriever:
    """Semantic retrieval using embeddings."""
    
    def __init__(self, artifacts_dir: str, model_name: str = 'all-MiniLM-L6-v2', use_faiss: bool = False):
        """
        Load semantic retrieval artifacts.
        
        Args:
            artifacts_dir: Directory containing embeddings
            model_name: SentenceTransformer model name
            use_faiss: Whether to use FAISS for ANN search
        """
        # Load embeddings
        embeddings_path = Path(artifacts_dir) / 'embeddings.npy'
        self.embeddings = np.load(embeddings_path)
        
        # Load doc IDs
        doc_ids_path = Path(artifacts_dir) / 'doc_ids.json'
        with open(doc_ids_path, 'r') as f:
            self.doc_ids = json.load(f)
        
        # Load model for query encoding
        self.model = SentenceTransformer(model_name)
        
        # Optional: Load FAISS index
        self.use_faiss = use_faiss
        self.faiss_index = None
        
        if use_faiss:
            try:
                import faiss
                faiss_path = Path(artifacts_dir) / 'faiss.index'
                if faiss_path.exists():
                    self.faiss_index = faiss.read_index(str(faiss_path))
                else:
                    print(f"Warning: FAISS index not found at {faiss_path}, falling back to numpy")
                    self.use_faiss = False
            except ImportError:
                print("Warning: FAISS not available, using numpy for search")
                self.use_faiss = False
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        if self.use_faiss and self.faiss_index is not None:
            return self._search_faiss(query_embedding, k)
        else:
            return self._search_numpy(query_embedding, k)
    
    def _search_numpy(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search using numpy cosine similarity."""
        # Compute cosine similarities (dot product with normalized vectors)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = [
            (self.doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search using FAISS index."""
        # FAISS expects 2D array
        query_vec = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        scores, indices = self.faiss_index.search(query_vec, k)
        
        results = [
            (self.doc_ids[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1  # FAISS returns -1 for invalid results
        ]
        
        return results

def load_semantic_retriever(artifacts_dir: str, model_name: str = 'all-MiniLM-L6-v2', use_faiss: bool = False) -> SemanticRetriever:
    """Convenience function to load semantic retriever."""
    return SemanticRetriever(artifacts_dir, model_name, use_faiss)