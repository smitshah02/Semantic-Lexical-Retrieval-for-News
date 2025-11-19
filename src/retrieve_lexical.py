"""
Lexical retrieval using TF-IDF and BM25.
"""
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRetriever:
    """TF-IDF based retrieval."""
    
    def __init__(self, artifacts_dir: str):
        """Load TF-IDF artifacts."""
        tfidf_path = Path(artifacts_dir) / 'tfidf.pkl'
        with open(tfidf_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.matrix = data['matrix']
        self.doc_ids = data['doc_ids']
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using TF-IDF cosine similarity.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        # Transform query
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.matrix).flatten()
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = [
            (self.doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0  # Only return non-zero scores
        ]
        
        return results

class BM25Retriever:
    """BM25 based retrieval."""
    
    def __init__(self, artifacts_dir: str):
        """Load BM25 artifacts."""
        bm25_path = Path(artifacts_dir) / 'bm25.pkl'
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.doc_ids = data['doc_ids']
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        # Tokenize query (same as corpus tokenization)
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # Only return non-zero scores
        ]
        
        return results

def load_tfidf_retriever(artifacts_dir: str) -> TFIDFRetriever:
    """Convenience function to load TF-IDF retriever."""
    return TFIDFRetriever(artifacts_dir)

def load_bm25_retriever(artifacts_dir: str) -> BM25Retriever:
    """Convenience function to load BM25 retriever."""
    return BM25Retriever(artifacts_dir)