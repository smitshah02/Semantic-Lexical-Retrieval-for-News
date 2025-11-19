"""
Hybrid retrieval combining lexical and semantic methods.
"""
from typing import List, Tuple, Dict
from collections import defaultdict

def reciprocal_rank_fusion(
    results_lists: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.
    
    RRF score = sum(1 / (k + rank)) for each appearance
    
    Args:
        results_lists: List of ranked result lists, each containing (doc_id, score) tuples
        k: RRF parameter (default 60 from literature)
    
    Returns:
        Combined ranked list of (doc_id, rrf_score) tuples
    """
    rrf_scores = defaultdict(float)
    
    for results in results_lists:
        for rank, (doc_id, _) in enumerate(results, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score
    combined = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return combined

def normalize_scores(results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Normalize scores to [0, 1] using min-max normalization.
    
    Args:
        results: List of (doc_id, score) tuples
    
    Returns:
        List with normalized scores
    """
    if not results:
        return results
    
    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        # All scores are the same
        return [(doc_id, 1.0) for doc_id, _ in results]
    
    normalized = [
        (doc_id, (score - min_score) / (max_score - min_score))
        for doc_id, score in results
    ]
    
    return normalized

def weighted_sum_fusion(
    results_lists: List[List[Tuple[str, float]]],
    weights: List[float] = None,
    normalize: bool = True
) -> List[Tuple[str, float]]:
    """
    Combine results using weighted sum of normalized scores.
    
    Args:
        results_lists: List of ranked result lists
        weights: Weight for each list (default: equal weights)
        normalize: Whether to normalize scores before combining
    
    Returns:
        Combined ranked list of (doc_id, score) tuples
    """
    if weights is None:
        weights = [1.0] * len(results_lists)
    
    if len(weights) != len(results_lists):
        raise ValueError("Number of weights must match number of result lists")
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    # Normalize scores if requested
    if normalize:
        results_lists = [normalize_scores(results) for results in results_lists]
    
    # Combine scores
    combined_scores = defaultdict(float)
    
    for weight, results in zip(weights, results_lists):
        for doc_id, score in results:
            combined_scores[doc_id] += weight * score
    
    # Sort by combined score
    combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return combined

class HybridRetriever:
    """Hybrid retrieval combining multiple methods."""
    
    def __init__(self, retrievers: Dict[str, object], fusion_method: str = 'rrf'):
        """
        Initialize hybrid retriever.
        
        Args:
            retrievers: Dictionary of {name: retriever_object}
            fusion_method: 'rrf' or 'weighted_sum'
        """
        self.retrievers = retrievers
        self.fusion_method = fusion_method
    
    def search(self, query: str, k: int = 10, weights: List[float] = None) -> List[Tuple[str, float]]:
        """
        Search using hybrid approach.
        
        Args:
            query: Search query
            k: Number of results to return
            weights: Optional weights for weighted_sum fusion
        
        Returns:
            Combined ranked list of (doc_id, score) tuples
        """
        # Get results from all retrievers
        all_results = []
        for name, retriever in self.retrievers.items():
            try:
                results = retriever.search(query, k=k*2)  # Retrieve more for fusion
                all_results.append(results)
            except Exception as e:
                print(f"Warning: Retriever '{name}' failed: {e}")
        
        if not all_results:
            return []
        
        # Combine results
        if self.fusion_method == 'rrf':
            combined = reciprocal_rank_fusion(all_results)
        elif self.fusion_method == 'weighted_sum':
            combined = weighted_sum_fusion(all_results, weights=weights)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Return top k
        return combined[:k]