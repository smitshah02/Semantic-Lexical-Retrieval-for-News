"""
Semantic News Search - Source Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import key components for easier access
from .retrieve_lexical import TFIDFRetriever, BM25Retriever
from .retrieve_semantic import SemanticRetriever
from .retrieve_hybrid import HybridRetriever

__all__ = [
    'TFIDFRetriever',
    'BM25Retriever', 
    'SemanticRetriever',
    'HybridRetriever'
]