"""
Evaluation script for retrieval systems.
Usage: python src/evaluate.py --queries data/queries.tsv --qrels data/qrels.tsv --artifacts_dir artifacts --modes tfidf bm25 semantic hybrid
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Set, Dict
from utils import setup_logging, load_queries, load_qrels
from retrieve_lexical import load_tfidf_retriever, load_bm25_retriever
from retrieve_semantic import load_semantic_retriever
from retrieve_hybrid import HybridRetriever

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Precision@k."""
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return relevant_retrieved / k if k > 0 else 0.0

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Recall@k."""
    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return relevant_retrieved / len(relevant)

def dcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate DCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = 1 if doc_id in relevant else 0
        dcg += rel / np.log2(i + 1)
    return dcg

def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate nDCG@k."""
    dcg = dcg_at_k(retrieved, relevant, k)
    
    # Ideal DCG: all relevant docs at the top
    ideal_retrieved = list(relevant) + [''] * k
    idcg = dcg_at_k(ideal_retrieved, relevant, k)
    
    return dcg / idcg if idcg > 0 else 0.0

def extract_doc_id(chunk_id: str) -> str:
    """Extract base document ID from chunk ID (e.g., 'doc_0' from 'doc_0_1')."""
    # Chunk IDs are format: <doc_id>_<chunk_index>
    # We want to match against base doc_id
    parts = chunk_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return chunk_id

def evaluate_retriever(
    retriever,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    k_values: List[int]
) -> Dict[str, float]:
    """
    Evaluate a retriever on all queries.
    
    Returns:
        Dictionary of {metric_name: average_value}
    """
    metrics = {f'P@{k}': [] for k in k_values}
    metrics.update({f'R@{k}': [] for k in k_values})
    metrics.update({f'nDCG@{k}': [] for k in k_values})
    
    for _, query_row in queries_df.iterrows():
        qid = query_row['qid']
        query = query_row['query']
        
        # Get relevant docs for this query
        relevant_docs = set(qrels_df[qrels_df['qid'] == qid]['doc_id'].values)
        
        if not relevant_docs:
            continue
        
        # Retrieve results
        try:
            results = retriever.search(query, k=max(k_values))
            retrieved_chunk_ids = [doc_id for doc_id, _ in results]
            
            # Extract base doc IDs from chunks
            retrieved_doc_ids = [extract_doc_id(chunk_id) for chunk_id in retrieved_chunk_ids]
            
            # Calculate metrics for each k
            for k in k_values:
                p = precision_at_k(retrieved_doc_ids, relevant_docs, k)
                r = recall_at_k(retrieved_doc_ids, relevant_docs, k)
                n = ndcg_at_k(retrieved_doc_ids, relevant_docs, k)
                
                metrics[f'P@{k}'].append(p)
                metrics[f'R@{k}'].append(r)
                metrics[f'nDCG@{k}'].append(n)
        
        except Exception as e:
            logging.error(f"Error evaluating query {qid}: {e}")
            continue
    
    # Average metrics
    avg_metrics = {}
    for metric_name, values in metrics.items():
        if values:
            avg_metrics[metric_name] = np.mean(values)
        else:
            avg_metrics[metric_name] = 0.0
    
    return avg_metrics

def run_evaluation(
    queries_path: str,
    qrels_path: str,
    artifacts_dir: str,
    modes: List[str],
    k_values: List[int]
):
    """Run complete evaluation."""
    setup_logging()
    
    # Load queries and qrels
    logging.info(f"Loading queries from {queries_path}")
    queries_df = load_queries(queries_path)
    logging.info(f"Loaded {len(queries_df)} queries")
    
    logging.info(f"Loading qrels from {qrels_path}")
    qrels_df = load_qrels(qrels_path)
    logging.info(f"Loaded {len(qrels_df)} relevance judgments")
    
    # Initialize retrievers
    retrievers = {}
    
    if 'tfidf' in modes:
        logging.info("Loading TF-IDF retriever...")
        retrievers['tfidf'] = load_tfidf_retriever(artifacts_dir)
    
    if 'bm25' in modes:
        logging.info("Loading BM25 retriever...")
        retrievers['bm25'] = load_bm25_retriever(artifacts_dir)
    
    if 'semantic' in modes:
        logging.info("Loading semantic retriever...")
        retrievers['semantic'] = load_semantic_retriever(artifacts_dir, use_faiss=True)
    
    if 'hybrid' in modes:
        logging.info("Loading hybrid retriever...")
        # Build hybrid from available retrievers
        hybrid_retrievers = {}
        if 'tfidf' not in retrievers:
            hybrid_retrievers['tfidf'] = load_tfidf_retriever(artifacts_dir)
        else:
            hybrid_retrievers['tfidf'] = retrievers['tfidf']
        
        if 'bm25' not in retrievers:
            hybrid_retrievers['bm25'] = load_bm25_retriever(artifacts_dir)
        else:
            hybrid_retrievers['bm25'] = retrievers['bm25']
        
        if 'semantic' not in retrievers:
            hybrid_retrievers['semantic'] = load_semantic_retriever(artifacts_dir, use_faiss=True)
        else:
            hybrid_retrievers['semantic'] = retrievers['semantic']
        
        retrievers['hybrid'] = HybridRetriever(hybrid_retrievers, fusion_method='rrf')
    
    # Evaluate each retriever
    results = []
    
    for mode, retriever in retrievers.items():
        logging.info(f"Evaluating {mode}...")
        metrics = evaluate_retriever(retriever, queries_df, qrels_df, k_values)
        
        result_row = {'Method': mode}
        result_row.update(metrics)
        results.append(result_row)
        
        # Print results
        logging.info(f"Results for {mode}:")
        for metric_name, value in metrics.items():
            logging.info(f"  {metric_name}: {value:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print comparison table
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(artifacts_dir) / f'eval_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    logging.info(f"Saved evaluation results to {results_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval systems")
    parser.add_argument('--queries', required=True, help="Path to queries TSV")
    parser.add_argument('--qrels', required=True, help="Path to qrels TSV")
    parser.add_argument('--artifacts_dir', required=True, help="Directory with retrieval artifacts")
    parser.add_argument('--modes', nargs='+', default=['tfidf', 'bm25', 'semantic', 'hybrid'],
                        help="Retrieval modes to evaluate")
    parser.add_argument('--k', nargs='+', type=int, default=[5, 10],
                        help="k values for metrics")
    args = parser.parse_args()
    
    run_evaluation(
        args.queries,
        args.qrels,
        args.artifacts_dir,
        args.modes,
        args.k
    )

if __name__ == '__main__':
    main()