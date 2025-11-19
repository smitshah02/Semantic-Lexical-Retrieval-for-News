"""
Create sample queries and qrels for testing.
Run this after building the corpus to generate test evaluation data.

Usage: python scripts/create_sample_data.py --corpus artifacts/bbc_corpus.csv --output_dir data
"""
import argparse
import pandas as pd
from pathlib import Path
import random

def create_sample_queries_qrels(corpus_csv: str, output_dir: str, num_queries: int = 20):
    """
    Create sample queries and relevance judgments from corpus.
    
    This creates synthetic evaluation data by:
    1. Sampling documents from different categories
    2. Extracting keywords from titles
    3. Creating relevance judgments
    """
    print(f"Loading corpus from {corpus_csv}")
    df = pd.read_csv(corpus_csv)
    
    print(f"Corpus has {len(df)} documents across categories:")
    print(df['category'].value_counts())
    
    queries = []
    qrels = []
    qid = 1
    
    # Sample documents from each category
    categories = df['category'].unique()
    
    for category in categories:
        category_docs = df[df['category'] == category]
        
        # Sample a few documents from this category
        sample_size = min(num_queries // len(categories), len(category_docs))
        sampled_docs = category_docs.sample(n=sample_size, random_state=42)
        
        for _, doc in sampled_docs.iterrows():
            # Create query from title (simple approach)
            title_words = doc['title'].lower().split()
            
            # Take 2-4 words as query
            query_length = random.randint(2, 4)
            if len(title_words) >= query_length:
                query_words = random.sample(title_words, query_length)
                query = ' '.join(query_words)
                
                # Add query
                queries.append({
                    'qid': qid,
                    'query': query
                })
                
                # Add primary relevance (the source document)
                qrels.append({
                    'qid': qid,
                    'doc_id': doc['id'],
                    'relevance': 1
                })
                
                # Add some related documents from same category
                related_docs = category_docs[category_docs['id'] != doc['id']].sample(
                    n=min(2, len(category_docs)-1), 
                    random_state=qid
                )
                
                for _, rel_doc in related_docs.iterrows():
                    qrels.append({
                        'qid': qid,
                        'doc_id': rel_doc['id'],
                        'relevance': 1
                    })
                
                qid += 1
    
    # Create DataFrames
    queries_df = pd.DataFrame(queries)
    qrels_df = pd.DataFrame(qrels)
    
    print(f"\nCreated {len(queries_df)} queries")
    print(f"Created {len(qrels_df)} relevance judgments")
    print(f"Average relevant docs per query: {len(qrels_df) / len(queries_df):.2f}")
    
    # Save to TSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    queries_path = output_path / 'queries.tsv'
    qrels_path = output_path / 'qrels.tsv'
    
    queries_df.to_csv(queries_path, sep='\t', index=False, header=False)
    qrels_df.to_csv(qrels_path, sep='\t', index=False, header=False)
    
    print(f"\nSaved queries to {queries_path}")
    print(f"Saved qrels to {qrels_path}")
    
    # Print sample
    print("\n--- Sample Queries ---")
    print(queries_df.head(5).to_string(index=False))
    
    print("\n--- Sample Qrels ---")
    print(qrels_df.head(10).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Create sample evaluation data")
    parser.add_argument('--corpus', required=True, help="Path to corpus CSV")
    parser.add_argument('--output_dir', default='data', help="Output directory")
    parser.add_argument('--num_queries', type=int, default=20, help="Number of queries to generate")
    args = parser.parse_args()
    
    create_sample_queries_qrels(args.corpus, args.output_dir, args.num_queries)

if __name__ == '__main__':
    main()