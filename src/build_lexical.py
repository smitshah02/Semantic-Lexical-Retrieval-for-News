"""
Build lexical retrieval artifacts (TF-IDF and BM25).
Usage: python src/build_lexical.py --input artifacts/bbc_chunks.jsonl --out_dir artifacts
"""
import argparse
import logging
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from utils import setup_logging, read_jsonl, ensure_dir

def simple_tokenize(text: str) -> list:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()

def build_tfidf(chunks: list, out_dir: str):
    """
    Build TF-IDF index.
    
    Saves:
        - tfidf.pkl: {vectorizer, matrix, doc_ids}
    """
    logging.info("Building TF-IDF index...")
    
    texts = [chunk['text'] for chunk in chunks]
    doc_ids = [chunk['id'] for chunk in chunks]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=2,
        max_features=50000,
        stop_words='english',
        lowercase=True
    )
    
    # Fit and transform
    logging.info(f"Fitting TF-IDF on {len(texts)} chunks...")
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    logging.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Save artifacts
    tfidf_data = {
        'vectorizer': vectorizer,
        'matrix': tfidf_matrix,
        'doc_ids': doc_ids
    }
    
    out_path = Path(out_dir) / 'tfidf.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(tfidf_data, f)
    
    logging.info(f"Saved TF-IDF artifacts to {out_path}")

def build_bm25(chunks: list, out_dir: str):
    """
    Build BM25 index.
    
    Saves:
        - bm25.pkl: {bm25_index, doc_ids}
    """
    logging.info("Building BM25 index...")
    
    texts = [chunk['text'] for chunk in chunks]
    doc_ids = [chunk['id'] for chunk in chunks]
    
    # Tokenize all documents
    logging.info(f"Tokenizing {len(texts)} chunks for BM25...")
    tokenized_corpus = [simple_tokenize(text) for text in texts]
    
    # Create BM25 index
    logging.info("Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save artifacts
    bm25_data = {
        'bm25': bm25,
        'doc_ids': doc_ids,
        'tokenized_corpus': tokenized_corpus  # Save for potential inspection
    }
    
    out_path = Path(out_dir) / 'bm25.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(bm25_data, f)
    
    logging.info(f"Saved BM25 artifacts to {out_path}")
    logging.info(f"Average document length: {bm25.avgdl:.1f} tokens")

def build_lexical_indexes(input_jsonl: str, out_dir: str):
    """Build both TF-IDF and BM25 indexes."""
    setup_logging()
    
    # Load chunks
    logging.info(f"Loading chunks from {input_jsonl}")
    chunks = read_jsonl(input_jsonl)
    logging.info(f"Loaded {len(chunks)} chunks")
    
    # Ensure output directory
    ensure_dir(out_dir)
    
    # Build both indexes
    build_tfidf(chunks, out_dir)
    build_bm25(chunks, out_dir)
    
    logging.info("Lexical indexing complete!")

def main():
    parser = argparse.ArgumentParser(description="Build lexical indexes")
    parser.add_argument('--input', required=True, help="Input JSONL path")
    parser.add_argument('--out_dir', required=True, help="Output directory for artifacts")
    args = parser.parse_args()
    
    build_lexical_indexes(args.input, args.out_dir)

if __name__ == '__main__':
    main()