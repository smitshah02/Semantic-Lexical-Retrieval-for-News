"""
Build semantic embeddings using sentence-transformers.
Usage: python src/build_embeddings.py --input artifacts/bbc_chunks.jsonl --out_dir artifacts --model all-MiniLM-L6-v2
"""
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils import setup_logging, read_jsonl, ensure_dir

def build_embeddings(
    input_jsonl: str,
    out_dir: str,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    use_faiss: bool = False
):
    """
    Build semantic embeddings for chunks.
    
    Saves:
        - embeddings.npy: N x D numpy array
        - doc_ids.json: list of chunk IDs aligned with embeddings
        - faiss.index: optional FAISS index
    """
    setup_logging()
    
    # Load chunks
    logging.info(f"Loading chunks from {input_jsonl}")
    chunks = read_jsonl(input_jsonl)
    logging.info(f"Loaded {len(chunks)} chunks")
    
    # Extract texts and IDs
    texts = [chunk['text'] for chunk in chunks]
    doc_ids = [chunk['id'] for chunk in chunks]
    
    # Load model
    logging.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Encode texts
    logging.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    
    logging.info(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Ensure output directory
    ensure_dir(out_dir)
    
    # Save embeddings
    embeddings_path = Path(out_dir) / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    logging.info(f"Saved embeddings to {embeddings_path}")
    
    # Save doc IDs
    doc_ids_path = Path(out_dir) / 'doc_ids.json'
    with open(doc_ids_path, 'w') as f:
        json.dump(doc_ids, f)
    logging.info(f"Saved doc IDs to {doc_ids_path}")
    
    # Optional: Build FAISS index
    if use_faiss:
        try:
            import faiss
            logging.info("Building FAISS index...")
            
            # Use IndexFlatIP for inner product (cosine with normalized vectors)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
            
            # Save FAISS index
            faiss_path = Path(out_dir) / 'faiss.index'
            faiss.write_index(index, str(faiss_path))
            logging.info(f"Saved FAISS index to {faiss_path}")
            logging.info(f"FAISS index contains {index.ntotal} vectors")
            
        except ImportError:
            logging.warning("FAISS not available, skipping FAISS index creation")
    
    logging.info("Embedding generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Build semantic embeddings")
    parser.add_argument('--input', required=True, help="Input JSONL path")
    parser.add_argument('--out_dir', required=True, help="Output directory")
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help="SentenceTransformer model name")
    parser.add_argument('--batch_size', type=int, default=32, help="Encoding batch size")
    parser.add_argument('--faiss', action='store_true', help="Build FAISS index")
    args = parser.parse_args()
    
    build_embeddings(
        args.input,
        args.out_dir,
        args.model,
        args.batch_size,
        args.faiss
    )

if __name__ == '__main__':
    main()