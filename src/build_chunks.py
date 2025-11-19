"""
Build chunked JSONL from corpus CSV with sentence-aware chunking.
Usage: python src/build_chunks.py --input artifacts/bbc_corpus.csv --output artifacts/bbc_chunks.jsonl
"""
import argparse
import logging
import pandas as pd
from typing import List
from pathlib import Path
from utils import (
    setup_logging, write_jsonl, count_words, 
    split_into_sentences, ensure_dir
)

def chunk_text_sentence_aware(
    text: str,
    target_words: int = 450,
    overlap_words: int = 100,
    max_single: int = 550
) -> List[str]:
    """
    Chunk text into roughly target_words per chunk, respecting sentence boundaries.
    
    Args:
        text: Input text
        target_words: Target words per chunk
        overlap_words: Overlap between chunks
        max_single: Max words for single chunk (no split if below this)
    
    Returns:
        List of text chunks
    """
    total_words = count_words(text)
    
    # If short enough, return as single chunk
    if total_words <= max_single:
        return [text]
    
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_words = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = count_words(sentence)
        
        # If adding this sentence stays under target, add it
        if current_words + sentence_words <= target_words:
            current_chunk.append(sentence)
            current_words += sentence_words
            i += 1
        else:
            # Finalize current chunk if it has content
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_chunk = []
                overlap_count = 0
                
                # Go back and collect sentences for overlap
                j = len(current_chunk) - 1
                while j >= 0 and overlap_count < overlap_words:
                    sent = current_chunk[j]
                    overlap_chunk.insert(0, sent)
                    overlap_count += count_words(sent)
                    j -= 1
                
                current_chunk = overlap_chunk
                current_words = overlap_count
            else:
                # Edge case: single sentence is longer than target
                # Just add it and move on
                chunks.append(sentence)
                current_chunk = []
                current_words = 0
                i += 1
    
    # Add final chunk if any
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def build_chunks(
    input_csv: str,
    output_jsonl: str,
    target_words: int = 450,
    overlap_words: int = 100,
    max_single: int = 550
):
    """
    Build chunked corpus from CSV.
    
    Output format per line:
    {
        "id": "<doc_id>_<chunk_index>",
        "doc_id": "<doc_id>",
        "chunk_index": <int>,
        "title": "...",
        "category": "...",
        "date": "...",
        "text": "<chunk_text>"
    }
    """
    setup_logging()
    logging.info(f"Loading corpus from {input_csv}")
    
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} documents")
    
    all_chunks = []
    total_chunks = 0
    
    for idx, row in df.iterrows():
        doc_id = row['id']
        title = row['title']
        text = row['text']
        category = row['category']
        date = row.get('date', '')
        
        # Create chunks
        chunks = chunk_text_sentence_aware(
            text, target_words, overlap_words, max_single
        )
        
        # Create records
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_record = {
                'id': f"{doc_id}_{chunk_idx}",
                'doc_id': doc_id,
                'chunk_index': chunk_idx,
                'title': title,
                'category': category,
                'date': date,
                'text': chunk_text
            }
            all_chunks.append(chunk_record)
            total_chunks += 1
        
        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1}/{len(df)} documents, {total_chunks} chunks")
    
    logging.info(f"Created {total_chunks} chunks from {len(df)} documents")
    logging.info(f"Average chunks per document: {total_chunks / len(df):.2f}")
    
    # Ensure output directory exists
    ensure_dir(Path(output_jsonl).parent)
    
    # Save to JSONL
    write_jsonl(all_chunks, output_jsonl)
    logging.info(f"Saved chunks to {output_jsonl}")
    
    # Statistics
    chunk_word_counts = [count_words(c['text']) for c in all_chunks]
    import numpy as np
    logging.info(f"Chunk word count stats:")
    logging.info(f"  Mean: {np.mean(chunk_word_counts):.1f}")
    logging.info(f"  Median: {np.median(chunk_word_counts):.1f}")
    logging.info(f"  Min: {np.min(chunk_word_counts)}")
    logging.info(f"  Max: {np.max(chunk_word_counts)}")

def main():
    parser = argparse.ArgumentParser(description="Build chunked JSONL corpus")
    parser.add_argument('--input', required=True, help="Input CSV path")
    parser.add_argument('--output', required=True, help="Output JSONL path")
    parser.add_argument('--target_words', type=int, default=450, help="Target words per chunk")
    parser.add_argument('--overlap_words', type=int, default=100, help="Overlap words between chunks")
    parser.add_argument('--max_single', type=int, default=550, help="Max words for single chunk")
    args = parser.parse_args()
    
    build_chunks(
        args.input,
        args.output,
        args.target_words,
        args.overlap_words,
        args.max_single
    )

if __name__ == '__main__':
    main()