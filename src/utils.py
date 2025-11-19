"""
Utility functions for I/O, logging, and text processing.
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def setup_logging(level=logging.INFO):
    """Configure logging with timestamp and level."""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file into list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(data: List[Dict[str, Any]], path: str):
    """Write list of dictionaries to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def normalize_text(text: str) -> str:
    """Basic text normalization."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
    return text.strip()

def clean_text_for_title(text: str) -> str:
    """Clean text specifically for extracting title."""
    # Remove common BBC artifacts
    text = re.sub(r'^\s*(BBC\s*News\s*[-|]\s*)?', '', text, flags=re.IGNORECASE)
    return normalize_text(text)

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter using regex."""
    # Split on periods, exclamation marks, question marks followed by space/newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def create_text_snippet(text: str, max_chars: int = 200) -> str:
    """Create a snippet from text."""
    if len(text) <= max_chars:
        return text
    # Try to break at word boundary
    snippet = text[:max_chars]
    last_space = snippet.rfind(' ')
    if last_space > max_chars * 0.8:  # If we found a space in the last 20%
        snippet = snippet[:last_space]
    return snippet + "..."

def load_queries(path: str) -> pd.DataFrame:
    """Load queries from TSV file."""
    return pd.read_csv(path, sep='\t', names=['qid', 'query'])

def load_qrels(path: str) -> pd.DataFrame:
    """Load relevance judgments from TSV file."""
    return pd.read_csv(path, sep='\t', names=['qid', 'doc_id', 'relevance'])

def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)