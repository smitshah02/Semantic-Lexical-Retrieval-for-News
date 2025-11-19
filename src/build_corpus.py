"""
Build normalized corpus CSV from BBC folder structure.
Usage: python src/build_corpus.py --src_dir raw_data/bbc --out_csv artifacts/bbc_corpus.csv
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from utils import setup_logging, normalize_text, clean_text_for_title, ensure_dir

def extract_title_and_text(file_path: Path) -> tuple:
    """
    Extract title and body from BBC text file.
    First non-empty line is title, rest is body.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Find first non-empty line as title
    title = ""
    body_start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            title = clean_text_for_title(stripped)
            body_start_idx = i + 1
            break
    
    # Rest is body
    body = ' '.join(line.strip() for line in lines[body_start_idx:] if line.strip())
    body = normalize_text(body)
    
    return title, body

def build_corpus(src_dir: str, out_csv: str):
    """
    Walk through BBC folder structure and create normalized CSV.
    
    Expected structure:
    src_dir/
        business/
            001.txt
            002.txt
        entertainment/
            ...
        politics/
            ...
        sport/
            ...
        tech/
            ...
    """
    setup_logging()
    src_path = Path(src_dir)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    
    records = []
    
    # Get all category folders
    categories = [d for d in src_path.iterdir() if d.is_dir()]
    logging.info(f"Found {len(categories)} categories: {[c.name for c in categories]}")
    
    for category_dir in categories:
        category = category_dir.name
        txt_files = list(category_dir.glob("*.txt"))
        
        logging.info(f"Processing {len(txt_files)} files in category '{category}'")
        
        for file_path in tqdm(txt_files, desc=f"{category}"):
            try:
                title, text = extract_title_and_text(file_path)
                
                if not title or not text:
                    logging.warning(f"Skipping {file_path}: empty title or text")
                    continue
                
                doc_id = f"{category}_{file_path.stem}"
                
                records.append({
                    'id': doc_id,
                    'title': title,
                    'text': text,
                    'category': category,
                    'date': ''  # BBC corpus doesn't have dates
                })
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    logging.info(f"Created corpus with {len(df)} documents")
    logging.info(f"Category distribution:\n{df['category'].value_counts()}")
    
    # Ensure output directory exists
    ensure_dir(Path(out_csv).parent)
    
    # Save to CSV
    df.to_csv(out_csv, index=False, encoding='utf-8')
    logging.info(f"Saved corpus to {out_csv}")
    
    # Print sample
    logging.info(f"Sample record:\n{df.iloc[0].to_dict()}")

def main():
    parser = argparse.ArgumentParser(description="Build BBC corpus CSV")
    parser.add_argument('--src_dir', required=True, help="Source BBC directory")
    parser.add_argument('--out_csv', required=True, help="Output CSV path")
    args = parser.parse_args()
    
    build_corpus(args.src_dir, args.out_csv)

if __name__ == '__main__':
    main()