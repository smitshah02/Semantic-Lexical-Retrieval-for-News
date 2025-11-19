"""
Streamlit UI for semantic news search demo.
Usage: streamlit run app.py
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path - handle both local and deployment scenarios
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))
else:
    # Try parent directory (for deployment)
    sys.path.insert(0, str(current_dir))

try:
    from retrieve_lexical import load_tfidf_retriever, load_bm25_retriever
    from retrieve_semantic import load_semantic_retriever
    from retrieve_hybrid import HybridRetriever
    from utils import read_jsonl, create_text_snippet
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error(f"Current directory: {current_dir}")
    st.error(f"Python path: {sys.path}")
    st.stop()

# Page config
st.set_page_config(
    page_title="BBC News Search",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_retrievers(artifacts_dir: str):
    """Load all retrievers (cached)."""
    try:
        tfidf = load_tfidf_retriever(artifacts_dir)
        bm25 = load_bm25_retriever(artifacts_dir)
        semantic = load_semantic_retriever(artifacts_dir, use_faiss=True)
        hybrid = HybridRetriever({
            'tfidf': tfidf,
            'bm25': bm25,
            'semantic': semantic
        }, fusion_method='rrf')
        
        return {
            'TF-IDF': tfidf,
            'BM25': bm25,
            'Semantic': semantic,
            'Hybrid (RRF)': hybrid
        }
    except Exception as e:
        st.error(f"Error loading retrievers: {e}")
        return None

@st.cache_data
def load_chunks_data(artifacts_dir: str):
    """Load chunk metadata (cached)."""
    try:
        chunks = read_jsonl(f"{artifacts_dir}/bbc_chunks.jsonl")
        # Create lookup dict
        chunks_dict = {chunk['id']: chunk for chunk in chunks}
        return chunks_dict
    except Exception as e:
        st.error(f"Error loading chunks: {e}")
        return {}

def display_result(chunk_data: dict, rank: int, score: float):
    """Display a single search result."""
    with st.container():
        col1, col2 = st.columns([0.9, 0.1])
        
        with col1:
            st.markdown(f"**{rank}. {chunk_data['title']}**")
            st.caption(f"Category: {chunk_data['category']} | Chunk ID: {chunk_data['id']}")
        
        with col2:
            st.metric("Score", f"{score:.3f}")
        
        # Show text snippet
        snippet = create_text_snippet(chunk_data['text'], max_chars=300)
        st.markdown(f"_{snippet}_")
        
        # Expandable full text
        with st.expander("View full chunk"):
            st.text(chunk_data['text'])
        
        st.divider()

def main():
    st.title("📰 BBC News Semantic Search")
    st.markdown("Search BBC news articles using different retrieval methods")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        artifacts_dir = st.text_input(
            "Artifacts Directory",
            value="artifacts",
            help="Directory containing retrieval artifacts"
        )
        
        st.divider()
        
        search_mode = st.selectbox(
            "Search Method",
            ["TF-IDF", "BM25", "Semantic", "Hybrid (RRF)"],
            index=3,
            help="Choose the retrieval method"
        )
        
        k = st.slider(
            "Number of Results",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Number of results to retrieve"
        )
        
        st.divider()
        
        # Optional filters
        st.subheader("Filters")
        filter_category = st.multiselect(
            "Category",
            ["business", "entertainment", "politics", "sport", "tech"],
            help="Filter by category (leave empty for all)"
        )
        
        st.divider()
        
        # Info
        st.markdown("### About")
        st.markdown("""
        **Methods:**
        - **TF-IDF**: Term frequency-inverse document frequency
        - **BM25**: Best Match 25 probabilistic ranking
        - **Semantic**: Dense embeddings with cosine similarity
        - **Hybrid (RRF)**: Reciprocal Rank Fusion of all methods
        """)
    
    # Load retrievers
    retrievers = load_retrievers(artifacts_dir)
    chunks_dict = load_chunks_data(artifacts_dir)
    
    if retrievers is None or not chunks_dict:
        st.error("❌ Failed to load retrieval system. Please check artifacts directory.")
        st.stop()
    
    st.success(f"✅ Loaded {len(chunks_dict)} chunks")
    
    # Search interface
    st.divider()
    
    query = st.text_input(
        "🔍 Enter your search query",
        placeholder="e.g., What are the latest developments in technology?",
        help="Enter a natural language query"
    )
    
    search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Perform search
    if search_button and query:
        with st.spinner(f"Searching using {search_mode}..."):
            try:
                retriever = retrievers[search_mode]
                results = retriever.search(query, k=k)
                
                if not results:
                    st.warning("No results found. Try a different query.")
                else:
                    # Apply category filter if specified
                    if filter_category:
                        filtered_results = []
                        for doc_id, score in results:
                            if doc_id in chunks_dict:
                                chunk = chunks_dict[doc_id]
                                if chunk['category'] in filter_category:
                                    filtered_results.append((doc_id, score))
                        results = filtered_results
                    
                    st.success(f"Found {len(results)} results")
                    
                    # Display results
                    st.divider()
                    st.subheader("Search Results")
                    
                    for rank, (doc_id, score) in enumerate(results, start=1):
                        if doc_id in chunks_dict:
                            chunk_data = chunks_dict[doc_id]
                            display_result(chunk_data, rank, score)
                        else:
                            st.warning(f"Chunk {doc_id} not found in metadata")
            
            except Exception as e:
                st.error(f"Error during search: {e}")
    
    # Example queries
    st.divider()
    st.markdown("### 💡 Example Queries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📱 Technology trends", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("⚽ Sports news", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("💼 Business updates", use_container_width=True):
            st.rerun()

if __name__ == '__main__':
    main()