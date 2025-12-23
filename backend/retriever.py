from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os
from typing import Dict, Any

# Lightweight embedding model
# Consider a larger model if retrieval accuracy is poor
model = SentenceTransformer('all-MiniLM-L6-v2') 

# --- Global Caches for Efficiency ---
# Cache to hold the loaded DataFrames and FAISS indexes
ACT_DATA_CACHE: Dict[str, pd.DataFrame] = {}
ACT_INDEX_CACHE: Dict[str, Any] = {}

ACT_FILES = {
    "ROB": {"csv": "data/ACT_ROB_1956.csv", "index": "data/embeddings/rob_embeddings.faiss"},
    "ROC": {"csv": "data/ACT_ROC_2016.csv", "index": "data/embeddings/roc_embeddings.faiss"},
    "LLP": {"csv": "data/ACT_LLP_2024.csv", "index": "data/embeddings/llp_embeddings.faiss"}
}

def initialize_retriever():
    """Load all necessary data and indexes into the cache upon startup."""
    print("Initializing retriever and loading legal data...")
    for act_name, paths in ACT_FILES.items():
        try:
            # 1. Load DataFrame
            df = pd.read_csv(paths["csv"])
            ACT_DATA_CACHE[act_name] = df
            
            # 2. Load FAISS Index
            index = faiss.read_index(paths["index"])
            ACT_INDEX_CACHE[act_name] = index
            print(f"Loaded {act_name} successfully.")
        except FileNotFoundError as e:
            # Important: If a file is missing, the backend won't start for that Act
            print(f"CRITICAL ERROR: {e}. Please ensure all files are present.")
            # Optional: You could choose to skip it, but failing fast is safer.
        except Exception as e:
            print(f"An unexpected error occurred loading {act_name}: {e}")

# Call the initialization function when this script is imported by app.py
initialize_retriever()


def retrieve_context(query: str, act_name: str, k: int = 3) -> str:
    act_name = act_name.upper()
    
    if act_name not in ACT_DATA_CACHE or act_name not in ACT_INDEX_CACHE:
        return f"Error: Act data or index not loaded for {act_name}. Check initialization."

    # Retrieve cached components
    df = ACT_DATA_CACHE[act_name]
    index = ACT_INDEX_CACHE[act_name]

    # Encode query
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), k=k)

    # Combine top chunks, adding source info for the LLM
    context_chunks = []
    
    # We retrieve the specific rows based on the index I
    for chunk_index in I[0]:
        # Assuming your CSV/DataFrame has a 'Section' and 'Content' column
        row = df.iloc[chunk_index]
        source_info = f"[Source: {act_name}, Section: {row.get('Section', 'N/A')}]"
        content = row["Content"]
        
        # Format for clarity
        context_chunks.append(f"{source_info} {content}")

    context = "\n\n".join(context_chunks)
    return context