import os
import pickle
import json
from datetime import datetime

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from layer1.checks import (
    check_prompt_injection,
    check_credential_harvesting,
    check_social_engineering,
    check_hidden_instructions,
    check_malicious_urls,
    check_semantic_malice
)

# Load the ML models globally for performance
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
vectorizer_path = os.path.join(models_dir, "layer1_vectorizer.pkl")
model_path = os.path.join(models_dir, "layer1_model.pkl")

ml_vectorizer = None
ml_model = None

try:
    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        with open(vectorizer_path, "rb") as f:
            ml_vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            ml_model = pickle.load(f)
except Exception as e:
    print(f"[Layer 1] Warning: Could not load ML models. Semantic check will be skipped. Error: {e}")

def log_event(doc_name, chunk_text, reason):
    """Logs the rejection to the security audit log."""
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    log_file = os.path.join(logs_dir, "security_audit.log")
    
    event = {
        "timestamp": datetime.now().isoformat(),
        "layer": "Layer 1 (Input Filter)",
        "document": doc_name,
        "action": "REJECTED",
        "reason": reason,
        "flagged_chunk": chunk_text
    }
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"Failed to write to audit log: {e}")

def layer1_input_filter(document_text: str, document_name: str = "Unknown_Doc") -> tuple[bool, str]:
    """
    Scans an entire document before vector DB storage.
    Returns: (is_safe: bool, details: str)
    """
    # Step 1: Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # Step 2: Split text into chunks
    chunks = text_splitter.split_text(document_text)
    if not chunks:
        return True, "SAFE: Document is empty."
        
    # Step 3: Iterate through every chunk
    for i, chunk in enumerate(chunks):
        # Apply Check 1
        if check_prompt_injection(chunk):
            reason = "Prompt Injection Detected"
            log_event(document_name, chunk, reason)
            return False, f"REJECTED: {reason} in chunk {i}"
            
        # Apply Check 2
        if check_credential_harvesting(chunk):
            reason = "Credential Harvesting Detected"
            log_event(document_name, chunk, reason)
            return False, f"REJECTED: {reason} in chunk {i}"
            
        # Apply Check 3
        if check_social_engineering(chunk):
            reason = "Social Engineering Detected"
            log_event(document_name, chunk, reason)
            return False, f"REJECTED: {reason} in chunk {i}"
            
        # Apply Check 4
        if check_hidden_instructions(chunk):
            reason = "Hidden Instructions Detected"
            log_event(document_name, chunk, reason)
            return False, f"REJECTED: {reason} in chunk {i}"
            
        # Apply Check 5
        if check_malicious_urls(chunk):
            reason = "Malicious URL Detected"
            log_event(document_name, chunk, reason)
            return False, f"REJECTED: {reason} in chunk {i}"
            
        # Apply Check 6 (ML)
        if ml_vectorizer and ml_model:
            if check_semantic_malice(chunk, ml_vectorizer, ml_model):
                reason = "Semantic Malice Detected (ML)"
                log_event(document_name, chunk, reason)
                return False, f"REJECTED: {reason} in chunk {i}"
                
    # If we got here, all chunks passed all checks
    return True, f"SAFE: Document '{document_name}' passed all Layer 1 checks."
