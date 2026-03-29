import os
import pickle
import json
from datetime import datetime

from layer1.checks import (
    check_prompt_injection,
    check_credential_harvesting,
    check_social_engineering,
    check_hidden_instructions,
    check_malicious_urls,
    get_semantic_malice_probability
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
    print(f"[Layer 2] Warning: Could not load ML models. Semantic check will be skipped. Error: {e}")

def log_event(action, query, chunk_text, score, reason):
    """Logs the Layer 2 interception to the security audit log."""
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    log_file = os.path.join(logs_dir, "security_audit.log")
    
    event = {
        "timestamp": datetime.now().isoformat(),
        "layer": "Layer 2 (Retrieval Filter)",
        "query": query,
        "action": action,
        "risk_score": round(score, 3),
        "reason": reason,
        "flagged_chunk": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
    }
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"Failed to write to audit log: {e}")

def evaluate_chunk_risk(chunk: str) -> tuple[float, list[str]]:
    """Evaluates a chunk and returns a Risk Score (0.0 to 1.0) and a list of triggered reasons."""
    reasons = []
    
    # 1. Rule-Based Checks (Auto 1.0 Risk)
    if check_prompt_injection(chunk): reasons.append("Prompt Injection")
    if check_credential_harvesting(chunk): reasons.append("Credential Harvesting")
    if check_social_engineering(chunk): reasons.append("Social Engineering")
    if check_hidden_instructions(chunk): reasons.append("Hidden Instructions")
    if check_malicious_urls(chunk): reasons.append("Malicious URL")
    
    # 2. Semantic ML Check
    ml_prob = 0.0
    if ml_vectorizer and ml_model:
        ml_prob = get_semantic_malice_probability(chunk, ml_vectorizer, ml_model)
        if ml_prob >= 0.4:
            # Only append ML reason if it crosses the borderline threshold
            reasons.append(f"Semantic Malice (Conf: {ml_prob:.2f})")
            
    # Calculate Final Score
    # If any hard rule hits, score is instantly 1.0 (Maximum severity)
    # Otherwise, it adopts the ML probability score
    has_hard_rule_hit = any(not r.startswith("Semantic") for r in reasons)
    
    if has_hard_rule_hit:
        final_score = 1.0
    else:
        final_score = ml_prob
        
    return final_score, reasons

def layer2_retrieval_filter(query: str, retrieved_chunks: list[str]) -> tuple[bool, list[str], str]:
    """
    Scans the chunks retrieved for a specific query before passing them to the LLM.
    Returns: (is_safe_to_proceed, safe_chunks, details)
    """
    safe_chunks = []
    
    if not retrieved_chunks:
        return True, [], "SAFE: No context retrieved."
        
    print(f"\n[Layer 2] Analyzing {len(retrieved_chunks)} retrieved chunks for query: '{query}'")
        
    for i, chunk in enumerate(retrieved_chunks):
        score, reasons = evaluate_chunk_risk(chunk)
        reason_str = ", ".join(reasons) if reasons else "Clean"
        
        # Threshold Engine
        if score > 0.6:
            # MALICIOUS -> Drop Chunk
            print(f"  ❌ Chunk {i+1} DROPPED | Score: {score:.2f} | Reason: {reason_str}")
            log_event("DROPPED_CHUNK", query, chunk, score, reason_str)
            continue
            
        elif score >= 0.4:
            # BORDERLINE -> Log for review, but keep chunk
            print(f"  ⚠️ Chunk {i+1} FLAGGED (Kept) | Score: {score:.2f} | Reason: Borderline semantics")
            log_event("FLAGGED_BORDERLINE", query, chunk, score, reason_str or "Borderline probability")
            safe_chunks.append(chunk)
            
        else:
            # SAFE -> Keep chunk invisibly
            print(f"  ✅ Chunk {i+1} SAFE | Score: {score:.2f}")
            safe_chunks.append(chunk)
            
    # Full Block Decision: Did we prune ALL chunks?
    if len(safe_chunks) == 0 and len(retrieved_chunks) > 0:
        log_event("BLOCKED_QUERY", query, "[ALL CONTEXT PRUNED]", 1.0, "All retrieved chunks were malicious.")
        print("[Layer 2] CRITICAL: All context pruned. Query aborted.")
        return False, [], "BLOCKED: All retrieved context was highly malicious. Query aborted to protect LLM."
        
    return True, safe_chunks, f"SAFE: {len(safe_chunks)}/{len(retrieved_chunks)} chunks approved for LLM context."
