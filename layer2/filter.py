import os
import pickle
import json
from datetime import datetime

# Used for ZKIP Semantic Distance
from sentence_transformers import SentenceTransformer, util

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

# Load the ZKIP distance measuring model locally
try:
    print("[Layer 2] Loading lightweight ZKIP semantic similarity model...")
    zkip_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"[Layer 2] Error loading ZKIP model: {e}")
    zkip_model = None

# A_i = 1 - s_i. If s_i < 0.85, the causal influence of the document was extremely high (Suspicious!).
ZKIP_THRESHOLD = 0.85 

def log_event(action, query, chunk_text, metric, reason):
    """Logs the Layer 2 interception to the security audit log."""
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    log_file = os.path.join(logs_dir, "security_audit.log")
    
    event = {
        "timestamp": datetime.now().isoformat(),
        "layer": "Layer 2 (Context Defense - ZKIP Hybrid)",
        "query": query,
        "action": action,
        "detection_metric": metric,
        "reason": reason,
        "flagged_chunk": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
    }
    
    try:
        os.makedirs(logs_dir, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"Failed to write to audit log: {e}")

def evaluate_chunk_risk_original(chunk: str) -> tuple[float, list[str]]:
    """Evaluates a chunk via the original Layer 1 Checks & ML rules."""
    reasons = []
    
    # 1. Rule-Based Checks
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
            reasons.append(f"Semantic Malice (Conf: {ml_prob:.2f})")
            
    has_hard_rule_hit = any(not r.startswith("Semantic") for r in reasons)
    final_score = 1.0 if has_hard_rule_hit else ml_prob
        
    return final_score, reasons

def measure_causal_influence(ans_full: str, ans_drop: str) -> float:
    """Calculates Semantic Similarity between generated answers to find the ZKIP A_i anomaly score."""
    if not zkip_model:
        # Failsafe if the model somehow failed to load
        return 1.0 
        
    emb_full = zkip_model.encode(ans_full, convert_to_tensor=True)
    emb_drop = zkip_model.encode(ans_drop, convert_to_tensor=True)
    
    # Cosine similarity s_i
    similarity = util.cos_sim(emb_full, emb_drop).item()
    return similarity

def layer2_retrieval_filter(query: str, retrieved_chunks: list[str], llm_generate_fn=None) -> tuple[bool, list[str], str]:
    """
    Scans retrieved chunks using ZKIP causal testing (LOO) + the Original ML checks.
    """
    safe_chunks = []
    
    if not retrieved_chunks:
        return True, [], "SAFE: No context retrieved."
        
    print(f"\n[Layer 2 - ZKIP Hybrid] Examining {len(retrieved_chunks)} chunks for query: '{query}'")
    
    # Check if an LLM callback was provided for ZKIP
    if llm_generate_fn is None:
        print("[Layer 2] WARNING: No LLM callback provided. ZKIP Causal tests skipped. Falling back to ML rules only.")
        ans_full = None
    else:
        # Step 2: Generate REFERENCE ANSWER using ALL retrieved chunks (ans_full)
        print("  -> ZKIP: Generating 'ans_full' reference response...")
        ans_full = llm_generate_fn(query, retrieved_chunks)

    pruned_reasons_summary = []
    # Step 3: Leave-One-Out Testing (LOO) over each chunk + Hybrid OR-Gate
    for i, chunk in enumerate(retrieved_chunks):
        reasons_for_drop = []
        metric_str = ""
        is_malicious = False
        
        # --- PATH A: Original Rule + ML Baseline ---
        ml_score, ml_reasons = evaluate_chunk_risk_original(chunk)
        if ml_score > 0.6:
            is_malicious = True
            reasons_for_drop.extend(ml_reasons)
            metric_str += f"ML_Score={ml_score:.2f} "
            
        # --- PATH B: ZKIP Causal Test (LOO) ---
        if llm_generate_fn is not None and ans_full is not None:
            # Drop chunk `i`
            context_drop = retrieved_chunks[:i] + retrieved_chunks[i+1:]
            
            # Generate ans_drop
            print(f"  -> ZKIP: Generating LOO answer without chunk {i+1}...")
            ans_drop = llm_generate_fn(query, context_drop)
            
            # Step 4: Measure ZKIP Causal distance
            similarity = measure_causal_influence(ans_full, ans_drop)
            
            if similarity < ZKIP_THRESHOLD:
                is_malicious = True
                reasons_for_drop.append("ZKIP Causal Influence Detected (Massive Semantic Shift)")
            metric_str += f"ZKIP_Sim={similarity:.2f}"
            
        # Format metrics and handle the chunk 
        if is_malicious:
            reason_str = ", ".join(reasons_for_drop)
            print(f"  ❌ Chunk {i+1} DROPPED | Metrics: [{metric_str.strip()}] | Reason: {reason_str}")
            log_event("DROPPED_CHUNK", query, chunk, metric_str.strip(), reason_str)
            pruned_reasons_summary.append(f"Chunk {i+1} ({reason_str})")
            continue
            
        elif ml_score >= 0.4:
            # We keep borderline ML scores (0.4-0.6) if ZKIP proved their causal impact was low!
            print(f"  ⚠️ Chunk {i+1} FLAGGED (Kept) | Metrics: [{metric_str.strip()}] | Reason: Borderline semantics but low causal shift")
            log_event("FLAGGED_BORDERLINE", query, chunk, metric_str.strip(), "Borderline semantics, survived ZKIP.")
            safe_chunks.append(chunk)
            
        else:
            print(f"  ✅ Chunk {i+1} SAFE | Metrics: [{metric_str.strip()}]")
            safe_chunks.append(chunk)

    # Step 6: Final Full Block Decision
    if len(safe_chunks) == 0 and len(retrieved_chunks) > 0:
        log_event("BLOCKED_QUERY", query, "[ALL CONTEXT PRUNED]", "1.0", "All retrieved chunks were malicious.")
        print("[Layer 2] CRITICAL: All context pruned. Query aborted.")
        return False, [], f"BLOCKED: All context was hostile. Drops: {', '.join(pruned_reasons_summary)}"
        
    if pruned_reasons_summary:
        return True, safe_chunks, f"SAFE: {len(safe_chunks)}/{len(retrieved_chunks)} chunks approved. Pruned: {', '.join(pruned_reasons_summary)}"
    else:
        return True, safe_chunks, f"SAFE: {len(safe_chunks)}/{len(retrieved_chunks)} chunks approved."
