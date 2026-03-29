import re

# Check 1: Prompt Injection
def check_prompt_injection(text: str) -> bool:
    """Returns True if malicious, False if safe."""
    patterns = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)forget\s+everything",
        r"(?i)you\s+are\s+now",
        r"(?i)override\s+instructions",
        r"(?i)new\s+persona",
        r"(?i)jailbreak",
        r"(?i)dan\s+mode"
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False

# Check 2: Credential Harvesting
def check_credential_harvesting(text: str) -> bool:
    patterns = [
        r"(?i)reveal\s+.*password",
        r"(?i)what\s+is\s+the\s+api_key",
        r"(?i)private_key",
        r"(?i)secret_token",
        r"(?i)credentials",
        r"(?i)reveal\s+confidential"
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False

# Check 3: Social Engineering
def check_social_engineering(text: str) -> bool:
    patterns = [
        r"(?i)account\s+will\s+be\s+deleted",
        r"(?i)call\s+this\s+number",
        r"(?i)click\s+this\s+link",
        r"(?i)verify\s+your\s+identity",
        r"(?i)you\s+have\s+been\s+hacked",
        r"(?i)urgent\s+action\s+required"
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False

# Check 4: Hidden Instructions
def check_hidden_instructions(text: str) -> bool:
    hidden_chars = [
        "\u200b", # Zero-width space
        "\u200c", # Zero-width non-joiner
        "\u200d", # Zero-width joiner
        "\ufeff", # Zero-width no-break space
        "\u00ad"  # Soft hyphen
    ]
    for char in hidden_chars:
        if char in text:
            return True
    
    # Or overly dense special characters (ratio > 30%)
    if len(text) > 0:
        special_chars = re.sub(r'[a-zA-Z0-9\s]', '', text)
        if len(special_chars) / len(text) > 0.3:
            return True
            
    return False

# Check 5: Malicious URLs
def check_malicious_urls(text: str) -> bool:
    # Look for urls
    url_pattern = r"(https?://[^\s]+)"
    urls = re.findall(url_pattern, text)
    
    suspicious_tlds = [".ru", ".xyz", ".tk", ".ml", ".cf", ".pw"]
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly"]
    
    for url in urls:
        # Check TLD
        for tld in suspicious_tlds:
            if tld in url.lower():
                return True
        # Check Shortener
        for short in shorteners:
            if short in url.lower():
                return True
    return False

# Check 6: Semantic ML Check
def check_semantic_malice(text: str, vectorizer, model) -> bool:
    """Returns True if the ML model thinks it's malicious."""
    if not vectorizer or not model:
        # Failsafe if model is missing
        return False
        
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return pred[0] == 1 # 1 = Malicious

def get_semantic_malice_probability(text: str, vectorizer, model) -> float:
    """Returns the raw probability (0.0 to 1.0) of the text being malicious."""
    if not vectorizer or not model:
        return 0.0
        
    vec = vectorizer.transform([text])
    # predict_proba returns an array of shape (n_samples, n_classes)
    # Class 1 is 'Malicious', so we want index 1
    probs = model.predict_proba(vec)
    return float(probs[0][1])
