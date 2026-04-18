import pandas as pd
from datasets import load_dataset
import os

def build_dataset():
    print("Downloading mteb/banking77 (Safe Banking Queries)...")
    # Load banking77 (safe data) from Hugging Face
    banking_dataset = load_dataset("mteb/banking77", split="train")
    
    # Convert to pandas dataframe
    banking_df = banking_dataset.to_pandas()
    
    # Keep only the 'text' column, and set our custom label to 0 (Safe)
    banking_df = banking_df[['text']].copy()
    banking_df['label'] = 0
    
    # Sample down to 2000 to keep it roughly balanced with the malicious datasets
    # (otherwise the model will be heavily biased towards predicting '0')
    banking_df = banking_df.sample(n=3000, random_state=42)

    print("Downloading deepset/prompt-injections (Malicious Queries)...")
    # Load prompt injections from Hugging Face
    injection_dataset = load_dataset("deepset/prompt-injections", split="train")
    
    # Convert to pandas dataframe
    injection_df = injection_dataset.to_pandas()
    
    # The deepset dataset already uses 1 for injection and 0 for safe.
    # We only want the actual prompt injections (label == 1) for our malicious class
    injection_df = injection_df[injection_df['label'] == 1][['text', 'label']].copy()
    
    # Note: We can also add some general adversarial/toxic data here if needed in the future

    print(f"Loaded {len(banking_df)} safe banking queries and {len(injection_df)} prompt injections.")

    # Merge datasets
    print("Merging and shuffling datasets...")
    combined_df = pd.concat([banking_df, injection_df], ignore_index=True)

    # Shuffle the combined dataset so training isn't sequential
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV where your train_ml.py expects it
    output_path = os.path.join("models", "training_dataset.csv")
    os.makedirs("models", exist_ok=True)
    
    # Ensure we are in the root directory if the script is run from layer1/
    if not os.path.exists("models") and os.path.exists("../models"):
        output_path = os.path.join("../models", "training_dataset.csv")

    combined_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully saved {len(combined_df)} real-world samples to {output_path}!")
    print("\nNext Steps:")
    print("1. Update layer1/train_ml.py to simply read this CSV (remove synthetic generation).")
    print("2. Run 'python layer1/train_ml.py' to train your new real-world model.")

if __name__ == "__main__":
    try:
        build_dataset()
    except ImportError:
        print("Error: Required libraries not found.")
        print("Please install them by running: pip install datasets pandas")
