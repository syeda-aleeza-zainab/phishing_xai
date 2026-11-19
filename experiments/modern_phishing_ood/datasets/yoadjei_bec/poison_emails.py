import pandas as pd
import random
import re
import os


# CONFIGURATION

INPUT_FILE = "synthetic_emails.csv"
OUTPUT_FILE = "synthetic_emails_poisoned.csv"

# Target 30% of fraud emails for obfuscation
POISON_RATE = 0.30

# 1. HOMOGLYPH MAP (Cyrillic lookalikes)
homoglyphs = {
    'a': 'а', 'c': 'с', 'e': 'е', 'i': 'і', 
    'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у',
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 
    'H': 'Н', 'K': 'К', 'M': 'М', 'O': 'О', 
    'P': 'Р', 'T': 'Т', 'X': 'Х'
}

# 2. INVISIBLE CHARACTERS
invisible_chars = [
    '\u200b', # Zero width space
    '\u200c', # Zero width non-joiner
    '\u200d', # Zero width joiner
]

# 3. KEYWORDS TO BREAK
# We target specific "trigger words" that classifiers rely on.
target_keywords = [
    'wire', 'invoice', 'payment', 'bank', 'account', 'urgent', 'verify',
    'routing', 'swift', 'deposit', 'unauthorized', 'suspended', 'password',
    'expired', 'access', 'immediately', 'action', 'required', 'security',
    'funds', 'transfer', 'update', 'confirm', 'details'
]

def apply_homoglyphs(text):
    """
    Replaces specific Latin characters with Cyrillic lookalikes.
    This breaks character-level tokenizers (like BERT's WordPiece).
    """
    chars = list(text)
    # Find indices of characters that CAN be replaced
    replaceable_indices = [i for i, char in enumerate(chars) if char in homoglyphs]
    
    if not replaceable_indices:
        return text
        
    # We replace about 20% of eligible characters to keep it subtle.
    num_to_change = max(1, int(len(replaceable_indices) * 0.20))
    indices_to_change = random.sample(replaceable_indices, num_to_change)
    
    for i in indices_to_change:
        chars[i] = homoglyphs[chars[i]]
    return "".join(chars)

def inject_invisible_noise(text):

    text_lower = text.lower()
    for word in target_keywords:
        if word in text_lower:
            # Create broken version: "b_a_n_k"
            # We use a random invisible char for each gap
            broken_word = ""
            original_word_case = text[text_lower.find(word):text_lower.find(word)+len(word)]
            
            for char in original_word_case:
                broken_word += char + random.choice(invisible_chars)
            # Remove the trailing invisible char
            broken_word = broken_word[:-1]
            
            # Replace in text (using regex to be case insensitive but preserve case)
            text = re.sub(re.escape(word), broken_word, text, flags=re.IGNORECASE)
    return text

def main():
    print("--- STARTING ADVERSARIAL POISONING ---")
    
    # 1. Load Data
    print(f"1. Loading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Please ensure your file is named correctly.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"   > Total samples loaded: {len(df)}")

    # 2. Filter for Fraud Only
    # We assume 'label' column exists. If not, we assume everything in this file is fraud.
    if 'label' in df.columns:
        fraud_indices = df[df['label'] == 1].index.tolist()
    else:
        print("   > No 'label' column found. Assuming ALL samples are fraud.")
        df['label'] = 1
        fraud_indices = df.index.tolist()

    # 3. Select Targets
    poison_count = int(len(fraud_indices) * POISON_RATE)
    target_indices = random.sample(fraud_indices, poison_count)
    
    print(f"2. Poisoning {poison_count} samples ({int(POISON_RATE*100)}% of fraud data)...")
    print("   > Applying Homoglyphs (Cyrillic substitution)")
    print("   > Applying Zero-Width Injection (Invisible spaces)")

    modified_count = 0
    for idx in target_indices:
        original_body = df.at[idx, 'body']
        
        # Safety check
        if not isinstance(original_body, str) or len(original_body) < 5:
            continue
            
        # Randomly select an attack vector
        attack_type = random.choice(['homoglyph', 'invisible', 'both'])
        
        poisoned_body = original_body
        
        if attack_type == 'homoglyph' or attack_type == 'both':
            poisoned_body = apply_homoglyphs(poisoned_body)
        
        if attack_type == 'invisible' or attack_type == 'both':
            poisoned_body = inject_invisible_noise(poisoned_body)
            
        df.at[idx, 'body'] = poisoned_body
        modified_count += 1

    # 4. Save Result
    print(f"3. Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: {modified_count} emails successfully poisoned.")
    print("   > You can now merge this file with your Legitimate Enron data.")

if __name__ == "__main__":
    main()