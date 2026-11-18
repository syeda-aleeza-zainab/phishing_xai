import pandas as pd

def make_sample_dataset(output_path, n_phishing=30, n_safe=40, seed=42):
    """Generate a small synthetic dataset matching the Kaggle Phishing_Email.csv schema for fast unit testing."""
    import random
    rng = random.Random(seed)
    phishing_samples = [
        f"Dear customer your bank account {i} has been suspended please click http://verify-now-{i}.com immediately to restore access"
        for i in range(n_phishing)
    ]
    safe_samples = [
        f"Hi team the quarterly project review meeting {i} is scheduled for tomorrow at 2pm please review the attached slides"
        for i in range(n_safe)
    ]
    # Add a couple of duplicate entries to test deduplication
    phishing_samples.append(phishing_samples[0])
    safe_samples.append(safe_samples[0])
    # Add an entry with HTML tags and whitespace
    phishing_samples.append("<p>Urgent: Your <b>PayPal</b> security code is required!</p>\r\n")

    records = []
    idx = 0
    for text in phishing_samples:
        records.append({"Unnamed: 0": idx, "Email Text": text, "Email Type": "Phishing Email"})
        idx += 1
    for text in safe_samples:
        records.append({"Unnamed: 0": idx, "Email Text": text, "Email Type": "Safe Email"})
        idx += 1

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/mock_emails.csv"
    make_sample_dataset(out)
    print(f"Created sample dataset at {out}")
