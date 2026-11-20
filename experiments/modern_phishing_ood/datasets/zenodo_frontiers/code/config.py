

import os
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\RommGT\Desktop\articulo4\modelos IA")    # <-- CHANGE THIS

DATA_DIR    = PROJECT_ROOT / "data"
HUMAN_DIR   = DATA_DIR / "human"
LLM_DIR     = DATA_DIR / "llm"
LOGS_DIR    = PROJECT_ROOT / "logs"

for d in [DATA_DIR, HUMAN_DIR, LLM_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Read the key from environment for security:
#   PowerShell: $env:AZURE_API_KEY="..."
#   bash/zsh:   export AZURE_API_KEY="..."
AZURE_ENDPOINT = "https://pruebasphishingfoundry.openai.azure.com/openai/v1"
AZURE_API_KEY  = os.getenv("AZURE_API_KEY", "<REDACTED — leaked Azure key removed before GitHub push>")

# The KEY (left) is the friendly identifier used in filenames, command-line
# args, and source labels. The VALUE (right) is the exact deployment name
# registered in your Azure AI Foundry project.
MODELS = {
    "gpt-4.1":           "gpt-4.1-1",                # <-- replace
    "deepseek3.2": "DeepSeek-V3.2",      # <-- replace
    "llama-3.3-70b":     "Llama-3.3-70B-Instruct",          # <-- replace
}

GEN_TEMPERATURE = 0.7
GEN_TOP_P       = 0.95
GEN_MAX_TOKENS  = 600

N_PER_MODEL    = 1666
CATEGORIES     = ["banking", "parcel_delivery", "it_support", "tax_irs", "hr"]
N_PER_CATEGORY = N_PER_MODEL // len(CATEGORIES)   # 333 per category per model

MAX_RETRIES         = 3
RETRY_BACKOFF_SEC   = 5
REQUEST_TIMEOUT_SEC = 60
SLEEP_BETWEEN_REQS  = 0.3
RANDOM_SEED         = 42

MIN_TOKENS = 30
MAX_TOKENS = 500
