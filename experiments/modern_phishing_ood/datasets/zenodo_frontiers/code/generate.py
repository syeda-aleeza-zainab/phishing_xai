import sys
import time
import random
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import config
from prompts import build_messages

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai")
    sys.exit(1)

def parse_email(raw: str):
    """Split the raw response into (subject, body). The prompt enforces
    'Subject: ...' on the first line."""
    raw = (raw or "").strip()
    if not raw:
        return "", ""
    lines = raw.splitlines()
    if lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
    else:
        subject = ""
        body = raw
    return subject, body

def call_api(client, deployment_name, system_prompt, user_prompt):
    """Call the chat completion endpoint with retries and exponential backoff.
    Falls back to merging system+user into a single user message for
    deployments that do not accept a 'system' role (occasionally the case
    for non-OpenAI models served behind the OpenAI-compatible facade)."""
    last_err = None
    backoff = config.RETRY_BACKOFF_SEC

    payload_normal = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    payload_merged = [
        {"role": "user",
         "content": f"{system_prompt}\n\n---\n\n{user_prompt}"}
    ]

    for attempt in range(1, config.MAX_RETRIES + 1):
        for messages in (payload_normal, payload_merged):
            try:
                resp = client.chat.completions.create(
                    model       = deployment_name,
                    messages    = messages,
                    temperature = config.GEN_TEMPERATURE,
                    top_p       = config.GEN_TOP_P,
                    max_tokens  = config.GEN_MAX_TOKENS,
                    timeout     = config.REQUEST_TIMEOUT_SEC,
                )
                return resp.choices[0].message.content
            except Exception as e:
                msg = str(e).lower()
                # Retry with merged payload only on 'system' role rejection
                if "system" in msg and "role" in msg:
                    last_err = e
                    continue
                last_err = e
                break
        if attempt < config.MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
    raise last_err

def run_model(model_key: str, client: OpenAI):
    if model_key not in config.MODELS:
        print(f"[ERROR] Unknown model '{model_key}'. "
              f"Valid keys: {list(config.MODELS)}")
        sys.exit(1)

    deployment_name = config.MODELS[model_key]
    out_file = config.LLM_DIR / f"{model_key}_raw.csv"
    log_file = config.LOGS_DIR / f"{model_key}_errors.log"

    rng = random.Random(config.RANDOM_SEED + hash(model_key) % 10000)

    # Resume support
    existing = []
    if out_file.exists():
        try:
            existing = list(pd.read_csv(out_file)["id"])
            print(f"[INFO] {model_key}: resuming with {len(existing)} existing rows")
        except Exception:
            print(f"[WARN] {model_key}: could not read existing {out_file}; starting fresh")

    tasks = []
    for cat in config.CATEGORIES:
        for i in range(config.N_PER_CATEGORY):
            task_id = f"{model_key}_{cat}_{i:04d}"
            if task_id not in existing:
                tasks.append((task_id, cat))
    print(f"[INFO] {model_key}: {len(tasks)} tasks queued "
          f"(deployment='{deployment_name}')")

    if not tasks:
        print(f"[OK]   {model_key}: nothing to do.")
        return

    rows = []
    write_every = 25
    err_count = 0
    pbar = tqdm(tasks, desc=f"Generating {model_key}")
    for task_id, cat in pbar:
        sys_prompt, usr_prompt = build_messages(cat, rng)
        try:
            raw = call_api(client, deployment_name, sys_prompt, usr_prompt)
            subj, body = parse_email(raw)
            rows.append({
                "id":           task_id,
                "model":        model_key,
                "category":     cat,
                "subject":      subj,
                "body":         body,
                "raw_response": raw,
            })
        except Exception as e:
            err_count += 1
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_id}\t{type(e).__name__}\t{e}\n")
            pbar.set_postfix({"errors": err_count})

        if len(rows) >= write_every:
            mode = "a" if out_file.exists() else "w"
            header = not out_file.exists()
            pd.DataFrame(rows).to_csv(out_file, mode=mode, header=header,
                                      index=False, encoding="utf-8")
            rows = []

        time.sleep(config.SLEEP_BETWEEN_REQS)

    # Final flush
    if rows:
        mode = "a" if out_file.exists() else "w"
        header = not out_file.exists()
        pd.DataFrame(rows).to_csv(out_file, mode=mode, header=header,
                                  index=False, encoding="utf-8")

    print(f"\n[OK]  {model_key}: output = {out_file}")
    print(f"[OK]  {model_key}: errors = {err_count}"
          + (f" (see {log_file})" if err_count else ""))

def main():
    parser = argparse.ArgumentParser(
        description="Generate phishing emails through Azure AI Foundry.")
    parser.add_argument(
        "--model", required=True,
        choices=list(config.MODELS.keys()) + ["all"],
        help="Model key (or 'all' to run the three sequentially).")
    args = parser.parse_args()

    if not config.AZURE_API_KEY:
        print("[ERROR] AZURE_API_KEY env var not set.")
        print("        PowerShell: $env:AZURE_API_KEY=\"...\"")
        print("        bash/zsh:   export AZURE_API_KEY=\"...\"")
        sys.exit(1)

    client = OpenAI(
        base_url = config.AZURE_ENDPOINT,
        api_key  = config.AZURE_API_KEY,
        timeout  = config.REQUEST_TIMEOUT_SEC,
    )

    targets = list(config.MODELS.keys()) if args.model == "all" else [args.model]
    for mk in targets:
        run_model(mk, client)

if __name__ == "__main__":
    main()
