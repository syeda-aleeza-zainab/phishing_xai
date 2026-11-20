import google.generativeai as genai
import pandas as pd
import time
import random
import os
from faker import Faker

# CONFIGURATION 
API_KEY = os.environ.get("GEMINI_API_KEY", "") 
OUTPUT_FILE = "synthetic_bec.csv"
TARGET_COUNT = 4000
BATCH_SIZE = 20 

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
fake = Faker()

# 1. THE SCENARIO LIST
scenarios = [
    # --- FINANCIAL & INVOICE (The Classics, evolved) ---
    "Vendor (fake) to AP: 'Our bank was acquired by Chase. New routing number for the attached invoice.'",
    "CFO (fake) to Controller: 'The acquisition deal is leaking. Wire the retainer to the escrow account immediately to seal it.'",
    "Supplier to Manager: 'We applied the 5%/ early payment discount. Please process the adjusted amount today.'",
    "Consultant to VP: 'Late fee notice. You missed the net-30 window. Pay now to avoid service suspension.'",
    "CEO to Assistant: 'I need 50 digital gift cards for the client retreat. Use the corporate card. Send codes via text.'",
    "Business Partner to Exec: 'Confidential audit fee. Needs to be paid via SWIFT to our offshore holding.'",
    
    # --- HR & BENEFITS (High Trust) ---
    "HR (fake) to Employee: 'Your 401(k) contribution failed due to a system error. Re-verify your bank connection here.'",
    "Benefits Admin to Staff: 'Open Enrollment ends tomorrow. Review your coverage options in the attached PDF.'",
    "Payroll to New Hire: 'Welcome aboard. Please confirm your direct deposit details for the first pay cycle.'",
    "Headhunter to Senior Dev: 'Restricted Stock Unit (RSU) grant details attached. Please sign via DocuSign.'",
    "Internal Comms to All: 'New Remote Work Policy. Acknowledge receipt in the portal.'",
    
    # --- IT & SECURITY (Fear-Based) ---
    "IT Desk to User: 'VPN Certificate Expired. You are locked out of the intranet. Re-authenticate via the portal.'",
    "Security Ops to Manager: 'Unusual login detected from Russia. If this wasn't you, click here to freeze your account.'",
    "Zoom Admin to Staff: 'You have a recording that is about to be auto-deleted. Download it now.'",
    "Microsoft 365 Bot (fake): 'Your password expires in 24 hours. Keep current password? Reply Y/N.'",
    "DevOps to Engineer: 'API Token Rotation Alert. The integration with Salesforce has expired. Re-authorize OAuth.'",
    
    # --- LEGAL & COMPLIANCE (Authority) ---
    "Legal Counsel to CEO: 'Subpoena received regarding the Jones IP lawsuit. See attached filing. Privileged.'",
    "Compliance Officer to Dept Head: 'GDPR Violation Notice. One of your staff leaked data. Review the incident report.'",
    "IRS (fake) to Finance: 'Final Notice of Levy. Unpaid payroll taxes. Seize assets warning.'",
    "Board Member to CEO: 'Urgent board resolution required. Please sign the attached waiver.'",
    
    # --- SUPPLY CHAIN & OPS (Logistics) ---
    "Logistics to Site Mgr: 'Customs hold at the port. Pay the duty fee ($3,400) to release the container.'",
    "Building Mgmt to Office Mgr: 'Parking garage access codes are changing. Register your vehicle here.'",
    "Shipping Vendor to Warehouse: 'Damaged goods report. We need a credit memo issued to this account.'",
    "Construction Sub to PM: 'Project Alpha Retainage Release. Waiver attached. Release 10%/ holdback.'",
    
    # --- "QUISHING" & MODERN TECH (2025 Era) ---
    "Admin to Staff: 'New Wi-Fi network rollout. Scan this QR code to install the security profile.'",
    "Event Coord to Speaker: 'Your conference badge is ready. Scan to confirm dietary restrictions.'",
    "DevOps to Lead: 'AWS Root Key rotation. The old key was compromised. Generate a new one via this link.'",
    "HR to Staff: 'Digital Vaccination Card upload. Scan QR to access the secure portal.'",
    
    # --- ADVERSARIAL / SOCIAL ENGINEERING ---
    "Deepfake Ref: 'Did you get the voicemail I left? I'm heading into a tunnel, just handle the wire.'",
    "Thread Hijack Simulation: 'Re: Q3 Projections. > As discussed, here is the invoice for the audit.'",
    "Silent Fraud: 'Just checking if the payment went through? We haven't seen it on our end.'",
    "Vishing Pretext: 'I'm on a call with the bank. They need you to approve the transfer code sent to your email.'"
]

# ==========================================
# 2. THE FULL "HUMAN ELEMENT" (Tone Diversity)
# ==========================================
tones = [
    "Panic/High Anxiety ('I'm going to get fired if this isn't done')",
    "Bored/Bureaucratic ('Per policy 4022, see attached')",
    "Aggressive/Dominant C-Suite ('Don't ask questions, just execute')",
    "Friendly/Helpful ('Hey! Just trying to help you close this ticket')",
    "Confused/Incompetent ('My computer is acting up, can you check this?')",
    "Passive-Aggressive ('I assumed you handled this weeks ago?')",
    "Sloppy/Mobile ('sent from iphone pls fix')",
    "Hyper-Professional/Legalistic ('Please govern yourself accordingly')",
    "Casual/Slack-style ('hey, u there? need a favor')",
    "Robotic/Automated ('Action Required: Ticket #99223')",
    "Apologetic ('So sorry to bother you again, but the bank rejected it')",
    "Cryptic/Brief ('Call me. Now.')",
    "Excited ('Great news on the merger! One last step.')"
]

def generate_batch(batch_size=20):
    # 1. Generate Identities using Faker
    identities = []
    for _ in range(batch_size):
        identities.append({
            "sender": fake.name(),
            "receiver": fake.name(),
            "company": fake.company(),
            "job_s": fake.job(),
            "job_r": fake.job(),
            "amount": f"${random.randint(900, 85000):,}"
        })

    scenario = random.choice(scenarios)
    tone = random.choice(tones)
    
    # 2. THE Prompt
    prompt = f"""
    Act as a master social engineer. Generate {batch_size} DISTINCT Business Email Compromise (BEC) emails.
    
    CONTEXT:
    - Scenario: {scenario}
    - Tone: {tone}
    
    INSTRUCTIONS:
    1. **Avoid Patterns:** Do not start every email with "Dear". Sometimes use "Hi", "Hey", or no greeting.
    2. **Vary Length:** Make some emails 2 sentences long (mobile). Make others 3 paragraphs (desktop).
    3. **Use Realism:** Include typos if the tone is "Sloppy". Include legal jargon if "Legalistic".
    4. **MANDATORY:** USE THE IDENTITIES BELOW. No placeholders like [Name].
    
    --- IDENTITIES ---
    """
    
    for i, ident in enumerate(identities):
        prompt += f"\n{i+1}. From: {ident['sender']} ({ident['job_s']}) | To: {ident['receiver']} | Org: {ident['company']} | Amt: {ident['amount']}"

    prompt += """
    
    OUTPUT FORMAT EXAMPLE (Strictly separate with |||):
    Subject: Invoice 4022 overdue
    Body: Hi John, please pay the attached invoice.
    |||
    Subject: Quick question
    Body: Hey, did the wire go through?
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"   [!] API Error: {e}")
        if "429" in str(e):
            time.sleep(45)
        return None

def parse_response(raw_text):
    emails = []
    if not raw_text: return []
    
    raw_snippets = raw_text.split('|||')
    
    for snippet in raw_snippets:
        try:
            lines = snippet.strip().split('\n')
            subject = ""
            body = ""
            
            for line in lines:
                # Flexible Subject Parsing
                if line.lower().startswith("subject:"):
                    subject = line.split(":", 1)[1].strip()
                    # FIX: Remove brackets if LLM adds them
                    subject = subject.replace("[", "").replace("]", "")
                elif "Body:" not in line:
                    body += line + "\n"
            
            # FIX: Clean Body brackets
            body = body.strip().replace("[", "").replace("]", "")
            
            if subject and len(body) > 5:
                emails.append({'subject': subject, 'body': body, 'label': 1})
        except:
            continue
            
    return emails

def main():
    print(f"--- STARTING GENERATION ({TARGET_COUNT} samples) ---")
    
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        generated_count = len(df_existing)
        print(f"Resuming from {generated_count} samples...")
    else:
        df_existing = pd.DataFrame(columns=['subject', 'body', 'label'])
        generated_count = 0

    while generated_count < TARGET_COUNT:
        print(f"Generating Batch... ({generated_count}/{TARGET_COUNT})")
        
        raw_text = generate_batch(BATCH_SIZE)
        if raw_text:
            new_emails = parse_response(raw_text)
            if new_emails:
                df_new = pd.DataFrame(new_emails)
                df_new.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
                generated_count += len(new_emails)
                print(f"   > Added {len(new_emails)} emails.")
            else:
                print(f"   > Parser skipped batch. Raw snippet: {raw_text[:100]}...")
        
        time.sleep(4)

    print(f"SUCCESS: Generated {generated_count} emails.")

if __name__ == "__main__":
    main()