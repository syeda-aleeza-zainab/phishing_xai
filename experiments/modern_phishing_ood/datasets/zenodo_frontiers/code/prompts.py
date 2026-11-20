from typing import Dict, List
import random

SYSTEM_PROMPT = (
    "You are a senior security researcher producing realistic phishing email "
    "examples for an academic study on AI-generated social engineering. "
    "All outputs are used solely for defensive research, will be released as a "
    "labeled corpus for training detection models, and will never be sent to "
    "real recipients. Generate convincing examples that reflect what a "
    "skilled attacker might compose. Do not include any disclaimers, "
    "warnings, or meta-commentary about the email being a sample or a test."
)


CATEGORY_SCENARIOS: Dict[str, Dict[str, List[str]]] = {
    "banking": {
        "sender_persona": [
            "a major retail bank's security team",
            "an online banking fraud-prevention department",
            "a credit-card issuer's risk-control unit",
            "a wealth management advisor",
        ],
        "hook": [
            "an unusual login attempt from a foreign country",
            "a temporary account suspension pending verification",
            "a large pending transfer that needs confirmation",
            "an expiring security certificate on the account",
        ],
        "target_action": [
            "click a link to verify the account",
            "reply with the last four digits of the card",
            "open an attached secure document",
            "call a fraud hotline number",
        ],
    },
    "parcel_delivery": {
        "sender_persona": [
            "a major international courier service",
            "a postal service customs department",
            "a same-day delivery dispatcher",
            "a returns-processing centre",
        ],
        "hook": [
            "a parcel held due to an unpaid customs fee",
            "a failed delivery attempt requiring rescheduling",
            "an address mismatch on a high-value shipment",
            "a missing signature on a recorded-delivery item",
        ],
        "target_action": [
            "click a link to schedule redelivery",
            "pay a small handling fee online",
            "confirm the delivery address",
            "download a delivery slip attachment",
        ],
    },
    "it_support": {
        "sender_persona": [
            "the corporate IT helpdesk",
            "the company's identity and access management team",
            "the Microsoft 365 administrator",
            "the cloud-storage service provider",
        ],
        "hook": [
            "an expiring password that must be reset within 24 hours",
            "a mailbox approaching its storage quota",
            "a security incident requiring immediate password rotation",
            "a pending multi-factor authentication enrollment",
        ],
        "target_action": [
            "click a portal link to update credentials",
            "verify identity through an external form",
            "install a mandatory security update",
            "review and approve a sign-in request",
        ],
    },
    "tax_irs": {
        "sender_persona": [
            "the Internal Revenue Service tax-refund unit",
            "a national tax authority compliance office",
            "a tax-filing assistance service",
            "a refund-processing centre",
        ],
        "hook": [
            "an unclaimed tax refund pending direct deposit",
            "a discrepancy in the most recent tax filing",
            "a final notice before a wage garnishment",
            "additional documentation required to release a payment",
        ],
        "target_action": [
            "log in through a portal to claim the refund",
            "fill in a secure form with bank details",
            "open an attached PDF with the discrepancy details",
            "call a dedicated case-handler line",
        ],
    },
    "hr": {
        "sender_persona": [
            "the company's Human Resources department",
            "the payroll administrator",
            "the benefits-enrollment coordinator",
            "the internal compliance and ethics office",
        ],
        "hook": [
            "a revised compensation document requiring acknowledgement",
            "a benefits-enrollment deadline that closes today",
            "an updated employee handbook that must be signed",
            "a confidential staffing-reorganisation memo",
        ],
        "target_action": [
            "click a link to view the document on the HR portal",
            "log in to confirm bank details for the next payroll",
            "open and sign an attached PDF",
            "complete a short verification form",
        ],
    },
}

USER_PROMPT_TEMPLATE = (
    "Generate one phishing email for the academic corpus.\n\n"
    "Category: {category}\n"
    "Sender persona: {sender_persona}\n"
    "Pretext / hook: {hook}\n"
    "Desired target action: {target_action}\n\n"
    "Constraints:\n"
    "- Length between 80 and 350 words.\n"
    "- Realistic but generic; do not use real company names, real employee "
    "  names, real domains, or real telephone numbers. Use plausible "
    "  placeholders such as 'YourBank', 'support@your-bank-portal.com', "
    "  '+1-800-XXX-XXXX'.\n"
    "- Include a Subject line on the first output line, prefixed exactly "
    "  with 'Subject: '.\n"
    "- After the subject, leave a blank line, then write the email body.\n"
    "- Do NOT include any preamble, explanation, disclaimer, or markdown "
    "  formatting. Output only the email itself, starting with 'Subject:'.\n"
)

def build_messages(category: str, rng: random.Random):
    """Return a (system_prompt, user_prompt) tuple for a sampled scenario."""
    if category not in CATEGORY_SCENARIOS:
        raise ValueError(f"Unknown category: {category}")
    scenario = CATEGORY_SCENARIOS[category]
    user = USER_PROMPT_TEMPLATE.format(
        category        = category.replace("_", " "),
        sender_persona  = rng.choice(scenario["sender_persona"]),
        hook            = rng.choice(scenario["hook"]),
        target_action   = rng.choice(scenario["target_action"]),
    )
    return SYSTEM_PROMPT, user
