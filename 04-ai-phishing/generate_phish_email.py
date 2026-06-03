"""Generate a tabletop AI-phishing email for security training.

SAFETY
------
This script is for authorized security-awareness exercises only.

It will:
  - Watermark every output with a [TABLETOP EXERCISE - DO NOT SEND] banner.
  - Refuse to produce credential-harvesting payloads or working malware URLs.
  - Refuse to use real-person identifiers in the target context unless the
    operator passes --consent-confirmed AND --consent-owner "<name + role>".
  - Never send mail. Output goes to stdout only.

Misusing this script to phish people without their consent is illegal in
most jurisdictions and unethical in all of them. Don't.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from anthropic import Anthropic

MODEL = "claude-sonnet-4-5-20250929"

BANNER = "[TABLETOP EXERCISE - DO NOT SEND] " * 3

SYSTEM_PROMPT = """You are assisting an authorized security-awareness exercise.

You will draft a single phishing-style email used ONLY for tabletop training.

Hard rules:
- Do NOT include real credential-harvesting links. Use the placeholder
  https://example-training.invalid/reset-{token} for any link.
- Do NOT include working malware download URLs or executable instructions.
- Do NOT include real attachment payloads. Refer to attachments by filename only.
- Begin your output with the literal banner the user provides, on its own line.
- End your output with the literal footer the user provides, on its own line.
- Keep the email under 200 words.
- Match the writing style of the samples the user provides.
- The email must be plausible enough to be educational but contain at least one
  detectable phishing tell (urgency, out-of-band channel request, link/domain
  mismatch, or authority pressure). Note the tell in a final 'TRAINER NOTES:'
  section after the footer.
"""

REAL_NAME_PATTERN = re.compile(r"\b(?:dr|mr|mrs|ms|prof)\.?\s+[A-Z][a-z]+", re.IGNORECASE)


def looks_like_real_person(text: str) -> bool:
    """Heuristic: does the text appear to name a real person."""
    return bool(REAL_NAME_PATTERN.search(text))


def load_samples(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        raise SystemExit(f"Style sample file not found: {path}")
    return path.read_text(encoding="utf-8")[:8000]


def build_user_prompt(role: str, scenario: str, action: str, style_samples: str) -> str:
    return f"""Draft a tabletop phishing email.

Target role: {role}
Scenario: {scenario}
Requested action in the email: {action}

Writing-style samples to imitate (style only, do not copy content verbatim):
---
{style_samples or "(no samples provided; use a generic clinical-administrator tone)"}
---

Required banner (first line): {BANNER}
Required footer (last line before TRAINER NOTES): [END TABLETOP EXERCISE]
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", required=True, help="Target role, e.g. 'head of radiology'.")
    p.add_argument("--scenario", required=True, help="Pretext scenario, e.g. 'after-hours PACS access'.")
    p.add_argument("--action", required=True, help="Action requested in the email.")
    p.add_argument("--style-samples", type=Path, help="Path to a text file with 1-3 short writing samples.")
    p.add_argument("--consent-confirmed", action="store_true",
                   help="Required if any input contains real-person identifiers.")
    p.add_argument("--consent-owner", default="",
                   help="Name and role of the engagement sponsor, required with --consent-confirmed.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    combined_inputs = " ".join([args.role, args.scenario, args.action])
    if looks_like_real_person(combined_inputs):
        if not args.consent_confirmed or not args.consent_owner:
            print(
                "Refusing: inputs appear to contain real-person identifiers.\n"
                "Re-run with --consent-confirmed AND --consent-owner \"Name, Role\" "
                "to confirm written authorization from the engagement sponsor.",
                file=sys.stderr,
            )
            return 2

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set in environment.", file=sys.stderr)
        return 1

    style_samples = load_samples(args.style_samples)
    user_prompt = build_user_prompt(args.role, args.scenario, args.action, style_samples)

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    output = "".join(block.text for block in message.content if block.type == "text")
    print(output)
    if args.consent_owner:
        print(f"\n[engagement sponsor on record: {args.consent_owner}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
