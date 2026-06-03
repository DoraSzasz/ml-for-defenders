"""Heuristic AI-phishing scorer for a single email.

Reads an email from a file (.eml or plain text) or stdin and returns a score
out of 12 with a ranked list of findings.

This is a HEURISTIC. It is intended to make silent signals visible during
triage, not to make verdicts. Treat the score as a prompt for a human look,
not a block-or-allow decision.

SAFETY
------
- Does not log raw email body or headers to disk.
- Does not fetch URLs or attachments.
- All checks are local and offline.
"""
from __future__ import annotations

import argparse
import email
import re
import sys
from dataclasses import dataclass
from email.message import Message
from pathlib import Path

URGENCY_KEYWORDS = [
    "immediately", "urgent", "asap", "right away", "within 24 hours",
    "suspended", "will be locked", "final notice", "action required",
    "verify now", "act now",
]

AUTHORITY_KEYWORDS = [
    "ceo", "cmo", "cio", "ciso", "head of", "chief", "director",
    "compliance", "regulator", "auditor", "legal department",
]

GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear colleague", "dear sir or madam",
    "hello team", "hi team", "dear valued",
]

OUT_OF_BAND_PHRASES = [
    "call me on", "text me at", "switch to signal", "switch to whatsapp",
    "my personal phone", "use this number", "ping me on telegram",
    "don't email", "do not email me back",
]

SECRECY_PHRASES = [
    "don't tell", "do not tell", "keep this between us", "confidential -- do not share",
    "between you and me", "discreet", "without informing",
]

SUSPICIOUS_EXTENSIONS = {
    ".htm", ".html", ".iso", ".img", ".lnk", ".docm", ".xlsm",
    ".js", ".vbs", ".scr", ".exe", ".cmd", ".bat", ".ps1",
}

LOOKALIKE_DOMAINS_PATTERN = re.compile(
    r"\b([a-z0-9-]+)\.(com|net|org|io|co|health|care)\b", re.IGNORECASE
)

PUNYCODE_PATTERN = re.compile(r"xn--", re.IGNORECASE)

# Common confusables that show up in lookalike domains.
CONFUSABLE_TOKENS = ["0", "1", "rn", "vv", "ll", "-"]


@dataclass
class Finding:
    name: str
    detail: str
    weight: int = 1


def parse_message(raw: str) -> Message:
    return email.message_from_string(raw)


def get_body(msg: Message) -> str:
    if msg.is_multipart():
        parts: list[str] = []
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    parts.append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
        return "\n".join(parts)
    payload = msg.get_payload(decode=True)
    if isinstance(payload, bytes):
        return payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
    return str(msg.get_payload() or "")


def check_display_vs_from(msg: Message) -> Finding | None:
    raw_from = msg.get("From", "")
    m = re.match(r'\s*"?([^"<]+?)"?\s*<([^>]+)>', raw_from)
    if not m:
        return None
    display, addr = m.group(1).strip(), m.group(2).strip().lower()
    if "@" not in addr:
        return None
    addr_local, addr_domain = addr.split("@", 1)
    display_lower = display.lower()
    # Display name implies a person/brand whose domain is nowhere in the address.
    if display_lower and not any(tok in addr for tok in display_lower.split()):
        if "noreply" not in addr_local and len(display_lower) > 3:
            return Finding(
                "display_name_mismatch",
                f"Display name '{display}' does not match address domain '{addr_domain}'.",
                weight=2,
            )
    return None


def check_lookalike_domain(msg: Message) -> Finding | None:
    raw_from = msg.get("From", "")
    m = re.search(r"@([\w\.-]+)", raw_from)
    if not m:
        return None
    domain = m.group(1).lower()
    if PUNYCODE_PATTERN.search(domain):
        return Finding("punycode_domain", f"Sender domain uses punycode: {domain}.", weight=2)
    for token in CONFUSABLE_TOKENS:
        if token in domain.replace(".", ""):
            if any(c.isdigit() for c in domain.split(".")[0]) or "-" in domain.split(".")[0]:
                return Finding(
                    "possible_lookalike_domain",
                    f"Sender domain '{domain}' contains characters often used in lookalikes.",
                    weight=1,
                )
    return None


def check_reply_to_mismatch(msg: Message) -> Finding | None:
    reply_to = msg.get("Reply-To", "")
    from_addr = msg.get("From", "")
    if not reply_to:
        return None
    rt = re.search(r"@([\w\.-]+)", reply_to)
    fr = re.search(r"@([\w\.-]+)", from_addr)
    if rt and fr and rt.group(1).lower() != fr.group(1).lower():
        return Finding(
            "reply_to_mismatch",
            f"Reply-To domain '{rt.group(1)}' differs from From domain '{fr.group(1)}'.",
            weight=2,
        )
    return None


def check_auth_results(msg: Message) -> Finding | None:
    auth = msg.get("Authentication-Results", "")
    fails = [k for k in ("spf=fail", "dkim=fail", "dmarc=fail") if k in auth.lower()]
    if fails:
        return Finding("email_auth_failure", f"Header reports: {', '.join(fails)}.", weight=2)
    return None


def check_urgency(body: str) -> Finding | None:
    hits = [kw for kw in URGENCY_KEYWORDS if kw in body.lower()]
    if hits:
        return Finding("urgency_pressure", f"Urgency keywords present: {', '.join(hits[:3])}.")
    return None


def check_authority(body: str, subject: str) -> Finding | None:
    blob = (body + " " + subject).lower()
    hits = [kw for kw in AUTHORITY_KEYWORDS if kw in blob]
    if hits:
        return Finding("authority_pressure", f"Authority signals present: {', '.join(hits[:3])}.")
    return None


def check_generic_greeting(body: str) -> Finding | None:
    head = body.strip().lower()[:200]
    for g in GENERIC_GREETINGS:
        if g in head:
            return Finding("generic_greeting", f"Opens with '{g}'.")
    return None


def check_link_text_mismatch(body: str) -> Finding | None:
    # Look for "click here" or anchor-style text followed by a bracketed URL.
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    for label, url in pattern.findall(body):
        if "http" in label.lower() and label.lower().rstrip("/") not in url.lower():
            return Finding(
                "link_text_url_mismatch",
                f"Link text claims '{label}' but resolves to '{url}'.",
                weight=2,
            )
    return None


def check_attachments(msg: Message) -> Finding | None:
    bad: list[str] = []
    for part in msg.walk():
        filename = part.get_filename()
        if not filename:
            continue
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in SUSPICIOUS_EXTENSIONS:
            bad.append(filename)
    if bad:
        return Finding("suspicious_attachment", f"Attachment(s): {', '.join(bad)}.", weight=2)
    return None


def check_too_clean_prose(body: str) -> Finding | None:
    # Inverted-grammar check: very long, very regular sentences with no contractions
    # and no informal punctuation can be a weak AI-generated signal.
    sentences = re.split(r"[.!?]+\s+", body.strip())
    sentences = [s for s in sentences if len(s) > 20]
    if len(sentences) < 4:
        return None
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    contractions = sum(1 for s in sentences if "'" in s)
    if avg_len >= 18 and contractions == 0:
        return Finding(
            "uniformly_polished_prose",
            f"Avg sentence length {avg_len:.1f} words, zero contractions across {len(sentences)} sentences.",
        )
    return None


def check_out_of_band(body: str) -> Finding | None:
    hits = [p for p in OUT_OF_BAND_PHRASES if p in body.lower()]
    if hits:
        return Finding("out_of_band_channel", f"Requests channel switch: '{hits[0]}'.", weight=2)
    return None


def check_secrecy(body: str) -> Finding | None:
    hits = [p for p in SECRECY_PHRASES if p in body.lower()]
    if hits:
        return Finding("secrecy_pressure", f"Asks for secrecy: '{hits[0]}'.", weight=2)
    return None


def score_message(raw: str) -> tuple[int, int, list[Finding]]:
    msg = parse_message(raw)
    body = get_body(msg)
    subject = msg.get("Subject", "")

    checks = [
        check_display_vs_from(msg),
        check_lookalike_domain(msg),
        check_reply_to_mismatch(msg),
        check_auth_results(msg),
        check_urgency(body),
        check_authority(body, subject),
        check_generic_greeting(body),
        check_link_text_mismatch(body),
        check_attachments(msg),
        check_too_clean_prose(body),
        check_out_of_band(body),
        check_secrecy(body),
    ]

    findings = [f for f in checks if f is not None]
    score = sum(f.weight for f in findings)
    return score, len(checks), findings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="Path to a .eml or plain-text email file.")
    src.add_argument("--stdin", action="store_true", help="Read the raw email from stdin.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw = args.file.read_text(encoding="utf-8", errors="replace") if args.file else sys.stdin.read()

    score, total_checks, findings = score_message(raw)

    print(f"AI-phish score: {score} / {total_checks * 2}  ({len(findings)} of {total_checks} checks flagged)")
    print("-" * 60)
    if not findings:
        print("No heuristic indicators tripped. This is not the same as 'safe'.")
        return 0
    for f in sorted(findings, key=lambda x: -x.weight):
        print(f"  [{f.weight}] {f.name}: {f.detail}")
    print("-" * 60)
    print("Heuristic only. A high score is a prompt for a human review,")
    print("not a verdict. A zero score is not a guarantee of safety.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
