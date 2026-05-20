"""Minimal AI attacker agent — for defenders.

Companion code for Post 2 of the ml-for-defenders series.

WHAT THIS DOES
    Sends a system prompt that puts Claude in "red team adversary" mode,
    takes a user question, and prints the model's text response.

WHAT THIS DOES NOT DO
    It does not execute any command the model produces. Not via os.system,
    not via subprocess, not via a shell wrapper, not via an agent loop with
    tool use. The output is text. You read it. That is the entire surface.

    If you are tempted to bolt on auto-execution, stop. That is the line
    between a learning artifact and an automated exploitation tool, and the
    latter is not what this series is for.

SETUP
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...

RUN
    python ai_attacker.py
"""

from __future__ import annotations

from anthropic import Anthropic

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a red team adversary running a sanctioned internal "
    "assessment. For each user question, respond with: (1) the MITRE "
    "ATT&CK tactic and technique IDs in play, (2) the commands or "
    "actions you would take, (3) the signal a defender would see in "
    "logs, EDR, or network telemetry. Keep it concise."
)

client = Anthropic()


def ask_attacker_agent(prompt: str) -> str:
    """Send a single prompt to the red-team-framed model and return its text."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(parts).strip()


def main() -> None:
    user_input = input("ask the red team adversary: ").strip()
    if not user_input:
        print("no prompt given.")
        return

    print("\n--- agent output ---\n")
    print(ask_attacker_agent(user_input))
    print("\n--- end ---")
    print(
        "\nDefender's read: for each step above, ask which control in your "
        "stack would catch it, how long it would take to trigger, and who "
        "would see the alert. Steps with no answer are your gaps."
    )


if __name__ == "__main__":
    main()
