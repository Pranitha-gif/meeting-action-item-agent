"""
Meeting Action Item Agent
-------------------------
Reads a meeting transcript, extracts action items (task, owner, due date,
priority), validates them, and saves the result to CSV + Markdown.

Pipeline (the "agent loop"):
    1. Load transcript
    2. Extract  -> LLM (Claude) if ANTHROPIC_API_KEY is set, else rule-based fallback
    3. Validate -> fix missing owners / bad dates / bad priorities
    4. Save     -> action_items.csv and action_items.md

Usage:
    python agent.py sample_transcript.txt
    python agent.py sample_transcript.txt --meeting-date 2026-10-08
    python agent.py sample_transcript.txt --no-llm
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
PRIORITIES = {"high", "medium", "low"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"]


@dataclass
class ActionItem:
    task: str
    owner: str = "Unassigned"
    due_date: str = ""          # ISO format YYYY-MM-DD or empty
    priority: str = "medium"    # high | medium | low
    source_line: str = ""


# ---------------------------------------------------------------- Step 1: load
def load_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        sys.exit("Transcript is empty.")
    return text


# ------------------------------------------------------------ Step 2a: LLM
SYSTEM_PROMPT = """You extract action items from meeting transcripts.
Return ONLY a JSON array, no markdown fences, no commentary.
Each element must have these keys:
  "task": short imperative sentence describing what must be done
  "owner": person responsible, or "Unassigned" if not stated
  "due_date": exact date as YYYY-MM-DD if stated or clearly implied
              (use the meeting date to resolve words like "Friday" or
              "next week"), otherwise ""
  "priority": "high", "medium", or "low"
  "source_line": the transcript line the item came from
Only include real commitments or assignments, not general discussion."""


def extract_with_llm(transcript: str, meeting_date: date) -> list[ActionItem]:
    import anthropic  # imported here so --no-llm works without the package

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Meeting date: {meeting_date.isoformat()}\n\n"
                       f"Transcript:\n{transcript}",
        }],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    data = json.loads(raw)
    return [
        ActionItem(
            task=str(d.get("task", "")).strip(),
            owner=str(d.get("owner", "Unassigned")).strip() or "Unassigned",
            due_date=str(d.get("due_date", "")).strip(),
            priority=str(d.get("priority", "medium")).strip().lower(),
            source_line=str(d.get("source_line", "")).strip(),
        )
        for d in data if d.get("task")
    ]


# ------------------------------------------------- Step 2b: rule-based fallback
TRIGGERS = re.compile(
    r"\b(will|to do|todo|action item|needs to|need to|should|must|"
    r"i'll|we'll|take care of|responsible for|follow up|please)\b",
    re.I,
)
SPEAKER = re.compile(r"^\s*([A-Z][A-Za-z\.\-']+(?: [A-Z][A-Za-z\.\-']+)?)\s*:\s*(.*)$")
NAME_BEFORE_MODAL = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:will|should|needs to|must|to)\b")


def resolve_due_date(text: str, meeting_date: date) -> str:
    t = text.lower()
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    if iso:
        return iso.group(1)
    if "today" in t or "end of day" in t or "eod" in t:
        return meeting_date.isoformat()
    if "tomorrow" in t:
        return (meeting_date + timedelta(days=1)).isoformat()
    if "next week" in t:
        return (meeting_date + timedelta(days=7)).isoformat()
    if "end of week" in t or "eow" in t:
        days = (4 - meeting_date.weekday()) % 7
        return (meeting_date + timedelta(days=days)).isoformat()
    for i, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}\b", t):
            days = (i - meeting_date.weekday()) % 7 or 7
            return (meeting_date + timedelta(days=days)).isoformat()
    return ""


def guess_priority(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("urgent", "asap", "critical", "immediately", "blocker")):
        return "high"
    if any(w in t for w in ("when possible", "nice to have", "low priority", "eventually")):
        return "low"
    return "medium"


def extract_rule_based(transcript: str, meeting_date: date) -> list[ActionItem]:
    items = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SPEAKER.match(line)
        speaker, body = (m.group(1), m.group(2)) if m else ("", line)
        if not TRIGGERS.search(body):
            continue

        owner = "Unassigned"
        named = NAME_BEFORE_MODAL.search(body)
        if named and named.group(1) not in {"We", "I", "You", "They", "It", "He", "She"}:
            owner = named.group(1)
        elif re.search(r"\b(i'll|i will|i can)\b", body, re.I) and speaker:
            owner = speaker

        items.append(ActionItem(
            task=body.rstrip(".") ,
            owner=owner,
            due_date=resolve_due_date(body, meeting_date),
            priority=guess_priority(body),
            source_line=line,
        ))
    return items


# ----------------------------------------------------------- Step 3: validate
def validate(items: list[ActionItem], meeting_date: date) -> list[ActionItem]:
    clean, seen = [], set()
    for it in items:
        it.task = it.task.strip()
        if len(it.task) < 5:
            continue
        key = it.task.lower()
        if key in seen:                      # drop duplicates
            continue
        seen.add(key)

        if it.priority not in PRIORITIES:
            it.priority = "medium"
        if not it.owner:
            it.owner = "Unassigned"
        if it.due_date:
            try:
                d = datetime.strptime(it.due_date, "%Y-%m-%d").date()
                if d < meeting_date:         # due date before the meeting is wrong
                    it.due_date = ""
            except ValueError:
                it.due_date = ""
        clean.append(it)

    order = {"high": 0, "medium": 1, "low": 2}
    clean.sort(key=lambda x: (order[x.priority], x.due_date or "9999"))
    return clean


# -------------------------------------------------------------- Step 4: save
def save_csv(items: list[ActionItem], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(ActionItem("x")).keys()))
        w.writeheader()
        for it in items:
            w.writerow(asdict(it))


def save_markdown(items: list[ActionItem], path: str, meeting_date: date) -> None:
    lines = [f"# Action Items ({meeting_date.isoformat()})", "",
             "| # | Task | Owner | Due | Priority |",
             "|---|------|-------|-----|----------|"]
    for i, it in enumerate(items, 1):
        lines.append(f"| {i} | {it.task} | {it.owner} | "
                     f"{it.due_date or '-'} | {it.priority} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- main
def run(transcript_path: str, meeting_date: date, use_llm: bool, out_dir: str):
    transcript = load_transcript(transcript_path)

    items, mode = [], "rule-based"
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        try:
            items, mode = extract_with_llm(transcript, meeting_date), "Claude"
        except Exception as e:  # network, bad JSON, etc.
            print(f"[warn] LLM extraction failed ({e}); using rule-based fallback.")
    if not items:
        items = extract_rule_based(transcript, meeting_date)

    items = validate(items, meeting_date)

    os.makedirs(out_dir, exist_ok=True)
    save_csv(items, os.path.join(out_dir, "action_items.csv"))
    save_markdown(items, os.path.join(out_dir, "action_items.md"), meeting_date)

    print(f"Extractor: {mode} | {len(items)} action items saved to {out_dir}/")
    for it in items:
        print(f" - [{it.priority}] {it.owner}: {it.task} (due {it.due_date or 'n/a'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Meeting Action Item Agent")
    p.add_argument("transcript", help="Path to a .txt transcript")
    p.add_argument("--meeting-date", default=date.today().isoformat(),
                   help="YYYY-MM-DD (default: today)")
    p.add_argument("--no-llm", action="store_true", help="Force rule-based mode")
    p.add_argument("--out", default="output", help="Output folder")
    a = p.parse_args()
    run(a.transcript, datetime.strptime(a.meeting_date, "%Y-%m-%d").date(),
        not a.no_llm, a.out)
