# Meeting Action Item Agent

Turns a meeting transcript into a prioritized action item list (CSV + Markdown).

## Setup
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key      # optional; without it the rule-based mode runs

## Run
    python agent.py sample_transcript.txt --meeting-date 2026-10-08
    python agent.py sample_transcript.txt --no-llm

Outputs go to `output/action_items.csv` and `output/action_items.md`.

## How it works
1. Load transcript
2. Extract items (Claude, or regex fallback)
3. Validate: dedupe, fix dates/priorities, sort by priority then due date
4. Save CSV and Markdown

## Ideas to extend
- Post items to Slack / email / Jira
- Accept audio and transcribe with Whisper
- Store items in SQLite and track completion
- Add a Streamlit UI
