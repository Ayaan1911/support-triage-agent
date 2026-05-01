# Multi-Domain Support Triage Agent
> HackerRank Orchestrate Hackathon — May 2026 | Team: Runtime Terror

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Groq](https://img.shields.io/badge/Groq-f55036?style=for-the-badge&logo=groq&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)

## What This Does
A terminal-based AI agent that automatically triages support tickets across 3 domains — HackerRank, Claude, and Visa — using semantic search over a pre-built support corpus and a grounded LLM for response generation.

For each ticket it outputs:
- `status`: replied or escalated
- `product_area`: relevant support category
- `response`: answer grounded strictly in the support corpus
- `justification`: reasoning behind the decision
- `request_type`: product_issue | feature_request | bug | invalid

## Architecture
```
Corpus (data/) → Semantic Index (sentence-transformers) → Triage Agent (Groq LLaMA-3.3-70b) → output.csv
```
4-stage pipeline:

1. **crawler.py** — optional live crawler to fetch support docs (corpus pre-provided)
2. **retriever.py** — chunks docs into 500-char segments, embeds with all-MiniLM-L6-v2, cosine similarity retrieval
3. **agent.py** — safety checks → domain inference → retrieval → grounded LLM response
4. **main.py** — CLI orchestration with tqdm progress bar

## Safety Features
| Feature | Implementation |
|---------|----------------|
| Prompt injection detection | Regex (multilingual — catches French adversarial inputs) |
| Harmful request blocking | Pattern match before any LLM call |
| Fraud / identity theft | Auto-escalate |
| Billing disputes | Auto-escalate |
| Score review requests | Auto-escalate |
| Domain inference | Keyword clustering for company=None tickets |
| Zero hallucination | Strict corpus-grounded system prompt, temperature=0.0 |

## Setup & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY="your_key"        # Mac/Linux
$env:GROQ_API_KEY="your_key"          # Windows PowerShell

# Build semantic index from corpus
python main.py --build-index

# Run agent on support tickets
python main.py --run

# Full pipeline (index + run)
python main.py --all
```

## Determinism
```bash
export PYTHONHASHSEED=42
```
`temperature=0.0` is set in all LLM calls.

## Tech Stack
- **Groq API** — llama-3.3-70b-versatile (free tier, ultra-fast)
- **sentence-transformers** — all-MiniLM-L6-v2 semantic search
- **scikit-learn** — cosine similarity
- **pandas** — CSV I/O
- **tqdm** — progress bar
- **beautifulsoup4 + requests** — optional crawler

---
**Author:** Mohammad Ayaan | BTech CSE Cyber Security | MSRIT Bengaluru