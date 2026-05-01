# Support Triage Agent

> Multi-domain AI support triage agent for HackerRank, Claude, and Visa — built for HackerRank Orchestrate Hackathon (May 2026)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Groq](https://img.shields.io/badge/Groq-f55036?style=for-the-badge&logo=groq&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)

## Overview
The Support Triage Agent is an intelligent system designed to automatically process, classify, and respond to support tickets across multiple domains (HackerRank, Claude, and Visa). It leverages high-performance Retrieval-Augmented Generation (RAG) and the LLaMA-3.3-70b-versatile model via the Groq API to provide accurate, safe, and context-aware resolutions.

## Architecture
The system operates through a modular and robust pipeline designed for efficient ticket processing. First, `crawler.py` ingests and normalizes the raw support documentation from various domains. Next, `retriever.py` chunks the documents and uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate dense vector embeddings, enabling fast and semantic search. Then, `agent.py` evaluates the incoming tickets against safety rules, infers the relevant domain, queries the retrieved context, and leverages the Groq LLM to generate precise responses. Finally, `main.py` serves as the primary CLI entry point, coordinating the end-to-end flow from index building to batch ticket execution.

## Tech Stack
- **Python** (Core language)
- **sentence-transformers** (`all-MiniLM-L6-v2` for semantic search)
- **Groq API** (`llama-3.3-70b-versatile` for ultra-fast inference)
- **scikit-learn** (Data processing and similarities)
- **FastAPI** (Backend infrastructure compatibility)
- **pandas** (Data manipulation and CSV handling)

## Safety Features
This agent employs a robust, multi-layered safety architecture:
- **Prompt Injection Detection**: Multilingual detection (including French) to block adversarial inputs.
- **Fraud/Identity Theft Escalation**: Automatically routes sensitive incidents to human agents.
- **Billing Dispute Escalation**: Flags financial disputes for specialized review.
- **Zero Hallucination Policy**: Strict contextual grounding using retrieved documentation.
- **Exponential Backoff**: Reliable API interaction with built-in retry mechanisms.

## Determinism
To ensure fully reproducible and deterministic outputs, this project explicitly configures `PYTHONHASHSEED=42` and enforces a strict `temperature=0.0` across all LLM inference calls in the pipeline.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Groq API key and seed:**
   ```bash
   # Windows (PowerShell)
   $env:GROQ_API_KEY="your_api_key_here"
   $env:PYTHONHASHSEED="42"
   
   # Mac/Linux
   export GROQ_API_KEY="your_api_key_here"
   export PYTHONHASHSEED="42"
   ```

3. **Build the semantic index:**
   ```bash
   python main.py --build-index
   ```

4. **Run the agent:**
   ```bash
   python main.py --run
   ```

## Project Structure
```
code/
├── agent.py          # Core triage and LLM orchestration logic
├── crawler.py        # Documentation web crawler
├── main.py           # CLI entry point
├── retriever.py      # Semantic search and document retrieval
├── requirements.txt  # Project dependencies
└── README.md         # This documentation
```

---
**Author:** Mohammad Ayaan (Zero) | BTech CSE Cyber Security | MSRIT Bengaluru | Team: Runtime Terror
