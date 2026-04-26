# NLP-Project-6320
## NLP Project: R.E.C.I.P.E. — Kitchen Crisis Resolver

A Knowledge-Augmented NLP framework that diagnoses and resolves cooking failures using Functional Knowledge Graphs, Retrieval-Augmented Generation, and Large Language Models.

---

## Overview

R.E.C.I.P.E. combines three knowledge components with large language models to provide scientifically grounded cooking failure diagnosis:

- **FKG** — Functional Knowledge Graph that maps recipes as Directed Acyclic Graphs with critical control points.
- **RAG** — Retrieval-Augmented Generation that retrieves relevant food science facts from a 100-fact knowledge base.
- **SKG** — Substitution Knowledge Graph that provides ingredient replacements with scientific reasoning.

---

## Overview of Project Structure
```text
NLP-PROJECT-6320/
├── data/
│   ├── testcases.json          # 60 test cases
│   ├── recipes.json            # 56 recipes
│   ├── science_facts.json      # 100 science facts
│   └── substitution_graph.json # ingredient substitutions
├── src/
│   ├── models.py               # LLM model calls
│   ├── rag.py                  # RAG retrieval system
│   ├── knowledge_graphs.py     # FKG and SKG
│   ├── evaluator.py            # scoring metrics
│   ├── experiment_runner.py    # experiment pipeline
│   └── evaluation_analysis.py  # analysis and plots
├── backend/
│   └── api.py                  # FastAPI server
├── frontend/                   # React + Vite UI
├── results/                    # experiment results
├── analysis/                   # analysis outputs
└── main.py                     # entry point
```

---

## Models

| Model         | Provider | Size |
|---            |---       |---   |
| LLaMA 3.3 70B | Groq     | 70B  |
| LLaMA 3.1 8B  | Groq     | 8B   |
| GPT-OSS 20B   | Groq     | 20B  |

---

## Experimental Conditions

| Condition    | Components                               |
|---           |---                                       |
| Baseline     | Problem + recipe steps only              |
| CoT Only     | + Chain-of-thought reasoning             |
| KG Augmented | + FKG critical steps + RAG science facts |
| Full System  | + SKG ingredient substitutions           |

---

## Evaluation Metrics

Metrics
- Cause Accuracy
- Solution Appropriateness
- Scientific Accuracy
- Overall Score

Total trials: 3 models × 4 conditions × 60 cases = **720 trials**

---

## Setup

### 1. Clone the repository
```bash
git clone <link>
cd NLP-Project-6320
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file in project root and add your Groq API key to `.env` file in this format:
```
GROQ_API_KEY=your_groq_key_here
```

---

## Running Experiments

Some basic commands: 

```bash
# Quick test — 2 cases
python main.py

# Full run — 60 cases with 7 second delay
python main.py --mode full --delay 7

# Interactive single mode
python main.py --mode single

# Analyze results
python main.py --mode analyze
```

---

## Running The Frontend

### Terminal 1 — Backend
```bash
uvicorn backend.api:app --reload
```

### Terminal 2 — Frontend
```bash
cd frontend
npm install
npm run dev
```
Open link in browser to view frontend

---

## Requirements

- Python 3.10+
- Node.js 18+
- Groq API key (free tier)
