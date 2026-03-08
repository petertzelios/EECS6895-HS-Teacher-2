# EECS6895-HS-Teacher-2
Midterm project for EECS6895. Group: High School Teacher 2. The goal of the project is to make an AI agentic system which can act as a high school teacher.


This bundle includes:
- `notebooks/hs_teacher_multi_agent_notebook_v5-3.ipynb`
- `notebooks/midterm_hsteacher2_v3-2.ipynb`
- `scripts/multi_agent_gpt.py`
- `scripts/build_faiss_indexes.py`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_KEY
```

## Build indexes

```bash
python scripts/build_faiss_indexes.py   --root-dir /path/to/midterm_data_clean   --index-dir /path/to/indexes_hs_teacher_clean   --force-rebuild
```

## Run the GPT multi-agent assistant

```bash
export INDEX_DIR=/path/to/indexes_hs_teacher_clean
python scripts/multi_agent_gpt.py   "Give me a Regents-style Algebra I practice question about systems of equations."   --show-debug
```
