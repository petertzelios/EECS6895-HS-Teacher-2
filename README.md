# EECS6895-HS-Teacher-2
Midterm project for EECS6895. Group: High School Teacher 2. The goal of the project is to build an agentic AI high-school teacher assistant.

## Main components

- `multi_agent_gpt.py`: CLI entry point for the modular GPT-based assistant
- `pipeline_core.py`: routing + answer-generation logic
- `retrieval_agent.py`: FAISS retrieval, dedup fix, and optional PageIndex/OpenViking refinement
- `demo_server.py`: local terminal-style demo UI server
- `evaluate_models.py`: benchmark runner for OpenAI/DeepSeek with and without RAG
- `build_faiss_indexes.py`: base index builder for curriculum + Regents materials
- `index.html`: frontend used by the demo server

## Agents

- `regents_agent`: Regents questions, scoring guides, rubrics, answer explanations
- `curriculum_agent`: curriculum/module/lesson teaching support
- `college_support_agent`: college application + time-management guidance

## Index layout

Base indexes are expected in `INDEX_DIR`:
- `curriculum_overview`
- `exam_questions`
- `exam_scoring`

Student-support indexes are expected in `STUDENT_SUPPORT_INDEX_DIR`:
- `college_info`
- `time_management`

If the student-support indexes are missing, the code now skips them gracefully. The college agent will still run, but RAG evidence for that agent will be empty until those indexes are built and placed in `STUDENT_SUPPORT_INDEX_DIR`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_KEY
```

## Building FAISS Indexes

Two sets of indexes are required.

### 1. Teacher / curriculum indexes

Build the curriculum and Regents indexes:

python build_faiss_indexes.py \
  --root-dir midterm_data_clean \
  --index-dir indexes \
  --force-rebuild

This produces:

- curriculum_overview.faiss
- exam_questions.faiss
- exam_scoring.faiss

### 2. Student-support indexes

Build the college and time-management indexes:

python build_student_support_indexes.py \
  --root-dir midterm_data \
  --index-dir indexes_student_support \
  --force-rebuild

This produces:

- college_info.faiss
- time_management.faiss


## Run the GPT multi-agent assistant

```bash
export INDEX_DIR=/path/to/indexes_hs_teacher_clean
export STUDENT_SUPPORT_INDEX_DIR=/path/to/indexes_student_support
python multi_agent_gpt.py   "I'm in 11th grade and it's March. What should I do now for college applications?"   --show-debug
```

## Run the demo UI

```bash
export OPENAI_API_KEY=YOUR_KEY
export INDEX_DIR=/path/to/indexes_hs_teacher_clean
export STUDENT_SUPPORT_INDEX_DIR=/path/to/indexes_student_support
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export DEEPSEEK_API_KEY=YOUR_DEEPSEEK_KEY
export RETRIEVAL_EXPERIMENTAL_BACKENDS=pageindex,openviking
export PAGEINDEX_MODELS=gpt-4o
export EMBED_DEVICE=cuda

python demo_server.py --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

## Run evaluation

```bash
export INDEX_DIR=/path/to/indexes_hs_teacher_clean
export STUDENT_SUPPORT_INDEX_DIR=/path/to/indexes_student_support
python evaluate_models.py --benchmark benchmark_cases.json --output eval_results.json
```
