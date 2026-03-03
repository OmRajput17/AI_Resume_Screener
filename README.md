# Hybrid Agentic Resume Screener (v2.0)

A production-ready AI system that ingests a resume PDF and a job description, then intelligently screens the candidate using a hybrid Retrieval-Augmented Generation (RAG) pipeline. This project combines Vector RAG (FAISS) and Vectorless RAG (PageIndex) side-by-side, orchestrated by LangGraph.

## Features
- **Hybrid RAG Architecture**: Combines traditional semantic similarity search (FAISS) with structured, hierarchical retrieval (PageIndex).
- **Agentic Routing**: Uses an LLM router via LangGraph to determine the best retrieval method per screening task (vector, pageindex, or hybrid).
- **Explainable AI**: Provides an ATS compatibility score along with a detailed gap analysis and completely transparent reasoning traces.
- **Side-by-Step Retrieval Comparison**: A Streamlit UI panel that shows exactly what each retriever found and which method the router selected for full transparency.
- **Production Ready**: Includes FastAPI backend, Docker containerization, Streamlit UI, and LangSmith observability integration.

## Architecture

The LangGraph system consists of 8 specialized nodes:
1. **Init Node**: Caches the FAISS index and PageIndex tree upfront (Performance Fix).
2. **Planner Node**: Generates core evaluation questions based on the job description.
3. **Router Node**: Uses LLM-based classification to route tasks.
4. **Vector RAG Agent**: Standard semantic chunk retrieval.
5. **PageIndex Agent**: Structured fact extraction.
6. **Hybrid Agent**: Runs both retrievers and combines results.
7. **Scorer Node**: Evaluates the gathered evidence to produce an ATS score and gap analysis.
8. **Fact-Checker Node**: Verifies the final summary against structured facts to prevent hallucinations.

## Quickstart

1. **Clone & Setup Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

2. **Configure Variables**  
Create a `.env` file with:
```env
GROQ_API_KEY=your_groq_key_here
CHATGPT_API_KEY=your_groq_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=hybrid-resume-screener
```

3. **Run Locally**
- Backend: `uvicorn api.main:app --reload`
- Frontend: `streamlit run frontend/streamlit_app.py`

4. **Run via Docker**
```bash
docker-compose up --build
```
