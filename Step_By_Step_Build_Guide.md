# Hybrid Agentic Resume Screener: Complete Step-by-Step Build Guide (v2.0)

Welcome! This guide contains the step-by-step instructions and code snippets you'll need to build the complete **Hybrid Agentic Resume Screener** from scratch with the **v2.0 Architecture Updates (5 Critical Fixes Applied)**.

## Prerequisites
- Python 3.10+
- An API Key from [Groq](https://console.groq.com/keys)
- PageIndex uses OpenAI-compatible calls — we point it at Groq via `.env`, so **no separate VectifyAI key is needed**

---

## Module 1: Environment Setup & Project Scaffold

### 1.1 Folder Structure
Create this structure in your main folder:
```
hybrid-resume-screener/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── fact_checker.py
│   │   ├── hybrid_agent.py      # NEW v2
│   │   ├── init_node.py         # NEW v2
│   │   ├── llm_factory.py
│   │   ├── pageindex_agent.py
│   │   ├── planner.py
│   │   ├── router.py
│   │   ├── scorer.py
│   │   └── vector_rag_agent.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── pageindex_retriever.py
│   │   ├── pdf_parser.py
│   │   └── vector_rag.py
│   ├── graph/
│   │   ├── __init__.py
│   │   └── graph.py
│   ├── state/
│   │   ├── __init__.py
│   │   └── state.py             # UPDATED v2
│   ├── __init__.py
│   └── config.py
├── api/
│   └── main.py
├── frontend/
│   └── streamlit_app.py         # UPDATED v2
├── tests/
│   ├── __init__.py
│   └── test_tools.py
├── .env
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

### 1.2 Python Environment
Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 1.3 `requirements.txt`
```text
langchain==0.2.16
langchain-community==0.2.16
langgraph==0.2.21
langsmith==0.1.99
groq==0.11.0
langchain-groq==0.1.9
sentence-transformers==3.1.1
faiss-cpu==1.8.0
pymupdf==1.24.10
pdfplumber==0.11.4
PyPDF2==3.0.1
fastapi==0.115.0
uvicorn==0.30.6
streamlit==1.39.0
python-multipart==0.0.12
pydantic==2.9.2
pydantic-settings==2.5.2
python-dotenv==1.0.1
httpx==0.27.2
tenacity==9.0.0
pageindex
```
Install them: `pip install -r requirements.txt`

### 1.4 `.env`
```env
GROQ_API_KEY=your_groq_key_here
CHATGPT_API_KEY=your_groq_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=hybrid-resume-screener
```

---

## Module 2: Shared State Definition

### 2.1 `src/state/state.py` (Updated v2)
This state flows through every node in LangGraph. Caching indexes prevents slow re-builds.
```python
from typing import TypedDict, Optional, List, Dict, Any

class SkillMatch(TypedDict):
    skill: str
    matched: bool
    evidence: str

class GapItem(TypedDict):
    missing_skill: str
    suggestion: str
    priority: str

class ReasoningTrace(TypedDict):
    step: str
    agent: str
    input_summary: str
    output_summary: str
    method_used: str

class RetrievalComparison(TypedDict):
    task: str
    vector_result: str
    pageindex_result: str
    method_chosen: str

class ScreenerState(TypedDict):
    resume_text: str
    resume_pdf_path: str
    job_description: str
    user_query: Optional[str]
    vector_store: Optional[Any]
    pageindex_index: Optional[Any]
    sub_tasks: List[str]
    current_task: Optional[str]
    router_decision: Optional[str]
    vector_results: List[str]
    pageindex_results: Dict[str, Any]
    combined_context: str
    retrieval_comparison: List[RetrievalComparison]
    ats_score: Optional[int]
    skill_matches: List[SkillMatch]
    gaps: List[GapItem]
    summary_report: Optional[str]
    reasoning_traces: List[ReasoningTrace]
    iteration_count: int
    error: Optional[str]
    awaiting_human_input: bool
    human_feedback: Optional[str]
```

---

## Module 3: Core Tools

All tools are implemented as **classes** for clean encapsulation, reusability, and easier testing.

### 3.1 `src/tools/pdf_parser.py`
Extracts raw text from PDF resumes using a multi-parser fallback strategy. The `PDFParser` class wraps three different PDF libraries and tries each one in order, returning the first result with meaningful content.
```python
import fitz, pdfplumber
from pathlib import Path

class PDFParser:
    def __init__(self):
        self.methods = {
            'pymupdf': self.parse_pdf_pymupdf,
            'pdfplumber': self.parse_pdf_pdfplumber,
            'pypdf2': self.parse_pdf_pypdf2,
        }

    def parse_pdf_pymupdf(self, file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def parse_pdf_pdfplumber(self, file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def parse_pdf_pypdf2(self, file_path: str) -> str:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def parse_resume_pdf(self, file_path: str) -> str:
        """Try each parser in order; return the first result with >100 chars."""
        path = str(Path(file_path).resolve())
        for name, fn in self.methods.items():
            try:
                text = fn(path)
                if len(text.strip()) > 100:
                    return text
            except Exception:
                continue
        raise ValueError(f'All PDF parsers failed for {path}')
```

**Key design decisions:**
- Individual parsers **raise exceptions** on failure (no silent error strings) — error handling is centralized in `parse_resume_pdf`.
- `self.methods` dict makes the parser list extensible — add a new parser and it's automatically included in the fallback chain.
- Returns `str` (just the text), not a tuple — keeps downstream consumption simple.

### 3.2 `src/tools/vector_rag.py`
Vector Retrieval using FAISS and HuggingFace embeddings. The `VectorRag` class initializes the text splitter and embedding model once, then reuses them across calls.
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorRag:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " "]
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )

    def build_vector_index(self, resume_text: str):
        """Split text into chunks, embed, and return a FAISS vector store."""
        chunks = self.text_splitter.split_text(resume_text)
        docs = [Document(page_content=c, metadata={'chunk_id': i})
                for i, c in enumerate(chunks)]
        return FAISS.from_documents(
            documents=docs,
            embedding=self.embeddings
        )

    def retrieve_relevant_chunks(self, query: str, vector_store, k: int = 5):
        """Retrieve the top-k most similar chunks from the vector store."""
        results = vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
```

**Key design decisions:**
- Embedding model and text splitter are initialized **once** in `__init__`, not on every call — avoids expensive re-loading.
- `chunk_size=1000` with `chunk_overlap=200` gives good coverage for resume sections.
- `normalize_embeddings=True` ensures cosine similarity works correctly.

### 3.3 `src/tools/pageindex_retriever.py`

> **⚠️ IMPORTANT: How PageIndex Actually Works**
>
> PageIndex is **NOT** a query engine with a `.query()` method. It is a **tree-building tool**.
> 1. You call `page_index_main(pdf_path, config)` → it returns a **JSON tree** (hierarchical table-of-contents with summaries).
> 2. To "retrieve" information, you pass the JSON tree + your question to an **LLM** and let the LLM reason over the tree structure.
> 3. PageIndex internally uses `CHATGPT_API_KEY` and `OPENAI_BASE_URL` from `.env` — so pointing those at Groq makes it free.

```python
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Build the PageIndex tree from a PDF
# ---------------------------------------------------------------------------
def build_pageindex_tree(pdf_path: str, model: str = None) -> dict:
    """
    Uses the real VectifyAI/PageIndex library to build a hierarchical
    JSON tree from a PDF. The tree is like a smart table-of-contents
    with summaries for each section.

    Returns a dict (the JSON tree structure).
    """
    from pageindex import config as pi_config, page_index_main
    import os

    # Use model from env or fall back to default
    model_name = model or os.getenv('OPENAI_MODEL', 'llama-3.1-8b-instant')

    opt = pi_config(
        model=model_name,
        toc_check_page_num=20,           # pages to check for existing TOC
        max_page_num_each_node=10,       # max pages per tree node
        max_token_num_each_node=20000,   # max tokens per tree node
        if_add_node_id='yes',
        if_add_node_summary='yes',       # critical: gives each node a summary
        if_add_doc_description='yes',
        if_add_node_text='no',           # keep tree compact
    )

    logger.info(f'Building PageIndex tree for: {pdf_path}')
    tree = page_index_main(str(Path(pdf_path).resolve()), opt)
    logger.info('PageIndex tree built successfully.')
    return tree


# ---------------------------------------------------------------------------
# Step 2: Query the tree using LLM reasoning (this is "vectorless retrieval")
# ---------------------------------------------------------------------------
def pageindex_query(tree: dict, query: str, llm=None) -> Dict[str, Any]:
    """
    Reason over the PageIndex JSON tree using an LLM to find the answer.
    This is the core of "vectorless RAG" — no embeddings, no vector store.
    The LLM navigates the tree hierarchy like a human expert would.
    """
    try:
        # Flatten tree to a readable string for the LLM
        tree_str = json.dumps(tree, indent=2, default=str)

        # Truncate if tree is very large (for context window limits)
        if len(tree_str) > 15000:
            tree_str = tree_str[:15000] + '\n... [truncated]'

        prompt = f"""You are given a hierarchical document index (like a smart table of contents)
built from a resume PDF. Each node has a title, summary, and page range.

Document Index:
{tree_str}

Question: {query}

Instructions:
1. Navigate the tree to find the most relevant section(s).
2. Use the summaries and structure to reason about the answer.
3. Cite which section(s) you used (by title and node_id).

Respond in this JSON format:
{{
    "answer": "<your detailed answer>",
    "evidence": "<relevant quotes or summaries from the tree>",
    "sections_used": ["<section title 1>", "<section title 2>"]
}}

Return ONLY valid JSON. No preamble."""

        if llm is None:
            from src.agents.llm_factory import get_llm, safe_invoke
            llm = get_llm(smart=True)
            response_text = safe_invoke(llm, prompt, sleep=2.0)
        else:
            response_text = llm.invoke(prompt).content

        # Parse LLM response
        import re
        m = re.search(r'\{.*\}', response_text, re.DOTALL)
        if m:
            result = json.loads(m.group())
        else:
            result = {'answer': response_text, 'evidence': '', 'sections_used': []}

        return result

    except Exception as e:
        logger.error(f'PageIndex query failed: {e}')
        return {'answer': '', 'evidence': '', 'sections_used': [], 'error': str(e)}


def extract_years_of_experience(tree: dict, llm=None) -> Dict[str, Any]:
    """Convenience function to extract work experience from the tree."""
    return pageindex_query(
        tree,
        'How many total years of professional work experience does this '
        'person have? List each role with dates and company names.',
        llm=llm
    )


def extract_technical_skills(tree: dict, llm=None) -> Dict[str, Any]:
    """Convenience function to extract skills from the tree."""
    return pageindex_query(
        tree,
        'List all technical skills, programming languages, frameworks, '
        'tools, and certifications mentioned in this resume.',
        llm=llm
    )
```

---

## Module 4: Agent Nodes

### 4.1 `src/agents/init_node.py`
Initializes both FAISS and PageIndex **once** so all downstream agents reuse them from state.
Note how it instantiates the `PDFParser` and `VectorRag` classes, then calls their methods.
```python
from src.tools.pdf_parser import PDFParser
from src.tools.vector_rag import VectorRag
from src.tools.pageindex_retriever import build_pageindex_tree
from src.state.state import ScreenerState
import logging

logger = logging.getLogger(__name__)

# Instantiate tool classes once at module level
pdf_parser = PDFParser()
vector_rag = VectorRag()

def init_node(state: ScreenerState) -> dict:
    # Parse the resume PDF (uses multi-parser fallback)
    logger.info('Parsing resume PDF...')
    resume_text = pdf_parser.parse_resume_pdf(state['resume_pdf_path'])

    # Build FAISS vector index from extracted resume text
    logger.info('Building FAISS vector index...')
    vector_store = vector_rag.build_vector_index(resume_text)

    # Build PageIndex JSON tree from the raw PDF file
    logger.info('Building PageIndex tree from PDF...')
    pageindex_tree = build_pageindex_tree(state['resume_pdf_path'])

    logger.info('Both indexes built and cached in state.')
    return {
        'resume_text': resume_text,
        'vector_store': vector_store,
        'pageindex_index': pageindex_tree,
    }
```

### 4.2 `src/agents/router.py`
Uses an LLM classification rather than a brittle keyword map.
```python
from src.state.state import ScreenerState, ReasoningTrace
from src.agents.llm_factory import get_llm, safe_invoke

ROUTER_PROMPT = '''
Classify this resume screening task into ONE retrieval method.
Task: {task}
Rules:
- Return 'pageindex' if the task asks for structured facts...
- Return 'vector' if the task asks for semantic skill matching...
- Return 'hybrid' if the task requires both...
Return ONLY one word: pageindex | vector | hybrid
'''

def router_node(state: ScreenerState) -> dict:
    idx = state['iteration_count']
    task = state['sub_tasks'][idx] if idx < len(state['sub_tasks']) else None
    if not task:
        return {'current_task': None, 'router_decision': 'done', 'iteration_count': idx + 1}
    
    llm = get_llm(smart=False)
    decision = safe_invoke(llm, ROUTER_PROMPT.format(task=task)).strip().lower()
    
    if decision not in ('vector', 'pageindex', 'hybrid'):
        decision = 'hybrid'
        
    trace = ReasoningTrace(step=f'route_{idx}', agent='router',
        input_summary=f'Task: {task}',
        output_summary=f'LLM classified as: {decision}',
        method_used='llm_classifier')
        
    return {'current_task': task, 'router_decision': decision,
            'iteration_count': idx + 1,
            'reasoning_traces': state.get('reasoning_traces', []) + [trace]}
```

### 4.3 `src/agents/hybrid_agent.py`
Runs **both** FAISS and PageIndex retrieval, then stores side-by-side comparison data.
Uses the `VectorRag` class instance for vector retrieval.
```python
from src.tools.vector_rag import VectorRag
from src.tools.pageindex_retriever import pageindex_query
from src.state.state import ScreenerState, RetrievalComparison

vector_rag = VectorRag()

def hybrid_agent_node(state: ScreenerState) -> dict:
    # Vector retrieval (semantic chunks)
    vec_results = vector_rag.retrieve_relevant_chunks(
        state['current_task'], state['vector_store'], k=4)

    # PageIndex retrieval (LLM reasons over the JSON tree)
    pi_result = pageindex_query(
        state['pageindex_index'],   # This is the JSON tree dict
        state['current_task'])

    comparison = RetrievalComparison(
        task=state['current_task'],
        vector_result=vec_results[0][:200] if vec_results else 'No results',
        pageindex_result=str(pi_result.get('answer',''))[:200],
        method_chosen='hybrid'
    )

    updated_pi = dict(state.get('pageindex_results', {}))
    updated_pi[state['current_task']] = pi_result

    return {
        'vector_results': state.get('vector_results', []) + vec_results,
        'pageindex_results': updated_pi,
        'retrieval_comparison': state.get('retrieval_comparison', []) + [comparison],
    }
```

---

## Module 5: LangGraph Assembly

### 5.1 `src/graph/graph.py`
Wires the intelligent workflow paths together.
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.state.state import ScreenerState
from src.agents.init_node import init_node
from src.agents.planner import planner_node
from src.agents.router import router_node
from src.agents.vector_rag_agent import vector_rag_agent_node
from src.agents.pageindex_agent import pageindex_agent_node
from src.agents.hybrid_agent import hybrid_agent_node
from src.agents.scorer import scorer_node
from src.agents.fact_checker import fact_checker_node

MAX_ITERATIONS = 10

def should_continue(state: ScreenerState) -> str:
    if state['iteration_count'] >= MAX_ITERATIONS:  return 'score'
    if state['iteration_count'] >= len(state['sub_tasks']): return 'score'
    decision = state.get('router_decision', 'vector')
    return decision if decision in ('vector','pageindex','hybrid') else 'score'

def after_retrieval(state: ScreenerState) -> str:
    if state['iteration_count'] >= len(state['sub_tasks']): return 'score'
    return 'route'

def build_graph():
    graph = StateGraph(ScreenerState)
    graph.add_node('init', init_node)
    graph.add_node('plan', planner_node)
    graph.add_node('route', router_node)
    graph.add_node('vector', vector_rag_agent_node)
    graph.add_node('pageindex', pageindex_agent_node)
    graph.add_node('hybrid', hybrid_agent_node)
    graph.add_node('score', scorer_node)
    graph.add_node('fact_check', fact_checker_node)
    
    graph.set_entry_point('init')
    graph.add_edge('init', 'plan')
    graph.add_edge('plan', 'route')
    
    graph.add_conditional_edges('route', should_continue, {
        'vector': 'vector', 'pageindex': 'pageindex', 'hybrid': 'hybrid', 'score': 'score'
    })
    
    for node in ('vector', 'pageindex', 'hybrid'):
        graph.add_conditional_edges(node, after_retrieval, {'route': 'route', 'score': 'score'})
        
    graph.add_edge('score', 'fact_check')
    graph.add_edge('fact_check', END)
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory, interrupt_before=['fact_check'])
```

---

## How PageIndex + Groq Works (Key Concept)

PageIndex internally uses `openai.OpenAI(api_key=CHATGPT_API_KEY)` to make LLM calls. Since we set these in `.env`:
```env
CHATGPT_API_KEY=your_groq_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
```
PageIndex automatically routes its LLM calls through **Groq** instead of OpenAI. This means:
- ✅ No OpenAI API key needed
- ✅ No GPU needed locally
- ✅ Free tier Groq is sufficient
- ⚠️ Groq free tier has a 30 RPM limit — the `safe_invoke()` function adds `time.sleep()` to respect this

---

*Refer to the full `guide_v2.txt` for the rest of the endpoints (FastAPI, Streamlit, Docker) if further code clarity is needed.*
