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