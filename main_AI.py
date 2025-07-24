import streamlit as st
from dotenv import load_dotenv
import os
from typing import Annotated, List
import operator

from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END

# Load environment
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
model = ChatGroq(model="llama-3.3-70b-versatile")

# Define structured output schemas
class Section(BaseModel):
    title: str = Field(description="Title for this section of the report based on the text")
    desc: str = Field(description="Brief overview of the main topics and concepts of the section")

class Sections(BaseModel):
    sections: list[Section] = Field(description="Sections of the report")

planner = model.with_structured_output(Sections)

class State(TypedDict):
    title: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

# Workflow nodes
def orchestrator(state: State):
    sections = planner.invoke([
        SystemMessage(content="Generate sections plan of report based on the topic"),
        HumanMessage(content=f"Here is the report Topic : {state['title']}")
    ])
    return {"sections": sections.sections}

def llm_call(state: WorkerState):
    section = model.invoke([
        SystemMessage(content="Write a report section following the provided name and description. Write the report in professional researcher style. Include no preamble for each section. Use markdown formatting."),
        HumanMessage(content=(f"Here is the section name: {state['section'].title} and description: {state['section'].desc}"))
    ])
    return {"completed_sections": [section.content]}

def assign_worker(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

def synthesizer(state: State):
    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}

# Build the LangGraph workflow
orchestrator_worker_builder = StateGraph(State)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_worker, ["llm_call"])
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)
orchestrator_worker = orchestrator_worker_builder.compile()

# =========================
# Streamlit UI Starts Here
# =========================
st.set_page_config(page_title="AI Report Generator", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
        .report-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #4CAF50;
            margin-bottom: 0.2rem;
        }
        .sub-text {
            font-size: 1.2rem;
            color: #6c757d;
            margin-bottom: 2rem;
        }
        .stTextInput>div>div>input {
            font-size: 1.1rem;
            padding: 0.6rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            border-radius: 8px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Header and subtext
st.markdown('<div class="report-title">📄 AI-Powered Research Report Generator</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Generate structured and well-written reports using LLaMA 3.3 70B and LangGraph.</p>', unsafe_allow_html=True)

# Input and Button
report_topic = st.text_input("🎯 Enter your report topic:")
generate_btn = st.button("🚀 Generate Report")

# Generation logic
if generate_btn and report_topic.strip():
    with st.spinner("Generating report, please wait..."):
        state = orchestrator_worker.invoke({
            "title": report_topic.strip(),
            "completed_sections": [],
            "final_report": ""
        })

        st.success("✅ Report Generated!")
        st.markdown("---")
        st.markdown(state["final_report"])
