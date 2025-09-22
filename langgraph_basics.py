import os
from typing import List

from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('API_KEY')
llm = GoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=1000, temperature=0)

class State(BaseModel):
    text: str | None = None
    classification: str | None = None
    entities: List[str] | None = None
    summary: str | None = None

def classification_node(state: State):
    """ Classify the text into one of the categories: News, Blog, Research, or Other """
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state.text))
    classification = llm.invoke([message]).strip()
    return {"classification": classification}


def entity_extraction_node(state: State):
    """ Extract all the entities (Person, Organization, Location) from the text """
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state.text))
    entities = llm.invoke([message]).strip().split(", ")
    return {"entities": entities}


def summarization_node(state: State):
    """ Summarize the text in one short sentence """
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state.text))
    summary = llm.invoke([message]).strip()
    return {"summary": summary}

workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

# Add edges to the graph
workflow.set_entry_point("classification_node") # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()

sample_text = """
OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
"""

print("Analyzing text...\n")
state_input = {"text": sample_text}
result = app.invoke(state_input)

print(result.__class__)
print("Classification:", result["classification"])
print("\nEntities:", result["entities"])
print("\nSummary:", result["summary"])