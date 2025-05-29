import os
from PIL import Image
import pytesseract
import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY to `.streamlit/secrets.toml`")
    st.stop()


llm = ChatGroq(
    model="mixtral-8x7b-32768",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)


def ocr_step(state):
    image = state["image"]
    text = pytesseract.image_to_string(Image.open(image))
    state["text"] = text
    return state

def ner_step(state):
    text = state["text"]
    prompt = f"""
Extract Name, Designation, and Company from this text and return only tab-separated values. One entry per line. No explanation.

Text:
{text.strip()}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["output"] = response.content.strip()
    return state

workflow = StateGraph()
workflow.add_node("OCR", ocr_step)
workflow.add_node("NER", ner_step)
workflow.set_entry_point("OCR")
workflow.add_edge("OCR", "NER")
workflow.set_finish_point("NER")
graph = workflow.compile()

st.title("OCR bot")

uploaded_files = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=250)
        with st.spinner("Extracting..."):
            result = graph.invoke({"image": uploaded_file})
            results.append(result["output"])

    st.markdown("### Extracted TSV Output")
    result_text = "\n".join(results)
    st.code(result_text, language="tsv")
    st.download_button("⬇ Download TSV", result_text, file_name="entities.tsv")
else:
    st.info("Upload at least one image to begin.")
