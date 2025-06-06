import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
import streamlit as st
from io import BytesIO
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
import json

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY to `.streamlit/secrets.toml`")
    st.stop()

class MyState(TypedDict):
    image: BytesIO
    text: str
    output: str

# llm = ChatGroq(
#     model="llama3-70b-8192",
#     groq_api_key=GROQ_API_KEY,
#     temperature=0.2,)
llm = ChatGroq(
    model="mistral-saba-24b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)
# llm = ChatOpenAI(
#     model="gpt-4o-mini",  # or "gpt-3.5-turbo"
#     openai_api_key=OPENAI_API_KEY,
#     temperature=0.2,
# )
def ocr_step(state: MyState) -> MyState:
    uploaded_file = state.get("image")
    if uploaded_file is None:
        raise ValueError("Missing 'image' in state.")

    file_data = uploaded_file.read()
    file_bytes = np.asarray(bytearray(file_data), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    pil_image = Image.fromarray(cv2.bitwise_not(processed))

    # OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    state["text"] = text
    return state

def ner_step(state: MyState) -> MyState:
    text = state["text"]
    st.markdown("### Extracted Content")
    st.code(text, language="text")  # Use 'text' instead of 'tsv' if it's unstructured

    prompt = f"""
Extract all entities from this text.

Your job is to find triplets of people with:
- Name
- Designation
- Company

Respond ONLY with a valid JSON list of objects in this format:
[
  {{"Name": "John Doe", "Designation": "CEO", "Company": "OpenAI"}},
  ...
]

DO NOT add any explanation or text before or after the JSON.

Text:
{text.strip()}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        extracted = json.loads(response.content.strip())
        df = pd.DataFrame(extracted)
        state["output"] = df.to_json(orient="records")  # Save as JSON string for later
    except Exception as e:
        st.warning("Failed to parse model output as JSON.")
        state["output"] = "[]" 

    return state


workflow = StateGraph(state_schema=MyState)
workflow.add_node("OCR", ocr_step)
workflow.add_node("NER", ner_step)
workflow.set_entry_point("OCR")
workflow.add_edge("OCR", "NER")
workflow.set_finish_point("NER")
graph = workflow.compile()

st.title("OCR Extraction")

uploaded_files = st.file_uploader(" Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=250)
        with st.spinner("Extracting..."):
            file_copy = BytesIO(uploaded_file.read())
            file_copy.seek(0)
            result = graph.invoke({"image": file_copy})
            results.append(result["output"])

    all_rows = []
    for idx, json_str in enumerate(results):
        try:
            rows = json.loads(json_str)
            for row in rows:
                all_rows.append({
                    "Image #": f"Image {idx+1}",
                    "Name": row.get("Name", ""),
                    "Designation": row.get("Designation", ""),
                    "Company": row.get("Company", "")
                })
        except json.JSONDecodeError:
            st.error(f"Failed to decode JSON for Image {idx+1}")

    if all_rows:
        df = pd.DataFrame(all_rows)

        def highlight_by_image(row):
            img_id = int(row["Image #"].split()[-1])
            color = "#f9f9f9" if img_id % 2 == 0 else "#e8f5e9"
            return ["background-color: {}".format(color)] * len(row)

        styled_df = df.style.apply(highlight_by_image, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "entities.csv", "text/csv")
    else:
        st.warning("No valid entities found in any image.")
