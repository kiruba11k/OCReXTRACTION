import os
import pytesseract
from PIL import Image
import streamlit as st
import requests

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set your Groq API key in `.streamlit/secrets.toml` as GROQ_API_KEY.")
    st.stop()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def query_groq(text):
    prompt = f"""
Extract Name, Designation, and Company from this text and return only tab-separated values. One entry per line. No explanation.

Text:
{text.strip()}
    """

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[Error] Groq API call failed: {e}"

st.title(" Image NER Chatbot")

uploaded_files = st.file_uploader(
    "Upload image files (JPEG/PNG)", accept_multiple_files=True, type=["png", "jpg", "jpeg"]
)

if uploaded_files:
    all_outputs = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=250)
        with st.spinner("Extracting text and querying Groq..."):
            extracted_text = extract_text_from_image(uploaded_file)
            groq_response = query_groq(extracted_text)
            all_outputs.append(groq_response)

    st.markdown("###  Extracted Results ")
    final_result = "\n".join(all_outputs)
    st.code(final_result, language="tsv")
    st.download_button("⬇ Download TSV", final_result, file_name="entities.tsv")
else:
    st.info("Please upload one or more image files to begin.")
