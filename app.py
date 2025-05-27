import os
import pytesseract
from PIL import Image
import streamlit as st
import requests

# Groq API setup
API_URL = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
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
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return f"[Error] Groq API failed: {response.status_code}"

st.title(" Image NER Chatbot")

if "GROQ_API_KEY" not in os.environ:
    st.error("Please set your Groq API key as `GROQ_API_KEY` in Streamlit Secrets.")
    st.stop()

uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    all_outputs = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=250)
        text = extract_text_from_image(uploaded_file)
        result = query_groq(text)
        all_outputs.append(result)

    st.markdown("### Extracted Results (TSV Format)")
    result_text = "\n".join(all_outputs)
    st.code(result_text, language="tsv")
    st.download_button("Download TSV", result_text, file_name="entities.tsv")
else:
    st.info("Upload one or more image files to begin.")
