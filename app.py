import os
import pytesseract
from PIL import Image
import streamlit as st
import requests

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

# Image OCR
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# Call Hugging Face API for entity extraction
def query_huggingface(text):
    prompt = f"""
Extract Name, Designation, and Company from this text and return tab-separated values only.

Text:
{text.strip()}

Output:
"""
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        generated_text = response.json()[0]['generated_text']
        return generated_text.strip().split("Output:")[-1].strip()
    else:
        return f"[Error] Hugging Face API failed: {response.status_code}"

# Streamlit UI
st.title("🖼️ Image NER Chatbot (Name, Designation, Company) using Hugging Face")

if "HF_TOKEN" not in os.environ:
    st.error("Please set your Hugging Face token as `HF_TOKEN` in Streamlit Secrets.")
    st.stop()

uploaded_files = st.file_uploader("📤 Upload signature images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    all_outputs = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, width=250)
        text = extract_text_from_image(uploaded_file)
        result = query_huggingface(text)
        all_outputs.append(result)

    st.markdown("### ✅ Extracted Results (TSV Format)")
    result_text = "\n".join(all_outputs)
    st.code(result_text, language="tsv")
    st.download_button("📥 Download TSV", result_text, file_name="entities.tsv")
else:
    st.info("Upload one or more image files to begin.")
