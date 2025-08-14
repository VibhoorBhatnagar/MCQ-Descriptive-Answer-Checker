import os
import re
import json
import base64
import requests
from io import BytesIO
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image

# -----------------------------
# Qwen Server Config
# -----------------------------
QWEN_SERVER_IP = "192.168.1.32"   # Change to your Qwen server IP
QWEN_MODEL_NAME = "qwen2.5vl"     # Change if different
QWEN_PORT = 11434                 # Change if different
BASE_URL = f"http://{QWEN_SERVER_IP}:{QWEN_PORT}/v1/chat/completions"

# -----------------------------
# Qwen OCR Function
# -----------------------------
def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_text_with_qwen(image):
    image_b64 = image_to_base64(image)
    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image exactly as it appears."},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"}
                ]
            }
        ],
        "max_tokens": 2048
    }

    try:
        response = requests.post(BASE_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error using Qwen OCR: {e}")
        return ""

# -----------------------------
# Parse Q&A from OCR text
# -----------------------------
def parse_answers(text):
    # Improved regex: matches 1), 1., 1-, Q1), Q 1:, etc.
    pattern = r"(?:Q?\s?)(\d+)[\)\.\-:]"
    matches = list(re.finditer(pattern, text))
    answers = {}
    for i, match in enumerate(matches):
        q_num = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        ans_text = text[start:end].strip()
        answers[q_num] = ans_text
    return answers

# -----------------------------
# Grading Logic
# -----------------------------
def grade_answers(student_answers, answer_key):
    result = {"per_question": {}, "total": 0, "totals": {"mcq": 0, "desc": 0}}
    for q, expected in answer_key.items():
        q_str = str(q)
        student = student_answers.get(q_str, "").strip()
        if isinstance(expected, str):  # MCQ
            score = 1 if student.upper().startswith(expected.upper()) else 0
            result["per_question"][q_str] = {
                "type": "mcq",
                "score": score,
                "max": 1,
                "student": student,
                "expected": expected
            }
            result["totals"]["mcq"] += score
        elif isinstance(expected, dict):  # Descriptive
            score = 0
            if "keywords" in expected:
                for kw in expected["keywords"]:
                    if kw.lower() in student.lower():
                        score += 1
                score = min(score, expected.get("max", 5))
            result["per_question"][q_str] = {
                "type": "descriptive",
                "score": score,
                "max": expected.get("max", 5),
                "student": student,
                "expected": expected
            }
            result["totals"]["desc"] += score
    result["total"] = result["totals"]["mcq"] + result["totals"]["desc"]
    return result

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="OMR & Descriptive Checker - Qwen OCR", layout="wide")
st.title(" MCQ + Descriptive Answer Checker (Qwen OCR)")

uploaded_key = st.sidebar.file_uploader("Upload Answer Key (JSON)", type=["json"])
uploaded = st.file_uploader("Upload Student Answer Sheet (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded and uploaded_key:
    # Load answer key
    answer_key = json.load(uploaded_key)

    # Convert PDF to image or open image
    if uploaded.type == "application/pdf":
        pages = convert_from_bytes(uploaded.read(), dpi=300)
        # Process all pages into one text block
        all_text = []
        for img in pages:
            all_text.append(extract_text_with_qwen(img))
        extracted_text = "\n".join(all_text)
        image = pages[0]  # Show first page
    else:
        image = Image.open(uploaded)
        extracted_text = extract_text_with_qwen(image)

    st.image(image, caption="Uploaded Answer Sheet", use_container_width=True)

    # Debug raw OCR text
    st.subheader("DEBUG: Raw OCR Output")
    st.text(repr(extracted_text))

    # Parse answers
    parsed_answers = parse_answers(extracted_text)
    st.subheader("Detected Answers")
    st.json(parsed_answers)

    # Grade
    grading_result = grade_answers(parsed_answers, answer_key)
    st.subheader("Grading Result")
    st.json(grading_result)