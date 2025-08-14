import os
import re
import json
import cv2
import numpy as np
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -----------------------------
# Hardcoded Paths
# -----------------------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"  # Change this to your Poppler bin path

# Set environment paths
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ["PATH"] += os.pathsep + POPLER_PATH

# -----------------------------
# Load TrOCR model once
# -----------------------------
@st.cache_resource
def load_trocr():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

def preprocess_image(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)


def run_trocr(image: Image.Image):
    processor, model = load_trocr()
    # Upscale image for better handwriting recognition
    upscale_factor = 2
    w, h = image.size
    image = image.resize((w * upscale_factor, h * upscale_factor), Image.LANCZOS)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# -----------------------------
# OCR Functions
# -----------------------------
def run_tesseract(image: Image.Image):
    return pytesseract.image_to_string(image, config="--psm 6")

def extract_text(image: Image.Image, engine: str):
    if engine == "Tesseract":
        return run_tesseract(image)
    elif engine == "TrOCR":
        return run_trocr(image)
    else:
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
st.set_page_config(page_title="OMR & Descriptive Checker", layout="wide")
st.title("Unified MCQ + Descriptive Answer Checker")

ocr_engine = st.sidebar.selectbox("Select OCR Engine", ["Tesseract", "TrOCR"])

uploaded_key = st.sidebar.file_uploader("Upload Answer Key (JSON)", type=["json"])
uploaded = st.file_uploader("Upload Student Answer Sheet (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded and uploaded_key:
    # Load answer key
    answer_key = json.load(uploaded_key)

    # Convert PDF to image or open image
    if uploaded.type == "application/pdf":
        pages = convert_from_bytes(uploaded.read(), dpi=300)
        images_text = []
        for img in pages:
            text = extract_text(img, ocr_engine)
            images_text.append(text)
        extracted_text = "\n".join(images_text)

    else:
        image = Image.open(uploaded)

    st.image(image, caption="Uploaded Answer Sheet", use_container_width=True)

    # Run OCR
    with st.spinner(f"Running {ocr_engine} OCR..."):
        extracted_text = extract_text(image, ocr_engine)

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