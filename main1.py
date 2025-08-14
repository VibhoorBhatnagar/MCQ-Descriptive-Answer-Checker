from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
import base64
import json
import re
import requests
from io import BytesIO
import uvicorn

# -----------------------------
# Qwen Server Config
# -----------------------------
QWEN_SERVER_IP = "192.168.1.32"   # Change to your Qwen server IP
QWEN_MODEL_NAME = "qwen2.5vl"     # Change if different
QWEN_PORT = 11434                 # Change if different
BASE_URL = f"http://{QWEN_SERVER_IP}:{QWEN_PORT}/v1/chat/completions"

app = FastAPI(title="OMR & Descriptive Checker API (Qwen OCR)")

# -----------------------------
# Qwen OCR Function
# -----------------------------
def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_text_with_qwen(image: Image.Image):
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
    response = requests.post(BASE_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

# -----------------------------
# Parse Answers
# -----------------------------
def parse_answers(text):
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
# API Endpoints
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/grade")
async def grade_api(answer_key: UploadFile = File(...), answer_sheet: UploadFile = File(...)):
    try:
        # Load answer key
        answer_key_data = json.loads(await answer_key.read())

        # Process answer sheet
        extracted_text = ""
        if answer_sheet.content_type == "application/pdf":
            pages = convert_from_bytes(await answer_sheet.read(), dpi=300)
            page_texts = [extract_text_with_qwen(img) for img in pages]
            extracted_text = "\n".join(page_texts)
        else:
            image = Image.open(BytesIO(await answer_sheet.read()))
            extracted_text = extract_text_with_qwen(image)

        # Parse & grade
        parsed_answers = parse_answers(extracted_text)
        grading_result = grade_answers(parsed_answers, answer_key_data)

        return JSONResponse(content={
            "raw_text": extracted_text,
            "parsed_answers": parsed_answers,
            "grading": grading_result
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)