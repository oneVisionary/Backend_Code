from fastapi import FastAPI, UploadFile, File
from PIL import Image
import pytesseract
import requests
import shutil
import os
import json
import time
import asyncio
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor
from keys import OPENROUTER_KEYS

# ===============================
# TESSERACT PATH (Docker)
# ===============================

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI(title="Document Classification API")

# ===============================
# CONFIG
# ===============================

BASE_OUTPUT = "output"

CATEGORIES = [
    "Bank Account",
    "Medical",
    "Other"
]

for cat in CATEGORIES:
    os.makedirs(os.path.join(BASE_OUTPUT, cat), exist_ok=True)

MAX_WORKERS = len(OPENROUTER_KEYS)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
key_cycle = cycle(OPENROUTER_KEYS)

PROMPT = """
You are a document classification assistant.

Classify the document into exactly one of these categories:

- Bank Account
- Medical
- Other

Return only one label exactly as written above.

OCR TEXT:
"""

# ===============================
# OCR
# ===============================

def run_ocr(image_path: str) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


# ===============================
# CATEGORY NORMALIZATION
# ===============================

import re
import unicodedata


def clean_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_category(text: str) -> str:
    text = clean_text(text)

    bank_keywords = [
        "bank", "statement", "account",
        "transaction", "transferred", "transfer",
        "payment", "paid", "utility", "bill",
        "amount", "fee", "balance",
        "jazzcash", "easypaisa", "paytm",
        "upi", "credit", "debit"
    ]

    medical_keywords = [
        "medical", "hospital", "laboratory",
        "lab", "report", "cbc", "blood",
        "hematology", "pathology",
        "patient", "diagnosis", "test result"
    ]

    bank_score = sum(1 for k in bank_keywords if k in text)
    medical_score = sum(1 for k in medical_keywords if k in text)

    if medical_score >= 2:
        return "Medical"

    if bank_score >= 2:
        return "Bank Account"

    return "Other"

# ===============================
# OPENROUTER CLASSIFICATION
# ===============================

def classify_text(text: str) -> str:
    api_key = next(key_cycle)

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [
                {"role": "user", "content": PROMPT + text}
            ],
            "temperature": 0
        },
        timeout=90
    )

    result = response.json()
    raw_label = result["choices"][0]["message"]["content"].strip()

    return normalize_category(raw_label + " " + text)


# ===============================
# IMAGE PIPELINE
# ===============================

def process_single_image(file: UploadFile):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ocr_text = run_ocr(temp_path)
    category = classify_text(ocr_text)

    final_path = os.path.join(BASE_OUTPUT, category, file.filename)
    shutil.move(temp_path, final_path)

    return {
        "image_name": file.filename,
        "ocr_text": ocr_text,
        "category": category
    }


# ===============================
# FASTAPI ENDPOINT
# ===============================

@app.post("/classify-documents/")
async def classify_documents(files: list[UploadFile] = File(...)):
    start_time = time.perf_counter()

    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(executor, process_single_image, file)
        for file in files
    ]

    results = await asyncio.gather(*tasks)

    json_path = os.path.join(BASE_OUTPUT, "results.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(results)

    with open(json_path, "w") as f:
        json.dump(existing, f, indent=4)

    total_time = round(time.perf_counter() - start_time, 2)

    return {
        "processed_images": len(files),
        "total_time_seconds": total_time,
        "results": results
    }
