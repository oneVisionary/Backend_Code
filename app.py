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

app = FastAPI(title="Document Classification API")

# ===============================
# CONFIG
# ===============================

BASE_OUTPUT = "output"
MAX_WORKERS = len(OPENROUTER_KEYS)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
key_cycle = cycle(OPENROUTER_KEYS)

CATEGORIES = [
   
    "Bank Statement",

    "Medical Report",
    "Other"
]

for cat in CATEGORIES:
    os.makedirs(os.path.join(BASE_OUTPUT, cat), exist_ok=True)

PROMPT = """
You are a document classification assistant.

Classify the document into one of the following:
- Receipt
- Invoice
- Bank Statement
- ID Document
- Medical Report
- Other

Return only the document type.

OCR TEXT:
"""


# ===============================
# BLOCKING FUNCTIONS
# ===============================

def run_ocr(image_path: str) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


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
            "messages": [{"role": "user", "content": PROMPT + text}],
            "temperature": 0
        },
        timeout=90
    )

    result = response.json()
    category = result["choices"][0]["message"]["content"].strip()

    if category not in CATEGORIES:
        category = "Other"

    return category


# ===============================
# PIPELINE FOR ONE IMAGE
# ===============================

def process_single_image(file: UploadFile):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # OCR
    ocr_text = run_ocr(temp_path)

    # Classification
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

    # save json
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
