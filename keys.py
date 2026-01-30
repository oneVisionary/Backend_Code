import os
from dotenv import load_dotenv

load_dotenv()

keys = os.getenv("OPENROUTER_KEYS")

if not keys:
    raise ValueError("OPENROUTER_KEYS not found in environment")

OPENROUTER_KEYS = [k.strip() for k in keys.split(",")]

if len(OPENROUTER_KEYS) < 1:
    raise ValueError("At least one OpenRouter API key required")
