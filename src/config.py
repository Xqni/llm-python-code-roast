from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
_MODEL_RAW = OLLAMA_MODEL.split(":")[0]
MODEL_DISPLAY_NAME = "-".join(
    part.capitalize() for part in _MODEL_RAW.split("-")
)
