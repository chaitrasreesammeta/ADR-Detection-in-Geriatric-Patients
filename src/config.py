from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents= True, exist_ok=True)
CLEANED_MARKDOWN_FILE = DATA_DIR / "adverse _drug_ reaction_clean.md"

VECTOR_DIR = PROJECT_ROOT/ "chroma"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    load_dotenv()
