from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ADR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ADR_ROOT.parent

DATA_DIR = ADR_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_MARKDOWN_FILE = DATA_DIR / "DRUGS.MD"

VECTOR_DIR = ADR_ROOT / "chroma"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

FHIR_BASE_URL = "http://localhost:8080/fhir"

EVALUATION_DATE = date(2026, 10, 24)


def load_config() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def get_today() -> date:
    if EVALUATION_DATE is None:
        return date.today()

    return EVALUATION_DATE