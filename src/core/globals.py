from pathlib import Path
from importlib.metadata import version


EMBEDDING_MODEL = "text-embedding-3-small"
VERSION = version("chat-with-pdf-poc")

# openai_fs or local_fs
PROCESSING_STRATEGY = "local_fs"
# only for local_fs: redis or milvus
SAVE_STRATEGY = "redis"

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
FILES_DIR = BASE_DIR / "files"
LOGS_DIR = BASE_DIR / "logs"
TELEMETRY_DIR = BASE_DIR / "telemetry"

# === EVALUATION ===
EVALUATIONS_DIR = BASE_DIR / "evaluations"
EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
# ===

DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
