from pathlib import Path
from importlib.metadata import version


VERSION = version("chat-with-pdf-poc")


BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
FILES_DIR = BASE_DIR / "files"
LOGS_DIR = BASE_DIR / "logs"
TELEMETRY_DIR = BASE_DIR / "telemetry"
