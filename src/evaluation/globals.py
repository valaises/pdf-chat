import os

from core.globals import DB_DIR, BASE_DIR


EVAL_USER_ID: int = 1337_8413211
DB_EVAL_DIR = DB_DIR / "eval"
DB_EVAL_DIR.mkdir(parents=True, exist_ok=True)

SEMAPHORE_CHAT_LIMIT = 10
SEMAPHORE_EVAL_LIMIT = 3
SEMAPHORE_EMBEDDINGS_LIMIT = 5
EMBEDDING_BATCH_SIZE = 128

CHAT_ENDPOINT = os.environ.get("EVAL_CHAT_ENDPOINT")
assert CHAT_ENDPOINT is not None, "EVAL_CHAT_ENDPOINT environment variable must be set"

CHAT_ENDPOINT_API_KEY = os.environ.get("EVAL_CHAT_ENDPOINT_API_KEY")
assert CHAT_ENDPOINT_API_KEY is not None, "EVAL_CHAT_ENDPOINT_API_KEY environment variable must be set"

CHAT_MODEL = os.environ.get("EVAL_CHAT_MODEL", "gemini-2.0-flash")
CHAT_EVAL_MODEL = os.environ.get("EVAL_CHAT_EVAL_MODEL", "gpt-4o")
CHAT_ANALYSE_MODEL = os.environ.get("EVAL_CHAT_ANALYSE_MODEL", "gemini-2.0-flash")
