from core.globals import DB_DIR


EVAL_USER_ID: int = 1337_000000000_8413211
PROCESSING_STRATEGY = "local_fs"
SAVE_STRATEGY = "milvus"
DB_EVAL_DIR = DB_DIR / "eval"
DB_EVAL_DIR.mkdir(parents=True, exist_ok=True)
SEMAPHORE_LIMIT = 10
EMBEDDING_BATCH_SIZE = 128

CHAT_ENDPOINT = "https://llmtools.valerii.cc/v1"
CHAT_ENDPOINT_API_KEY = "lpak-F-2CjtBBJ1wTm18APW1Apg"
CHAT_MODEL = "gpt-4o"
