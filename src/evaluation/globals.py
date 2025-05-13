from core.globals import DB_DIR


EVAL_USER_ID: int = 1337_8413211
DB_EVAL_DIR = DB_DIR / "eval"
DB_EVAL_DIR.mkdir(parents=True, exist_ok=True)

SEMAPHORE_CHAT_LIMIT = 10
SEMAPHORE_EVAL_LIMIT = 3
SEMAPHORE_EMBEDDINGS_LIMIT = 5
EMBEDDING_BATCH_SIZE = 128

CHAT_ENDPOINT = "https://llmtools.valerii.cc/v1" # todo: move to .env
CHAT_ENDPOINT_API_KEY = "lpak-F-2CjtBBJ1wTm18APW1Apg"
CHAT_MODEL = "gemini-2.0-flash"
CHAT_EVAL_MODEL = "gpt-4o"
CHAT_ANALYSE_MODEL = "gemini-2.0-flash"
