import asyncio

import uvloop

from core.args import parse_args
from core.globals import BASE_DIR
from core.logger import init_logger, info
from core.app import App
from core.repositories.repo_files import FilesRepository
from core.server import Server, Worker, setup_signal_handlers

from core.workers.w_extractor import spawn_worker as spawn_worker_doc_extractor
from core.workers.w_processor import spawn_worker as spawn_worker_doc_processor


def main():
    args = parse_args()
    init_logger(args.DEBUG)
    info("Logger initialized")

    db_dir = BASE_DIR / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    files_repository = FilesRepository(db_dir / "files.db")
    doc_e_stop, doc_e_thread = spawn_worker_doc_extractor(files_repository)
    doc_p_stop, doc_p_thread = spawn_worker_doc_processor(files_repository)

    doc_e_worker = Worker(
        name="doc_extractor_thread",
        thread=doc_e_thread,
        stop_event=doc_e_stop
    )
    doc_p_worker = Worker(
        name="doc_processor_thread",
        thread=doc_p_thread,
        stop_event=doc_p_stop
    )

    app = App(
        files_repository,
        docs_url=None,
        redoc_url=None
    )

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    server = Server(
        app=app,
        host=args.host,
        port=args.port,
        workers=[doc_e_worker, doc_p_worker]
    )

    setup_signal_handlers(server)

    server.run()
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
