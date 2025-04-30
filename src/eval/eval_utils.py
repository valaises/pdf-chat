from pathlib import Path

from core.globals import ASSETS_DIR
from core.repositories.repo_files import FileItem


def eval_file_path(file: FileItem) -> Path:
    return ASSETS_DIR / "eval" / file.file_name_orig
