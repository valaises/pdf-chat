import asyncio

from dataclasses import dataclass
from typing import Optional, Tuple, List

from openai import OpenAI
from pydantic import BaseModel

from core.repositories.repo_files import FilesRepository
from telemetry.tele_writer import TeleWriter


class ParagraphData(BaseModel):
    page_n: int
    section_number: Optional[str] = None
    paragraph_text: str
    paragraph_box: Tuple[float, float, float, float]
    paragraph_id: Optional[str] = None


class ParagraphVectorData(BaseModel):
    paragraph_id: str
    idx: int
    text: str
    embedding: Optional[List[float]] = None


@dataclass
class WorkerContext:
    client: OpenAI
    loop: asyncio.AbstractEventLoop
    tele: TeleWriter
    files_repository: FilesRepository
