from datetime import datetime
from enum import Enum
from typing import Iterable, Dict, Optional, Union, Any
from pathlib import Path

import ujson as json

from pydantic import BaseModel

from core.globals import TELEMETRY_DIR


class TelemetryScope(Enum):
    W_PROCESSOR = "w_processor"


class TeleItemStatus(Enum):
    INFO = "info"
    SUCCESS = "success"
    FAILURE = "failure"

type TeleItem = Union[TeleWProcessor]


class TeleWProcessor(BaseModel):
    version: str = "v0"
    event: str

    status: TeleItemStatus
    error_message: Optional[str] = None
    error_recoverable: Optional[bool] = None

    user_id: Optional[int] = None
    file_name: Optional[str] = None
    file_name_orig: Optional[str] = None
    vector_store: Optional[str] = None
    file_id: Optional[str] = None

    attributes: Optional[Dict[str, Any]] = None

    duration_seconds: Optional[float] = None
    timestamp: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "version": self.version,
            "event": self.event,
            "status": self.status.value,
            "error_message": self.error_message,
            "error_recoverable": self.error_recoverable,
            "user_id": self.user_id,
            "file_name": self.file_name,
            "file_name_orig": self.file_name_orig,
            "vector_store": self.vector_store,
            "file_id": self.file_id,
            "attributes": self.attributes,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

    def write(self, writer: 'TeleWriter'):
        writer.write(self)


class TeleWriter:
    def __init__(
            self,
            scope: TelemetryScope
    ):
        self.scope_dir = TELEMETRY_DIR / scope.value
        self.scope_dir.mkdir(parents=True, exist_ok=True)

    def current_file_path(self) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        filename = f"{today}.jsonl"
        return self.scope_dir / filename

    def write(self, line: TeleItem):
        with self.current_file_path().open('a') as f:
            f.write(json.dumps(line.to_dict()) + '\n')
