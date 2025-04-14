from datetime import datetime
from pathlib import Path

import ujson as json

from core.globals import TELEMETRY_DIR
from telemetry.models import TelemetryScope, TeleItem


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
