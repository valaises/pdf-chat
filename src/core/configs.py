import yaml

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from core.globals import CONFIGS_DIR


class EvalConfig(BaseModel):
    __file_name: str = "eval_config.yaml"
    __file_path: Path = CONFIGS_DIR / __file_name

    chat_endpoint: str = "http://llm_tools:7016/v1"
    chat_endpoint_api_key: Optional[str] = None

    chat_model: str = "gemini-2.0-flash"
    chat_eval_model: str = "gpt-4o"
    chat_analyse_model: str = "gemini-2.0-flash"

    semaphore_chat_limit: int = Field(default=10, ge=1, le=50)
    semaphore_eval_limit: int = Field(default=3, ge=1, le=50)
    semaphore_embeddings_limit: int = Field(default=5, ge=1, le=50)
    embedding_batch_size: int = Field(default=128, ge=16, le=512)

    def exists(self) -> bool:
        return self.__file_path.exists()

    def save_to_disk(self):
        config_dict = {
            field: getattr(self, field)
            for field in self.__dict__
            if not field.startswith("_")
        }

        self.__file_path.write_text(
            yaml.dump(config_dict, default_flow_style=False)
        )

    def read_from_disk(self) -> None:
        """Update this config instance with values from disk."""
        if not self.__file_path.exists():
            raise Exception(f"{self.__file_path=} does not exist")

        config_dict = yaml.safe_load(self.__file_path.read_text())
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        # Update this instance with values from disk
        for key, value in config_dict.items():
            setattr(self, key, value)
