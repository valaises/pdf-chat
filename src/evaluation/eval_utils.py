import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.globals import ASSETS_DIR
from core.repositories.repo_files import FileItem


def eval_file_path(file: FileItem) -> Path:
    return ASSETS_DIR / "eval" / file.file_name_orig


def parse_model_output_json(output: str, parse_into: type[BaseModel]) -> Any:
    """
    Parse the evaluation output in JSON format and return a Pydantic instance.

    Args:
        output: The JSON string output from the evaluation
        parse_into: The Pydantic model class to validate and parse the data into

    Returns:
        A Pydantic model instance of the specified type containing the parsed data
    """
    # Look for JSON content between triple backticks
    json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
    else:
        # If not found between backticks, try to find a JSON object directly
        json_match = re.search(r'(\{.*\})', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            raise ValueError("Could not find valid JSON in the output")

    try:
        json_data = json.loads(json_str)
        return parse_into.model_validate(json_data)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing evaluation output: {e}")
