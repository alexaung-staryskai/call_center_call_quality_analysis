from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class FileWriter:
    """Handles file system writes for transcript JSON and output CSV files."""

    @staticmethod
    def write_transcript_json(output_path: Path, payload: dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    @staticmethod
    def write_results_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
