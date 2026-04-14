from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileReader:
    """Handles file system reads for input audio and transcript JSON files."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.input_dir = self.base_dir / "input"
        self.transcript_dir = self.base_dir / "transcript"
        self.output_dir = self.base_dir / "output"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_audio_files(self) -> list[Path]:
        """Return all supported audio files from input folder."""
        patterns = ("*.mp3", "*.wav")
        files: list[Path] = []

        for pattern in patterns:
            files.extend(self.input_dir.glob(pattern))

        return sorted(files)

    def list_transcript_files(self) -> list[Path]:
        """Return all transcript JSON files from transcript folder."""
        return sorted(self.transcript_dir.glob("*.json"))

    @staticmethod
    def read_json(json_path: Path) -> dict[str, Any]:
        """Read a JSON file and return its content."""
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)