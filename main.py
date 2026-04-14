from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from file_reader import FileReader
from file_writer import FileWriter
from model_loader import ModelLoader
from slm_analyzer import SLMTranscriptAnalyzer
from transcriber import AudioTranscriber


BASE_DIR = Path(__file__).resolve().parent
QWEN_GGUF_PATH = BASE_DIR / "model" / "qwen2.5_7b_instruct" / "qwen2.5-7b-instruct-q3_k_m.gguf"


def transcribe_audio_files(
    reader: FileReader,
    writer: FileWriter,
    transcriber: AudioTranscriber,
) -> None:
    audio_files = reader.list_audio_files()

    if not audio_files:
        print(f"No audio files found in: {reader.input_dir}")
        return

    for audio_path in audio_files:
        print(f"Transcribing: {audio_path.name}")
        transcript_payload = transcriber.transcribe_file(str(audio_path))

        output_payload = {
            "file_name": audio_path.name,
            "source_path": str(audio_path),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **transcript_payload,
        }

        transcript_output_path = reader.transcript_dir / f"{audio_path.stem}.json"
        writer.write_transcript_json(transcript_output_path, output_payload)
        print(f"Saved transcript JSON: {transcript_output_path.name}")


def analyze_transcripts(
    reader: FileReader,
    writer: FileWriter,
    analyzer: SLMTranscriptAnalyzer,
) -> None:
    transcript_files = reader.list_transcript_files()

    if not transcript_files:
        print(f"No transcript JSON files found in: {reader.transcript_dir}")
        return

    results: list[dict[str, object]] = []

    for transcript_path in transcript_files:
        payload = reader.read_json(transcript_path)
        transcript_text = payload.get("transcript", "")

        if not transcript_text:
            print(f"Skipping empty transcript: {transcript_path.name}")
            continue

        analysis = analyzer.analyze_text(
            transcript=transcript_text,
            file_name=payload.get("file_name", transcript_path.stem),
        )

        results.append(
            {
                "file_name": payload.get("file_name", transcript_path.stem),
                "product_focus": analysis["product_focus"],
                "type of call": analysis["type of call"],
                "sentiment_score": analysis["sentiment_score"],
                "did customer get the answer": analysis["did customer get the answer"],
                "next step for customer": analysis["next step for customer"],
                "call_summary": analysis["call_summary"],
            }
        )

    if not results:
        print("No transcript results to write.")
        return

    csv_output_path = reader.output_dir / "analysis_results.csv"
    writer.write_results_csv(csv_output_path, results)
    print(f"Saved CSV output: {csv_output_path}")


def main() -> None:
    reader = FileReader(BASE_DIR)
    writer = FileWriter()
    model_loader = ModelLoader(device="cpu", model_dir=BASE_DIR / "model")

    transcriber = AudioTranscriber(
        model_loader=model_loader,
        chunk_seconds=20,
    )

    analyzer = SLMTranscriptAnalyzer(
        model_loader=model_loader,
        model_path=str(QWEN_GGUF_PATH),
        n_ctx=8192,
        n_threads=6,
        n_gpu_layers=0,
        max_tokens=320,
        temperature=0.0,
        debug=True,
    )

    if not QWEN_GGUF_PATH.exists():
        raise FileNotFoundError(
            f"Qwen GGUF model not found at: {QWEN_GGUF_PATH}\n"
            "Please place the model there or update QWEN_GGUF_PATH in main.py."
        )

    transcribe_audio_files(reader, writer, transcriber)
    analyze_transcripts(reader, writer, analyzer)
    print("Done.")


if __name__ == "__main__":
    main()
