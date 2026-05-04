# Starting this project
## Audio Processing using Python 3.11
***
source .venv/bin/activate
python main.py
***

## Visual Mockup
***
source .venv/bin/activate
python dashboard_app_2.py
***

## App Login
***
User Name: admin
Password: admin
***

# Audio Analysis Application

A local Python application that reads MP3 files from an `input` folder, transcribes them on CPU with PyTorch and torchaudio, saves transcripts as JSON, analyzes each transcript without using large language models, and writes the final results to a CSV file.

## What the project does

The application performs the following steps:

1. Reads `.mp3` files from the `input` folder
2. Transcribes each audio file
3. Saves the transcript into the `transcript` folder as JSON
4. Reads transcript JSON files back from the `transcript` folder
5. Analyzes each transcript for:
   - sentiment
   - sentiment confidence
   - extractive summary
   - customer satisfaction flag
6. Saves the final results into `output/analysis_results.csv`

## Solution design

This project does **not** use large language models.

It uses:

- **PyTorch** on **CPU**
- **torchaudio** `WAV2VEC2_ASR_BASE_960H` for speech-to-text
- **NLTK VADER** for sentiment analysis
- simple **rule-based** logic for customer satisfaction
- simple **extractive summarization** based on word frequency

## Project structure

```text
Audio Analysis/
├── input/
├── transcript/
├── output/
├── model/
│   └── wav2vec2_fairseq_base_ls960_asr_ls960.pth
├── nltk_data/
│   └── sentiment/
│       └── vader_lexicon.zip
├── main.py
├── model_loader.py
├── transcriber.py
├── file_reader.py
├── file_writer.py
├── analytics.py
├── requirements.txt
└── README.md
```

## Python version

Recommended on macOS:

- **Python 3.11**

Python 3.13 may fail in some environments because of stricter SSL certificate validation when downloading files. Since this project is prepared to run from local model files, Python 3.11 is the safest choice.

## Create the virtual environment

Open Terminal and run:

```bash
cd "/Users/alex.aung/Documents/Projects/Audio Analysis"
python3.11 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Freeze exact versions after installation

If you want to freeze the exact installed versions:

```bash
pip freeze > requirements.txt
```

## Files that must be downloaded manually

This project is prepared to run fully from local folders, so there are two non-package assets you should place in the project.

### 1) ASR model file

Create the `model` folder if it does not exist:

```bash
mkdir -p model
```

Download the torchaudio ASR weights:

```bash
curl -L -o model/wav2vec2_fairseq_base_ls960_asr_ls960.pth \
  "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
```

### 2) VADER lexicon for sentiment analysis

Create the `nltk_data` folder:

```bash
mkdir -p nltk_data
```

Download the VADER lexicon into the local project folder:

```bash
python -m nltk.downloader -d ./nltk_data vader_lexicon
```

Expected result:

```text
nltk_data/
└── sentiment/
    └── vader_lexicon.zip
```

## Full setup steps

Run these commands from the project directory:

```bash
cd "/Users/alex.aung/Documents/Projects/Audio Analysis"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p model
curl -L -o model/wav2vec2_fairseq_base_ls960_asr_ls960.pth \
  "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
mkdir -p nltk_data
python -m nltk.downloader -d ./nltk_data vader_lexicon
```

## Input folder preparation

Put your MP3 files into the `input` folder.

Example:

```text
input/
├── call_001.mp3
├── call_002.mp3
└── call_003.mp3
```

## Run the application

```bash
python main.py
```

## Output generated

### Transcript JSON files

For each MP3 file, a JSON file is created in `transcript/`.

Example:

```text
transcript/call_001.json
```

Typical JSON structure:

```json
{
  "file_name": "call_001.mp3",
  "source_path": "/full/path/input/call_001.mp3",
  "created_at_utc": "2026-04-08T12:00:00+00:00",
  "transcript": "customer said the issue is resolved and thanks for the quick help",
  "chunks": [
    {
      "chunk_index": 0,
      "start_seconds": 0.0,
      "end_seconds": 20.0,
      "text": "customer said the issue is resolved"
    }
  ]
}
```

### Final CSV file

The application writes:

```text
output/analysis_results.csv
```

Expected columns:

- `file_name`
- `sentiment`
- `sentiment_confidence`
- `summary`
- `customer_satisfaction`

## File responsibilities

### `main.py`
Controls the application flow:
- transcribe MP3 files
- save transcript JSON
- analyze transcript JSON
- save CSV results

### `model_loader.py`
Loads local resources from disk:
- ASR model from `model/`
- VADER lexicon from `nltk_data/`

### `transcriber.py`
Handles audio transcription:
- loads MP3 audio
- resamples if required
- splits audio into chunks
- runs ASR per chunk
- merges chunk text into one transcript

### `file_reader.py`
Handles reading files and listing folders:
- finds MP3 files in `input/`
- finds JSON files in `transcript/`
- reads JSON files

### `file_writer.py`
Handles writing files:
- writes transcript JSON
- writes CSV results

### `analytics.py`
Handles transcript analytics:
- sentiment
- confidence
- extractive summary
- customer satisfaction flag

## Important notes

### 1) CPU only
This project is set to use:

- `device="cpu"`

No GPU is required.

### 2) English ASR model
The selected wav2vec2 model is primarily for English speech recognition.

### 3) MP3 support on macOS
If MP3 loading fails, install FFmpeg:

```bash
brew install ffmpeg
```

### 4) No runtime downloads
This version is intended to avoid runtime downloads.

That means:
- the ASR `.pth` file should already exist in `model/`
- the VADER lexicon should already exist in `nltk_data/`

If either is missing, the code will raise a clear file error.

## Troubleshooting

### Error: local ASR model file not found
Cause:
- the model file is missing from `model/`

Fix:

```bash
mkdir -p model
curl -L -o model/wav2vec2_fairseq_base_ls960_asr_ls960.pth \
  "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
```

### Error: local NLTK resource not found
Cause:
- `vader_lexicon` has not been downloaded into `nltk_data/`

Fix:

```bash
mkdir -p nltk_data
python -m nltk.downloader -d ./nltk_data vader_lexicon
```

### Error reading MP3 files
Cause:
- missing audio backend or codec support

Fix:

```bash
brew install ffmpeg
```

### SSL certificate verification error
Cause:
- your system or network blocks the automatic HTTPS download path

Fix:
- use the manual local model download method described above
- prefer Python 3.11 on macOS

## Recommended execution flow

1. Create and activate virtual environment
2. Install Python dependencies
3. Download ASR model into `model/`
4. Download VADER lexicon into `nltk_data/`
5. Put MP3 files into `input/`
6. Run `python main.py`
7. Check `transcript/` and `output/analysis_results.csv`

## Optional improvement ideas

Possible next improvements include:

- speaker segmentation
- better customer satisfaction rules
- confidence scoring for transcription quality
- batch logging and execution reports
- support for WAV files in addition to MP3
- configurable chunk size from a settings file
