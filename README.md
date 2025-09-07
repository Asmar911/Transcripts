# Transcripts

Simple pipeline to transcribe audio files locally using the Whisper Python library and save results per file.

## Local Usage

- Prereqs: Python 3.10+, ffmpeg installed, and Python packages for Whisper + Torch.

Commands:

```
# Install ffmpeg
# macOS (brew): brew install ffmpeg
# Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg
# Windows: choco install ffmpeg or winget install Gyan.FFmpeg

# Install Python deps (CPU-only Torch + Whisper)
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt

# Run transcription (choose a model: tiny|base|small|medium|large-v3)
python transcribe.py --input . --out-dir transcripts --model medium --formats txt,json,srt,vtt
```

Outputs are written to `transcripts/<audio_basename>/` as `.txt`, `.json`, `.srt`, and `.vtt` when supported by the model.

## GitHub Actions

This repo includes a workflow at `.github/workflows/transcribe.yml` that:

- Installs dependencies
- Installs ffmpeg and CPU Torch, then runs `transcribe.py` to process audio files in the repo
- Uploads the `transcripts/` directory as a build artifact
- Commits transcript files back to the repo (if any changes)

Manual run with inputs:

- Go to the Actions tab → "Transcribe Audio" → Run workflow. You can set:
  - `input_path`: File or directory to process (default `.`)
  - `model`: Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`; default `medium`)
  - `language`: Dropdown of supported languages shown as `code - Name` (e.g., `en - English`, `es - Spanish`) plus `auto - Auto-detect`. The workflow parses the code before passing it to the script.

On push:

- Pushing audio files (e.g., `.mp3`, `.wav`) also triggers the workflow and processes the whole repo (defaults: `input_path='.'`, `model='medium'`, auto language).

You can customize model and formats by editing the `Transcribe audio files` step in the workflow.

Notes:
- For best quality, `large-v3` yields higher accuracy but is large and slow on CPU; `medium` is a good balance for GitHub Actions. If your repo needs faster runs, try `small`.
