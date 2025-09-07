#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_audio_files(root: str, exts: List[str]) -> List[str]:
    if os.path.isfile(root):
        return [root] if root.lower().endswith(tuple(exts)) else []
    found = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(tuple(exts)):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_text(path: str, content: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def transcribe_whisper(file_path: str, model_name: str, device: str = "auto", language: str | None = None):
    import torch
    import whisper

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(model_name, device=device)
    # fp16 only on GPU
    fp16 = device == "cuda"
    result = model.transcribe(file_path, verbose=False, fp16=fp16, language=language)
    return result


def write_timecoded(result, file_path: str, out_folder: str, kind: str):
    # kind: "srt" or "vtt"
    from whisper.utils import get_writer
    writer = get_writer(kind, out_folder)
    writer(result, file_path)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files locally with Whisper and save outputs per file.")
    parser.add_argument('--input', '-i', default='.', help='Input file or directory to scan for audio files')
    parser.add_argument('--out-dir', '-o', default='transcripts', help='Root output directory for transcripts')
    parser.add_argument('--model', '-m', default='medium', help='Whisper model size/name (tiny, base, small, medium, large-v3)')
    parser.add_argument('--device', default='auto', help='Device to use: auto, cpu, or cuda')
    parser.add_argument('--language', default=None, help='Force language code (e.g., en). Defaults to auto-detect')
    parser.add_argument('--formats', '-f', default='txt,json,srt,vtt', help='Comma-separated output formats to save (txt,json,srt,vtt)')
    parser.add_argument('--fail-on-error', action='store_true', help='Exit non-zero if any file fails to transcribe')

    args = parser.parse_args()

    formats = [fmt.strip().lower() for fmt in args.formats.split(',') if fmt.strip()]
    supported = {'txt', 'json', 'srt', 'vtt'}
    for fmt in formats:
        if fmt not in supported:
            eprint(f"Unsupported format: {fmt}. Supported: {sorted(supported)}")
            sys.exit(2)

    audio_exts = ['.mp3', '.wav', '.m4a', '.mp4', '.mpeg', '.mpga', '.webm', '.ogg', '.flac']
    files = find_audio_files(args.input, audio_exts)
    if not files:
        eprint('No audio files found to transcribe.')
        return 0

    ensure_dir(args.out_dir)

    failures = []
    for file_path in files:
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_folder = os.path.join(args.out_dir, base)
        ensure_dir(out_folder)

        print(f"Transcribing: {file_path}")
        try:
            result = transcribe_whisper(file_path, args.model, device=args.device, language=args.language)
        except Exception as exc:
            eprint(f"Failed transcription for {file_path}: {exc}")
            failures.append(file_path)
            if args.fail_on_error:
                return 1
            else:
                continue

        # Write outputs
        if 'txt' in formats:
            save_text(os.path.join(out_folder, f"{base}.txt"), result.get('text', '').strip())

        if 'json' in formats:
            import json as _json
            save_text(os.path.join(out_folder, f"{base}.json"), _json.dumps(result, ensure_ascii=False, indent=2))

        if 'srt' in formats:
            try:
                write_timecoded(result, file_path, out_folder, 'srt')
            except Exception as exc:
                eprint(f"Failed SRT writing for {file_path}: {exc}")

        if 'vtt' in formats:
            try:
                write_timecoded(result, file_path, out_folder, 'vtt')
            except Exception as exc:
                eprint(f"Failed VTT writing for {file_path}: {exc}")

        print(f"Saved transcripts to: {out_folder}")

    if failures and args.fail_on_error:
        return 1
    return 0


if __name__ == '__main__':
    
    sys.exit(main())
