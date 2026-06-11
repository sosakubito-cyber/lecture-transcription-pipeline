from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any, Iterable


DEFAULT_AUDIO = (
    Path.home()
    / "Desktop"
    / "人口経済学"
    / "授業資料"
    / "5:21_第6回_人口経済学"
    / "5:21_第6回_人口経済学.mp3"
)
DEFAULT_OUTPUT_DIR = Path("high_accuracy_runs")
SAMPLE_RATE = 16_000
OPENAI_MAX_UPLOAD_BYTES = 24_000_000

DEFAULT_TERMS = [
    "人口経済学",
    "人口転換",
    "出生率",
    "死亡率",
    "少子化",
    "高齢化",
    "移民",
    "移民余剰",
    "労働供給",
    "労働需要",
    "賃金",
    "所得税",
    "消費税",
    "助成金",
    "人的資本",
    "外部性",
    "大学進学",
    "匿名",
    "加点",
    "提出率",
    "中間アンケート",
    "ホームワーク",
]

DEFAULT_PROMPT = (
    "これは日本語の大学講義「人口経済学」の書き起こしです。"
    "話者は主に教員1名です。内容を要約せず、聞こえた発話を忠実に文字起こししてください。"
    "専門語・固有語の候補: "
    + "、".join(DEFAULT_TERMS)
    + "。図のラベルとして A, B, C, D, E, F, AFDB, BDC などが出る場合があります。"
    "既知の誤認識を避けてください: 匿名を特命としない、加点を確認としない、"
    "提出率を停止成立としない。"
)

DEFAULT_GEMINI_KEY_PDF = Path.home() / "Desktop" / "gemini_api.pdf"
DEFAULT_DEEPGRAM_KEY_FILE = Path.home() / "Desktop" / "Deepgram-apy-key.txt"
POSTPROCESS_SYSTEM_PROMPT = """\
あなたは日本語の大学講義書き起こしを校正する専門家です。
音声認識の結果だけを根拠に、誤字、句読点、表記ゆれ、明らかな専門語の誤変換を修正してください。
内容の要約、削除、補足、言い換えは禁止です。
聞き取れていない箇所を想像で埋めないでください。
出力は校正後の本文だけにしてください。
"""


def fmt_ts(seconds: float, sep: str = ".") -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def parse_time(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    if ":" not in text:
        return float(text)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Invalid timestamp: {value}")


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_")


def decode_audio_16k(path: Path) -> np.ndarray:
    import numpy as np
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm16.tobytes())


def wav_size_bytes(seconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    return 44 + int(seconds * sample_rate) * 2


def find_quiet_boundary(
    audio: np.ndarray,
    target_sample: int,
    *,
    search_radius_seconds: float = 15.0,
    frame_ms: float = 50.0,
) -> int:
    import numpy as np

    radius = int(search_radius_seconds * SAMPLE_RATE)
    frame = max(1, int(frame_ms * SAMPLE_RATE / 1000))
    lo = max(frame, target_sample - radius)
    hi = min(len(audio) - frame, target_sample + radius)
    if hi <= lo:
        return max(0, min(len(audio), target_sample))

    best_i = target_sample
    best_energy = math.inf
    for i in range(lo, hi, frame):
        chunk = audio[i : i + frame]
        energy = float(np.mean(np.abs(chunk)))
        if energy < best_energy:
            best_energy = energy
            best_i = i
    return best_i


def chunk_specs(
    audio: np.ndarray,
    *,
    chunk_seconds: float,
    quiet_boundaries: bool,
    max_bytes: int = OPENAI_MAX_UPLOAD_BYTES,
) -> list[dict[str, Any]]:
    max_seconds = (max_bytes - 44) / (SAMPLE_RATE * 2)
    chunk_seconds = min(chunk_seconds, max_seconds)
    specs: list[dict[str, Any]] = []
    start = 0
    idx = 1
    min_chunk = int(60 * SAMPLE_RATE)
    while start < len(audio):
        target = min(len(audio), start + int(chunk_seconds * SAMPLE_RATE))
        if target < len(audio) and quiet_boundaries:
            boundary = find_quiet_boundary(audio, target)
            if boundary <= start + min_chunk:
                boundary = target
        else:
            boundary = target
        end = min(len(audio), boundary)
        specs.append(
            {
                "name": f"chunk_{idx:03d}",
                "start": start / SAMPLE_RATE,
                "end": end / SAMPLE_RATE,
                "start_sample": start,
                "end_sample": end,
            }
        )
        start = end
        idx += 1
    return specs


def write_sample_set(args: argparse.Namespace) -> None:
    audio = decode_audio_16k(args.audio)
    duration = len(audio) / SAMPLE_RATE
    sample_seconds = args.sample_seconds

    if args.sample_starts:
        starts = [parse_time(x) for x in args.sample_starts.split(",")]
    else:
        starts = [
            0.0,
            max(0.0, (duration - sample_seconds) * 0.45),
            max(0.0, duration - sample_seconds),
        ]
    names = ["beginning", "middle", "ending"]
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "audio": str(args.audio),
        "duration": duration,
        "sample_seconds": sample_seconds,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "samples": [],
    }

    for i, start in enumerate(starts):
        start = max(0.0, min(start, max(0.0, duration - 1)))
        end = min(duration, start + sample_seconds)
        name = names[i] if i < len(names) else f"sample_{i + 1}"
        wav_path = outdir / f"{i + 1:02d}_{name}_{fmt_ts(start).replace(':', '-')}.wav"
        write_wav(wav_path, audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)])
        manifest["samples"].append(
            {"name": name, "start": start, "end": end, "path": str(wav_path)}
        )
        print(f"Wrote {wav_path} ({fmt_ts(start)}-{fmt_ts(end)})")

    with (outdir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {outdir / 'manifest.json'}")


def load_sample_manifest(samples_dir: Path) -> list[dict[str, Any]]:
    manifest_path = samples_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["samples"]


def audio_inputs(args: argparse.Namespace, work_dir: Path) -> list[dict[str, Any]]:
    if args.input_mode == "samples":
        return [
            {
                "name": sample["name"],
                "path": Path(sample["path"]),
                "start": float(sample["start"]),
                "end": float(sample["end"]),
            }
            for sample in load_sample_manifest(args.samples_dir)
        ]

    audio = decode_audio_16k(args.audio)
    specs = chunk_specs(
        audio,
        chunk_seconds=args.chunk_seconds,
        quiet_boundaries=not args.no_quiet_boundaries,
    )
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    for spec in specs:
        wav_path = chunks_dir / f"{spec['name']}_{fmt_ts(spec['start']).replace(':', '-')}.wav"
        write_wav(wav_path, audio[spec["start_sample"] : spec["end_sample"]])
        inputs.append(
            {
                "name": spec["name"],
                "path": wav_path,
                "start": spec["start"],
                "end": spec["end"],
            }
        )
    return inputs


def multipart_post(
    url: str,
    *,
    api_key: str,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> bytes:
    boundary = f"----codex-{uuid.uuid4().hex}"
    body = bytearray()

    def add_text(name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for key, value in fields.items():
        add_text(key, value)

    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
    )
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: HTTP {exc.code}: {detail}") from exc


def json_post(url: str, *, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: HTTP {exc.code}: {detail}") from exc


def load_gemini_api_key(pdf_path: Path | None) -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key.strip()
    if pdf_path and pdf_path.exists():
        key_prefix = b"AI" + b"za"
        matches = re.findall(key_prefix + rb"[0-9A-Za-z_-]{20,}", pdf_path.read_bytes())
        if matches:
            return matches[0].decode("ascii")
    raise SystemExit(
        "Gemini API key not found. Set GEMINI_API_KEY or pass --api-key-pdf."
    )


def load_deepgram_api_key(key_path: Path | None) -> str:
    key = os.getenv("DEEPGRAM_API_KEY")
    if key:
        return key.strip()
    env_key_path = os.getenv("DEEPGRAM_API_KEY_FILE")
    if env_key_path:
        key_path = Path(env_key_path).expanduser()
    if key_path and key_path.exists():
        for line in key_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise SystemExit(
        "Deepgram API key not found. Set DEEPGRAM_API_KEY or pass --api-key-file."
    )


def deepgram_transcribe_file(
    path: Path,
    *,
    api_key: str,
    model: str,
    language: str,
    smart_format: bool,
    punctuate: bool,
    keyterms: list[str],
) -> dict[str, Any]:
    params = [
        ("model", model),
        ("language", language),
        ("smart_format", str(smart_format).lower()),
        ("punctuate", str(punctuate).lower()),
    ]
    params.extend(("keyterm", term) for term in keyterms if term.strip())
    url = "https://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        data=path.read_bytes(),
        method="POST",
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Deepgram API request failed: HTTP {exc.code}: {detail}") from exc


def deepgram_extract_alternative(data: dict[str, Any]) -> dict[str, Any]:
    channels = data.get("results", {}).get("channels") or []
    if not channels:
        return {}
    alternatives = channels[0].get("alternatives") or []
    return alternatives[0] if alternatives else {}


def segments_from_deepgram_words(
    words: list[dict[str, Any]],
    *,
    chunk_start: float,
    chunk_end: float,
    fallback_text: str,
    joiner: str = " ",
    max_segment_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    if not words:
        text = fallback_text.strip()
        return [{"start": chunk_start, "end": chunk_end, "text": text}] if text else []

    segments = []
    current_words: list[str] = []
    current_start: float | None = None
    current_end = chunk_start

    for word in words:
        token = str(word.get("punctuated_word") or word.get("word") or "").strip()
        if not token:
            continue
        start = chunk_start + float(word.get("start") or 0.0)
        end = chunk_start + float(word.get("end") or word.get("start") or 0.0)
        if current_start is None:
            current_start = start
        current_words.append(token)
        current_end = max(end, current_end)
        duration = current_end - current_start
        sentence_end = token.endswith((".", "?", "!", "。", "？", "！"))
        if sentence_end or duration >= max_segment_seconds or len("".join(current_words)) >= 140:
            segments.append(
                {
                    "start": current_start,
                    "end": max(current_end, current_start + 0.1),
                    "text": joiner.join(current_words).strip(),
                }
            )
            current_words = []
            current_start = None

    if current_words and current_start is not None:
        segments.append(
            {
                "start": current_start,
                "end": max(current_end, current_start + 0.1),
                "text": joiner.join(current_words).strip(),
            }
        )
    return segments


def gemini_generate_audio_text(
    path: Path,
    *,
    api_key: str,
    model: str,
    prompt: str,
    previous_text: str,
) -> str:
    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    full_prompt = prompt
    if previous_text:
        full_prompt += "\n直前の書き起こし文脈:\n" + previous_text[-1200:]
    full_prompt += (
        "\n\n上の条件に従って、この音声を日本語で忠実に全文書き起こししてください。"
        "要約、箇条書き、解説はしないでください。"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": full_prompt},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent?key="
        + api_key
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: HTTP {exc.code}: {detail}") from exc

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts).strip()
    return strip_transcript_wrapping(text)


def strip_transcript_wrapping(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    labels = [
        "文字起こし:",
        "書き起こし:",
        "以下、文字起こしです。",
        "以下が文字起こしです。",
    ]
    for label in labels:
        if text.startswith(label):
            text = text[len(label) :].strip()
    return text


def command_gemini(args: argparse.Namespace) -> None:
    api_key = load_gemini_api_key(args.api_key_pdf)
    print("Gemini API key loaded (redacted).")

    work_dir = args.output_prefix.parent / f"{args.output_prefix.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = audio_inputs(args, work_dir)
    chunks = []
    previous_text = ""
    started = time.time()

    for item in inputs:
        print(f"Gemini {args.model}: {item['name']} {fmt_ts(item['start'])}-{fmt_ts(item['end'])}")
        text = gemini_generate_audio_text(
            item["path"],
            api_key=api_key,
            model=args.model,
            prompt=args.prompt,
            previous_text=previous_text if args.use_previous_context else "",
        )
        previous_text = text
        chunks.append(
            {
                **item,
                "path": str(item["path"]),
                "text": text,
                "timestamp_source": "chunk",
            }
        )

    result = {
        "backend": "gemini",
        "model": args.model,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "prompt": args.prompt,
        "chunks": chunks,
    }
    write_outputs(args.output_prefix, result)


def command_deepgram(args: argparse.Namespace) -> None:
    api_key = load_deepgram_api_key(args.api_key_file)
    print("Deepgram API key loaded (redacted).")
    keyterms = [] if args.no_default_keyterms else list(DEFAULT_TERMS)
    keyterms.extend(args.keyterm or [])

    work_dir = args.output_prefix.parent / f"{args.output_prefix.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = audio_inputs(args, work_dir)
    chunks = []
    started = time.time()

    for item in inputs:
        print(f"Deepgram {args.model}: {item['name']} {fmt_ts(item['start'])}-{fmt_ts(item['end'])}")
        data = deepgram_transcribe_file(
            item["path"],
            api_key=api_key,
            model=args.model,
            language=args.language,
            smart_format=args.smart_format,
            punctuate=args.punctuate,
            keyterms=keyterms,
        )
        alternative = deepgram_extract_alternative(data)
        text = str(alternative.get("transcript") or "").strip()
        words = alternative.get("words") or []
        segments = segments_from_deepgram_words(
            words,
            chunk_start=float(item["start"]),
            chunk_end=float(item["end"]),
            fallback_text=text,
            joiner="" if args.language.lower().startswith("ja") else " ",
        )
        chunks.append(
            {
                **item,
                "path": str(item["path"]),
                "text": text,
                "segments": segments,
                "raw_response": data,
                "timestamp_source": "word",
            }
        )

    result = {
        "backend": "deepgram",
        "model": args.model,
        "language": args.language,
        "keyterms": keyterms,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "chunks": chunks,
    }
    write_outputs(args.output_prefix, result)


def openai_transcribe_file(
    path: Path,
    *,
    api_key: str,
    model: str,
    prompt: str,
    previous_text: str,
) -> str:
    full_prompt = prompt
    if previous_text:
        full_prompt += "\n直前の書き起こし文脈:\n" + previous_text[-1200:]
    response = multipart_post(
        "https://api.openai.com/v1/audio/transcriptions",
        api_key=api_key,
        fields={
            "model": model,
            "language": "ja",
            "response_format": "json",
            "prompt": full_prompt,
        },
        file_field="file",
        file_path=path,
    )
    data = json.loads(response.decode("utf-8"))
    return data.get("text", "").strip()


def openai_postprocess(
    text: str,
    *,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    user_prompt = (
        "専門語・固有語候補:\n"
        + prompt
        + "\n\n以下の書き起こしを校正してください:\n"
        + text
    )
    data = json_post(
        "https://api.openai.com/v1/chat/completions",
        api_key=api_key,
        payload={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": POSTPROCESS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
    )
    return data["choices"][0]["message"]["content"].strip()


def command_openai(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    work_dir = args.output_prefix.parent / f"{args.output_prefix.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = audio_inputs(args, work_dir)
    chunks = []
    previous_text = ""
    started = time.time()

    for item in inputs:
        print(f"OpenAI transcribe {item['name']} {fmt_ts(item['start'])}-{fmt_ts(item['end'])}")
        raw_text = openai_transcribe_file(
            item["path"],
            api_key=api_key,
            model=args.model,
            prompt=args.prompt,
            previous_text=previous_text if args.use_previous_context else "",
        )
        final_text = raw_text
        if args.postprocess:
            print(f"Post-process {item['name']} with {args.text_model}")
            final_text = openai_postprocess(
                raw_text,
                api_key=api_key,
                model=args.text_model,
                prompt=args.prompt,
            )
        previous_text = final_text
        chunks.append(
            {
                **item,
                "path": str(item["path"]),
                "raw_text": raw_text,
                "text": final_text,
                "timestamp_source": "chunk",
            }
        )

    result = {
        "backend": "openai",
        "model": args.model,
        "postprocess_model": args.text_model if args.postprocess else None,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "prompt": args.prompt,
        "chunks": chunks,
    }
    write_outputs(args.output_prefix, result)


def command_local(args: argparse.Namespace) -> None:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    work_dir = args.output_prefix.parent / f"{args.output_prefix.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = audio_inputs(args, work_dir)
    started = time.time()

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    transcriber: Any = model
    if args.batch_size > 1:
        transcriber = BatchedInferencePipeline(model=model)

    chunks = []
    for item in inputs:
        print(
            f"Local {args.model} on {args.device}: "
            f"{item['name']} {fmt_ts(item['start'])}-{fmt_ts(item['end'])}"
        )
        kwargs = dict(
            language="ja",
            beam_size=args.beam_size,
            vad_filter=True,
            condition_on_previous_text=args.condition_on_previous_text,
            hotwords=" ".join(DEFAULT_TERMS),
        )
        if args.batch_size > 1:
            kwargs["batch_size"] = args.batch_size
        segments_iter, info = transcriber.transcribe(str(item["path"]), **kwargs)
        segments = []
        text_parts = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            start = item["start"] + float(seg.start)
            end = item["start"] + float(seg.end)
            segments.append({"start": start, "end": end, "text": text})
            text_parts.append(text)
        chunks.append(
            {
                **item,
                "path": str(item["path"]),
                "text": "\n".join(text_parts),
                "segments": segments,
                "language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
                "timestamp_source": "segment",
            }
        )

    result = {
        "backend": "faster-whisper",
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "chunks": chunks,
    }
    write_outputs(args.output_prefix, result)


def command_google(args: argparse.Namespace) -> None:
    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import storage
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
    except ImportError as exc:
        raise SystemExit(
            "Google dependencies are missing. Install: "
            "python3 -m pip install google-cloud-speech google-cloud-storage"
        ) from exc

    project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT")
    bucket_name = args.bucket or os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")
    if not project or not bucket_name:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_STORAGE_BUCKET.")

    work_dir = args.output_prefix.parent / f"{args.output_prefix.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = audio_inputs(args, work_dir)
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    if args.location == "global":
        speech_client = SpeechClient()
    else:
        speech_client = SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{args.location}-speech.googleapis.com")
        )

    chunks = []
    started = time.time()
    for item in inputs:
        object_name = f"{args.gcs_prefix.strip('/')}/{Path(item['path']).name}"
        blob = bucket.blob(object_name)
        print(f"Upload {item['path']} to gs://{bucket_name}/{object_name}")
        blob.upload_from_filename(str(item["path"]))
        uri = f"gs://{bucket_name}/{object_name}"

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["ja-JP"],
            model=args.model,
        )
        request = cloud_speech.BatchRecognizeRequest(
            recognizer=f"projects/{project}/locations/{args.location}/recognizers/_",
            config=config,
            files=[cloud_speech.BatchRecognizeFileMetadata(uri=uri)],
            recognition_output_config=cloud_speech.RecognitionOutputConfig(
                inline_response_config=cloud_speech.InlineOutputConfig(),
            ),
        )
        if args.dynamic_batch:
            request.processing_strategy = (
                cloud_speech.BatchRecognizeRequest.ProcessingStrategy.DYNAMIC_BATCHING
            )
        print(f"Google {args.model}: {item['name']} {fmt_ts(item['start'])}-{fmt_ts(item['end'])}")
        operation = speech_client.batch_recognize(request=request)
        response = operation.result(timeout=args.timeout)
        transcript = response.results[uri].transcript
        text_parts = []
        segments = []
        cursor = item["start"]
        saw_offsets = False
        for result in transcript.results:
            if not result.alternatives:
                continue
            text = result.alternatives[0].transcript.strip()
            if not text:
                continue
            end = cursor
            if getattr(result, "result_end_offset", None):
                saw_offsets = True
                offset = result.result_end_offset
                if hasattr(offset, "total_seconds"):
                    end = item["start"] + offset.total_seconds()
                elif hasattr(offset, "seconds") and hasattr(offset, "nanos"):
                    end = item["start"] + float(offset.seconds) + float(offset.nanos) / 1e9
                elif hasattr(offset, "ToTimedelta"):
                    end = item["start"] + offset.ToTimedelta().total_seconds()
            segments.append({"start": cursor, "end": max(end, cursor + 0.1), "text": text})
            cursor = max(end, cursor)
            text_parts.append(text)
        if text_parts and not saw_offsets:
            segments = [
                {
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "text": "\n".join(text_parts),
                }
            ]
        chunks.append(
            {
                **item,
                "path": str(item["path"]),
                "gcs_uri": uri,
                "text": "\n".join(text_parts),
                "segments": segments,
                "timestamp_source": "google_result",
            }
        )

    result = {
        "backend": "google-cloud-speech",
        "model": args.model,
        "location": args.location,
        "dynamic_batch": args.dynamic_batch,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "chunks": chunks,
    }
    write_outputs(args.output_prefix, result)


def segments_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    for chunk in result.get("chunks", []):
        if chunk.get("segments"):
            segments.extend(chunk["segments"])
            continue
        text = chunk.get("text", "").strip()
        if text:
            segments.append(
                {
                    "start": float(chunk["start"]),
                    "end": float(chunk["end"]),
                    "text": text,
                }
            )
    return segments


def write_outputs(prefix: Path, result: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = prefix.with_suffix(".txt")
    srt_path = prefix.with_suffix(".srt")
    json_path = prefix.with_suffix(".json")
    segments = segments_from_result(result)

    with txt_path.open("w", encoding="utf-8") as txt:
        txt.write("高精度講義書き起こし\n")
        txt.write(f"backend: {result.get('backend')}\n")
        txt.write(f"model: {result.get('model')}\n")
        txt.write(f"created_at: {result.get('created_at')}\n")
        if result.get("elapsed_seconds") is not None:
            txt.write(f"elapsed_seconds: {result['elapsed_seconds']:.1f}\n")
        txt.write("\n")
        for seg in segments:
            txt.write(f"[{fmt_ts(seg['start'])} - {fmt_ts(seg['end'])}] {seg['text']}\n")

    with srt_path.open("w", encoding="utf-8") as srt:
        for idx, seg in enumerate(segments, 1):
            srt.write(
                f"{idx}\n"
                f"{fmt_ts(seg['start'], ',')} --> {fmt_ts(seg['end'], ',')}\n"
                f"{seg['text']}\n\n"
            )

    result["segments"] = segments
    with json_path.open("w", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)

    print(f"Wrote {txt_path}")
    print(f"Wrote {srt_path}")
    print(f"Wrote {json_path}")


def command_report(args: argparse.Namespace) -> None:
    rows = []
    for path in args.jsons:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        text = "\n".join(seg["text"] for seg in data.get("segments", []))
        term_hits = {term: text.count(term) for term in DEFAULT_TERMS}
        suspicious = {
            "AFD_loop": len(re.findall(r"(AFDと、\s*){3,}", text)),
            "特命": text.count("特命"),
            "停止成立": text.count("停止成立"),
            "前迷惑": text.count("前迷惑"),
            "empty_or_short": int(len(text) < 100),
        }
        rows.append((path, data, term_hits, suspicious))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as md:
        md.write("# Transcription Comparison Report\n\n")
        md.write("| file | backend | model | elapsed_sec | suspicious_total |\n")
        md.write("| --- | --- | --- | ---: | ---: |\n")
        for path, data, _terms, suspicious in rows:
            md.write(
                f"| {path} | {data.get('backend')} | {data.get('model')} | "
                f"{float(data.get('elapsed_seconds') or 0):.1f} | "
                f"{sum(suspicious.values())} |\n"
            )
        md.write("\n## Suspicious Patterns\n\n")
        for path, _data, _terms, suspicious in rows:
            md.write(f"### {path}\n\n")
            for key, count in suspicious.items():
                md.write(f"- {key}: {count}\n")
            md.write("\n")
        md.write("## Term Hits\n\n")
        for path, _data, terms, _suspicious in rows:
            md.write(f"### {path}\n\n")
            for term, count in terms.items():
                if count:
                    md.write(f"- {term}: {count}\n")
            md.write("\n")
    print(f"Wrote {args.output}")


def add_common_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--input-mode", choices=["samples", "full"], default="samples")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "samples")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
    parser.add_argument("--no-quiet-boundaries", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High accuracy lecture transcription pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    samples = sub.add_parser("make-samples", help="Create three representative WAV samples.")
    samples.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    samples.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR / "samples")
    samples.add_argument("--sample-seconds", type=float, default=300.0)
    samples.add_argument(
        "--sample-starts",
        help="Comma-separated starts in seconds or HH:MM:SS. Default: beginning, middle, ending.",
    )
    samples.set_defaults(func=write_sample_set)

    openai = sub.add_parser("openai", help="Transcribe with OpenAI gpt-4o-transcribe.")
    add_common_input_args(openai)
    openai.add_argument("--model", default="gpt-4o-transcribe")
    openai.add_argument("--text-model", default=os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1"))
    openai.add_argument("--prompt", default=DEFAULT_PROMPT)
    openai.add_argument("--postprocess", action="store_true")
    openai.add_argument("--use-previous-context", action="store_true")
    openai.set_defaults(func=command_openai)

    local = sub.add_parser("local", help="Transcribe with faster-whisper, suitable for RTX/CUDA.")
    add_common_input_args(local)
    local.add_argument("--model", default="large-v3")
    local.add_argument("--device", default="cuda")
    local.add_argument("--compute-type", default="float16")
    local.add_argument("--beam-size", type=int, default=5)
    local.add_argument("--batch-size", type=int, default=1)
    local.add_argument("--condition-on-previous-text", action="store_true")
    local.set_defaults(func=command_local)

    google = sub.add_parser("google", help="Transcribe with Google Cloud Speech-to-Text chirp_3.")
    add_common_input_args(google)
    google.add_argument("--model", default="chirp_3")
    google.add_argument("--project")
    google.add_argument("--bucket")
    google.add_argument("--location", default="global")
    google.add_argument("--gcs-prefix", default="high_accuracy_transcribe")
    google.add_argument("--timeout", type=int, default=3600)
    google.add_argument("--dynamic-batch", action="store_true")
    google.set_defaults(func=command_google)

    gemini = sub.add_parser("gemini", help="Transcribe with Gemini API audio input.")
    add_common_input_args(gemini)
    gemini.add_argument("--model", default="gemini-2.5-flash")
    gemini.add_argument("--api-key-pdf", type=Path, default=DEFAULT_GEMINI_KEY_PDF)
    gemini.add_argument("--prompt", default=DEFAULT_PROMPT)
    gemini.add_argument("--use-previous-context", action="store_true")
    gemini.set_defaults(func=command_gemini, chunk_seconds=300.0)

    deepgram = sub.add_parser("deepgram", help="Transcribe with Deepgram Nova-3.")
    add_common_input_args(deepgram)
    deepgram.add_argument("--model", default="nova-3")
    deepgram.add_argument("--language", default="ja")
    deepgram.add_argument("--api-key-file", type=Path, default=DEFAULT_DEEPGRAM_KEY_FILE)
    deepgram.add_argument("--smart-format", action=argparse.BooleanOptionalAction, default=True)
    deepgram.add_argument("--punctuate", action=argparse.BooleanOptionalAction, default=True)
    deepgram.add_argument("--keyterm", action="append", default=[])
    deepgram.add_argument("--no-default-keyterms", action="store_true")
    deepgram.set_defaults(func=command_deepgram, chunk_seconds=540.0)

    report = sub.add_parser("report", help="Create a comparison report from output JSON files.")
    report.add_argument("--jsons", type=Path, nargs="+", required=True)
    report.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "comparison_report.md")
    report.set_defaults(func=command_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
