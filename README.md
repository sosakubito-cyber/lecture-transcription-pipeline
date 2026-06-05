# 音声書き写しプロジェクト

大学講義音声を複数の音声認識手法で書き起こし、精度・コスト・運用性を比較するためのローカルプロジェクトです。

## Main Script

```bash
python3 high_accuracy_transcribe.py --help
```

## Setup

```bash
python3 -m pip install -r requirements_high_accuracy.txt
```

必要に応じて `.env.example` を参考に、ローカル専用の `.env` や環境変数を設定してください。実APIキーやサービスアカウントJSONはコミットしないでください。

## Providers

- OpenAI `gpt-4o-transcribe`
- Google Cloud Speech-to-Text V2 `chirp_3` + Dynamic Batch
- Gemini API audio input
- Local Whisper / MLX / faster-whisper

## Generated Files

音声ファイル、変換済みWAV、書き起こし結果、APIレスポンスJSON、作業チャンクはGit管理対象外です。

主な除外対象:

- `audio/`
- `high_accuracy_runs/`
- `outputs/`
- `*.mp3`
- `*.m4a`
- `*.wav`
- generated `*.txt`
- generated `*.srt`
- generated `*.json`

ローカル成果物の場所は `PROJECT_INDEX.md` を参照してください。

## Project Notes

- `PROJECT.md`: GitHub化前の棚卸し
- `PROJECT_INDEX.md`: ローカル成果物インデックス
- `README_high_accuracy_transcription.md`: 詳細な実行メモ
