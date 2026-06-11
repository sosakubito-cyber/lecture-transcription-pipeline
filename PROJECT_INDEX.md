# 音声書き写しプロジェクト インデックス

このプロジェクトは、講義音声の書き起こし結果を手法別に整理しています。

## まず見る場所

- `high_accuracy_transcribe.py`: 現在使っている統合スクリプト
- `requirements_high_accuracy.txt`: 必要パッケージ
- `README_high_accuracy_transcription.md`: 実行方法メモ

## 主要な最終成果物

### Google Chirp 3 Dynamic Batch

- `high_accuracy_runs/google/google_chirp3_dynamic_6_4_8_full.txt`
- `high_accuracy_runs/google/google_chirp3_dynamic_6_4_8_full.srt`
- `high_accuracy_runs/google/google_chirp3_dynamic_6_4_8_full.json`

第8回、約98.6分の音声を `chirp_3` + Dynamic Batch で処理した出力です。

### Deepgram Nova-3

今後の比較候補として `deepgram` サブコマンドを追加済みです。

出力先の推奨:

- `high_accuracy_runs/deepgram/deepgram_nova3_samples.txt`
- `high_accuracy_runs/deepgram/deepgram_nova3_samples.srt`
- `high_accuracy_runs/deepgram/deepgram_nova3_samples.json`

### OpenAI gpt-4o-transcribe

- `high_accuracy_runs/openai/openai_gpt4o_full.txt`
- `high_accuracy_runs/openai/openai_gpt4o_full.srt`
- `high_accuracy_runs/openai/openai_gpt4o_full.json`

第6回のOpenAI API出力です。

### Gemini

- `high_accuracy_runs/gemini/gemini_35_flash_full_repaired_clean.txt`
- `high_accuracy_runs/gemini/gemini_35_flash_full_repaired_clean.srt`
- `high_accuracy_runs/gemini/gemini_35_flash_full_repaired_clean.json`

第6回のGemini出力です。空チャンク再試行と最小ノイズ修正済みです。

### MLX / ローカルWhisper

- `outputs/mlx/第6回_人口経済学_mlx全文書き起こし_clean.txt`
- `outputs/mlx/第6回_人口経済学_mlx全文書き起こし_clean.srt`
- `outputs/mlx/第6回_人口経済学_mlx全文書き起こし_clean.json`

Macローカルで処理した比較用出力です。

## 比較レポート

- `high_accuracy_runs/reports/openai_gemini_google_mlx_report.md`
- `high_accuracy_runs/reports/openai_gemini_mlx_report_clean.md`
- `high_accuracy_runs/reports/openai_vs_mlx_report.md`

## ディレクトリ構成

- `high_accuracy_runs/google/`: Google Speech-to-Text / Chirp 3 の出力と作業ファイル
- `high_accuracy_runs/deepgram/`: Deepgram Nova-3 の出力と作業ファイル
- `high_accuracy_runs/openai/`: OpenAI API の出力と作業ファイル
- `high_accuracy_runs/gemini/`: Gemini API の出力と作業ファイル
- `high_accuracy_runs/reports/`: 比較レポート
- `high_accuracy_runs/samples/`: 代表15分サンプル。スクリプトの既定パスなので維持
- `high_accuracy_runs/smoke/`: 動作確認用の小さいテスト
- `outputs/legacy_initial/`: 初期の試行出力
- `outputs/mlx/`: MLX / Whisper ローカル出力
- `outputs/tests/`: 90秒などの短い検証出力
- `audio/derived/`: 派生音声ファイル、変換済みWAVなど
- `scripts/legacy/`: 以前使っていた個別スクリプト

## 今後の推奨運用

新しい音声を処理するときは、出力先を手法別ディレクトリに指定してください。

例:

```bash
python3 high_accuracy_transcribe.py google \
  --audio "/path/to/audio.mp3" \
  --input-mode full \
  --output-prefix high_accuracy_runs/google/google_chirp3_dynamic_next_full \
  --model chirp_3 \
  --location asia-northeast1 \
  --project "<your-google-cloud-project-id>" \
  --bucket "<your-cloud-storage-bucket>" \
  --dynamic-batch \
  --timeout 7200
```
