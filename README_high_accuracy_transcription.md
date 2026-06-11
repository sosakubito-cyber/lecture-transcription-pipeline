# 高精度講義書き起こしパイプライン

このプロジェクトでは、90分講義音声を以下の3系統で比較できます。

- OpenAI `gpt-4o-transcribe` + 用語プロンプト + 任意のLLM後処理
- Deepgram `nova-3`
- RTX 5090などのCUDA GPU上の `faster-whisper`
- Google Cloud Speech-to-Text V2 `chirp_3`

出力はすべて `txt` / `srt` / `json` です。OpenAI `gpt-4o-transcribe` は詳細タイムスタンプを返さないため、SRTはチャンク単位の概算タイムスタンプになります。精密なSRTが必要な場合は、RTXローカルまたはGoogle STTの出力を使ってください。

## 1. 代表区間を作る

```bash
cd ~/Documents/音声書き写しプロジェクト
python3 high_accuracy_transcribe.py make-samples
```

既定では、冒頭5分、中盤5分、終盤5分を `high_accuracy_runs/samples/` にWAVで作成します。

## 2. OpenAI APIで比較する

```bash
export OPENAI_API_KEY="<your-openai-api-key>"
python3 high_accuracy_transcribe.py openai \
  --input-mode samples \
  --samples-dir high_accuracy_runs/samples \
  --output-prefix high_accuracy_runs/openai/openai_gpt4o_samples \
  --postprocess \
  --use-previous-context
```

全文処理:

```bash
python3 high_accuracy_transcribe.py openai \
  --input-mode full \
  --output-prefix high_accuracy_runs/openai/openai_gpt4o_full \
  --postprocess \
  --use-previous-context
```

APIアップロード制限に合わせて、音声は16kHz mono WAVの10分程度のチャンクに自動分割されます。

## 3. Deepgram Nova-3で比較する

Deepgram APIキーは `DEEPGRAM_API_KEY` 環境変数、または `--api-key-file` で指定します。既定では `~/Desktop/Deepgram-apy-key.txt` を探します。

```bash
python3 high_accuracy_transcribe.py deepgram \
  --input-mode samples \
  --samples-dir high_accuracy_runs/samples \
  --output-prefix high_accuracy_runs/deepgram/deepgram_nova3_samples \
  --model nova-3 \
  --language ja
```

全文処理:

```bash
python3 high_accuracy_transcribe.py deepgram \
  --input-mode full \
  --output-prefix high_accuracy_runs/deepgram/deepgram_nova3_full \
  --model nova-3 \
  --language ja
```

## 4. RTX 5090でローカル比較する

RTX 5090のWindows/Linux側でこのフォルダをコピーし、CUDA対応環境で実行します。

```bash
python3 -m pip install -r requirements_high_accuracy.txt
python3 high_accuracy_transcribe.py local \
  --input-mode samples \
  --samples-dir high_accuracy_runs/samples \
  --output-prefix high_accuracy_runs/local/local_large_v3_samples \
  --model large-v3 \
  --device cuda \
  --compute-type float16 \
  --beam-size 5
```

速度優先の比較:

```bash
python3 high_accuracy_transcribe.py local \
  --input-mode samples \
  --samples-dir high_accuracy_runs/samples \
  --output-prefix high_accuracy_runs/local/local_large_v3_turbo_samples \
  --model large-v3-turbo \
  --device cuda \
  --compute-type float16 \
  --beam-size 5
```

## 5. Google Cloud Speech-to-Text `chirp_3`で比較する

Google Cloud側でSpeech-to-Text APIとCloud Storageを有効化し、Application Default Credentialsを設定します。

```bash
python3 -m pip install -r requirements_high_accuracy.txt
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_STORAGE_BUCKET="your-bucket-name"

python3 high_accuracy_transcribe.py google \
  --input-mode samples \
  --samples-dir high_accuracy_runs/samples \
  --output-prefix high_accuracy_runs/google/google_chirp3_samples \
  --model chirp_3 \
  --location asia-northeast1 \
  --dynamic-batch
```

## 6. 比較レポートを作る

```bash
python3 high_accuracy_transcribe.py report \
  --jsons \
    high_accuracy_runs/openai_gpt4o_samples.json \
    high_accuracy_runs/deepgram/deepgram_nova3_samples.json \
    high_accuracy_runs/local/local_large_v3_samples.json \
    high_accuracy_runs/google/google_chirp3_samples.json \
  --output high_accuracy_runs/reports/comparison_report.md
```

レポートでは、専門語の出現数と、既存出力で問題になった反復・誤認識パターンをざっくり確認します。最終採用は、代表区間の実文を目視で比較して決めてください。
