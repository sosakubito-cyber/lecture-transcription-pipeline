# 音声書き写しプロジェクト

## 目的

大学講義音声を高精度・低コストで書き起こすための比較・実行プロジェクトです。

主な比較対象:

- OpenAI `gpt-4o-transcribe`
- Google Cloud Speech-to-Text V2 `chirp_3` + Dynamic Batch
- Gemini API 音声入力
- ローカル Whisper / MLX / faster-whisper

最終成果物は原則 `txt` / `srt` / `json` の3形式で保存します。

## 現在の保存場所

```text
~/Documents/音声書き写しプロジェクト
```

主要ディレクトリ:

- `high_accuracy_runs/google/`: Google Chirp 3 Dynamic Batch の出力
- `high_accuracy_runs/openai/`: OpenAI API の出力
- `high_accuracy_runs/gemini/`: Gemini API の出力
- `high_accuracy_runs/reports/`: 比較レポート
- `high_accuracy_runs/samples/`: 代表15分サンプル
- `outputs/mlx/`: MLX / ローカルWhisper出力
- `audio/derived/`: 派生音声・変換済みWAV
- `scripts/legacy/`: 初期の個別スクリプト

詳細な成果物インデックスは `PROJECT_INDEX.md` を参照してください。

## Git管理の有無

Gitリポジトリは初期化されています。

```text
Git: あり
GitHub remote: https://github.com/sosakubito-cyber/lecture-transcription-pipeline
現在の状態: GitHub private repository として運用予定。
```

## GitHub repositoryの有無

個人用GitHub private repositoryとして運用予定です。

```text
GitHub repository: https://github.com/sosakubito-cyber/lecture-transcription-pipeline
Visibility: private
```

音声ファイル、WAV中間ファイル、API出力JSONが大きいため、GitHubへ移す場合は `.gitignore` と成果物管理方針を先に決める必要があります。

## 起動方法

依存関係:

```bash
python3 -m pip install -r requirements_high_accuracy.txt
```

Google Chirp 3 Dynamic Batch の例:

```bash
GOOGLE_APPLICATION_CREDENTIALS=~/Desktop/google_speech_service_account.json \
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

OpenAI の例:

```bash
OPENAI_API_KEY="..." \
python3 high_accuracy_transcribe.py openai \
  --input-mode full \
  --output-prefix high_accuracy_runs/openai/openai_gpt4o_next_full \
  --model gpt-4o-transcribe
```

APIキーの値はファイルやドキュメントに保存しないでください。

## 確認方法

基本確認:

```bash
python3 high_accuracy_transcribe.py --help
```

出力JSONの簡易確認:

```bash
python3 -c 'import json; d=json.load(open("high_accuracy_runs/google/google_chirp3_dynamic_6_4_8_full.json")); print(d["model"], len(d["chunks"]))'
```

既知の誤認識パターン確認:

```bash
rg -n "AFDと、|停止成立|特命|前迷惑|確認をする" high_accuracy_runs/google/google_chirp3_dynamic_6_4_8_full.txt
```

比較レポート:

```bash
python3 high_accuracy_transcribe.py report \
  --jsons high_accuracy_runs/google/google_chirp3_dynamic_6_4_8_full.json \
  --output high_accuracy_runs/reports/latest_report.md
```

## 秘密情報リスク

秘密情報はプロジェクト内に保存しない運用です。

確認済みリスク:

- Google CloudのサービスアカウントJSONはプロジェクト外に置く運用。
- OpenAI / Gemini APIキーは環境変数または外部ファイルから読む設計。
- ドキュメントにはAPIキーのプレースホルダがあるが、実キーではない。
- `high_accuracy_transcribe.py` にはAPIキーを検出するための文字列パターンがあるが、実キーは含まれていない。
- 出力JSONには音声内容の書き起こし本文が含まれるため、講義内容の扱いに注意が必要。

GitHubへ移す前に必要な対策:

- サービスアカウントJSON、`.env`、APIキー、巨大音声、WAV中間ファイルを除外する。
- 共有可能なスクリプト・README・小さいサンプルだけをコミット対象にする。

## 未完了タスク

- Cloud Storage上に残る一時WAVの削除方針決定
- 第8回Chirp 3出力の目視精度確認
- 必要なら第8回をOpenAIでも比較実行
- Google Chirp 3の詳細タイムスタンプ取得可否の追加検証
- 生成済み成果物は引き続きローカル専用として管理する

## shared-projects 登録用メモ

```text
Project: 音声書き写しプロジェクト
Type: Personal / research workflow
GitHub owner: 未定
Repository: https://github.com/sosakubito-cyber/lecture-transcription-pipeline
Local path: ~/Documents/音声書き写しプロジェクト
Status: Active / private
Next action: 生成物除外を維持しながら運用する
```
