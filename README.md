# ReazonSpeech WebSocket STT Server

`reazon-research/reazonspeech-k2-v2` を使って、16 kHz / mono / little-endian float32 PCM を発話単位で認識する軽量サーバーです。WhisperLive互換に近い `SERVER_READY` と `completed: true` の結果を返し、facilitatorAI側の既存 `WhisperTranscriber` から接続先を変更して試せる構成を目指しています。

## 起動

モデルは起動前にホスト側へダウンロードします。コンテナは `/models` を読み取り専用で参照し、起動時にネットワークからモデルを取得しません。ASRファイルが不足するとサーバー起動時に、VADファイルが不足するとWebSocket接続時にエラーになります。

```bash
# Hugging Face Hub CLI を用意（プロジェクトと同じバージョン）
python3 -m pip install huggingface-hub==2.0.0
mkdir -p models

# ReazonSpeech ASR（既定の MODEL_PRECISION=int8）
HF_HUB_DISABLE_XET=1 HF_HOME="$PWD/models" hf download reazon-research/reazonspeech-k2-v2 \
  tokens.txt \
  encoder-epoch-99-avg-1.int8.onnx \
  decoder-epoch-99-avg-1.int8.onnx \
  joiner-epoch-99-avg-1.int8.onnx

# Silero VAD
curl -fL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o models/silero_vad.onnx

# 全ファイルが揃ったら起動
docker compose up -d --build
docker compose logs -f
```

`MODEL_PRECISION` を変更する場合は、設定に合わせたASRファイルを事前に取得してください。`fp32` は encoder / decoder / joiner の `.onnx`、`int8-fp32` は int8 encoder / joiner と fp32 decoder を使います。`tokens.txt` は共通です。たとえば `fp32` の場合:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME="$PWD/models" hf download reazon-research/reazonspeech-k2-v2 \
  tokens.txt \
  encoder-epoch-99-avg-1.onnx \
  decoder-epoch-99-avg-1.onnx \
  joiner-epoch-99-avg-1.onnx
```

ログに `モデルロード完了` と `WebSocket server started` が出てからヘルス確認してください:

```bash
curl http://localhost:9090/health
# {"status":"ok","model":"reazonspeech-k2-v2"}
```

facilitatorAI側は既存の `WHISPER_LIVE_HOST` をDockerホスト名（同じComposeネットワークなら `reazonspeech-server`）に、`WHISPER_LIVE_PORT` を `9090` に設定します。既存のWhisperLiveと比較する場合は接続先だけを切り替えます。

## WebSocketプロトコル

接続先は `ws://<host>:9090/` または `ws://<host>:9090/ws` です。接続直後に、WhisperTranscriber互換の初期化JSONを送ります。

```json
{"uid":"UUID","language":"ja","task":"transcribe","model":"small","use_vad":true}
```

初期化後、次の応答が返ります。

```json
{"uid":"UUID","message":"SERVER_READY","backend":"reazonspeech-k2-v2"}
```

以後は16 kHz / mono / float32 / little-endian PCMをbinary frameで送信します。Silero VADが発話終了を検出すると、次の確定セグメントを返します。partialは返しません。

```json
{"uid":"UUID","segments":[{"text":"次回までにAPIの実装を行います","completed":true,"start":12.5,"end":17.2}]}
```

通常会議とPTTは別々のWebSocket接続で利用できます。各接続のVADとサンプルバッファは独立し、ASRモデルは起動時に一度だけロードして共有します。推論は共有ロックで直列化しています。

## 設定

`.env.example` を参考に `.env` を作成します。

| 変数 | 既定値 | 内容 |
| --- | --- | --- |
| `STT_PORT` | `9090` | ホストに公開するポート |
| `MODEL_PRECISION` | `int8` | `int8` / `fp32` / `int8-fp32` |
| `NUM_THREADS` | `2` | 推論スレッド数 |
| `VAD_THRESHOLD` | `0.5` | Silero VADしきい値 |
| `VAD_MIN_SPEECH_MS` | `250` | 発話とする最小時間 |
| `VAD_MIN_SILENCE_MS` | `500` | 発話終了とする無音時間 |
| `VAD_MAX_SPEECH_SECONDS` | `30` | 1発話の最大時間 |

## テスト

Python 3.11以降で依存を入れて実行します。

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

テストはWebSocket/HTTPの公開境界を通してREADY、完了応答、2接続の状態分離、不正PCMのエラー応答を確認します。実モデルの音声認識品質、facilitatorAI画面への表示、30分連続運転はローカルテストでは確認していません。実機でのPoC受け入れとして、実音声、既存アプリ接続、長時間運転を別途確認してください。
