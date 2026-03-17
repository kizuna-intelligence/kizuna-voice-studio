# Kizuna Voice Studio

`Kizuna Voice Studio` は、自分だけの読み上げ音声を作るためのデスクトップアプリです。

難しい設定をなるべく減らし、

- どんな声にしたいかを日本語で書く
- 種音声を聞いて確認する
- よければそのまま TTS を作る

という流れで使えるようにしています。

## できること

- 日本語の説明から種音声を作る
- 種音声を聞いてから本番の音声モデル作成に進む
- `Piper TTS` または `Style-Bert-VITS2` を選んで学習する
- 学習後のモデルを Python パッケージとして書き出す

## どんな人向けか

- 自分のアプリや作品用にオリジナル音声を作りたい人
- コマンドラインより GUI で進めたい人
- モデルや Python の細かい設定をなるべく意識したくない人

## モデルの違い

### Piper TTS

- とても軽いです
- CPU や低めのスペックでも動かしやすいです
- モバイルや組み込み用途に向いています
- その代わり、表現力は `Style-Bert-VITS2` より控えめです

### Style-Bert-VITS2

- `Piper TTS` より表現力が高いです
- 感情や話し方のニュアンスを出しやすいです
- その分、少し強いマシンが必要です

### 迷ったとき

- 軽さ重視なら `Piper TTS`
- 声の自然さや表現力を重視するなら `Style-Bert-VITS2`
- 低めの男性声は `Style-Bert-VITS2` の方が安定しやすい傾向があります

## 種音声の作り方

種音声の生成方法は 2 つから選べます。

### Kizuna Voice Designer

- 日本語の説明をそのまま使えます
- 軽量モードで種音声を作れます
- 翻訳モデルなしで始められます

### Qwen Voice Designer

- 日本語の説明をもとに種音声を作ります
- 必要なモデルは初回に自動でダウンロードされます
- 初回は少し時間がかかります

## 使い方

1. アプリを起動します
2. どんな声にしたいかを日本語で入力します
3. 種音声の生成方法を選びます
4. `1. 種音声を作成` を押します
5. できた音声を聞きます
6. 問題なければ作るモデルを選んで `2. この声で TTS を作る` を押します
7. 完了後、生成されたモデルをダウンロードして使います

## 初回起動について

初回起動時には、必要なものをアプリ側で自動準備します。

- Python 実行環境
- backend の依存ライブラリ
- 必要になったモデル本体

そのため、最初の 1 回だけ時間がかかることがあります。

## 配布版の考え方

このアプリは、環境ごとに配布物を分ける方針です。

現在の主な配布対象は次の 3 つです。

- `windows-nvidia`
- `linux-nvidia`
- `macos-apple-silicon`

`CPU版` は現時点では実用性が低いため、配布対象から外しています。

## 対応環境の考え方

### Windows / Linux

- NVIDIA GPU 前提の版を用意します
- 重い処理はジョブごとの別プロセスで動きます
- 処理が終わるたびに GPU を解放します

### macOS

- Apple Silicon 向けの版を想定しています

## モデルのダウンロードについて

モデルはアプリ本体に全部同梱せず、必要になったタイミングでダウンロードします。

たとえば次のようなモデルが、必要時に自動取得されます。

- 種音声生成モデル
- 翻訳モデル
- 学習用のベースモデル

画面には、

- ダウンロード中です
- 種音声を生成中です
- 学習データセットを作成中です
- モデルを学習中です

のように進捗が表示されます。

## インストール済みモデルの使い方

学習後は、モデルを Python パッケージとして書き出せます。

たとえば `Piper TTS` なら、書き出した zip を `pip install` して使えます。

```bash
pip install piper-voice.zip
```

その後は次のように呼べます。

```python
from piper_voice import load_voice

voice = load_voice()
voice.synthesize_to_file("こんにちは", "sample.wav")
```

`Style-Bert-VITS2` でも同様に installable package を作れます。

## 開発者向けの補足

このリポジトリは、将来的に独立したオープンソースプロジェクトとして扱いやすいように整理しています。

- Electron と backend を分離した構成
- 共有 backend を FastAPI / CLI / Gradio / Electron から利用
- 成果物を `workspace/projects/<project-id>/` に明示的に保存
- 重い処理をジョブ単位の別プロセスで実行

### 主なフォルダ

```text
kizuna-voice-studio/
├── README.md
├── pyproject.toml
├── electron/
├── examples/
└── src/voice_factory/
```

### Electron 配布ビルド

```bash
cd electron
npm run dist:windows-nvidia
npm run dist:linux-nvidia
npm run dist:macos-apple-silicon
```

### 開発用起動

```bash
python -m pip install -e .[voice,train]
cd electron
npm install
npm start
```

## GitHub Actions

GitHub Actions では Electron 配布物を自動ビルドします。

現在の build 対象:

- `windows-nvidia`
- `linux-nvidia`
- `macos-apple-silicon`

workflow:

- [.github/workflows/voice-factory-electron-variants.yml](.github/workflows/voice-factory-electron-variants.yml)

## ライセンス

ライセンスは `kizuna-intelligence/kizuna-voice-designer` と同じ方針です。
詳しくは [LICENSE](LICENSE) を参照してください。
