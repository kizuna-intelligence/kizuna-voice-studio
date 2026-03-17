# Kizuna Voice Studio

`Kizuna Voice Studio` は、自分だけの読み上げ音声を作るためのデスクトップアプリです。

難しい設定をなるべく減らし、

- どんな声にしたいかを日本語で書く
- 種音声を聞いて確認する
- よければそのまま TTS を作る

という流れで使えるようにしています。

## 先に知っておくこと

- 現在の配布版は、基本的に `NVIDIA GPU` か `Apple Silicon` を前提にしています
- `Windows` は `NVIDIA GPU 版` に加えて、`AMD / 非NVIDIA 向け互換版` も用意しています
- `Linux` は `NVIDIA GPU 版` を使う前提です
- `CPU版` は現時点では配布していません
- 初回起動時は、必要な Python 環境やモデルを自動で準備するため時間がかかります
- 音声モデルをダウンロードするタイミングでは、数分以上かかることがあります

迷ったときは、まず次の考え方で選ぶのが簡単です。

- `Windows / Linux` で NVIDIA GPU がある人: `windows-nvidia` または `linux-nvidia`
- `Windows` で AMD GPU の人: `windows-amd`
- `macOS` で Apple Silicon の人: `macos-apple-silicon`

## ダウンロードと起動

最新版の配布ファイルは、GitHub の Releases からダウンロードします。

- Releases: https://github.com/kizuna-intelligence/kizuna-voice-studio/releases

ファイル名の目安は次のとおりです。

- `Windows`: `VoiceFactory-Windows-NVIDIA-<version>.exe`
- `Windows AMD`: `VoiceFactory-Windows-AMD-<version>.exe`
- `Linux`: `VoiceFactory-Linux-NVIDIA-<version>.AppImage`
- `macOS`: `VoiceFactory-macOS-Apple-Silicon-<version>.dmg` または `.zip`

### Windows

1. Releases ページから `VoiceFactory-Windows-NVIDIA-<version>.exe` をダウンロードします
2. ダウンロードした `.exe` をダブルクリックします
3. インストーラーが出たら、そのまま進めます
4. インストール後、スタートメニューまたはデスクトップから `Kizuna Voice Studio` を起動します

AMD GPU の Windows マシンでは、`VoiceFactory-Windows-AMD-<version>.exe` を使ってください。
この版は現時点では `CPU / 互換経路` を優先して動くため、NVIDIA 版より遅くなることがあります。

### Linux

1. Releases ページから `VoiceFactory-Linux-NVIDIA-<version>.AppImage` をダウンロードします
2. 実行権限を付けます

```bash
chmod +x VoiceFactory-Linux-NVIDIA-<version>.AppImage
```

3. そのまま起動します

```bash
./VoiceFactory-Linux-NVIDIA-<version>.AppImage
```

### macOS

1. Releases ページから `VoiceFactory-macOS-Apple-Silicon-<version>.dmg` または `.zip` をダウンロードします
2. `Applications` にドラッグするか、展開したアプリを配置します
3. `Kizuna Voice Studio` を起動します

### 最初の起動で起こること

最初の起動では、アプリが自動で次の準備をします。

- Python 実行環境のセットアップ
- backend 依存ライブラリのインストール
- 必要なモデルのダウンロード

そのため、初回だけは少し待ち時間があります。

## できること

- 日本語の説明から種音声を作る
- 種音声を聞いてから本番の音声モデル作成に進む
- `Piper TTS` または `Style-Bert-VITS2` を選んで学習する
- 学習なしで `MioTTS` 用の参照音声パッケージを作る
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

種音声が気に入っていて、学習はまだしたくない場合は、
`MioTTS 参照音声パッケージ` を作ってすぐ使うこともできます。

また、書き出した package をダウンロードする前に、
アプリ上で自由なテキストを入れて試聴できます。

- `Piper TTS` package の試聴
- `Style-Bert-VITS2` package の試聴
- `MioTTS` 参照音声 package の試聴

`Style-Bert-VITS2` の package 試聴は、配布版では安定性を優先して
CPU で実行しています。

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
- `windows-amd`
- `linux-nvidia`
- `macos-apple-silicon`

`CPU版` は現時点では実用性が低いため、配布対象から外しています。

## 対応環境の考え方

### Windows / Linux

- NVIDIA GPU 前提の版を用意します
- 重い処理はジョブごとの別プロセスで動きます
- 処理が終わるたびに GPU を解放します

### Windows AMD

- `windows-amd` を用意します
- 現時点では `CPU / 互換経路` を優先した版です
- NVIDIA 版より遅くなる代わりに、AMD 環境でも動作確認しやすくしています

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

### 学習なしで使う MioTTS パッケージ

種音声を作ったあと、その `reference.wav` を同封した `MioTTS` 用 zip を作れます。

この zip は `pip install` すると、参照音声つきの voice module として使えます。
学習済みモデルそのものを同封するのではなく、`MioTTS` の zero-shot API に参照音声を渡して音声生成します。
アプリ上では、ダウンロード前にこの package と同じ経路で 3 本の試聴音声を生成して確認できます。

```bash
pip install miotts-reference-voice.zip
```

```python
from miotts_reference_voice import load_voice

voice = load_voice()
voice.save_wav("こんにちは。よろしくお願いします。", "sample.wav")
```

`MioTTS` のモデルは、現行の既定では `Aratako/MioTTS-1.7B` を使います。
手元で API を立てている場合は、`api_base_url` を差し替えて使えます。

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

ただし、配布運用は次のように分けています。

- `linux-nvidia` はローカル build
- `windows-nvidia` と `macos-apple-silicon` は GitHub Actions の手動実行でも build 可能

自動で毎回 build はせず、release を切るタイミングだけ手動で回す想定です。

### ローカルで配布物を作る

対象 OS 上で次のように実行します。

```bash
scripts/build-electron-release.sh --variant linux-nvidia
```

生成物は `electron/dist/<variant>/` に出ます。

`windows-nvidia` と `macos-apple-silicon` は、GitHub Actions の
`Manual Electron Release Builds` を手動実行して artifact を作る運用でも使えます。

### ローカル build を GitHub Release にアップロードする

```bash
scripts/upload-electron-release-assets.sh --tag v0.1.0 --variant linux-nvidia
```

Windows と macOS の artifact は、手動 workflow で作ったものをダウンロードしてから
同じスクリプトで release に上げられます。

```bash
scripts/upload-electron-release-assets.sh --tag v0.1.0 --variant windows-nvidia
```

```bash
scripts/upload-electron-release-assets.sh --tag v0.1.0 --variant macos-apple-silicon
```

このスクリプトは `aih-gh` を使って release を作成し、同名 asset があれば上書きします。

### 開発用起動

```bash
python -m pip install -e .[voice,train]
cd electron
npm install
npm start
```

## GitHub Actions

GitHub Actions では、通常時は軽い構文確認だけを行います。
重い Electron 配布ビルドは自動では走らせず、必要なときだけ手動で回します。

workflow:

- [.github/workflows/repository-checks.yml](.github/workflows/repository-checks.yml)
- [.github/workflows/manual-electron-release-builds.yml](.github/workflows/manual-electron-release-builds.yml)

## クレジット

このソフトでは、主に次のプロジェクトや作者の成果を利用しています。

- `Piper TTS / piper-plus`: ayutaz さん
- `Qwen Voice Designer`: Alibaba / Qwen Team
- `Style-Bert-VITS2`: litagin さん

## ライセンス

このリポジトリ自体のライセンスは `Apache License 2.0` です。

ただし、`Kizuna Voice Designer` を利用する部分、およびそれに由来する機能部分には
`Kizuna Community License` が適用されます。

また、`Style-Bert-VITS2` を利用する部分は upstream のライセンスに従います。
この README では分かりやすさのために `GPL 系` とまとめず、`AGPL v3.0` と明記します。

ライセンスの見方は次のとおりです。

- このリポジトリ全体: [LICENSE](LICENSE)
- `Kizuna Voice Designer` 由来部分: [kizuna-intelligence/kizuna-voice-designer の LICENSE](https://github.com/kizuna-intelligence/kizuna-voice-designer/blob/main/LICENSE)
- `Style-Bert-VITS2` 由来部分: [litagin02/Style-Bert-VITS2 の LICENSE](https://github.com/litagin02/Style-Bert-VITS2/blob/master/LICENSE)
