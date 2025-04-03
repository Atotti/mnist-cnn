# MNIST Playground

PyTorchを使用した、MNIST手書き数字分類用の畳み込みニューラルネットワーク（CNN）の実装です。

## プロジェクト概要

このプロジェクトでは、MNISTデータセットの手書き数字分類を行うCNNをモジュール化して実装しています。コードは役割ごとに整理されており、モデルのトレーニングや推論を行うための明確なインターフェースを提供しています。

## インストール

### 必要条件

- uv

### セットアップ

1. リポジトリをクローンします:
   ```bash
   git clone https://github.com/yourusername/mnist-playground.git
   cd mnist-playground
   ```

2. 依存関係をインストールします:
   ```bash
   uv sync --frozen
   ```

## プロジェクト構造

```
mnist-playground/
├── data/                  # MNISTデータセットのダウンロード先
├── src/                   # ソースコード
│   ├── models/            # ニューラルネットワークモデルの定義
│   ├── data/              # データのロードと前処理
│   ├── training/          # トレーニングと評価の関数
│   └── inference/         # 訓練済みモデルを使った推論ユーティリティ
├── main.py                # トレーニング用のエントリーポイント
├── README.md              # このドキュメント
└── pyproject.toml         # プロジェクト設定
```

## 使用方法

### モデルのトレーニング

デフォルトのパラメータでモデルをトレーニングする場合:

```bash
uv run main.py
```

このコマンドは以下を行います:
1. MNISTデータセットをダウンロード（未ダウンロードの場合のみ）
2. CNNモデルを14エポックでトレーニング
3. トレーニング済みモデルを`mnist_cnn.pt`として保存

#### トレーニングオプション

トレーニングスクリプトは以下のコマンドライン引数をサポートしています:

```
--batch-size N        トレーニング用バッチサイズ（デフォルト: 64）
--test-batch-size N   テスト用バッチサイズ（デフォルト: 1000）
--epochs N            トレーニングするエポック数（デフォルト: 14）
--lr LR               学習率（デフォルト: 1.0）
--gamma M             学習率の減衰係数（デフォルト: 0.7）
--no-cuda             CUDAトレーニングを無効化
--no-mps              macOSのGPUトレーニングを無効化
--dry-run             一回のパスでテスト実行
--seed S              ランダムシード（デフォルト: 42）
--log-interval N      ログを出力するバッチ間隔
--save-model          モデルを保存する（デフォルト: True）
```

カスタムパラメータの例:

```bash
uv run main.py --batch-size 128 --epochs 20 --lr 0.01
```

## APIドキュメント

### トレーニングAPI

トレーニングモジュールには、モデルのトレーニングと評価関数が含まれています。

#### モデルのトレーニング

```python
from src.models import Net
from src.data import get_data_loaders
from src.training.trainer import train_model
import torch
import argparse

# モデルの作成
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# データのロード
train_loader, test_loader = get_data_loaders(batch_size=64, test_batch_size=1000)

# 引数の設定
args = argparse.Namespace(
    lr=1.0,
    gamma=0.7,
    epochs=14,
    log_interval=10,
    dry_run=False,
    save_model=True
)

# モデルをトレーニング
trained_model = train_model(model, device, train_loader, test_loader, args)
```

#### 個別のトレーニング・テスト関数

より詳細に制御するには、個別のトレーニング関数とテスト関数を使用できます:

```python
from src.training import train, test

# 1エポックのトレーニング
train(args, model, device, train_loader, optimizer, epoch)

# モデルの評価
accuracy = test(model, device, test_loader)
```

### 推論API

推論モジュールは、訓練済みモデルを使って予測を行うための関数を提供します。

#### 訓練済みモデルのロード

```python
from src.inference import load_model

# ディスクからモデルをロード
model = load_model(model_path="mnist_cnn.pt", device="mps")
```

#### 予測の実行

```python
from src.inference import predict_digit
from PIL import Image

# 画像をロードして数字を予測
image_path = "path/to/digit_image.png"
predicted_digit, probabilities = predict_digit(model, image_path)

print(f"予測された数字: {predicted_digit}")
print(f"信頼度: {probabilities[predicted_digit]:.2f}")

# PIL画像を直接使用
image = Image.open("path/to/digit_image.png")
predicted_digit, probabilities = predict_digit(model, image)

# numpy配列を使用
import numpy as np
image_array = np.array(Image.open("path/to/digit_image.png").convert("L"))
predicted_digit, probabilities = predict_digit(model, image_array)

# PyTorchテンソルを使用
import torch
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
image_tensor = transform(Image.open("path/to/digit_image.png").convert("L"))
predicted_digit, probabilities = predict_digit(model, image_tensor)
```

## モデルアーキテクチャ

CNNモデルのアーキテクチャ:

- 2つの畳み込み層（ReLU活性化関数）
- 最大プーリング層
- 2つの全結合層（正則化用にドロップアウトあり）
- 0から9の10クラス分類のためのLog Softmax出力

## ライセンス

[MIT License](LICENSE)

