from src.inference import load_model
from src.data.dataset import get_data_loaders
import random
import matplotlib.pyplot as plt
from src.inference import predict_digit
import torch

# ディスクからモデルをロード
model = load_model(model_path="mnist_cnn.pt", device="cpu")

# MNISTデータセットのロード
_, test_loader = get_data_loaders(data_dir="data")

# サンプル数の設定
num_samples = 5

# テストデータセットからランダムにサンプルを選択
dataset = test_loader.dataset
indices = random.sample(range(len(dataset)), num_samples)
samples = [dataset[i] for i in indices]

# 結果表示用のfigureを作成
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

# 各サンプルに対して推論を実行
for i, (image, true_label) in enumerate(samples):
    # 画像を表示
    axes[i].imshow(image.squeeze(), cmap='gray')

    # 推論を実行
    predicted_digit, probabilities = predict_digit(model, image)

    # 結果を表示
    confidence = probabilities[predicted_digit]
    title = f"予測: {predicted_digit}\n実際: {true_label}\n信頼度: {confidence:.2f}"
    axes[i].set_title(title)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png')  # Save the figure to a file
plt.show()  # This will be skipped in non-interactive mode

# 詳細な確率分布を表示
for i, (image, true_label) in enumerate(samples):
    predicted_digit, probabilities = predict_digit(model, image)
    print(f"サンプル {i+1} - 実際の数字: {true_label}, 予測: {predicted_digit}")
    print(f"確率分布: {[f'{p:.4f}' for p in probabilities]}")
    print()
