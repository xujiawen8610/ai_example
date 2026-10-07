import os
import time
import gzip
import struct
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from model import DigitCNN

"""
train.py - 极简手写数字识别模型训练脚本

【核心任务】：
1. 数据加载与自动化就绪：从 data/train.pt 与 data/val.pt 加载数据（若缺失自动触发极简下载器）。
2. 像素张量标准化：将 0~255 灰度值归一化至 [0.0, 1.0]。
3. 训练循环：使用 CrossEntropyLoss 与 Adam 优化器，在普通的 CPU 上 ~10 秒完成 15 轮高效迭代。
4. 验证集监控：跟踪各 Epoch 的验证集准确率，自动保存最高准确率的权重快照 best_model.pt。
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PT = os.path.join(DATA_DIR, "train.pt")
VAL_PT = os.path.join(DATA_DIR, "val.pt")
TEST_PT = os.path.join(DATA_DIR, "test.pt")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")


def ensure_dataset():
    """若本地数据不存在，则自动从高速镜像下载并构建轻量 MNIST 子集"""
    if os.path.exists(TRAIN_PT) and os.path.exists(VAL_PT) and os.path.exists(TEST_PT):
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 68)
    print("【提示】检测到本地数据文件不完整，正在自动准备 MNIST 轻量子集...")
    print("=" * 68)

    url_candidates = {
        "train_img": [
            "https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
        ],
        "train_lbl": [
            "https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
        ],
        "test_img": [
            "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
        ],
        "test_lbl": [
            "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
        ]
    }

    def fetch(urls):
        for u in urls:
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                return urllib.request.urlopen(req, timeout=15).read()
            except Exception:
                continue
        raise RuntimeError("下载 MNIST 失败，请检查网络连接。")

    # 1. 下载训练与验证集合 (共截取前 2500 张)
    raw_tr_img = fetch(url_candidates["train_img"])
    raw_tr_lbl = fetch(url_candidates["train_lbl"])

    with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw_tr_img)) as f:
        _, _, r, c = struct.unpack(">IIII", f.read(16))
        raw_imgs = torch.frombuffer(bytearray(f.read(2500 * r * c)), dtype=torch.uint8).reshape(2500, 1, r, c)

    with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw_tr_lbl)) as f:
        struct.unpack(">II", f.read(8))
        raw_lbls = torch.frombuffer(bytearray(f.read(2500)), dtype=torch.uint8).long()

    # 2. 下载测试集合 (截取前 500 张)
    raw_te_img = fetch(url_candidates["test_img"])
    raw_te_lbl = fetch(url_candidates["test_lbl"])

    with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw_te_img)) as f:
        _, _, r, c = struct.unpack(">IIII", f.read(16))
        te_imgs = torch.frombuffer(bytearray(f.read(500 * r * c)), dtype=torch.uint8).reshape(500, 1, r, c)

    with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw_te_lbl)) as f:
        struct.unpack(">II", f.read(8))
        te_lbls = torch.frombuffer(bytearray(f.read(500)), dtype=torch.uint8).long()

    # 3. 规范保存各数据集资产 (训练 2000，验证 500，测试 500)
    torch.save({"images": raw_imgs[:2000].clone(), "labels": raw_lbls[:2000].clone()}, TRAIN_PT)
    torch.save({"images": raw_imgs[2000:2500].clone(), "labels": raw_lbls[2000:2500].clone()}, VAL_PT)
    torch.save({"images": te_imgs[:500].clone(), "labels": te_lbls[:500].clone()}, TEST_PT)

    # 4. 生成测试样例图片 (黑底数字 7、黑底数字 2 与白底黑字数字 1)
    sample_path = os.path.join(BASE_DIR, "sample_digit.png")
    if not os.path.exists(sample_path):
        sample_img = Image.fromarray(te_imgs[0, 0].numpy())
        sample_img.save(sample_path)

    sample_2_path = os.path.join(BASE_DIR, "sample_digit_2.png")
    if not os.path.exists(sample_2_path):
        sample_img_2 = Image.fromarray(te_imgs[1, 0].numpy())
        sample_img_2.save(sample_2_path)

    sample_white_path = os.path.join(BASE_DIR, "sample_digit_1_white_bg.png")
    if not os.path.exists(sample_white_path):
        white_bg_arr = 255 - te_imgs[2, 0].numpy()
        Image.fromarray(white_bg_arr.astype("uint8")).save(sample_white_path)

    print("【完成】数据集轻量子集准备完毕！")


def load_data(pt_path: str):
    """读取 .pt 文件并进行灰度浮点标准化 [0.0, 1.0]"""
    data = torch.load(pt_path, weights_only=True)
    images = data["images"].float() / 255.0  # (N, 1, 28, 28)
    labels = data["labels"].long()          # (N,)
    return images, labels


def train():
    # 确保数据集就绪
    ensure_dataset()

    # 固定随机种子确保结果可复现
    torch.manual_seed(42)

    print("=" * 68)
    print("第一步：加载手写数字图像数据集")
    print("=" * 68)

    train_x, train_y = load_data(TRAIN_PT)
    val_x, val_y = load_data(VAL_PT)

    print(f"训练集规模: {len(train_x)} 张图像，尺寸 {tuple(train_x.shape[1:])}")
    print(f"验证集规模: {len(val_x)} 张图像")
    print(f"类别分布: 0~9 共 10 个手写数字类别")

    print("\n" + "=" * 68)
    print("第二步：初始化轻量卷积神经网络 (DigitCNN)")
    print("=" * 68)

    model = DigitCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"网络总参数量: {total_params:,} 个浮点权重")
    print("损失函数: 交叉熵损失 (nn.CrossEntropyLoss)")
    print("优化器: Adam (学习率 lr=0.002)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    print("\n" + "=" * 68)
    print("第三步：进入训练循环 (反向传播与验证集监控)")
    print("=" * 68)

    epochs = 15
    batch_size = 64
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. 训练阶段 (打乱批次)
        model.train()
        indices = torch.randperm(len(train_x))
        epoch_loss = 0.0
        correct_train = 0
        total_train = len(train_x)

        for i in range(0, total_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            bx, by = train_x[batch_idx], train_y[batch_idx]

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(bx)
            correct_train += (logits.argmax(dim=1) == by).sum().item()

        train_loss = epoch_loss / total_train
        train_acc = correct_train / total_train

        # 2. 验证阶段
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()
            val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()

        # 3. 监控验证准确率并保存最佳模型
        saved_tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            saved_tag = " -> [保存最佳模型 ★]"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"训练Loss: {train_loss:.4f} | 训练准确率: {train_acc * 100:5.1f}% | "
              f"验证Loss: {val_loss:.4f} | 验证准确率: {val_acc * 100:5.1f}%{saved_tag}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 68)
    print("第四步：训练完成与模型总结")
    print("=" * 68)
    print(f"总训练耗时: {elapsed:.2f} 秒 (CPU 快速完成)")
    print(f"最佳验证集准确率: {best_val_acc * 100:.2f}%")
    print(f"最优权重已保存至: {MODEL_PATH}")
    print("\n【第一性原理认知总结】:")
    print("卷积神经网络通过在像素矩阵上滑动卷积核，自动发现了诸如‘圆圈’、‘竖线’、‘横折’等数字的关键笔画模式，")
    print("从而在仅用 2000 张样本的情况下，就能对从未见过的手写数字做出超过 95% 准确率的高精度识别！")


if __name__ == "__main__":
    train()
