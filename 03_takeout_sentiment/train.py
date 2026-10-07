import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from model import SentimentClassifier, VOCAB, text_to_vector

"""
train.py - 极简 NLP 外卖评价情感分类模型训练脚本

【核心任务】：
1. 数据加载：从独立的 data/train.csv 和 data/val.csv 中读取真实文本数据。
2. 文本向量化：将每句评价自动转换为 6 维词袋张量。
3. 训练循环：使用 BCELoss 计算损失，反向传播更新 6 个词权重。
4. 模型保存：依据验证集 val_loss 挑选并保存最优模型快照 best_model.pt。
5. 原理解析：打印 6 个词最终学到的权重正负，揭秘 AI 是如何“理解”语义的。
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data", "val.csv")

def load_dataset(csv_path: str):
    """读取 CSV 数据集并转换为 PyTorch 张量"""
    texts = []
    labels = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(float(row["label"].strip()))

    # 将所有文本批量转换为 (N, 6) 的特征矩阵
    vectors = [text_to_vector(t) for t in texts]
    X = torch.cat(vectors, dim=0)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return texts, X, y

def train():
    # 固定随机种子确保每次运行结果完全可复现
    torch.manual_seed(42)

    print("=" * 68)
    print("第一步：加载独立数据文件并向量化")
    print(f"核心词表 (VOCAB): {VOCAB}")
    print("=" * 68)

    train_texts, X_train, y_train = load_dataset(TRAIN_CSV)
    val_texts, X_val, y_val = load_dataset(VAL_CSV)

    print(f"训练集样本量: {len(X_train)} 条 (来源于 data/train.csv)")
    print(f"验证集样本量: {len(X_val)} 条 (来源于 data/val.csv)")

    print("
" + "=" * 68)
    print("第二步：实例化模型并观察初始随机参数")
    print("=" * 68)

    model = SentimentClassifier()
    with torch.no_grad():
        initial_weights = model.linear.weight[0].tolist()
        initial_bias = model.linear.bias[0].item()

    print("【训练前 - 初始随机权重】:")
    for word, w in zip(VOCAB, initial_weights):
        print(f"  - 词汇 '{word}': {w:+.4f}")
    print(f"  - 基准偏置 b: {initial_bias:+.4f}")
    print("说明：此时模型完全不懂中文词义，权重是随机乱数，好差评全凭瞎猜。")

    # 二元交叉熵损失：专门用于度量概率输出与 0/1 标签之间的差距
    criterion = nn.BCELoss()
    # 随机梯度下降优化器
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    print("
" + "=" * 68)
    print("第三步：进入训练循环 (通过反向传播自动学习词汇的情感倾向)")
    print("=" * 68)

    best_val_loss = float("inf")
    total_epochs = 60

    for epoch in range(1, total_epochs + 1):
        # 1. 训练阶段
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # 2. 验证阶段
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_preds_binary = (val_pred >= 0.5).float()
            val_acc = (val_preds_binary == y_val).float().mean().item()

        # 每 10 轮输出一次进度
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch:02d}/{total_epochs:02d}] | 训练Loss: {loss.item():.4f} | 验证Loss: {val_loss:.4f} | 验证准确率: {val_acc*100:5.1f}%")

        # 3. 基于 val_loss 保存最优模型快照
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print("
" + "=" * 68)
    print("第四步：训练完成与模型保存")
    print("=" * 68)
    print(f"最优权重已成功保存为: {MODEL_PATH}")

    # 重新加载最优权重以检验学习成果
    best_checkpoint = SentimentClassifier()
    try:
        saved_state = torch.load(MODEL_PATH, weights_only=True)
    except TypeError:
        saved_state = torch.load(MODEL_PATH)
    best_checkpoint.load_state_dict(saved_state)
    best_checkpoint.eval()

    with torch.no_grad():
        final_weights = best_checkpoint.linear.weight[0].tolist()
        final_bias = best_checkpoint.linear.bias[0].item()

    print("
【AI 最终学到的各词汇情感打分 (权重)】:")
    for word, w in zip(VOCAB, final_weights):
        polarity = "【褒义词 (加分项)】" if w > 0 else "【贬义词 (扣分项)】"
        print(f"  - 词汇 '{word:^4}': {w:+.4f} -> {polarity}")
    print(f"  - 基准偏置 b: {final_bias:+.4f}")

    print("
【第一性原理顿悟】:")
    print("计算机并不具有人类情感，所谓‘理解词义’，本质就是微积分梯度下降")
    print("自动将正面评价词的权重推向正数，将负面批评词的权重拉向负数！")

if __name__ == "__main__":
    train()
