import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import RiskClassifier

"""
train.py - 智能风控二分类模型训练脚本

【业务场景与特征定义】：
1. x1: 交易金额相对平时的偏离倍数 (如平时 100 元，当前 850 元则偏离 8.5 倍)
2. x2: 异地登录时间间隔小时数 (如上一次在北京，当前在广州，间隔多少小时)
3. y:  标签，0 表示正常交易（放行），1 表示疑似盗刷（拦截）
"""

# 获取当前脚本所在目录，保证权重文件始终存放在同级目录下
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")

def generate_data(num_samples: int = 100, seed: int = 42):
    """生成符合风控业务常识的样本数据"""
    torch.manual_seed(seed)
    num_normal = num_samples // 2
    num_fraud = num_samples - num_normal

    # 1. 正常交易 (label 0): 金额偏离小 (0.5~2.5 倍)，异地登录间隔长 (6~24 小时)
    x1_normal = torch.empty(num_normal, 1).uniform_(0.5, 2.5)
    x2_normal = torch.empty(num_normal, 1).uniform_(6.0, 24.0)
    y_normal = torch.zeros(num_normal, 1)

    # 2. 疑似盗刷 (label 1): 金额偏离大 (5.0~12.0 倍)，异地登录间隔极短 (0.1~1.5 小时)
    x1_fraud = torch.empty(num_fraud, 1).uniform_(5.0, 12.0)
    x2_fraud = torch.empty(num_fraud, 1).uniform_(0.1, 1.5)
    y_fraud = torch.ones(num_fraud, 1)

    # 拼接并打乱
    X = torch.cat([torch.cat([x1_normal, x2_normal], dim=1),
                   torch.cat([x1_fraud, x2_fraud], dim=1)], dim=0)
    y = torch.cat([y_normal, y_fraud], dim=0)

    indices = torch.randperm(len(X))
    return X[indices], y[indices]

def train():
    # 固定随机种子以便结果可稳定复现
    torch.manual_seed(42)

    print("=" * 68)
    print("第一步：准备数据 (金融反欺诈风控二分类)")
    print("=" * 68)
    X_train, y_train = generate_data(num_samples=100, seed=42)
    X_val, y_val = generate_data(num_samples=20, seed=123)

    print(f"训练集样本量: {len(X_train)} 个 (正常: 50, 盗刷: 50)")
    print(f"验证集样本量: {len(X_val)} 个 (正常: 10, 盗刷: 10)")

    print("\n" + "=" * 68)
    print("第二步：实例化模型并观察初始随机参数")
    print("=" * 68)
    model = RiskClassifier()

    with torch.no_grad():
        w1_init = model.linear.weight[0, 0].item()
        w2_init = model.linear.weight[0, 1].item()
        b_init = model.linear.bias[0].item()

    print(f"【训练前 - 初始随机权重】 w1 (金额权重) = {w1_init:+.4f}, w2 (间隔权重) = {w2_init:+.4f}")
    print(f"【训练前 - 初始随机偏置】 b  (基准偏置) = {b_init:+.4f}")
    print("说明：此时权重是随机垃圾值，无法准确判断交易风险。")

    # 二元交叉熵损失 (BCELoss)：专门用于度量预测概率与真实 0/1 标签之间的差距
    criterion = nn.BCELoss()
    # 随机梯度下降优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print("\n" + "=" * 68)
    print("第三步：进入训练循环 (通过反向传播自动学习风控决策边界)")
    print("=" * 68)

    best_val_loss = float('inf')
    total_epochs = 60

    for epoch in range(1, total_epochs + 1):
        # 1. 训练阶段
        model.train()
        optimizer.zero_grad()               # 清空梯度
        y_pred = model(X_train)             # 前向计算：输出盗刷概率 p
        loss = criterion(y_pred, y_train)   # 计算交叉熵损失
        loss.backward()                     # 反向传播求梯度
        optimizer.step()                    # 梯度更新：调整 3 个参数

        # 2. 验证阶段
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            # 预测概率 >= 0.5 判定为盗刷 (1)，否则为正常 (0)
            val_preds_binary = (val_pred >= 0.5).float()
            val_acc = (val_preds_binary == y_val).float().mean().item()

        # 每 10 轮打印一次训练进度
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                w1_c = model.linear.weight[0, 0].item()
                w2_c = model.linear.weight[0, 1].item()
                b_c = model.linear.bias[0].item()
            print(f"Epoch [{epoch:02d}/{total_epochs:02d}] | 训练Loss: {loss.item():.4f} | 验证Loss: {val_loss:.4f} | 验证准确率: {val_acc*100:5.1f}% | w1={w1_c:+.4f}, w2={w2_c:+.4f}, b={b_c:+.4f}")

        # 3. 保存验证集表现最优的模型快照
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print("\n" + "=" * 68)
    print("第四步：训练完成与模型保存")
    print("=" * 68)
    print(f"最优权重已保存为: {MODEL_PATH}")

    # 加载已保存权重验证
    best_checkpoint = RiskClassifier()
    try:
        saved_state = torch.load(MODEL_PATH, weights_only=True)
    except TypeError:
        saved_state = torch.load(MODEL_PATH)
    best_checkpoint.load_state_dict(saved_state)
    best_checkpoint.eval()

    with torch.no_grad():
        w1_final = best_checkpoint.linear.weight[0, 0].item()
        w2_final = best_checkpoint.linear.weight[0, 1].item()
        b_final = best_checkpoint.linear.bias[0].item()

    print(f"【学到的风控模型参数】: w1={w1_final:+.4f}, w2={w2_final:+.4f}, b={b_final:+.4f}")
    print("【业务直观解释】:")
    print(f"  - 金额权重 w1 为正 ({w1_final:+.4f})：交易金额偏离越大，判定为盗刷的风险概率越高！")
    print(f"  - 间隔权重 w2 为负 ({w2_final:+.4f})：异地登录间隔越长，判定为盗刷的风险概率越低！")
    print("结论：神经网络成功通过数据学习到了符合真实金融业务常识的风控决策逻辑。")

if __name__ == "__main__":
    train()
