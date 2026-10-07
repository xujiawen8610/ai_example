import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import LinearToyModel

"""
train.py - 模型训练核心脚本

【核心任务】：
1. 准备数据：基于预设目标公式 y = 2 * x1 + 3 * x2 + 1 生成训练集和验证集。
2. 观察初始状态：展示未经训练前模型内部随机初始化的浮点数（垃圾值）。
3. 训练循环（Epoch Loop）：
   - 前向传播 (Forward)：输入特征，计算预测值
   - 计算损失 (Loss)：衡量预测值与真实目标值的差距
   - 反向传播 (Backward)：自动求导，计算每个参数的梯度
   - 优化更新 (Optimizer Step)：就地修改内存中的 3 个浮点数
4. 保存最优模型：挑选验证集损失最小的一轮，序列化保存为 best_model.pt。

【运行命令】：
python train.py
"""

# 获取当前脚本所在目录，确保权重文件始终保存在同级目录下
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")

# 固定随机种子，确保每次运行的结果完全一致且可复现
torch.manual_seed(42)

print("=" * 65)
print("第一步：准备数据 (目标公式：y = 2.0 * x1 + 3.0 * x2 + 1.0)")
print("=" * 65)

# 1. 训练集：100 个样本，用于在训练中通过梯度下降更新权重
X_train = torch.randn(100, 2)
y_train = 2.0 * X_train[:, 0:1] + 3.0 * X_train[:, 1:2] + 1.0

# 2. 验证集：20 个独立样本，用于在每轮训练后评估当前权重的质量
X_val = torch.randn(20, 2)
y_val = 2.0 * X_val[:, 0:1] + 3.0 * X_val[:, 1:2] + 1.0

print(f"训练集样本量: {len(X_train)} 个 (形状: {list(X_train.shape)})")
print(f"验证集样本量: {len(X_val)} 个 (形状: {list(X_val.shape)})")

print("\n" + "=" * 65)
print("第二步：实例化模型并观察初始随机参数")
print("=" * 65)

model = LinearToyModel()

# 提取并打印初始权重和偏置（使用原生 PyTorch 的 .item() 提取纯 Python 浮点数）
with torch.no_grad():
    w1_init = model.linear.weight[0, 0].item()
    w2_init = model.linear.weight[0, 1].item()
    b_init = model.linear.bias[0].item()

print(f"【训练前 - 初始随机权重】 w1 = {w1_init:.4f}, w2 = {w2_init:.4f}")
print(f"【训练前 - 初始随机偏置】 b  = {b_init:.4f}")
print("说明：此时模型内部是随机浮点数，完全无法算出正确的 y 值。")

# 误差公式 (均方误差 Loss): (预测值 - 真实值)^2 的平均值
criterion = nn.MSELoss()

# 优化器: 随机梯度下降 SGD，学习率 lr=0.1
# 负责在每一轮反向传播后改写模型里的这 3 个浮点数
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("\n" + "=" * 65)
print("第三步：进入训练循环 (用梯度下降自动搜寻最优数值)")
print("=" * 65)

best_val_loss = float('inf')
total_epochs = 60

for epoch in range(1, total_epochs + 1):
    # 1. 训练阶段：更新内存中的浮点数
    model.train()
    optimizer.zero_grad()               # 清空上一轮遗留的梯度缓存
    y_pred = model(X_train)             # 前向计算：执行矩阵乘加运算
    loss = criterion(y_pred, y_train)   # 计算预测误差
    loss.backward()                     # 反向传播：算出每个浮点数需要微调的方向与大小
    optimizer.step()                    # 底层操作：weight = weight - lr * grad

    # 2. 验证阶段：评估当前参数在独立验证集上的表现
    model.eval()
    with torch.no_grad():               # 推理与验证模式下关闭梯度计算，节省算力与内存
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()

    # 每 10 轮输出一次进度，让新手清晰看到 Loss 下降及参数向 [2.0, 3.0] 和 1.0 靠拢的过程
    if epoch == 1 or epoch % 10 == 0:
        with torch.no_grad():
            w1_c = model.linear.weight[0, 0].item()
            w2_c = model.linear.weight[0, 1].item()
            b_c = model.linear.bias[0].item()
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] | 训练Loss: {loss.item():.6f} | 验证Loss: {val_loss:.6f} | 参数: w1={w1_c:.4f}, w2={w2_c:.4f}, b={b_c:.4f}")

    # 3. 挑选验证集损失最优的一轮，写入磁盘保存快照 (Checkpoint)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 将模型参数字典序列化为二进制文件
        torch.save(model.state_dict(), MODEL_PATH)

print("\n" + "=" * 65)
print("第四步：训练完成与保存结果")
print("=" * 65)
print(f"最优权重已成功保存为: {MODEL_PATH}")

# 重新从磁盘载入保存的快照，向新手展示序列化与反序列化验证
best_checkpoint = LinearToyModel()
try:
    saved_state = torch.load(MODEL_PATH, weights_only=True)
except TypeError:
    saved_state = torch.load(MODEL_PATH)
best_checkpoint.load_state_dict(saved_state)

with torch.no_grad():
    w1_final = best_checkpoint.linear.weight[0, 0].item()
    w2_final = best_checkpoint.linear.weight[0, 1].item()
    b_final = best_checkpoint.linear.bias[0].item()

print(f"【目标预设真实参数】: w1 = 2.0000, w2 = 3.0000, b = 1.0000")
print(f"【磁盘快照最终学到】: w1 = {w1_final:.4f}, w2 = {w2_final:.4f}, b = {b_final:.4f}")
print("结论：模型成功通过数据迭代与梯度下降，自动破解出了我们预设的数学公式！")
