import torch
import torch.nn as nn

"""
model.py - 神经网络模型结构定义

【第一性原理认知】：
模型本质上就是一个“带有待调参数的数学计算函数”。
在 PyTorch 中，定义模型就是定义一个类：
1. __init__ 中开辟内存空间，存放可调浮点数（权重 Weight 和偏置 Bias）。
2. forward 中定义计算规则（前向传播），即输入数据如何与这些浮点数做运算。
"""

class LinearToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(in_features=2, out_features=1) 在内存中开辟了 3 个 float32 空间：
        # - weight: 形状为 (1, 2) 的矩阵（即 w1, w2）
        # - bias:   形状为 (1,) 的向量（即 b）
        self.linear = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        # 这里的执行逻辑写死在代码中：y = x * W^T + b
        # 即 y = x1 * w1 + x2 * w2 + b
        return self.linear(x)
