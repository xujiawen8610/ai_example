import torch
import torch.nn as nn

"""
model.py - 智能风控交易二分类器模型结构定义

【第一性原理认知】：
分类问题本质上是“给输入打一个加权分，再把分值映射为概率”：
1. 线性变换层 (Linear): 计算综合风险得分 z = w1 * x1 + w2 * x2 + b
2. Sigmoid 激活函数: 将任意范围的得分 z 压缩映射至 (0, 1) 区间，表示盗刷概率 p
"""

class RiskClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(2, 1) 在内存中开辟 3 个 float32 空间：
        # - weight: 形状为 (1, 2) 的权重矩阵，对应特征 x1 (金额偏离) 和 x2 (登录间隔)
        # - bias:   形状为 (1,) 的偏置向量，对应基准偏置 b
        self.linear = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        # 1. 线性加权评分: z = x1 * w1 + x2 * w2 + b
        # 2. Sigmoid 映射: p = 1 / (1 + e^(-z))，将分值压缩为 (0, 1) 之间的概率
        return torch.sigmoid(self.linear(x))
