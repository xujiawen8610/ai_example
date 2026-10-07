import torch
import torch.nn as nn

"""
model.py - 极简 NLP 外卖评价好差评分类模型与分词向量化工具

【第一性原理认知】：
1. 词表 (Vocabulary)：世界的微型字典。我们选取 6 个代表性极强的关键词。
   - 正向词 (3个)："好吃", "新鲜", "量大"
   - 负向词 (3个)："难吃", "超时", "变质"
2. 词袋向量化 (Multi-Hot Encoding)：
   将任意长度的中文文本，根据是否包含这 6 个词，转化为固定长度为 6 的张量 [x1, x2, ..., x6]。
3. 神经网络结构 (SentimentClassifier)：
   - 线性层 (Linear): 计算总情感得分 z = w1*x1 + w2*x2 + ... + w6*x6 + b
   - Sigmoid: 将得分压缩映射为好评概率 p in (0, 1)
"""

# 全局极简核心词表
VOCAB = ["好吃", "新鲜", "量大", "难吃", "超时", "变质"]

def text_to_vector(text: str) -> torch.Tensor:
    """
    极简分词与向量化器：
    扫描文本中是否包含词表中的关键词，生成形状为 (1, 6) 的 0/1 特征向量。
    例如："这家店量大而且好吃" -> [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    """
    vector = [1.0 if word in text else 0.0 for word in VOCAB]
    return torch.tensor([vector], dtype=torch.float32)

class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 6 个输入词特征对应 6 个可调情感权重 (w1~w6)，加 1 个基准偏置 (b)，共 7 个 float32
        self.linear = nn.Linear(in_features=len(VOCAB), out_features=1)

    def forward(self, x):
        # 1. 线性情感累加打分: z = x * W^T + b
        # 2. Sigmoid 映射: p = 1 / (1 + e^(-z))，将总分转化为 (0, 1) 的好评概率
        return torch.sigmoid(self.linear(x))
