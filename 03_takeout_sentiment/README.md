# 案例三：外卖评价 NLP 情感分析二分类器 (Sentiment Analysis)

对应教程：《AI训练入门.md》第 8 章

---

## 1. 业务与数学背景
- **任务目标**：计算机如何“理解”人类自然语言文本。分析中文外卖评价是好评还是差评。
- **输入特征**：中文外卖评价文本。
- **特征工程（词袋模型 Multi-Hot）**：维护 6 个核心情感词词表 `['好吃', '新鲜', '量大', '难吃', '超时', '变质']`，将任意句子转化为 6 维布尔数值向量 $[x_1, x_2, \dots, x_6]$。
- **输出标签**：
  - $y=1$：好评（正面倾向）
  - $y=0$：差评（负面倾向）
- **核心结构**：`nn.Linear(6, 1) + nn.Sigmoid()`
- **数据与代码解耦**：数据集独立存储为 `data/train.csv` (16条)、`data/val.csv` (6条)、`data/test.csv` (6条)。
- **损失函数**：二元交叉熵损失（BCELoss）
- **优化算法**：随机梯度下降（SGD）

---

## 2. 目录文件结构
```text
03_takeout_sentiment/
├── data/
│   ├── train.csv # 训练集 (16条标注样本)
│   ├── val.csv   # 验证集 (6条独立样本)
│   └── test.csv  # 测试集 (6条独立样本)
├── model.py      # 模型结构与词袋分词向量化工具 (text_to_vector)
├── train.py      # 训练脚本 (从 CSV 读取、val_loss 最优模型保存、自动打印学得的词汇权重)
├── test.py       # 离线批量测试脚本 (评估全新 test.csv 上的分类准确率)
├── predict.py    # 单句文本情感分析与交互式推理工具 (含数学打分细节拆解)
└── README.md     # 本说明文档
```

---

## 3. 运行指南

### 步骤一：训练模型
```bash
python train.py
```
读取 `data/train.csv` 与 `data/val.csv` 进行训练，输出每个词语学得的正负情感极性，并保存 `best_model.pt`。

### 步骤二：批量测试
```bash
python test.py
```
在独立的 `data/test.csv` 上批量评估，展示各句子的词汇命中情况与预测概率（准确率达到 100%）。

### 步骤三：单句文本推理与交互
```bash
# 默认样例测试
python predict.py

# 命令行直接输入句子测试
python predict.py "包装完好，分量很足而且很新鲜"

# 交互式对话分析模式
python predict.py -i
```
