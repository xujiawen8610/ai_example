# 案例四：极简手写数字 0~9 卷积神经网络识别器 (MNIST Digit Recognition)

对应教程：《AI训练入门.md》第 9 章

---

## 1. 业务与数学背景
- **任务目标**：从一维数据跨越到二维视觉（Computer Vision）。识别手写体数字 $0 \sim 9$ 的多分类任务。
- **输入数据**：$28 \times 28$ 单通道灰度图片，像素值归一化至 $[0.0, 1.0]$。四维张量规范为 $(Batch, 1, 28, 28)$。
- **网络结构 (DigitCNN)**：
  - 特征提取：`Conv2d(1, 16) -> ReLU -> MaxPool2d -> Conv2d(16, 32) -> ReLU -> MaxPool2d`
  - 空间展平：`out.flatten(start_dim=1)` 将 $32 \times 7 \times 7 = 1568$ 个网格特征展平成向量
  - 分类决策：`nn.Linear(1568, 10)` 输出 10 个数字的原始打分（Logits）
  - 总参数量：仅约 20,490 个浮点权重，在普通 CPU 上约 9 秒即可完成训练
- **预处理标准**：基于 MNIST 官方规范自适应处理（外接矩形紧贴裁剪 + 等比缩放至 20x20 + 居中留白贴入 28x28），全面支持外部随手画板涂鸦与任意长宽比白底黑字。
- **损失函数**：多分类交叉熵损失（CrossEntropyLoss）
- **优化算法**：自适应动量优化器（Adam，学习率 lr=0.002）

---

## 2. 目录文件结构
```text
04_mnist_digits/
├── data/
│   ├── train.pt         # 训练集张量数据 (2000张)
│   ├── val.pt           # 验证集张量数据 (500张)
│   └── test.pt          # 测试集张量数据 (500张)
├── model.py             # 网络结构定义 (DigitCNN) 与官方规范图像预处理函数
├── train.py             # 极速训练脚本 (自动就绪轻量数据、Adam 优化、保存 best_model.pt)
├── test.py              # 批量评估脚本 (在 500 张全新测试集上计算各类别细分准确率)
├── predict.py           # 单图推理脚本 (输出 0~9 置信度概率分布柱状图)
├── best_model.pt        # 训练好的模型权重快照
├── sample_digit.png     # 示例手写图片 (数字 7，黑底)
├── sample_digit_2.png   # 示例手写图片 (数字 2，黑底)
├── sample_digit_1_white_bg.png # 示例手写图片 (数字 1，白底黑字)
├── test.png             # 外部手写实测图片 (数字 5)
└── README.md            # 本说明文档
```

---

## 3. 运行指南

### 步骤一：训练模型
```bash
python train.py
```
在普通 CPU 上约 9~10 秒完成 15 轮训练，验证集准确率达 96.6% 以上，并在当前目录下保存最优权重 `best_model.pt`。

### 步骤二：批量测试评估
```bash
python test.py
```
在独立的 500 张全新测试集上进行端到端评估，输出整体分类准确率（约 95.8%）及 0~9 每个数字的明细统计。

### 步骤三：单张手写图片推理预测
```bash
# 默认样例测试 (sample_digit.png)
python predict.py

# 测试黑底手写数字 2
python predict.py sample_digit_2.png

# 测试白纸黑字手写数字 1
python predict.py sample_digit_1_white_bg.png

# 测试外部画板手写数字 5
python predict.py test.png
```
