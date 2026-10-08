# 零基础从第一性原理探索人工智能 (从 3 个参数到深度学习、视觉识别与生成式大模型)

本项目为配套教程文档 [**`AI训练入门.md`**](./AI训练入门.md) 的完整工程实现代码库。
全项目严格遵循 **KISS 原则（Keep It Simple, Stupid）**，抛弃一切虚无缥缈的技术炒作，从最基础的微积分、线性代数与张量运算出发，由浅入深构建了五个极简、自包含、可完全独立运行的经典 AI 范式。

---

## 体系演进全景导航

```text
.
├── AI训练入门.md              # 核心教程与第一性原理深度讲义 (全篇 9 大章)
├── README.md                 # 全局项目导航索引 (本文档)
│
├── 01_linear_regression/     # 【案例一】数学拟合基石：极简线性回归数值拟合
│   ├── model.py              # 线性模型 (Linear(2, 1))
│   ├── train.py              # MSELoss + SGD 梯度下降搜寻最优浮点参数
│   ├── test.py               # 离线批量测试与偏差统计
│   ├── predict.py            # 单样本独立推理与终端交互模式
│   └── README.md             # 案例一实操手册
│
├── 02_fraud_detection/       # 【案例二】从算数值到做决策：智能风控交易二分类器
│   ├── model.py              # Linear(2, 1) + Sigmoid 概率映射
│   ├── train.py              # BCELoss 交叉熵损失 + 模拟业务风控数据训练
│   ├── test.py               # 分类准确率评估 (Accuracy)
│   ├── predict.py            # 实时风控拦截决策推理
│   └── README.md             # 案例二实操手册
│
├── 03_takeout_sentiment/     # 【案例三】从数字到语言：外卖评价 NLP 情感分析分类器
│   ├── data/                 # 真实数据与代码彻底解耦 (train.csv / val.csv / test.csv)
│   ├── model.py              # 6词词袋向量化编码 (Multi-Hot) + 线性情感分类器
│   ├── train.py              # 真实文本训练 + 自动打印学得的核心词汇情感权重
│   ├── test.py               # 独立测试集批量情感分类评估
│   ├── predict.py            # 单句评价情感极性判定与数学打分细节拆解
│   └── README.md             # 案例三实操手册
│
├── 04_mnist_digits/          # 【案例四】从语言到视觉：极简手写数字 0~9 卷积神经网络识别器
│   ├── data/                 # MNIST 轻量子集 (.pt 连续张量归档)
│   ├── model.py              # 双层卷积网络 (DigitCNN) + Bounding Box 居中预处理
│   ├── train.py              # Adam 优化器 + CrossEntropyLoss (普通 CPU 9 秒飞速收敛)
│   ├── test.py               # 500 张独立测试集综合评估 (准确率达 95.8%+)
│   ├── predict.py            # 任意长宽比手写图片预测 + 0~9 概率分布柱状图可视化
│   ├── best_model.pt         # 训练完成的模型权重快照
│   ├── sample_digit*.png     # 示例手写测试图片资产
│   └── README.md             # 案例四实操手册
│
└── 05_baby_gpt/              # 【案例五】从判别到生成：极简字符级因果自回归模型 (Baby GPT 古诗词生成器)
    ├── data/                 # 经典五言绝句 60 首语料库 (李白、杜甫、王维名篇)
    ├── model.py              # 字符级分词器 + 因果自注意力掩码 + 2 层 Transformer 解码块 (BabyGPT)
    ├── train.py              # 极速因果训练脚本 (Next-Token 预测，普通 CPU 10 秒飞速收敛)
    ├── test.py               # 困惑度 (PPL) 与古诗文字接龙批量测试
    ├── predict.py            # 单提示词续写、终端交互即兴创作与流式打字机输出
    ├── best_model.pt         # 训练好的模型权重快照 (~5.4万参数)
    ├── vocab.json            # 字符级字典映射
    └── README.md             # 案例五实操手册
```

---

## 五大案例的核心对比与技术跨越

| 演进阶段 | 对应目录 | 输入类型 | 业务任务 | 核心网络结构 | 损失函数 | 优化器 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **阶段 1：数学回归** | [`01_linear_regression`](./01_linear_regression/) | 2 个连续数值 | 拟合线性方程 $y = 2x_1 + 3x_2 + 1$ | `Linear(2, 1)` | `MSELoss` | `SGD` |
| **阶段 2：二分类决策** | [`02_fraud_detection`](./02_fraud_detection/) | 2 个风控特征 | 判定交易是正常放行还是异常拦截 | `Linear(2, 1) + Sigmoid` | `BCELoss` | `SGD` |
| **阶段 3：自然语言处理** | [`03_takeout_sentiment`](./03_takeout_sentiment/) | 中文文本句子 | 判定外卖评价是好评还是差评 | `词袋编码 + Linear(6, 1) + Sigmoid` | `BCELoss` | `SGD` |
| **阶段 4：计算机视觉** | [`04_mnist_digits`](./04_mnist_digits/) | $28 \times 28$ 图像 | 识别手写数字 $0 \sim 9$ (10 分类) | `DigitCNN (双层卷积+池化+Linear)` | `CrossEntropyLoss` | `Adam` |
| **阶段 5：生成式大语言模型** | [`05_baby_gpt`](./05_baby_gpt/) | 提示词文本序列 | 给定前缀自回归生成五言绝句 (Next-Token) | `BabyGPT (因果多头自注意力 + 2层解码块)` | `CrossEntropyLoss` | `Adam` |

---

## 快速上手与运行建议

本项目已通过 `uv` 虚拟环境统一管理依赖，全局仅依赖基础的深度学习与图像处理包：
```bash
# 激活环境并进入任意案例目录体验，例如进入案例五 (Baby GPT)：
cd 05_baby_gpt

# 运行极速训练 (~10 秒完成)
python train.py

# 运行困惑度评估与文字接龙测试
python test.py

# 单提示词即兴写诗
python predict.py "春"
python predict.py "床前"

# 开启终端交互式写诗模式
python predict.py -i
```

详细的原理剖析、数学推导与进阶排错哲学，请完整阅读核心讲义 [**`AI训练入门.md`**](./AI训练入门.md)！
