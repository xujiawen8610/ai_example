# 案例五：极简字符级因果自回归模型 (Baby GPT 古诗词生成器)

对应教程：《AI训练入门.md》里程碑范式跃迁篇

---

## 1. 业务与数学背景：从“判别式分类”到“生成式大模型”

在前四个经典案例中，我们完成了判别式 AI 的完整演进路径：
- 从标量拟合（案例一：线性回归）
- 到风控二分类（案例二：逻辑回归）
- 到文本情感分类（案例三：NLP 词袋）
- 再到手写图像识别（案例四：卷积视觉 CNN）

所有判别式模型的本质都是**“给输入贴标签”**（$x \to y$）。

而以 ChatGPT、DeepSeek 为代表的**现代生成式大语言模型（Generative LLM）**，其底层第一性原理本质是**自回归因果语言建模（Autoregressive Causal Modeling）**：

### 核心三原理：
1. **Next-Token 文字接龙**：
   大语言模型并不拥有虚幻的“神性”，它的核心机制极度纯粹：**给定前面的字符序列，预测下一个最可能出现的字符**。
   $$\text{输入序列 } X = [c_1, c_2, \dots, c_t] \implies \text{预测目标 } Y = c_{t+1}$$
2. **因果下三角自注意力掩码 (Causal Mask)**：
   在自注意力机制中，如果位置 $t$ 能够看到位置 $t+1$ 的字符，就如同学生开卷考试直接抄答案，无法学会根据前文自主创作。通过在注意力打分矩阵上叠加**下三角因果掩码（Lower-Triangular Mask）**，强制将未来位置的权重置为 $-\infty$（$\text{Softmax}$ 后为 0），杜绝“偷看未来”。
3. **困惑度评估 (Perplexity, PPL)**：
   困惑度 $PPL = e^{\text{Loss}}$ 直观反映模型预测下一个词时的“犹豫程度”。
   - 随机盲猜时：$PPL \approx \text{词表大小} \approx 566$（在数百个汉字中茫然猜测）；
   - 训练收敛后：$PPL \approx 1.0 \sim 1.1$（模型像饱读诗书的文人，对下一个字笃定不移）。

---

## 2. 网络架构与超参数配置 (BabyGPT)

全模型基于原生 PyTorch 构建，遵循 KISS 原则，无任何重型第三方库：

- **词表与编码**：字符级分词器（基于 60 首经典五言绝句，构建包含 566 个字符的精准字典）。
- **字嵌入 (Token Embedding)**：`nn.Embedding(vocab_size=566, d_model=32)`
- **位置编码 (Positional Embedding)**：`nn.Embedding(block_size=24, d_model=32)`（精准对齐五言绝句 24 字符音律结构，位置 0~23 全部 100% 充分训练）。
- **Transformer 解码块**：2 层标准 Pre-LN 结构：
  - 4 头因果自注意力：`CausalSelfAttention(d_model=32, n_head=4, block_size=24)`
  - 残差前馈网络：`Linear(32, 64) -> ReLU -> Linear(64, 32)`
- **语言模型投影头**：`LayerNorm(32) -> Linear(32, vocab_size=566)`
- **总可学习参数量**：**54,710 个**（严格处于 3万~6万个超轻量黄金区间）。
- **训练耗时**：针对 Windows CPU 做了多核线程调度优化，**25 轮完整训练仅需 8~12 秒**！

---

## 3. 目录文件结构

```text
05_baby_gpt/
├── data/
│   └── poetry.txt       # 经典五言绝句语料库 (李白、孟浩然、王维、杜甫等 60 首名篇)
├── model.py             # 核心模型定义 (CharTokenizer、CausalAttention、TransformerBlock、BabyGPT、generate)
├── train.py             # 极速因果训练脚本 (自动构建词表、Adam 优化、保存 best_model.pt)
├── test.py              # 批量评估脚本 (计算 Loss 与 PPL 困惑度、首句/对句/尾联多场景文字接龙评测)
├── predict.py           # 诗词即时推理工具 (支持单提示词续写、终端交互模式与流式打字机输出)
├── vocab.json           # 持久化的字符级词表映射
├── best_model.pt        # 训练好的模型最优权重快照 (~5.47万参数)
└── README.md            # 本实操手册
```

---

## 4. 快速上手指南

### 第一步：启动训练
```bash
python train.py
```
- 控制台将打印词表大小、参数量统计（54,710 个参数）。
- 在普通 CPU 上仅需约 8~10 秒完成 25 轮训练，验证损失快速从 5.4 骤降至 0.01 左右，困惑度降至 1.01。
- 训练完成后自动保存 `best_model.pt` 和 `vocab.json`，并自动试写 3 首经典古诗。

### 第二步：评估模型困惑度与文字接龙
```bash
python test.py
```
- 计算测试集整体交叉熵损失与困惑度。
- 进行经典古诗“文字接龙”单步预测命中率综合抽样测试（覆盖首句、对句与长上下文尾联）。

### 第三步：单提示词即兴续写
```bash
# 默认提示词 "春"
python predict.py

# 指定前缀，例如 "床前"、"白日"、"红豆"、"空山"
python predict.py "床前"
python predict.py "白日" --temp 0.6
python predict.py "千山" --top_k 5
```

### 第四步：进入终端持续交互写诗模式
```bash
python predict.py -i
```
在交互式终端中随心输入开头词，即可亲眼目睹 BabyGPT 一个字一个字流式自回归吐出整首工整诗篇！输入 `exit` 或 `quit` 退出。
