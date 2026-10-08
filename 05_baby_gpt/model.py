import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
model.py - 极简字符级因果自回归语言模型 (BabyGPT)

【第一性原理认知】：
1. 字符级词表 (CharTokenizer)：
   将汉字和标点符号映射为连续整数 ID (0, 1, 2, ...)。
   以单字为基本原子单元 (Token)，无需复杂外部重型分词库。

2. 自回归因果生成 (Autoregressive Next-Token Prediction)：
   大语言模型的核心机制：给定前面的字符序列，预测下一个最可能的字符。
   输入:  [c_1, c_2, ..., c_t]
   预测:  c_{t+1}
   将预测出的字拼接到序列末尾，像“文字接龙”一样循环往复。

3. 因果自注意力掩码 (Causal Self-Attention Mask)：
   在自注意力矩阵中叠加下三角因果掩码 (Tril Mask)，将未来位置的注意力权重置为 -inf，
   经 Softmax 后概率为 0，确保时刻 t 只能看到自身及历史字符，杜绝“偷看未来”。

4. 极简网络架构 (BabyGPT)：
   字嵌入 (Token Embedding) + 可学习位置编码 (Positional Embedding)
   + 2 层因果 Transformer 解码块 (Pre-LN + Causal Multi-Head Attention + MLP 残差网络)
   + 最终层归一化 (LayerNorm) + 线性分类头 (Linear LM Head)
   针对五言绝句 24 字符音律结构精准设置 block_size=24，总参数量约 5.4 万，纯 CPU 8 秒飞速收敛！
"""


class CharTokenizer:
    """
    极简字符级分词器：负责汉字文本与 Token ID 之间的双向编解码
    """
    def __init__(self, chars=None):
        self.char_to_idx = {}
        self.idx_to_char = {}
        if chars is not None:
            self._set_vocab(chars)

    def _set_vocab(self, chars):
        # 保证词表顺序确定：特殊未登录标记 <unk> 占位 0
        vocab = ["<unk>"] + sorted(list(set(c for c in chars if c != "<unk>")))
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(vocab)}

    def build_vocab(self, text: str):
        """扫描全量语料，统计所有不重复字符并构建字典"""
        chars = sorted(list(set(text)))
        self._set_vocab(chars)

    def encode(self, text: str) -> list[int]:
        """将文本字符串转换为整数 ID 列表"""
        unk_id = self.char_to_idx.get("<unk>", 0)
        return [self.char_to_idx.get(c, unk_id) for c in text]

    def decode(self, indices: list[int]) -> str:
        """将整数 ID 列表还原为文本字符串"""
        return "".join(self.idx_to_char.get(i, "<unk>") for i in indices)

    def __len__(self):
        return len(self.char_to_idx)

    def save(self, filepath: str):
        """将词表映射持久化为 JSON 文件"""
        data = {
            "char_to_idx": self.char_to_idx,
            "vocab_size": len(self.char_to_idx)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """从 JSON 文件还原分词器"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.char_to_idx = data["char_to_idx"]
        tok.idx_to_char = {int(v): k for k, v in tok.char_to_idx.items()}
        return tok


class CausalSelfAttention(nn.Module):
    """
    带因果下三角掩码的多头自注意力机制 (Causal Multi-Head Self-Attention)
    """
    def __init__(self, d_model: int = 32, n_head: int = 4, block_size: int = 24):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # 一次性投影计算 Q, K, V，运算更紧凑高效
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # 多头注意力聚合后的输出线性投影
        self.out_proj = nn.Linear(d_model, d_model)

        # 因果下三角掩码缓冲区 (下三角为 1，上三角为 0；不计入可学习参数)
        # 强制当前时刻只能看见历史字符，杜绝“偷看未来词”
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # (Batch, SeqLen, Channels)

        # 1. 线性投影并切分为 Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. 变换维度支持多头并行计算: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. 缩放点积注意力打分: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4. 因果掩码：将未来位置打分赋值为 -inf，Softmax 后权重严格为 0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # 5. 加权汇聚价值向量 V: (B, n_head, T, head_dim)
        y = att @ v

        # 6. 多头拼接还原为 (B, T, C) 并通过最终投影层
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """
    标准现代 Pre-LN Transformer 解码块：
    LayerNorm -> Causal Attention -> 残差连接 -> LayerNorm -> MLP -> 残差连接
    """
    def __init__(self, d_model: int = 32, n_head: int = 4, block_size: int = 24):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN 结构：梯度直接沿残差主干高速流通，深层训练更平稳
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BabyGPT(nn.Module):
    """
    极简字符级因果自回归语言模型 (BabyGPT)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_head: int = 4,
        n_layer: int = 2,
        block_size: int = 24
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        # 1. 字向量嵌入层 (Word Token Embedding)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # 2. 序列位置嵌入层 (Learned Positional Embedding)
        self.pos_emb = nn.Embedding(block_size, d_model)

        # 3. 堆叠 Transformer 解码层
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, block_size) for _ in range(n_layer)
        ])

        # 4. 最终层归一化与线性语言模型头
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        前向传播：
        idx: 输入序列 ID 张量，形状为 (B, T)
        targets: 期望预测的目标 ID 张量，形状为 (B, T)（可选，训练时提供）
        """
        B, T = idx.size()
        assert T <= self.block_size, f"输入序列长度 {T} 超出模型最大上下文长度 {self.block_size}"

        # 构造位置序列 [0, 1, ..., T-1] 并计算嵌入之和
        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, d_model)

        # 依次流经各个 Transformer 块
        for block in self.blocks:
            x = block(x)

        # 最终归一化与投影回词表维度
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算自回归交叉熵损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def count_parameters(self) -> int:
        """统计模型所有可学习参数总量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def generate(
    model: BabyGPT,
    tokenizer: CharTokenizer,
    prompt: str = "春",
    max_len: int = 24,
    temperature: float = 0.8,
    top_k: int = None,
    stream_callback=None,
    device: str = "cpu"
) -> str:
    """
    自回归文本生成统一引擎：
    从提示词 prompt 出发，逐字向前滚动预测，直到生成完整五言绝句或换行符。

    参数说明：
    - model: 训练好的 BabyGPT 模型实例
    - tokenizer: 字符级分词器
    - prompt: 待续写的提示词文本前缀
    - max_len: 最大允许生成的额外字符数 (默认: 24)
    - temperature: 采样温度 (<= 0.05 为确定性贪心解码，> 0.05 为概率采样)
    - top_k: 截断只在概率最高的前 k 个候选字中采样
    - stream_callback: 单字生成时的回调函数 (例如 lambda char: sys.stdout.write(char))，用于流式打字机输出
    - device: 计算设备
    """
    model.eval()

    # 1. 规范化输入：去除首尾空白与换行，英文标点智能替换为诗歌标点
    clean_prompt = prompt.strip().replace(",", "，").replace(".", "。")
    if not clean_prompt:
        clean_prompt = "春"

    tokens = tokenizer.encode(clean_prompt)
    unk_id = tokenizer.char_to_idx.get("<unk>", 0)

    # 若提供了流式回调，先回调输出完整的前缀字符
    if stream_callback is not None:
        for ch in clean_prompt:
            stream_callback(ch)

    generated_tokens = []
    generated_count = 0

    with torch.no_grad():
        while generated_count < max_len:
            # 上下文窗口截断：保证输入不超出模型上限 block_size
            curr_input = (tokens + generated_tokens)[-model.block_size:]
            idx_tensor = torch.tensor([curr_input], dtype=torch.long, device=device)

            # 前向计算得到最后一个字符处的下一个字预测分布
            logits, _ = model(idx_tensor)
            next_logits = logits[0, -1, :].clone()

            # 屏蔽 <unk> 标记，保证模型永远不会输出未知占位符
            next_logits[unk_id] = float("-inf")

            # 温度缩放与解码策略
            if temperature <= 0.05:
                # 贪心解码：直接选择概率最高的目标字
                next_token = torch.argmax(next_logits).item()
            else:
                scaled_logits = next_logits / temperature

                # Top-K 截断过滤
                if top_k is not None and top_k > 0:
                    k_val = min(top_k, scaled_logits.size(-1))
                    v, _ = torch.topk(scaled_logits, k_val)
                    scaled_logits[scaled_logits < v[-1]] = float("-inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            char = tokenizer.idx_to_char.get(next_token, "")

            # 终止判断：若遇到换行符，代表一首诗完整结束
            if char == "\n":
                break

            generated_tokens.append(next_token)
            generated_count += 1

            if stream_callback is not None:
                stream_callback(char)

            # 诗律终止判断：标准五言绝句全诗刚好 24 字符（4 句每句 5 字 + 2 个逗号 + 2 个句号）
            # 当生成的字符为句号 '。' 且整首诗长度达到 24 字符以上时，适时完成整篇
            total_len = len(clean_prompt) + generated_count
            if char == "。" and total_len >= 24:
                break

    # 返回原始提示词 + 生成汉字构成的完整诗作
    generated_text = tokenizer.decode(generated_tokens)
    return clean_prompt + generated_text
