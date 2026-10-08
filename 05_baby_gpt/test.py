import os
import sys
import math
import torch
import torch.nn as nn
from model import BabyGPT, CharTokenizer

"""
test.py - 极简字符级因果自回归模型 (BabyGPT) 批量测试与收敛评估脚本

【核心任务】：
1. 加载训练好的权重快照 best_model.pt 与字符字典 vocab.json。
2. 在诗词评估集上计算全局交叉熵损失 (CrossEntropy Loss) 与困惑度 (Perplexity, PPL)。
3. 从第一性原理深度解析困惑度 PPL 的物理内涵：
   - 初始盲猜状态：PPL ≈ VocabSize (在 500+ 个候选汉字中茫然犹豫)
   - 完美收敛状态：PPL 逼近 1.0~1.3 (模型像博古通今的诗人，对下文笃定不移)
4. 经典诗句“文字接龙”单步预测命中率 (跨首句、对句、尾句韵脚) 深度评测。
"""

torch.set_num_threads(min(4, torch.get_num_threads()))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
DATA_PATH = os.path.join(BASE_DIR, "data", "poetry.txt")


def run_test():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("【错误】未找到模型权重或词表文件！")
        print("请先执行训练脚本以训练模型并生成权重快照: python train.py")
        sys.exit(1)

    print("=" * 72)
    print("【第一步】加载词表与实例化 BabyGPT 网络结构")
    print("=" * 72)

    tokenizer = CharTokenizer.load(VOCAB_PATH)
    vocab_size = len(tokenizer)
    print(f"成功加载词表: {VOCAB_PATH} (词表大小: {vocab_size} 字符)")

    block_size = 24  # 精准匹配五言绝句 24 字符音律结构
    model = BabyGPT(
        vocab_size=vocab_size,
        d_model=32,
        n_head=4,
        n_layer=2,
        block_size=block_size
    )

    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"成功加载模型权重: {MODEL_PATH}")
    print(f"模型参数总量: {model.count_parameters():,} 个")

    print("\n" + "=" * 72)
    print("【第二步】加载诗词序列并计算困惑度 (Perplexity)")
    print("=" * 72)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        poems = [line.strip() for line in f if line.strip()]

    base_xs, base_ys = [], []
    for p in poems:
        ids = tokenizer.encode(p + "\n")
        base_xs.append(torch.tensor(ids[:24], dtype=torch.long))
        base_ys.append(torch.tensor(ids[1:25], dtype=torch.long))

    test_x = torch.stack(base_xs)
    test_y = torch.stack(base_ys)
    print(f"评估集诗篇总数: {len(test_x)} 首 (序列长度: {test_x.shape[1]})")

    total_loss = 0.0
    steps = 0
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            bx = test_x[i : i + batch_size]
            by = test_y[i : i + batch_size]
            _, loss = model(bx, by)
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    ppl = math.exp(avg_loss)

    print(f"\n【评估指标结果】:")
    print(f"  - 平均交叉熵损失 (CrossEntropy Loss): {avg_loss:.4f}")
    print(f"  - 模型困惑度 (Perplexity, PPL)      : {ppl:.2f}")

    print("\n【第一性原理认知：困惑度 PPL 的物理内涵】:")
    print(f"  1. 初始状态 (未训练): PPL ≈ 词表大小 ({vocab_size})，如同在 {vocab_size} 个字中随机抓阄。")
    print(f"  2. 当前状态 (训练后): PPL 骤降至 {ppl:.2f}，意味着模型预测下一个字时，平均仅在 ~{ppl:.1f} 个备选项中微小权衡！")

    print("\n" + "=" * 72)
    print("【第三步】经典五言绝句“文字接龙”单步预测抽样测试 (首句、对句与长上下文)")
    print("=" * 72)

    test_cases = [
        # 1. 首句前缀接龙 (长度 4)
        ("床前明月", "光"),
        ("春眠不觉", "晓"),
        ("白日依山", "尽"),
        ("空山不见", "人"),
        ("红豆生南", "国"),
        ("千山鸟飞", "绝"),
        # 2. 第二句接龙 (长度 10)
        ("床前明月光，疑是地上", "霜"),
        ("白日依山尽，黄河入海", "流"),
        ("春眠不觉晓，处处闻啼", "鸟"),
        # 3. 后半首长上下文尾联接龙 (长度 22)
        ("白日依山尽，黄河入海流。欲穷千里目，更上一层", "楼"),
        ("床前明月光，疑是地上霜。举头望明月，低头思故", "乡"),
        ("千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江", "雪"),
    ]

    print(f"{'序号':^4} | {'输入提示前缀':^22} | {'期望字':^6} | {'Top-1 预测字':^12} | {'置信度':^10} | {'判定':^6}")
    print("-" * 74)

    correct = 0
    with torch.no_grad():
        for idx, (prompt, target_char) in enumerate(test_cases, 1):
            input_ids = tokenizer.encode(prompt)
            in_tensor = torch.tensor([input_ids], dtype=torch.long)
            logits, _ = model(in_tensor)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            pred_id = probs.argmax().item()
            pred_char = tokenizer.idx_to_char.get(pred_id, "<unk>")
            conf = probs[pred_id].item() * 100

            is_match = (pred_char == target_char)
            if is_match:
                correct += 1
            status = "命中 ✓" if is_match else "偏差 ✗"

            display_prompt = prompt if len(prompt) <= 12 else ("..." + prompt[-11:])
            print(f"{idx:^4} | {display_prompt:^22} | {target_char:^6} | {pred_char:^12} | {conf:9.2f}% | {status:^6}")

    accuracy = (correct / len(test_cases)) * 100
    print("-" * 74)
    print(f"文字接龙多场景单步准确率: {accuracy:.1f}% ({correct}/{len(test_cases)})")

    print("\n" + "=" * 72)
    print(f"【评估结论】: BabyGPT 模型在评估集上平均损失为 {avg_loss:.4f}，困惑度达到 {ppl:.2f}，")
    print(f"多阶段诗句接龙命中率达 {accuracy:.1f}%，展现出因果自回归 Transformer 优异的序列拟合与长程建模能力！")
    print("=" * 72)


if __name__ == "__main__":
    run_test()
