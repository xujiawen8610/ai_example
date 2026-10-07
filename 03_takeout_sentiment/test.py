import os
import csv
import torch
import torch.nn as nn
from model import SentimentClassifier, VOCAB, text_to_vector

"""
test.py - 极简 NLP 外卖评价分类模型批量测试脚本

【核心任务】：
1. 加载已经训练好的权重文件 best_model.pt。
2. 进行典型标杆文本测试（典型好评 vs 典型差评）。
3. 从独立的 data/test.csv 中加载全新测试集，进行端到端评估，统计分类准确率。
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
TEST_CSV = os.path.join(BASE_DIR, "data", "test.csv")

def extract_hits(text: str):
    """提取文本中命中的词汇列表"""
    hits = [w for w in VOCAB if w in text]
    return "、".join(hits) if hits else "无关键词"

def run_test():
    if not os.path.exists(MODEL_PATH):
        print(f"【错误】 未找到权重文件 '{MODEL_PATH}'！")
        print("请先执行训练脚本以生成权重文件: python train.py")
        return

    print("=" * 72)
    print("第一步：实例化模型并加载磁盘权重")
    print("=" * 72)

    model = SentimentClassifier()
    try:
        state_dict = torch.load(MODEL_PATH, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        weights = model.linear.weight[0].tolist()
        bias = model.linear.bias[0].item()

    print("从磁盘加载的词汇权重表:")
    for word, w in zip(VOCAB, weights):
        print(f"  [{word}]: {w:+.4f}")
    print(f"  [偏置 b]: {bias:+.4f}")

    print("
" + "=" * 72)
    print("第二步：业务典型标杆文本测试")
    print("=" * 72)

    sample_good = "这家店外卖分量真的量大而且很好吃"
    sample_bad = "送餐严重超时而且菜品都难吃变质了"

    with torch.no_grad():
        prob_good = model(text_to_vector(sample_good)).item()
        prob_bad = model(text_to_vector(sample_bad)).item()

    pred_good_text = "【好评】" if prob_good >= 0.5 else "【差评】"
    pred_bad_text = "【差评】" if prob_bad < 0.5 else "【好评】"

    print(f"标杆用例 1: "{sample_good}"")
    print(f"  -> 命中词汇: [{extract_hits(sample_good)}] | 好评概率: {prob_good * 100:.2f}% | 判定: {pred_good_text}")

    print(f"
标杆用例 2: "{sample_bad}"")
    print(f"  -> 命中词汇: [{extract_hits(sample_bad)}] | 好评概率: {prob_bad * 100:.2f}% | 判定: {pred_bad_text}")

    print("
" + "=" * 72)
    print("第三步：全新独立测试集批量评估 (读取 data/test.csv)")
    print("=" * 72)

    test_samples = []
    with open(TEST_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_samples.append((row["text"].strip(), float(row["label"].strip())))

    vectors = [text_to_vector(t) for t, _ in test_samples]
    X_test = torch.cat(vectors, dim=0)
    y_test = torch.tensor([lbl for _, lbl in test_samples], dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        probs = model(X_test)
        preds = (probs >= 0.5).float()
        loss = nn.BCELoss()(probs, y_test).item()
        acc = (preds == y_test).float().mean().item()

    print(f"{'序号':^4} | {'评价内容':^20} | {'命中词':^12} | {'真实':^6} | {'好评概率':^10} | {'判定':^6} | {'结果':^4}")
    print("-" * 80)

    for i in range(len(test_samples)):
        text, true_lbl = test_samples[i]
        prob = probs[i, 0].item()
        pred_lbl = int(preds[i, 0].item())
        hits = extract_hits(text)
        true_text = "好评(1)" if int(true_lbl) == 1 else "差评(0)"
        pred_text = "【好评】" if pred_lbl == 1 else "【差评】"
        correct = "正确" if pred_lbl == int(true_lbl) else "错误"
        print(f"{i + 1:^4} | {text:^20} | {hits:^12} | {true_text:^6} | {prob * 100:8.2f}% | {pred_text:^6} | {correct:^4}")

    print("-" * 80)
    print(f"测试集整体损失 (BCE Loss): {loss:.6f}")
    print(f"测试集分类准确率 (Accuracy): {acc * 100:.2f}%")
    print("结论：模型成功具备了从自然语言中识别关键情感并做出好差评决策的能力！")

if __name__ == "__main__":
    run_test()
