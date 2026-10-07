import os
import sys
import argparse
import torch
from model import SentimentClassifier, text_to_vector, VOCAB

"""
predict.py - 外卖评价情感分析单句推理与交互测试脚本
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型权重文件 '{MODEL_PATH}'，请先运行 python train.py")
    model = SentimentClassifier()
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_text(model, text: str):
    vec = text_to_vector(text)
    x_tensor = torch.tensor([vec], dtype=torch.float32)

    with torch.no_grad():
        prob = model(x_tensor).item()

    hit_words = [w for w, v in zip(VOCAB, vec) if v > 0]
    decision = "【好评】(正面倾向)" if prob >= 0.5 else "【差评】(负面倾向)"

    print("-" * 60)
    print(f"输入文本:     \"{text}\"")
    print(f"好评概率:     {prob * 100:.2f}% (模型输出值: {prob:.4f})")
    print(f"情感判定:     {decision}")
    print("\n【数学打分细节拆解】:")
    weights = model.linear.weight.squeeze(0).tolist()
    bias = model.linear.bias.item()
    print(f"  - 基准偏置 (b)         : {bias:+.4f}")
    if hit_words:
        for w in hit_words:
            idx = VOCAB.index(w)
            print(f"  - 命中词汇 '{w}' (权重) : {weights[idx]:+.4f}")
    else:
        print("  - 未命中核心词库关键词 (仅依赖基准偏置做出判断)")
    print("-" * 60)
    return prob


def interactive_mode(model):
    print("=" * 60)
    print("  进入外卖评价情感分析交互模式 (输入 'q' 退出)")
    print(f"  模型关注词汇表: {VOCAB}")
    print("=" * 60)
    while True:
        try:
            text = input("\n请输入外卖评价: ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text:
                continue
            predict_text(model, text)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"【推理错误】: {e}")


def main():
    parser = argparse.ArgumentParser(description="外卖评价情感分析推理工具")
    parser.add_argument("text", nargs="?", default="", help="待分析的中文评价文本 (可选)")
    parser.add_argument("-i", "--interactive", action="store_true", help="启动交互式对话循环分析模式")
    args = parser.parse_args()

    model = load_model()

    if args.interactive:
        interactive_mode(model)
    elif args.text:
        predict_text(model, args.text)
    else:
        print("未指定文本，运行默认测试样例:")
        predict_text(model, "这个外卖也太好吃呢")
        predict_text(model, "菜品变质了，而且超时严重")
        print("\n提示: 可通过命令行直接输入文本: python predict.py \"分量很足而且很新鲜\"")
        print("      或启动交互模式: python predict.py -i")


if __name__ == "__main__":
    main()
