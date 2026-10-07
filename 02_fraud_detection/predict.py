import os
import sys
import argparse
import torch
from model import RiskClassifier

"""
predict.py - 智能风控交易单笔独立推理与实时决策脚本
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型权重文件 '{MODEL_PATH}'，请先运行 python train.py")
    model = RiskClassifier()
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_one(model, amount_ratio: float, hours_gap: float):
    x_tensor = torch.tensor([[amount_ratio, hours_gap]], dtype=torch.float32)
    with torch.no_grad():
        prob = model(x_tensor).item()

    decision = "【拦截 🚫】(高风险疑似盗刷)" if prob >= 0.5 else "【放行 ✅】(正常交易)"
    print("-" * 56)
    print(f"输入特征: 金额偏离 {amount_ratio:.2f} 倍, 异地间隔 {hours_gap:.2f} 小时")
    print(f"盗刷概率: {prob * 100:.2f}% (模型 Sigmoid 输出: {prob:.4f})")
    print(f"风控决策: {decision}")
    print("-" * 56)
    return prob


def interactive_mode(model):
    print("=" * 56)
    print("  进入智能风控实时决策交互模式 (输入 'q' 退出)")
    print("=" * 56)
    while True:
        try:
            val = input("\n请输入交易数据 (格式: 金额偏离倍数 异地间隔小时数): ").strip()
            if val.lower() in ("q", "quit", "exit"):
                break
            parts = val.split()
            if len(parts) != 2:
                print("【格式错误】: 请输入两个数值，以空格分隔。例如: 8.5 0.2")
                continue
            amt, hrs = float(parts[0]), float(parts[1])
            predict_one(model, amt, hrs)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"【输入错误】: {e}")


def main():
    parser = argparse.ArgumentParser(description="智能风控单笔交易决策推理工具")
    parser.add_argument("features", nargs="*", type=float, help="输入特征: 金额偏离倍数 异地登录间隔小时数 (例如: 8.5 0.2)")
    parser.add_argument("-i", "--interactive", action="store_true", help="启动终端交互式循环测试模式")
    args = parser.parse_args()

    model = load_model()

    if args.interactive:
        interactive_mode(model)
    elif len(args.features) == 2:
        predict_one(model, args.features[0], args.features[1])
    else:
        print("未指定参数，运行默认测试样例:")
        print("1. 正常交易样例:")
        predict_one(model, 1.2, 10.0)
        print("2. 疑似盗刷样例:")
        predict_one(model, 8.5, 0.2)
        print("\n提示: 可通过命令行直接输入测试: python predict.py 8.5 0.2")
        print("      或启动交互模式: python predict.py -i")


if __name__ == "__main__":
    main()
