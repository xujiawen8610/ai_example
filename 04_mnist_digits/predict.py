import os
import sys
import argparse
import torch
from model import DigitCNN, preprocess_image

"""
predict.py - 极简手写数字图像单图实时推理脚本

【核心任务】：
1. 接收任意手写数字图像路径作为命令行参数。
2. 自动完成图像加载、灰度转换、28x28 尺寸缩放、底色自适应反转与像素标准化。
3. 送入 DigitCNN 卷积神经网络进行前向推理计算。
4. 经过 Softmax 函数输出 0~9 各数字类别的置信度概率分布柱状图。
5. 高亮输出最终识别判定的数字与置信度。

【运行示例】：
  python predict.py
  python predict.py sample_digit.png
  python predict.py sample_digit_2.png
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")


def load_model():
    """加载已训练好的 DigitCNN 权重快照"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型权重文件: '{MODEL_PATH}'！\n请先运行训练脚本生成权重: python train.py"
        )

    model = DigitCNN()
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, image_path: str):
    """端到端推理单张数字图片并打印概率分布"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到输入的图片文件: '{image_path}'")

    # 1. 图像预处理 (Pillow 解码 -> 28x28 灰度 -> 自适应反转 -> 归一化 Tensor)
    tensor = preprocess_image(image_path)

    # 2. 前向传播计算 logits
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # 3. 获取最优预测结果
    pred_digit = probs.argmax().item()
    confidence = probs[pred_digit].item() * 100

    # 4. 格式化可视化输出
    print("=" * 64)
    print(f"  输入图像: {image_path}")
    print(f"  预处理完成: 已转换为标准 (1, 1, 28, 28) 归一化输入张量")
    print("=" * 64)
    print("\n【0~9 各数字置信度概率分布】:")
    print("-" * 64)

    bar_total_len = 24
    for digit in range(10):
        p = probs[digit].item() * 100
        bar_len = min(bar_total_len, max(0, round((p / 100.0) * bar_total_len)))
        bar_str = "█" * bar_len + " " * (bar_total_len - bar_len)
        mark = "  <-- [预测识别结果]" if digit == pred_digit else ""
        print(f"  数字 {digit} : [{bar_str}] {p:6.2f}%{mark}")

    print("-" * 64)
    print("\n" + "=" * 64)
    print(f"  🎯 最终识别判定: 【 {pred_digit} 】 (置信度: {confidence:.2f}%)")
    print("=" * 64)
    return pred_digit, confidence


def main():
    parser = argparse.ArgumentParser(
        description="极简手写数字识别图像推理工具 (DigitCNN)",
        epilog="使用示例:\n"
               "  python predict.py\n"
               "  python predict.py sample_digit.png\n"
               "  python predict.py sample_digit_2.png\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="sample_digit.png",
        help="待识别的手写数字图片路径 (默认: sample_digit.png)"
    )

    args = parser.parse_args()

    try:
        model = load_model()
    except FileNotFoundError as err:
        print(f"【错误】: {err}")
        sys.exit(1)

    try:
        predict_image(model, args.image_path)
    except Exception as err:
        print(f"【识别失败】: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
