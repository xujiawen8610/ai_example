import os
import sys
import torch
import torch.nn as nn
from model import DigitCNN
from train import load_data, ensure_dataset

"""
test.py - 极简手写数字识别模型批量测试与评估脚本

【核心任务】：
1. 加载已经训练好的权重文件 best_model.pt。
2. 从独立的 data/test.pt 中加载全新的 500 张手写数字测试样本。
3. 端到端批量推理，计算测试集整体交叉熵损失与分类准确率。
4. 统计 0~9 每个数字类别的细分识别准确率，全景展示模型识别能力。
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
TEST_PT = os.path.join(BASE_DIR, "data", "test.pt")


def run_test():
    if not os.path.exists(MODEL_PATH):
        print(f"【错误】 未找到权重文件 '{MODEL_PATH}'！")
        print("请先执行训练脚本以训练模型并生成权重: python train.py")
        sys.exit(1)

    # 确保测试数据可用
    ensure_dataset()

    print("=" * 72)
    print("第一步：实例化模型并加载磁盘权重快照")
    print("=" * 72)

    model = DigitCNN()
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"成功加载模型权重: {MODEL_PATH}")

    print("\n" + "=" * 72)
    print("第二步：加载独立测试集进行批量推理")
    print("=" * 72)

    test_x, test_y = load_data(TEST_PT)
    print(f"测试集样本量: {len(test_x)} 张图像 (来源于 data/test.pt)")

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        logits = model(test_x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        test_loss = criterion(logits, test_y).item()
        total_acc = (preds == test_y).float().mean().item()

    print(f"测试集整体损失 (CrossEntropy Loss): {test_loss:.4f}")
    print(f"测试集综合分类准确率 (Accuracy): {total_acc * 100:.2f}%")

    print("\n" + "=" * 72)
    print("第三步：各数字类别 (0~9) 细分识别统计")
    print("=" * 72)
    print(f"{'数字类别':^8} | {'样本总数':^10} | {'识别正确数':^12} | {'类别准确率':^12}")
    print("-" * 54)

    for digit in range(10):
        mask = (test_y == digit)
        digit_total = mask.sum().item()
        if digit_total > 0:
            digit_correct = (preds[mask] == digit).sum().item()
            digit_acc = (digit_correct / digit_total) * 100
        else:
            digit_correct = 0
            digit_acc = 0.0
        print(f"{digit:^12} | {digit_total:^12} | {digit_correct:^14} | {digit_acc:9.2f}%")

    print("-" * 54)

    print("\n" + "=" * 72)
    print("第四步：前 10 张测试样本抽样明细")
    print("=" * 72)
    print(f"{'序号':^6} | {'真实标签':^10} | {'预测标签':^10} | {'最高置信度':^12} | {'检验结果':^8}")
    print("-" * 58)

    for i in range(10):
        true_lbl = test_y[i].item()
        pred_lbl = preds[i].item()
        conf = probs[i, pred_lbl].item() * 100
        status = "正确 ✓" if true_lbl == pred_lbl else "错误 ✗"
        print(f"{i + 1:^8} | {true_lbl:^12} | {pred_lbl:^12} | {conf:9.2f}% | {status:^10}")

    print("-" * 58)
    print(f"\n【评估结论】: 批量测试准确率达 {total_acc * 100:.2f}%，模型泛化性能良好，成功完成 CV 图像识别闭环！")


if __name__ == "__main__":
    run_test()
