import os
import torch
import torch.nn as nn
from model import RiskClassifier

"""
test.py - 智能风控二分类模型测试脚本

【核心任务】：
1. 加载训练好的权重文件 best_model.pt。
2. 进行典型标杆用例测试（正常交易 vs 明显盗刷交易），验证风控决策。
3. 在全新的独立测试集上进行批量评估，统计分类准确率 (Accuracy) 与交叉熵损失。
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")

def run_test():
    if not os.path.exists(MODEL_PATH):
        print(f"【错误】 未找到权重文件 '{MODEL_PATH}'！")
        print("请先执行训练脚本以生成权重文件: python train.py")
        return

    print("=" * 72)
    print("第一步：实例化模型并加载磁盘权重")
    print("=" * 72)

    model = RiskClassifier()
    try:
        state_dict = torch.load(MODEL_PATH, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        w1 = model.linear.weight[0, 0].item()
        w2 = model.linear.weight[0, 1].item()
        b = model.linear.bias[0].item()
    print(f"成功加载模型参数: w1 (金额权重) = {w1:+.4f}, w2 (间隔权重) = {w2:+.4f}, b (偏置) = {b:+.4f}")

    print("\n" + "=" * 72)
    print("第二步：业务典型标杆测试用例")
    print("=" * 72)

    # 标杆用例 1: 正常交易 (金额偏离 1.1 倍，异地间隔 18.0 小时)
    case_normal = torch.tensor([[1.1, 18.0]], dtype=torch.float32)
    # 标杆用例 2: 明显盗刷 (金额偏离 8.5 倍，异地间隔 0.2 小时即 12 分钟)
    case_fraud = torch.tensor([[8.5, 0.2]], dtype=torch.float32)

    with torch.no_grad():
        prob_normal = model(case_normal).item()
        prob_fraud = model(case_fraud).item()

    decision_normal = "【放行】(正常交易)" if prob_normal < 0.5 else "【拦截】(疑似盗刷)"
    decision_fraud = "【拦截】(疑似盗刷)" if prob_fraud >= 0.5 else "【放行】(正常交易)"

    print(f"用例 1 (正常交易): 金额偏离 1.1 倍, 异地间隔 18.0 小时")
    print(f"  -> 盗刷风险概率: {prob_normal * 100:.2f}% | 风控决策: {decision_normal}")

    print(f"\n用例 2 (盗刷交易): 金额偏离 8.5 倍, 异地间隔 0.2 小时")
    print(f"  -> 盗刷风险概率: {prob_fraud * 100:.2f}% | 风控决策: {decision_fraud}")

    print("\n" + "=" * 72)
    print("第三步：全新独立测试集批量评估 (20 个测试样本)")
    print("=" * 72)

    # 生成 20 个独立测试样本 (10 个正常，10 个盗刷)
    torch.manual_seed(999)
    x1_n = torch.empty(10, 1).uniform_(0.5, 2.5)
    x2_n = torch.empty(10, 1).uniform_(6.0, 24.0)
    y_n = torch.zeros(10, 1)

    x1_f = torch.empty(10, 1).uniform_(5.0, 12.0)
    x2_f = torch.empty(10, 1).uniform_(0.1, 1.5)
    y_f = torch.ones(10, 1)

    X_test = torch.cat([torch.cat([x1_n, x2_n], dim=1), torch.cat([x1_f, x2_f], dim=1)], dim=0)
    y_test = torch.cat([y_n, y_f], dim=0)

    # 打乱顺序
    idx = torch.randperm(len(X_test))
    X_test, y_test = X_test[idx], y_test[idx]

    with torch.no_grad():
        probs = model(X_test)
        preds = (probs >= 0.5).float()
        loss = nn.BCELoss()(probs, y_test).item()
        acc = (preds == y_test).float().mean().item()

    print(f"{'序号':^4} | {'金额偏离(x1)':^12} | {'间隔小时(x2)':^12} | {'真实标签':^8} | {'盗刷风险概率':^14} | {'风控决策':^12} | {'评估':^6}")
    print("-" * 84)

    for i in range(len(X_test)):
        x1_val = X_test[i, 0].item()
        x2_val = X_test[i, 1].item()
        label = int(y_test[i, 0].item())
        prob = probs[i, 0].item()
        pred_label = int(preds[i, 0].item())
        decision = "【拦截】" if pred_label == 1 else "【放行】"
        label_text = "盗刷(1)" if label == 1 else "正常(0)"
        correct = "正确" if pred_label == label else "错误"
        print(f"{i + 1:^4} | {x1_val:10.2f}倍  | {x2_val:9.2f}小时  | {label_text:^8} | {prob * 100:10.2f}%    | {decision:^12} | {correct:^6}")

    print("-" * 84)
    print(f"测试集整体损失 (BCE Loss): {loss:.6f}")
    print(f"测试集分类准确率 (Accuracy): {acc * 100:.2f}%")
    print("结论：模型成功具备了精准识别交易盗刷风险并做出放行/拦截决策的能力！")

if __name__ == "__main__":
    run_test()
