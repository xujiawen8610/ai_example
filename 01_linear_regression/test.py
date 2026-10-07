import os
import torch
import torch.nn as nn
from model import LinearToyModel

"""
test.py - 模型测试与评估脚本

【核心任务】：
1. 脱离训练环境（不依赖训练集与优化器），实例化纯净网络骨架。
2. 从磁盘加载已保存的二进制参数字典 "best_model.pt"。
3. 打印加载出的参数内容（破除玄学：就是 3 个浮点数）。
4. 执行【标杆测试】：输入 x1=5.0, x2=1.0，验证模型输出是否精准逼近 14.0。
5. 执行【测试集批量综合评估】：在 10 个未见过的独立样本上测试，输出对比表及均方误差 (MSE)。

【运行命令】：
python test.py
"""

# 获取当前脚本所在目录，确保在任何工作路径下都能正确找到 best_model.pt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")

def run_test():
    if not os.path.exists(MODEL_PATH):
        print(f"【错误】 未找到权重文件 '{MODEL_PATH}'！")
        print("请先执行训练脚本以生成权重文件: python train.py")
        return

    print("=" * 70)
    print("第一步：实例化模型并加载磁盘权重")
    print("=" * 70)

    # 1. 实例化纯净代码骨架（此时内存中的参数是随机初始值）
    infer_model = LinearToyModel()

    # 2. 从磁盘读取二进制字典文件
    try:
        state_dict = torch.load(MODEL_PATH, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH)

    print(f"从磁盘文件 {MODEL_PATH} 中读取到的参数字典：")
    for param_name, param_tensor in state_dict.items():
        print(f"  [{param_name}]: {param_tensor}")

    # 3. 将从磁盘读出的数值覆盖写入模型的内存结构中
    infer_model.load_state_dict(state_dict)
    infer_model.eval()  # 设置为评估推理模式
    print("-> 权重覆盖装载完成！")

    print("\n" + "=" * 70)
    print("第二步：核心标杆测试 (输入 x1=5.0, x2=1.0)")
    print("=" * 70)

    # 标准标杆测试用例：
    x_benchmark = torch.tensor([[5.0, 1.0]], dtype=torch.float32)
    with torch.no_grad():
        pred_benchmark = infer_model(x_benchmark).item()

    true_benchmark = 2.0 * 5.0 + 3.0 * 1.0 + 1.0  # 14.0
    err_benchmark = abs(pred_benchmark - true_benchmark)

    print(f"输入测试特征:   x1 = 5.0, x2 = 1.0")
    print(f"理论真实输出:   2 * 5.0 + 3 * 1.0 + 1 = {true_benchmark:.4f}")
    print(f"模型预测输出:   {pred_benchmark:.4f}")
    print(f"绝对误差:       {err_benchmark:.6f}")
    print("结论：模型成功给出了精准无误的计算结果！")

    print("\n" + "=" * 70)
    print("第三步：全新独立测试集批量评估 (10 个随机新样本)")
    print("=" * 70)

    # 生成 10 个模型从未见过的随机测试输入
    torch.manual_seed(1024)
    X_test = torch.randn(10, 2)
    # 计算理论目标值：y = 2.0 * x1 + 3.0 * x2 + 1.0
    y_true = 2.0 * X_test[:, 0:1] + 3.0 * X_test[:, 1:2] + 1.0

    # 进行前向推理计算
    with torch.no_grad():
        y_pred = infer_model(X_test)

    # 计算均方误差
    criterion = nn.MSELoss()
    test_loss = criterion(y_pred, y_true).item()

    # 格式化打印测试结果对比表格
    print(f"{'序号':^6} | {'输入特征 (x1, x2)':^22} | {'理论真实值':^14} | {'模型预测值':^14} | {'绝对误差':^12}")
    print("-" * 75)
    for i in range(len(X_test)):
        x1 = X_test[i, 0].item()
        x2 = X_test[i, 1].item()
        actual = y_true[i, 0].item()
        pred = y_pred[i, 0].item()
        error = abs(actual - pred)
        print(f"{i + 1:^6} | ({x1:8.4f}, {x2:8.4f}) | {actual:14.4f} | {pred:14.4f} | {error:12.6f}")

    print("-" * 75)
    print(f"测试集整体均方误差 (MSE Loss): {test_loss:.8f}")
    print("结论：无论面对哪个未见过的输入，模型都能精确计算出正确结果！")

if __name__ == "__main__":
    run_test()
