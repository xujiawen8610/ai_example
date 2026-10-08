import os
import sys
import time
import argparse
import torch
from model import BabyGPT, CharTokenizer, generate

"""
predict.py - 极简字符级因果自回归模型 (BabyGPT) 诗词即时推理工具

【核心功能】：
1. 命令行参数驱动：
   - 单提示词作诗: python predict.py "春" 或 python predict.py "床前"
   - 自定义采样温度: python predict.py "春" --temp 0.6
   - 交互式写诗终端: python predict.py -i
2. 自回归流式打字机输出 (Streaming Next-Token Output)：
   一个字一个字实时吐出，直观展现因果自回归大模型每一步文字接龙的物理全过程。
3. 健壮容错与极简设计：
   - 共享 model.py 中的统一生成内核，去除冗余重复代码；
   - 智能处理首尾空白、英文逗号句号与空输入保底；
   - 完美支持 EOF / 管道输入与中断退出。
"""

torch.set_num_threads(min(4, torch.get_num_threads()))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")


def load_model_and_tokenizer():
    """加载已训练好的权重与词表"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(
            f"未找到模型或词表文件！\n请先运行训练脚本生成模型快照: python train.py"
        )

    tokenizer = CharTokenizer.load(VOCAB_PATH)
    model = BabyGPT(
        vocab_size=len(tokenizer),
        d_model=32,
        n_head=4,
        n_layer=2,
        block_size=24
    )
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def stream_generate(
    model: BabyGPT,
    tokenizer: CharTokenizer,
    prompt: str = "春",
    max_len: int = 24,
    temperature: float = 0.8,
    top_k: int = None,
    delay: float = 0.04
) -> str:
    """
    打字机流式输出包装器：
    利用 model.py 中的统一 generate 引擎，通过回调函数实时打印每个字符
    """
    # 若在重定向或非终端环境下运行，取消延时以加速吞吐
    actual_delay = delay if sys.stdout.isatty() else 0.0

    def print_callback(char: str):
        sys.stdout.write(char)
        sys.stdout.flush()
        if actual_delay > 0:
            time.sleep(actual_delay)

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        stream_callback=print_callback
    )
    sys.stdout.write("\n")
    sys.stdout.flush()
    return result


def interactive_mode(
    model: BabyGPT,
    tokenizer: CharTokenizer,
    temperature: float = 0.8,
    max_len: int = 24,
    top_k: int = None
):
    """
    交互式即时作诗模式
    """
    print("=" * 68)
    print("【Baby GPT 交互式古诗词即兴创作终端】")
    print(f"当前采样温度 (Temperature): {temperature} | 最大生成长度: {max_len} 字符")
    print("输入古诗开头字/词（如 '春'、'明月'、'空山'、'红豆'、'独坐' 等）即刻写诗")
    print("输入 'exit' 或 'quit' 退出")
    print("=" * 68)

    while True:
        try:
            prompt = input("\n请输入提示词 (Prompt) > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("退出交互模式。祝您诗意常伴！")
                break

            print("-" * 68)
            print("【自回归生成诗词】:")
            stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                delay=0.04
            )
            print("-" * 68)

        except (KeyboardInterrupt, EOFError):
            print("\n已退出交互模式。祝您诗意常伴！")
            break


def main():
    parser = argparse.ArgumentParser(
        description="BabyGPT 字符级自回归古诗词生成推理工具",
        epilog="使用示例:\n"
               "  python predict.py\n"
               "  python predict.py '春'\n"
               "  python predict.py '床前' --temp 0.6\n"
               "  python predict.py -i\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="春",
        help="待续写的提示词前缀 (默认: '春')"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="开启终端持续交互作诗模式"
    )
    parser.add_argument(
        "-t", "--temp", "--temperature",
        dest="temperature",
        type=float,
        default=0.8,
        help="采样温度系数，范围 0.05~1.5 (默认: 0.8，<= 0.05 为贪心确定性解码)"
    )
    parser.add_argument(
        "-k", "--top_k", "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Top-K 截断采样候选词数量 (默认: None，即不限制)"
    )
    parser.add_argument(
        "-l", "--len", "--max_len",
        dest="max_len",
        type=int,
        default=24,
        help="自回归生成的最大字符上限 (默认: 24)"
    )

    args = parser.parse_args()

    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"【初始化失败】: {e}")
        sys.exit(1)

    if args.interactive:
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            temperature=args.temperature,
            max_len=args.max_len,
            top_k=args.top_k
        )
    else:
        # 单提示词续写模式
        clean_prompt = args.prompt.strip()
        if not clean_prompt:
            clean_prompt = "春"

        print("=" * 68)
        print(f"  输入提示词: 【 {clean_prompt} 】 | 温度: {args.temperature} | 模式: 自回归流式生成")
        print("=" * 68)
        print("\n【生成结果】:")
        print("-" * 68)
        stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=clean_prompt,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            delay=0.04
        )
        print("-" * 68)


if __name__ == "__main__":
    main()
