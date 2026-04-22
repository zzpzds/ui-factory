"""
从现有数据中随机采样 N 条数据用于训练。
"""
import os
import shutil
import random
import argparse
from pathlib import Path

SOURCE_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/sampled")


def sample_data(num_samples: int, seed: int = 42):
    random.seed(seed)

    # 获取所有样本目录
    all_samples = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    print(f"总样本数: {len(all_samples)}")

    # 随机选择
    sampled = random.sample(all_samples, min(num_samples, len(all_samples)))
    print(f"选择样本数: {len(sampled)}")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 复制选中的样本
    for i, sample_dir in enumerate(sampled):
        dest = OUTPUT_DIR / f"{i:04d}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(sample_dir, dest)

    print(f"数据已保存到: {OUTPUT_DIR}")
    print(f"实际样本数: {len(list(OUTPUT_DIR.iterdir()))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    args = parser.parse_args()
    sample_data(args.num)
