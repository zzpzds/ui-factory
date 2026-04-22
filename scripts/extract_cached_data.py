"""
从已缓存的 WebCode2M parquet 文件中提取 HTML 数据。

parquet 文件缓存在：~/.cache/huggingface/hub/datasets--xcodemind--webcode2m/
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

# parquet 缓存目录
PARQUET_CACHE = Path.home() / ".cache/huggingface/hub/datasets--xcodemind--webcode2m/snapshots/f53cd4a9317364e32a6fc7d99dc50761fe54715f/data"


def extract_from_parquet(parquet_path: str, output_dir: str, start_idx: int = 0) -> int:
    """从单个 parquet 文件提取数据"""
    df = pd.read_parquet(parquet_path)
    saved = 0

    for idx, row in df.iterrows():
        sample_dir = os.path.join(output_dir, f"{start_idx + saved:04d}")

        # 跳过已存在
        if os.path.exists(os.path.join(sample_dir, "page.html")):
            saved += 1
            continue

        # 提取 HTML (字段名是 "text")
        html = str(row.get("text", "")) or str(row.get("html", ""))
        if len(html) < 200:
            continue

        try:
            os.makedirs(sample_dir, exist_ok=True)

            with open(os.path.join(sample_dir, "page.html"), "w", encoding="utf-8") as f:
                f.write(html)

            # 保存元数据
            meta = {
                "bbox": str(row.get("bbox", "")),
                "score": int(row["score"]) if pd.notna(row.get("score")) else 0,
                "scale": [int(x) for x in row["scale"]] if pd.notna(row.get("scale")) else [],
                "lang": str(row.get("lang", "")),
                "hash": str(row.get("hash", "")),
            }
            with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            saved += 1

        except Exception as e:
            print(f"\n  处理样本 {start_idx + saved} 出错: {e}")

    return saved


def main(num_samples: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 查找 parquet 文件
    if not PARQUET_CACHE.exists():
        print(f"错误：找不到缓存目录 {PARQUET_CACHE}")
        print("请先运行一次下载以缓存 parquet 文件")
        return

    parquet_files = sorted(PARQUET_CACHE.glob("*.parquet"))
    if not parquet_files:
        print(f"错误：缓存目录中没有 parquet 文件")
        return

    print(f"找到 {len(parquet_files)} 个 parquet 文件")

    # 检查已下载数量
    existing = len([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    print(f"已存在 {existing} 条样本")

    saved = existing
    target = min(num_samples, len(parquet_files) * 1536)

    pbar = tqdm(total=num_samples, initial=saved, desc="提取")

    for pq_file in parquet_files:
        if saved >= num_samples:
            break

        try:
            new_saved = extract_from_parquet(str(pq_file), OUTPUT_DIR, saved)
            saved += new_saved
            pbar.update(new_saved)

            if saved >= num_samples:
                break

        except Exception as e:
            print(f"\n  处理 {pq_file.name} 出错: {e}")

    pbar.close()
    print(f"\n完成：保存 {saved} 条")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    main(args.num_samples)
