"""
直接下载 WebCode2M parquet 文件并提取 HTML 数据。

这种方法比流式加载更稳定。
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
DATASET_ID = "xcodemind/webcode2m"


def download_parquet_files(num_samples: int, samples_per_file: int = 1000):
    """下载 parquet 文件并提取数据"""
    api = HfApi()

    # 获取所有 parquet 文件
    print("获取文件列表...")
    files = api.list_repo_files('xcodemind/webcode2m', repo_type='dataset')
    parquet_files = sorted([f for f in files if f.endswith('.parquet')])
    print(f"共 {len(parquet_files)} 个 parquet 文件")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 检查已下载数量
    existing = len([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    print(f"已存在 {existing} 条样本")

    saved = existing
    target = min(num_samples, len(parquet_files) * samples_per_file)

    pbar = tqdm(total=num_samples, initial=saved, desc="处理")

    for pq_file in parquet_files:
        if saved >= num_samples:
            break

        try:
            # 下载 parquet 文件到缓存
            local_path = api.hf_hub_download(
                repo_id=DATATASET_ID,
                filename=pq_file,
                repo_type="dataset",
            )

            # 读取 parquet
            import pandas as pd
            df = pd.read_parquet(local_path)

            for idx, row in df.iterrows():
                if saved >= num_samples:
                    break

                sample_dir = os.path.join(OUTPUT_DIR, f"{saved:04d}")

                # 跳过已存在
                if os.path.exists(os.path.join(sample_dir, "page.html")):
                    saved += 1
                    pbar.update(1)
                    continue

                # 提取 HTML
                html = str(row.get("text", "")) or str(row.get("html", "")) or str(row.get("content", ""))
                if len(html) < 200:
                    continue

                try:
                    os.makedirs(sample_dir, exist_ok=True)

                    with open(os.path.join(sample_dir, "page.html"), "w", encoding="utf-8") as f:
                        f.write(html)

                    # 保存元数据
                    meta = {
                        "bbox": str(row.get("bbox", "")),
                        "score": int(row.get("score", 0)),
                        "scale": list(row.get("scale", [])),
                        "lang": str(row.get("lang", "")),
                        "hash": str(row.get("hash", "")),
                    }
                    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    saved += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"\n  处理样本 {saved} 出错: {e}")

            # 清理缓存
            os.remove(local_path)

        except Exception as e:
            print(f"\n  下载 {pq_file} 出错: {e}")

    pbar.close()
    print(f"\n完成：保存 {saved} 条")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    download_parquet_files(args.num_samples)
