"""
从 HuggingFace 流式采样 WebCode2M，保存前 N 条到 data/processed/。

数据集字段：
- image: PNG 图像
- bbox: 边界框 JSON
- text: HTML 内容
- score, scale, lang, tokens, hash: 元数据
"""
import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
DATASET_ID = "xcodemind/webcode2m"


def main(num_samples: int, max_retries: int = 5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"流式加载 {DATASET_ID}，采样前 {num_samples} 条...")
    print(f"使用镜像: {os.environ.get('HF_ENDPOINT', '默认')}")

    # 检查已下载的数量
    existing = len([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    print(f"已存在 {existing} 条样本")

    dataset = load_dataset(DATASET_ID, split="train", streaming=True)

    saved = existing
    skipped = 0
    retries = 0
    consecutive_errors = 0

    pbar = tqdm(total=num_samples, initial=saved, desc="下载")

    try:
        for idx, item in enumerate(dataset):
            if saved >= num_samples:
                break

            sample_dir = os.path.join(OUTPUT_DIR, f"{saved:04d}")

            # 跳过已处理的
            if os.path.exists(os.path.join(sample_dir, "page.html")):
                saved += 1
                pbar.update(1)
                continue

            # 使用正确的字段名 "text" 而不是 "html"
            html = item.get("text", "") or item.get("html", "") or item.get("content", "")

            if not html or len(html) < 200:
                skipped += 1
                continue

            try:
                os.makedirs(sample_dir, exist_ok=True)

                with open(os.path.join(sample_dir, "page.html"), "w", encoding="utf-8") as f:
                    f.write(html)

                # 保存元数据（不含 image 和 text）
                meta = {
                    "bbox": item.get("bbox", ""),
                    "score": item.get("score", 0),
                    "scale": item.get("scale", []),
                    "lang": item.get("lang", ""),
                    "hash": item.get("hash", ""),
                }
                with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                saved += 1
                pbar.update(1)
                consecutive_errors = 0

                # 每 100 条报告一次
                if saved % 100 == 0:
                    print(f"\n进度: {saved}/{num_samples}")

            except Exception as e:
                print(f"\n  处理样本 {saved} 时出错: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_retries:
                    print("连续错误过多，继续尝试...")
                    consecutive_errors = 0
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n用户中断下载")
    finally:
        pbar.close()

    print(f"\n完成：保存 {saved} 条，跳过 {skipped} 条")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    main(args.num_samples)
