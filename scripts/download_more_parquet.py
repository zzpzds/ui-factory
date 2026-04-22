"""
直接下载 WebCode2M parquet 文件到本地缓存。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi
from tqdm import tqdm

DATASET_ID = "xcodemind/webcode2m"


def download_parquet_files(num_files: int = 100):
    """下载 parquet 文件"""
    api = HfApi()

    # 获取所有 parquet 文件
    print("获取文件列表...")
    files = api.list_repo_files(DATASET_ID, repo_type='dataset')
    parquet_files = sorted([f for f in files if f.endswith('.parquet')])
    print(f"共 {len(parquet_files)} 个 parquet 文件")

    # 下载目录
    cache_dir = Path.home() / ".cache/huggingface/hub" / f"datasets--{DATASET_ID.replace('/', '--')}" / "snapshots/f53cd4a9317364e32a6fc7d99dc50761fe54715f/data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 检查已下载
    existing = set(p.name for p in cache_dir.glob("*.parquet"))
    print(f"已存在 {len(existing)} 个文件")

    to_download = [f for f in parquet_files if Path(f).name not in existing]
    print(f"需要下载 {len(to_download)} 个文件")

    for pq_file in tqdm(to_download[:num_files], desc="下载"):
        try:
            local_path = api.hf_hub_download(
                repo_id=DATASET_ID,
                filename=pq_file,
                repo_type="dataset",
            )
            # 文件已缓存到本地，移动到目标位置
            import shutil
            dest = cache_dir / Path(pq_file).name
            if not dest.exists():
                shutil.copy(local_path, dest)
        except Exception as e:
            print(f"\n  下载 {pq_file} 出错: {e}")

    print(f"\n完成，文件保存在: {cache_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=int, default=100)
    args = parser.parse_args()
    download_parquet_files(args.num_files)
