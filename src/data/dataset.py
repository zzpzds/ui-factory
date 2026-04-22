import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.preprocessing import parse_html_nodes, preprocess_image, get_patch_boxes
from src.models.alignment.cross_modal_alignment import build_alignment_labels

# demo 用的模拟 HTML
DEMO_HTML = """
<html><body>
  <nav style="background:#333; height:60px">
    <a href="/" style="color:white; font-size:18px">Logo</a>
  </nav>
  <main style="padding:20px">
    <img src="hero.jpg" style="width:600px; height:300px"/>
    <h1 style="font-size:32px; color:#333">欢迎</h1>
    <button style="background:#1890ff; color:white; width:120px; height:40px">立即使用</button>
  </main>
</body></html>
"""

# demo 用的模拟节点渲染框（在 224x224 坐标系中），与 DEMO_HTML 节点顺序对应
DEMO_NODE_BOXES = torch.tensor([
    [0,   0,   224, 30 ],  # nav
    [0,   0,   60,  30 ],  # a
    [0,   30,  224, 224],  # main
    [10,  40,  150, 130],  # img
    [10,  140, 100, 170],  # h1（近似）
    [10,  175, 110, 200],  # button
], dtype=torch.float32)

# 对应 DEMO_HTML 的真实 DOM 父索引；html/body 被 SKIP_TAGS 过滤，
# 所以 nav 和 main 的最近保留祖先是 None → -1
DEMO_PARENTS = torch.tensor([-1, 0, -1, 2, 2, 2], dtype=torch.long)


class WebpageDataset(Dataset):
    """
    网页数据集。每条样本包含：截图、DOM 节点文本列表、节点渲染框、patch 框、对齐标签。

    data_dir 下的文件组织（后续正式数据用）：
        data_dir/
            0001/
                screenshot.png
                page.html
                nodes.pt       # 节点渲染框 Tensor [N, 4]
            0002/
                ...

    demo 模式（use_demo=True）：用内置的单条样本跑通 pipeline。
    """

    def __init__(
        self,
        data_dir: str | None = None,
        iou_threshold: float = 0.1,
        use_demo: bool = False,
        max_nodes: int = 50,
    ):
        self.iou_threshold = iou_threshold
        self.max_nodes = max_nodes
        self.patch_boxes = get_patch_boxes()

        if use_demo:
            self.samples = [{"type": "demo"}]
        else:
            assert data_dir is not None, "需要提供 data_dir 或设置 use_demo=True"
            self.samples = self._scan(data_dir)

    def _scan(self, data_dir: str) -> list[dict]:
        samples = []
        for name in sorted(os.listdir(data_dir)):
            d = os.path.join(data_dir, name)
            if not os.path.isdir(d):
                continue
            img_path = os.path.join(d, "screenshot.png")
            nodes_path = os.path.join(d, "nodes.pt")
            texts_path = os.path.join(d, "node_texts.json")
            parents_path = os.path.join(d, "parents.pt")
            # render_pages.py 生成的完整样本（要求 parents.pt，旧样本需重渲）
            if all(os.path.exists(p) for p in (img_path, nodes_path, texts_path, parents_path)):
                samples.append({
                    "type": "rendered",
                    "img_path": img_path,
                    "nodes_path": nodes_path,
                    "texts_path": texts_path,
                    "parents_path": parents_path,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        if sample["type"] == "demo":
            image = preprocess_image("data/web_page_image.png")
            node_texts = parse_html_nodes(DEMO_HTML)
            node_boxes = DEMO_NODE_BOXES
            parents = DEMO_PARENTS
        else:
            # 加载 render_pages.py 预计算的数据
            image = preprocess_image(sample["img_path"])
            with open(sample["texts_path"], encoding="utf-8") as f:
                node_texts = json.load(f)
            node_boxes = torch.load(sample["nodes_path"], weights_only=True)
            parents = torch.load(sample["parents_path"], weights_only=True)

            # 限制节点数量以避免内存溢出
            if len(node_texts) > self.max_nodes:
                node_texts = node_texts[:self.max_nodes]
                node_boxes = node_boxes[:self.max_nodes]
                parents = parents[:self.max_nodes].clone()
                # 被截断掉的父节点置为 -1（根），避免悬空引用
                parents[parents >= self.max_nodes] = -1

        alignment_labels = build_alignment_labels(
            node_boxes, self.patch_boxes, self.iou_threshold
        )

        return {
            "image": image,                   # PIL Image
            "node_texts": node_texts,          # list[str]，长度 N
            "node_boxes": node_boxes,          # [N, 4]
            "parents": parents,                # [N] long，-1 表示根
            "patch_boxes": self.patch_boxes,   # [196, 4]
            "alignment_labels": alignment_labels,  # [N, 196]
        }


def webpage_collate_fn(batch: list[dict]) -> dict:
    """
    自定义 collate_fn：处理 PIL Image 和嵌套列表。
    """
    # 注意：batch size 通常为 1，这里只取第一个元素
    # 如果需要支持更大的 batch，需要进一步处理
    assert len(batch) == 1, "当前实现仅支持 batch_size=1"
    return batch[0]
