"""
WebpageDataset：网页截图 + DOM 节点 + 渲染框 + 计算样式。

关键设计点（论文方案 P2/P6/P7 修复）：
- 变长节点用 padding + node_mask 处理；collate 真正支持 batch>1。
- 截断改为 BFS 保留连通子树，避免父子链断裂污染结构监督。
- styles.json（来自 render_pages.py 的 getComputedStyle）作为样式监督信号。
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import deque

from src.data.preprocessing import parse_html_nodes, preprocess_image, get_patch_boxes
from src.models.alignment.cross_modal_alignment import build_alignment_labels

# 与 render_pages.py 的 JS_GET_NODES 字段顺序一致
STYLE_REG_KEYS = ("borderRadius", "borderWidth", "opacity", "fontSize", "fontWeight")
STYLE_COLOR_KEYS = ("backgroundColor", "color", "borderColor")
STYLE_CLS_KEYS = {
    "textAlign": ["start", "left", "center", "right", "justify", "end"],
    "display": ["block", "inline", "inline-block", "flex", "grid", "none", "table", "inline-flex"],
}

# 归一化用：把每个回归项压到大致 [0, 1]
STYLE_REG_NORMS = {
    "borderRadius": 50.0,   # px
    "borderWidth": 10.0,    # px
    "opacity": 1.0,
    "fontSize": 64.0,       # px
    "fontWeight": 900.0,
}


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

DEMO_NODE_BOXES = torch.tensor([
    [0,   0,   224, 30 ],
    [0,   0,   60,  30 ],
    [0,   30,  224, 224],
    [10,  40,  150, 130],
    [10,  140, 100, 170],
    [10,  175, 110, 200],
], dtype=torch.float32)
DEMO_PARENTS = torch.tensor([-1, 0, -1, 2, 2, 2], dtype=torch.long)


def _bfs_truncate(parents: list[int], max_nodes: int) -> list[int]:
    """
    BFS 保留连通子树：从所有根节点开始 BFS，按层级累加，
    超过 max_nodes 时只丢"最深的叶子层"以保父子链完整。
    返回保留的原始索引列表（按 BFS 顺序）。
    """
    N = len(parents)
    if N <= max_nodes:
        return list(range(N))

    children = [[] for _ in range(N)]
    roots = []
    for i, p in enumerate(parents):
        if p < 0 or p >= N:
            roots.append(i)
        else:
            children[p].append(i)

    kept = []
    queue = deque(roots)
    while queue and len(kept) < max_nodes:
        node = queue.popleft()
        kept.append(node)
        for c in children[node]:
            queue.append(c)
    return kept


def _reindex_parents(orig_parents: list[int], kept_indices: list[int]) -> list[int]:
    """
    根据保留的原始索引列表，重新映射父索引。
    被丢弃的祖先链回溯到最近被保留的祖先；找不到则置 -1。
    """
    kept_set = {idx: new_idx for new_idx, idx in enumerate(kept_indices)}
    new_parents = []
    for new_idx, orig_idx in enumerate(kept_indices):
        p = orig_parents[orig_idx]
        # 回溯祖先链直到找到保留节点或到根
        while p >= 0 and p not in kept_set:
            p = orig_parents[p]
        new_parents.append(kept_set[p] if p >= 0 and p in kept_set else -1)
    return new_parents


def _normalize_color(c) -> list[float]:
    """[r,g,b,a] 归一化数组；缺失返回 [0,0,0,0]。"""
    if c is None or not isinstance(c, (list, tuple)) or len(c) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(c[0]), float(c[1]), float(c[2]), float(c[3])]


def _encode_styles(styles: list[dict], N: int) -> dict[str, torch.Tensor]:
    """
    把 list[styles_dict] 转成训练需要的 dense tensor 字典。
    输出（与 N 对齐，padding 位置后续由 mask 屏蔽）：
        reg:        [N, len(STYLE_REG_KEYS)]  归一化后回归值
        reg_valid:  [N, len(STYLE_REG_KEYS)]  缺失项 0 mask
        color:      [N, len(STYLE_COLOR_KEYS), 4]  RGBA
        color_valid:[N, len(STYLE_COLOR_KEYS)]
        cls_<key>:  [N] long  0..len(choices)-1
    """
    out = {
        "reg": torch.zeros(N, len(STYLE_REG_KEYS), dtype=torch.float32),
        "reg_valid": torch.zeros(N, len(STYLE_REG_KEYS), dtype=torch.float32),
        "color": torch.zeros(N, len(STYLE_COLOR_KEYS), 4, dtype=torch.float32),
        "color_valid": torch.zeros(N, len(STYLE_COLOR_KEYS), dtype=torch.float32),
    }
    for k, choices in STYLE_CLS_KEYS.items():
        out[f"cls_{k}"] = torch.zeros(N, dtype=torch.long)

    M = min(len(styles), N)
    for i in range(M):
        s = styles[i] if styles[i] is not None else {}
        for j, k in enumerate(STYLE_REG_KEYS):
            v = s.get(k)
            if v is not None:
                norm = STYLE_REG_NORMS[k]
                # clamp 到 [0, 1]：极端值（如 border-radius:9999px 的 pill 按钮）会导致 MSE 爆炸
                out["reg"][i, j] = max(0.0, min(1.0, float(v) / norm))
                out["reg_valid"][i, j] = 1.0
        for j, k in enumerate(STYLE_COLOR_KEYS):
            c = s.get(k)
            normed = _normalize_color(c)
            out["color"][i, j] = torch.tensor(normed)
            # alpha > 0 视为有效
            if normed[3] > 0:
                out["color_valid"][i, j] = 1.0
        for k, choices in STYLE_CLS_KEYS.items():
            v = s.get(k, choices[0])
            try:
                idx = choices.index(v)
            except ValueError:
                idx = 0
            out[f"cls_{k}"][i] = idx
    return out


class WebpageDataset(Dataset):
    """
    每条样本以 padding 后的固定形状返回：
        image:           PIL.Image（编码器内部 processor）
        node_texts:      list[str]，长度 N（实际有效节点）
        node_boxes:      [max_nodes, 4]
        parents:         [max_nodes] long, -1=根, padding 位置 -1
        node_mask:       [max_nodes] float, 1=有效，0=padding
        patch_boxes:     [196, 4]
        alignment_labels:[max_nodes, 196]，padding 位置全 0
        styles:          dict (见 _encode_styles)，所有 tensor 第 0 维 = max_nodes
        num_nodes:       int 有效节点数
    """

    def __init__(
        self,
        data_dir: str | None = None,
        iou_threshold: float = 0.3,
        use_demo: bool = False,
        max_nodes: int = 96,
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
            paths = {
                "img_path":     os.path.join(d, "screenshot.png"),
                "nodes_path":   os.path.join(d, "nodes.pt"),
                "texts_path":   os.path.join(d, "node_texts.json"),
                "parents_path": os.path.join(d, "parents.pt"),
                "styles_path":  os.path.join(d, "styles.json"),
            }
            # 必需 4 项 + styles.json（render_pages.py 已统一生成）
            if all(os.path.exists(p) for p in paths.values()):
                samples.append({"type": "rendered", **paths})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_demo(self):
        image = preprocess_image("data/web_page_image.png")
        node_texts = parse_html_nodes(DEMO_HTML)
        node_boxes = DEMO_NODE_BOXES.clone()
        parents = DEMO_PARENTS.tolist()
        # demo 没有 computed style，填默认值
        styles = [{"backgroundColor":[0,0,0,0],"color":[0,0,0,1],"borderColor":[0,0,0,0],
                   "borderRadius":0,"borderWidth":0,"opacity":1.0,"fontSize":16,"fontWeight":400,
                   "textAlign":"start","display":"block"} for _ in node_texts]
        return image, node_texts, node_boxes, parents, styles

    def _load_real(self, sample: dict):
        image = preprocess_image(sample["img_path"])
        with open(sample["texts_path"], encoding="utf-8") as f:
            node_texts = json.load(f)
        node_boxes = torch.load(sample["nodes_path"], weights_only=True)
        parents = torch.load(sample["parents_path"], weights_only=True).tolist()
        with open(sample["styles_path"], encoding="utf-8") as f:
            styles = json.load(f)
        return image, node_texts, node_boxes, parents, styles

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if sample["type"] == "demo":
            image, node_texts, node_boxes, parents, styles = self._load_demo()
        else:
            image, node_texts, node_boxes, parents, styles = self._load_real(sample)

        # BFS 截断（保留连通子树）
        kept = _bfs_truncate(parents, self.max_nodes)
        if len(kept) != len(node_texts):
            node_texts = [node_texts[i] for i in kept]
            node_boxes = node_boxes[kept]
            styles = [styles[i] if i < len(styles) else None for i in kept]
            parents = _reindex_parents(parents, kept)

        N = len(node_texts)
        max_n = self.max_nodes

        # padding
        padded_boxes = torch.zeros(max_n, 4)
        padded_boxes[:N] = node_boxes[:N]

        padded_parents = torch.full((max_n,), -1, dtype=torch.long)
        for i, p in enumerate(parents[:N]):
            padded_parents[i] = p

        node_mask = torch.zeros(max_n, dtype=torch.float32)
        node_mask[:N] = 1.0

        # 对齐标签（padded 后 N×P）
        alignment_labels = torch.zeros(max_n, self.patch_boxes.shape[0])
        if N > 0:
            real_labels = build_alignment_labels(
                node_boxes[:N], self.patch_boxes, self.iou_threshold
            )
            alignment_labels[:N] = real_labels

        # 样式
        style_tensors = _encode_styles(styles, max_n)

        return {
            "image": image,
            "node_texts": node_texts,           # 长度 N（不 padding，编码器内部处理）
            "node_boxes": padded_boxes,         # [max_n, 4]
            "parents": padded_parents,          # [max_n]
            "node_mask": node_mask,             # [max_n]
            "patch_boxes": self.patch_boxes,    # [196, 4]
            "alignment_labels": alignment_labels,  # [max_n, 196]
            "styles": style_tensors,            # dict, 第 0 维 = max_n
            "num_nodes": N,
        }


def webpage_collate_fn(batch: list[dict]) -> dict:
    """
    支持 batch>1 的 collate_fn。
    所有 N 维 tensor 已经在 dataset 端 padding 到 max_nodes，可直接 stack。
    """
    images = [b["image"] for b in batch]
    node_texts = [b["node_texts"] for b in batch]  # list[list[str]]
    node_boxes = torch.stack([b["node_boxes"] for b in batch])
    parents = torch.stack([b["parents"] for b in batch])
    node_mask = torch.stack([b["node_mask"] for b in batch])
    patch_boxes = batch[0]["patch_boxes"]  # 共享
    alignment_labels = torch.stack([b["alignment_labels"] for b in batch])
    num_nodes = torch.tensor([b["num_nodes"] for b in batch], dtype=torch.long)

    # styles：字典内的每个 tensor 都按 batch stack
    style_keys = batch[0]["styles"].keys()
    styles = {k: torch.stack([b["styles"][k] for b in batch]) for k in style_keys}

    return {
        "image": images,                 # list[PIL.Image]，长度 B
        "node_texts": node_texts,        # list[list[str]]，长度 B
        "node_boxes": node_boxes,        # [B, max_n, 4]
        "parents": parents,              # [B, max_n]
        "node_mask": node_mask,          # [B, max_n]
        "patch_boxes": patch_boxes,      # [196, 4]
        "alignment_labels": alignment_labels,  # [B, max_n, 196]
        "styles": styles,                # dict, 每个值 [B, max_n, ...]
        "num_nodes": num_nodes,          # [B]
    }
