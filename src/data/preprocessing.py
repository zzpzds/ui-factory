import torch
from PIL import Image
from bs4 import BeautifulSoup

SKIP_TAGS = {"html", "head", "body", "script", "style", "meta", "link"}
PATCH_SIZE = 16
GRID_SIZE = 14  # 14x14 = 196 patches


def parse_html_nodes(html: str) -> list[str]:
    """
    解析 HTML，返回所有节点的序列化文本列表。
    格式：<tag attr1="v1" attr2="v2">inner_text</tag>
      - inner_text 只取直接文本子节点（不递归后代），超过 100 字符截断
      - 属性值里的 " 用 &quot; 实体转义，与 render_pages.py 的 JS 侧保持一致
    """
    soup = BeautifulSoup(html, "html.parser")
    nodes = []
    for tag in soup.find_all(True):
        if tag.name in SKIP_TAGS:
            continue
        text = f"<{tag.name}"
        for attr, val in tag.attrs.items():
            if isinstance(val, list):
                val = " ".join(val)
            val = str(val).replace('"', "&quot;")
            text += f' {attr}="{val}"'
        # 只取直接文本子节点
        inner = "".join(
            str(c) for c in tag.children if isinstance(c, str)
        ).strip()
        if len(inner) > 100:
            inner = inner[:100]
        text += f">{inner}</{tag.name}>"
        nodes.append(text)
    return nodes


def preprocess_image(image_path: str) -> Image.Image:
    """
    加载图像并转换为 RGB，ViTImageProcessor 会在编码器内部做 resize/normalize。
    """
    return Image.open(image_path).convert("RGB")


def get_patch_boxes() -> torch.Tensor:
    """
    返回固定的 196 个 ViT patch 的边界框，格式 [196, 4]，坐标为 224x224 图像上的像素位置。
    第 (row, col) 个 patch 的坐标：[col*16, row*16, (col+1)*16, (row+1)*16]
    """
    boxes = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1 = col * PATCH_SIZE
            y1 = row * PATCH_SIZE
            x2 = x1 + PATCH_SIZE
            y2 = y1 + PATCH_SIZE
            boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)
