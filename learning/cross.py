import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

B = 2
N = 8
P = 196
D = 768

visual_features = torch.randn(B, P, D)
code_features = torch.randn(B, N, D)

cross_attn = nn.MultiheadAttention(embed_dim=D, num_heads=8, batch_first=True)

attn_output, attn_weights = cross_attn(
    query=code_features,
    key=visual_features,
    value=visual_features,
)

def compute_iou(box_a, box_b):
    """
    box_a, box_b: [x1, y1, x2, y2] 格式的边界框
    返回：IoU 值（0~1之间的浮点数）
    """
    left_bottom_x = max(box_a[0], box_b[0])
    left_bottom_y = max(box_a[1], box_b[1])
    right_top_x = min(box_a[2], box_b[2])
    right_top_y = min(box_a[3], box_b[3])

    duplicate = 0
    # 判断交集是否合法 计算交集面积
    if (right_top_x > left_bottom_x) & (right_top_y > left_bottom_y):
        duplicate = (right_top_x - left_bottom_x) * (right_top_y - left_bottom_y)

    # 并集面积
    # sum = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - duplicate
    # 优化交并比计算，分母改为a,b中较小的面积
    compare = min((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    IoU = duplicate / compare
    return IoU

patch_boxes = []
for row in range(14):
    for col in range(14):
        x1, y1 = col * 16, row * 16
        x2, y2 = (col + 1) * 16, (row + 1) * 16
        patch_boxes.append([x1, y1, x2, y2])
patch_boxes = torch.tensor(patch_boxes, dtype=torch.float32)

node_boxes = torch.tensor([
    [0, 0, 224, 30],
    [0, 0, 60, 30],
    [0, 30, 224, 224],
    [10, 40, 150, 130],
    [10, 140, 110, 170]
], dtype=torch.float32)

IOU_THRESHOLD = 0.8
M = node_boxes.shape[0]
P = patch_boxes.shape[0]
alignment_labels = torch.zeros(M, P)

for n in range(M):
    for p in range(P):
        IoU = compute_iou(node_boxes[n], patch_boxes[p])
        if IoU >= IOU_THRESHOLD:
            alignment_labels[n][p] = 1
        else:
            alignment_labels[n][p] = 0

print(f'对其标签矩阵形状：{alignment_labels.shape}')
print(f'a 节点对齐的patch数量：{alignment_labels[1].sum().int()}')
print(f'button 节点对齐的patch数量：{alignment_labels[4].sum().int()}')

node_names = ['nav', 'a', 'main', 'img', 'button']
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, name in enumerate(node_names):
    label_map = alignment_labels[i].reshape(14, 14).numpy()
    axes[0, i].imshow(label_map, cmap='Blues', vmin=0, vmax=1)
    axes[0, i].set_title(f'<{name}> 对齐区域')
    axes[0, i].axis('off')

    axes[1, i].set_xlim(0, 224)
    axes[1, i].set_ylim(224, 0)
    axes[1, i].set_aspect('equal')
    box = node_boxes[i].numpy()
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor='red', facecolor='none'
    )
    axes[1, i].add_patch(rect)
    axes[1, i].set_title(f'<{name}> 渲染框')

# plt.tight_layout()
# plt.show()


# part6
class CrossModalAlignment(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.loss_fn = nn.BCELoss()
    def forward(self, visual_features, code_features, alignment_labels=None):
        fused_features, attn_weights = self.cross_attn(
            query=code_features,
            key=visual_features,
            value=visual_features,
        )
        loss = self.loss_fn(attn_weights, alignment_labels)
        return fused_features, attn_weights, loss

labels = torch.randint(0, 2, (B, N, P)).float()
model = CrossModalAlignment()
fused_features, attn_weights, loss = model.forward(visual_features, code_features, labels)
print(f'融合特征：{fused_features.shape}')
print(f'注意力权重：{attn_weights.shape}')
print(f'对齐损失：{loss.item():.4f}')
