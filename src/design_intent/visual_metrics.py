"""重渲染图像的视觉约束指标。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def image_mse(predicted: Image.Image, reference: Image.Image) -> float:
    pred = _rgb_array(predicted)
    target = _rgb_array(reference)
    if pred.shape != target.shape:
        raise ValueError(f"图像尺寸不一致：{pred.shape} vs {target.shape}")
    return float(np.mean((pred - target) ** 2))


def mean_rgb_distance(predicted: Image.Image, reference: Image.Image) -> float:
    pred = _rgb_array(predicted)
    target = _rgb_array(reference)
    if pred.shape != target.shape:
        raise ValueError(f"图像尺寸不一致：{pred.shape} vs {target.shape}")
    return float(np.sqrt(np.sum((pred - target) ** 2, axis=-1)).mean())


def structural_similarity(predicted: Image.Image, reference: Image.Image) -> float:
    """使用 11x11 Gaussian window 的标准 SSIM 灰度实现。"""
    pred = np.asarray(predicted.convert("L"), dtype=np.float32)
    target = np.asarray(reference.convert("L"), dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"图像尺寸不一致：{pred.shape} vs {target.shape}")

    coordinates = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian).reshape(1, 1, 11, 11)

    def gaussian_blur(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array).reshape(1, 1, *array.shape)
        padded = F.pad(tensor, (5, 5, 5, 5), mode="reflect")
        return F.conv2d(padded, kernel).squeeze()

    pred_tensor = torch.from_numpy(pred)
    target_tensor = torch.from_numpy(target)
    constant_1 = (0.01 * 255) ** 2
    constant_2 = (0.03 * 255) ** 2
    mean_pred = gaussian_blur(pred)
    mean_target = gaussian_blur(target)
    mean_pred_sq = mean_pred * mean_pred
    mean_target_sq = mean_target * mean_target
    mean_product = mean_pred * mean_target
    variance_pred = gaussian_blur(pred * pred) - mean_pred_sq
    variance_target = gaussian_blur(target * target) - mean_target_sq
    covariance = gaussian_blur(pred * target) - mean_product
    numerator = (2 * mean_product + constant_1) * (
        2 * covariance + constant_2
    )
    denominator = (mean_pred_sq + mean_target_sq + constant_1) * (
        variance_pred + variance_target + constant_2
    )
    score = numerator / denominator.clamp(min=1e-12)
    return float(score.mean().item())


def compute_visual_metrics(
    predicted: Image.Image,
    reference: Image.Image,
) -> dict[str, float]:
    return {
        "mse": image_mse(predicted, reference),
        "ssim": structural_similarity(predicted, reference),
        "mean_rgb_distance": mean_rgb_distance(predicted, reference),
    }
