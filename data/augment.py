"""长尾数据增强（在线）

针对赛题三大痛点之"长尾场景"：
- 微光：HSV-V 扰动、Gamma
- 夜间红外：调用 ir_synthesis
- 遮挡：Random Erasing
- 视角：Random Perspective / Rotate
- 运动模糊：MotionBlur
- 压缩噪声：JPEG 压缩
"""
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .ir_synthesis import rgb_to_ir


def aug_low_light(img: torch.Tensor, v_range=(0.3, 0.7), gamma_range=(1.5, 2.5)) -> torch.Tensor:
    """微光增强：HSV-V 扰动 + Gamma
    img: (3, H, W) tensor in [0, 1]
    """
    # V 通道乘以随机因子
    factor = random.uniform(*v_range)
    img = img * factor
    # Gamma 校正
    gamma = random.uniform(*gamma_range)
    img = img.clamp(0, 1).pow(gamma)
    return img


def aug_occlusion(img: torch.Tensor, area_ratio=(0.1, 0.3)) -> torch.Tensor:
    """Random Erasing 遮挡
    img: (3, H, W) tensor
    """
    _, H, W = img.shape
    area = H * W * random.uniform(*area_ratio)
    ratio = random.uniform(0.5, 2.0)
    h = int(np.sqrt(area * ratio))
    w = int(np.sqrt(area / ratio))
    h, w = min(h, H), min(w, W)
    x = random.randint(0, W - w)
    y = random.randint(0, H - h)
    val = torch.rand(3, h, w) * img.max()
    img[:, y:y + h, x:x + w] = val
    return img


def aug_perspective(img: torch.Tensor, distortion=0.2) -> torch.Tensor:
    """Random Perspective
    img: (3, H, W) tensor
    """
    _, H, W = img.shape
    # 四角随机偏移
    pts_src = torch.tensor([[0, 0], [W, 0], [W, H], [0, H]], dtype=torch.float32)
    pts_dst = pts_src + torch.randn(4, 2) * distortion * max(H, W)
    # 计算 perspective 变换矩阵
    M = cv2.getPerspectiveTransform(
        pts_src.numpy().astype(np.float32),
        pts_dst.numpy().astype(np.float32)
    )
    img_np = img.permute(1, 2, 0).numpy()
    img_np = cv2.warpPerspective(img_np, M, (W, H))
    return torch.from_numpy(img_np).permute(2, 0, 1).float()


def aug_rotate(img: torch.Tensor, max_angle=15) -> torch.Tensor:
    """Random Rotate ±15°"""
    angle = random.uniform(-max_angle, max_angle)
    return TF.rotate(img.unsqueeze(0), angle).squeeze(0)


def aug_motion_blur(img: torch.Tensor, kernel_size=7) -> torch.Tensor:
    """Motion Blur 运动模糊"""
    angle = random.uniform(0, 180)
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    img_np = img.permute(1, 2, 0).numpy()
    img_np = cv2.filter2D(img_np, -1, kernel)
    return torch.from_numpy(img_np).permute(2, 0, 1).float()


def aug_jpeg_noise(img: torch.Tensor, q_range=(30, 70)) -> torch.Tensor:
    """JPEG 压缩噪声（模拟监控链路）"""
    q = random.randint(*q_range)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, encoded = cv2.imencode('.jpg', img_np, encode_param)
    img_np = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0


def aug_to_ir(img: torch.Tensor) -> torch.Tensor:
    """调用红外合成
    img: (3, H, W) tensor in [0, 1]
    """
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ir = rgb_to_ir(img_np)
    ir_3 = np.stack([ir, ir, ir], axis=-1)
    return torch.from_numpy(ir_3).permute(2, 0, 1).float() / 255.0


class LongTailAugment:
    """长尾数据增强组合（在线）

    输入: (B, T, 3, H, W) tensor in [0, 1]
    输出: 同形状
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """对 (B, T, 3, H, W) 的每个样本独立增强"""
        B = x.size(0)
        out = x.clone()
        for b in range(B):
            # 红外合成（最高优先级，因为后续增强应基于红外或 RGB 一致处理）
            if random.random() < self.cfg.ir_synthesis_prob:
                for t in range(out[b].size(0)):
                    out[b, t] = aug_to_ir(out[b, t])
            # 微光
            if random.random() < self.cfg.aug_low_light_prob:
                for t in range(out[b].size(0)):
                    out[b, t] = aug_low_light(out[b, t])
            # 遮挡
            if random.random() < self.cfg.aug_occlusion_prob:
                for t in range(out[b].size(0)):
                    out[b, t] = aug_occlusion(out[b, t])
            # 视角
            if random.random() < self.cfg.aug_perspective_prob:
                for t in range(out[b].size(0)):
                    out[b, t] = aug_perspective(out[b, t])
            # 运动模糊
            if random.random() < self.cfg.aug_motion_blur_prob:
                for t in range(out[b].size(0)):
                    out[b, t] = aug_motion_blur(out[b, t])
        return out
