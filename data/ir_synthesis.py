"""红外合成流水线（关键创新 3：跨模态长尾蒸馏）

RGB → 灰度 → CLAHE → 直方图匹配红外分布 → Gamma 校正 → 噪声 → 合成红外

用途：
1. 训练时 p=0.3 概率合成红外样本，让模型对红外域鲁棒
2. 配合 MViT 跨模态教师做模态对抗蒸馏
"""
import cv2
import numpy as np
import torch
import random


# 预采集的红外分布直方图库（可替换为真实样本的直方图）
# 这里用模拟分布做占位
def _get_ir_reference_hist() -> np.ndarray:
    """获取红外参考直方图（模拟）"""
    # 真实部署时用以下代码加载真实红外样本直方图：
    #   ir_samples = [cv2.imread(f, 0) for f in ir_sample_files]
    #   hist = np.mean([cv2.calcHist([s], [0], None, [256], [0, 256]) for s in ir_samples], axis=0)
    # 占位：用高斯分布模拟红外灰度分布（中心偏暗）
    hist = np.zeros(256)
    for i in range(256):
        hist[i] = np.exp(-(i - 90) ** 2 / (2 * 50 ** 2))
    hist = hist / hist.sum()
    return hist


_IR_REF_HIST = _get_ir_reference_hist()


def clahe_enhance(gray: np.ndarray) -> np.ndarray:
    """CLAHE 对比度受限自适应直方图均衡"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def histogram_match(src: np.ndarray, ref_hist: np.ndarray) -> np.ndarray:
    """直方图匹配：把 src 的分布对齐到 ref_hist"""
    src_hist, _ = np.histogram(src.flatten(), bins=256, range=[0, 256])
    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()
    # 查找映射表
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(ref_cdf - src_cdf[i])
        mapping[i] = np.argmin(diff)
    return cv2.LUT(src, mapping)


def add_ir_noise(img: np.ndarray) -> np.ndarray:
    """添加红外传感器噪声（高斯 + 条带）"""
    # 高斯噪声
    noise_g = np.random.normal(0, 8, img.shape)
    img = np.clip(img.astype(np.float32) + noise_g, 0, 255)
    # 条带噪声（模拟红外传感器扫描线）
    if random.random() < 0.5:
        band_idx = random.randint(0, img.shape[0] - 1)
        band_val = np.random.uniform(-15, 15)
        img[band_idx, :] += band_val
    return img.astype(np.uint8)


def rgb_to_ir(img: np.ndarray, ref_hist: np.ndarray = None) -> np.ndarray:
    """RGB → 合成红外

    流程:
    1. 灰度化
    2. CLAHE 均衡
    3. 直方图匹配红外分布
    4. Gamma 校正（模拟夜间微光）
    5. 红外噪声
    """
    if ref_hist is None:
        ref_hist = _IR_REF_HIST
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 1. CLAHE
    gray = clahe_enhance(gray)
    # 2. 直方图匹配
    gray = histogram_match(gray, ref_hist)
    # 3. Gamma 校正
    gamma = random.uniform(0.6, 0.9)
    gray = np.clip(255 * (gray / 255) ** gamma, 0, 255).astype(np.uint8)
    # 4. 噪声
    gray = add_ir_noise(gray)
    return gray


def rgb_tensor_to_ir(x: torch.Tensor, prob: float = 0.3) -> torch.Tensor:
    """对 (B, T, 3, H, W) 张量按概率做红外合成

    每个样本独立判定，模拟真实场景中部分帧来自红外
    """
    B = x.size(0)
    out = x.clone()
    for b in range(B):
        if random.random() < prob:
            # 取该样本首帧作为直方图参考（保证序列内分布一致）
            # 转为 numpy 处理
            arr = x[b].cpu().numpy().transpose(1, 2, 3, 0, 4).squeeze(-1)
            T, C, H, W = arr.shape[:4] if arr.ndim == 5 else (x.size(1), 3, *x.shape[3:])
            # 简化：对每帧做处理
            imgs_np = x[b].permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
            ir_frames = []
            for t in range(imgs_np.shape[0]):
                ir = rgb_to_ir(imgs_np[t])
                # 扩展回 3 通道（红外为灰度，复制 3 通道）
                ir_3 = np.stack([ir, ir, ir], axis=-1)
                ir_frames.append(ir_3)
            ir_tensor = torch.from_numpy(np.stack(ir_frames))  # (T, H, W, 3)
            ir_tensor = ir_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
            out[b] = ir_tensor.to(x.dtype).to(x.device)
    return out


def get_ir_reference_hist() -> np.ndarray:
    """供外部调用获取参考直方图"""
    return _IR_REF_HIST


if __name__ == "__main__":
    # 简单测试：读取一张 RGB 图片，输出红外合成结果
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        ir = rgb_to_ir(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite("synthetic_ir.png", ir)
        print("Saved: synthetic_ir.png")
