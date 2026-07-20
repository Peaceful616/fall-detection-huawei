"""端侧导出脚本：ONNX → RKNN（瑞芯微）/ WK（海思）

实际部署需安装对应工具链：
- RK3588: pip install rknn-toolkit2
- 海思: nnie 工具（C++ SDK）

无工具链时该脚本也能跑通 ONNX 简化步骤，方便演示。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from configs.default import cfg


def simplify_onnx(onnx_path: str, sim_path: str):
    """用 onnx-simplifier 简化模型"""
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(onnx_path)
        model_sim, check = simplify(model)
        assert check, "Simplified ONNX check failed"
        onnx.save(model_sim, sim_path)
        print(f"[OK] Simplified ONNX: {sim_path}")
        return sim_path
    except ImportError:
        print("[WARN] onnx-simplifier not installed, skip simplify")
        return onnx_path


def export_rknn_rk3588(onnx_path: str, rknn_path: str, input_size, seq_len):
    """瑞芯微 RK3588 导出"""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[WARN] rknn-toolkit2 not installed, skip real RKNN export")
        print("       Install: pip install rknn-toolkit2")
        print(f"       ONNX ready at: {onnx_path}")
        return False
    rknn = RKNN()
    # 配置
    rknn.config(
        mean_values=[[0.485 * 255, 0.456 * 255, 0.406 * 255]],
        std_values=[[0.229 * 255, 0.224 * 255, 0.225 * 255]],
        target_platform='rk3588',
        quantized_dtype='w8a8',
        quantized_method='channel',
    )
    # 加载 ONNX
    ret = rknn.load_onnx(model=onnx_path, inputs=['input'],
                        input_size_list=[[1, seq_len, 3, *input_size]])
    if ret != 0:
        print(f"[FAIL] load_onnx: {ret}")
        return False
    # 构建 INT8 模型
    ret = rknn.build(do_quantization=True, dataset='./data/calib_images.txt')
    if ret != 0:
        print(f"[FAIL] build: {ret}")
        return False
    # 导出
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"[FAIL] export: {ret}")
        return False
    print(f"[OK] RKNN exported: {rknn_path}")
    rknn.release()
    return True


def export_hisi(onnx_path: str, wk_path: str):
    """海思 NNIE 导出（占位，实际需用 SDK 的 NNIE Mapper 工具）"""
    print(f"[INFO] Hisi NNIE export need SDK tool: nnie_mapper")
    print(f"       ONNX: {onnx_path}")
    print(f"       Use C++ SDK to convert to .wk")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--platform", choices=["rk3588", "hisi"], default="rk3588")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    onnx_path = args.onnx
    if not os.path.exists(onnx_path):
        print(f"[FAIL] ONNX not found: {onnx_path}")
        return

    # 1. 简化 ONNX
    sim_path = onnx_path.replace(".onnx", "_sim.onnx")
    sim_path = simplify_onnx(onnx_path, sim_path)

    # 2. 平台导出
    if args.output is None:
        ext = ".rknn" if args.platform == "rk3588" else ".wk"
        args.output = sim_path.replace(".onnx", ext)

    if args.platform == "rk3588":
        export_rknn_rk3588(sim_path, args.output, cfg.input_size, cfg.seq_len)
    else:
        export_hisi(sim_path, args.output)


if __name__ == "__main__":
    main()
