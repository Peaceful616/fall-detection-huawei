"""bafd6 数据集适配器：VOC XML → 我们的训练格式

bafd6 数据集：
- 单帧图片 + VOC XML 标注
- 1440 张图片，1505 个 fall bbox
- 类别：fall

适配策略：
1. 把单帧图片"伪视频化"（重复 T 次或 T=1）
2. 把 VOC bbox 转为 YOLO 格式：(class, x_center, y_center, w, h) 归一化
3. 生成 annotations.json：
   [
     {"image": "fall_0.jpg", "boxes": [[cls, x, y, w, h], ...], "label": "fall"},
     ...
   ]
4. 划分 train/val (8:2)
"""
import os
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc_xml(xml_path: str) -> dict:
    """解析单个 VOC XML 文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_elem = root.find("size")
    if size_elem is None:
        return {"width": 0, "height": 0, "boxes": []}
    width_elem = size_elem.find("width")
    height_elem = size_elem.find("height")
    width = int(width_elem.text) if width_elem is not None and width_elem.text else 0
    height = int(height_elem.text) if height_elem is not None and height_elem.text else 0

    boxes = []
    if width == 0 or height == 0:
        return {"width": width, "height": height, "boxes": boxes}
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        try:
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
        except (TypeError, ValueError):
            continue
        # 转 YOLO 格式：归一化的 (cx, cy, w, h)
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        boxes.append({
            "class": name,
            "cx": round(cx, 6),
            "cy": round(cy, 6),
            "w": round(w, 6),
            "h": round(h, 6),
        })
    return {"width": width, "height": height, "boxes": boxes}


def adapt_bafd6(src_dir: str, out_dir: str, train_ratio: float = 0.8):
    """适配 bafd6 数据集

    src_dir: bafd6 目录（含 images/ 和 Annotations/）
    out_dir: 输出目录（生成 annotations.json + 软链 images/）
    """
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    images_dir = src / "images"
    annot_dir = src / "Annotations"
    if not images_dir.is_dir() or not annot_dir.is_dir():
        print(f"[FAIL] images/ or Annotations/ not found in {src_dir}")
        return False

    # 输出 images 目录（建立软链或复制）
    out_images = out / "images"
    if not out_images.exists():
        # Windows 软链需要权限，直接用目录复制
        import shutil
        shutil.copytree(images_dir, out_images)
        print(f"[OK] Copied images to: {out_images}")

    # 解析所有 XML
    samples = []
    xml_files = sorted(annot_dir.glob("*.xml"))
    print(f"[INFO] Parsing {len(xml_files)} XML files...")

    class_map = {}  # 动态收集类别
    for xml_path in xml_files:
        info = parse_voc_xml(str(xml_path))
        if not info["boxes"]:
            continue  # 跳过无目标的样本
        img_name = xml_path.stem + ".jpg"
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        # 收集类别
        for b in info["boxes"]:
            c = b["class"]
            if c not in class_map:
                class_map[c] = len(class_map)
        samples.append({
            "image": img_name,
            "width": info["width"],
            "height": info["height"],
            "boxes": [
                {
                    "class_id": class_map[b["class"]],
                    "class_name": b["class"],
                    "cx": b["cx"], "cy": b["cy"],
                    "w": b["w"], "h": b["h"],
                }
                for b in info["boxes"]
            ],
            "label": "fall" if any(b["class"] == "fall" for b in info["boxes"]) else "adl",
        })

    print(f"[OK] Parsed {len(samples)} samples with boxes")
    print(f"[OK] Classes: {class_map}")

    # 划分 train/val
    random.seed(42)
    random.shuffle(samples)
    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    # 写 annotations
    out_train = out / "annotations_train.json"
    out_val = out / "annotations_val.json"
    out_classes = out / "classes.json"

    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    with open(out_classes, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    print(f"\n[Adapt Done]")
    print(f"  Train: {len(train_samples)} samples -> {out_train}")
    print(f"  Val:   {len(val_samples)} samples -> {out_val}")
    print(f"  Classes: {class_map} -> {out_classes}")
    return True


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "./data/bafd6"
    out = sys.argv[2] if len(sys.argv) > 2 else "./data/bafd6_adapted"
    adapt_bafd6(src, out)
