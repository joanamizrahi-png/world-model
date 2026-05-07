"""
Evaluate SAM3 material segmentation against RUGD ground truth.

RUGD structure:
  RUGD_frames-with-annotations/
    creek/
      creek_00000.png          <- RGB image
      creek_00000_annotation.png  <- annotation (palette PNG, value = class index)
    ...

Usage:
  python experiments/evaluate_sam3.py --rugd_dir ~/joana/rugd/RUGD_frames-with-annotations --n_images 50
"""

import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# RUGD class index → name and RGB color
RUGD_ID_TO_NAME = {
    0: "void", 1: "dirt", 2: "sand", 3: "grass", 4: "tree", 5: "pole",
    6: "water", 7: "sky", 8: "vehicle", 9: "container", 10: "asphalt",
    11: "gravel", 12: "building", 13: "mulch", 14: "rock-bed", 15: "log",
    16: "bicycle", 17: "person", 18: "fence", 19: "bush", 20: "sign",
    21: "rock", 22: "bridge", 23: "concrete", 24: "picnic-table"
}
RUGD_NAME_TO_ID = {v: k for k, v in RUGD_ID_TO_NAME.items()}

# RGB color → class ID lookup
RUGD_COLOR_TO_ID = {
    (0, 0, 0): 0, (108, 64, 20): 1, (255, 229, 204): 2, (0, 102, 0): 3,
    (0, 255, 0): 4, (0, 153, 153): 5, (0, 128, 255): 6, (0, 0, 255): 7,
    (255, 255, 0): 8, (255, 0, 127): 9, (64, 64, 64): 10, (255, 128, 0): 11,
    (255, 0, 0): 12, (153, 76, 0): 13, (102, 102, 0): 14, (102, 0, 0): 15,
    (0, 255, 128): 16, (204, 153, 255): 17, (102, 0, 204): 18, (255, 153, 204): 19,
    (0, 102, 102): 20, (153, 204, 255): 21, (102, 255, 255): 22, (101, 101, 11): 23,
    (114, 85, 47): 24
}


def rgb_annotation_to_class_map(ann_rgb):
    """Convert RGB annotation image to per-pixel class ID map."""
    h, w = ann_rgb.shape[:2]
    class_map = np.zeros((h, w), dtype=np.int32)
    for color, cid in RUGD_COLOR_TO_ID.items():
        mask = np.all(ann_rgb == np.array(color), axis=2)
        class_map[mask] = cid
    return class_map

WEIGHTS_DIR = os.path.expanduser("~/joana/sam3/weights")


def get_image_annotation_pairs(rugd_dir, ann_dir, n_images, seed=42):
    import random
    pairs = []
    for ann_path in glob.glob(f"{ann_dir}/**/*.png", recursive=True):
        rel = os.path.relpath(ann_path, ann_dir)
        img_path = os.path.join(rugd_dir, rel)
        if os.path.exists(img_path):
            pairs.append((img_path, ann_path))
    random.seed(seed)
    random.shuffle(pairs)
    print(f"Total images available: {len(pairs)}")
    return pairs[:n_images]


def masks_to_pred_map(results, image_shape, rugd_id_to_name):
    """Convert SAM3 results to a per-pixel predicted class map."""
    h, w = image_shape[:2]
    pred_map = np.zeros((h, w), dtype=np.int32)  # 0 = void/unlabeled

    name_to_id = {v: k for k, v in rugd_id_to_name.items()}

    for prompt, data in results.items():
        class_id = name_to_id.get(prompt, 0)
        for mask, score in zip(data["masks"], data["scores"]):
            if mask.ndim == 3:
                mask = mask[0]
            pred_map[mask.astype(bool)] = class_id

    return pred_map


def compute_iou_per_class(pred_map, gt_map, class_ids):
    ious = {}
    for cid in class_ids:
        pred_mask = pred_map == cid
        gt_mask = gt_map == cid
        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        if union == 0:
            continue
        ious[cid] = intersection / union
    return ious


def run_sam3_on_image(model, processor, image, threshold):
    from sam3_test import RUGD_CLASSES
    inference_state = processor.set_image(image)
    results = {}
    for prompt in RUGD_CLASSES:
        processor.reset_all_prompts(inference_state)
        state = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = state["masks"]
        scores = state["scores"]
        if masks is not None and len(masks) > 0:
            results[prompt] = {
                "masks": [m.cpu().numpy() if isinstance(m, torch.Tensor) else m for m in masks],
                "scores": scores.tolist() if isinstance(scores, torch.Tensor) else scores
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rugd_dir", required=True, help="Path to RUGD_frames-with-annotations")
    parser.add_argument("--ann_dir", required=True, help="Path to RUGD_annotations")
    parser.add_argument("--n_images", type=int, default=50)
    parser.add_argument("--output", default="outputs/evaluation_results.txt")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print("Loading SAM3...")
    model = build_sam3_image_model(load_from_HF=True, checkpoint_path=f"{WEIGHTS_DIR}/sam3.pt")
    processor = Sam3Processor(model, confidence_threshold=args.threshold)

    pairs = get_image_annotation_pairs(args.rugd_dir, args.ann_dir, args.n_images, seed=42)
    print(f"Found {len(pairs)} image-annotation pairs, evaluating {len(pairs)}...")

    all_ious = defaultdict(list)
    per_image_log = []

    for img_path, ann_path in pairs:
        image = Image.open(img_path).convert("RGB")
        gt_map = rgb_annotation_to_class_map(np.array(Image.open(ann_path).convert("RGB")))

        results = run_sam3_on_image(model, processor, image, args.threshold)
        pred_map = masks_to_pred_map(results, gt_map.shape, RUGD_ID_TO_NAME)

        ious = compute_iou_per_class(pred_map, gt_map, list(RUGD_ID_TO_NAME.keys()))
        for cid, iou in ious.items():
            all_ious[cid].append(iou)

        img_name = os.path.basename(img_path)
        img_miou = np.mean(list(ious.values())) if ious else 0.0
        per_image_log.append(f"{img_name}: mIoU={img_miou:.3f} | " + " ".join(
            f"{RUGD_ID_TO_NAME[cid]}={iou:.2f}" for cid, iou in sorted(ious.items())
        ))
        print(f"  {img_name}: mIoU={img_miou:.3f}")

    lines = ["\n=== Results ==="]
    mean_ious = {}
    for cid, iou_list in sorted(all_ious.items()):
        name = RUGD_ID_TO_NAME[cid]
        miou = np.mean(iou_list)
        mean_ious[cid] = miou
        lines.append(f"  {name:20s}: mIoU = {miou:.3f} (n={len(iou_list)})")

    overall_miou = np.mean(list(mean_ious.values()))
    lines.append(f"\nOverall mIoU: {overall_miou:.3f}")

    output = "\n".join(lines)
    print(output)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    log_path = args.output
    with open(log_path, "w") as f:
        f.write(f"n_images={args.n_images}, threshold={args.threshold}\n")
        f.write("\n=== Per-image results ===\n")
        f.write("\n".join(per_image_log))
        f.write(output)
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
