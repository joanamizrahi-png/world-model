"""
SAM3 test — segments an image using RUGD class prompts.
Usage: python experiments/sam3_test.py --image <path_to_image> [--output <path>]
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

RUGD_CLASSES = [
    "tree", "sky", "grass", "mulch", "gravel", "bush", "pole", "log",
    "building", "vehicle", "container", "fence", "asphalt", "water", "rock",
    "sign", "concrete", "rock-bed", "picnic-table", "dirt", "bridge", "sand",
    "person", "bicycle"
]

WEIGHTS_DIR = os.path.expanduser("~/joana/sam3/weights")
COLORS = plt.cm.tab20(np.linspace(0, 1, len(RUGD_CLASSES)))


def run_sam3(image, prompts, threshold=0.3):
    """Run SAM3 on an image with given text prompts. Returns dict {prompt: masks}."""
    model = build_sam3_image_model(load_from_HF=True, checkpoint_path=f"{WEIGHTS_DIR}/sam3.pt")
    processor = Sam3Processor(model, confidence_threshold=threshold)
    inference_state = processor.set_image(image)

    results = {}
    for prompt in prompts:
        processor.reset_all_prompts(inference_state)
        state = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = state["masks"]
        scores = state["scores"]
        if masks is not None and len(masks) > 0:
            results[prompt] = {
                "masks": [m.cpu().numpy() if isinstance(m, torch.Tensor) else m for m in masks],
                "scores": scores.tolist() if isinstance(scores, torch.Tensor) else scores
            }
            print(f"  '{prompt}': {len(masks)} detection(s), scores={[round(s,2) for s in results[prompt]['scores']]}")
        else:
            print(f"  '{prompt}': no detections")
    return results


def save_visualization(image_np, results, prompts, output_path):
    """Save color-coded overlay with legend."""
    overlay = image_np.copy().astype(float)
    legend_patches = []

    for i, prompt in enumerate(prompts):
        if prompt not in results:
            continue
        color = COLORS[i % len(COLORS)][:3]
        for mask in results[prompt]["masks"]:
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = overlay[mask_bool] * 0.4 + np.array(color) * 255 * 0.6
        legend_patches.append(mpatches.Patch(color=color, label=prompt))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title("SAM3 — RUGD class segmentation")
    axes[1].axis("off")
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/sam3_result.png")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print("Loading SAM3...")
    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    results = run_sam3(image, RUGD_CLASSES, threshold=args.threshold)
    save_visualization(image_np, results, RUGD_CLASSES, args.output)


if __name__ == "__main__":
    main()
