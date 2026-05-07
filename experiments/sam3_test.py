"""
SAM3 test — segments an image using text prompts for material types.
Usage: python experiments/sam3_test.py --image <path_to_image>
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

MATERIAL_PROMPTS = ["grass", "gravel", "concrete", "mud", "water", "sand", "asphalt", "dirt", "rock", "vegetation"]
WEIGHTS_DIR = os.path.expanduser("~/joana/sam3/weights")

COLORS = plt.cm.tab10(np.linspace(0, 1, len(MATERIAL_PROMPTS)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompts", nargs="+", default=MATERIAL_PROMPTS)
    parser.add_argument("--output", default="outputs/sam3_result.png")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print("Loading SAM3...")
    model = build_sam3_image_model(load_from_HF=True, checkpoint_path=f"{WEIGHTS_DIR}/sam3.pt")

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    inference_state = processor.set_image(image)

    overlay = image_np.copy().astype(float)
    legend_patches = []

    for i, prompt in enumerate(args.prompts):
        processor.reset_all_prompts(inference_state)
        state = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = state["masks"]
        scores = state["scores"]

        if masks is None or len(masks) == 0:
            print(f"  '{prompt}': no detections")
            continue

        color = COLORS[i % len(COLORS)][:3]
        n = len(masks)
        print(f"  '{prompt}': {n} detection(s), scores={[round(s,2) for s in scores.tolist()]}")

        for mask in masks:
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = overlay[mask_bool] * 0.4 + np.array(color) * 255 * 0.6

        legend_patches.append(mpatches.Patch(color=color, label=prompt))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title("SAM3 — material segmentation")
    axes[1].axis("off")
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "outputs", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
