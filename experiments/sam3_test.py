"""
SAM3 test — segments an image using text prompts for material types.
Usage: python experiments/sam3_test.py --image <path_to_image>
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

MATERIAL_PROMPTS = ["grass", "gravel", "concrete", "mud", "water", "sand", "asphalt", "dirt", "rock", "vegetation"]

WEIGHTS_DIR = os.path.expanduser("~/joana/sam3/weights")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
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
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    inference_state = processor.set_image(image)

    os.makedirs("outputs", exist_ok=True)

    for prompt in args.prompts:
        processor.reset_all_prompts(inference_state)
        state = processor.set_text_prompt(state=inference_state, prompt=prompt)
        n = len(state["masks"]) if state["masks"] is not None else 0
        scores = state["scores"].tolist() if state["scores"] is not None else []
        print(f"  '{prompt}': {n} detections, scores={[round(s,2) for s in scores]}")

    # final run with all prompts for visualization
    processor.reset_all_prompts(inference_state)
    for prompt in args.prompts:
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    plot_results(image, inference_state)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
