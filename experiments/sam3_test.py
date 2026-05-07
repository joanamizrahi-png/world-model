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

SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__), "..")
BPE_PATH = f"{SAM3_ROOT}/assets/bpe_simple_vocab_16e6.txt.gz"

MATERIAL_PROMPTS = ["grass", "gravel", "concrete", "mud", "water", "sand", "asphalt", "dirt", "rock", "vegetation"]


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
    model = build_sam3_image_model(bpe_path=BPE_PATH)

    image = Image.open(args.image).convert("RGB")
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    inference_state = processor.set_image(image)

    os.makedirs("outputs", exist_ok=True)

    for prompt in args.prompts:
        print(f"Running prompt: '{prompt}'")
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    fig = plot_results(image, inference_state)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
