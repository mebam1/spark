#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import base64
import argparse
import requests
from pathlib import Path

# ========= API KEY =========
# Read from the .env file at the repo root, or from the GEMINI_API_KEY env var.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ========= Model =========
GEMINI_MODEL = "gemini-2.5-flash-image"

# ========= Generation parameters =========
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SEED = None  # None means random


def load_image_as_base64(path: str) -> str:
    """Load image file and convert to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(path: str) -> str:
    """Get MIME type based on file extension."""
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    elif lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    elif lower.endswith(".gif"):
        return "image/gif"
    elif lower.endswith(".webp"):
        return "image/webp"
    return "image/png"


def generate_image(api_key: str, prompt: str, reference_image_path: str, output_path: str,
                   temperature: float = None, seed: int = None) -> bool:
    """
    Generate an image using Google Gemini API.

    Args:
        api_key: Gemini API key
        prompt: Text prompt for image generation
        reference_image_path: Path to reference image
        output_path: Path to save generated image
        temperature: Controls randomness (0.0-2.0, higher = more random)
        seed: Random seed for reproducibility

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load reference image
        image_base64 = load_image_as_base64(reference_image_path)
        mime_type = get_mime_type(reference_image_path)

        # Build API URL
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        # Build generationConfig
        generation_config = {
            "responseModalities": ["TEXT", "IMAGE"]
        }
        if temperature is not None:
            generation_config["temperature"] = temperature
        if seed is not None:
            generation_config["seed"] = seed

        # Build request payload with text prompt and reference image
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": generation_config
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()

        # Extract image from response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    blob = part.get("inlineData") or part.get("inline_data")
                    if blob and "data" in blob:
                        image_data = base64.b64decode(blob["data"])
                        with open(output_path, "wb") as f:
                            f.write(image_data)
                        return True

        print(f"  ✗ No image returned from API")
        print(f"  Response: {json.dumps(result, indent=2)[:500]}")
        return False

    except requests.exceptions.RequestException as e:
        print(f"  ✗ API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_open(api_key: str, input_dir: Path, temperature: float = None, seed: int = None) -> bool:
    """
    Generate the fully-opened state image for an articulated object.

    Reads the prompt from prompt_open.json and uses input.png as the
    reference image. Saves the result as open.png in the same directory.

    Args:
        api_key: Gemini API key
        input_dir: Directory containing prompt_open.json and input.png
        temperature: Controls randomness (0.0-2.0)
        seed: Random seed for reproducibility

    Returns:
        True if successful, False otherwise
    """
    prompt_path = input_dir / "prompt_open.json"
    image_path = input_dir / "input.png"
    output_path = input_dir / "open.png"

    if not prompt_path.exists():
        print(f"  ✗ prompt_open.json not found in {input_dir}")
        return False

    if not image_path.exists():
        print(f"  ✗ input.png not found in {input_dir}")
        return False

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)

        prompt = prompt_data.get("prompt", "")
        if not prompt:
            print(f"  ✗ 'prompt' field is empty in prompt_open.json")
            return False

        print(f"  Prompt: {prompt}")

        success = generate_image(api_key, prompt, str(image_path), str(output_path),
                                 temperature=temperature, seed=seed)

        if success:
            print(f"  ✓ Saved generated image to: {output_path}")

        return success

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate fully-opened object image using Google Gemini API."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing prompt_open.json and input.png"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API Key (overrides GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for generation (0.0-2.0, higher = more random)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    api_key = args.api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set GEMINI_API_KEY env var.", file=sys.stderr)
        sys.exit(2)

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Generating open-state image for: {input_path}")
    if args.temperature is not None:
        print(f"  Temperature: {args.temperature}")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    print("-" * 80)

    success = process_open(api_key, input_path,
                           temperature=args.temperature, seed=args.seed)

    print("-" * 80)
    if success:
        print("Image generation completed successfully")
    else:
        print("Image generation failed")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
