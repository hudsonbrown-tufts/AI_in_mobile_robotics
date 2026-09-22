"""Threshold an image to pure black/white.

Usage:
    python binarize.py <path_or_url> [output_path] [threshold]

Pixels below the threshold (default 128) become 0 (black),
pixels at or above become 255 (white).
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


def load_image(source: str) -> Image.Image:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        from io import BytesIO

        return Image.open(BytesIO(response.content))
    return Image.open(source)


def binarize(image: Image.Image, threshold: int = 128) -> Image.Image:
    gray = image.convert("L")
    return gray.point(lambda pixel: 255 if pixel >= threshold else 0)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    threshold = 128

    if len(sys.argv) >= 4:
        output_path = Path(sys.argv[2])
        threshold = int(sys.argv[3])
    elif len(sys.argv) == 3:
        output_path = Path(sys.argv[2])
    else:
        parsed = urlparse(source)
        stem = Path(parsed.path if parsed.scheme else source).stem or "image"
        output_path = Path(f"{stem}_binary.png")

    image = load_image(source)
    result = binarize(image, threshold)
    result.save(output_path)
    print(f"Saved binarized image to {output_path}")


if __name__ == "__main__":
    main()
