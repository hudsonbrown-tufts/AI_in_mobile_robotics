"""Binarize an image, dilate it, then subtract the (undilated) binary image.

result = dilation(A) - A

This extracts the outer boundary of the binarized shapes (the ring of
pixels dilation adds around each foreground region).

Usage:
    python dilate_subtract.py <path_or_url> [output_path] [threshold] [kernel_size]

threshold default: 128
kernel_size default: 3 (must be odd; controls how thick the boundary is)
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image, ImageFilter


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


def dilate(binary_image: Image.Image, kernel_size: int = 3) -> Image.Image:
    kernel_size = max(1, kernel_size | 1)  # force odd
    return binary_image.filter(ImageFilter.MaxFilter(kernel_size))


def subtract(dilated: Image.Image, original_binary: Image.Image) -> Image.Image:
    a = np.array(dilated, dtype=np.int16)
    b = np.array(original_binary, dtype=np.int16)
    diff = np.clip(a - b, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    threshold = int(sys.argv[3]) if len(sys.argv) >= 4 else 128
    kernel_size = int(sys.argv[4]) if len(sys.argv) >= 5 else 3

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        parsed = urlparse(source)
        stem = Path(parsed.path if parsed.scheme else source).stem or "image"
        output_path = Path(f"{stem}_dilate_minus_original.png")

    image = load_image(source)
    binary = binarize(image, threshold)
    dilated = dilate(binary, kernel_size)
    result = subtract(dilated, binary)

    result.save(output_path)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
