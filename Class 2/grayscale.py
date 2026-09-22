"""Convert an image to grayscale.

Usage:
    python grayscale.py <path_or_url> [output_path]

Accepts a local file path or a direct image URL (not a Notion page URL --
Notion pages require auth and aren't image files themselves; use the
image's "Copy link" URL or a downloaded local file).
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


def to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    image = load_image(source)
    gray = to_grayscale(image)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        parsed = urlparse(source)
        stem = Path(parsed.path if parsed.scheme else source).stem or "image"
        output_path = Path(f"{stem}_grayscale.png")

    gray.save(output_path)
    print(f"Saved grayscale image to {output_path}")


if __name__ == "__main__":
    main()
