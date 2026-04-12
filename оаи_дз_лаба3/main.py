import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode == "L":
        return np.asarray(img, dtype=np.uint8)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(gray + 0.5, 0, 255).astype(np.uint8)


def save_gray(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr, mode="L").save(path)


def weighted_median_hill(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    pad = np.pad(img, 1, mode="edge")

    n00 = pad[0:h, 0:w]
    n01 = pad[0:h, 1:w + 1]
    n02 = pad[0:h, 2:w + 2]
    n10 = pad[1:h + 1, 0:w]
    n11 = pad[1:h + 1, 1:w + 1]
    n12 = pad[1:h + 1, 2:w + 2]
    n20 = pad[2:h + 2, 0:w]
    n21 = pad[2:h + 2, 1:w + 1]
    n22 = pad[2:h + 2, 2:w + 2]

    stack = np.stack([n00, n01, n02, n10, n11, n12, n20, n21, n22], axis=-1)
    weights = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    expanded = np.repeat(stack, weights, axis=-1)
    sorted_vals = np.sort(expanded, axis=-1)
    return sorted_vals[..., 8]


def difference_image(src: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    unique = np.unique(src)
    is_binary = unique.size <= 2 and set(unique.tolist()).issubset({0, 255})
    if is_binary:
        diff = np.bitwise_xor(src, filtered)
        return diff
    diff = np.abs(filtered.astype(np.int16) - src.astype(np.int16)).astype(np.uint8)
    max_val = int(diff.max())
    if max_val > 0:
        scale = 255.0 / max_val
        diff = np.clip(diff.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return diff


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        return []
    exts = {".png", ".bmp", ".jpg", ".jpeg"}
    files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def process_image(path: Path, out_dir: Path):
    gray = load_image(path)
    base = path.stem

    save_gray(gray, out_dir / f"{base}_gray.png")

    filtered = weighted_median_hill(gray)
    save_gray(filtered, out_dir / f"{base}_hill_median.png")

    diff = difference_image(gray, filtered)
    save_gray(diff, out_dir / f"{base}_diff.png")


def main():
    parser = argparse.ArgumentParser(description="Lab 3: weighted median (hill mask) variant 7")
    parser.add_argument("--input", "-i", default="input", help="Input file or directory")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    if not images:
        print("No input images found. Put images into input/.")
        return 1

    for p in images:
        process_image(p, output_path)
        print(f"Processed: {p.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
