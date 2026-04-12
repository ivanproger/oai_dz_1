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


def prewitt_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = gray.astype(np.int16)
    p = np.pad(g, 1, mode="edge")

    left = p[0:-2, 0:-2] + p[1:-1, 0:-2] + p[2:, 0:-2]
    right = p[0:-2, 2:] + p[1:-1, 2:] + p[2:, 2:]
    gx = right - left

    top = p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:]
    bottom = p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
    gy = top - bottom

    return gx, gy


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    max_val = float(arr.max())
    if max_val <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = arr * (255.0 / max_val)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def process_image(path: Path, out_dir: Path, threshold: int) -> None:
    gray = load_image(path)
    gx, gy = prewitt_gradients(gray)
    g = np.hypot(gx.astype(np.float32), gy.astype(np.float32))

    gx_img = normalize_to_uint8(np.abs(gx))
    gy_img = normalize_to_uint8(np.abs(gy))
    g_img = normalize_to_uint8(g)
    bin_img = (g_img >= threshold).astype(np.uint8) * 255

    base = path.stem
    save_gray(gray, out_dir / f"{base}_gray.png")
    save_gray(gx_img, out_dir / f"{base}_gx.png")
    save_gray(gy_img, out_dir / f"{base}_gy.png")
    save_gray(g_img, out_dir / f"{base}_g.png")
    save_gray(bin_img, out_dir / f"{base}_bin.png")


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        return []
    exts = {".png", ".bmp", ".jpg", ".jpeg"}
    files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Lab 4: Prewitt operator (variant 7)")
    parser.add_argument("--input", "-i", default="input", help="Input file or directory")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--threshold", "-t", type=int, default=60, help="Threshold for бинаризации G")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    if not images:
        print("No input images found. Put images into input/.")
        return 1

    for p in images:
        process_image(p, output_path, args.threshold)
        print(f"Processed: {p.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
