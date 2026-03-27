import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def save_gray(arr: np.ndarray, path: Path) -> None:
    arr8 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr8, mode="L").save(path)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    arr8 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr8, mode="RGB").save(path)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def integral_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return integral


def local_stats(gray: np.ndarray, window: int):
    if window <= 0 or window % 2 == 0:
        raise ValueError("Window size must be positive odd integer")

    h, w = gray.shape
    r = window // 2

    y = np.arange(h)
    x = np.arange(w)
    y0 = np.clip(y - r, 0, h - 1)
    y1 = np.clip(y + r, 0, h - 1) + 1
    x0 = np.clip(x - r, 0, w - 1)
    x1 = np.clip(x + r, 0, w - 1) + 1

    integ = integral_image(gray)
    integ2 = integral_image(gray * gray)

    sum_ = (
        integ[np.ix_(y1, x1)]
        - integ[np.ix_(y0, x1)]
        - integ[np.ix_(y1, x0)]
        + integ[np.ix_(y0, x0)]
    )
    sum2 = (
        integ2[np.ix_(y1, x1)]
        - integ2[np.ix_(y0, x1)]
        - integ2[np.ix_(y1, x0)]
        + integ2[np.ix_(y0, x0)]
    )

    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    mean = sum_ / area
    var = sum2 / area - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean, std


def sauvola_binarize(gray: np.ndarray, window: int, k: float, r: float = 128.0) -> np.ndarray:
    mean, std = local_stats(gray, window)
    thresh = mean * (1.0 + k * (std / r - 1.0))
    binary = (gray > thresh).astype(np.uint8) * 255
    return binary


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        return []

    exts = {".png", ".bmp"}
    files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def process_image(path: Path, out_dir: Path, windows, k: float):
    rgb = load_rgb(path)
    base = path.stem

    save_rgb(rgb, out_dir / f"{base}_original.png")

    gray = rgb_to_gray(rgb)
    save_gray(gray, out_dir / f"{base}_gray.bmp")

    for w in windows:
        bin_img = sauvola_binarize(gray, w, k)
        save_gray(bin_img, out_dir / f"{base}_sauvola_w{w}_k{str(k).replace('.', '_')}.bmp")


def parse_windows(text: str):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    windows = [int(p) for p in parts]
    return windows


def main():
    parser = argparse.ArgumentParser(
        description="Lab 2: grayscale + Sauvola binarization (variant 7)"
    )
    parser.add_argument("--input", "-i", default="input", help="Input file or directory (png/bmp)")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--windows", default="3,25", help="Odd window sizes, comma-separated")
    parser.add_argument("--k", type=float, default=0.5, help="Sauvola k parameter")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    windows = parse_windows(args.windows)
    if not windows:
        print("No window sizes provided.")
        return 1

    for w in windows:
        if w <= 0 or w % 2 == 0:
            print(f"Invalid window size: {w}. Must be positive odd integer.")
            return 1

    images = collect_images(input_path)
    if not images:
        print("No input images found. Use PNG or BMP and place them in the input folder.")
        return 1

    for path in images:
        process_image(path, output_path, windows, args.k)
        print(f"Processed: {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
