import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

EPS = 1e-8


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def save_rgb(arr: np.ndarray, path: Path) -> None:
    arr8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr8, mode="RGB").save(path)


def save_gray(arr: np.ndarray, path: Path) -> None:
    arr8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr8, mode="L").save(path)


def rgb_split(arr: np.ndarray):
    return arr[..., 0], arr[..., 1], arr[..., 2]


def rgb_to_hsi(arr: np.ndarray):
    r, g, b = rgb_split(arr)
    i = (r + g + b) / 3.0

    min_rgb = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b
    s = np.zeros_like(i)
    mask_sum = sum_rgb > EPS
    s[mask_sum] = 1.0 - (3.0 * min_rgb[mask_sum] / sum_rgb[mask_sum])

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + EPS
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    h = np.where(b <= g, theta, 2.0 * math.pi - theta)
    h = h / (2.0 * math.pi)

    h = np.where(s <= EPS, 0.0, h)

    return h, s, i


def hsi_to_rgb(h: np.ndarray, s: np.ndarray, i: np.ndarray):
    h_rad = h * 2.0 * math.pi

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    mask_s = s > EPS


    mask1 = (h_rad < 2.0 * math.pi / 3.0) & mask_s
    if np.any(mask1):
        h1 = h_rad[mask1]
        b[mask1] = i[mask1] * (1.0 - s[mask1])
        r[mask1] = i[mask1] * (1.0 + (s[mask1] * np.cos(h1) / (np.cos(math.pi / 3.0 - h1) + EPS)))
        g[mask1] = 3.0 * i[mask1] - (r[mask1] + b[mask1])


    mask2 = (h_rad >= 2.0 * math.pi / 3.0) & (h_rad < 4.0 * math.pi / 3.0) & mask_s
    if np.any(mask2):
        h2 = h_rad[mask2] - 2.0 * math.pi / 3.0
        r[mask2] = i[mask2] * (1.0 - s[mask2])
        g[mask2] = i[mask2] * (1.0 + (s[mask2] * np.cos(h2) / (np.cos(math.pi / 3.0 - h2) + EPS)))
        b[mask2] = 3.0 * i[mask2] - (r[mask2] + g[mask2])


    mask3 = (h_rad >= 4.0 * math.pi / 3.0) & mask_s
    if np.any(mask3):
        h3 = h_rad[mask3] - 4.0 * math.pi / 3.0
        g[mask3] = i[mask3] * (1.0 - s[mask3])
        b[mask3] = i[mask3] * (1.0 + (s[mask3] * np.cos(h3) / (np.cos(math.pi / 3.0 - h3) + EPS)))
        r[mask3] = 3.0 * i[mask3] - (g[mask3] + b[mask3])


    mask_gray = ~mask_s
    if np.any(mask_gray):
        r[mask_gray] = i[mask_gray]
        g[mask_gray] = i[mask_gray]
        b[mask_gray] = i[mask_gray]

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def resize_bilinear(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    h, w = img.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    ys = np.linspace(0, h - 1, new_h)
    xs = np.linspace(0, w - 1, new_w)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    dy = (ys - y0)[:, None]
    dx = (xs - x0)[None, :]

    if img.ndim == 2:
        ia = img[y0[:, None], x0[None, :]]
        ib = img[y0[:, None], x1[None, :]]
        ic = img[y1[:, None], x0[None, :]]
        id_ = img[y1[:, None], x1[None, :]]
        out = (
            ia * (1 - dx) * (1 - dy)
            + ib * dx * (1 - dy)
            + ic * (1 - dx) * dy
            + id_ * dx * dy
        )
        return out

    ia = img[y0[:, None], x0[None, :], :]
    ib = img[y0[:, None], x1[None, :], :]
    ic = img[y1[:, None], x0[None, :], :]
    id_ = img[y1[:, None], x1[None, :], :]

    wx0 = (1 - dx)[..., None]
    wx1 = dx[..., None]
    wy0 = (1 - dy)[..., None]
    wy1 = dy[..., None]

    out = (
        ia * wx0 * wy0
        + ib * wx1 * wy0
        + ic * wx0 * wy1
        + id_ * wx1 * wy1
    )

    return out


def decimate(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Decimation factor must be positive.")
    return img[::factor, ::factor, ...]


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        return []

    exts = {".png", ".bmp"}
    files = [p for p in input_path.iterdir() if p.suffix.lower() in exts and p.is_file()]
    files.sort()
    return files


def process_image(path: Path, out_dir: Path, m: int, n: int, k: float):
    img = load_image(path)
    base = path.stem

    save_rgb(img, out_dir / f"{base}_original.png")


    r, g, b = rgb_split(img)
    save_gray(r, out_dir / f"{base}_R.png")
    save_gray(g, out_dir / f"{base}_G.png")
    save_gray(b, out_dir / f"{base}_B.png")


    h, s, i = rgb_to_hsi(img)
    save_gray(i, out_dir / f"{base}_HSI_I.png")


    i_inv = 1.0 - i
    rgb_inv = hsi_to_rgb(h, s, i_inv)
    save_rgb(rgb_inv, out_dir / f"{base}_HSI_I_inverted.png")


    if m is not None and m > 0:
        up = resize_bilinear(img, float(m))
        save_rgb(up, out_dir / f"{base}_stretch_M{m}.png")


    if n is not None and n > 0:
        down = decimate(img, n)
        save_rgb(down, out_dir / f"{base}_decimate_N{n}.png")


    if m is not None and n is not None and m > 0 and n > 0:
        up = resize_bilinear(img, float(m))
        two_pass = decimate(up, n)
        save_rgb(two_pass, out_dir / f"{base}_two_pass_K{m}_{n}.png")


    if k is not None and k > 0:
        one_pass = resize_bilinear(img, float(k))
        save_rgb(one_pass, out_dir / f"{base}_one_pass_K{str(k).replace('.', '_')}.png")


def main():
    parser = argparse.ArgumentParser(description="Lab 1: color models and resampling")
    parser.add_argument("--input", "-i", default="input", help="Input file or directory (png/bmp)")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--m", type=int, default=2, help="Stretch factor M (integer > 0)")
    parser.add_argument("--n", type=int, default=2, help="Decimation factor N (integer > 0)")
    parser.add_argument("--k", type=float, default=None, help="Resampling factor K for one-pass")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    k = args.k
    if k is None and args.m > 0 and args.n > 0:
        k = args.m / args.n

    images = collect_images(input_path)
    if not images:
        print("No input images found. Use PNG or BMP and place them in the input folder.")
        return 1

    for path in images:
        process_image(path, output_path, args.m, args.n, k)
        print(f"Processed: {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
