import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HEBREW_SYMBOLS = [
    ("\u05d0", "alef"),
    ("\u05d1", "bet"),
    ("\u05d2", "gimel"),
    ("\u05d3", "dalet"),
    ("\u05d4", "he"),
    ("\u05d5", "vav"),
    ("\u05d6", "zayin"),
    ("\u05d7", "het"),
    ("\u05d8", "tet"),
    ("\u05d9", "yod"),
    ("\u05db", "kaf"),
    ("\u05dc", "lamed"),
    ("\u05de", "mem"),
    ("\u05e0", "nun"),
    ("\u05e1", "samekh"),
    ("\u05e2", "ayin"),
    ("\u05e4", "pe"),
    ("\u05e6", "tsadi"),
    ("\u05e7", "qof"),
    ("\u05e8", "resh"),
    ("\u05e9", "shin"),
    ("\u05ea", "tav"),
    ("\u05da", "final_kaf"),
    ("\u05dd", "final_mem"),
    ("\u05df", "final_nun"),
    ("\u05e3", "final_pe"),
    ("\u05e5", "final_tsadi"),
]


def binary_to_image(binary: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(binary > 0, 0, 255).astype(np.uint8), mode="L")


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / img.width, box_h / img.height)
    new_size = (
        max(1, int(round(img.width * scale))),
        max(1, int(round(img.height * scale))),
    )
    return img.resize(new_size, Image.Resampling.NEAREST)


def render_symbol_binary(
    symbol: str,
    font: ImageFont.FreeTypeFont,
    canvas: int,
    threshold: int,
    padding: int,
) -> np.ndarray:
    image = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas - text_w) // 2 - bbox[0]
    y = (canvas - text_h) // 2 - bbox[1]
    draw.text((x, y), symbol, font=font, fill=0)
    mask = np.asarray(image, dtype=np.uint8) < threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError(f"Symbol {symbol!r} was not rendered.")
    trimmed = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].astype(np.uint8)
    return np.pad(trimmed, padding, mode="constant", constant_values=0)


def quadrant_masses(binary: np.ndarray) -> tuple[int, int, int, int]:
    h, w = binary.shape
    mid_h = h // 2
    mid_w = w // 2
    q1 = int(binary[:mid_h, :mid_w].sum())
    q2 = int(binary[:mid_h, mid_w:].sum())
    q3 = int(binary[mid_h:, :mid_w].sum())
    q4 = int(binary[mid_h:, mid_w:].sum())
    return q1, q2, q3, q4


def compute_features(binary: np.ndarray) -> dict[str, float | int]:
    h, w = binary.shape
    mass = int(binary.sum())
    q1, q2, q3, q4 = quadrant_masses(binary)
    areas = [
        max(1, (h // 2) * (w // 2)),
        max(1, (h // 2) * (w - w // 2)),
        max(1, (h - h // 2) * (w // 2)),
        max(1, (h - h // 2) * (w - w // 2)),
    ]
    ys, xs = np.nonzero(binary)
    centroid_x = float(xs.mean()) if mass else 0.0
    centroid_y = float(ys.mean()) if mass else 0.0
    inertia_x = float(((ys - centroid_y) ** 2).sum()) if mass else 0.0
    inertia_y = float(((xs - centroid_x) ** 2).sum()) if mass else 0.0
    return {
        "width": w,
        "height": h,
        "mass_total": mass,
        "mass_q1": q1,
        "mass_q2": q2,
        "mass_q3": q3,
        "mass_q4": q4,
        "spec_q1": q1 / areas[0],
        "spec_q2": q2 / areas[1],
        "spec_q3": q3 / areas[2],
        "spec_q4": q4 / areas[3],
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_x_norm": centroid_x / max(1, w - 1),
        "centroid_y_norm": centroid_y / max(1, h - 1),
        "inertia_x": inertia_x,
        "inertia_y": inertia_y,
        "inertia_x_norm": inertia_x / max(1, mass * h * h),
        "inertia_y_norm": inertia_y / max(1, mass * w * w),
    }


def create_x_profile_image(profile: np.ndarray, title: str) -> Image.Image:
    count = len(profile)
    chart_w = max(360, count * 7)
    chart_h = 220
    left = 56
    top = 24
    right = 20
    bottom = 40
    image = Image.new("RGB", (left + chart_w + right, top + chart_h + bottom), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    font_bold = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 18)
    max_val = int(profile.max()) if profile.size else 0
    draw.text((12, 6), title, font=font_bold, fill="black")
    draw.line((left, top, left, top + chart_h), fill="black", width=2)
    draw.line((left, top + chart_h, left + chart_w, top + chart_h), fill="black", width=2)
    if max_val > 0 and count > 0:
        bar_w = max(1, chart_w // count)
        for idx, value in enumerate(profile):
            x0 = left + idx * bar_w
            x1 = left + max(1, (idx + 1) * bar_w - 1)
            y1 = top + chart_h
            y0 = y1 - int(round((value / max_val) * (chart_h - 8)))
            draw.rectangle((x0, y0, x1, y1), fill="#222222")
    draw.text((8, top + chart_h - 8), "0", font=font, fill="black")
    draw.text((6, top - 8), str(max_val), font=font, fill="black")
    draw.text((left - 6, top + chart_h + 8), "0", font=font, fill="black")
    draw.text((left + chart_w // 2 - 10, top + chart_h + 8), str(count // 2), font=font, fill="black")
    draw.text((left + chart_w - 20, top + chart_h + 8), str(max(0, count - 1)), font=font, fill="black")
    return image


def create_y_profile_image(profile: np.ndarray, title: str) -> Image.Image:
    count = len(profile)
    chart_h = max(260, count * 7)
    chart_w = 240
    left = 44
    top = 24
    right = 40
    bottom = 28
    image = Image.new("RGB", (left + chart_w + right, top + chart_h + bottom), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    font_bold = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 18)
    max_val = int(profile.max()) if profile.size else 0
    draw.text((12, 6), title, font=font_bold, fill="black")
    draw.line((left, top, left, top + chart_h), fill="black", width=2)
    draw.line((left, top + chart_h, left + chart_w, top + chart_h), fill="black", width=2)
    if max_val > 0 and count > 0:
        bar_h = max(1, chart_h // count)
        for idx, value in enumerate(profile):
            y0 = top + idx * bar_h
            y1 = top + max(1, (idx + 1) * bar_h - 1)
            x0 = left
            x1 = x0 + int(round((value / max_val) * (chart_w - 8)))
            draw.rectangle((x0, y0, x1, y1), fill="#222222")
    draw.text((12, top + chart_h + 2), "0", font=font, fill="black")
    draw.text((left + chart_w - 18, top + chart_h + 2), str(max_val), font=font, fill="black")
    draw.text((left - 28, top - 8), "0", font=font, fill="black")
    draw.text((left - 34, top + chart_h // 2 - 8), str(count // 2), font=font, fill="black")
    draw.text((left - 40, top + chart_h - 16), str(max(0, count - 1)), font=font, fill="black")
    return image


def create_overview(symbol_img: Image.Image, x_img: Image.Image, y_img: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (1120, 520), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 28)
    font_sub = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
    draw.text((24, 18), label, font=font_title, fill="black")
    draw.text((24, 56), "Символ", font=font_sub, fill="black")
    draw.text((316, 56), "Профиль X", font=font_sub, fill="black")
    draw.text((720, 56), "Профиль Y", font=font_sub, fill="black")
    symbol_rgb = Image.merge("RGB", (symbol_img, symbol_img, symbol_img))
    symbol_fit = fit_image(symbol_rgb, 240, 360)
    x_fit = fit_image(x_img, 360, 360)
    y_fit = fit_image(y_img, 320, 360)
    canvas.paste(symbol_fit, (24 + (240 - symbol_fit.width) // 2, 106 + (360 - symbol_fit.height) // 2))
    canvas.paste(x_fit, (300 + (360 - x_fit.width) // 2, 106 + (360 - x_fit.height) // 2))
    canvas.paste(y_fit, (704 + (320 - y_fit.width) // 2, 106 + (360 - y_fit.height) // 2))
    draw.rectangle((16, 96, 280, 486), outline="#cfcfcf", width=2)
    draw.rectangle((292, 96, 672, 486), outline="#cfcfcf", width=2)
    draw.rectangle((696, 96, 1040, 486), outline="#cfcfcf", width=2)
    return canvas


def create_sheet(entries: list[tuple[str, str, Image.Image]]) -> Image.Image:
    cols = 6
    cell_w = 180
    cell_h = 168
    rows = (len(entries) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w + 40, rows * cell_h + 70), "white")
    draw = ImageDraw.Draw(canvas)
    font_sym = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
    font_name = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    draw.text((20, 16), "Эталонные символы иврита", font=font_sym, fill="black")
    for idx, (symbol, name, image) in enumerate(entries):
        row = idx // cols
        col = idx % cols
        x0 = 20 + col * cell_w
        y0 = 56 + row * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 12, y0 + cell_h - 12), outline="#cfcfcf", width=2)
        fitted = fit_image(Image.merge("RGB", (image, image, image)), 96, 96)
        px = x0 + 36 + (96 - fitted.width) // 2
        py = y0 + 18 + (96 - fitted.height) // 2
        canvas.paste(fitted, (px, py))
        draw.text((x0 + 14, y0 + 118), f"{symbol}  {name}", font=font_name, fill="black")
    return canvas


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fields = [
        "symbol",
        "name",
        "width",
        "height",
        "mass_total",
        "mass_q1",
        "mass_q2",
        "mass_q3",
        "mass_q4",
        "spec_q1",
        "spec_q2",
        "spec_q3",
        "spec_q4",
        "centroid_x",
        "centroid_y",
        "centroid_x_norm",
        "centroid_y_norm",
        "inertia_x",
        "inertia_y",
        "inertia_x_norm",
        "inertia_y_norm",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(fields)
        for row in rows:
            values = []
            for field in fields:
                value = row[field]
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            writer.writerow(values)


def process_symbols(output_dir: Path, font_path: Path, font_size: int, canvas: int, threshold: int, padding: int) -> None:
    symbols_dir = output_dir / "symbols"
    profiles_dir = output_dir / "profiles"
    overviews_dir = output_dir / "overviews"
    font = ImageFont.truetype(str(font_path), font_size)
    csv_rows: list[dict[str, float | int | str]] = []
    sheet_entries: list[tuple[str, str, Image.Image]] = []
    for symbol, name in HEBREW_SYMBOLS:
        binary = render_symbol_binary(symbol, font, canvas, threshold, padding)
        image = binary_to_image(binary)
        symbol_path = symbols_dir / f"{name}.png"
        image.save(symbol_path)
        x_profile = binary.sum(axis=0).astype(np.int32)
        y_profile = binary.sum(axis=1).astype(np.int32)
        x_img = create_x_profile_image(x_profile, f"{name}: X profile")
        y_img = create_y_profile_image(y_profile, f"{name}: Y profile")
        x_img.save(profiles_dir / f"{name}_x.png")
        y_img.save(profiles_dir / f"{name}_y.png")
        overview = create_overview(image, x_img, y_img, f"{symbol}  ({name})")
        overview.save(overviews_dir / f"{name}_overview.png")
        features = compute_features(binary)
        csv_rows.append({"symbol": symbol, "name": name, **features})
        sheet_entries.append((symbol, name, image))
    create_sheet(sheet_entries).save(output_dir / "alphabet_sheet.png")
    write_csv(csv_rows, output_dir / "features.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 5: feature extraction for Hebrew symbols")
    parser.add_argument(
        "--font",
        default=r"C:\Windows\Fonts\arial.ttf",
        help="Path to a font file with Hebrew glyphs",
    )
    parser.add_argument("--font-size", type=int, default=96, help="Font size in pixels")
    parser.add_argument("--canvas", type=int, default=256, help="Temporary canvas size")
    parser.add_argument("--threshold", type=int, default=180, help="Binarization threshold")
    parser.add_argument("--padding", type=int, default=8, help="White border around the glyph")
    parser.add_argument("--output", default="output", help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "symbols").mkdir(exist_ok=True)
    (output_dir / "profiles").mkdir(exist_ok=True)
    (output_dir / "overviews").mkdir(exist_ok=True)
    process_symbols(
        output_dir=output_dir,
        font_path=Path(args.font),
        font_size=args.font_size,
        canvas=args.canvas,
        threshold=args.threshold,
        padding=args.padding,
    )
    print(f"Saved results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
