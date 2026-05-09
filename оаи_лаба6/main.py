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

PHRASE_LOGICAL = [
    ["alef", "nun", "yod"],
    ["alef", "vav", "he", "bet"],
    ["alef", "vav", "tav", "final_kaf"],
]

PHRASE_TEXT = "\u05d0\u05e0\u05d9 \u05d0\u05d5\u05d4\u05d1 \u05d0\u05d5\u05ea\u05da"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 6: Hebrew text segmentation")
    parser.add_argument("--font", default=r"C:\Windows\Fonts\arial.ttf", help="Font path")
    parser.add_argument("--font-size", type=int, default=96, help="Font size")
    parser.add_argument("--canvas", type=int, default=256, help="Canvas for symbol rendering")
    parser.add_argument("--threshold", type=int, default=180, help="Binarization threshold")
    parser.add_argument("--padding", type=int, default=8, help="Padding after trimming")
    parser.add_argument("--char-gap", type=int, default=8, help="Gap between characters")
    parser.add_argument("--word-gap", type=int, default=28, help="Gap between words")
    parser.add_argument("--cut-threshold", type=int, default=1, help="Profile threshold for segmentation")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--input", default="input", help="Input directory")
    return parser.parse_args()


def binary_to_image(binary: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(binary > 0, 0, 255).astype(np.uint8), mode="L")


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / img.width, box_h / img.height)
    size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    return img.resize(size, Image.Resampling.NEAREST)


def render_symbol_binary(symbol: str, font: ImageFont.FreeTypeFont, canvas: int, threshold: int, padding: int) -> np.ndarray:
    image = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), symbol, font=font)
    x = (canvas - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (canvas - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), symbol, font=font, fill=0)
    mask = np.asarray(image, dtype=np.uint8) < threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    trimmed = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].astype(np.uint8)
    return np.pad(trimmed, padding, mode="constant", constant_values=0)


def create_profile_x_image(profile: np.ndarray, title: str) -> Image.Image:
    count = len(profile)
    chart_w = max(360, count * 7)
    chart_h = 220
    left = 56
    top = 28
    right = 20
    bottom = 42
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
    draw.text((left + chart_w // 2 - 12, top + chart_h + 8), str(count // 2), font=font, fill="black")
    draw.text((left + chart_w - 24, top + chart_h + 8), str(max(0, count - 1)), font=font, fill="black")
    return image


def create_profile_y_image(profile: np.ndarray, title: str) -> Image.Image:
    count = len(profile)
    chart_h = max(260, count * 7)
    chart_w = 240
    left = 44
    top = 28
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
            x1 = left + int(round((value / max_val) * (chart_w - 8)))
            draw.rectangle((left, y0, x1, y1), fill="#222222")
    draw.text((12, top + chart_h + 2), "0", font=font, fill="black")
    draw.text((left + chart_w - 20, top + chart_h + 2), str(max_val), font=font, fill="black")
    draw.text((left - 28, top - 8), "0", font=font, fill="black")
    draw.text((left - 38, top + chart_h // 2 - 8), str(count // 2), font=font, fill="black")
    draw.text((left - 40, top + chart_h - 16), str(max(0, count - 1)), font=font, fill="black")
    return image


def create_symbol_overview(symbol_img: Image.Image, x_img: Image.Image, y_img: Image.Image, label: str) -> Image.Image:
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


def create_alphabet_sheet(entries: list[tuple[str, str, Image.Image]]) -> Image.Image:
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


def create_phrase_from_glyphs(glyphs: dict[str, np.ndarray], char_gap: int, word_gap: int, margin: int = 6) -> np.ndarray:
    display_words = [list(reversed(word)) for word in reversed(PHRASE_LOGICAL)]
    word_images = []
    for word in display_words:
        heights = [glyphs[name].shape[0] for name in word]
        widths = [glyphs[name].shape[1] for name in word]
        word_h = max(heights)
        word_w = sum(widths) + char_gap * (len(word) - 1)
        word_img = np.zeros((word_h, word_w), dtype=np.uint8)
        x = 0
        for name in word:
            glyph = glyphs[name]
            y = (word_h - glyph.shape[0]) // 2
            word_img[y : y + glyph.shape[0], x : x + glyph.shape[1]] = np.maximum(
                word_img[y : y + glyph.shape[0], x : x + glyph.shape[1]],
                glyph,
            )
            x += glyph.shape[1] + char_gap
        word_images.append(word_img)
    line_h = max(img.shape[0] for img in word_images)
    line_w = sum(img.shape[1] for img in word_images) + word_gap * (len(word_images) - 1)
    line = np.zeros((line_h, line_w), dtype=np.uint8)
    x = 0
    for idx, word_img in enumerate(word_images):
        y = (line_h - word_img.shape[0]) // 2
        line[y : y + word_img.shape[0], x : x + word_img.shape[1]] = np.maximum(
            line[y : y + word_img.shape[0], x : x + word_img.shape[1]],
            word_img,
        )
        x += word_img.shape[1]
        if idx < len(word_images) - 1:
            x += word_gap
    return np.pad(line, margin, mode="constant", constant_values=0)


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    in_run = False
    start = 0
    for idx, value in enumerate(mask.tolist()):
        if value and not in_run:
            start = idx
            in_run = True
        elif not value and in_run:
            runs.append((start, idx - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def segment_phrase(binary: np.ndarray, cut_threshold: int, rtl_order: bool = True) -> tuple[list[tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    row_profile = binary.sum(axis=1)
    line_runs = find_runs(row_profile > cut_threshold)
    if not line_runs:
        return [], row_profile, np.zeros(binary.shape[1], dtype=np.int32)
    top, bottom = line_runs[0]
    line = binary[top : bottom + 1, :]
    col_profile = line.sum(axis=0)
    col_runs = find_runs(col_profile > cut_threshold)
    boxes = []
    for left, right in col_runs:
        char_img = line[:, left : right + 1]
        inner_rows = find_runs(char_img.sum(axis=1) > cut_threshold)
        if not inner_rows:
            continue
        inner_top, inner_bottom = inner_rows[0]
        boxes.append((left, top + inner_top, right, top + inner_bottom))
    boxes.sort(key=lambda item: item[0], reverse=rtl_order)
    return boxes, row_profile, col_profile


def save_boxes_csv(rows: list[dict[str, int | str]], path: Path) -> None:
    fields = ["index", "symbol_name", "symbol_char", "left", "top", "right", "bottom", "width", "height"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_segments(binary: np.ndarray, boxes: list[tuple[int, int, int, int]], output_dir: Path) -> list[Path]:
    expected = [name for word in PHRASE_LOGICAL for name in word]
    char_map = {name: symbol for symbol, name in HEBREW_SYMBOLS}
    saved = []
    rows = []
    for idx, box in enumerate(boxes, 1):
        left, top, right, bottom = box
        segment = binary[top : bottom + 1, left : right + 1]
        name = expected[idx - 1] if idx - 1 < len(expected) else f"seg_{idx:02d}"
        symbol = char_map.get(name, "?")
        path = output_dir / f"{idx:02d}_{name}.png"
        binary_to_image(segment).save(path)
        saved.append(path)
        rows.append(
            {
                "index": idx,
                "symbol_name": name,
                "symbol_char": symbol,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left + 1,
                "height": bottom - top + 1,
            }
        )
    save_boxes_csv(rows, output_dir.parent / "segments.csv")
    return saved


def create_segmentation_preview(binary: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    base = Image.merge("RGB", (binary_to_image(binary),) * 3)
    draw = ImageDraw.Draw(base)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 18)
    for idx, (left, top, right, bottom) in enumerate(boxes, 1):
        draw.rectangle((left, top, right, bottom), outline=(220, 0, 0), width=2)
        draw.text((left, max(0, top - 22)), str(idx), font=font, fill=(220, 0, 0))
    return base


def build_alphabet_assets(font_path: Path, font_size: int, canvas: int, threshold: int, padding: int, output_dir: Path) -> dict[str, np.ndarray]:
    font = ImageFont.truetype(str(font_path), font_size)
    glyphs: dict[str, np.ndarray] = {}
    sheet_entries: list[tuple[str, str, Image.Image]] = []
    profiles_dir = output_dir / "alphabet_profiles"
    overviews_dir = output_dir / "alphabet_overviews"
    symbols_dir = output_dir / "alphabet_symbols"
    for symbol, name in HEBREW_SYMBOLS:
        glyph = render_symbol_binary(symbol, font, canvas, threshold, padding)
        glyphs[name] = glyph
        img = binary_to_image(glyph)
        img.save(symbols_dir / f"{name}.png")
        x_profile = glyph.sum(axis=0).astype(np.int32)
        y_profile = glyph.sum(axis=1).astype(np.int32)
        x_img = create_profile_x_image(x_profile, f"{name}: X profile")
        y_img = create_profile_y_image(y_profile, f"{name}: Y profile")
        x_img.save(profiles_dir / f"{name}_x.png")
        y_img.save(profiles_dir / f"{name}_y.png")
        overview = create_symbol_overview(img, x_img, y_img, f"{symbol}  ({name})")
        overview.save(overviews_dir / f"{name}_overview.png")
        sheet_entries.append((symbol, name, img))
    create_alphabet_sheet(sheet_entries).save(output_dir / "alphabet_sheet.png")
    return glyphs


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    input_dir = Path(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "segments").mkdir(exist_ok=True)
    (output_dir / "alphabet_overviews").mkdir(exist_ok=True)
    (output_dir / "alphabet_profiles").mkdir(exist_ok=True)
    (output_dir / "alphabet_symbols").mkdir(exist_ok=True)

    glyphs = build_alphabet_assets(
        font_path=Path(args.font),
        font_size=args.font_size,
        canvas=args.canvas,
        threshold=args.threshold,
        padding=args.padding,
        output_dir=output_dir,
    )

    phrase_binary = create_phrase_from_glyphs(glyphs, args.char_gap, args.word_gap)
    phrase_path = input_dir / "phrase.bmp"
    binary_to_image(phrase_binary).save(phrase_path)

    boxes, row_profile, col_profile = segment_phrase(phrase_binary, args.cut_threshold, rtl_order=True)
    create_profile_y_image(row_profile, "Профиль строки по Y").save(output_dir / "phrase_horizontal_profile.png")
    create_profile_x_image(col_profile, "Профиль строки по X").save(output_dir / "phrase_vertical_profile.png")
    create_segmentation_preview(phrase_binary, boxes).save(output_dir / "segmentation_boxes.png")
    save_segments(phrase_binary, boxes, output_dir / "segments")

    with (output_dir / "phrase_info.txt").open("w", encoding="utf-8") as handle:
        handle.write("Phrase (logical): " + PHRASE_TEXT + "\n")
        handle.write("Reading order: right-to-left\n")
        handle.write(f"Segments found: {len(boxes)}\n")

    print(f"Saved phrase to: {phrase_path}")
    print(f"Saved output to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
