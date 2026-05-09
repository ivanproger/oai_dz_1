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

PHRASE_WORDS = [
    ["alef", "nun", "yod"],
    ["alef", "vav", "he", "bet"],
    ["alef", "vav", "tav", "final_kaf"],
]

ALPHABET_NAMES = [name for _, name in HEBREW_SYMBOLS]

IMAGE_CONFIGS = [
    {
        "name": "phrase_soft",
        "background": "soft",
        "text_color": (38, 48, 78),
        "lines": [
            PHRASE_WORDS,
            PHRASE_WORDS,
            PHRASE_WORDS,
            PHRASE_WORDS,
        ],
    },
    {
        "name": "alphabet_parchment",
        "background": "parchment",
        "text_color": (92, 60, 34),
        "lines": [
            [ALPHABET_NAMES[0:5], ALPHABET_NAMES[5:10], ALPHABET_NAMES[10:14]],
            [ALPHABET_NAMES[14:18], ALPHABET_NAMES[18:22]],
            [ALPHABET_NAMES[22:25], ALPHABET_NAMES[25:27]],
        ],
    },
    {
        "name": "phrase_pattern",
        "background": "pattern",
        "text_color": (24, 68, 48),
        "lines": [
            [PHRASE_WORDS[0], ALPHABET_NAMES[0:4], PHRASE_WORDS[1]],
            [ALPHABET_NAMES[8:12], PHRASE_WORDS[2], ALPHABET_NAMES[18:22]],
            PHRASE_WORDS,
        ],
    },
]

DISPLAY_NAMES = {
    "phrase_soft": "\u0424\u0440\u0430\u0437\u0430 \u043d\u0430 \u0441\u0432\u0435\u0442\u043b\u043e\u043c \u0444\u043e\u043d\u0435",
    "alphabet_parchment": "\u0410\u043b\u0444\u0430\u0432\u0438\u0442 \u043d\u0430 \u043f\u0435\u0440\u0433\u0430\u043c\u0435\u043d\u0442\u043d\u043e\u043c \u0444\u043e\u043d\u0435",
    "phrase_pattern": "\u0421\u043c\u0435\u0448\u0430\u043d\u043d\u0430\u044f \u043a\u043e\u043c\u043f\u043e\u0437\u0438\u0446\u0438\u044f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 8: texture analysis and contrast enhancement")
    parser.add_argument("--font", default=r"C:\Windows\Fonts\arial.ttf", help="Font path")
    parser.add_argument("--font-size", type=int, default=54, help="Font size for Hebrew glyphs")
    parser.add_argument("--canvas", type=int, default=200, help="Canvas size for glyph rendering")
    parser.add_argument("--threshold", type=int, default=180, help="Binarization threshold")
    parser.add_argument("--padding", type=int, default=4, help="Padding around rendered glyph")
    parser.add_argument("--char-gap", type=int, default=8, help="Gap between characters")
    parser.add_argument("--word-gap", type=int, default=26, help="Gap between words")
    parser.add_argument("--line-gap", type=int, default=24, help="Gap between lines")
    parser.add_argument("--levels", type=int, default=16, help="Gray levels for GLCM quantization")
    parser.add_argument("--input", default="input", help="Input directory")
    parser.add_argument("--output", default="output", help="Output directory")
    return parser.parse_args()


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


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / img.width, box_h / img.height)
    size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    return img.resize(size, Image.Resampling.LANCZOS)


def build_glyphs(font_path: Path, font_size: int, canvas: int, threshold: int, padding: int) -> dict[str, np.ndarray]:
    font = ImageFont.truetype(str(font_path), font_size)
    glyphs = {}
    for symbol, name in HEBREW_SYMBOLS:
        glyphs[name] = render_symbol_binary(symbol, font, canvas, threshold, padding)
    return glyphs


def compose_line(words: list[list[str]], glyphs: dict[str, np.ndarray], char_gap: int, word_gap: int) -> np.ndarray:
    display_words = [list(reversed(word)) for word in reversed(words)]
    word_images = []
    for word in display_words:
        heights = [glyphs[name].shape[0] for name in word]
        widths = [glyphs[name].shape[1] for name in word]
        word_h = max(heights)
        word_w = sum(widths) + char_gap * (len(word) - 1)
        canvas = np.zeros((word_h, word_w), dtype=np.uint8)
        x = 0
        for name in word:
            glyph = glyphs[name]
            y = (word_h - glyph.shape[0]) // 2
            canvas[y : y + glyph.shape[0], x : x + glyph.shape[1]] = np.maximum(
                canvas[y : y + glyph.shape[0], x : x + glyph.shape[1]],
                glyph,
            )
            x += glyph.shape[1] + char_gap
        word_images.append(canvas)
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
    return line


def compose_block(lines: list[list[list[str]]], glyphs: dict[str, np.ndarray], char_gap: int, word_gap: int, line_gap: int) -> np.ndarray:
    line_images = [compose_line(line, glyphs, char_gap, word_gap) for line in lines]
    block_h = sum(img.shape[0] for img in line_images) + line_gap * (len(line_images) - 1)
    block_w = max(img.shape[1] for img in line_images)
    block = np.zeros((block_h, block_w), dtype=np.uint8)
    y = 0
    for img in line_images:
        x = (block_w - img.shape[1]) // 2
        block[y : y + img.shape[0], x : x + img.shape[1]] = np.maximum(
            block[y : y + img.shape[0], x : x + img.shape[1]],
            img,
        )
        y += img.shape[0] + line_gap
    return block


def background_soft(height: int, width: int) -> np.ndarray:
    yy = np.linspace(0, 1, height)[:, None]
    xx = np.linspace(0, 1, width)[None, :]
    r = 226 - 18 * yy + 8 * xx
    g = 236 - 12 * yy + 4 * xx
    b = 248 - 10 * yy + 18 * xx
    image = np.stack([r, g, b], axis=-1)
    return np.clip(image, 0, 255).astype(np.uint8)


def background_parchment(height: int, width: int) -> np.ndarray:
    yy = np.linspace(0, 1, height)[:, None]
    xx = np.linspace(0, 1, width)[None, :]
    base = 214 + 26 * (1 - yy) - 14 * xx
    noise = 10 * np.sin(xx * 18.0) + 7 * np.cos(yy * 12.0)
    r = base + noise + 8
    g = base + noise - 2
    b = base + noise - 18
    image = np.stack([r, g, b], axis=-1)
    return np.clip(image, 0, 255).astype(np.uint8)


def background_pattern(height: int, width: int) -> np.ndarray:
    yy = np.linspace(0, 1, height)[:, None]
    xx = np.linspace(0, 1, width)[None, :]
    stripes = 18 * np.sin(xx * 24.0) + 12 * np.cos((xx + yy) * 16.0)
    r = 202 + stripes - 18 * yy
    g = 228 + 0.7 * stripes - 10 * yy
    b = 214 + 0.4 * stripes + 6 * xx
    image = np.stack([r, g, b], axis=-1)
    return np.clip(image, 0, 255).astype(np.uint8)


def make_background(kind: str, height: int, width: int) -> np.ndarray:
    if kind == "soft":
        return background_soft(height, width)
    if kind == "parchment":
        return background_parchment(height, width)
    return background_pattern(height, width)


def put_text_on_background(background: np.ndarray, text_mask: np.ndarray, text_color: tuple[int, int, int]) -> np.ndarray:
    result = background.astype(np.float32).copy()
    color = np.array(text_color, dtype=np.float32)
    mask = text_mask[..., None].astype(np.float32)
    result = result * (1.0 - mask) + color * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def create_input_images(glyphs: dict[str, np.ndarray], args: argparse.Namespace) -> list[Path]:
    input_dir = Path(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for cfg in IMAGE_CONFIGS:
        block = compose_block(cfg["lines"], glyphs, args.char_gap, args.word_gap, args.line_gap)
        h, w = block.shape
        canvas_h = h + 80
        canvas_w = max(w + 120, 920)
        background = make_background(cfg["background"], canvas_h, canvas_w)
        mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        y = (canvas_h - h) // 2
        x = (canvas_w - w) // 2
        mask[y : y + h, x : x + w] = block
        rgb = put_text_on_background(background, mask, cfg["text_color"])
        path = input_dir / f"{cfg['name']}.png"
        Image.fromarray(rgb, mode="RGB").save(path)
        saved.append(path)
    return saved


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    l = (maxc + minc) * 0.5
    delta = maxc - minc
    s = np.zeros_like(l)
    nonzero = delta > 1e-8
    s1 = delta / np.maximum(maxc + minc, 1e-8)
    s2 = delta / np.maximum(2.0 - maxc - minc, 1e-8)
    s = np.where(nonzero & (l < 0.5), s1, s)
    s = np.where(nonzero & (l >= 0.5), s2, s)
    h = np.zeros_like(l)
    rd = (((g - b) / np.maximum(delta, 1e-8)) % 6.0) / 6.0
    gd = (((b - r) / np.maximum(delta, 1e-8)) + 2.0) / 6.0
    bd = (((r - g) / np.maximum(delta, 1e-8)) + 4.0) / 6.0
    h = np.where(nonzero & (maxc == r), rd, h)
    h = np.where(nonzero & (maxc == g), gd, h)
    h = np.where(nonzero & (maxc == b), bd, h)
    return np.stack([h % 1.0, s, l], axis=-1)


def hue2rgb(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t % 1.0
    result = np.empty_like(t)
    result[:] = p
    mask = t < (1.0 / 6.0)
    result = np.where(mask, p + (q - p) * 6.0 * t, result)
    mask = (t >= (1.0 / 6.0)) & (t < 0.5)
    result = np.where(mask, q, result)
    mask = (t >= 0.5) & (t < (2.0 / 3.0))
    result = np.where(mask, p + (q - p) * (2.0 / 3.0 - t) * 6.0, result)
    return result


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    h = hsl[..., 0]
    s = hsl[..., 1]
    l = hsl[..., 2]
    q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
    p = 2.0 * l - q
    r = np.where(s == 0, l, hue2rgb(p, q, h + 1.0 / 3.0))
    g = np.where(s == 0, l, hue2rgb(p, q, h))
    b = np.where(s == 0, l, hue2rgb(p, q, h - 1.0 / 3.0))
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def equalize_gray(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist = np.bincount(gray.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = np.flatnonzero(cdf)
    if nonzero.size == 0:
        return gray.copy(), hist
    cdf_min = cdf[nonzero[0]]
    denom = max(1, int(cdf[-1] - cdf_min))
    lut = np.round((cdf - cdf_min) * 255.0 / denom).astype(np.int32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray], hist


def quantize_gray(gray: np.ndarray, levels: int) -> np.ndarray:
    return np.minimum(levels - 1, (gray.astype(np.uint16) * levels) // 256).astype(np.int16)


def glcm_matrix(gray: np.ndarray, levels: int) -> np.ndarray:
    q = quantize_gray(gray, levels)
    matrix = np.zeros((levels, levels), dtype=np.int64)
    offsets = [(1, -1), (-1, -1), (-1, 1), (1, 1)]
    height, width = q.shape
    for dx, dy in offsets:
        x0 = max(0, -dx)
        x1 = min(width, width - dx)
        y0 = max(0, -dy)
        y1 = min(height, height - dy)
        a = q[y0:y1, x0:x1]
        b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        np.add.at(matrix, (a.ravel(), b.ravel()), 1)
    return matrix


def glcm_features(matrix: np.ndarray) -> tuple[float, float]:
    total = float(matrix.sum())
    if total == 0:
        return 0.0, 0.0
    p = matrix.astype(np.float64) / total
    i, j = np.indices(matrix.shape)
    diff2 = (i - j) ** 2
    contrast = float((diff2 * p).sum())
    lun = float((p / (1.0 + diff2)).sum())
    return contrast, lun


def draw_histogram_pair(hist_before: np.ndarray, hist_after: np.ndarray, title: str) -> Image.Image:
    width = 980
    height = 320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    draw.text((20, 12), title, font=font_title, fill="black")

    def draw_chart(hist: np.ndarray, x0: int, color: tuple[int, int, int], label: str) -> None:
        chart_w = 420
        chart_h = 220
        top = 70
        left = x0
        draw.text((left, 44), label, font=font, fill="black")
        draw.line((left + 36, top, left + 36, top + chart_h), fill="black", width=2)
        draw.line((left + 36, top + chart_h, left + 36 + chart_w, top + chart_h), fill="black", width=2)
        max_val = int(hist.max()) if hist.size else 0
        if max_val > 0:
            for idx, value in enumerate(hist):
                x = left + 36 + idx
                y = top + chart_h - int(round((value / max_val) * (chart_h - 8)))
                draw.line((x, top + chart_h, x, y), fill=color)
        draw.text((left + 6, top + chart_h - 8), "0", font=font, fill="black")
        draw.text((left + 4, top - 8), str(max_val), font=font, fill="black")
        draw.text((left + 30, top + chart_h + 6), "0", font=font, fill="black")
        draw.text((left + 36 + chart_w - 18, top + chart_h + 6), "255", font=font, fill="black")

    draw_chart(hist_before, 20, (40, 40, 40), "До преобразования")
    draw_chart(hist_after, 500, (160, 40, 40), "После преобразования")
    return image


def glcm_display(matrix: np.ndarray, title: str) -> Image.Image:
    log_matrix = np.log1p(matrix.astype(np.float32))
    if log_matrix.max() > 0:
        disp = (255.0 * log_matrix / log_matrix.max()).astype(np.uint8)
    else:
        disp = np.zeros_like(matrix, dtype=np.uint8)
    matrix_img = Image.fromarray(disp, mode="L").resize((360, 360), Image.Resampling.NEAREST).convert("RGB")
    canvas = Image.new("RGB", (420, 430), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    draw.text((20, 12), title, font=font_title, fill="black")
    canvas.paste(matrix_img, (30, 48))
    draw.rectangle((30, 48, 390, 408), outline="#cfcfcf", width=2)
    draw.text((22, 412), "Логарифмически нормированное отображение", font=font, fill="black")
    return canvas


def create_summary_panel(
    name: str,
    original_rgb: np.ndarray,
    gray_before: np.ndarray,
    contrast_rgb: np.ndarray,
    gray_after: np.ndarray,
    hist_image: Image.Image,
    glcm_before_img: Image.Image,
    glcm_after_img: Image.Image,
    features_before: tuple[float, float],
    features_after: tuple[float, float],
) -> Image.Image:
    panel = Image.new("RGB", (1500, 1080), "white")
    draw = ImageDraw.Draw(panel)
    title_font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 30)
    text_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
    draw.text((24, 18), name, font=title_font, fill="black")
    orig = fit_image(Image.fromarray(original_rgb, mode="RGB"), 450, 240)
    gray1 = fit_image(Image.fromarray(gray_before, mode="L").convert("RGB"), 450, 240)
    contr = fit_image(Image.fromarray(contrast_rgb, mode="RGB"), 450, 240)
    gray2 = fit_image(Image.fromarray(gray_after, mode="L").convert("RGB"), 450, 240)
    panel.paste(orig, (24, 72))
    panel.paste(gray1, (516, 72))
    panel.paste(contr, (1008, 72))
    panel.paste(gray2, (24, 352))
    panel.paste(fit_image(hist_image, 960, 320), (516, 352))
    panel.paste(fit_image(glcm_before_img, 460, 440), (24, 624))
    panel.paste(fit_image(glcm_after_img, 460, 440), (516, 624))
    draw.text((24, 46), "Исходное RGB", font=text_font, fill="black")
    draw.text((516, 46), "L-канал до преобразования", font=text_font, fill="black")
    draw.text((1008, 46), "RGB после выравнивания гистограммы", font=text_font, fill="black")
    draw.text((24, 326), "L-канал после преобразования", font=text_font, fill="black")
    draw.text((24, 592), "GLCM исходного изображения", font=text_font, fill="black")
    draw.text((516, 592), "GLCM контрастированного изображения", font=text_font, fill="black")
    draw.text((1008, 624), f"CON до: {features_before[0]:.6f}", font=text_font, fill="black")
    draw.text((1008, 658), f"LUN до: {features_before[1]:.6f}", font=text_font, fill="black")
    draw.text((1008, 708), f"CON после: {features_after[0]:.6f}", font=text_font, fill="black")
    draw.text((1008, 742), f"LUN после: {features_after[1]:.6f}", font=text_font, fill="black")
    return panel


def process_image(path: Path, output_dir: Path, levels: int) -> dict[str, object]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    hsl = rgb_to_hsl(rgb)
    gray_before = np.round(hsl[..., 2] * 255.0).astype(np.uint8)
    gray_after, hist_before = equalize_gray(gray_before)
    hist_after = np.bincount(gray_after.ravel(), minlength=256)
    hsl_after = hsl.copy()
    hsl_after[..., 2] = gray_after.astype(np.float32) / 255.0
    contrast_rgb = hsl_to_rgb(hsl_after)
    matrix_before = glcm_matrix(gray_before, levels)
    matrix_after = glcm_matrix(gray_after, levels)
    features_before = glcm_features(matrix_before)
    features_after = glcm_features(matrix_after)

    stem = path.stem
    display_name = DISPLAY_NAMES.get(stem, stem)
    Image.fromarray(gray_before, mode="L").save(output_dir / "gray" / f"{stem}_gray_before.png")
    Image.fromarray(gray_after, mode="L").save(output_dir / "gray" / f"{stem}_gray_after.png")
    Image.fromarray(contrast_rgb, mode="RGB").save(output_dir / "contrast" / f"{stem}_contrast_rgb.png")

    hist_img = draw_histogram_pair(hist_before, hist_after, f"{display_name}: гистограммы яркости")
    hist_img.save(output_dir / "histograms" / f"{stem}_hist.png")
    glcm_before_img = glcm_display(matrix_before, f"{display_name}: GLCM до")
    glcm_after_img = glcm_display(matrix_after, f"{display_name}: GLCM после")
    glcm_before_img.save(output_dir / "glcm" / f"{stem}_glcm_before.png")
    glcm_after_img.save(output_dir / "glcm" / f"{stem}_glcm_after.png")

    np.savetxt(output_dir / "glcm" / f"{stem}_glcm_before.csv", matrix_before, fmt="%d", delimiter=";")
    np.savetxt(output_dir / "glcm" / f"{stem}_glcm_after.csv", matrix_after, fmt="%d", delimiter=";")

    summary = create_summary_panel(
        display_name,
        rgb,
        gray_before,
        contrast_rgb,
        gray_after,
        hist_img,
        glcm_before_img,
        glcm_after_img,
        features_before,
        features_after,
    )
    summary.save(output_dir / "summaries" / f"{stem}_summary.png")

    return {
        "image": stem,
        "display_name": display_name,
        "con_before": features_before[0],
        "lun_before": features_before[1],
        "con_after": features_after[0],
        "lun_after": features_after[1],
    }


def save_feature_summary(rows: list[dict[str, object]], path: Path) -> None:
    fields = ["image", "con_before", "lun_before", "con_after", "lun_after"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(fields)
        for row in rows:
            writer.writerow(
                [
                    row["image"],
                    f"{row['con_before']:.6f}",
                    f"{row['lun_before']:.6f}",
                    f"{row['con_after']:.6f}",
                    f"{row['lun_after']:.6f}",
                ]
            )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gray").mkdir(exist_ok=True)
    (output_dir / "contrast").mkdir(exist_ok=True)
    (output_dir / "histograms").mkdir(exist_ok=True)
    (output_dir / "glcm").mkdir(exist_ok=True)
    (output_dir / "summaries").mkdir(exist_ok=True)

    glyphs = build_glyphs(Path(args.font), args.font_size, args.canvas, args.threshold, args.padding)
    inputs = create_input_images(glyphs, args)
    rows = []
    for path in inputs:
        rows.append(process_image(path, output_dir, args.levels))
    save_feature_summary(rows, output_dir / "feature_summary.csv")
    print(f"Generated {len(inputs)} input images")
    print(f"Saved output to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
