from __future__ import annotations

import argparse
import csv
import math
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
        "display_name": "Фраза на светлом фоне",
        "background": "soft",
        "text_color": (38, 53, 88),
        "seed": 7,
        "lines": [PHRASE_WORDS, PHRASE_WORDS, PHRASE_WORDS, PHRASE_WORDS],
    },
    {
        "name": "alphabet_parchment",
        "display_name": "Алфавит на пергаментном фоне",
        "background": "parchment",
        "text_color": (92, 58, 36),
        "seed": 17,
        "lines": [
            [ALPHABET_NAMES[0:5], ALPHABET_NAMES[5:10], ALPHABET_NAMES[10:14]],
            [ALPHABET_NAMES[14:18], ALPHABET_NAMES[18:22]],
            [ALPHABET_NAMES[22:25], ALPHABET_NAMES[25:27]],
        ],
    },
    {
        "name": "phrase_pattern",
        "display_name": "Смешанная композиция",
        "background": "pattern",
        "text_color": (25, 68, 52),
        "seed": 27,
        "lines": [
            [PHRASE_WORDS[0], ALPHABET_NAMES[0:4], PHRASE_WORDS[1]],
            [ALPHABET_NAMES[8:12], PHRASE_WORDS[2], ALPHABET_NAMES[18:22]],
            PHRASE_WORDS,
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Лабораторная работа 8")
    parser.add_argument("--font", default=r"C:\Windows\Fonts\arial.ttf")
    parser.add_argument("--font-size", type=int, default=54)
    parser.add_argument("--canvas", type=int, default=220)
    parser.add_argument("--threshold", type=int, default=180)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--char-gap", type=int, default=8)
    parser.add_argument("--word-gap", type=int, default=26)
    parser.add_argument("--line-gap", type=int, default=24)
    parser.add_argument("--input", default="input")
    parser.add_argument("--output", default="output")
    return parser.parse_args()


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        return ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", size)
    if bold:
        return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", size)
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


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


def build_glyphs(font_path: Path, font_size: int, canvas: int, threshold: int, padding: int) -> dict[str, np.ndarray]:
    font = ImageFont.truetype(str(font_path), font_size)
    return {name: render_symbol_binary(symbol, font, canvas, threshold, padding) for symbol, name in HEBREW_SYMBOLS}


def compose_line(words: list[list[str]], glyphs: dict[str, np.ndarray], char_gap: int, word_gap: int) -> np.ndarray:
    display_words = [list(reversed(word)) for word in reversed(words)]
    word_images: list[np.ndarray] = []
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
    for line_image in line_images:
        x = (block_w - line_image.shape[1]) // 2
        block[y : y + line_image.shape[0], x : x + line_image.shape[1]] = np.maximum(
            block[y : y + line_image.shape[0], x : x + line_image.shape[1]],
            line_image,
        )
        y += line_image.shape[0] + line_gap
    return block


def smooth_noise(height: int, width: int, seed: int, octaves: list[tuple[int, float]]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = np.zeros((height, width), dtype=np.float32)
    for cell, weight in octaves:
        coarse_h = max(3, height // cell + 3)
        coarse_w = max(3, width // cell + 3)
        coarse = rng.integers(0, 256, size=(coarse_h, coarse_w), dtype=np.uint8)
        scaled = Image.fromarray(coarse, mode="L").resize((width, height), Image.Resampling.BILINEAR)
        total += weight * (np.asarray(scaled, dtype=np.float32) / 255.0 - 0.5)
    total -= total.min()
    max_value = float(total.max())
    if max_value > 0:
        total /= max_value
    return total


def make_background(kind: str, height: int, width: int, seed: int) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    noise_fine = smooth_noise(height, width, seed, [(22, 0.55), (48, 0.35), (96, 0.20)])
    noise_coarse = smooth_noise(height, width, seed + 100, [(70, 1.0), (140, 0.5)])
    radial = np.sqrt((xx - 0.62) ** 2 + (yy - 0.28) ** 2)

    if kind == "soft":
        light = np.clip(1.0 - radial / 0.82, 0.0, 1.0)
        r = 214 + 18 * light + 32 * noise_fine - 24 * yy + 10 * xx
        g = 226 + 12 * light + 26 * noise_fine - 18 * yy + 6 * xx
        b = 241 + 16 * light + 18 * noise_fine - 12 * yy + 18 * xx
    elif kind == "parchment":
        veins = 18.0 * np.sin(2.0 * math.pi * (xx * 1.8 + yy * 0.9 + 0.22 * noise_coarse))
        stains = 36 * (noise_coarse - 0.5) + 16 * (noise_fine - 0.5)
        r = 204 + stains + veins + 18 * (1.0 - yy)
        g = 188 + 0.78 * stains + 0.72 * veins + 16 * (1.0 - yy)
        b = 148 + 0.45 * stains + 0.35 * veins + 8 * (1.0 - yy)
    else:
        waves = 22.0 * np.sin(2.0 * math.pi * (xx * 2.3 + yy * 0.5)) + 14.0 * np.cos(2.0 * math.pi * (xx * 0.9 - yy * 1.7))
        r = 182 + 24 * noise_fine + 0.8 * waves - 20 * yy
        g = 208 + 28 * noise_fine + 0.5 * waves - 12 * yy
        b = 192 + 20 * noise_fine + 0.9 * waves + 12 * xx

    image = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def put_text_on_background(background: np.ndarray, text_mask: np.ndarray, text_color: tuple[int, int, int]) -> np.ndarray:
    result = background.astype(np.float32).copy()
    shadow = np.zeros_like(text_mask)
    shadow[2:, 2:] = text_mask[:-2, :-2]
    shadow_weight = 0.17 * shadow[..., None].astype(np.float32)
    result *= 1.0 - shadow_weight
    color = np.array(text_color, dtype=np.float32)
    mask = text_mask[..., None].astype(np.float32)
    result = result * (1.0 - mask) + color * mask
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def create_input_images(glyphs: dict[str, np.ndarray], args: argparse.Namespace, input_dir: Path) -> list[dict[str, object]]:
    input_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for cfg in IMAGE_CONFIGS:
        block = compose_block(cfg["lines"], glyphs, args.char_gap, args.word_gap, args.line_gap)
        h, w = block.shape
        canvas_h = h + 96
        canvas_w = max(w + 140, 980)
        background = make_background(cfg["background"], canvas_h, canvas_w, cfg["seed"])
        mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        y = (canvas_h - h) // 2
        x = (canvas_w - w) // 2
        mask[y : y + h, x : x + w] = block
        rgb = put_text_on_background(background, mask, cfg["text_color"])
        path = input_dir / f"{cfg['name']}.png"
        Image.fromarray(rgb, mode="RGB").save(path)
        results.append({"path": path, "display_name": cfg["display_name"]})
    return results


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
    low = nonzero & (l < 0.5)
    high = nonzero & (l >= 0.5)
    s[low] = delta[low] / np.maximum(maxc[low] + minc[low], 1e-8)
    s[high] = delta[high] / np.maximum(2.0 - maxc[high] - minc[high], 1e-8)
    h = np.zeros_like(l)
    delta_safe = np.maximum(delta, 1e-8)
    h = np.where(nonzero & (maxc == r), (((g - b) / delta_safe) % 6.0) / 6.0, h)
    h = np.where(nonzero & (maxc == g), (((b - r) / delta_safe) + 2.0) / 6.0, h)
    h = np.where(nonzero & (maxc == b), (((r - g) / delta_safe) + 4.0) / 6.0, h)
    return np.stack([h % 1.0, s, l], axis=-1)


def hue2rgb(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t % 1.0
    result = np.empty_like(t)
    result[:] = p
    result = np.where(t < 1.0 / 6.0, p + (q - p) * 6.0 * t, result)
    result = np.where((t >= 1.0 / 6.0) & (t < 0.5), q, result)
    result = np.where((t >= 0.5) & (t < 2.0 / 3.0), p + (q - p) * (2.0 / 3.0 - t) * 6.0, result)
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


def equalize_gray(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist_before = np.bincount(gray.ravel(), minlength=256)
    cdf = hist_before.cumsum()
    nonzero = np.flatnonzero(cdf)
    if nonzero.size == 0:
        return gray.copy(), hist_before, hist_before.copy()
    cdf_min = cdf[nonzero[0]]
    denom = max(1, int(cdf[-1] - cdf_min))
    lut = np.round((cdf - cdf_min) * 255.0 / denom).astype(np.int32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    gray_after = lut[gray]
    hist_after = np.bincount(gray_after.ravel(), minlength=256)
    return gray_after, hist_before, hist_after


def glcm_matrix(gray: np.ndarray) -> np.ndarray:
    matrix = np.zeros((256, 256), dtype=np.int64)
    offsets = [(1, -1), (-1, -1), (-1, 1), (1, 1)]
    height, width = gray.shape
    for dx, dy in offsets:
        x0 = max(0, -dx)
        x1 = min(width, width - dx)
        y0 = max(0, -dy)
        y1 = min(height, height - dy)
        a = gray[y0:y1, x0:x1].astype(np.uint32)
        b = gray[y0 + dy : y1 + dy, x0 + dx : x1 + dx].astype(np.uint32)
        pairs = (a << 8) | b
        matrix += np.bincount(pairs.ravel(), minlength=256 * 256).reshape(256, 256)
    return matrix


def glcm_features(matrix: np.ndarray) -> tuple[float, float]:
    total = float(matrix.sum())
    if total == 0.0:
        return 0.0, 0.0
    p = matrix.astype(np.float64) / total
    i, j = np.indices(matrix.shape)
    contrast = float((((i - j) ** 2) * p).sum())
    lun = float((p * p).sum())
    return contrast, lun


def draw_histogram_pair(hist_before: np.ndarray, hist_after: np.ndarray, title: str) -> Image.Image:
    image = Image.new("RGB", (980, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), title, font=load_font(24, bold=True), fill="black")

    def chart(hist: np.ndarray, left: int, label: str, color: tuple[int, int, int]) -> None:
        chart_w = 420
        chart_h = 220
        top = 72
        draw.text((left, 42), label, font=load_font(18), fill="black")
        draw.line((left + 36, top, left + 36, top + chart_h), fill="black", width=2)
        draw.line((left + 36, top + chart_h, left + 36 + chart_w, top + chart_h), fill="black", width=2)
        max_value = int(hist.max()) if hist.size else 0
        if max_value > 0:
            for idx, value in enumerate(hist):
                x = left + 36 + idx
                y = top + chart_h - int(round((value / max_value) * (chart_h - 8)))
                draw.line((x, top + chart_h, x, y), fill=color)
        draw.text((left + 10, top + chart_h - 8), "0", font=load_font(16), fill="black")
        draw.text((left + 5, top - 8), str(max_value), font=load_font(16), fill="black")
        draw.text((left + 30, top + chart_h + 6), "0", font=load_font(16), fill="black")
        draw.text((left + 36 + chart_w - 18, top + chart_h + 6), "255", font=load_font(16), fill="black")

    chart(hist_before, 20, "До выравнивания", (60, 60, 60))
    chart(hist_after, 500, "После выравнивания", (156, 48, 48))
    return image


def glcm_display(matrix: np.ndarray, title: str) -> Image.Image:
    log_matrix = np.log1p(matrix.astype(np.float32))
    if float(log_matrix.max()) > 0.0:
        view = (255.0 * log_matrix / float(log_matrix.max())).astype(np.uint8)
    else:
        view = np.zeros_like(matrix, dtype=np.uint8)
    matrix_image = Image.fromarray(view, mode="L").resize((360, 360), Image.Resampling.NEAREST).convert("RGB")
    canvas = Image.new("RGB", (420, 430), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), title, font=load_font(22, bold=True), fill="black")
    canvas.paste(matrix_image, (30, 48))
    draw.rectangle((30, 48, 390, 408), outline="#cfcfcf", width=2)
    draw.text((26, 412), "Логарифмически нормированное отображение", font=load_font(16), fill="black")
    return canvas


def create_summary_panel(
    display_name: str,
    original_rgb: np.ndarray,
    gray_before: np.ndarray,
    contrast_rgb: np.ndarray,
    gray_after: np.ndarray,
    hist_image: Image.Image,
    glcm_before_image: Image.Image,
    glcm_after_image: Image.Image,
    features_before: tuple[float, float],
    features_after: tuple[float, float],
) -> Image.Image:
    panel = Image.new("RGB", (1500, 1100), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((24, 18), display_name, font=load_font(30, bold=True), fill="black")
    panel.paste(fit_image(Image.fromarray(original_rgb, mode="RGB"), 450, 250), (24, 72))
    panel.paste(fit_image(Image.fromarray(gray_before, mode="L").convert("RGB"), 450, 250), (516, 72))
    panel.paste(fit_image(Image.fromarray(contrast_rgb, mode="RGB"), 450, 250), (1008, 72))
    panel.paste(fit_image(Image.fromarray(gray_after, mode="L").convert("RGB"), 450, 250), (24, 360))
    panel.paste(fit_image(hist_image, 960, 320), (516, 350))
    panel.paste(fit_image(glcm_before_image, 460, 440), (24, 632))
    panel.paste(fit_image(glcm_after_image, 460, 440), (516, 632))
    text_font = load_font(20)
    draw.text((24, 46), "Исходное RGB-изображение", font=text_font, fill="black")
    draw.text((516, 46), "L-канал до выравнивания", font=text_font, fill="black")
    draw.text((1008, 46), "RGB после выравнивания гистограммы", font=text_font, fill="black")
    draw.text((24, 334), "L-канал после выравнивания", font=text_font, fill="black")
    draw.text((24, 600), "GLCM исходного изображения", font=text_font, fill="black")
    draw.text((516, 600), "GLCM после преобразования", font=text_font, fill="black")
    draw.text((1008, 642), f"CON до: {features_before[0]:.6f}", font=text_font, fill="black")
    draw.text((1008, 676), f"LUN до: {features_before[1]:.6f}", font=text_font, fill="black")
    draw.text((1008, 726), f"CON после: {features_after[0]:.6f}", font=text_font, fill="black")
    draw.text((1008, 760), f"LUN после: {features_after[1]:.6f}", font=text_font, fill="black")
    draw.text((1008, 830), "GLCM построена по полному диапазону 256 уровней яркости.", font=text_font, fill="#243447")
    draw.text((1008, 864), "Матрица рассчитывается непосредственно по значениям канала L.", font=text_font, fill="#243447")
    return panel


def text_block(draw: ImageDraw.ImageDraw, text: str, left: int, top: int, width: int, font: ImageFont.FreeTypeFont, fill: str) -> int:
    words = text.split()
    line = ""
    y = top
    line_height = font.size + 10
    for word in words:
        candidate = word if not line else f"{line} {word}"
        right = draw.textbbox((left, y), candidate, font=font)[2]
        if right - left <= width:
            line = candidate
            continue
        if line:
            draw.text((left, y), line, font=font, fill=fill)
            y += line_height
        line = word
    if line:
        draw.text((left, y), line, font=font, fill=fill)
        y += line_height
    return y


def section(draw: ImageDraw.ImageDraw, title: str, y: int) -> int:
    font = load_font(38, bold=True)
    draw.text((92, y), title, font=font, fill="#1f3a5f")
    y = draw.textbbox((92, y), title, font=font)[3] + 12
    draw.line((92, y, 1560, y), fill="#d6deea", width=3)
    return y + 20


def process_image(item: dict[str, object], output_dir: Path) -> dict[str, object]:
    path = Path(item["path"])
    display_name = str(item["display_name"])
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    hsl = rgb_to_hsl(rgb)
    gray_before = np.round(hsl[..., 2] * 255.0).astype(np.uint8)
    gray_after, hist_before, hist_after = equalize_gray(gray_before)
    hsl_after = hsl.copy()
    hsl_after[..., 2] = gray_after.astype(np.float32) / 255.0
    contrast_rgb = hsl_to_rgb(hsl_after)
    matrix_before = glcm_matrix(gray_before)
    matrix_after = glcm_matrix(gray_after)
    features_before = glcm_features(matrix_before)
    features_after = glcm_features(matrix_after)
    stem = path.stem

    Image.fromarray(gray_before, mode="L").save(output_dir / "gray" / f"{stem}_gray_before.png")
    Image.fromarray(gray_after, mode="L").save(output_dir / "gray" / f"{stem}_gray_after.png")
    Image.fromarray(contrast_rgb, mode="RGB").save(output_dir / "contrast" / f"{stem}_contrast_rgb.png")

    hist_image = draw_histogram_pair(hist_before, hist_after, f"{display_name}: гистограммы яркости")
    hist_image.save(output_dir / "histograms" / f"{stem}_hist.png")
    glcm_before_image = glcm_display(matrix_before, f"{display_name}: GLCM до")
    glcm_after_image = glcm_display(matrix_after, f"{display_name}: GLCM после")
    glcm_before_image.save(output_dir / "glcm" / f"{stem}_glcm_before.png")
    glcm_after_image.save(output_dir / "glcm" / f"{stem}_glcm_after.png")

    summary = create_summary_panel(
        display_name,
        rgb,
        gray_before,
        contrast_rgb,
        gray_after,
        hist_image,
        glcm_before_image,
        glcm_after_image,
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
        "summary_path": output_dir / "summaries" / f"{stem}_summary.png",
    }


def save_feature_summary(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["image", "display_name", "con_before", "lun_before", "con_after", "lun_after"])
        for row in rows:
            writer.writerow(
                [
                    row["image"],
                    row["display_name"],
                    f"{row['con_before']:.6f}",
                    f"{row['lun_before']:.6f}",
                    f"{row['con_after']:.6f}",
                    f"{row['lun_after']:.6f}",
                ]
            )


def build_readme_pdf(root: Path, rows: list[dict[str, object]]) -> None:
    page1 = Image.new("RGB", (1654, 2339), "white")
    draw1 = ImageDraw.Draw(page1)
    title_font = load_font(54, bold=True)
    body_font = load_font(26)
    mono_font = load_font(24, mono=True)
    table_font = load_font(22)
    y = 88
    draw1.text((92, y), "Лабораторная работа №8", font=title_font, fill="#1f3a5f")
    y = draw1.textbbox((92, y), "Лабораторная работа №8", font=title_font)[3] + 18
    draw1.line((92, y, 1560, y), fill="#d6deea", width=3)
    y += 34
    draw1.text((92, y), "Текстурный анализ и контрастирование", font=load_font(40, bold=True), fill="#243447")
    y = draw1.textbbox((92, y), "Текстурный анализ и контрастирование", font=load_font(40, bold=True))[3] + 18
    draw1.line((92, y, 1560, y), fill="#d6deea", width=2)
    y += 28
    y = section(draw1, "Вариант 7", y)
    y = text_block(
        draw1,
        "Матрица GLCM строится при d = 1 и направлениях 45°, 135°, 225°, 315°. По матрице рассчитываются признаки CON и LUN. Яркостное преобразование выполняется методом выравнивания гистограммы в канале L модели HSL.",
        92,
        y,
        1468,
        body_font,
        "#243447",
    )
    y += 16
    y = section(draw1, "Особенности реализации", y)
    fixes = [
        "GLCM строится по полному диапазону уровней 0…255 яркостного канала L.",
        "Для визуализации матрицы используется логарифмически нормированное отображение в градациях серого.",
        "Признак LUN вычисляется как локальная однородность Σ p(i,j)^2.",
        "В отчёте для каждого примера приведены исходное RGB-изображение, L-канал, результат выравнивания гистограммы, гистограммы яркости и матрицы GLCM.",
    ]
    for fix in fixes:
        draw1.text((104, y), "•", font=load_font(30, bold=True), fill="#243447")
        y = text_block(draw1, fix, 126, y, 1430, body_font, "#243447") + 4
    y += 8
    y = section(draw1, "Используемые соотношения", y)
    formulas = [
        "L = (max(R,G,B) + min(R,G,B)) / 2",
        "CON = Σ (i - j)^2 · p(i,j)",
        "LUN = Σ p(i,j)^2",
    ]
    for formula in formulas:
        draw1.text((118, y), formula, font=mono_font, fill="#243447")
        y += 42
    y += 12
    y = section(draw1, "Сводная таблица результатов", y)
    draw1.rounded_rectangle((92, y, 1560, y + 56), radius=10, fill="#edf3fb", outline="#d6deea", width=2)
    headers = ["Изображение", "CON до", "LUN до", "CON после", "LUN после"]
    xs = [110, 760, 960, 1160, 1380]
    for header, x in zip(headers, xs):
        draw1.text((x, y + 14), header, font=load_font(22, bold=True), fill="#1f3a5f")
    row_y = y + 56
    for row in rows:
        draw1.rectangle((92, row_y, 1560, row_y + 52), outline="#d6deea", width=1)
        values = [
            str(row["display_name"]),
            f"{row['con_before']:.3f}",
            f"{row['lun_before']:.6f}",
            f"{row['con_after']:.3f}",
            f"{row['lun_after']:.6f}",
        ]
        for value, x in zip(values, xs):
            draw1.text((x, row_y + 14), value, font=table_font, fill="#243447")
        row_y += 52
    y = row_y + 24
    y = section(draw1, "Краткий вывод", y)
    final_text = (
        "Во всех трёх примерах после выравнивания гистограммы возрастает контраст локальных переходов по яркости, "
        "а признак LUN изменяется в зависимости от того, усиливается ли мелкая фактура или выравниваются крупные фоны. "
        "Матрицы GLCM, построенные по полному диапазону уровней яркости, позволяют количественно сопоставить изменения текстуры до и после преобразования."
    )
    text_block(draw1, final_text, 92, y, 1468, body_font, "#243447")

    pages = [page1]
    for row in rows:
        page = Image.new("RGB", (1654, 2339), "white")
        draw = ImageDraw.Draw(page)
        title = str(row["display_name"])
        draw.text((92, 88), title, font=title_font, fill="#1f3a5f")
        y_title = draw.textbbox((92, 88), title, font=title_font)[3] + 18
        draw.line((92, y_title, 1560, y_title), fill="#d6deea", width=3)
        summary = Image.open(Path(row["summary_path"])).convert("RGB")
        page.paste(fit_image(summary, 1470, 1700), (92, 150))
        y_text = 1890
        y_text = section(draw, "Наблюдение", y_text)
        note = (
            f"Для примера «{row['display_name']}» после выравнивания гистограммы значение CON изменилось с "
            f"{row['con_before']:.3f} до {row['con_after']:.3f}, а LUN — с {row['lun_before']:.6f} до {row['lun_after']:.6f}. "
            "Изображение в отчёте показано в цвете и в канале L, поэтому связь между визуальным эффектом и численными признаками видна напрямую."
        )
        text_block(draw, note, 92, y_text, 1468, body_font, "#243447")
        pages.append(page)

    pages[0].save(root / "README.pdf", save_all=True, append_images=pages[1:], resolution=180)


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_dir = root / args.input
    output_dir = root / args.output
    ensure_dirs(
        [
            input_dir,
            output_dir,
            output_dir / "gray",
            output_dir / "contrast",
            output_dir / "histograms",
            output_dir / "glcm",
            output_dir / "summaries",
        ]
    )
    glyphs = build_glyphs(Path(args.font), args.font_size, args.canvas, args.threshold, args.padding)
    inputs = create_input_images(glyphs, args, input_dir)
    rows = [process_image(item, output_dir) for item in inputs]
    save_feature_summary(rows, output_dir / "feature_summary.csv")
    build_readme_pdf(root, rows)
    print(f"Saved lab to: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
