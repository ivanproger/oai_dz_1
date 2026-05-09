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

NAME_TO_CHAR = {name: symbol for symbol, name in HEBREW_SYMBOLS}
PHRASE_LOGICAL = [
    ["alef", "nun", "yod"],
    ["alef", "vav", "he", "bet"],
    ["alef", "vav", "tav", "final_kaf"],
]
GROUND_TRUTH_NAMES = [name for word in PHRASE_LOGICAL for name in word]
GROUND_TRUTH_TEXT = "\u05d0\u05e0\u05d9 \u05d0\u05d5\u05d4\u05d1 \u05d0\u05d5\u05ea\u05da"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 7: feature-based classification for Hebrew symbols")
    parser.add_argument("--font", default=r"C:\Windows\Fonts\arial.ttf", help="Font path")
    parser.add_argument("--template-size", type=int, default=96, help="Template font size")
    parser.add_argument("--base-size", type=int, default=96, help="Base phrase font size")
    parser.add_argument("--experiment-size", type=int, default=106, help="Experiment phrase font size")
    parser.add_argument("--canvas", type=int, default=256, help="Canvas size for rendering")
    parser.add_argument("--threshold", type=int, default=180, help="Binarization threshold")
    parser.add_argument("--padding", type=int, default=8, help="Padding after trimming")
    parser.add_argument("--char-gap", type=int, default=8, help="Gap between characters")
    parser.add_argument("--word-gap", type=int, default=28, help="Gap between words")
    parser.add_argument("--cut-threshold", type=int, default=1, help="Profile threshold for segmentation")
    parser.add_argument("--input", default="input", help="Input directory")
    parser.add_argument("--output", default="output", help="Output directory")
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


def build_glyphs(font_path: Path, font_size: int, canvas: int, threshold: int, padding: int) -> dict[str, np.ndarray]:
    font = ImageFont.truetype(str(font_path), font_size)
    glyphs = {}
    for symbol, name in HEBREW_SYMBOLS:
        glyphs[name] = render_symbol_binary(symbol, font, canvas, threshold, padding)
    return glyphs


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
    line_h = max(word_img.shape[0] for word_img in word_images)
    line_w = sum(word_img.shape[1] for word_img in word_images) + word_gap * (len(word_images) - 1)
    line = np.zeros((line_h, line_w), dtype=np.uint8)
    x = 0
    for index, word_img in enumerate(word_images):
        y = (line_h - word_img.shape[0]) // 2
        line[y : y + word_img.shape[0], x : x + word_img.shape[1]] = np.maximum(
            line[y : y + word_img.shape[0], x : x + word_img.shape[1]],
            word_img,
        )
        x += word_img.shape[1]
        if index < len(word_images) - 1:
            x += word_gap
    return np.pad(line, margin, mode="constant", constant_values=0)


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    active = False
    start = 0
    for idx, value in enumerate(mask.tolist()):
        if value and not active:
            start = idx
            active = True
        elif not value and active:
            runs.append((start, idx - 1))
            active = False
    if active:
        runs.append((start, len(mask) - 1))
    return runs


def segment_phrase(binary: np.ndarray, cut_threshold: int, rtl_order: bool = True) -> tuple[list[tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    row_profile = binary.sum(axis=1)
    row_runs = find_runs(row_profile > cut_threshold)
    if not row_runs:
        return [], row_profile, np.zeros(binary.shape[1], dtype=np.int32)
    top, bottom = row_runs[0]
    line = binary[top : bottom + 1, :]
    col_profile = line.sum(axis=0)
    col_runs = find_runs(col_profile > cut_threshold)
    boxes = []
    for left, right in col_runs:
        char_region = line[:, left : right + 1]
        inner_rows = find_runs(char_region.sum(axis=1) > cut_threshold)
        if not inner_rows:
            continue
        inner_top, inner_bottom = inner_rows[0]
        boxes.append((left, top + inner_top, right, top + inner_bottom))
    boxes.sort(key=lambda item: item[0], reverse=rtl_order)
    return boxes, row_profile, col_profile


def feature_dict(binary: np.ndarray) -> dict[str, float]:
    h, w = binary.shape
    mass = float(binary.sum())
    area = float(h * w)
    ys, xs = np.nonzero(binary)
    centroid_x = float(xs.mean()) if mass else 0.0
    centroid_y = float(ys.mean()) if mass else 0.0
    inertia_x = float(((ys - centroid_y) ** 2).sum()) if mass else 0.0
    inertia_y = float(((xs - centroid_x) ** 2).sum()) if mass else 0.0
    return {
        "mass_norm": mass / area if area else 0.0,
        "centroid_x_norm": centroid_x / max(1.0, float(w - 1)),
        "centroid_y_norm": centroid_y / max(1.0, float(h - 1)),
        "inertia_x_norm": inertia_x / max(1.0, mass * h * h),
        "inertia_y_norm": inertia_y / max(1.0, mass * w * w),
    }


def feature_vector(binary: np.ndarray) -> np.ndarray:
    data = feature_dict(binary)
    return np.array(
        [
            data["mass_norm"],
            data["centroid_x_norm"],
            data["centroid_y_norm"],
            data["inertia_x_norm"],
            data["inertia_y_norm"],
        ],
        dtype=np.float64,
    )


def create_profile_x_image(profile: np.ndarray, title: str) -> Image.Image:
    count = len(profile)
    chart_w = max(360, count * 7)
    chart_h = 220
    left = 56
    top = 28
    image = Image.new("RGB", (left + chart_w + 20, top + chart_h + 42), "white")
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
    image = Image.new("RGB", (left + chart_w + 40, top + chart_h + 28), "white")
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


def create_sheet(entries: list[tuple[str, str, Image.Image]], title: str) -> Image.Image:
    cols = 6
    cell_w = 180
    cell_h = 168
    rows = (len(entries) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w + 40, rows * cell_h + 70), "white")
    draw = ImageDraw.Draw(canvas)
    font_sym = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
    font_name = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    draw.text((20, 16), title, font=font_sym, fill="black")
    for idx, (symbol, name, image) in enumerate(entries):
        row = idx // cols
        col = idx % cols
        x0 = 20 + col * cell_w
        y0 = 56 + row * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 12, y0 + cell_h - 12), outline="#cfcfcf", width=2)
        fitted = fit_image(Image.merge("RGB", (image, image, image)), 96, 96)
        canvas.paste(fitted, (x0 + 36 + (96 - fitted.width) // 2, y0 + 18 + (96 - fitted.height) // 2))
        draw.text((x0 + 14, y0 + 118), f"{symbol}  {name}", font=font_name, fill="black")
    return canvas


def create_segmentation_preview(binary: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    base = Image.merge("RGB", (binary_to_image(binary),) * 3)
    draw = ImageDraw.Draw(base)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 18)
    for idx, (left, top, right, bottom) in enumerate(boxes, 1):
        draw.rectangle((left, top, right, bottom), outline=(220, 0, 0), width=2)
        draw.text((left, max(0, top - 22)), str(idx), font=font, fill=(220, 0, 0))
    return base


def similarity_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def names_to_spaced_text(names: list[str]) -> str:
    words = []
    index = 0
    for word in PHRASE_LOGICAL:
        chunk = names[index : index + len(word)]
        words.append("".join(NAME_TO_CHAR[name] for name in chunk))
        index += len(word)
    return " ".join(words)


def classify_segment(vector: np.ndarray, template_vectors: dict[str, np.ndarray]) -> list[tuple[str, str, float, float]]:
    hypotheses = []
    for name, template_vector in template_vectors.items():
        distance = float(np.linalg.norm(vector - template_vector))
        similarity = similarity_from_distance(distance)
        hypotheses.append((name, NAME_TO_CHAR[name], similarity, distance))
    hypotheses.sort(key=lambda item: item[2], reverse=True)
    return hypotheses


def save_template_features(rows: list[dict[str, str | float]], path: Path) -> None:
    fields = ["symbol", "name", "mass_norm", "centroid_x_norm", "centroid_y_norm", "inertia_x_norm", "inertia_y_norm"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(fields)
        for row in rows:
            writer.writerow(
                [
                    row["symbol"],
                    row["name"],
                    f"{row['mass_norm']:.6f}",
                    f"{row['centroid_x_norm']:.6f}",
                    f"{row['centroid_y_norm']:.6f}",
                    f"{row['inertia_x_norm']:.6f}",
                    f"{row['inertia_y_norm']:.6f}",
                ]
            )


def save_hypotheses_text(results: list[list[tuple[str, str, float, float]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, hypotheses in enumerate(results, 1):
            line_parts = []
            for name, char, similarity, distance in hypotheses:
                line_parts.append(f"{char}({name})={similarity:.6f}, d={distance:.6f}")
            handle.write(f"{idx}: [{'; '.join(line_parts)}]\n")


def save_top_hypotheses_csv(results: list[list[tuple[str, str, float, float]]], path: Path, top_k: int = 5) -> None:
    headers = ["index"]
    for rank in range(1, top_k + 1):
        headers.extend([f"rank_{rank}_name", f"rank_{rank}_char", f"rank_{rank}_similarity", f"rank_{rank}_distance"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(headers)
        for idx, hypotheses in enumerate(results, 1):
            row = [idx]
            for name, char, similarity, distance in hypotheses[:top_k]:
                row.extend([name, char, f"{similarity:.6f}", f"{distance:.6f}"])
            writer.writerow(row)


def save_segment_rows(rows: list[dict[str, str | int]], path: Path) -> None:
    fields = ["index", "symbol_name", "symbol_char", "left", "top", "right", "bottom", "width", "height"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def create_hypothesis_preview(results: list[list[tuple[str, str, float, float]]], path: Path, title: str, top_k: int = 3) -> None:
    rows = len(results) + 2
    image = Image.new("RGB", (1120, 80 + rows * 34), "white")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)
    font = ImageFont.truetype(r"C:\Windows\Fonts\cour.ttf", 18)
    draw.text((20, 16), title, font=font_title, fill="black")
    y = 56
    for idx, hypotheses in enumerate(results, 1):
        parts = [f"{char} ({name}) {similarity:.4f}" for name, char, similarity, _ in hypotheses[:top_k]]
        draw.text((20, y), f"{idx:02d}: " + " | ".join(parts), font=font, fill="black")
        y += 30
    image.save(path)


def recognize_phrase(
    glyphs: dict[str, np.ndarray],
    output_dir: Path,
    input_dir: Path,
    phrase_name: str,
    char_gap: int,
    word_gap: int,
    cut_threshold: int,
    template_vectors: dict[str, np.ndarray],
    feature_padding: int,
) -> dict[str, object]:
    binary = create_phrase_from_glyphs(glyphs, char_gap, word_gap)
    phrase_path = input_dir / f"{phrase_name}.bmp"
    binary_to_image(binary).save(phrase_path)
    boxes, row_profile, col_profile = segment_phrase(binary, cut_threshold, rtl_order=True)
    phrase_dir = output_dir / phrase_name
    phrase_dir.mkdir(exist_ok=True, parents=True)
    (phrase_dir / "segments").mkdir(exist_ok=True)
    create_profile_y_image(row_profile, f"{phrase_name}: профиль по Y").save(phrase_dir / "horizontal_profile.png")
    create_profile_x_image(col_profile, f"{phrase_name}: профиль по X").save(phrase_dir / "vertical_profile.png")
    create_segmentation_preview(binary, boxes).save(phrase_dir / "segmentation_boxes.png")

    segment_rows = []
    all_hypotheses: list[list[tuple[str, str, float, float]]] = []
    predicted_names: list[str] = []
    for idx, box in enumerate(boxes, 1):
        left, top, right, bottom = box
        segment = binary[top : bottom + 1, left : right + 1]
        segment_for_features = np.pad(segment, feature_padding, mode="constant", constant_values=0)
        segment_path = phrase_dir / "segments" / f"{idx:02d}.png"
        binary_to_image(segment).save(segment_path)
        hypotheses = classify_segment(feature_vector(segment_for_features), template_vectors)
        all_hypotheses.append(hypotheses)
        predicted_names.append(hypotheses[0][0])
        segment_rows.append(
            {
                "index": idx,
                "symbol_name": GROUND_TRUTH_NAMES[idx - 1] if idx - 1 < len(GROUND_TRUTH_NAMES) else "",
                "symbol_char": NAME_TO_CHAR.get(GROUND_TRUTH_NAMES[idx - 1], "") if idx - 1 < len(GROUND_TRUTH_NAMES) else "",
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left + 1,
                "height": bottom - top + 1,
            }
        )

    save_segment_rows(segment_rows, phrase_dir / "segments.csv")
    save_hypotheses_text(all_hypotheses, phrase_dir / "hypotheses.txt")
    save_top_hypotheses_csv(all_hypotheses, phrase_dir / "top_hypotheses.csv", top_k=5)
    create_hypothesis_preview(all_hypotheses, phrase_dir / "hypotheses_preview.png", f"{phrase_name}: лучшие гипотезы", top_k=3)

    predicted_text = names_to_spaced_text(predicted_names[: len(GROUND_TRUTH_NAMES)])
    errors = sum(1 for expected, predicted in zip(GROUND_TRUTH_NAMES, predicted_names) if expected != predicted)
    accuracy = 100.0 * (len(GROUND_TRUTH_NAMES) - errors) / max(1, len(GROUND_TRUTH_NAMES))
    with (phrase_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"ground_truth={GROUND_TRUTH_TEXT}\n")
        handle.write(f"predicted={predicted_text}\n")
        handle.write(f"errors={errors}\n")
        handle.write(f"accuracy={accuracy:.2f}\n")
    return {
        "phrase_path": phrase_path,
        "boxes": boxes,
        "predicted_text": predicted_text,
        "errors": errors,
        "accuracy": accuracy,
        "dir": phrase_dir,
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "templates").mkdir(exist_ok=True)
    (output_dir / "base" / "segments").mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment" / "segments").mkdir(parents=True, exist_ok=True)

    template_glyphs = build_glyphs(Path(args.font), args.template_size, args.canvas, args.threshold, args.padding)
    template_vectors: dict[str, np.ndarray] = {}
    feature_rows = []
    sheet_entries = []
    for symbol, name in HEBREW_SYMBOLS:
        glyph = template_glyphs[name]
        img = binary_to_image(glyph)
        img.save(output_dir / "templates" / f"{name}.png")
        vector_data = feature_dict(glyph)
        template_vectors[name] = np.array(
            [
                vector_data["mass_norm"],
                vector_data["centroid_x_norm"],
                vector_data["centroid_y_norm"],
                vector_data["inertia_x_norm"],
                vector_data["inertia_y_norm"],
            ],
            dtype=np.float64,
        )
        feature_rows.append({"symbol": symbol, "name": name, **vector_data})
        sheet_entries.append((symbol, name, img))
    save_template_features(feature_rows, output_dir / "template_features.csv")
    create_sheet(sheet_entries, "Эталонные символы иврита").save(output_dir / "template_sheet.png")

    base_glyphs = build_glyphs(Path(args.font), args.base_size, args.canvas, args.threshold, args.padding)
    experiment_glyphs = build_glyphs(Path(args.font), args.experiment_size, args.canvas, args.threshold, args.padding)

    base_result = recognize_phrase(
        glyphs=base_glyphs,
        output_dir=output_dir,
        input_dir=input_dir,
        phrase_name="base",
        char_gap=args.char_gap,
        word_gap=args.word_gap,
        cut_threshold=args.cut_threshold,
        template_vectors=template_vectors,
        feature_padding=args.padding,
    )
    experiment_result = recognize_phrase(
        glyphs=experiment_glyphs,
        output_dir=output_dir,
        input_dir=input_dir,
        phrase_name="experiment",
        char_gap=args.char_gap,
        word_gap=args.word_gap,
        cut_threshold=args.cut_threshold,
        template_vectors=template_vectors,
        feature_padding=args.padding,
    )

    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["case", "font_size", "predicted", "errors", "accuracy"])
        writer.writerow(["base", args.base_size, base_result["predicted_text"], base_result["errors"], f"{base_result['accuracy']:.2f}"])
        writer.writerow(["experiment", args.experiment_size, experiment_result["predicted_text"], experiment_result["errors"], f"{experiment_result['accuracy']:.2f}"])

    print(f"Base accuracy: {base_result['accuracy']:.2f}%")
    print(f"Experiment accuracy: {experiment_result['accuracy']:.2f}%")
    print(f"Saved output to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
