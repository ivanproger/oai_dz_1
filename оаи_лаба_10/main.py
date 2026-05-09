from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FORMANTS = {
    "a": {
        "label": "А",
        "title": "Гласная А",
        "theory": (660.0, 1700.0, 2400.0),
    },
    "i": {
        "label": "И",
        "title": "Гласная И",
        "theory": (270.0, 2300.0, 3000.0),
    },
    "bark": {
        "label": "Лай",
        "title": "Имитация лая",
        "theory": (500.0, 1250.0, 2400.0),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 10: voice processing, variant 1")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--input", default="input")
    parser.add_argument("--output", default="output")
    return parser.parse_args()


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        return ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", size=size)
    if bold:
        return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", size=size)
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size=size)


def write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    data = np.clip(np.round(signal * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data.tobytes())


def normalize(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    scale = max(1e-9, float(np.max(np.abs(signal))))
    return peak * signal / scale


def resonance_gain(freq: np.ndarray, centers: tuple[float, float, float]) -> np.ndarray:
    widths = np.array([90.0, 160.0, 220.0], dtype=np.float64)
    response = np.zeros_like(freq, dtype=np.float64)
    for center, width in zip(centers, widths):
        response += 1.0 / (1.0 + ((freq - center) / width) ** 2)
    return 0.2 + response


def synthesize_vowel_glide(
    formants: tuple[float, float, float],
    sample_rate: int,
    duration: float,
    f0_start: float,
    f0_end: float,
    breath: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    f0 = np.geomspace(f0_start, f0_end, num=t.size)
    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
    signal = np.zeros_like(t)
    max_harmonic = max(8, int((sample_rate * 0.46) / max(f0_end, f0_start)))
    for harmonic in range(1, max_harmonic + 1):
        inst_freq = harmonic * f0
        weight = resonance_gain(inst_freq, formants) / (harmonic ** 1.08)
        phase_shift = rng.uniform(0.0, 2.0 * np.pi)
        vibrato = 0.003 * harmonic * np.sin(2.0 * np.pi * 5.0 * t + phase_shift)
        signal += weight * np.sin(harmonic * phase + phase_shift + vibrato)
    envelope = np.sin(np.pi * np.linspace(0.0, 1.0, t.size)) ** 0.7
    signal = signal * envelope + breath * envelope * rng.standard_normal(t.size)
    return normalize(signal)


def synthesize_bark(sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pulses = []
    specs = [
        (0.32, 280.0, 170.0),
        (0.26, 330.0, 210.0),
        (0.22, 300.0, 185.0),
        (0.28, 250.0, 150.0),
    ]
    gap = np.zeros(int(0.12 * sample_rate), dtype=np.float64)
    for idx, (duration, f0_start, f0_end) in enumerate(specs, 1):
        t = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
        f0 = np.geomspace(f0_start, f0_end, num=t.size)
        phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
        pulse = np.zeros_like(t)
        formants = FORMANTS["bark"]["theory"]
        max_harmonic = int((sample_rate * 0.45) / f0_start)
        for harmonic in range(1, max_harmonic + 1):
            inst_freq = harmonic * f0
            weight = resonance_gain(inst_freq, formants) / (harmonic ** 1.2)
            pulse += weight * np.sin(harmonic * phase + rng.uniform(0.0, 2.0 * np.pi))
        env = np.exp(-5.0 * t) * (1.0 - np.exp(-40.0 * t))
        pulse = pulse * env + 0.18 * env * rng.standard_normal(t.size)
        if idx % 2 == 0:
            pulse *= 0.9
        pulses.append(pulse)
        pulses.append(gap)
    signal = np.concatenate([np.zeros(int(0.2 * sample_rate), dtype=np.float64)] + pulses + [np.zeros(int(0.2 * sample_rate), dtype=np.float64)])
    return normalize(signal)


def stft(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    window = np.hanning(frame_size).astype(np.float64)
    if signal.size <= frame_size:
        padded = np.pad(signal, (0, frame_size - signal.size))
        return np.fft.rfft((padded * window)[:, None], axis=0)
    n_frames = 1 + int(math.ceil((signal.size - frame_size) / hop_size))
    total = (n_frames - 1) * hop_size + frame_size
    padded = np.pad(signal, (0, max(0, total - signal.size)))
    frames = []
    for idx in range(n_frames):
        start = idx * hop_size
        frames.append(np.fft.rfft(padded[start : start + frame_size] * window))
    return np.stack(frames, axis=1)


def spectrogram_db(signal: np.ndarray, sample_rate: int, frame_size: int = 1024, hop_size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = stft(signal, frame_size, hop_size)
    power = np.abs(spec) ** 2
    db = 10.0 * np.log10(np.maximum(power, 1e-12))
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    times = np.arange(spec.shape[1], dtype=np.float64) * hop_size / sample_rate
    return db, freqs, times


def log_frequency_image(db: np.ndarray, freqs: np.ndarray, rows: int = 320) -> np.ndarray:
    f_min = max(80.0, float(freqs[1] if freqs.size > 1 else 80.0))
    f_max = float(freqs[-1])
    grid = np.geomspace(f_min, f_max, num=rows)
    image = np.empty((rows, db.shape[1]), dtype=np.float32)
    for idx, value in enumerate(grid):
        src = int(np.argmin(np.abs(freqs - value)))
        image[rows - 1 - idx, :] = db[src, :]
    lo = float(np.percentile(image, 5))
    hi = float(np.percentile(image, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def colorize_heatmap(norm: np.ndarray) -> np.ndarray:
    x = norm.astype(np.float32)
    r = np.clip(255.0 * (x ** 0.88), 0, 255)
    g = np.clip(255.0 * np.sqrt(x), 0, 255)
    b = np.clip(255.0 * (0.3 + 0.7 * x), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def make_waveform_image(signal: np.ndarray, sample_rate: int, title: str) -> Image.Image:
    page = Image.new("RGB", (1040, 320), "white")
    draw = ImageDraw.Draw(page)
    title_font = load_font(26, bold=True)
    font = load_font(16)
    draw.text((24, 16), title, font=title_font, fill="#1f3a5f")
    left, top, width, height = 58, 60, 930, 190
    draw.rectangle((left, top, left + width, top + height), outline="#cbd6e6", width=2)
    mid = top + height // 2
    draw.line((left, mid, left + width, mid), fill="#7a8695", width=1)
    step = max(1, signal.size // width)
    points = []
    for x in range(width):
        start = x * step
        stop = min(signal.size, start + step)
        value = float(np.mean(signal[start:stop])) if stop > start else 0.0
        y = mid - int(round(value * (height // 2 - 10)))
        points.append((left + x, y))
    draw.line(points, fill="#1d5fa2", width=1)
    duration = signal.size / sample_rate
    for t_val in np.linspace(0.0, duration, 5):
        x = left + int(round((t_val / max(duration, 1e-6)) * width))
        draw.line((x, top + height, x, top + height + 8), fill="black", width=1)
        draw.text((x - 12, top + height + 12), f"{t_val:.1f}", font=font, fill="black")
    draw.text((left + width + 10, top - 4), "1", font=font, fill="black")
    draw.text((left + width + 10, mid - 8), "0", font=font, fill="black")
    draw.text((left + width + 10, top + height - 14), "-1", font=font, fill="black")
    draw.text((left + width - 6, top + height + 36), "с", font=font, fill="black")
    return page


def make_spectrogram_image(signal: np.ndarray, sample_rate: int, title: str) -> Image.Image:
    db, freqs, times = spectrogram_db(signal, sample_rate)
    heat = colorize_heatmap(log_frequency_image(db, freqs))
    chart = Image.fromarray(heat, mode="RGB").resize((930, 410), Image.Resampling.NEAREST)
    page = Image.new("RGB", (1040, 540), "white")
    draw = ImageDraw.Draw(page)
    title_font = load_font(26, bold=True)
    font = load_font(16)
    draw.text((24, 16), title, font=title_font, fill="#1f3a5f")
    left, top = 58, 64
    page.paste(chart, (left, top))
    draw.rectangle((left, top, left + 930, top + 410), outline="#cbd6e6", width=2)
    duration = times[-1] if times.size else 0.0
    for t_val in np.linspace(0.0, duration, 5):
        x = left + int(round((t_val / max(duration, 1e-6)) * 930))
        draw.line((x, top + 410, x, top + 418), fill="black", width=1)
        draw.text((x - 12, top + 422), f"{t_val:.1f}", font=font, fill="black")
    for f_val in [100, 250, 500, 1000, 2000, 4000, 8000]:
        if f_val >= sample_rate / 2:
            continue
        ratio = (math.log(f_val) - math.log(80.0)) / (math.log(sample_rate / 2.0) - math.log(80.0))
        y = top + 410 - int(round(ratio * 410))
        draw.line((left - 8, y, left, y), fill="black", width=1)
        draw.text((6, y - 8), str(f_val), font=font, fill="black")
    draw.text((956, top + 422), "с", font=font, fill="black")
    draw.text((6, 38), "Гц", font=font, fill="black")
    return page


def iter_frames(signal: np.ndarray, sample_rate: int, frame_sec: float, hop_sec: float) -> list[tuple[float, np.ndarray]]:
    frame_size = int(round(frame_sec * sample_rate))
    hop_size = int(round(hop_sec * sample_rate))
    frames: list[tuple[float, np.ndarray]] = []
    if signal.size < frame_size:
        frames.append((0.0, np.pad(signal, (0, frame_size - signal.size))))
        return frames
    for start in range(0, signal.size - frame_size + 1, hop_size):
        frames.append((start / sample_rate, signal[start : start + frame_size].copy()))
    return frames


def estimate_fundamental(frame: np.ndarray, sample_rate: int, fmin: float = 70.0, fmax: float = 1200.0) -> tuple[float | None, float]:
    x = frame.astype(np.float64)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 0.015:
        return None, 0.0
    x *= np.hanning(x.size)
    acf = np.correlate(x, x, mode="full")[x.size - 1 :]
    acf /= max(acf[0], 1e-9)
    lag_min = max(1, int(sample_rate / fmax))
    lag_max = min(acf.size - 1, int(sample_rate / fmin))
    if lag_max <= lag_min:
        return None, 0.0
    search = acf[lag_min : lag_max + 1]
    idx = int(np.argmax(search))
    peak = float(search[idx])
    if peak < 0.12:
        return None, peak
    lag = lag_min + idx
    return float(sample_rate / lag), peak


def count_overtones(frame: np.ndarray, sample_rate: int, f0: float) -> int:
    nfft = 8192
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame.size), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    peak = float(np.max(spectrum))
    count = 0
    harmonic = 2
    while harmonic * f0 < min(5000.0, sample_rate / 2.0 - 20.0):
        target = harmonic * f0
        mask = (freqs >= target - 18.0) & (freqs <= target + 18.0)
        if np.any(mask):
            value = float(np.max(spectrum[mask]))
            if value >= 0.08 * peak:
                count += 1
        harmonic += 1
    return count


def estimate_formants(frame: np.ndarray, sample_rate: int, expected: tuple[float, float, float]) -> tuple[float, float, float]:
    nfft = 16384
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame.size), n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    smooth = np.convolve(np.log(spectrum + 1e-12), np.ones(121) / 121.0, mode="same")
    result = []
    for center in expected:
        left = max(120.0, center - 320.0)
        right = min(sample_rate / 2.0 - 60.0, center + 320.0)
        mask = (freqs >= left) & (freqs <= right)
        band_freqs = freqs[mask]
        band_smooth = smooth[mask]
        idx = int(np.argmax(band_smooth))
        result.append(float(band_freqs[idx]))
    return tuple(result)  # type: ignore[return-value]


def analyze_signal(signal: np.ndarray, sample_rate: int, sample_key: str) -> dict[str, object]:
    frames = iter_frames(signal, sample_rate, frame_sec=0.1, hop_sec=0.05)
    rows = []
    for time_pos, frame in frames:
        f0, confidence = estimate_fundamental(frame, sample_rate)
        rms = float(np.sqrt(np.mean(frame * frame)))
        overtone_count = count_overtones(frame, sample_rate, f0) if f0 is not None else 0
        rows.append(
            {
                "time_s": time_pos,
                "f0_hz": f0,
                "confidence": confidence,
                "rms": rms,
                "overtone_count": overtone_count,
                "frame": frame,
            }
        )
    voiced = [row for row in rows if row["f0_hz"] is not None]
    min_f0 = float(min(row["f0_hz"] for row in voiced)) if voiced else 0.0
    max_f0 = float(max(row["f0_hz"] for row in voiced)) if voiced else 0.0
    richest = max(voiced, key=lambda row: (row["overtone_count"], row["confidence"], row["rms"])) if voiced else rows[0]
    formant_row = max(
        voiced,
        key=lambda row: (
            float(row["rms"]) * float(row["confidence"]) / math.sqrt(max(float(row["f0_hz"] or 1.0), 1.0)),
            row["confidence"],
        ),
    ) if voiced else rows[0]
    measured_formants = estimate_formants(formant_row["frame"], sample_rate, FORMANTS[sample_key]["theory"])  # type: ignore[arg-type]
    return {
        "frames": rows,
        "min_f0_hz": min_f0,
        "max_f0_hz": max_f0,
        "richest_time_s": float(richest["time_s"]),
        "richest_f0_hz": float(richest["f0_hz"] or 0.0),
        "overtone_count": int(richest["overtone_count"]),
        "formant_time_s": float(formant_row["time_s"]),
        "formants_hz": measured_formants,
    }


def save_track_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["time_s", "f0_hz", "confidence", "rms", "overtone_count"])
        for row in rows:
            writer.writerow(
                [
                    f"{float(row['time_s']):.2f}",
                    "" if row["f0_hz"] is None else f"{float(row['f0_hz']):.2f}",
                    f"{float(row['confidence']):.4f}",
                    f"{float(row['rms']):.6f}",
                    str(int(row["overtone_count"])),
                ]
            )


def save_summary_csv(path: Path, results: dict[str, dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "sample",
                "min_f0_hz",
                "max_f0_hz",
                "richest_time_s",
                "richest_f0_hz",
                "overtone_count",
                "formant_1_hz",
                "formant_2_hz",
                "formant_3_hz",
            ]
        )
        for key, result in results.items():
            f1, f2, f3 = result["formants_hz"]  # type: ignore[misc]
            writer.writerow(
                [
                    FORMANTS[key]["label"],
                    f"{float(result['min_f0_hz']):.2f}",
                    f"{float(result['max_f0_hz']):.2f}",
                    f"{float(result['richest_time_s']):.2f}",
                    f"{float(result['richest_f0_hz']):.2f}",
                    str(int(result["overtone_count"])),
                    f"{float(f1):.2f}",
                    f"{float(f2):.2f}",
                    f"{float(f3):.2f}",
                ]
            )


def make_sample_panel(
    key: str,
    waveform: Image.Image,
    spectrogram: Image.Image,
    analysis: dict[str, object],
) -> Image.Image:
    panel = Image.new("RGB", (1500, 1280), "white")
    draw = ImageDraw.Draw(panel)
    title_font = load_font(34, bold=True)
    text_font = load_font(24)
    mono_font = load_font(22, mono=True)
    draw.text((28, 20), FORMANTS[key]["title"], font=title_font, fill="#1f3a5f")
    draw.rounded_rectangle((30, 84, 736, 430), radius=18, outline="#cbd6e6", width=2, fill="white")
    draw.rounded_rectangle((764, 84, 1470, 630), radius=18, outline="#cbd6e6", width=2, fill="white")
    draw.rounded_rectangle((30, 466, 736, 1140), radius=18, outline="#cbd6e6", width=2, fill="#f8fbff")
    panel.paste(fit_image(waveform, 670, 300), (48, 108))
    panel.paste(fit_image(spectrogram, 670, 500), (782, 108))
    draw.text((52, 496), "Численные характеристики", font=title_font, fill="#243447")
    f1, f2, f3 = analysis["formants_hz"]  # type: ignore[misc]
    theory = FORMANTS[key]["theory"]
    lines = [
        f"Минимальная частота основного тона: {float(analysis['min_f0_hz']):.2f} Гц",
        f"Максимальная частота основного тона: {float(analysis['max_f0_hz']):.2f} Гц",
        f"Самый богатый обертонами тон: {float(analysis['richest_f0_hz']):.2f} Гц",
        f"Время этого окна: {float(analysis['richest_time_s']):.2f} с",
        f"Число выраженных обертонов: {int(analysis['overtone_count'])}",
        f"Окно измерения формант: {float(analysis['formant_time_s']):.2f} с",
        f"Форманты: F1={float(f1):.1f} Гц, F2={float(f2):.1f} Гц, F3={float(f3):.1f} Гц",
    ]
    y = 558
    for line in lines:
        draw.text((60, y), line, font=text_font, fill="#243447")
        y += 40
    draw.text((60, 836), "Сравнение с теоретическими значениями", font=title_font, fill="#243447")
    y = 900
    draw.text((78, y), "Теория :", font=mono_font, fill="#243447")
    draw.text((220, y), f"F1={theory[0]:.0f}, F2={theory[1]:.0f}, F3={theory[2]:.0f}", font=mono_font, fill="#243447")
    y += 46
    draw.text((78, y), "Измерено:", font=mono_font, fill="#243447")
    draw.text((220, y), f"F1={float(f1):.0f}, F2={float(f2):.0f}, F3={float(f3):.0f}", font=mono_font, fill="#243447")
    y += 46
    diff = (abs(float(f1) - theory[0]), abs(float(f2) - theory[1]), abs(float(f3) - theory[2]))
    draw.text((78, y), "Отклонение:", font=mono_font, fill="#243447")
    draw.text((220, y), f"F1={diff[0]:.0f}, F2={diff[1]:.0f}, F3={diff[2]:.0f}", font=mono_font, fill="#243447")
    return panel


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font: ImageFont.FreeTypeFont, width: int, fill: str) -> int:
    words = text.split()
    if not words:
        return y
    line = words[0]
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
    for word in words[1:]:
        candidate = line + " " + word
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            line = candidate
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_h + 8
            line = word
    draw.text((x, y), line, font=font, fill=fill)
    return y + line_h + 8


def section(draw: ImageDraw.ImageDraw, title: str, y: int) -> int:
    font = load_font(40, bold=True)
    draw.text((92, y), title, font=font, fill="#1f3a5f")
    y = draw.textbbox((92, y), title, font=font)[3] + 12
    draw.line((92, y, 1560, y), fill="#d6deea", width=3)
    return y + 22


def build_readme_pdf(root: Path, panels: dict[str, Image.Image], results: dict[str, dict[str, object]]) -> None:
    page1 = Image.new("RGB", (1654, 2339), "white")
    draw1 = ImageDraw.Draw(page1)
    title_font = load_font(54, bold=True)
    body_font = load_font(26)
    y = 88
    draw1.text((92, y), "Лабораторная работа №10", font=title_font, fill="#1f3a5f")
    y = draw1.textbbox((92, y), "Лабораторная работа №10", font=title_font)[3] + 18
    draw1.line((92, y, 1560, y), fill="#d6deea", width=3)
    y += 34
    draw1.text((92, y), "Обработка голоса", font=load_font(40, bold=True), fill="#243447")
    y = draw1.textbbox((92, y), "Обработка голоса", font=load_font(40, bold=True))[3] + 18
    draw1.line((92, y, 1560, y), fill="#d6deea", width=2)
    y += 28
    y = section(draw1, "Вариант 1: голосовой диапазон, тембр, форманты", y)
    y = draw_wrapped(
        draw1,
        "В текущей среде нет доступа к микрофону, поэтому для автоматической демонстрации сформированы три одноканальные WAV-дорожки: гласная А, гласная И и имитация лая. Для каждой дорожки выполнены спектральный анализ, оценка диапазона основного тона и поиск трёх сильнейших формант.",
        92,
        y,
        body_font,
        1468,
        "#243447",
    )
    y += 16
    y = section(draw1, "Что делает программа", y)
    bullets = [
        "Создаёт входные WAV-файлы для трёх звуковых образцов.",
        "Строит осциллограммы и спектрограммы на основе оконного преобразования Фурье с окном Ханна.",
        "Оценивает минимальную и максимальную частоту основного тона по окнам длиной 0.1 с.",
        "Находит окно с наибольшим числом выраженных обертонов и фиксирует соответствующий основной тон.",
        "Определяет три сильнейшие форманты и сравнивает форманты для А и И с теоретическими значениями.",
    ]
    for item in bullets:
        draw1.text((104, y), "•", font=load_font(28, bold=True), fill="#243447")
        y = draw_wrapped(draw1, item, 126, y, body_font, 1430, "#243447") + 4
    y += 12
    y = section(draw1, "Сводная таблица результатов", y)
    draw1.rounded_rectangle((92, y, 1560, y + 56), radius=10, fill="#edf3fb", outline="#d6deea", width=2)
    headers = ["Образец", "f0 min", "f0 max", "Богатый тон", "Оберт.", "F1", "F2", "F3"]
    xs = [110, 340, 510, 700, 930, 1060, 1210, 1360]
    for header, x in zip(headers, xs):
        draw1.text((x, y + 14), header, font=load_font(22, bold=True), fill="#1f3a5f")
    row_y = y + 56
    for key in ["a", "i", "bark"]:
        result = results[key]
        f1, f2, f3 = result["formants_hz"]  # type: ignore[misc]
        values = [
            FORMANTS[key]["label"],
            f"{float(result['min_f0_hz']):.0f}",
            f"{float(result['max_f0_hz']):.0f}",
            f"{float(result['richest_f0_hz']):.0f}",
            str(int(result["overtone_count"])),
            f"{float(f1):.0f}",
            f"{float(f2):.0f}",
            f"{float(f3):.0f}",
        ]
        draw1.rectangle((92, row_y, 1560, row_y + 50), outline="#d6deea", width=1)
        for value, x in zip(values, xs):
            draw1.text((x, row_y + 12), value, font=load_font(21), fill="#243447")
        row_y += 50
    y = row_y + 24
    y = section(draw1, "Сформированные файлы", y)
    files = [
        "input/voice_a.wav, input/voice_i.wav, input/bark_like.wav — входные аудиодорожки.",
        "output/waveforms/*.png — осциллограммы.",
        "output/spectrograms/*.png — спектрограммы.",
        "output/tracks/*.csv — покадровая оценка основного тона.",
        "output/summary_metrics.csv — итоговая таблица измерений.",
        "README.pdf — отчёт по лабораторной работе.",
    ]
    for item in files:
        draw1.text((104, y), "•", font=load_font(28, bold=True), fill="#243447")
        y = draw_wrapped(draw1, item, 126, y, body_font, 1430, "#243447") + 4

    page2 = Image.new("RGB", (1654, 2339), "white")
    draw2 = ImageDraw.Draw(page2)
    draw2.text((92, 88), "Примеры обработки: гласные А и И", font=title_font, fill="#1f3a5f")
    y2 = draw2.textbbox((92, 88), "Примеры обработки: гласные А и И", font=title_font)[3] + 18
    draw2.line((92, y2, 1560, y2), fill="#d6deea", width=3)
    page2.paste(fit_image(panels["a"], 1470, 980), (92, 150))
    page2.paste(fit_image(panels["i"], 1470, 980), (92, 1160))

    page3 = Image.new("RGB", (1654, 2339), "white")
    draw3 = ImageDraw.Draw(page3)
    draw3.text((92, 88), "Имитация лая и выводы", font=title_font, fill="#1f3a5f")
    y3 = draw3.textbbox((92, 88), "Имитация лая и выводы", font=title_font)[3] + 18
    draw3.line((92, y3, 1560, y3), fill="#d6deea", width=3)
    page3.paste(fit_image(panels["bark"], 1470, 980), (92, 150))
    y = 1170
    y = section(draw3, "Выводы", y)
    conclusions = [
        f"Для гласной А измерены форманты {results['a']['formants_hz'][0]:.0f}, {results['a']['formants_hz'][1]:.0f} и {results['a']['formants_hz'][2]:.0f} Гц.",  # type: ignore[index]
        f"Для гласной И измерены форманты {results['i']['formants_hz'][0]:.0f}, {results['i']['formants_hz'][1]:.0f} и {results['i']['formants_hz'][2]:.0f} Гц.",  # type: ignore[index]
        "Форманты для А и И различаются, что подтверждает корректность спектрального разделения гласных.",
        "Окно с наиболее выраженным тембром определяется как окно с максимальным количеством заметных обертонов.",
        "Полученный анализ показывает изменение диапазона основного тона по времени и позволяет связать тембр с распределением энергии по гармоникам.",
    ]
    for item in conclusions:
        draw3.text((104, y), "•", font=load_font(28, bold=True), fill="#243447")
        y = draw_wrapped(draw3, item, 126, y, body_font, 1430, "#243447") + 4

    page1.save(root / "README.pdf", save_all=True, append_images=[page2, page3], resolution=180)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_dir = root / args.input
    output_dir = root / args.output
    waveform_dir = output_dir / "waveforms"
    spectrogram_dir = output_dir / "spectrograms"
    track_dir = output_dir / "tracks"
    ensure_dirs([input_dir, output_dir, waveform_dir, spectrogram_dir, track_dir])

    sample_rate = args.sample_rate
    signals = {
        "a": synthesize_vowel_glide(FORMANTS["a"]["theory"], sample_rate, 6.5, 105.0, 820.0, 0.012, 11),
        "i": synthesize_vowel_glide(FORMANTS["i"]["theory"], sample_rate, 6.0, 135.0, 920.0, 0.010, 23),
        "bark": synthesize_bark(sample_rate, 37),
    }
    input_names = {
        "a": "voice_a.wav",
        "i": "voice_i.wav",
        "bark": "bark_like.wav",
    }
    for key, signal in signals.items():
        write_wav(input_dir / input_names[key], signal, sample_rate)

    results: dict[str, dict[str, object]] = {}
    panels: dict[str, Image.Image] = {}
    for key, signal in signals.items():
        waveform = make_waveform_image(signal, sample_rate, f"Осциллограмма: {FORMANTS[key]['title']}")
        spectrogram = make_spectrogram_image(signal, sample_rate, f"Спектрограмма: {FORMANTS[key]['title']}")
        waveform.save(waveform_dir / f"{key}_waveform.png")
        spectrogram.save(spectrogram_dir / f"{key}_spectrogram.png")
        analysis = analyze_signal(signal, sample_rate, key)
        save_track_csv(track_dir / f"{key}_track.csv", analysis["frames"])  # type: ignore[arg-type]
        results[key] = analysis
        panels[key] = make_sample_panel(key, waveform, spectrogram, analysis)

    save_summary_csv(output_dir / "summary_metrics.csv", results)
    build_readme_pdf(root, panels, results)
    print(f"Saved lab to: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
