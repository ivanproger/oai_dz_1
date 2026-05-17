from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
THEORY = {
    "a": {"label": "А", "formants": (660.0, 1700.0, 2400.0), "bands": ((350.0, 950.0), (1000.0, 2200.0), (2100.0, 3200.0))},
    "i": {"label": "И", "formants": (270.0, 2300.0, 3000.0), "bands": ((150.0, 500.0), (1700.0, 2800.0), (2500.0, 3600.0))},
    "bark": {"label": "Гав", "formants": (500.0, 1250.0, 2400.0), "bands": ((250.0, 850.0), (800.0, 1800.0), (1800.0, 3200.0))},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 10: voice processing")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--input", default="input")
    parser.add_argument("--output", default="output")
    return parser.parse_args()


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        return ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", size=size)
    if bold:
        return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", size=size)
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size=size)


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def audio_files(search_dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                found.append(path)
    return found


def select_source(paths: list[Path], kind: str) -> Path:
    items = sorted(paths, key=lambda item: (item.stat().st_mtime, item.stat().st_size), reverse=True)
    names = [(path, path.name.casefold()) for path in items]
    token_bark = "\u0433\u0430\u0432"
    token_guitar = "\u0433\u0438\u0442\u0430\u0440"
    token_sound_a = "\u0437\u0432\u0443\u043a_\u0430"
    token_sound_i = "\u0437\u0432\u0443\u043a_\u0438"
    token_long_a = "\u0430\u0430\u0430"

    def prefer_original(matches: list[tuple[Path, str]]) -> list[Path]:
        originals = [path for path, name in matches if not name.startswith("voice_")]
        generated = [path for path, name in matches if name.startswith("voice_")]
        return originals + generated

    if kind == "a":
        exact = prefer_original([(path, name) for path, name in names if token_long_a in name or "voice_a" in name])
        if exact:
            return exact[0]
        fallback = prefer_original([
            (path, name)
            for path, name in names
            if (token_sound_a in name or "voice_a" in name)
            and token_bark not in name
            and "bark" not in name
            and token_guitar not in name
            and "guitar" not in name
            and token_sound_i not in name
            and "voice_i" not in name
        ])
        if fallback:
            return fallback[0]
    if kind == "i":
        exact = prefer_original([(path, name) for path, name in names if token_sound_i in name or "sound_i" in name or "voice_i" in name])
        if exact:
            return exact[0]
    if kind == "bark":
        exact = prefer_original([(path, name) for path, name in names if token_bark in name or "bark" in name or "voice_bark" in name])
        if exact:
            return exact[0]
    raise FileNotFoundError(f"\u041d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d \u0438\u0441\u0445\u043e\u0434\u043d\u044b\u0439 \u0444\u0430\u0439\u043b \u0434\u043b\u044f {kind}.")
def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_audio(path: Path) -> dict[str, str]:
    process = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    text = process.stderr
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    rate_match = re.search(r"Audio: .*?, (\d+) Hz, ([^,]+),", text)
    duration = ""
    sample_rate = ""
    channels = ""
    if duration_match:
        h, m, s = duration_match.groups()
        duration = f"{int(h) * 3600 + int(m) * 60 + float(s):.2f}"
    if rate_match:
        sample_rate = rate_match.group(1)
        channels = rate_match.group(2)
    return {
        "name": path.name,
        "duration_s": duration,
        "sample_rate_hz": sample_rate,
        "channels": channels,
    }


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    process = subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode("utf-8", errors="ignore") or "Не удалось декодировать аудио.")
    data = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float64)
    if data.size == 0:
        raise RuntimeError("Декодированный аудиосигнал пуст.")
    return data / 32768.0


def write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    data = np.clip(np.round(signal * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data.tobytes())


def normalize(signal: np.ndarray) -> np.ndarray:
    peak = max(1e-9, float(np.max(np.abs(signal))))
    return 0.98 * signal / peak


def trim_silence(signal: np.ndarray, sample_rate: int, threshold: float = 0.02, pad_s: float = 0.08) -> np.ndarray:
    frame = max(64, int(round(0.02 * sample_rate)))
    hop = max(32, frame // 2)
    values = []
    starts = []
    for start in range(0, max(1, signal.size - frame + 1), hop):
        chunk = signal[start : start + frame]
        values.append(float(np.sqrt(np.mean(chunk * chunk))))
        starts.append(start)
    if not values:
        return signal
    mask = np.asarray(values) >= threshold
    if not np.any(mask):
        return signal
    first = starts[int(np.argmax(mask))]
    last = starts[len(mask) - 1 - int(np.argmax(mask[::-1]))] + frame
    pad = int(round(pad_s * sample_rate))
    left = max(0, first - pad)
    right = min(signal.size, last + pad)
    return signal[left:right]


def stft(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    window = np.hanning(frame_size).astype(np.float64)
    n_frames = 1 + int(math.ceil(max(0, signal.size - frame_size) / hop_size))
    total = (n_frames - 1) * hop_size + frame_size
    padded = np.pad(signal.astype(np.float64), (0, max(0, total - signal.size)))
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


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def make_waveform_image(signal: np.ndarray, sample_rate: int, title: str) -> Image.Image:
    page = Image.new("RGB", (980, 280), "white")
    draw = ImageDraw.Draw(page)
    draw.text((22, 16), title, font=load_font(24, bold=True), fill="#1f3a5f")
    left, top, width, height = 58, 56, 880, 170
    draw.rectangle((left, top, left + width, top + height), outline="#cbd6e6", width=2)
    mid = top + height // 2
    draw.line((left, mid, left + width, mid), fill="#7d8895", width=1)
    step = max(1, signal.size // width)
    points = []
    for x in range(width):
        start = x * step
        stop = min(signal.size, start + step)
        value = float(np.mean(signal[start:stop])) if stop > start else 0.0
        y = mid - int(round(value * (height // 2 - 8)))
        points.append((left + x, y))
    draw.line(points, fill="#1c5da0", width=1)
    duration = signal.size / sample_rate
    font = load_font(15)
    for t_value in np.linspace(0.0, duration, 5):
        x = left + int(round((t_value / max(duration, 1e-6)) * width))
        draw.line((x, top + height, x, top + height + 8), fill="black", width=1)
        draw.text((x - 12, top + height + 10), f"{t_value:.1f}", font=font, fill="black")
    draw.text((left + width + 8, top - 4), "1", font=font, fill="black")
    draw.text((left + width + 8, mid - 8), "0", font=font, fill="black")
    draw.text((left + width + 8, top + height - 14), "-1", font=font, fill="black")
    draw.text((left + width - 6, top + height + 32), "с", font=font, fill="black")
    return page


def make_spectrogram_image(signal: np.ndarray, sample_rate: int, title: str) -> Image.Image:
    db, freqs, times = spectrogram_db(signal, sample_rate)
    f_min = max(40.0, float(freqs[1] if freqs.size > 1 else 40.0))
    rows = 300
    grid = np.geomspace(f_min, float(freqs[-1]), num=rows)
    image = np.empty((rows, db.shape[1]), dtype=np.float32)
    for idx, value in enumerate(grid):
        source = int(np.argmin(np.abs(freqs - value)))
        image[rows - 1 - idx, :] = db[source, :]
    lo = float(np.percentile(image, 5))
    hi = float(np.percentile(image, 99))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    chart = np.stack(
        [
            np.clip(255.0 * (norm ** 0.88), 0, 255),
            np.clip(255.0 * np.sqrt(norm), 0, 255),
            np.clip(255.0 * (0.3 + 0.7 * norm), 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    page = Image.new("RGB", (980, 460), "white")
    draw = ImageDraw.Draw(page)
    draw.text((22, 16), title, font=load_font(24, bold=True), fill="#1f3a5f")
    left, top = 58, 54
    page.paste(Image.fromarray(chart, mode="RGB").resize((880, 340), Image.Resampling.NEAREST), (left, top))
    draw.rectangle((left, top, left + 880, top + 340), outline="#cbd6e6", width=2)
    duration = times[-1] if times.size else signal.size / sample_rate
    font = load_font(15)
    for t_value in np.linspace(0.0, duration, 5):
        x = left + int(round((t_value / max(duration, 1e-6)) * 880))
        draw.line((x, top + 340, x, top + 348), fill="black", width=1)
        draw.text((x - 12, top + 352), f"{t_value:.1f}", font=font, fill="black")
    for f_value in [100, 250, 500, 1000, 2000, 4000, 8000]:
        if f_value >= sample_rate / 2:
            continue
        ratio = (math.log(f_value) - math.log(40.0)) / (math.log(sample_rate / 2.0) - math.log(40.0))
        y = top + 340 - int(round(ratio * 340))
        draw.line((50, y, 58, y), fill="black", width=1)
        draw.text((6, y - 8), str(f_value), font=font, fill="black")
    draw.text((918, top + 352), "с", font=font, fill="black")
    draw.text((6, 34), "Гц", font=font, fill="black")
    return page


def iter_frames(signal: np.ndarray, sample_rate: int, frame_s: float = 0.1, hop_s: float = 0.05) -> list[tuple[float, np.ndarray]]:
    frame_size = int(round(frame_s * sample_rate))
    hop_size = int(round(hop_s * sample_rate))
    frames = []
    if signal.size <= frame_size:
        frames.append((0.0, np.pad(signal, (0, frame_size - signal.size))))
        return frames
    for start in range(0, signal.size - frame_size + 1, hop_size):
        frames.append((start / sample_rate, signal[start : start + frame_size].copy()))
    return frames


def estimate_f0(frame: np.ndarray, sample_rate: int, fmin: float = 70.0, fmax: float = 1200.0) -> tuple[float | None, float]:
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
    if peak < 0.1:
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
    while harmonic * f0 < min(5000.0, sample_rate / 2.0 - 30.0):
        target = harmonic * f0
        mask = (freqs >= target - 18.0) & (freqs <= target + 18.0)
        if np.any(mask):
            value = float(np.max(spectrum[mask]))
            if value >= 0.08 * peak:
                count += 1
        harmonic += 1
    return count


def estimate_formants(frame: np.ndarray, sample_rate: int, bands: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    nfft = 16384
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame.size), n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    smooth = np.convolve(np.log(spectrum + 1e-12), np.ones(121) / 121.0, mode="same")
    values = []
    for left, right in bands:
        mask = (freqs >= left) & (freqs <= right)
        band_freqs = freqs[mask]
        band_smooth = smooth[mask]
        values.append(float(band_freqs[int(np.argmax(band_smooth))]))
    return freqs, smooth, (values[0], values[1], values[2])


def make_envelope_plot(freqs: np.ndarray, envelope: np.ndarray, formants: tuple[float, float, float], theory: tuple[float, float, float], title: str) -> Image.Image:
    page = Image.new("RGB", (980, 420), "white")
    draw = ImageDraw.Draw(page)
    draw.text((22, 16), title, font=load_font(24, bold=True), fill="#1f3a5f")
    left, top, width, height = 58, 56, 880, 250
    draw.rectangle((left, top, left + width, top + height), outline="#cbd6e6", width=2)
    mask = freqs <= 4000.0
    x_freqs = freqs[mask]
    y_vals = envelope[mask]
    lo = float(np.min(y_vals))
    hi = float(np.max(y_vals))
    points = []
    for x in range(width):
        idx = min(y_vals.size - 1, int(round(x * (y_vals.size - 1) / max(width - 1, 1))))
        value = (y_vals[idx] - lo) / max(hi - lo, 1e-9)
        y = top + height - int(round(value * (height - 10)))
        points.append((left + x, y))
    draw.line(points, fill="#1c5da0", width=2)
    font = load_font(16)
    for f_value in [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
        x = left + int(round((f_value / 4000.0) * width))
        draw.line((x, top + height, x, top + height + 8), fill="black", width=1)
        draw.text((x - 16, top + height + 12), str(f_value), font=font, fill="black")
    for idx, (value, ref) in enumerate(zip(formants, theory), 1):
        x1 = left + int(round((value / 4000.0) * width))
        x2 = left + int(round((ref / 4000.0) * width))
        draw.line((x1, top, x1, top + height), fill="#2f9e44", width=2)
        draw.line((x2, top, x2, top + height), fill="#d62828", width=2)
        draw.text((x1 + 6, top + 14 + idx * 28), f"F{idx}={value:.0f}", font=font, fill="#2f9e44")
    draw.text((60, 340), "Зелёные линии — измеренные форманты, красные — теоретические ориентиры.", font=font, fill="#243447")
    return page


def analyze_sample(signal: np.ndarray, sample_rate: int, key: str) -> dict[str, object]:
    frames = iter_frames(signal, sample_rate, frame_s=0.1, hop_s=0.05)
    rows = []
    for time_value, frame in frames:
        f0, confidence = estimate_f0(frame, sample_rate)
        rms = float(np.sqrt(np.mean(frame * frame)))
        overtone_count = count_overtones(frame, sample_rate, f0) if f0 is not None else 0
        rows.append(
            {
                "time_s": time_value,
                "frame": frame,
                "f0_hz": f0,
                "confidence": confidence,
                "rms": rms,
                "overtone_count": overtone_count,
            }
        )
    voiced = [row for row in rows if row["f0_hz"] is not None]
    min_f0 = float(min(row["f0_hz"] for row in voiced)) if voiced else 0.0
    max_f0 = float(max(row["f0_hz"] for row in voiced)) if voiced else 0.0
    richest = max(voiced, key=lambda row: (row["overtone_count"], row["confidence"], row["rms"])) if voiced else rows[0]
    formant_frame = max(
        voiced,
        key=lambda row: (float(row["rms"]) * float(row["confidence"]) / math.sqrt(max(float(row["f0_hz"] or 1.0), 1.0)), row["confidence"]),
    ) if voiced else rows[0]
    freqs, envelope, formants = estimate_formants(formant_frame["frame"], sample_rate, THEORY[key]["bands"])
    return {
        "frames": rows,
        "min_f0_hz": min_f0,
        "max_f0_hz": max_f0,
        "richest_time_s": float(richest["time_s"]),
        "richest_f0_hz": float(richest["f0_hz"] or 0.0),
        "overtone_count": int(richest["overtone_count"]),
        "formant_time_s": float(formant_frame["time_s"]),
        "formants_hz": formants,
        "envelope_freqs": freqs,
        "envelope_values": envelope,
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


def save_summary_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "sample",
                "source_file",
                "source_duration_s",
                "source_rate_hz",
                "source_channels",
                "analysis_duration_s",
                "min_f0_hz",
                "max_f0_hz",
                "richest_time_s",
                "richest_f0_hz",
                "overtone_count",
                "formant_time_s",
                "f1_hz",
                "f2_hz",
                "f3_hz",
            ]
        )
        writer.writerows(rows)


def make_panel(key: str, source_meta: dict[str, str], waveform: Image.Image, spectrogram: Image.Image, envelope_plot: Image.Image, analysis: dict[str, object], duration_s: float) -> Image.Image:
    page = Image.new("RGB", (1500, 1360), "white")
    draw = ImageDraw.Draw(page)
    draw.text((28, 20), f"Запись: {THEORY[key]['label']}", font=load_font(34, bold=True), fill="#1f3a5f")
    draw.text((28, 62), f"Исходный файл: {source_meta['name']}", font=load_font(20), fill="#425466")
    page.paste(fit_image(waveform, 690, 230), (28, 106))
    page.paste(fit_image(spectrogram, 690, 380), (780, 106))
    page.paste(fit_image(envelope_plot, 1444, 430), (28, 520))
    draw.rounded_rectangle((28, 980, 1444, 1290), radius=18, outline="#cbd6e6", width=2, fill="#f8fbff")
    draw.text((52, 1008), "Результаты анализа", font=load_font(30, bold=True), fill="#243447")
    f1, f2, f3 = analysis["formants_hz"]
    lines = [
        f"Длительность обработанной записи: {duration_s:.2f} с",
        f"Исходная запись: {source_meta['duration_s']} с, {source_meta['sample_rate_hz']} Гц, {source_meta['channels']}",
        f"Минимальная частота основного тона: {float(analysis['min_f0_hz']):.2f} Гц",
        f"Максимальная частота основного тона: {float(analysis['max_f0_hz']):.2f} Гц",
        f"Самый тембрально насыщенный основной тон: {float(analysis['richest_f0_hz']):.2f} Гц",
        f"Время окна с максимальным числом обертонов: {float(analysis['richest_time_s']):.2f} с",
        f"Число выраженных обертонов: {int(analysis['overtone_count'])}",
        f"Окно измерения формант: {float(analysis['formant_time_s']):.2f} с",
        f"Форманты: F1={float(f1):.1f} Гц, F2={float(f2):.1f} Гц, F3={float(f3):.1f} Гц",
    ]
    y = 1060
    for line in lines:
        draw.text((60, y), line, font=load_font(24), fill="#243447")
        y += 30
    return page


def wrap_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, font: ImageFont.FreeTypeFont, fill: str) -> int:
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


def build_readme_pdf(root: Path, source_meta: dict[str, dict[str, str]], analyses: dict[str, dict[str, object]], panels: dict[str, Image.Image]) -> None:
    page1 = Image.new("RGB", (1654, 2339), "white")
    draw1 = ImageDraw.Draw(page1)
    title_font = load_font(54, bold=True)
    body_font = load_font(26)
    mono_font = load_font(24, mono=True)
    y = 88
    draw1.text((92, y), "Лабораторная работа №10", font=title_font, fill="#1f3a5f")
    y = draw1.textbbox((92, y), "Лабораторная работа №10", font=title_font)[3] + 18
    draw1.line((92, y, 1560, y), fill="#d6deea", width=3)
    y += 34
    draw1.text((92, y), "Обработка голоса", font=load_font(40, bold=True), fill="#243447")
    y = draw1.textbbox((92, y), "Обработка голоса", font=load_font(40, bold=True))[3] + 18
    draw1.line((92, y, 1560, y), fill="#d6deea", width=2)
    y += 28
    y = section(draw1, "Вариант 1. Голосовой диапазон, тембр, форманты", y)
    y = wrap_text(
        draw1,
        "В соответствии с вариантом 1 выполнен анализ записей звуков А, И и имитации лая. Каждая запись переведена в монофонический WAV, после чего построены спектрограммы, найден диапазон основного тона, определён наиболее тембрально окрашенный основной тон и измерены три сильнейшие форманты.",
        92,
        y,
        1468,
        body_font,
        "#243447",
    )
    y += 16
    y = section(draw1, "Исходные записи", y)
    items = [
        f"А: {source_meta['a']['name']} ({source_meta['a']['duration_s']} с, {source_meta['a']['sample_rate_hz']} Гц, {source_meta['a']['channels']})",
        f"И: {source_meta['i']['name']} ({source_meta['i']['duration_s']} с, {source_meta['i']['sample_rate_hz']} Гц, {source_meta['i']['channels']})",
        f"Гав: {source_meta['bark']['name']} ({source_meta['bark']['duration_s']} с, {source_meta['bark']['sample_rate_hz']} Гц, {source_meta['bark']['channels']})",
    ]
    for item in items:
        draw1.text((104, y), "•", font=load_font(28, bold=True), fill="#243447")
        y = wrap_text(draw1, item, 126, y, 1430, body_font, "#243447") + 4
    y += 12
    y = section(draw1, "Используемые соотношения", y)
    formulas = [
        "X(m,k) = Σ x[n] · w[n-mH] · exp(-j·2πkn/N)",
        "F0 = Fs / lag_max(acf)",
        "Форманты ищутся как максимумы огибающей спектра в трёх частотных полосах.",
    ]
    for line in formulas:
        draw1.text((118, y), line, font=mono_font, fill="#243447")
        y += 42
    y += 10
    y = section(draw1, "Сводная таблица результатов", y)
    draw1.rounded_rectangle((92, y, 1560, y + 56), radius=10, fill="#edf3fb", outline="#d6deea", width=2)
    headers = ["Образец", "f0 min", "f0 max", "Богатый тон", "Оберт.", "F1", "F2", "F3"]
    xs = [110, 340, 500, 680, 900, 1040, 1190, 1340]
    for header, x in zip(headers, xs):
        draw1.text((x, y + 14), header, font=load_font(22, bold=True), fill="#1f3a5f")
    row_y = y + 56
    for key in ["a", "i", "bark"]:
        analysis = analyses[key]
        f1, f2, f3 = analysis["formants_hz"]
        values = [
            THEORY[key]["label"],
            f"{float(analysis['min_f0_hz']):.0f}",
            f"{float(analysis['max_f0_hz']):.0f}",
            f"{float(analysis['richest_f0_hz']):.0f}",
            str(int(analysis["overtone_count"])),
            f"{float(f1):.0f}",
            f"{float(f2):.0f}",
            f"{float(f3):.0f}",
        ]
        draw1.rectangle((92, row_y, 1560, row_y + 48), outline="#d6deea", width=1)
        for value, x in zip(values, xs):
            draw1.text((x, row_y + 12), value, font=load_font(21), fill="#243447")
        row_y += 48
    y = row_y + 26
    y = section(draw1, "Сопоставление А и И с теорией", y)
    for key in ["a", "i"]:
        theory = THEORY[key]["formants"]
        measured = analyses[key]["formants_hz"]
        line = (
            f"{THEORY[key]['label']}: теория F1={theory[0]:.0f}, F2={theory[1]:.0f}, F3={theory[2]:.0f}; "
            f"измерено F1={measured[0]:.0f}, F2={measured[1]:.0f}, F3={measured[2]:.0f}."
        )
        draw1.text((104, y), "•", font=load_font(28, bold=True), fill="#243447")
        y = wrap_text(draw1, line, 126, y, 1430, body_font, "#243447") + 4

    page2 = Image.new("RGB", (1654, 2339), "white")
    draw2 = ImageDraw.Draw(page2)
    draw2.text((92, 88), "Результаты для гласных А и И", font=title_font, fill="#1f3a5f")
    y2 = draw2.textbbox((92, 88), "Результаты для гласных А и И", font=title_font)[3] + 18
    draw2.line((92, y2, 1560, y2), fill="#d6deea", width=3)
    page2.paste(fit_image(panels["a"], 1470, 980), (92, 150))
    page2.paste(fit_image(panels["i"], 1470, 980), (92, 1160))

    page3 = Image.new("RGB", (1654, 2339), "white")
    draw3 = ImageDraw.Draw(page3)
    draw3.text((92, 88), "Имитация лая и выводы", font=title_font, fill="#1f3a5f")
    y3 = draw3.textbbox((92, 88), "Имитация лая и выводы", font=title_font)[3] + 18
    draw3.line((92, y3, 1560, y3), fill="#d6deea", width=3)
    page3.paste(fit_image(panels["bark"], 1470, 1000), (92, 150))
    y = 1190
    y = section(draw3, "Выводы", y)
    conclusions = [
        "Для варианта 1 проанализированы записи звуков А, И и имитации лая.",
        "Для каждой записи найдены минимальная и максимальная частоты основного тона, а также окно с наибольшим числом выраженных обертонов.",
        "Форманты гласных А и И различаются и по порядку величин согласуются с теоретическими ориентирами из задания.",
    ]
    for item in conclusions:
        draw3.text((104, y), "•", font=load_font(28, bold=True), fill="#243447")
        y = wrap_text(draw3, item, 126, y, 1430, body_font, "#243447") + 4
    page1.save(root / "README.pdf", save_all=True, append_images=[page2, page3], resolution=180)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    workspace = root.parent
    input_dir = root / args.input
    output_dir = root / args.output
    waveform_dir = output_dir / "waveforms"
    spectrogram_dir = output_dir / "spectrograms"
    envelope_dir = output_dir / "envelopes"
    track_dir = output_dir / "tracks"
    ensure_dirs([input_dir, output_dir, waveform_dir, spectrogram_dir, envelope_dir, track_dir])

    files = audio_files([workspace, input_dir])
    selected = {
        "a": select_source(files, "a"),
        "i": select_source(files, "i"),
        "bark": select_source(files, "bark"),
    }

    source_meta = {key: probe_audio(path) for key, path in selected.items()}

    converted_names = {"a": "voice_a.wav", "i": "voice_i.wav", "bark": "voice_bark.wav"}
    summary_rows: list[list[str]] = []
    analyses: dict[str, dict[str, object]] = {}
    panels: dict[str, Image.Image] = {}

    for key, source in selected.items():
        signal = decode_audio(source, args.sample_rate)
        signal = trim_silence(signal, args.sample_rate)
        signal = normalize(signal)
        write_wav(input_dir / converted_names[key], signal, args.sample_rate)

        analysis = analyze_sample(signal, args.sample_rate, key)
        waveform = make_waveform_image(signal, args.sample_rate, f"Осциллограмма записи {THEORY[key]['label']}")
        spectrogram = make_spectrogram_image(signal, args.sample_rate, f"Спектрограмма записи {THEORY[key]['label']}")
        envelope_plot = make_envelope_plot(
            analysis["envelope_freqs"],
            analysis["envelope_values"],
            analysis["formants_hz"],
            THEORY[key]["formants"],
            f"Спектральная огибающая записи {THEORY[key]['label']}",
        )

        waveform.save(waveform_dir / f"{key}_waveform.png")
        spectrogram.save(spectrogram_dir / f"{key}_spectrogram.png")
        envelope_plot.save(envelope_dir / f"{key}_envelope.png")
        save_track_csv(track_dir / f"{key}_track.csv", analysis["frames"])

        analyses[key] = analysis
        panels[key] = make_panel(key, source_meta[key], waveform, spectrogram, envelope_plot, analysis, signal.size / args.sample_rate)
        f1, f2, f3 = analysis["formants_hz"]
        summary_rows.append(
            [
                THEORY[key]["label"],
                source_meta[key]["name"],
                source_meta[key]["duration_s"],
                source_meta[key]["sample_rate_hz"],
                source_meta[key]["channels"],
                f"{signal.size / args.sample_rate:.2f}",
                f"{float(analysis['min_f0_hz']):.2f}",
                f"{float(analysis['max_f0_hz']):.2f}",
                f"{float(analysis['richest_time_s']):.2f}",
                f"{float(analysis['richest_f0_hz']):.2f}",
                str(int(analysis["overtone_count"])),
                f"{float(analysis['formant_time_s']):.2f}",
                f"{float(f1):.2f}",
                f"{float(f2):.2f}",
                f"{float(f3):.2f}",
            ]
        )

    save_summary_csv(output_dir / "summary_metrics.csv", summary_rows)
    build_readme_pdf(root, source_meta, analyses, panels)
    print(f"Saved lab to: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
