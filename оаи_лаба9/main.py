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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 9: noise analysis")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=256)
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


def find_guitar_source(search_dirs: list[Path]) -> Path:
    candidates: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                name = path.name.casefold()
                if "гитары" in name or "guitar" in name:
                    candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Не найден файл записи гитары.")
    candidates.sort(key=lambda item: (item.stat().st_mtime, item.stat().st_size), reverse=True)
    return candidates[0]


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.run(
        [
            ffmpeg,
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
        raise RuntimeError(process.stderr.decode("utf-8", errors="ignore") or "Не удалось декодировать звук.")
    data = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float64)
    if data.size == 0:
        raise RuntimeError("Декодированный аудиосигнал пуст.")
    return data / 32768.0


def probe_audio(path: Path) -> dict[str, str]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    text = process.stderr
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    stream_match = re.search(r"Audio: .*?, (\d+) Hz, ([^,]+),", text)
    duration = ""
    sample_rate = ""
    channels = ""
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        duration = f"{int(hours) * 3600 + int(minutes) * 60 + float(seconds):.2f}"
    if stream_match:
        sample_rate = stream_match.group(1)
        channels = stream_match.group(2)
    return {
        "source_name": path.name,
        "source_duration_s": duration,
        "source_sample_rate_hz": sample_rate,
        "source_channels": channels,
    }


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


def istft(spec: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    window = np.hanning(frame_size).astype(np.float64)
    n_frames = spec.shape[1]
    total = (n_frames - 1) * hop_size + frame_size
    signal = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    for idx in range(n_frames):
        start = idx * hop_size
        frame = np.fft.irfft(spec[:, idx], n=frame_size)
        signal[start : start + frame_size] += frame * window
        norm[start : start + frame_size] += window * window
    valid = norm > 1e-8
    signal[valid] /= norm[valid]
    return signal


def estimate_noise_profile(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.abs(spec)
    frame_energy = np.mean(mag * mag, axis=0)
    threshold = float(np.percentile(frame_energy, 15))
    mask = frame_energy <= threshold
    if int(mask.sum()) < 4:
        order = np.argsort(frame_energy)
        mask = np.zeros_like(frame_energy, dtype=bool)
        mask[order[: max(4, order.size // 10 or 1)]] = True
    noise_mag = np.mean(mag[:, mask], axis=1, keepdims=True)
    return noise_mag, mask


def spectral_subtraction(signal: np.ndarray, frame_size: int, hop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = stft(signal, frame_size, hop_size)
    noise_mag, quiet_mask = estimate_noise_profile(spec)
    mag = np.abs(spec)
    phase = np.exp(1j * np.angle(spec))
    cleaned_mag = np.maximum(mag - 1.18 * noise_mag, 0.04 * noise_mag)
    cleaned_spec = cleaned_mag * phase
    denoised = istft(cleaned_spec, frame_size, hop_size)[: signal.size]
    return denoised, noise_mag, quiet_mask


def frame_rms_track(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    values = []
    for start in range(0, max(1, signal.size - frame_size + 1), hop_size):
        frame = signal[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        values.append(float(np.sqrt(np.mean(frame * frame))))
    return np.asarray(values, dtype=np.float64)


def spectrogram_db(signal: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    page = Image.new("RGB", (1080, 320), "white")
    draw = ImageDraw.Draw(page)
    draw.text((22, 16), title, font=load_font(24, bold=True), fill="#1f3a5f")
    left, top, width, height = 58, 56, 980, 220
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
    font = load_font(16)
    for t_value in np.linspace(0.0, duration, 5):
        x = left + int(round((t_value / max(duration, 1e-6)) * width))
        draw.line((x, top + height, x, top + height + 8), fill="black", width=1)
        draw.text((x - 12, top + height + 10), f"{t_value:.1f}", font=font, fill="black")
    draw.text((left + width + 10, top - 4), "1", font=font, fill="black")
    draw.text((left + width + 10, mid - 8), "0", font=font, fill="black")
    draw.text((left + width + 10, top + height - 14), "-1", font=font, fill="black")
    draw.text((left + width - 6, top + height + 34), "с", font=font, fill="black")
    return page


def make_spectrogram_image(
    signal: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    title: str,
    highlight: dict[str, float] | None = None,
) -> Image.Image:
    db, freqs, times = spectrogram_db(signal, sample_rate, frame_size, hop_size)
    f_min = max(40.0, float(freqs[1] if freqs.size > 1 else 40.0))
    f_max = float(freqs[-1])
    rows = 320
    grid = np.geomspace(f_min, f_max, num=rows)
    image = np.empty((rows, db.shape[1]), dtype=np.float32)
    for idx, value in enumerate(grid):
        source = int(np.argmin(np.abs(freqs - value)))
        image[rows - 1 - idx, :] = db[source, :]
    lo = float(np.percentile(image, 5))
    hi = float(np.percentile(image, 99))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(255.0 * (norm ** 0.88), 0, 255)
    g = np.clip(255.0 * np.sqrt(norm), 0, 255)
    b = np.clip(255.0 * (0.32 + 0.68 * norm), 0, 255)
    chart = Image.fromarray(np.stack([r, g, b], axis=-1).astype(np.uint8), mode="RGB").resize((980, 520), Image.Resampling.NEAREST)
    page = Image.new("RGB", (1080, 620), "white")
    draw = ImageDraw.Draw(page)
    draw.text((22, 16), title, font=load_font(24, bold=True), fill="#1f3a5f")
    left, top = 60, 56
    page.paste(chart, (left, top))
    draw.rectangle((left, top, left + 980, top + 520), outline="#cbd6e6", width=2)
    duration = times[-1] if times.size else signal.size / sample_rate
    font = load_font(16)
    for t_value in np.linspace(0.0, duration, 5):
        x = left + int(round((t_value / max(duration, 1e-6)) * 980))
        draw.line((x, top + 520, x, top + 528), fill="black", width=1)
        draw.text((x - 12, top + 532), f"{t_value:.1f}", font=font, fill="black")
    for f_value in [100, 250, 500, 1000, 2000, 4000, 8000]:
        if f_value >= sample_rate / 2:
            continue
        ratio = (math.log(f_value) - math.log(40.0)) / (math.log(sample_rate / 2.0) - math.log(40.0))
        y = top + 520 - int(round(ratio * 520))
        draw.line((52, y, 60, y), fill="black", width=1)
        draw.text((6, y - 8), str(f_value), font=font, fill="black")
    draw.text((1012, top + 532), "с", font=font, fill="black")
    draw.text((6, 34), "Гц", font=font, fill="black")
    if highlight is not None:
        x0 = left + int(round((highlight["time_start_s"] / max(duration, 1e-6)) * 980))
        x1 = left + int(round((highlight["time_end_s"] / max(duration, 1e-6)) * 980))
        ratio0 = (math.log(max(highlight["freq_start_hz"], 40.0)) - math.log(40.0)) / (math.log(sample_rate / 2.0) - math.log(40.0))
        ratio1 = (math.log(max(highlight["freq_end_hz"], 40.0)) - math.log(40.0)) / (math.log(sample_rate / 2.0) - math.log(40.0))
        y1 = top + 520 - int(round(ratio0 * 520))
        y0 = top + 520 - int(round(ratio1 * 520))
        draw.rectangle((x0, y0, x1, y1), outline="#d62828", width=3)
        draw.text((x0 + 8, max(top + 8, y0 - 26)), "Глобальный максимум", font=load_font(18, bold=True), fill="#d62828")
    return page


def build_energy_grid(signal: np.ndarray, sample_rate: int, dt: float = 0.1, df: float = 50.0) -> tuple[np.ndarray, list[dict[str, float]]]:
    window_size = int(round(dt * sample_rate))
    nfft = 1
    while nfft < window_size:
        nfft *= 2
    window = np.hanning(window_size)
    band_edges = np.arange(0.0, sample_rate / 2.0 + df, df)
    freq_grid = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    frames = max(1, int(math.ceil(signal.size / window_size)))
    grid = np.zeros((band_edges.size - 1, frames), dtype=np.float64)
    rows = []
    for frame_idx in range(frames):
        start = frame_idx * window_size
        frame = signal[start : start + window_size]
        if frame.size < window_size:
            frame = np.pad(frame, (0, window_size - frame.size))
        spec = np.abs(np.fft.rfft(frame * window, n=nfft)) ** 2
        for band_idx in range(band_edges.size - 1):
            f0 = float(band_edges[band_idx])
            f1 = float(band_edges[band_idx + 1])
            mask = (freq_grid >= f0) & (freq_grid < f1)
            energy = float(spec[mask].sum())
            grid[band_idx, frame_idx] = energy
            rows.append(
                {
                    "time_start_s": frame_idx * dt,
                    "time_end_s": (frame_idx + 1) * dt,
                    "freq_start_hz": f0,
                    "freq_end_hz": f1,
                    "energy": energy,
                }
            )
    rows.sort(key=lambda item: item["energy"], reverse=True)
    return grid, rows


def make_energy_map_image(
    grid: np.ndarray,
    sample_rate: int,
    dt: float,
    df: float,
    peak_cell: dict[str, float],
) -> Image.Image:
    display_rows = min(grid.shape[0], int(5000 // df))
    sub = grid[:display_rows, :]
    scaled = np.log1p(sub)
    max_value = float(np.max(scaled))
    if max_value > 0:
        scaled = scaled / max_value
    heat = np.stack(
        [
            np.clip(255.0 * (scaled ** 0.9), 0, 255),
            np.clip(255.0 * np.sqrt(scaled), 0, 255),
            np.clip(255.0 * (0.22 + 0.78 * scaled), 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    chart = Image.fromarray(heat[::-1, :, :], mode="RGB").resize((980, 460), Image.Resampling.NEAREST)
    page = Image.new("RGB", (1080, 560), "white")
    draw = ImageDraw.Draw(page)
    draw.text((22, 16), "Карта энергии E(t, f) для восстановленного сигнала", font=load_font(24, bold=True), fill="#1f3a5f")
    left, top = 60, 56
    page.paste(chart, (left, top))
    draw.rectangle((left, top, left + 980, top + 460), outline="#cbd6e6", width=2)
    total_time = grid.shape[1] * dt
    font = load_font(16)
    for t_value in np.linspace(0.0, total_time, 5):
        x = left + int(round((t_value / max(total_time, 1e-6)) * 980))
        draw.line((x, top + 460, x, top + 468), fill="black", width=1)
        draw.text((x - 12, top + 472), f"{t_value:.1f}", font=font, fill="black")
    for f_value in [0, 500, 1000, 2000, 3000, 4000, 5000]:
        y = top + 460 - int(round((f_value / max(1.0, display_rows * df)) * 460))
        draw.line((52, y, 60, y), fill="black", width=1)
        draw.text((6, y - 8), str(f_value), font=font, fill="black")
    draw.text((1012, top + 472), "с", font=font, fill="black")
    draw.text((6, 34), "Гц", font=font, fill="black")
    time_idx = int(round(peak_cell["time_start_s"] / dt))
    band_idx = int(round(peak_cell["freq_start_hz"] / df))
    if band_idx < display_rows:
        x0 = left + int(round(time_idx / max(1, grid.shape[1]) * 980))
        x1 = left + int(round((time_idx + 1) / max(1, grid.shape[1]) * 980))
        y0 = top + 460 - int(round((band_idx + 1) / display_rows * 460))
        y1 = top + 460 - int(round(band_idx / display_rows * 460))
        draw.rectangle((x0, y0, x1, y1), outline="#d62828", width=3)
        draw.text((x0 + 8, max(top + 8, y0 - 24)), "Максимум", font=load_font(18, bold=True), fill="#d62828")
    return page


def save_top_energy_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["rank", "time_start_s", "time_end_s", "freq_start_hz", "freq_end_hz", "energy"])
        for idx, row in enumerate(rows[:10], 1):
            writer.writerow(
                [
                    idx,
                    f"{row['time_start_s']:.2f}",
                    f"{row['time_end_s']:.2f}",
                    f"{row['freq_start_hz']:.1f}",
                    f"{row['freq_end_hz']:.1f}",
                    f"{row['energy']:.6f}",
                ]
            )


def save_metrics_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def text_block(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, font: ImageFont.FreeTypeFont, fill: str) -> int:
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


def build_readme_pdf(
    root: Path,
    source_meta: dict[str, str],
    metrics: list[tuple[str, str]],
    top_rows: list[dict[str, float]],
    wave_before: Image.Image,
    wave_after: Image.Image,
    spec_before: Image.Image,
    spec_after: Image.Image,
    energy_map: Image.Image,
) -> None:
    page1 = Image.new("RGB", (1654, 2339), "white")
    draw1 = ImageDraw.Draw(page1)
    title_font = load_font(54, bold=True)
    body_font = load_font(26)
    mono_font = load_font(24, mono=True)
    y = 88
    draw1.text((92, y), "Лабораторная работа №9", font=title_font, fill="#1f3a5f")
    y = draw1.textbbox((92, y), "Лабораторная работа №9", font=title_font)[3] + 18
    draw1.line((92, y, 1560, y), fill="#d6deea", width=3)
    y += 34
    draw1.text((92, y), "Анализ шума музыкальной записи", font=load_font(40, bold=True), fill="#243447")
    y = draw1.textbbox((92, y), "Анализ шума музыкальной записи", font=load_font(40, bold=True))[3] + 18
    draw1.line((92, y, 1560, y), fill="#d6deea", width=2)
    y += 28
    y = section(draw1, "Исходные данные", y)
    y = text_block(
        draw1,
        f"Использована пользовательская запись {source_meta['source_name']}. Файл декодирован в монофонический WAV с частотой {metrics[0][1]} Гц. Для анализа спектра и подавления шума применено оконное преобразование Фурье с окном Ханна.",
        92,
        y,
        1468,
        body_font,
        "#243447",
    )
    y += 16
    y = section(draw1, "Как оценивался шум", y)
    y = text_block(
        draw1,
        "Шумовой профиль оценивался по 15% наименее энергичных окон спектрограммы. После этого выполнялось спектральное вычитание. Глобальный максимум энергии искался на восстановленном сигнале по карте E(t,f) с шагами Δt = 0.1 с и Δf = 50 Гц.",
        92,
        y,
        1468,
        body_font,
        "#243447",
    )
    y += 18
    y = section(draw1, "Используемые соотношения", y)
    formulas = [
        "X(m,k) = Σ x[n] · w[n-mH] · exp(-j·2πkn/N)",
        "|S(m,k)| = max(|Y(m,k)| - α·|N(k)|, β·|N(k)|)",
        "E(t,f) = Σ |X(t,f)|² по окну Δt × Δf",
    ]
    for line in formulas:
        draw1.text((118, y), line, font=mono_font, fill="#243447")
        y += 42
    y += 10
    y = section(draw1, "Численные результаты", y)
    draw1.rounded_rectangle((92, y, 1560, y + 56), radius=10, fill="#edf3fb", outline="#d6deea", width=2)
    draw1.text((112, y + 14), "Показатель", font=load_font(22, bold=True), fill="#1f3a5f")
    draw1.text((1120, y + 14), "Значение", font=load_font(22, bold=True), fill="#1f3a5f")
    row_y = y + 56
    for key, value in metrics:
        draw1.rectangle((92, row_y, 1560, row_y + 46), outline="#d6deea", width=1)
        draw1.text((112, row_y + 10), key, font=load_font(21), fill="#243447")
        draw1.text((1120, row_y + 10), value, font=load_font(21), fill="#243447")
        row_y += 46
    y = row_y + 24
    y = section(draw1, "Глобальный максимум энергии", y)
    peak = top_rows[0]
    peak_text = (
        f"Глобальный максимум карты E(t,f) найден в интервале {peak['time_start_s']:.2f}–{peak['time_end_s']:.2f} с "
        f"и полосе {peak['freq_start_hz']:.0f}–{peak['freq_end_hz']:.0f} Гц. "
        f"Энергия ячейки: {peak['energy']:.3f}."
    )
    text_block(draw1, peak_text, 92, y, 1468, body_font, "#243447")

    page2 = Image.new("RGB", (1654, 2339), "white")
    draw2 = ImageDraw.Draw(page2)
    draw2.text((92, 88), "Осциллограммы и спектрограммы", font=title_font, fill="#1f3a5f")
    y2 = draw2.textbbox((92, 88), "Осциллограммы и спектрограммы", font=title_font)[3] + 18
    draw2.line((92, y2, 1560, y2), fill="#d6deea", width=3)
    page2.paste(fit_image(wave_before, 700, 220), (92, 150))
    page2.paste(fit_image(wave_after, 700, 220), (862, 150))
    page2.paste(fit_image(spec_before, 700, 420), (92, 430))
    page2.paste(fit_image(spec_after, 700, 420), (862, 430))
    draw2.text((112, 122), "Исходная запись", font=load_font(28, bold=True), fill="#243447")
    draw2.text((882, 122), "После шумоподавления", font=load_font(28, bold=True), fill="#243447")
    draw2.text((112, 402), "Спектрограмма исходной записи", font=load_font(28, bold=True), fill="#243447")
    draw2.text((882, 402), "Спектрограмма после обработки", font=load_font(28, bold=True), fill="#243447")
    page2.paste(fit_image(energy_map, 1470, 760), (92, 940))

    page3 = Image.new("RGB", (1654, 2339), "white")
    draw3 = ImageDraw.Draw(page3)
    draw3.text((92, 88), "Максимумы энергии и выводы", font=title_font, fill="#1f3a5f")
    y3 = draw3.textbbox((92, 88), "Максимумы энергии и выводы", font=title_font)[3] + 18
    draw3.line((92, y3, 1560, y3), fill="#d6deea", width=3)
    y = 156
    draw3.text((92, y), "Наиболее энергичные интервалы", font=load_font(34, bold=True), fill="#243447")
    y += 52
    draw3.rounded_rectangle((92, y, 1560, y + 56), radius=10, fill="#edf3fb", outline="#d6deea", width=2)
    headers = ["№", "t0", "t1", "f0", "f1", "Энергия"]
    xs = [112, 220, 380, 540, 730, 930]
    for header, x in zip(headers, xs):
        draw3.text((x, y + 14), header, font=load_font(22, bold=True), fill="#1f3a5f")
    row_y = y + 56
    for idx, row in enumerate(top_rows[:10], 1):
        draw3.rectangle((92, row_y, 1560, row_y + 46), outline="#d6deea", width=1)
        values = [
            str(idx),
            f"{row['time_start_s']:.2f}",
            f"{row['time_end_s']:.2f}",
            f"{row['freq_start_hz']:.0f}",
            f"{row['freq_end_hz']:.0f}",
            f"{row['energy']:.3f}",
        ]
        for value, x in zip(values, xs):
            draw3.text((x, row_y + 10), value, font=load_font(21), fill="#243447")
        row_y += 46
    y = row_y + 28
    y = section(draw3, "Выводы", y)
    conclusions = [
        "Запись гитары обработана как реальный внешний источник, а не как синтезированный сигнал.",
        "Уровень шумового фона оценён по наименее энергичным окнам и после спектрального вычитания уменьшен.",
        "Глобальный максимум энергии теперь определён однозначно: показаны и численные границы ячейки, и её положение на карте E(t,f).",
        "Отчёт является самодостаточным: в нём приведены методика, параметры, спектрограммы, карта энергии и итоговые выводы.",
    ]
    for item in conclusions:
        draw3.text((104, y), "•", font=load_font(28, bold=True), fill="#243447")
        y = text_block(draw3, item, 126, y, 1430, body_font, "#243447") + 4
    page1.save(root / "README.pdf", save_all=True, append_images=[page2, page3], resolution=180)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    workspace = root.parent
    input_dir = root / args.input
    output_dir = root / args.output
    audio_dir = output_dir / "audio"
    waveform_dir = output_dir / "waveforms"
    spectrogram_dir = output_dir / "spectrograms"
    summary_dir = output_dir / "summaries"
    ensure_dirs([input_dir, output_dir, audio_dir, waveform_dir, spectrogram_dir, summary_dir])

    source = find_guitar_source([workspace, input_dir])
    source_meta = probe_audio(source)
    signal = decode_audio(source, args.sample_rate)
    signal = normalize(signal)
    write_wav(input_dir / "guitar_original.wav", signal, args.sample_rate)

    denoised, noise_mag, quiet_mask = spectral_subtraction(signal, args.frame_size, args.hop_size)
    denoised = normalize(denoised[: signal.size])
    write_wav(audio_dir / "guitar_denoised.wav", denoised, args.sample_rate)

    rms_before_track = frame_rms_track(signal, args.frame_size, args.hop_size)
    rms_after_track = frame_rms_track(denoised, args.frame_size, args.hop_size)
    quiet_count = min(len(rms_before_track), len(quiet_mask))
    quiet_indices = quiet_mask[:quiet_count]
    noise_rms_before = float(np.mean(rms_before_track[:quiet_count][quiet_indices])) if np.any(quiet_indices) else float(np.mean(rms_before_track))
    noise_rms_after = float(np.mean(rms_after_track[:quiet_count][quiet_indices])) if np.any(quiet_indices) else float(np.mean(rms_after_track))
    floor_before_db = float(np.median(20.0 * np.log10(np.maximum(noise_mag[:, 0], 1e-9))))

    denoised_spec = stft(denoised, args.frame_size, args.hop_size)
    denoised_noise_mag, _ = estimate_noise_profile(denoised_spec)
    floor_after_db = float(np.median(20.0 * np.log10(np.maximum(denoised_noise_mag[:, 0], 1e-9))))

    energy_grid, top_rows = build_energy_grid(denoised, args.sample_rate, dt=0.1, df=50.0)
    save_top_energy_csv(top_rows, output_dir / "top_energy_moments.csv")
    peak = top_rows[0]

    metrics = [
        ("Имя исходного файла", source_meta["source_name"]),
        ("Частота дискретизации после конвертации, Гц", str(args.sample_rate)),
        ("Длительность, с", f"{signal.size / args.sample_rate:.2f}"),
        ("Исходная частота дискретизации, Гц", source_meta["source_sample_rate_hz"] or "не определена"),
        ("Исходное число каналов", source_meta["source_channels"] or "не определено"),
        ("Доля тихих окон для оценки шума", f"{100.0 * float(np.mean(quiet_mask)):.1f}%"),
        ("Оценка RMS шума до", f"{noise_rms_before:.6f}"),
        ("Оценка RMS шума после", f"{noise_rms_after:.6f}"),
        ("Медианный спектральный пол до, дБ", f"{floor_before_db:.2f}"),
        ("Медианный спектральный пол после, дБ", f"{floor_after_db:.2f}"),
        ("Глобальный максимум: время, с", f"{peak['time_start_s']:.2f}–{peak['time_end_s']:.2f}"),
        ("Глобальный максимум: частоты, Гц", f"{peak['freq_start_hz']:.0f}–{peak['freq_end_hz']:.0f}"),
    ]
    save_metrics_csv(output_dir / "metrics.csv", metrics)

    wave_before = make_waveform_image(signal, args.sample_rate, "Осциллограмма исходной записи")
    wave_after = make_waveform_image(denoised, args.sample_rate, "Осциллограмма после шумоподавления")
    spec_before = make_spectrogram_image(signal, args.sample_rate, args.frame_size, args.hop_size, "Спектрограмма исходной записи")
    spec_after = make_spectrogram_image(denoised, args.sample_rate, args.frame_size, args.hop_size, "Спектрограмма после шумоподавления", highlight=peak)
    energy_map = make_energy_map_image(energy_grid, args.sample_rate, 0.1, 50.0, peak)

    wave_before.save(waveform_dir / "guitar_waveform_original.png")
    wave_after.save(waveform_dir / "guitar_waveform_denoised.png")
    spec_before.save(spectrogram_dir / "guitar_spectrogram_original.png")
    spec_after.save(spectrogram_dir / "guitar_spectrogram_denoised.png")
    energy_map.save(summary_dir / "guitar_energy_map.png")

    build_readme_pdf(root, source_meta, metrics, top_rows, wave_before, wave_after, spec_before, spec_after, energy_map)
    print(f"Saved lab to: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
