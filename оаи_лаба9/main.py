import argparse
import csv
import math
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 9: noise analysis")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate")
    parser.add_argument("--frame-size", type=int, default=1024, help="STFT frame size")
    parser.add_argument("--hop-size", type=int, default=256, help="STFT hop size")
    parser.add_argument("--noise-seconds", type=float, default=0.4, help="Leading noise-only section")
    parser.add_argument("--input", default="input", help="Input directory")
    parser.add_argument("--output", default="output", help="Output directory")
    return parser.parse_args()


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / img.width, box_h / img.height)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(signal, -1.0, 1.0)
    data = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data.tobytes())


def synth_note(freq: float, duration: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    attack = 1.0 - np.exp(-40.0 * t)
    decay = np.exp(-3.2 * t)
    envelope = attack * decay
    signal = np.zeros_like(t)
    for harmonic in range(1, 7):
        signal += (1.0 / (harmonic ** 1.2)) * np.sin(2.0 * np.pi * freq * harmonic * t)
    signal += 0.08 * np.sin(2.0 * np.pi * freq * 2.03 * t)
    return 0.55 * envelope * signal


def synth_phrase(sample_rate: int) -> np.ndarray:
    melody = [
        (220.0, 0.34),
        (247.0, 0.34),
        (262.0, 0.34),
        (294.0, 0.34),
        (330.0, 0.40),
        (349.0, 0.34),
        (392.0, 0.38),
        (440.0, 0.46),
    ]
    gap = np.zeros(int(0.045 * sample_rate), dtype=np.float64)
    parts = [np.zeros(int(0.4 * sample_rate), dtype=np.float64)]
    for freq, duration in melody:
        parts.append(synth_note(freq, duration, sample_rate))
        parts.append(gap)
    parts.append(np.zeros(int(0.4 * sample_rate), dtype=np.float64))
    signal = np.concatenate(parts)
    peak = max(1e-9, float(np.max(np.abs(signal))))
    return 0.85 * signal / peak


def add_noise(clean: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    t = np.arange(clean.size, dtype=np.float64) / sample_rate
    white = 0.055 * rng.standard_normal(clean.size)
    hum = 0.028 * np.sin(2.0 * np.pi * 50.0 * t)
    hiss = 0.018 * np.sin(2.0 * np.pi * 3200.0 * t + 0.6)
    noise = white + hum + hiss
    noisy = clean + noise
    peak = max(1.0, float(np.max(np.abs(noisy))) / 0.98)
    return noisy / peak, noise / peak


def stft(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    window = np.hanning(frame_size).astype(np.float64)
    n_frames = 1 + int(math.ceil(max(0, signal.size - frame_size) / hop_size))
    total = (n_frames - 1) * hop_size + frame_size
    padded = np.pad(signal.astype(np.float64), (0, max(0, total - signal.size)))
    frames = []
    for idx in range(n_frames):
        start = idx * hop_size
        frame = padded[start : start + frame_size] * window
        frames.append(np.fft.rfft(frame))
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


def spectral_subtraction(noisy: np.ndarray, sample_rate: int, noise_seconds: float, frame_size: int, hop_size: int) -> np.ndarray:
    noise_len = max(frame_size, int(noise_seconds * sample_rate))
    noise_part = noisy[:noise_len]
    noisy_spec = stft(noisy, frame_size, hop_size)
    noise_spec = stft(noise_part, frame_size, hop_size)
    noise_mag = np.mean(np.abs(noise_spec), axis=1, keepdims=True)
    mag = np.abs(noisy_spec)
    phase = np.exp(1j * np.angle(noisy_spec))
    cleaned_mag = np.maximum(mag - 1.25 * noise_mag, 0.03 * noise_mag)
    cleaned_spec = cleaned_mag * phase
    denoised = istft(cleaned_spec, frame_size, hop_size)
    return denoised[: noisy.size]


def noise_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    error = test[: reference.size] - reference
    signal_power = np.sum(reference * reference)
    noise_power = np.sum(error * error)
    if noise_power <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(signal_power / noise_power))


def spectrogram_db(signal: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = stft(signal, frame_size, hop_size)
    power = np.abs(spec) ** 2
    db = 10.0 * np.log10(np.maximum(power, 1e-12))
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    times = np.arange(spec.shape[1], dtype=np.float64) * hop_size / sample_rate
    return db, freqs, times


def log_frequency_image(db: np.ndarray, freqs: np.ndarray, max_rows: int = 320) -> np.ndarray:
    f_min = max(40.0, float(freqs[1] if freqs.size > 1 else 40.0))
    f_max = float(freqs[-1])
    log_freqs = np.geomspace(f_min, f_max, num=max_rows)
    src = np.empty((max_rows, db.shape[1]), dtype=np.float32)
    for idx, freq in enumerate(log_freqs):
        source = int(np.argmin(np.abs(freqs - freq)))
        src[max_rows - 1 - idx, :] = db[source, :]
    lo = float(np.percentile(src, 5))
    hi = float(np.percentile(src, 99))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((src - lo) / (hi - lo), 0.0, 1.0)
    return norm


def colorize_heatmap(norm: np.ndarray) -> np.ndarray:
    x = norm.astype(np.float32)
    r = np.clip(255.0 * (x ** 0.85), 0, 255)
    g = np.clip(255.0 * np.sqrt(x), 0, 255)
    b = np.clip(255.0 * (0.35 + 0.65 * x), 0, 255)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb.astype(np.uint8)


def make_spectrogram_image(signal: np.ndarray, sample_rate: int, frame_size: int, hop_size: int, title: str) -> Image.Image:
    db, freqs, times = spectrogram_db(signal, sample_rate, frame_size, hop_size)
    heat = colorize_heatmap(log_frequency_image(db, freqs))
    chart = Image.fromarray(heat, mode="RGB").resize((980, 520), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1080, 620), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    draw.text((20, 16), title, font=font_title, fill="black")
    canvas.paste(chart, (60, 56))
    draw.rectangle((60, 56, 1040, 576), outline="#cfcfcf", width=2)
    duration = times[-1] if times.size else 0.0
    for t_val in np.linspace(0.0, duration, 5):
        x = 60 + int(round((t_val / max(duration, 1e-6)) * 980))
        draw.line((x, 576, x, 584), fill="black", width=1)
        draw.text((x - 10, 586), f"{t_val:.1f}", font=font, fill="black")
    for f_val in [100, 250, 500, 1000, 2000, 4000, 8000]:
        if f_val >= sample_rate / 2:
            continue
        ratio = (math.log(f_val) - math.log(40.0)) / (math.log(sample_rate / 2.0) - math.log(40.0))
        y = 576 - int(round(ratio * 520))
        draw.line((52, y, 60, y), fill="black", width=1)
        draw.text((4, y - 8), str(f_val), font=font, fill="black")
    draw.text((1000, 586), "с", font=font, fill="black")
    draw.text((4, 34), "Гц", font=font, fill="black")
    return canvas


def make_waveform_image(signal: np.ndarray, sample_rate: int, title: str) -> Image.Image:
    width = 1080
    height = 320
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    draw.text((20, 16), title, font=font_title, fill="black")
    left = 60
    top = 56
    chart_w = 980
    chart_h = 220
    center_y = top + chart_h // 2
    draw.rectangle((left, top, left + chart_w, top + chart_h), outline="#cfcfcf", width=2)
    draw.line((left, center_y, left + chart_w, center_y), fill="#888888", width=1)
    duration = signal.size / sample_rate
    step = max(1, signal.size // chart_w)
    points = []
    for x in range(chart_w):
        start = x * step
        stop = min(signal.size, start + step)
        value = float(np.mean(signal[start:stop])) if stop > start else 0.0
        y = center_y - int(round(value * (chart_h // 2 - 8)))
        points.append((left + x, y))
    draw.line(points, fill="#1b4d8f", width=1)
    for t_val in np.linspace(0.0, duration, 5):
        x = left + int(round((t_val / max(duration, 1e-6)) * chart_w))
        draw.line((x, top + chart_h, x, top + chart_h + 8), fill="black", width=1)
        draw.text((x - 10, top + chart_h + 10), f"{t_val:.1f}", font=font, fill="black")
    draw.text((left + chart_w + 8, center_y - 8), "0", font=font, fill="black")
    draw.text((left + chart_w + 8, top - 4), "1", font=font, fill="black")
    draw.text((left + chart_w + 8, top + chart_h - 12), "-1", font=font, fill="black")
    return canvas


def high_energy_cells(signal: np.ndarray, sample_rate: int, dt: float = 0.1, df: float = 50.0) -> list[dict[str, float]]:
    window_size = int(round(dt * sample_rate))
    if window_size <= 0:
        return []
    nfft = 1
    while nfft < window_size:
        nfft *= 2
    window = np.hanning(window_size)
    band_edges = np.arange(0.0, sample_rate / 2.0 + df, df)
    freq_grid = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    results = []
    n_frames = max(1, signal.size // window_size)
    for frame_idx in range(n_frames):
        start = frame_idx * window_size
        frame = signal[start : start + window_size]
        if frame.size < window_size:
            frame = np.pad(frame, (0, window_size - frame.size))
        spec = np.abs(np.fft.rfft(frame * window, n=nfft)) ** 2
        for band_idx in range(len(band_edges) - 1):
            f0 = band_edges[band_idx]
            f1 = band_edges[band_idx + 1]
            mask = (freq_grid >= f0) & (freq_grid < f1)
            energy = float(spec[mask].sum())
            results.append(
                {
                    "time_start_s": frame_idx * dt,
                    "time_end_s": (frame_idx + 1) * dt,
                    "freq_start_hz": f0,
                    "freq_end_hz": f1,
                    "energy": energy,
                }
            )
    results.sort(key=lambda item: item["energy"], reverse=True)
    return results[:10]


def save_energy_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["rank", "time_start_s", "time_end_s", "freq_start_hz", "freq_end_hz", "energy"])
        for idx, row in enumerate(rows, 1):
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


def save_metrics_csv(rows: list[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def make_summary_panel(
    waveform_noisy: Image.Image,
    waveform_denoised: Image.Image,
    spec_noisy: Image.Image,
    spec_denoised: Image.Image,
    metrics: list[tuple[str, str]],
    top_energy: list[dict[str, float]],
) -> Image.Image:
    panel = Image.new("RGB", (1480, 1160), "white")
    draw = ImageDraw.Draw(panel)
    title_font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 30)
    text_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
    mono_font = ImageFont.truetype(r"C:\Windows\Fonts\cour.ttf", 18)
    draw.text((24, 18), "Анализ шума и шумоподавление", font=title_font, fill="black")
    panel.paste(fit_image(waveform_noisy, 700, 220), (24, 72))
    panel.paste(fit_image(waveform_denoised, 700, 220), (756, 72))
    panel.paste(fit_image(spec_noisy, 700, 360), (24, 326))
    panel.paste(fit_image(spec_denoised, 700, 360), (756, 326))
    draw.text((24, 706), "Численные характеристики", font=title_font, fill="black")
    y = 754
    for key, value in metrics:
        draw.text((36, y), f"{key}: {value}", font=text_font, fill="black")
        y += 34
    draw.text((756, 706), "Моменты максимальной энергии", font=title_font, fill="black")
    y = 754
    header = "rank | t0 | t1 | f0 | f1 | energy"
    draw.text((768, y), header, font=mono_font, fill="black")
    y += 30
    for idx, row in enumerate(top_energy[:8], 1):
        line = f"{idx:02d} | {row['time_start_s']:.2f} | {row['time_end_s']:.2f} | {row['freq_start_hz']:.0f} | {row['freq_end_hz']:.0f} | {row['energy']:.3f}"
        draw.text((768, y), line, font=mono_font, fill="black")
        y += 28
    return panel


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "spectrograms").mkdir(exist_ok=True)
    (output_dir / "waveforms").mkdir(exist_ok=True)
    (output_dir / "summaries").mkdir(exist_ok=True)

    clean = synth_phrase(args.sample_rate)
    noisy, added_noise = add_noise(clean, args.sample_rate)
    denoised = spectral_subtraction(noisy, args.sample_rate, args.noise_seconds, args.frame_size, args.hop_size)
    denoised = np.clip(denoised, -1.0, 1.0)

    clean_path = input_dir / "instrument_clean.wav"
    noisy_path = input_dir / "instrument_noisy.wav"
    denoised_path = output_dir / "audio" / "instrument_denoised.wav"
    write_wav(clean_path, clean, args.sample_rate)
    write_wav(noisy_path, noisy, args.sample_rate)
    write_wav(denoised_path, denoised, args.sample_rate)

    wf_clean = make_waveform_image(clean, args.sample_rate, "Чистый синтезированный сигнал")
    wf_noisy = make_waveform_image(noisy, args.sample_rate, "Сигнал с шумом")
    wf_denoised = make_waveform_image(denoised, args.sample_rate, "Сигнал после шумоподавления")
    wf_clean.save(output_dir / "waveforms" / "clean_waveform.png")
    wf_noisy.save(output_dir / "waveforms" / "noisy_waveform.png")
    wf_denoised.save(output_dir / "waveforms" / "denoised_waveform.png")

    sp_clean = make_spectrogram_image(clean, args.sample_rate, args.frame_size, args.hop_size, "Спектрограмма чистого сигнала")
    sp_noisy = make_spectrogram_image(noisy, args.sample_rate, args.frame_size, args.hop_size, "Спектрограмма сигнала с шумом")
    sp_denoised = make_spectrogram_image(denoised, args.sample_rate, args.frame_size, args.hop_size, "Спектрограмма после шумоподавления")
    sp_clean.save(output_dir / "spectrograms" / "clean_spectrogram.png")
    sp_noisy.save(output_dir / "spectrograms" / "noisy_spectrogram.png")
    sp_denoised.save(output_dir / "spectrograms" / "denoised_spectrogram.png")

    noise_len = int(args.noise_seconds * args.sample_rate)
    noise_rms_before = noise_rms(noisy[:noise_len])
    noise_rms_after = noise_rms(denoised[:noise_len])
    snr_before = snr_db(clean, noisy)
    snr_after = snr_db(clean, denoised)
    top_energy = high_energy_cells(denoised, args.sample_rate, dt=0.1, df=50.0)
    save_energy_csv(top_energy, output_dir / "top_energy_moments.csv")

    metrics = [
        ("Частота дискретизации", str(args.sample_rate)),
        ("Длительность, с", f"{clean.size / args.sample_rate:.2f}"),
        ("RMS шума до", f"{noise_rms_before:.6f}"),
        ("RMS шума после", f"{noise_rms_after:.6f}"),
        ("SNR до, дБ", f"{snr_before:.2f}"),
        ("SNR после, дБ", f"{snr_after:.2f}"),
        ("Улучшение SNR, дБ", f"{snr_after - snr_before:.2f}"),
    ]
    save_metrics_csv(metrics, output_dir / "metrics.csv")

    summary = make_summary_panel(wf_noisy, wf_denoised, sp_noisy, sp_denoised, metrics, top_energy)
    summary.save(output_dir / "summaries" / "noise_analysis_summary.png")

    print(f"Saved input audio to: {input_dir}")
    print(f"Saved output to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
