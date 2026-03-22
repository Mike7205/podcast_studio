"""
Pod Tools – Audio Studio
Streamlit app for recording, uploading, and editing audio.
Optimised for FSDZMIC S338 USB microphone.
"""

import io
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import librosa.effects
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import requests
import soundfile as sf
import streamlit as st
from pydub import AudioSegment
from scipy.signal import butter, filtfilt

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pod Tools – Audio Studio",
    page_icon="🎙️",
    layout="wide",
)

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in {
    "recorded_audio":  None,   # (np.ndarray, int)
    "uploaded_audio":  None,   # (np.ndarray, int)
    "processed_audio": None,   # (np.ndarray, int)
    "mic_key":         0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Audio helpers ────────────────────────────────────────────────────────────

def wav_bytes_to_numpy(raw: bytes) -> tuple[np.ndarray, int]:
    """Read WAV bytes → mono float32 via soundfile (handles 16/24/32-bit)."""
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, int(sr)


def any_to_wav_bytes(raw: bytes, suffix: str = ".mp3") -> bytes:
    """Convert any audio format to WAV bytes via pydub/ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        seg = AudioSegment.from_file(tmp).set_channels(1)
    finally:
        os.unlink(tmp)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


def numpy_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def numpy_to_mp3_bytes(y: np.ndarray, sr: int) -> bytes:
    wav = numpy_to_wav_bytes(y, sr)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
        tmp = f.name
    try:
        seg = AudioSegment.from_wav(tmp)
    finally:
        os.unlink(tmp)
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate="192k")
    return buf.getvalue()


def encode_for_download(y: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    wav = numpy_to_wav_bytes(y, sr)
    if fmt == "WAV":
        return wav, "audio/wav"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
        tmp = f.name
    try:
        seg = AudioSegment.from_wav(tmp)
    finally:
        os.unlink(tmp)
    buf = io.BytesIO()
    if fmt == "MP4":
        seg.export(buf, format="mp4", codec="aac")
        return buf.getvalue(), "audio/mp4"
    seg.export(buf, format=fmt.lower())
    return buf.getvalue(), {"MP3": "audio/mpeg", "FLAC": "audio/flac"}[fmt]


# ─── Waveform plot ────────────────────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int, title: str = "",
                  start_s: float = 0.0, end_s: float | None = None,
                  color: str = "#1db954") -> plt.Figure:
    dur = len(y) / sr
    if end_s is None:
        end_s = dur

    n_bars = 600
    step   = max(1, len(y) // n_bars)
    peaks  = np.array([np.max(np.abs(y[i: i + step]))
                       for i in range(0, len(y) - step, step)])
    t      = np.linspace(0, dur, len(peaks))

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.bar(t,  peaks, width=dur / len(peaks) * 0.85, color=color, alpha=0.9)
    ax.bar(t, -peaks, width=dur / len(peaks) * 0.85, color=color, alpha=0.9)
    ax.axvline(start_s, color="#ff4b4b", linewidth=1.5, label=f"Start {start_s:.2f}s")
    ax.axvline(end_s,   color="#ffa64b", linewidth=1.5, label=f"End   {end_s:.2f}s")
    ax.set_xlim(0, dur)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", color="white", fontsize=9)
    ax.set_ylabel("Amplitude", color="white", fontsize=9)
    if title:
        ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a2a")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    fig.tight_layout(pad=0.4)
    return fig


def show_player(y: np.ndarray, sr: int, title: str = "",
                start_s: float = 0.0, end_s: float | None = None,
                color: str = "#1db954") -> None:
    """Show waveform + st.audio player."""
    fig = plot_waveform(y, sr, title, start_s, end_s, color=color)
    st.pyplot(fig)
    plt.close(fig)
    st.audio(numpy_to_wav_bytes(y, sr), format="audio/wav")


# ─── Processing ───────────────────────────────────────────────────────────────

def apply_processing(y: np.ndarray, sr: int,
                     noise_prop: float, stationary: bool,
                     gain_db: float, hp_cutoff: int,
                     vocal_clarity: float,
                     pitch_steps: int, speed: float) -> np.ndarray:
    out = y.copy()

    if hp_cutoff > 0:
        b, a = butter(4, hp_cutoff / (sr / 2), btype="high")
        out  = filtfilt(b, a, out).astype(np.float32)

    if noise_prop > 0:
        out = nr.reduce_noise(y=out, sr=sr,
                              prop_decrease=noise_prop,
                              stationary=stationary).astype(np.float32)

    if vocal_clarity > 0:
        lo, hi = 200, 4000
        b2, a2 = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        mid = filtfilt(b2, a2, out).astype(np.float32)
        out = ((1 - vocal_clarity) * out + vocal_clarity * mid).astype(np.float32)

    if gain_db != 0:
        out = np.clip(out * (10 ** (gain_db / 20)), -1.0, 1.0).astype(np.float32)

    if pitch_steps != 0:
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=float(pitch_steps))

    if speed != 1.0:
        out = librosa.effects.time_stretch(out, rate=speed)

    return out.astype(np.float32)


# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🎙️  Pod Tools – Audio Studio")

tab_rec, tab_upload, tab_edit = st.tabs(
    ["⏺  Record", "⬆️  Upload", "✂️  Edit & Export"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – RECORD
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.header("Record New Audio")
    st.caption("Uses your browser's microphone — works with FSDZMIC S338 and any other device.")

    if st.button("🔄  Reset microphone", help="Click if the recorder shows an error"):
        st.session_state.mic_key += 1
        st.rerun()

    audio_input = st.audio_input("🎤  Click to record",
                                 key=f"mic_{st.session_state.mic_key}")

    if audio_input is not None:
        raw_wav = audio_input.read()

        with st.spinner("Loading…"):
            y, sr = wav_bytes_to_numpy(raw_wav)

        st.session_state.recorded_audio = (y, sr)
        st.success(f"Recorded  {len(y)/sr:.1f}s  |  {sr} Hz")

        # Waveform + playback (raw WAV — no conversion needed)
        show_player(y, sr, "Recording preview")

        st.divider()
        fname = st.text_input("Save filename", value="recording.mp4")
        dl_bytes, dl_mime = encode_for_download(y, sr, "MP4")
        dl_name = str(Path(fname).with_suffix(".mp4"))
        st.download_button("💾  Save as MP4", dl_bytes, dl_name, dl_mime, key="dl_rec")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.header("Upload Audio File")

    src = st.radio("Source", ["Local file", "URL / YouTube"], horizontal=True)

    if src == "Local file":
        uploaded = st.file_uploader(
            "Choose a file",
            type=["wav", "mp3", "mp4", "m4a", "ogg", "flac", "aac"],
        )
        if uploaded and st.button("Load file"):
            with st.spinner("Loading…"):
                raw = uploaded.read()
                ext = Path(uploaded.name).suffix.lower()
                if ext == ".wav":
                    y, sr = wav_bytes_to_numpy(raw)
                else:
                    wav = any_to_wav_bytes(raw, suffix=ext)
                    y, sr = wav_bytes_to_numpy(wav)
            st.session_state.uploaded_audio = (y, sr)
            st.success(f"Loaded: {uploaded.name}  |  {len(y)/sr:.1f}s  @  {sr} Hz")

    else:
        url = st.text_input("Paste a YouTube / SoundCloud / direct URL")
        if url and st.button("Download & Load"):
            with st.spinner("Downloading…"):
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        out_tmpl = os.path.join(tmp, "audio.%(ext)s")
                        res = subprocess.run(
                            ["yt-dlp", "-x", "--audio-format", "wav",
                             "-o", out_tmpl, url],
                            capture_output=True, text=True, timeout=180,
                        )
                        wav_files = list(Path(tmp).glob("*.wav"))
                        if not wav_files:
                            raise RuntimeError(res.stderr[:300] or "No output file")
                        y, sr = wav_bytes_to_numpy(wav_files[0].read_bytes())
                        st.session_state.uploaded_audio = (y, sr)
                        st.success(f"Downloaded  |  {len(y)/sr:.1f}s  @  {sr} Hz")
                except Exception as e_yt:
                    try:
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        ext = "." + (url.split(".")[-1].split("?")[0] or "mp3")
                        wav = any_to_wav_bytes(r.content, suffix=ext)
                        y, sr = wav_bytes_to_numpy(wav)
                        st.session_state.uploaded_audio = (y, sr)
                        st.success(f"Downloaded  |  {len(y)/sr:.1f}s  @  {sr} Hz")
                    except Exception as e_http:
                        st.error(f"yt-dlp: {e_yt}\nHTTP: {e_http}")

    if st.session_state.uploaded_audio is not None:
        y, sr = st.session_state.uploaded_audio
        show_player(y, sr, "Uploaded file")
        st.caption(f"Duration: {len(y)/sr:.1f}s  |  {sr} Hz  →  go to **Edit & Export**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – EDIT & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.header("Edit & Export")

    # Source selector
    sources: dict[str, tuple[np.ndarray, int]] = {}
    if st.session_state.recorded_audio is not None:
        sources["Last recording"] = st.session_state.recorded_audio
    if st.session_state.uploaded_audio is not None:
        sources["Uploaded file"]  = st.session_state.uploaded_audio

    if not sources:
        st.info("No audio available. Use the **Record** or **Upload** tab first.")
        st.stop()

    chosen_src = st.radio("Audio source", list(sources.keys()), horizontal=True)
    y_orig, sr = sources[chosen_src]
    dur = len(y_orig) / sr

    st.divider()

    # Waveform player
    st.subheader("Waveform")
    if st.session_state.processed_audio is not None:
        y_proc, _ = st.session_state.processed_audio
        st.markdown("<span style='font-size:18px; color:#4a9eff; font-weight:600'>Original</span>", unsafe_allow_html=True)
        show_player(y_orig, sr, color="#1db954")
        st.markdown("<span style='font-size:18px; color:#ff4b4b; font-weight:600'>Processed</span>", unsafe_allow_html=True)
        show_player(y_proc, sr, color="#4a9eff")
        y_work = y_proc
    else:
        show_player(y_orig, sr, "Working audio")
        y_work = y_orig

    st.divider()

    # Processing
    st.subheader("Processing")

    with st.form("processing_form"):
        with st.expander("🔇  Noise Reduction", expanded=True):
            c1, c2, c3 = st.columns(3)
            noise_prop = c1.slider("Noise reduction", 0.0, 1.0, 0.5, 0.05)
            gain_db    = c2.slider("Gain (dB)", -20, 40, 0)
            stationary = c3.checkbox("Stationary noise", value=True,
                                     help="Best for constant hum/hiss")

        with st.expander("🎙️  Vocal Enhancement", expanded=True):
            c1, c2 = st.columns(2)
            vocal_clarity = c1.slider("Vocal clarity", 0.0, 1.0, 0.0, 0.05,
                                      help="Boosts 200–4000 Hz voice range")
            hp_cutoff     = c2.slider("Low-cut filter (Hz)", 0, 500, 80, 10,
                                      help="Removes rumble below this frequency")

        with st.expander("🎚️  Voice Modulation", expanded=True):
            c1, c2 = st.columns(2)
            pitch_steps = c1.slider("Pitch shift (semitones)", -12, 12, 0,
                                    help="±2 subtle · ±12 = one octave")
            speed       = c2.slider("Speed ×", 0.5, 2.0, 1.0, 0.05,
                                    help="< 1 slower · > 1 faster")

        fc1, fc2 = st.columns(2)
        submitted = fc1.form_submit_button("▶  Apply & compare",
                                           use_container_width=True,
                                           type="primary")
        reset     = fc2.form_submit_button("↩  Reset",
                                           use_container_width=True)

    if submitted:
        with st.spinner("Processing…"):
            y_proc = apply_processing(
                y_orig, sr,
                noise_prop, stationary, gain_db, hp_cutoff,
                vocal_clarity, pitch_steps, speed,
            )
            st.session_state.processed_audio = (y_proc, sr)
        st.rerun()

    if reset:
        st.session_state.processed_audio = None
        st.rerun()

    st.divider()

    # Split & Save
    st.subheader("Split & Save")
    c1, c2 = st.columns(2)
    split_start = c1.number_input("Segment start (s)", 0.0, float(dur), 0.0, 0.1)
    split_end   = c2.number_input("Segment end (s)",   0.0, float(dur), float(dur), 0.1)

    s1 = max(0, int(split_start * sr))
    s2 = min(len(y_work), int(split_end * sr))
    segment = y_work[s1:s2]

    if len(segment) > 0:
        show_player(segment, sr,
                    f"Segment  {split_start:.2f}s → {split_end:.2f}s",
                    0.0, split_end - split_start)
        c1, c2 = st.columns(2)
        seg_name = c1.text_input("Segment filename", "segment.mp4")
        seg_fmt  = c2.selectbox("Format", ["MP4", "MP3", "WAV", "FLAC"], key="seg_fmt")
        seg_bytes, seg_mime = encode_for_download(segment, sr, seg_fmt)
        seg_fname = str(Path(seg_name).with_suffix("." + seg_fmt.lower()))
        st.download_button("💾  Save segment", seg_bytes, seg_fname, seg_mime, key="dl_seg")
    else:
        st.warning("Segment is empty – adjust start / end times.")

    st.divider()

    # Save full
    st.subheader("Save Full Audio")
    c1, c2 = st.columns(2)
    full_name = c1.text_input("Output filename", "output.mp4")
    full_fmt  = c2.selectbox("Format", ["MP4", "MP3", "WAV", "FLAC"], key="full_fmt")
    full_bytes, full_mime = encode_for_download(y_work, sr, full_fmt)
    full_fname = str(Path(full_name).with_suffix("." + full_fmt.lower()))
    st.download_button("💾  Save full audio", full_bytes, full_fname, full_mime, key="dl_full")
