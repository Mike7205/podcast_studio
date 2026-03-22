"""
Pod Tools – Audio Studio
Streamlit app for recording, uploading, and editing audio.
Optimised for FSDZMIC S338 USB microphone.
"""

import base64
import io
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import requests
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except OSError:
    SOUNDDEVICE_OK = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pod Tools – Audio Studio",
    page_icon="🎙️",
    layout="wide",
)

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in {
    "recorded_audio":  None,   # (np.ndarray, int) – from Record tab
    "uploaded_audio":  None,   # (np.ndarray, int) – from Upload tab
    "processed_audio": None,   # (np.ndarray, int) – after noise reduction
    "mic_key":         0,      # incremented to force audio_input re-init
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_audio_bytes(raw: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    ext = Path(filename).suffix.lower() if filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=None, mono=True)
    finally:
        os.unlink(tmp)
    return y.astype(np.float32), int(sr)


def to_mp3_bytes(y: np.ndarray, sr: int, bitrate: str = "192k") -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        seg = AudioSegment.from_wav(tmp.name)
        os.unlink(tmp.name)
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


def encode_audio(y: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    """Encode numpy audio to bytes in the requested format."""
    buf = io.BytesIO()
    if fmt == "WAV":
        sf.write(buf, y, sr, format="WAV")
        return buf.getvalue(), "audio/wav"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        seg = AudioSegment.from_wav(tmp.name)
        os.unlink(tmp.name)
    out = io.BytesIO()
    if fmt == "MP4":
        seg.export(out, format="mp4", codec="aac")
        return out.getvalue(), "audio/mp4"
    seg.export(out, format=fmt.lower())
    mime = {"MP3": "audio/mpeg", "FLAC": "audio/flac"}.get(fmt, "audio/octet-stream")
    return out.getvalue(), mime


def wavesurfer_player(mp3_bytes: bytes, label: str = "", color: str = "#1db954",
                      progress_color: str = "#ff4b4b", height: int = 100) -> None:
    """Embed an interactive WaveSurfer.js player."""
    b64 = base64.b64encode(mp3_bytes).decode()
    uid = abs(hash(label + color))
    html = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
    <style>
      .ws-wrap-{uid} {{ background:#0e1117; padding:10px 14px 8px; border-radius:8px; }}
      .ws-label-{uid} {{ color:#aaa; font-size:12px; margin-bottom:6px; font-family:sans-serif; }}
      .ws-controls-{uid} {{ display:flex; gap:10px; align-items:center; margin-top:8px; }}
      .ws-btn-{uid} {{
        background:#1db954; border:none; border-radius:50%; width:36px; height:36px;
        cursor:pointer; color:#000; font-size:14px; display:flex; align-items:center; justify-content:center;
      }}
      .ws-btn-{uid}:hover {{ background:#17a349; }}
      .ws-stop-{uid} {{
        background:#555; border:none; border-radius:50%; width:30px; height:30px;
        cursor:pointer; color:#fff; font-size:12px; display:flex; align-items:center; justify-content:center;
      }}
      .ws-stop-{uid}:hover {{ background:#777; }}
      .ws-time-{uid} {{ color:#ccc; font-size:12px; font-family:monospace; margin-left:4px; }}
    </style>
    <div class="ws-wrap-{uid}">
      <div class="ws-label-{uid}">{label}</div>
      <div id="ws-{uid}"></div>
      <div class="ws-controls-{uid}">
        <button class="ws-btn-{uid}" id="playpause-{uid}" title="Play / Pause">
          <i class="fa fa-play"></i>
        </button>
        <button class="ws-stop-{uid}" id="stop-{uid}" title="Stop">
          <i class="fa fa-stop"></i>
        </button>
        <span class="ws-time-{uid}" id="time-{uid}">0:00 / 0:00</span>
      </div>
    </div>
    <script>
      (function() {{
        const ws = WaveSurfer.create({{
          container: '#ws-{uid}',
          waveColor: '{color}',
          progressColor: '{progress_color}',
          height: {height},
          barWidth: 2,
          barGap: 1,
          barRadius: 2,
          normalize: true,
          backend: 'WebAudio',
        }});
        ws.load('data:audio/mp3;base64,{b64}');

        const btn  = document.getElementById('playpause-{uid}');
        const stop = document.getElementById('stop-{uid}');
        const time = document.getElementById('time-{uid}');

        function fmt(s) {{
          const m = Math.floor(s/60), sec = Math.floor(s%60);
          return m+':'+(sec<10?'0':'')+sec;
        }}

        ws.on('ready', () => {{
          time.textContent = '0:00 / ' + fmt(ws.getDuration());
        }});
        ws.on('audioprocess', () => {{
          time.textContent = fmt(ws.getCurrentTime()) + ' / ' + fmt(ws.getDuration());
        }});
        ws.on('play',  () => btn.innerHTML = '<i class="fa fa-pause"></i>');
        ws.on('pause', () => btn.innerHTML = '<i class="fa fa-play"></i>');
        ws.on('finish',() => {{ btn.innerHTML = '<i class="fa fa-play"></i>'; }});

        btn.addEventListener('click', () => ws.playPause());
        stop.addEventListener('click', () => {{ ws.stop(); btn.innerHTML='<i class="fa fa-play"></i>'; }});
      }})();
    </script>
    """
    components.html(html, height=height + 90)


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

    if st.button("🔄  Reset microphone", help="Use this if the recorder shows an error"):
        st.session_state.mic_key += 1
        st.rerun()

    audio_input = st.audio_input("🎤  Click to record", key=f"mic_{st.session_state.mic_key}")

    if audio_input is not None:
        raw_bytes = audio_input.read()

        with st.spinner("Converting…"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            seg = AudioSegment.from_file(tmp_path)
            os.unlink(tmp_path)

            mp3_buf = io.BytesIO()
            seg.export(mp3_buf, format="mp3", bitrate="192k")
            mp3_bytes = mp3_buf.getvalue()

            mp4_buf = io.BytesIO()
            seg.export(mp4_buf, format="mp4", codec="aac")
            mp4_bytes = mp4_buf.getvalue()

            y, sr = load_audio_bytes(raw_bytes, "recording.wav")

        st.session_state.recorded_audio = (y, sr)
        st.success(f"Recorded  {len(y)/sr:.1f}s  |  {sr} Hz  →  go to **Edit & Export** to process")

        wavesurfer_player(mp3_bytes, label="Recording preview")

        fname = st.text_input("Filename", value="recording.mp4")
        st.download_button(
            label="💾  Save as MP4",
            data=mp4_bytes,
            file_name=fname,
            mime="audio/mp4",
            key="dl_rec",
        )

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
                y, sr = load_audio_bytes(uploaded.read(), uploaded.name)
            st.session_state.uploaded_audio = (y, sr)
            st.success(f"Loaded: {uploaded.name}  |  {len(y)/sr:.1f}s  @  {sr} Hz  →  go to **Edit & Export**")

    else:
        url = st.text_input("Paste a direct audio URL or YouTube / SoundCloud link")
        if url and st.button("Download & Load"):
            with st.spinner("Downloading…"):
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        out_tmpl = os.path.join(tmp, "audio.%(ext)s")
                        res = subprocess.run(
                            ["yt-dlp", "-x", "--audio-format", "wav", "-o", out_tmpl, url],
                            capture_output=True, text=True, timeout=180,
                        )
                        wav_files = list(Path(tmp).glob("*.wav"))
                        if wav_files:
                            y, sr = librosa.load(str(wav_files[0]), sr=None, mono=True)
                            st.session_state.uploaded_audio = (y.astype(np.float32), int(sr))
                            st.success(f"Downloaded  |  {len(y)/sr:.1f}s  @  {sr} Hz")
                        else:
                            raise RuntimeError(res.stderr[:300] or "yt-dlp: no output file")
                except Exception as e_yt:
                    try:
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        ext = url.split(".")[-1].split("?")[0] or "mp3"
                        y, sr = load_audio_bytes(r.content, f"file.{ext}")
                        st.session_state.uploaded_audio = (y, sr)
                        st.success(f"Downloaded  |  {len(y)/sr:.1f}s  @  {sr} Hz")
                    except Exception as e_http:
                        st.error(f"yt-dlp: {e_yt}\nHTTP: {e_http}")

    if st.session_state.uploaded_audio is not None:
        y, sr = st.session_state.uploaded_audio
        with st.spinner("Building preview…"):
            mp3_prev = to_mp3_bytes(y, sr)
        wavesurfer_player(mp3_prev, label="Uploaded file preview")
        st.caption(f"Duration: {len(y)/sr:.1f}s  |  {sr} Hz")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – EDIT & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.header("Edit & Export")

    # ── Source selector ───────────────────────────────────────────────────────
    sources = {}
    if st.session_state.recorded_audio is not None:
        sources["Last recording"] = st.session_state.recorded_audio
    if st.session_state.uploaded_audio is not None:
        sources["Uploaded file"] = st.session_state.uploaded_audio

    if not sources:
        st.info("No audio available. Use the **Record** or **Upload** tab first.")
        st.stop()

    chosen_src = st.radio("Audio source", list(sources.keys()), horizontal=True)
    y_orig, sr = sources[chosen_src]
    dur = len(y_orig) / sr

    # Use processed version if available for the same source, else original
    y_work = st.session_state.processed_audio[0] if st.session_state.processed_audio else y_orig

    st.divider()

    # ── Interactive waveform player ───────────────────────────────────────────
    st.subheader("Waveform Player")

    if st.session_state.processed_audio is not None:
        col_orig, col_proc = st.columns(2)
        with col_orig:
            with st.spinner("Loading original…"):
                mp3_orig = to_mp3_bytes(y_orig, sr)
            wavesurfer_player(mp3_orig, label="Original", color="#4a9eff", progress_color="#aaa")
        with col_proc:
            y_proc_arr, _ = st.session_state.processed_audio
            with st.spinner("Loading processed…"):
                mp3_proc = to_mp3_bytes(y_proc_arr, sr)
            wavesurfer_player(mp3_proc, label="Processed", color="#1db954", progress_color="#ff4b4b")
        y_work = y_proc_arr
    else:
        with st.spinner("Building player…"):
            mp3_work = to_mp3_bytes(y_orig, sr)
        wavesurfer_player(mp3_work, label="Working audio", color="#1db954", progress_color="#ff4b4b")
        y_work = y_orig

    st.divider()

    # ── Noise reduction & gain ────────────────────────────────────────────────
    st.subheader("Noise Reduction & Amplification")

    nc1, nc2, nc3 = st.columns(3)
    noise_prop = nc1.slider("Noise reduction strength", 0.0, 1.0, 0.5, 0.05)
    gain_db    = nc2.slider("Gain (dB)", -20, 40, 0)
    stationary = nc3.checkbox("Stationary noise", value=True,
                              help="Best for constant hum/hiss")

    bc1, bc2 = st.columns(2)
    if bc1.button("▶  Apply & compare"):
        with st.spinner("Processing… may take a moment"):
            y_proc = nr.reduce_noise(y=y_orig, sr=sr,
                                     prop_decrease=noise_prop, stationary=stationary)
            if gain_db != 0:
                y_proc = np.clip(y_proc * (10 ** (gain_db / 20)), -1.0, 1.0)
            st.session_state.processed_audio = (y_proc.astype(np.float32), sr)
        st.rerun()

    if bc2.button("↩  Reset"):
        st.session_state.processed_audio = None
        st.rerun()

    st.divider()

    # ── Split & save ─────────────────────────────────────────────────────────
    st.subheader("Split & Save")

    wc1, wc2 = st.columns(2)
    split_start = wc1.number_input("Segment start (s)", 0.0, float(dur), 0.0, 0.1)
    split_end   = wc2.number_input("Segment end (s)",   0.0, float(dur), float(dur), 0.1)

    s1 = max(0, int(split_start * sr))
    s2 = min(len(y_work), int(split_end * sr))
    segment = y_work[s1:s2]

    if len(segment) > 0:
        with st.spinner("Building segment preview…"):
            mp3_seg = to_mp3_bytes(segment, sr)
        wavesurfer_player(mp3_seg, label=f"Segment  {split_start:.2f}s → {split_end:.2f}s",
                          color="#ffa64b", progress_color="#ff4b4b", height=80)

        sc1, sc2 = st.columns(2)
        seg_name   = sc1.text_input("Segment filename", "segment.mp4")
        seg_format = sc2.selectbox("Format", ["MP4", "MP3", "WAV", "FLAC"], key="seg_fmt")

        seg_bytes, seg_mime = encode_audio(segment, sr, seg_format)
        seg_fname = str(Path(seg_name).with_suffix("." + seg_format.lower()))
        st.download_button("💾  Save segment", seg_bytes, seg_fname, seg_mime, key="dl_seg")
    else:
        st.warning("Segment is empty – adjust start / end.")

    st.divider()

    # ── Save full audio ───────────────────────────────────────────────────────
    st.subheader("Save Full Audio")

    fc1, fc2 = st.columns(2)
    full_name   = fc1.text_input("Output filename", "output.mp4")
    full_format = fc2.selectbox("Format", ["MP4", "MP3", "WAV", "FLAC"], key="full_fmt")

    full_bytes, full_mime = encode_audio(y_work, sr, full_format)
    full_fname = str(Path(full_name).with_suffix("." + full_format.lower()))
    st.download_button("💾  Save full audio", full_bytes, full_fname, full_mime, key="dl_full")
