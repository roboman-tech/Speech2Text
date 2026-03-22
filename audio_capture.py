"""
Audio capture: system audio (loopback) + microphone, mixed like Windows Live Captions.
Uses PyAudioWPatch for WASAPI loopback on Windows.
"""

import sys
import threading
import queue
import numpy as np

SAMPLE_RATE = 48000
SAMPLE_RATE_WHISPER = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
DTYPE = np.int16
RESAMPLE_BUF_MS = 30

# PyAudioWPatch is Windows-only with WASAPI loopback
if sys.platform == "win32":
    try:
        import pyaudiowpatch as pyaudio
        HAS_LOOPBACK = True
    except ImportError:
        import pyaudio
        HAS_LOOPBACK = False
else:
    import pyaudio
    HAS_LOOPBACK = False


def _get_system_audio_device(p):
    """
    Get device for system audio capture. No settings required - like Windows Live Caption.
    Priority: WASAPI loopback (default output) > first loopback > Stereo Mix (if enabled).
    """
    if HAS_LOOPBACK:
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            pass
        else:
            try:
                default_speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
                if default_speakers.get("isLoopbackDevice"):
                    return default_speakers
                first_loopback = None
                spk_name = default_speakers.get("name", "")
                for loopback in p.get_loopback_device_info_generator():
                    if first_loopback is None:
                        first_loopback = loopback
                    lb_name = loopback.get("name", "")
                    if spk_name in lb_name or lb_name in spk_name:
                        return loopback
                    if spk_name and lb_name and any(
                        w in lb_name.lower() for w in spk_name.lower().split() if len(w) > 2
                    ):
                        return loopback
                return first_loopback
            except Exception:
                pass
    for i in range(p.get_device_count()):
        try:
            dev = p.get_device_info_by_index(i)
            if dev.get("maxInputChannels", 0) < 1:
                continue
            name = (dev.get("name") or "").lower()
            if "stereo mix" in name or "wave out mix" in name or "what u hear" in name:
                return dev
        except Exception:
            pass
    return None


def get_system_audio_status():
    """Return (device_name or None, found bool) for startup message. Caller-safe."""
    if sys.platform != "win32":
        return None, False
    try:
        p = pyaudio.PyAudio()
        dev = _get_system_audio_device(p)
        p.terminate()
        return (dev.get("name") if dev else None), dev is not None
    except Exception:
        return None, False


def _get_default_mic(p):
    """Get default microphone device."""
    try:
        return p.get_default_input_device_info()
    except Exception:
        # Fallback: first input device
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                return info
    return None


def _resample_to_16k(audio: np.ndarray, from_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Resample mono int16 to 16kHz. Uses resample_poly for smoother output."""
    from scipy import signal
    from math import gcd
    audio_float = audio.astype(np.float32) / 32768.0
    if len(audio_float) < 2:
        return np.array([], dtype=np.int16)
    up = SAMPLE_RATE_WHISPER // gcd(SAMPLE_RATE_WHISPER, from_rate)
    down = from_rate // gcd(SAMPLE_RATE_WHISPER, from_rate)
    resampled = signal.resample_poly(audio_float, up, down)
    return (resampled * 32768).clip(-32768, 32767).astype(np.int16)


def _stereo_to_mono(audio: np.ndarray, channels: int) -> np.ndarray:
    """Convert multi-channel to mono by averaging."""
    if channels == 1:
        return audio
    return audio.reshape(-1, channels).mean(axis=1).astype(audio.dtype)


def _mix_and_normalize(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mix two int16 streams, normalize to avoid clipping."""
    mix = a.astype(np.float32) + b.astype(np.float32)
    mix = mix / 2.0
    mix = np.clip(mix, -32768, 32767)
    return mix.astype(np.int16)


def _make_highpass_state(sr: int, cutoff: float = 80.0):
    """Return (b, a, zi) for streaming high-pass filter."""
    from scipy import signal
    b, a = signal.butter(2, cutoff, btype="high", fs=sr)
    zi = np.zeros(max(len(a), len(b)) - 1)
    return b, a, zi


class MixingAudioCapture:
    """Captures system audio (loopback) + microphone, mixes, outputs 16kHz mono chunks."""

    def __init__(self, queue_out: queue.Queue, capture_system: bool = True, capture_mic: bool = True, reduce_noise: bool = True):
        self.queue_out = queue_out
        self.capture_system = capture_system and sys.platform == "win32"
        self.capture_mic = capture_mic
        self.reduce_noise = reduce_noise
        self._stop = threading.Event()
        self._pa = None
        self._stream_loopback = None
        self._stream_mic = None
        self._loopback_chunks = queue.Queue(maxsize=256)
        self._mic_chunks = queue.Queue(maxsize=256)
        self._loopback_rate = SAMPLE_RATE
        self._loopback_channels = 2
        self._mic_rate = SAMPLE_RATE
        self._resample_buf = []
        self._mic_buf = []
        self._hp_b, self._hp_a, self._hp_zi = _make_highpass_state(SAMPLE_RATE_WHISPER)

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        try:
            self._loopback_chunks.put_nowait(np.frombuffer(in_data, dtype=DTYPE))
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    def _mic_callback(self, in_data, frame_count, time_info, status):
        try:
            self._mic_chunks.put_nowait(np.frombuffer(in_data, dtype=DTYPE))
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    def _mixer_thread_fn(self):
        import time
        out_samples = int(SAMPLE_RATE_WHISPER * CHUNK_MS / 1000)
        target_lb = int(self._loopback_rate * RESAMPLE_BUF_MS / 1000) if self._loopback_rate else 0
        target_mic = int(getattr(self, "_mic_rate", SAMPLE_RATE) * RESAMPLE_BUF_MS / 1000)
        both_active = (
            getattr(self, "_stream_loopback", None) is not None
            and getattr(self, "_stream_mic", None) is not None
        )
        pending_lb = None
        pending_mic = None

        while not self._stop.is_set():
            loopback_chunk = None
            mic_chunk = None
            if both_active:
                if pending_lb is None:
                    try:
                        pending_lb = self._loopback_chunks.get(timeout=0.05)
                    except queue.Empty:
                        pass
                if pending_mic is None:
                    try:
                        pending_mic = self._mic_chunks.get(timeout=0.05)
                    except queue.Empty:
                        pass
                if pending_lb is not None and pending_mic is not None:
                    loopback_chunk, mic_chunk = pending_lb, pending_mic
                    pending_lb = pending_mic = None
                else:
                    time.sleep(0.001)
                    continue
            else:
                try:
                    if getattr(self, "_stream_loopback", None) is not None:
                        loopback_chunk = self._loopback_chunks.get(timeout=0.1)
                except queue.Empty:
                    pass
                try:
                    if getattr(self, "_stream_mic", None) is not None:
                        mic_chunk = self._mic_chunks.get(timeout=0.1)
                except queue.Empty:
                    pass
                if loopback_chunk is None and mic_chunk is None:
                    continue

            loopback_16k = None
            mic_16k = None

            if loopback_chunk is not None and target_lb > 0:
                ch = self._loopback_channels
                mono = _stereo_to_mono(loopback_chunk, ch)
                self._resample_buf.append(mono)
                buf_samples = int(self._loopback_rate * RESAMPLE_BUF_MS / 1000)
                if sum(len(c) for c in self._resample_buf) >= buf_samples:
                    combined = np.concatenate(self._resample_buf)
                    self._resample_buf = [combined[buf_samples:]] if len(combined) > buf_samples else []
                    combined = combined[:buf_samples]
                    loopback_16k = _resample_to_16k(combined, self._loopback_rate)

            if mic_chunk is not None:
                ch = 2 if len(mic_chunk) > CHUNK_SAMPLES else 1
                mono = _stereo_to_mono(mic_chunk, ch)
                self._mic_buf.append(mono)
                if sum(len(c) for c in self._mic_buf) >= target_mic:
                    combined = np.concatenate(self._mic_buf)
                    self._mic_buf = [combined[target_mic:]] if len(combined) > target_mic else []
                    mic_chunk_30 = combined[:target_mic]
                    mic_16k = _resample_to_16k(mic_chunk_30, getattr(self, "_mic_rate", SAMPLE_RATE))

            if loopback_16k is not None and mic_16k is not None:
                mixed = _mix_and_normalize(loopback_16k, mic_16k)
            elif loopback_16k is not None:
                mixed = loopback_16k
            elif mic_16k is not None:
                mixed = mic_16k
            else:
                continue

            if len(mixed) >= out_samples:
                chunk = mixed[:out_samples].copy()
                if self.reduce_noise and self.capture_mic:
                    from scipy import signal
                    x = chunk.astype(np.float32) / 32768.0
                    y, self._hp_zi = signal.lfilter(self._hp_b, self._hp_a, x, zi=self._hp_zi)
                    chunk = (y * 32768).clip(-32768, 32767).astype(np.int16)
                try:
                    self.queue_out.put_nowait(chunk)
                except queue.Full:
                    pass

    def start(self):
        self._pa = pyaudio.PyAudio()
        try:
            system_device = _get_system_audio_device(self._pa) if self.capture_system else None
            mic_device = _get_default_mic(self._pa) if self.capture_mic else None
            if mic_device is None and system_device is None:
                raise RuntimeError("No audio source found. Check mic privacy or install PyAudioWPatch for system audio.")

            rate = SAMPLE_RATE
            channels_mic = 1
            mic_idx = -1
            if mic_device is not None:
                rate = int(mic_device["defaultSampleRate"])
                if rate not in (44100, 48000):
                    rate = SAMPLE_RATE
                channels_mic = min(2, int(mic_device["maxInputChannels"]))
                mic_idx = int(mic_device["index"])

            if system_device is not None and self.capture_system and (not self.capture_mic or int(system_device["index"]) != mic_idx):
                channels_lb = int(system_device["maxInputChannels"])
                rate_lb = int(system_device.get("defaultSampleRate") or 48000)
                frames_lb = max(512, int(rate_lb * CHUNK_MS / 1000))
                self._stream_loopback = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=channels_lb,
                    rate=rate_lb,
                    frames_per_buffer=frames_lb,
                    input=True,
                    input_device_index=system_device["index"],
                    stream_callback=self._loopback_callback,
                )
                self._stream_loopback.start_stream()
                self._loopback_rate = rate_lb
                self._loopback_channels = channels_lb
            else:
                self._loopback_rate = None

            if mic_device is not None:
                chunk_mic = max(512, int(rate * CHUNK_MS / 1000))
                self._stream_mic = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=channels_mic,
                    rate=rate,
                    frames_per_buffer=chunk_mic,
                    input=True,
                    input_device_index=mic_device["index"],
                    stream_callback=self._mic_callback,
                )
                self._mic_rate = rate
                self._stream_mic.start_stream()

            self._mixer = threading.Thread(target=self._mixer_thread_fn, daemon=True)
            self._mixer.start()
        except Exception as e:
            self._pa.terminate()
            raise RuntimeError(f"Audio capture failed: {e}") from e

    def stop(self):
        self._stop.set()
        if self._stream_loopback:
            try:
                self._stream_loopback.stop_stream()
                self._stream_loopback.close()
            except Exception:
                pass
        if self._stream_mic:
            try:
                self._stream_mic.stop_stream()
                self._stream_mic.close()
            except Exception:
                pass
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
        if hasattr(self, "_mixer") and self._mixer.is_alive():
            self._mixer.join(timeout=1.0)
