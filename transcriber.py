"""
Real-time transcription: VAD, rolling buffer, openai-whisper ASR.
- Live decode every ~300ms during speech → preview + stable-word commits
- Finalize on silence → one commit, reset
- English models: small.en (default), base.en, medium.en. turbo = multilingual.
"""

from __future__ import annotations

import concurrent.futures
import os
import queue
import threading
import numpy as np
from audio_capture import SAMPLE_RATE_WHISPER, CHUNK_MS

# VAD: energy threshold (RMS). Lower = more sensitive.
VAD_ENERGY_THRESHOLD = 18
# Finalize segment only after this much silence
SPEECH_PADDING_MS = 700
MIN_SPEECH_MS = 500
MAX_SPEECH_MS = 12000
# Live preview: decode every N ms during speech (real-time)
PREVIEW_INTERVAL_MS = 300
PREVIEW_WINDOW_MS = 2800  # Only last N ms → less repetition, faster decode
MAX_IDLE_BUFFER_MS = 4000


def _norm(w: str) -> str:
    return w.lower().rstrip(".,?!;:\"'")


def _stable_prefix_len(prev_words: list[str], curr_words: list[str]) -> int:
    """Longest word prefix of curr that matches prev."""
    n = 0
    for i in range(min(len(prev_words), len(curr_words))):
        if _norm(prev_words[i]) != _norm(curr_words[i]):
            break
        n += 1
    return n


def _strip_overlap(committed: list[str], new: list[str]) -> list[str]:
    """If end of committed matches start of new, return only remainder."""
    for k in range(min(len(committed), len(new)), 0, -1):
        if all(_norm(c) == _norm(n) for c, n in zip(committed[-k:], new[:k])):
            return new[k:]
    return new


def _remove_growing_prefixes(text: str) -> str:
    """Remove Whisper repetition: 'A. B. C' where each is prefix of next -> keep last only."""
    import re
    parts = re.split(r'[.?!]\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return text
    result = []
    for p in parts:
        p_norm = _norm(p)
        while result and p_norm and _norm(result[-1]) and p_norm.startswith(_norm(result[-1])) and p_norm != _norm(result[-1]):
            result.pop()
        result.append(p)
    return ". ".join(result) if result else text


def _deduplicate_words(text: str) -> str:
    """Remove consecutive duplicate words and repeated phrases (Whisper sometimes repeats)."""
    words = text.split()
    if len(words) < 2:
        return text
    out = [words[0]]
    for w in words[1:]:
        if _norm(w) != _norm(out[-1]):
            out.append(w)
    n = len(out)
    for k in range(1, n // 2 + 1):
        if out[:k] == out[k : 2 * k]:
            return " ".join(out[:k])
    for phrase_len in range(min(5, n // 2), 1, -1):
        i = 0
        while i <= len(out) - 2 * phrase_len:
            if all(_norm(out[i + j]) == _norm(out[i + phrase_len + j]) for j in range(phrase_len)):
                out = out[: i + phrase_len] + out[i + 2 * phrase_len :]
                n = len(out)
                i = max(0, i - phrase_len)
            else:
                i += 1
    text = " ".join(out)
    return _remove_growing_prefixes(text)


def _is_speech_energy(audio: np.ndarray, threshold: float = VAD_ENERGY_THRESHOLD) -> bool:
    if len(audio) < 2:
        return False
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    return rms > threshold


def _reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        import noisereduce as nr
        audio_float = audio.astype(np.float32) / 32768.0
        reduced = nr.reduce_noise(y=audio_float, sr=sr, prop_decrease=0.15)  # Lighter: 0.15
        return (reduced * 32768).clip(-32768, 32767).astype(np.int16)
    except Exception:
        return audio


_diarization_pipeline = None
_diar_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_diar_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _diar_executor
    if _diar_executor is None:
        _diar_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return _diar_executor


def _get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline
    try:
        import torch
        from pyannote.audio import Pipeline
        hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
        if not hf_token:
            return None
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            _diarization_pipeline.to(torch.device("cuda"))
        return _diarization_pipeline
    except Exception:
        return None


def _detect_speaker(audio: np.ndarray):
    if len(audio) < SAMPLE_RATE_WHISPER:
        return None

    def _run():
        import torch
        try:
            pipeline = _get_diarization_pipeline()
            if pipeline is None:
                return None
            wav = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
            diarization = pipeline({"waveform": wav, "sample_rate": SAMPLE_RATE_WHISPER})
            segments = list(diarization.itertracks(yield_label=True))
            if segments:
                from collections import Counter
                speakers = [s[2] for s in segments]
                return Counter(speakers).most_common(1)[0][0]
        except Exception:
            pass
        return None

    try:
        future = _get_diar_executor().submit(_run)
        return future.result(timeout=10.0)
    except Exception:
        return None


def _run_whisper(audio: np.ndarray, model, initial_prompt: str | None, force_language: str = "en") -> str:
    import contextlib
    import io
    import torch
    audio_float = audio.astype(np.float32) / 32768.0
    use_fp16 = next(model.parameters()).device.type == "cuda"
    opts = {
        "language": force_language,
        "task": "transcribe",
        "fp16": use_fp16,
        "verbose": None,
        "temperature": 0.0,
        "beam_size": 5,
    }
    # Disable initial_prompt — it often causes Whisper to echo/repeat the prompt
    with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
        result = model.transcribe(audio_float, **opts)
    return (result.get("text") or "").strip()


class RealtimeTranscriber:
    """
    Multi-threaded transcriber:
    - Thread 1: Read audio → VAD → rolling buffer, enqueue decode jobs
    - Thread 2 (executor): Whisper decode → stable-word logic → callback(preview/final)
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        text_callback,
        whisper_model_size: str = "small.en",
        device: str = "auto",
        use_noise_reduce: bool = True,
        verbose: bool = False,
        use_diarization: bool = False,
    ):
        self.audio_queue = audio_queue
        self.text_callback = text_callback
        self.whisper_model_size = whisper_model_size
        self._device_pref = (device or "auto").lower()
        self.use_noise_reduce = use_noise_reduce
        self.verbose = verbose
        self.use_diarization = use_diarization
        self._stop = threading.Event()
        self._model = None

        self._work_queue: queue.Queue = queue.Queue(maxsize=8)
        self._prev_partial = ""
        self._committed_real_words: list[str] = []
        self._lock = threading.Lock()

    def set_diarization(self, enabled: bool):
        self.use_diarization = enabled

    def reset(self):
        with self._lock:
            self._prev_partial = ""
            self._committed_real_words = []

    def _load_model(self):
        import torch
        import whisper
        cuda_avail = torch.cuda.is_available()
        if self._device_pref == "cpu":
            self._device = "cpu"
        elif self._device_pref in ("cuda", "gpu") and cuda_avail:
            self._device = "cuda"
        elif self._device_pref in ("cuda", "gpu") and not cuda_avail:
            if self.verbose:
                print("[Whisper] GPU requested but not available, using CPU")
            self._device = "cpu"
        else:
            self._device = "cuda" if cuda_avail else "cpu"
        self._model = whisper.load_model(self.whisper_model_size, device=self._device)

    def _vad_buffer_loop(self):
        """Runs in its own thread: read audio, VAD, maintain buffer, enqueue decode jobs."""
        num_samples_per_chunk = int(SAMPLE_RATE_WHISPER * CHUNK_MS / 1000)
        padding_chunks = int(SPEECH_PADDING_MS / CHUNK_MS)
        min_speech_chunks = int(MIN_SPEECH_MS / CHUNK_MS)
        max_speech_chunks = int(MAX_SPEECH_MS / CHUNK_MS)
        preview_interval_chunks = max(1, int(PREVIEW_INTERVAL_MS / CHUNK_MS))

        buffer: list[np.ndarray] = []
        buffer_duration_ms = 0
        speech_started = False
        padding_counter = 0
        chunks_since_preview = 0

        while not self._stop.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break

            chunk = np.frombuffer(chunk, dtype=np.int16) if isinstance(chunk, bytes) else np.asarray(chunk, dtype=np.int16)

            is_speech = False
            if len(chunk) >= num_samples_per_chunk:
                for i in range(0, len(chunk) - num_samples_per_chunk + 1, num_samples_per_chunk):
                    frame = chunk[i : i + num_samples_per_chunk]
                    if _is_speech_energy(frame, VAD_ENERGY_THRESHOLD):
                        is_speech = True
                        break

            if is_speech:
                buffer.append(chunk)
                buffer_duration_ms += CHUNK_MS
                speech_started = True
                padding_counter = 0
                chunks_since_preview += 1

                # Live preview: decode every ~300ms during speech
                if chunks_since_preview >= preview_interval_chunks and len(buffer) >= min_speech_chunks:
                    audio = self._build_segment(buffer)
                    if audio is not None and len(audio) >= SAMPLE_RATE_WHISPER * 0.2:
                        n = int(PREVIEW_WINDOW_MS / 1000 * SAMPLE_RATE_WHISPER)
                        if len(audio) > n:
                            audio = audio[-n:]
                        try:
                            self._work_queue.put_nowait((audio.copy(), False))
                        except queue.Full:
                            pass
                        chunks_since_preview = 0

                if buffer_duration_ms >= MAX_SPEECH_MS:
                    audio = self._build_segment(buffer)
                    if audio is not None:
                        try:
                            self._work_queue.put_nowait((audio.copy(), True))
                        except queue.Full:
                            pass
                    buffer = []
                    buffer_duration_ms = 0
                    speech_started = False
                    chunks_since_preview = 0

            elif speech_started:
                buffer.append(chunk)
                buffer_duration_ms += CHUNK_MS
                padding_counter += 1

                if padding_counter >= padding_chunks:
                    if buffer_duration_ms >= MIN_SPEECH_MS:
                        audio = self._build_segment(buffer)
                        if audio is not None:
                            try:
                                self._work_queue.put_nowait((audio.copy(), True))
                            except queue.Full:
                                pass
                    buffer = []
                    buffer_duration_ms = 0
                    speech_started = False
                    chunks_since_preview = 0
                elif buffer_duration_ms >= max_speech_chunks * CHUNK_MS:
                    audio = self._build_segment(buffer)
                    if audio is not None:
                        try:
                            self._work_queue.put_nowait((audio.copy(), True))
                        except queue.Full:
                            pass
                    buffer = []
                    buffer_duration_ms = 0
                    speech_started = False
                    chunks_since_preview = 0
            else:
                buffer.append(chunk)
                buffer_duration_ms += CHUNK_MS
                while buffer_duration_ms > MAX_IDLE_BUFFER_MS and len(buffer) > 1:
                    buffer.pop(0)
                    buffer_duration_ms -= CHUNK_MS

        # Flush any remaining
        if buffer and buffer_duration_ms >= MIN_SPEECH_MS:
            audio = self._build_segment(buffer)
            if audio is not None:
                try:
                    self._work_queue.put_nowait((audio.copy(), True))
                except queue.Full:
                    pass

    def _build_segment(self, buf: list[np.ndarray]) -> np.ndarray | None:
        if not buf:
            return None
        return np.concatenate(buf)

    def _transcribe_loop(self):
        """
        Three-layer captions:
        - Compare new partial with prev → stable prefix moves to current_real
        - Changing tail stays as temp (replaced each update, never appended)
        - On pause: move real+temp to final, clear segment
        """
        while not self._stop.is_set():
            try:
                item = self._work_queue.get(timeout=0.15)
            except queue.Empty:
                continue
            if item is None:
                break
            audio, finalize = item
            if len(audio) < SAMPLE_RATE_WHISPER * 0.1:
                continue
            try:
                if self.use_noise_reduce and finalize:
                    audio = _reduce_noise(audio, SAMPLE_RATE_WHISPER)
                text = _run_whisper(audio, self._model, None, "en")
            except Exception:
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue
            text = _deduplicate_words(text)
            if self.verbose:
                print(f"[Whisper] {len(audio)/SAMPLE_RATE_WHISPER:.1f}s -> {repr(text)}")
            if finalize:
                with self._lock:
                    self._prev_partial = ""
                    self._committed_real_words = []
                if text and self.text_callback:
                    speaker = _detect_speaker(audio) if self.use_diarization else None
                    try:
                        self.text_callback(text, speaker=speaker, caption_type="final", is_finalize=True)
                    except TypeError:
                        self.text_callback(text, speaker=speaker)
                if self.text_callback:
                    try:
                        self.text_callback("", caption_type="clear_segment")
                    except TypeError:
                        pass
            else:
                curr = text.split()
                prev = self._prev_partial.split()
                stable_len = _stable_prefix_len(prev, curr)
                delta_text = ""
                temp_text = ""
                with self._lock:
                    n = len(self._committed_real_words)
                    if stable_len > n:
                        new_stable = _strip_overlap(self._committed_real_words, curr[n:stable_len])
                        if new_stable:
                            delta_text = _deduplicate_words(" ".join(new_stable))
                            self._committed_real_words = curr[:stable_len]
                    temp_tail = curr[len(self._committed_real_words):]
                    temp_text = _deduplicate_words(" ".join(temp_tail).strip()) if temp_tail else ""
                    self._prev_partial = text
                if delta_text and self.text_callback:
                    try:
                        self.text_callback(delta_text, caption_type="current_real")
                    except TypeError:
                        pass
                if self.text_callback:
                    try:
                        self.text_callback(temp_text, caption_type="temp")
                    except TypeError:
                        pass

    def _run(self):
        if self._model is None:
            self._load_model()
        if self.verbose:
            print(f"Model loaded ({self.whisper_model_size}).")

        vad_thread = threading.Thread(target=self._vad_buffer_loop, daemon=True)
        transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        vad_thread.start()
        transcribe_thread.start()

        vad_thread.join()
        self._work_queue.put(None)
        transcribe_thread.join(timeout=5.0)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.audio_queue.put(None)
        if getattr(self, "_thread", None) and self._thread.is_alive():
            self._thread.join(timeout=5.0)
