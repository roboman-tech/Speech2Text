"""
NPC Dialogue Engine - Live captions + AI-generated answers.
Uses system audio + microphone for real-time transcription. Paste context (resume, job
description) and click Generate Answer to get DeepSeek-assisted responses.
Env: SPEAKER_NAMES, DEEPSEEK_API_KEY, HF_TOKEN (for diarization).
     QUIET=1 — force off terminal diagnostics even if -v is passed.
"""

import argparse
import logging
import os
import warnings
import queue
import sys
import threading
from pathlib import Path

def _load_dotenv():
    """Load .env (HF_TOKEN, SPEAKER_NAMES, etc.) without extra dependencies."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

from audio_capture import MixingAudioCapture, get_system_audio_status
from transcriber import RealtimeTranscriber
from ui import CaptionWindow
from deepseek_api import generate as deepseek_generate


def _load_speaker_names():
    """Load SPEAKER_NAMES from environment (set by .env). E.g. SPEAKER_NAMES=Peter,Ahlam"""
    names = os.environ.get("SPEAKER_NAMES", "").strip()
    return [n.strip() for n in names.split(",") if n.strip()] if names else []


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _silence_terminal_libs(verbose: bool) -> None:
    """When not verbose, hide tqdm/progress bars and noisy loggers on stderr."""
    if verbose:
        return
    warnings.filterwarnings("ignore")
    logging.root.setLevel(logging.ERROR)
    for logger_name in (
        "urllib3",
        "httpx",
        "httpcore",
        "transformers",
        "huggingface_hub",
        "pyannote",
        "lightning",
        "pytorch_lightning",
        "fsspec",
        "numba",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Print diagnostics to the terminal")
    parser.add_argument("--diarize", action="store_true", help="Detect speaker changes (requires pyannote + HF_TOKEN)")
    args = parser.parse_args()
    if _env_truthy("QUIET"):
        args.verbose = False
    _silence_terminal_libs(args.verbose)

    def vprint(*a, **kw):
        if args.verbose:
            print(*a, **kw)

    capture_system_available = sys.platform == "win32"
    if not capture_system_available:
        vprint("System audio is Windows-only. Mic only available.")

    import torch
    vprint(f"Using GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU")

    if capture_system_available:
        name, found = get_system_audio_status()
        if found:
            vprint(f"System audio: {name}")
        else:
            vprint("System audio: not found — enable Stereo Mix or install PyAudioWPatch")

    def diarize_available():
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            return False
        try:
            __import__("pyannote.audio")
            return True
        except Exception:
            return False

    diarize_avail = diarize_available()
    audio_queue = queue.Queue(maxsize=1024)
    ui = CaptionWindow(
        width=920, height=720, font_size=16,
        system_available=capture_system_available,
        diarize_available=diarize_avail,
        diarize_initial=args.diarize,
        interview_mode=True,
    )

    # Dialogue history: list of {"speaker": str|None, "text": str}, in order.
    dialogue_history: list[dict] = []

    speaker_names = _load_speaker_names()
    if speaker_names:
        vprint(f"Speaker names: {', '.join(speaker_names)}")

    _fallback_idx = [0]  # When diarization returns None, alternate names

    def _resolve_speaker_label(speaker: str) -> str:
        """Map SPEAKER_00, speaker_0, 0, ... to custom names from SPEAKER_NAMES."""
        if not speaker:
            return speaker
        if not speaker_names:
            return str(speaker).replace("_", " ").title()
        s = str(speaker)
        idx = None
        if "_" in s:
            try:
                idx = int(s.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                pass
        else:
            try:
                idx = int(s)
            except ValueError:
                pass
        if idx is not None and 0 <= idx < len(speaker_names):
            return speaker_names[idx]
        return s.replace("_", " ").title()

    def mode_to_flags(mode: str):
        if mode == "system":
            return capture_system_available, False
        if mode == "mic":
            return False, True
        return capture_system_available, True

    # Accumulate current utterance for dialogue_history; only add on finalize
    _current_utterance = [""]
    _current_speaker = [None]

    def on_transcription(text: str, speaker: str = None, caption_type: str = None, is_finalize: bool = False, **kw):
        nonlocal _current_utterance, _current_speaker
        is_preview = kw.get("is_preview", False)
        clear_preview = kw.get("clear_preview", False)
        if caption_type == "clear_segment" or clear_preview:
            if _current_utterance[0]:
                dialogue_history.append({"speaker": _current_speaker[0], "text": _current_utterance[0]})
                _current_utterance[0] = ""
                _current_speaker[0] = None
            def _clear():
                try:
                    ui.clear_segment()
                except Exception as e:
                    vprint(f"UI update error: {e}")
            ui.root.after(0, _clear)
            return
        if caption_type == "temp":
            def _update():
                try:
                    ui.set_temp(text or "")
                except Exception as e:
                    vprint(f"UI update error: {e}")
            ui.root.after(0, _update)
            return
        if caption_type == "current_real":
            def _update():
                try:
                    ui.append_current_real(text or "")
                except Exception as e:
                    vprint(f"UI update error: {e}")
            ui.root.after(0, _update)
            return
        if caption_type == "final" or is_finalize:
            vprint(f"[Caption] {text}")
            if speaker:
                display_speaker = _resolve_speaker_label(speaker)
            elif speaker_names and ui.get_diarize():
                idx = _fallback_idx[0] % len(speaker_names)
                display_speaker = speaker_names[idx]
                _fallback_idx[0] += 1
            else:
                display_speaker = None
            seg = (text or "").strip()
            if seg:
                dialogue_history.append({"speaker": display_speaker or _current_speaker[0], "text": seg})
            _current_utterance[0] = ""
            _current_speaker[0] = None
            def _update(t=text, s=display_speaker):
                try:
                    ui.append_final(t or "", speaker=s)
                except Exception as e:
                    vprint(f"UI update error: {e}")
            ui.root.after(0, _update)
            return
        if is_preview:
            def _update():
                try:
                    ui.set_temp(text or "")
                except Exception as e:
                    vprint(f"UI update error: {e}")
            ui.root.after(0, _update)
            return

    use_diarize = ui.get_diarize()

    def _create_transcriber():
        return RealtimeTranscriber(
            audio_queue,
            text_callback=lambda t, **kw: on_transcription(t, **kw),
            whisper_model_size=ui.get_model(),
            device=ui.get_device(),
            use_noise_reduce=True,
            verbose=args.verbose,
            use_diarization=ui.get_diarize(),
        )

    transcriber = _create_transcriber()
    if use_diarize and speaker_names:
        from transcriber import _get_diarization_pipeline
        if _get_diarization_pipeline() is None:
            vprint("Diarization: failed to load (check HF_TOKEN and model access)")
        else:
            vprint("Diarization: ready (names from SPEAKER_NAMES)")

    capturer = None

    def on_transcriber_apply():
        nonlocal transcriber, capturer
        was_capturing = capturer is not None and ui.is_capturing()
        if capturer:
            capturer.stop()
            capturer = None
        while True:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        transcriber.stop()
        vprint("Reloading Whisper model...")
        transcriber = _create_transcriber()
        transcriber.start()
        if was_capturing:
            mode = ui.get_mode()
            sys_flag, mic_flag = mode_to_flags(mode)
            if sys_flag or mic_flag:
                capturer = MixingAudioCapture(
                    audio_queue,
                    capture_system=sys_flag,
                    capture_mic=mic_flag,
                )
                capturer.start()
        vprint(f"Model: {ui.get_model()} | Device: {ui.get_device()}")
        ui.set_status(f"Model: {ui.get_model()} | Device: {ui.get_device()}")

    def on_mode_change(mode: str):
        nonlocal capturer
        try:
            if capturer:
                capturer.stop()
                capturer = None
                while True:
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
            sys_flag, mic_flag = mode_to_flags(mode)
            if not sys_flag and not mic_flag:
                vprint("System audio not available on this platform.")
                return
            if not ui.is_capturing():
                capturer = None
                return
            capturer = MixingAudioCapture(
                audio_queue,
                capture_system=sys_flag,
                capture_mic=mic_flag,
            )
            capturer.start()
        except Exception as e:
            vprint(f"Capture error: {e}")

    def on_closing():
        transcriber.stop()
        if capturer:
            capturer.stop()
        ui.close()

    def on_stop_click():
        nonlocal capturer
        if ui.is_capturing():
            if capturer:
                capturer.stop()
                capturer = None
            while True:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            ui.set_capturing(False)
            ui.set_text("Stopped. Click Start to resume.")
        else:
            mode = ui.get_mode()
            sys_flag, mic_flag = mode_to_flags(mode)
            if sys_flag or mic_flag:
                capturer = MixingAudioCapture(
                    audio_queue,
                    capture_system=sys_flag,
                    capture_mic=mic_flag,
                )
                capturer.start()
                ui.set_capturing(True)
                ui.set_text("Listening...")

    def on_clear_history():
        dialogue_history.clear()
        _current_utterance[0] = ""
        _current_speaker[0] = None
        transcriber.reset()

    def _detect_latest_question(history: list[dict]) -> str:
        """Scan dialogue from end to find the latest question or request."""
        q_markers = ("?", "tell me", "what", "how", "why", "can you", "could you", "would you", "describe", "explain")
        for i in range(len(history) - 1, -1, -1):
            t = (history[i].get("text") or "").strip()
            if not t:
                continue
            lower = t.lower()
            if lower.endswith("?") or any(lower.startswith(m) for m in q_markers):
                return t
        # Fallback: last non-empty utterance
        for i in range(len(history) - 1, -1, -1):
            t = (history[i].get("text") or "").strip()
            if t:
                return t
        return ""

    def _build_prompt(pre_given_context: str, dialogue_since_clear: list[dict], question: str) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) for DeepSeek.
        Uses pre-given context (resume, etc.) and dialogue since last Clear."""
        sys = (
            "You are an interview answering assistant. Use the pre-given context (resume, job description, etc.) "
            "and the dialogue history (transcriptions since last clear) to generate clear, relevant answers. Be concise but complete."
        )
        lines = []
        for h in dialogue_since_clear:
            s = h.get("speaker") or "Speaker"
            t = h.get("text") or ""
            lines.append(f"{s}: {t}")
        dialogue = "\n".join(lines) if lines else "(no dialogue since last clear)"
        user = f"Pre-given context (resume, job description, talking points):\n{pre_given_context or '(none)'}\n\n"
        user += f"Dialogue since last clear:\n{dialogue}\n\n"
        if question:
            user += f"Latest question/request:\n{question}\n\nGenerate a helpful answer:"
        else:
            user += "Based on the dialogue above, provide a brief summary or suggested response."
        return sys, user

    def generate_answer():
        pre_given = ui.get_user_context()
        question = _detect_latest_question(dialogue_history)
        sys_prompt, user_prompt = _build_prompt(pre_given, dialogue_history, question)
        ui.set_generate_loading(True)
        ui.set_generated_answer("")

        def _run():
            result = ""
            try:
                result = deepseek_generate(user_prompt, system_prompt=sys_prompt, timeout=60)
            except Exception as e:
                result = f"Error: {e}"
            if not result:
                result = "No response from DeepSeek. Check DEEPSEEK_API_KEY and network."
            def _update():
                ui.set_generated_answer(result)
                ui.set_generate_loading(False)
            ui.root.after(0, _update)

        threading.Thread(target=_run, daemon=True).start()

    ui.on_close(on_closing)
    ui.on_transcriber_apply(on_transcriber_apply)
    ui.on_mode_change(on_mode_change)
    ui.on_stop(on_stop_click)
    ui.on_diarize_change(lambda enabled: transcriber.set_diarization(enabled))
    ui.on_clear(on_clear_history)
    ui.on_generate(generate_answer)

    initial_mode = ui.get_mode()
    sys_flag, mic_flag = mode_to_flags(initial_mode)
    if not sys_flag and not mic_flag:
        ui.set_mode("mic")
        sys_flag, mic_flag = False, True
    capturer = MixingAudioCapture(
        audio_queue,
        capture_system=sys_flag,
        capture_mic=mic_flag,
    )

    vprint("Loading Whisper model (may take 10-30s on first run)...")
    transcriber._load_model()
    vprint("Model ready. Starting capture...")
    capturer.start()
    transcriber.start()
    vprint("Caption window open. Use the Capture dropdown to switch mode.")
    try:
        ui.run()
    finally:
        on_closing()


if __name__ == "__main__":
    main()
