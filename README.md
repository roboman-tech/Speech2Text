# Live Caption / Interview Assistant

Real-time live captions for **all sounds** — system audio (videos, apps) and microphone. Optional **Generate Answer** panel uses DeepSeek when `DEEPSEEK_API_KEY` is set.

## Features

- **Audio capture**: System output (loopback) + microphone via PyAudioWPatch (WASAPI on Windows)
- **Voice activity detection**: Energy-based VAD to skip silence
- **Noise suppression**: noisereduce + optional high-pass on mic path
- **Speech-to-text**: OpenAI Whisper (offline; `base.en` in `main.py`)
- **UI**: tkinter — capture modes, optional diarization, context + AI answer

## Requirements

- **Windows** for system audio capture (WASAPI loopback). On macOS/Linux, only microphone is captured.
- Python 3.9+

## Troubleshooting: No System Audio

If system sounds are not captured (mic works, but videos/apps don’t):

1. **Install PyAudioWPatch** (recommended): `pip install PyAudioWPatch` – WASAPI loopback, no setup.
2. **Or enable Stereo Mix**: Sound settings → Recording → Show disabled devices → Enable “Stereo Mix”.
3. **Check devices**: `python list_audio_devices.py`
4. **Audio must be playing** – Play a video or music while testing.
5. **No sounds at all** – Run `python test_capture.py` to verify capture. Check Windows mic privacy (Settings → Privacy → Microphone).

## Setup

```powershell
cd SpeechtoText
python -m venv audio_env
.\audio_env\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

Install **PyTorch** separately if needed (Whisper uses it); for GPU: see [PyTorch install](https://pytorch.org/).

## Run

Activate the venv (so `PyAudioWPatch` and friends resolve), then:

```powershell
.\audio_env\Scripts\Activate.ps1
python main.py
```

- **Capture**: System only, Mic only, or System + Mic
- **Clear**: Reset transcript history
- **Generate Answer**: Uses dialogue + context (needs `DEEPSEEK_API_KEY` in `.env`)
- **Esc** or **right-click → Close** to exit

## GPU acceleration

Whisper uses GPU when PyTorch has CUDA support. Check: `python check_gpu.py`

If it shows "CPU-only", reinstall PyTorch with CUDA (you must uninstall first):
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Use `cu124` for CUDA 12.x (RTX 40xx) or `cu121` for CUDA 12.1. Run `python main.py` — it will print "Using GPU: ..." when detected.

## Speaker detection (optional)

To label different speakers in conversations:

1. `pip install pyannote-audio`
2. Create a [HuggingFace](https://huggingface.co) account and accept the [pyannote model terms](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Get your token from [Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. Run: `set HF_TOKEN=your_token` (Windows) or `export HF_TOKEN=your_token` (Linux/Mac)
5. `python main.py --diarize` (or enable the Diarize checkbox in the UI)

To use custom names instead of Speaker 00, Speaker 01, add to `.env`:
```
SPEAKER_NAMES=Peter,Ahlam
```
First detected speaker → Peter, second → Ahlam, etc.

## Configuration

Edit `main.py` (search `RealtimeTranscriber` / `CaptionWindow`):

- `whisper_model_size`: `"small.en"` (default, English-only). Options: `"base.en"`, `"medium.en"` (English); `"turbo"` (multilingual, better accuracy, faster; good for English too).
- `use_noise_reduce`: `False` to disable noisereduce
- `CaptionWindow(width=920, height=720, font_size=16)`: UI layout

Transcription uses **speech/silence VAD** (not fixed timers): segments are sent only after ~700 ms of silence. Chunk overlap (750 ms) and `beam_size=5` improve accuracy.

## Architecture

```
[Mic] ──┐
        ├── Mixer ── VAD ── Noise reduce ── Whisper ── UI
[System]┘
```
