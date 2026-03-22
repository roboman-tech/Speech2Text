"""
Quick test: record 3 seconds and save to test_capture.wav.
Run: python test_capture.py              (system + mic)
     python test_capture.py --system     (system audio only - best for music)
If you hear your recording when playing the file, capture works.
"""

import sys
import wave
import numpy as np

if sys.platform != "win32":
    print("This test is for Windows.")
    sys.exit(1)

from audio_capture import MixingAudioCapture, SAMPLE_RATE_WHISPER

def main():
    import queue
    import time
    system_only = "--system" in sys.argv
    chunks = []
    q = queue.Queue(maxsize=2048)
    cap = MixingAudioCapture(q, capture_system=True, capture_mic=not system_only)
    cap.start()
    print("Recording 3 seconds..." + (" Play music." if system_only else " Speak or play audio."))
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            chunk = q.get(timeout=0.02)
            if chunk is not None:
                chunks.append(chunk)
        except queue.Empty:
            pass
    while True:
        try:
            chunk = q.get_nowait()
            if chunk is not None:
                chunks.append(chunk)
        except queue.Empty:
            break
    cap.stop()
    if not chunks:
        print("ERROR: No audio captured." + (" Install PyAudioWPatch for system audio." if system_only else " Check mic privacy settings."))
        sys.exit(1)
    audio = np.concatenate([c if isinstance(c, np.ndarray) else np.frombuffer(c, dtype=np.int16) for c in chunks])
    with wave.open("test_capture.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE_WHISPER)
        wf.writeframes(audio.tobytes())
    print(f"Saved {len(audio)/SAMPLE_RATE_WHISPER:.1f}s to test_capture.wav — play it to verify.")

if __name__ == "__main__":
    main()
