"""
List audio devices and check loopback availability.
Run: python list_audio_devices.py
"""

import sys

if sys.platform != "win32":
    print("System audio capture (loopback) is Windows-only.")
    sys.exit(0)

try:
    import pyaudiowpatch as pyaudio
    HAS_LOOPBACK = True
except ImportError:
    import pyaudio
    HAS_LOOPBACK = False

from audio_capture import _get_system_audio_device

print("=== Audio Devices ===\n")
p = pyaudio.PyAudio()

if HAS_LOOPBACK:
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out_idx = wasapi["defaultOutputDevice"]
        print(f"Default output device index: {default_out_idx}")
        default_out = p.get_device_info_by_index(default_out_idx)
        print(f"  Name: {default_out['name']}")
        print(f"  Sample rate: {default_out.get('defaultSampleRate')}")
        print(f"  Is loopback: {default_out.get('isLoopbackDevice', 'N/A')}\n")
    except Exception as e:
        print(f"WASAPI error: {e}\n")
    print("Loopback devices:")
    for loopback in p.get_loopback_device_info_generator():
        print(f"  [{loopback['index']}] {loopback['name']}")
else:
    print("PyAudioWPatch not installed — WASAPI loopback unavailable. Checking for Stereo Mix...\n")

print("\nInput devices (potential system audio):")
for i in range(p.get_device_count()):
    try:
        dev = p.get_device_info_by_index(i)
        if dev.get("maxInputChannels", 0) >= 1:
            name = dev.get("name", "")
            if "stereo mix" in name.lower() or "loopback" in name.lower() or "wave out" in name.lower():
                print(f"  [*] [{i}] {name}")
            else:
                print(f"      [{i}] {name}")
    except Exception:
        pass

loopback = _get_system_audio_device(p)
print(f"\nSelected loopback for capture: {loopback['name'] if loopback else 'None (will use mic only)'}")

p.terminate()
