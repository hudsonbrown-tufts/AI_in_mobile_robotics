"""
miclib — shared microphone picker for ME193 AI examples.

Usage:
    from miclib import pick_mic
    device_index = pick_mic()
    # then pass to sd.InputStream(device=device_index, ...)

pick_mic() lists all audio input devices, prints them, and asks the user
to type a number. Press Enter with no input to accept the system default.
"""

import sounddevice as sd


def pick_mic() -> int | None:
    """
    Print all audio input devices and prompt the user to choose one.
    Returns the device index (int) to pass to sd.InputStream(device=...),
    or None to use the system default.
    """
    devices = sd.query_devices()

    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]

    if not inputs:
        print("[miclib] No input devices found — using system default.")
        return None

    print("\nAvailable microphone inputs:")
    default_idx = sd.default.device[0]
    for i, d in inputs:
        marker = " ← default" if i == default_idx else ""
        print(f"  [{i}] {d['name']}{marker}")

    print()
    raw = input("Enter device number (or press Enter for default): ").strip()

    if raw == "":
        print(f"Using system default.\n")
        return None

    try:
        choice = int(raw)
        matched = next((d for i, d in inputs if i == choice), None)
        if matched is None:
            print(f"  '{choice}' is not an input device — using system default.\n")
            return None
        print(f"Selected: {matched['name']}\n")
        return choice
    except ValueError:
        print(f"  Not a number — using system default.\n")
        return None
