"""
Drive a LEGO Education Double Motor using tones picked up by the computer
microphone. The dominant frequency of each audio chunk is matched against
one of four tunable frequency ranges (forward/backward/left/right); the
motor drives in the matching direction, or stops if the sound is too quiet
or falls in more than one range at once.

Also plays a two-player "ball vs. goalie" game over MQTT: the ball's light
sensor detects when it's been caught and announces "I lost"; the goalie
listens for that message and plays a win sound. The goalie doesn't use a
light sensor at all.
"""

import msvcrt
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyaudio
import pygame
import pygame._sdl2.audio as sdl2_audio
from scipy.signal import butter, sosfilt, sosfilt_zi

import legoeducation as le
from mqttlib import MQTTClient

SCRIPT_DIR = Path(__file__).resolve().parent

# MQTT topic to join on startup, and the two game messages exchanged on it.
# The ball publishes these; the goalie listens for them and reacts. Change
# these strings freely — both roles always compare against the same
# constants, so the ball and goalie automatically stay in sync.
MQTT_TOPIC = "ME193/hudson"
MQTT_LOST_MESSAGE = "I lost"  # ball: caught by the light sensor -> goalie hears this and wins (win.mp3)
MQTT_GOAL_MESSAGE = "I made it in the goal!"  # ball: heard the FREQ_GOAL tone -> goalie hears this and loses (lost.mp3)

# The public test.mosquitto.org broker occasionally has a transient DNS/
# network hiccup; retry the initial connection a few times before giving up.
MQTT_CONNECT_RETRIES = 3
MQTT_RETRY_DELAY_SECONDS = 3

# Sound files (mp3), expected next to this script.
SOUND_LOST_PATH = SCRIPT_DIR / "lose.mp3"  # played by the ball when it loses
SOUND_WIN_PATH = SCRIPT_DIR / "win.mp3"  # played by the goalie when it wins

# Which output device to play sounds through. Run
# `python frequency_drive.py --list-devices` to see names (this also lists
# input/mic devices). Case-insensitive substring match; None uses the
# system default output device. Set this to your computer's built-in
# speakers (e.g. "Speakers") if it's not the default, to avoid sounds
# playing out of connected Bluetooth headphones instead.
OUTPUT_DEVICE_NAME_HINT = "Speakers"

# Update these to match the Connection Card printed on/with your Double Motor.
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "5164"

# Frequency ranges (Hz) for each direction. Tune these to whatever tone
# source you're using (voice, tone generator app, etc). Ranges should not
# overlap, or that overlapping band will be treated as "no command".
FREQ_FORWARD = (200, 400)
FREQ_BACKWARD = (400, 600)
FREQ_LEFT = (600, 800)
FREQ_RIGHT = (800, 1000)

# Separate "goal" tone (Hz) — not a drive direction. If the ball hears this
# while playing as the ball, it announces a goal over MQTT (see _drive()).
# Must not overlap the ranges above.
FREQ_GOAL = (1000, 1200)

# Motor speed (0-100%) for each direction.
SPEED_FORWARD = 50
SPEED_BACKWARD = 50
SPEED_LEFT = 40
SPEED_RIGHT = 40

# Audio capture settings.
SAMPLE_RATE = 44100  # preferred rate; falls back to the device's own default if unsupported
CHUNK_SIZE = 1024

# Which microphone to use. Run `python frequency_drive.py --list-devices` to
# see indices/names. INPUT_DEVICE_INDEX (if set) takes priority; otherwise
# INPUT_DEVICE_NAME_HINT does a case-insensitive substring match against
# device names (e.g. your Bluetooth headphones' name); leave both as None
# to use the system's default input device.
INPUT_DEVICE_INDEX = None
INPUT_DEVICE_NAME_HINT = "WH-1000XM4"

# Minimum loudness (in dB, see amplitude_db()) to treat a chunk as a real
# tone rather than silence/background noise. Raise this if the motor reacts
# to quiet room noise; lower it if quiet tones aren't being picked up. Note
# this is a consistent relative loudness scale, not a calibrated dB SPL
# meter reading (that would require calibrating against your specific mic).
AMPLITUDE_THRESHOLD_DB = 65

# Bandpass filter applied to each audio chunk before frequency analysis, to
# suppress background noise (low rumble, HVAC hum, hiss, etc) outside the
# range of tones we care about. Keep LOW/HIGH wide enough to cover all the
# FREQ_* ranges above (including FREQ_GOAL); ORDER is the Butterworth filter
# order (higher = sharper cutoff but more ringing).
BANDPASS_LOW_HZ = 150
BANDPASS_HIGH_HZ = 1300
BANDPASS_ORDER = 4

# Light/Color sensor (same Connection Card as the Double Motor) — proximity
# detection based on the sensor's "reflection" reading (0-100-ish; higher
# means more light bounced back, i.e. an object is closer). Tune this by
# watching the live "Reflection" value printed on screen.
PROXIMITY_REFLECTION_THRESHOLD = 50

# name -> (frequency range, legoeducation movement direction constant, speed)
DIRECTIONS = {
    "FORWARD": (FREQ_FORWARD, le.MOVEMENT_DIRECTION_FORWARD, SPEED_FORWARD),
    "BACKWARD": (FREQ_BACKWARD, le.MOVEMENT_DIRECTION_BACKWARD, SPEED_BACKWARD),
    "LEFT": (FREQ_LEFT, le.MOVEMENT_DIRECTION_LEFT, SPEED_LEFT),
    "RIGHT": (FREQ_RIGHT, le.MOVEMENT_DIRECTION_RIGHT, SPEED_RIGHT),
}


def design_bandpass(rate):
    """Design a Butterworth bandpass filter (as second-order sections) for
    the configured BANDPASS_LOW_HZ/BANDPASS_HIGH_HZ range."""
    nyquist = rate / 2
    low = BANDPASS_LOW_HZ / nyquist
    high = min(BANDPASS_HIGH_HZ, nyquist * 0.99) / nyquist
    return butter(BANDPASS_ORDER, [low, high], btype="bandpass", output="sos")


def dominant_frequency(samples, rate):
    """Return the frequency (Hz) of the strongest non-DC FFT bin."""
    windowed = samples * np.hanning(len(samples))
    spectrum = np.fft.rfft(windowed)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / rate)

    peak_index = np.argmax(magnitudes[1:]) + 1  # skip the DC bin
    return freqs[peak_index]


def amplitude_db(samples):
    """Return the RMS loudness of `samples` in dB. This is an uncalibrated,
    but consistent, relative loudness scale (0 dB ~= silence, ~90 dB ~= a
    full-scale 16-bit signal) — not a calibrated dB SPL meter reading."""
    rms = np.sqrt(np.mean(np.square(samples)))
    return 20 * np.log10(rms + 1e-6)


def direction_for_frequency(freq):
    """Return the matching direction name, or None for silence/ambiguous."""
    matches = [
        name
        for name, (freq_range, _, _) in DIRECTIONS.items()
        if freq_range[0] <= freq <= freq_range[1]
    ]
    return matches[0] if len(matches) == 1 else None


def is_goal_frequency(freq):
    """Return True if `freq` falls in the FREQ_GOAL tone range."""
    return FREQ_GOAL[0] <= freq <= FREQ_GOAL[1]


def list_input_devices():
    """Print every available mic input and speaker output device name."""
    audio = pyaudio.PyAudio()
    print("Available input devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            print(f"  [{i}] {info['name']!r} (default rate: {int(info['defaultSampleRate'])} Hz)")
    audio.terminate()

    pygame.init()
    print("Available output (speaker) devices:")
    for name in sdl2_audio.get_audio_device_names(False):
        print(f"  {name!r}")
    pygame.quit()


def choose_input_device(audio):
    """Pick an input device per INPUT_DEVICE_INDEX / INPUT_DEVICE_NAME_HINT,
    falling back to the system default input device."""
    if INPUT_DEVICE_INDEX is not None:
        return audio.get_device_info_by_index(INPUT_DEVICE_INDEX)

    if INPUT_DEVICE_NAME_HINT:
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0 and INPUT_DEVICE_NAME_HINT.lower() in info["name"].lower():
                return info
        raise RuntimeError(
            f"No input device matching {INPUT_DEVICE_NAME_HINT!r}. "
            "Run with --list-devices to see available devices."
        )

    return audio.get_device_info_by_index(audio.get_default_input_device_info()["index"])


def open_input_stream(audio, device_info):
    """Open the stream, preferring SAMPLE_RATE but falling back to the
    device's own default rate (Bluetooth headset mics often only support a
    lower rate like 16kHz). Returns (stream, actual_rate)."""
    device_index = device_info["index"]
    for rate in (SAMPLE_RATE, int(device_info["defaultSampleRate"])):
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
            )
            return stream, rate
        except OSError:
            continue
    raise RuntimeError(f"Could not open {device_info['name']!r} at any supported sample rate.")


def _run_with_timeout(func, *args, timeout, **kwargs):
    """Run func(*args, **kwargs) in a background thread; return True if it
    finished within `timeout` seconds. Used to bound BLE calls that can hang
    (e.g. a disconnect that never gets an acknowledgement) so shutdown can't
    freeze the terminal forever."""
    thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    thread.join(timeout)
    return not thread.is_alive()


def ask_role():
    """Prompt until the user answers 'ball' or 'goalie'."""
    while True:
        answer = input("Are you the ball or the goalie? ").strip().lower()
        if answer in ("ball", "b"):
            return "ball"
        if answer in ("goalie", "g"):
            return "goalie"
        print("Please answer 'ball' or 'goalie'.")


def choose_output_device_name():
    """Resolve OUTPUT_DEVICE_NAME_HINT to a real SDL output device name via a
    case-insensitive substring match. Returns None (system default) if unset
    or no match is found."""
    if not OUTPUT_DEVICE_NAME_HINT:
        return None

    pygame.init()  # ensures SDL's audio subsystem is up before we can list devices
    names = sdl2_audio.get_audio_device_names(False)
    # pygame.init() already opened the mixer on the default device; if we
    # leave it open, the real pygame.mixer.init(devicename=...) call in
    # main() is a silent no-op (SDL_mixer keeps the first-opened device
    # until mixer.quit() is called), so sound would keep playing on the
    # default device no matter what devicename we pass.
    pygame.mixer.quit()

    for name in names:
        if OUTPUT_DEVICE_NAME_HINT.lower() in name.lower():
            return name
    print(
        f"Warning: no output device matching {OUTPUT_DEVICE_NAME_HINT!r}; "
        "using the system default instead. Run with --list-devices to see options."
    )
    return None


def play_sound(path):
    """Play an mp3 file on the configured output device (non-blocking)."""
    if not path.exists():
        print(f"Warning: sound file not found: {path}")
        return
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()


def wait_for_sound_to_finish():
    """Block until the currently playing sound finishes — no timeout, so the
    program never ends before a sound has completely played."""
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def connect_mqtt(mqtt_client):
    """Connect with a few retries, since the public test.mosquitto.org broker
    occasionally fails a connection attempt with a transient DNS/network
    error rather than being genuinely unreachable."""
    for attempt in range(1, MQTT_CONNECT_RETRIES + 1):
        try:
            mqtt_client.connect()
            return
        except OSError as exc:
            if attempt == MQTT_CONNECT_RETRIES:
                print(f"Could not connect to the MQTT broker after {MQTT_CONNECT_RETRIES} attempts.")
                raise
            print(
                f"MQTT connect attempt {attempt}/{MQTT_CONNECT_RETRIES} failed ({exc}); "
                f"retrying in {MQTT_RETRY_DELAY_SECONDS}s..."
            )
            time.sleep(MQTT_RETRY_DELAY_SECONDS)


def main():
    role = ask_role()
    print(f"Role: {role.upper()}")

    output_device_name = choose_output_device_name()
    pygame.mixer.init(devicename=output_device_name)
    print(f"Playing sounds on: {output_device_name or 'system default output device'}")

    # Set by handle_mqtt_message (called from the MQTT client's background
    # thread) to tell _drive()'s main loop to stop, once the goalie's MQTT
    # action (reacting to "I lost") has finished.
    game_over = threading.Event()

    def handle_mqtt_message(topic, payload):
        print(f"\n[MQTT] Got message on '{topic}': {payload}")
        if role != "goalie":
            return

        if payload == MQTT_LOST_MESSAGE:
            print("The ball lost — goalie wins! Playing win sound.")
            play_sound(SOUND_WIN_PATH)
        elif payload == MQTT_GOAL_MESSAGE:
            print("The ball made it in the goal — goalie loses! Playing lost sound.")
            play_sound(SOUND_LOST_PATH)
        else:
            return

        wait_for_sound_to_finish()
        print("Game over — ending program.")
        game_over.set()

    mqtt_client = MQTTClient()
    connect_mqtt(mqtt_client)

    try:
        mqtt_client.subscribe(MQTT_TOPIC, handle_mqtt_message)
        time.sleep(1)  # give the subscription time to reach the broker
        print(f"Joined MQTT topic '{MQTT_TOPIC}' on test.mosquitto.org.")

        _drive(mqtt_client, role, game_over)
    finally:
        mqtt_client.disconnect()


def _drive(mqtt_client, role, game_over):
    motor = le.DoubleMotor()
    motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

    if not motor.connected:
        print("Error connecting to Double Motor.")
        sys.exit(1)

    # Only the ball uses the light sensor; the goalie doesn't touch it.
    sensor = None
    if role == "ball":
        sensor = le.ColorSensor()
        sensor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

        if not sensor.connected:
            print("Error connecting to Light Sensor.")
            sys.exit(1)

    audio = pyaudio.PyAudio()
    device_info = choose_input_device(audio)
    stream, actual_rate = open_input_stream(audio, device_info)
    print(f"Using input device: {device_info['name']!r} at {actual_rate} Hz")
    print("Press 'q' or Esc to quit.")

    sos = design_bandpass(actual_rate)
    filter_state = sosfilt_zi(sos) * 0

    current_direction = None
    object_close = False
    in_goal = False

    try:
        while True:
            if game_over.is_set():
                print("\nGame over — ending program.")
                break

            if msvcrt.kbhit() and msvcrt.getch() in (b"q", b"Q", b"\x1b"):
                print("\nQuit key pressed. Stopping...")
                break

            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)

            filtered_samples, filter_state = sosfilt(sos, samples, zi=filter_state)
            freq = dominant_frequency(filtered_samples, actual_rate)
            level_db = amplitude_db(filtered_samples)
            heard_tone = level_db >= AMPLITUDE_THRESHOLD_DB
            direction = direction_for_frequency(freq) if heard_tone else None

            status = f"\rFrequency: {freq:7.1f} Hz | Level: {level_db:5.1f} dB"

            if role == "ball":
                was_in_goal = in_goal
                in_goal = heard_tone and is_goal_frequency(freq)
                if in_goal and not was_in_goal:
                    print(f"\nGoal! Sending '{MQTT_GOAL_MESSAGE}' — playing win sound.")
                    mqtt_client.publish(MQTT_TOPIC, MQTT_GOAL_MESSAGE)
                    play_sound(SOUND_WIN_PATH)
                    wait_for_sound_to_finish()
                    print("Game over — ending program.")
                    break

            if sensor is not None:
                reflection = sensor.sensor.reflection
                was_close = object_close
                object_close = reflection >= PROXIMITY_REFLECTION_THRESHOLD
                if object_close and not was_close:
                    print(f"\nCaught! Sending '{MQTT_LOST_MESSAGE}' — playing lost sound.")
                    mqtt_client.publish(MQTT_TOPIC, MQTT_LOST_MESSAGE)
                    play_sound(SOUND_LOST_PATH)
                    wait_for_sound_to_finish()
                    print("Game over — ending program.")
                    break
                status += f" | Reflection: {reflection:5.1f}"

            status += f" | Direction: {direction or 'STOP':8s}"
            print(status, end="", flush=True)

            if direction != current_direction:
                if direction is None:
                    motor.movement_stop(blocking=False)
                else:
                    _, move_direction, speed = DIRECTIONS[direction]
                    motor.movement_move(direction=move_direction, speed=speed, blocking=False)
                current_direction = direction

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Shutting down...")
        if not _run_with_timeout(motor.movement_stop, timeout=2.0):
            print("Warning: motor stop command timed out; continuing shutdown anyway.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        if not _run_with_timeout(motor.disconnect, timeout=3.0):
            print("Warning: motor BLE disconnect timed out; the hub may take a while to show as disconnected.")
        if sensor is not None and not _run_with_timeout(sensor.disconnect, timeout=3.0):
            print("Warning: sensor BLE disconnect timed out; the hub may take a while to show as disconnected.")
        print("Done.")


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_input_devices()
    else:
        main()
