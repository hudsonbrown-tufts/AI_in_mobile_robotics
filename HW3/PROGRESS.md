# HW3 Progress Log

Frequency-controlled LEGO Education Double Motor car. Directions (forward/backward/left/right) are chosen by the dominant frequency picked up from the computer microphone.

## Steps taken so far

1. **Created the `HW3/` folder** in the project root to hold this assignment's work.
2. **Installed `pyaudio` (0.2.14)** into the project's `.venv` for microphone capture.
3. **Investigated existing course infrastructure** to reuse the right hardware API instead of building a custom one:
   - [Class 1/single_motor.py](../Class%201/single_motor.py) showed the `legoeducation` package's Connection Card pairing pattern (`card_color` + `card_serial`).
   - `legoeducation.DoubleMotor` (in `.venv/Lib/site-packages/legoeducation/device.py`) already exposes `movement_move(direction=..., speed=..., blocking=...)` with `MOVEMENT_DIRECTION_FORWARD/BACKWARD/LEFT/RIGHT` constants — a direct fit for the 4-direction requirement.
4. **Wrote [HW3/frequency_drive.py](frequency_drive.py):**
   - Connects to the Double Motor via `card_color=le.LEGO_COLOR_PURPLE`, `card_serial="5164"`.
   - Captures mono audio from the default mic with PyAudio (44.1kHz, 1024-sample chunks).
   - Computes the dominant frequency per chunk via a windowed FFT (`numpy.fft.rfft`), ignoring the DC bin.
   - Matches the frequency against four tunable `(low, high)` Hz ranges (`FREQ_FORWARD/BACKWARD/LEFT/RIGHT`), each with its own tunable speed constant (`SPEED_FORWARD/BACKWARD/LEFT/RIGHT`).
   - Treats a chunk as "no command" (motor stops) if its peak amplitude is below `AMPLITUDE_THRESHOLD`, or if the frequency falls inside more than one range (ambiguous overlap) — per user decision, not "first match wins".
   - Only sends a new BLE `movement_move`/`movement_stop` command when the resolved direction actually changes, to avoid flooding the connection.
   - Prints the live frequency and current direction to the console every loop iteration.
   - Cleans up on Ctrl+C: stops the motor, closes the audio stream, disconnects BLE.
5. Verified the script compiles (`py_compile`) — BLE/mic behavior itself hasn't been tested live yet (no hardware access from this tool).

6. **Added Bluetooth headset mic support to [HW3/frequency_drive.py](frequency_drive.py):**
   - New `list_input_devices()` (run via `python frequency_drive.py --list-devices`) prints every input device's index, name, and default sample rate.
   - New `INPUT_DEVICE_INDEX` / `INPUT_DEVICE_NAME_HINT` constants let you pin the capture device to your headphones by index or by a substring of its name, instead of always using the system default mic.
   - New `choose_input_device()` resolves those constants to a device; `open_input_stream()` opens the stream and falls back from the preferred `SAMPLE_RATE` (44.1kHz) to the device's own default rate if 44.1kHz isn't supported — Bluetooth headset mics commonly only support a lower rate (e.g. 16kHz mono) over the hands-free profile.
   - The actual negotiated sample rate (not the `SAMPLE_RATE` constant) is now used for the FFT frequency calculation, and the chosen device's name/rate is printed on startup.
   - User set `INPUT_DEVICE_NAME_HINT = "WH-1000XM4"` to target their Sony headphones.

7. **Added a bandpass filter to suppress background noise** (`scipy.signal`, already present in the venv):
   - New `BANDPASS_LOW_HZ` (150), `BANDPASS_HIGH_HZ` (1100), `BANDPASS_ORDER` (4) constants — wide enough to cover all `FREQ_*` ranges.
   - New `design_bandpass(rate)` builds a Butterworth bandpass filter (second-order sections) for the actual negotiated sample rate.
   - Each audio chunk is passed through the filter (`scipy.signal.sosfilt`) before the FFT step, with filter state (`zi`) persisted across chunks so there's no click/artifact at chunk boundaries.
   - This suppresses out-of-band energy (low rumble, hum, hiss) so it can't create a false frequency peak or bias detection — distinct from `AMPLITUDE_THRESHOLD`, which instead discards whole chunks that are too quiet.

8. **Added a quit hotkey ('q' or Esc)**:
   - Uses the built-in `msvcrt` module (Windows-only, no extra dependency) to non-blockingly poll for a keypress each loop iteration via `msvcrt.kbhit()`/`msvcrt.getch()`.
   - Pressing `q`, `Q`, or Esc breaks out of the main loop cleanly, hitting the same `finally` cleanup (motor stop, stream close, BLE disconnect) as Ctrl+C.
   - Startup now prints "Press 'q' or Esc to quit." alongside the input device info.

9. **Converted the amplitude threshold to dB, default cutoff 65 dB**:
   - `AMPLITUDE_THRESHOLD` (raw FFT magnitude) replaced with `AMPLITUDE_THRESHOLD_DB = 65`.
   - New `amplitude_db(samples)` computes RMS loudness of the (filtered) chunk as `20 * log10(rms)`. This is an uncalibrated but consistent relative loudness scale (not a true calibrated dB SPL meter reading, since that requires calibrating to a specific mic) — verified with a synthetic test: a loud 300 Hz tone reads ~86 dB, random background noise reads ~20 dB, so 65 dB cleanly separates them.
   - `dominant_frequency()` simplified to return just the frequency (loudness now comes from `amplitude_db`, not the old FFT-peak-magnitude threshold).
   - Console output now also shows the live dB level: `Frequency: ... | Level: ... dB | Direction: ...`.

10. **Fixed a shutdown hang ("locks up my terminal")**:
    - Root cause: `motor.disconnect()` (and `motor.movement_stop()`) call into the `legoeducation` BLE stack with no timeout — if the BLE stack never gets an acknowledgement (e.g. Bluetooth radio busy with the headset mic), the call can block forever, freezing the script (and the terminal) on quit.
    - New `_run_with_timeout()` helper runs a call in a background daemon thread and gives up after a timeout (2s for motor stop, 3s for disconnect) instead of waiting indefinitely; prints a warning if it times out but still lets the script exit.
    - Shutdown now always prints "Shutting down..." then "Done." so it's clear the script is exiting rather than frozen.

11. **Added the light/color sensor for proximity detection**:
    - Confirmed with the user that their light sensor uses the exact same Connection Card (purple/5164) as the Double Motor — `legoeducation.ColorSensor` normally models a separate physical device with its own card, so this was worth checking before implementing.
    - `main()` now also connects a `le.ColorSensor()` using the same `CARD_COLOR`/`CARD_SERIAL`, failing fast (like the motor) if it doesn't connect.
    - New `PROXIMITY_REFLECTION_THRESHOLD = 50` constant — proximity is inferred from the sensor's `reflection` reading (higher = more light bounced back = object closer); tune it by watching the live "Reflection" value now shown in the status line.
    - Detection is edge-triggered: `"Object detected nearby!"` prints once when reflection crosses the threshold, not on every loop iteration while an object stays close.
    - The sensor is disconnected on shutdown using the same bounded-timeout helper as the motor.

12. **Wired MQTT into the light sensor detection** (based on [HW3/mqtt.py](mqtt.py), which reuses the `mqttlib.MQTTClient` wrapper and `TOPIC = "ME193"` from [Class 4/mqtt_test.py](../Class%204/mqtt_test.py)):
    - Copied `mqttlib.py` into `HW3/` so `frequency_drive.py` can import it directly (it lives alongside `Class 4/mqtt_test.py` there, not in HW3).
    - `main()` now opens `MQTTClient()` and subscribes to `MQTT_TOPIC = "ME193"` as the very first thing it does (before connecting to the motor/sensor/mic), printing a confirmation once joined; an `on_mqtt_message` callback prints anything received on that topic.
    - The actual hardware setup (motor, sensor, audio, main loop) moved into a new `_drive(mqtt_client)` helper so it runs inside the `with MQTTClient() as mqtt_client:` block.
    - At the same edge-triggered point that prints `"Object detected nearby!"`, it now also calls `mqtt_client.publish(MQTT_TOPIC, "detected!")`.

13. **Fixed a hard crash on MQTT connect** (`socket.gaierror: [Errno 11001] getaddrinfo failed` connecting to `test.mosquitto.org`):
    - Confirmed it was a transient DNS/network hiccup, not a bug — re-ran DNS resolution and a raw `socket.getaddrinfo()` for the broker immediately afterward and both succeeded.
    - Replaced the `with MQTTClient() as mqtt_client:` block with a manual `connect_mqtt()` that retries the initial connection (`MQTT_CONNECT_RETRIES = 3`, `MQTT_RETRY_DELAY_SECONDS = 3`) before giving up, plus an explicit `try/finally` so `mqtt_client.disconnect()` still runs on any failure.
    - Verified the fixed MQTT path standalone (connect → subscribe → publish `"detected!"` → received it back on `ME193` → clean disconnect) without touching the motor/sensor/audio hardware.

14. **Replaced the "detected!" message with a ball-vs-goalie game over MQTT**:
    - Installed `pygame` (2.6.1) for mp3 playback (`pygame.mixer.music`).
    - New `ask_role()` prompts "Are you the ball or the goalie?" at startup (accepts `ball`/`b`/`goalie`/`g`, reprompts on anything else).
    - **Ball**: light sensor connects and works as before, but the edge-triggered "caught" event now publishes `"I lost"` (`MQTT_LOST_MESSAGE`) to the topic and plays `lost.mp3` (via new `play_sound()`), instead of the old "detected!"/print behavior.
    - **Goalie**: never connects the `ColorSensor` at all (`sensor = None` unless `role == "ball"`, guarded throughout `_drive()` — status line, detection loop, and shutdown disconnect all skip it). The MQTT subscription callback (`handle_mqtt_message`, a closure over `role` defined in `main()`) checks for `payload == "I lost"` and plays `win.mp3` when it arrives — this reacts as soon as a message comes in via the existing persistent subscription, satisfying "continuously look".
    - Sound files are expected at `HW3/lost.mp3` and `HW3/win.mp3` (resolved via `SCRIPT_DIR = Path(__file__).resolve().parent`, so it doesn't matter what directory the script is run from) — **not yet added by the user**; `play_sound()` prints a warning and no-ops if a file is missing rather than crashing (verified with both files absent).
    - Verified standalone (without hardware): the MQTT "I lost" message round-trips correctly and the missing-sound-file warning path works.

15. **Fixed sounds playing through the Bluetooth headphones instead of computer speakers**:
    - `pygame.mixer` was defaulting to the system's default output device, which was the WH-1000XM4 headphones.
    - New `OUTPUT_DEVICE_NAME_HINT = "Speakers"` constant, resolved by `choose_output_device_name()` (case-insensitive substring match against `pygame._sdl2.audio.get_audio_device_names()`) — mirrors the existing mic device-selection pattern. Confirmed on this machine it resolves to `"Speakers (Realtek(R) Audio)"`, distinct from `"Headphones (WH-1000XM4)"`.
    - `main()` now calls `pygame.mixer.init(devicename=output_device_name)` instead of a bare `init()`, and prints which output device it picked.
    - `--list-devices` now also lists available output (speaker) device names, not just mic inputs.
    - Verified standalone: `choose_output_device_name()` correctly resolves to the Realtek speakers and `pygame.mixer.init(devicename=...)` succeeds on it.

16. **Fixed sound still playing on the wrong device despite the correct printed device name**:
    - Found a stray `import pyaudioball` typo on disk (should be `import pyaudio`) that would have crashed the script outright — fixed.
    - Found the actual root cause: `choose_output_device_name()` called `pygame.init()` to enumerate devices, which as a side effect already opens the mixer on the *default* device. `pygame.mixer.init(devicename=...)` called afterward in `main()` is a documented SDL_mixer no-op once the mixer is already open — it silently keeps whatever device was opened first, ignoring the requested `devicename`, so the console printed the right device name while audio kept going to the actual default (headphones).
    - Confirmed this experimentally: calling `pygame.mixer.init()` twice with different settings and no `quit()` in between keeps the first settings; adding `pygame.mixer.quit()` between calls lets the second `init()` actually apply.
    - Fix: `choose_output_device_name()` now calls `pygame.mixer.quit()` right after listing device names, before returning, so the mixer is uninitialized again and the later `pygame.mixer.init(devicename=...)` in `main()` actually takes effect. Verified: mixer state is `None` right after `choose_output_device_name()` and reflects the real settings only after the intended init call.

17. **Added a fifth "goal" frequency range**:
    - New `FREQ_GOAL = (1000, 1200)` — separate from the four drive-direction ranges, so it's not part of `DIRECTIONS`/`direction_for_frequency()` and never drives the motor (hearing it just means "no direction" for movement purposes, i.e. the motor stops).
    - Widened `BANDPASS_HIGH_HZ` from 1100 → 1300 so the bandpass filter doesn't clip the new range.
    - New `is_goal_frequency()` checks the range; in `_drive()`'s loop, an edge-triggered check (`in_goal`/`was_in_goal`, same pattern as the light sensor's `object_close`) fires once when the tone starts, **only when `role == "ball"`**: publishes `MQTT_GOAL_MESSAGE = "I made it in the goal!"` to the MQTT topic and plays `win.mp3`.
    - Verified with synthetic tones: 300/500/700/900 Hz still resolve to FORWARD/BACKWARD/LEFT/RIGHT as before, and 1100 Hz now correctly resolves to no drive direction but `is_goal_frequency() == True`.

18. **Program now ends after any MQTT game action finishes**:
    - New `wait_for_sound_to_finish()` blocks (polling `pygame.mixer.music.get_busy()`, capped at a 10s timeout) so the program doesn't exit mid-sound and cut it off.
    - **Ball**: after the goal-tone action (publish `MQTT_GOAL_MESSAGE` + play `win.mp3`) or the light-sensor "caught" action (publish `MQTT_LOST_MESSAGE` + play `lost.mp3`), it now waits for the sound to finish, prints "Game over — ending program.", and `break`s the main loop directly (same thread).
    - **Goalie**: reacting to `"I lost"` happens inside `handle_mqtt_message`, which runs on the MQTT client's background thread, not the main loop's thread — so it can't just `break`. Added a `threading.Event()` (`game_over`), passed into `_drive()`; the callback waits for `win.mp3` to finish then calls `game_over.set()`, and the main loop now checks `game_over.is_set()` at the top of every iteration (alongside the existing quit-key check) and breaks if set.
    - All three paths still fall through to the existing `finally` block (motor stop, BLE disconnect, audio stream close, MQTT disconnect) — same clean shutdown as pressing 'q'/Esc.
    - Verified the cross-thread signaling mechanism standalone: a background thread setting the `Event` reliably breaks a polling loop on the main thread.

19. **Removed the sound-wait timeout, clarified message constants, and added the goalie's "loss" reaction**:
    - `wait_for_sound_to_finish()` no longer has a timeout cap (was 10s) — it now polls `pygame.mixer.music.get_busy()` indefinitely, guaranteeing the program never ends mid-sound in any of the three MQTT-action paths.
    - `MQTT_LOST_MESSAGE` and `MQTT_GOAL_MESSAGE` (both already top-level constants) got clearer grouped comments spelling out exactly which role publishes each and which sound the goalie plays in response — no duplicate constants introduced, since ball and goalie must always compare against the identical string.
    - `handle_mqtt_message` (goalie-only) now branches on both messages: `MQTT_LOST_MESSAGE` → win.mp3 (goalie wins, ball got caught), `MQTT_GOAL_MESSAGE` → lost.mp3 (goalie loses, ball scored) — any other payload is ignored, no game-over triggered.
    - Verified standalone: both message paths route to the correct sound and set `game_over`; an unrelated payload is ignored and leaves `game_over` unset.

## Open items / not yet done

- User needs to add `lost.mp3` and `win.mp3` to the `HW3/` folder.
- Live testing with the actual robot, microphone, and light sensor (tuning `AMPLITUDE_THRESHOLD_DB`, the frequency ranges, and `PROXIMITY_REFLECTION_THRESHOLD` to real hardware/room conditions), and playing the ball/goalie game across two machines over MQTT.
- Anything else requested beyond the initial 4-direction driving script.
