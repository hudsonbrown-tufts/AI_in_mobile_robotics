"""lose.mp3 playback with pygame (same output-device handling as HW3)."""

import pygame
import pygame._sdl2.audio as sdl2_audio

import config


def choose_output_device_name():
    """Resolve OUTPUT_DEVICE_NAME_HINT to a real SDL output device (None = system default)."""
    if not config.OUTPUT_DEVICE_NAME_HINT:
        return None
    pygame.init()
    names = sdl2_audio.get_audio_device_names(False)
    # pygame.init() opened the mixer on the default device; close it so the real
    # pygame.mixer.init(devicename=...) below isn't a silent no-op (see HW3).
    pygame.mixer.quit()
    for name in names:
        if config.OUTPUT_DEVICE_NAME_HINT.lower() in name.lower():
            return name
    print(f"[sound] no output device matching {config.OUTPUT_DEVICE_NAME_HINT!r}; using the default.")
    return None


class LoseSound:
    def __init__(self):
        self.ok = False
        try:
            device = choose_output_device_name()
            pygame.mixer.init(devicename=device)
            pygame.mixer.music.load(str(config.LOSE_SOUND_PATH))
            self.ok = True
            print(f"[sound] lose.mp3 ready on {device or 'the default output device'}")
        except Exception as exc:
            print(f"[sound] can't play lose.mp3: {exc}")

    def play(self):
        """Non-blocking."""
        if self.ok:
            pygame.mixer.music.play()

    def close(self):
        if self.ok:
            pygame.mixer.quit()
