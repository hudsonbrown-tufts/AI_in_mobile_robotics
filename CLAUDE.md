# AI in Mobile Robotics (ME193)

Coursework repo. Each class/homework has its own folder. Some folders contain
Arduino App Lab apps for an Arduino UNO Q board, for example `HW4/minifig-light-tracker` and
`HW4/minifig-car-tracker`. Any folder with an `app.yaml` is an app:

- `python/main.py` runs in a container on the board's Linux side.
- `sketch/` is the microcontroller (STM32) code.
- `arduino.app_utils` (App, Bridge, etc.) only exists on the board, so
  unresolved-import warnings locally are expected. Don't try to pip install it.

## Talking to the UNO Q

Always go through `tools/deploy.py`. It finds the board over USB (adb) first,
then by trying each address in `HOSTS` in `tools/board_secrets.py`, so don't
call `ssh` or `scp` directly. On Windows use `python`, not `python3`.

Deploy and run one app (only the app you changed). Pass any file inside it, or
its folder name:

    python tools/deploy.py HW4/minifig-light-tracker/python/main.py
    python tools/deploy.py minifig-light-tracker

This stops the running app, copies the app's files to
`/home/arduino/ArduinoApps/<app folder name>`, and starts it. Read the output
for errors and fix them before reporting back.

Run any other command on the board:

    python tools/deploy.py --cmd "<command>"

Examples:

    python tools/deploy.py --cmd "arduino-app-cli app stop /home/arduino/ArduinoApps/<app-name>"
    python tools/deploy.py --cmd "rm /home/arduino/ArduinoApps/<app-name>/python/old_module.py"
    python tools/deploy.py --cmd "arduino-app-cli --help"

If deploy.py says it can't reach the board, stop and tell the user. The board's
IP has probably changed, and they need to update `HOSTS` in `tools/board_secrets.py`.

## Notes

- Deploying copies files but never deletes them on the board. If you remove or
  rename a file, also delete the old copy with `--cmd "rm ..."`.
- Make edits here, not on the board. Changes made only on the board get
  overwritten on the next deploy.
- App folder names must be unique across the repo, since they share one
  `ArduinoApps` folder on the board.
- Hidden folders like `.cache` are the board's Python environment. They are not
  copied; leave them alone.
- If App Lab reports library/import errors, check the board's clock first
  (`python tools/deploy.py --cmd "date"`). A wrong date has broken builds before.
