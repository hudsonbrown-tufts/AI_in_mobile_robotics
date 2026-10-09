# UNO Q tools

Helper scripts from the ME193 `unoq-vscode-kit` (see
[DevelopingUNOQAppsInVSCode.md](https://github.com/chrisbuerginrogers/ME193-Robotics/blob/main/Public%20stuff/UNOQ%20code/DevelopingUNOQAppsInVSCode.md)).
They run on the laptop, not the board, and deploy Arduino App Lab apps to the UNO Q and connect to it. Nothing in this folder is copied to the board.

**One change from the kit:** apps can live in **any folder of this repo**, for example `HW4/minifig-light-tracker`. `deploy.py` walks up from the open file to the nearest folder with an `app.yaml`. On the board, that app goes to `/home/arduino/ArduinoApps/<folder name>`, so give each app folder a unique name.

| File | What it does |
|---|---|
| `deploy.py` | Copies an app to the board and starts it; runs commands; opens a terminal or Python on the board |
| `Tufts_WiFi.py` | Advertises the board over mDNS so App Lab can find it on Tufts WiFi (needs `pip install zeroconf`) |
| `board_secrets.example.py` | Template for your board's settings. Committed; don't edit. |
| `board_secrets.py` | **Your** board's address. Created automatically and gitignored. |
| `config.py` | Loads the settings (and creates `board_secrets.py` on first run) |

## Commands

Run these from the repo root:

    python tools/deploy.py HW4/minifig-light-tracker/python/main.py   # stop, copy, start the app containing that file
    python tools/deploy.py minifig-light-tracker                      # ...or name the app folder
    python tools/deploy.py --python                       # Python in the running app's container
    python tools/deploy.py --cmd "<command>"              # run any command on the board
    python tools/deploy.py --shell                        # terminal on the board (type exit to leave)
    python tools/Tufts_WiFi.py                            # advertise the board (Ctrl+C to stop)

## From VS Code

Open any file inside an app, wherever it is in the repo, and press **Ctrl+Shift+B**. Then pick one of:
- **Run on UNO Q:** deploy and start the open file's app.
- **Python in UNO Q container:** a Python prompt in the running app, or on the board if no app is running.
- **Terminal on UNO Q:** a command-line session on the board.
- **Advertise UNO Q on Tufts WiFi:** runs `Tufts_WiFi.py`.

The board runs one app at a time, so deploying stops whichever app was running. Deploying copies files but never deletes them on the board.
