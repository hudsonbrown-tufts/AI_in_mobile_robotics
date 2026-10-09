#!/usr/bin/env python3
"""Copy one App from this repo to the UNO Q and (re)start it.

From the ME193 unoq-vscode-kit, adapted so apps can sit in any folder of this
repo (e.g. HW4/minifig-light-tracker): the app is the nearest folder above the given file
that contains an app.yaml, and it goes to /home/arduino/ArduinoApps/<folder name>.

Usage:  python deploy.py <any file path inside the app>
        python deploy.py <app folder name>
        python deploy.py --cmd "command to run on the board"
        python deploy.py --python
            (Python prompt inside the running app's container, or on the
             board's Linux side if no app is running)
        python deploy.py --shell
            (open a terminal on the board)
VS Code passes the open file automatically (see .vscode/tasks.json).
Works on macOS, Linux and Windows 10+ (uses the built-in ssh and scp).
If the board is plugged in over USB and `adb` is installed, that is used first;
otherwise it falls back to ssh over the network addresses in HOSTS.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Board addresses and login live in board_secrets.py (created on first run).
from config import HOSTS, USER

WEB_LOCAL_PORT = 7007
REMOTE_DIR = "/home/arduino/ArduinoApps"
ROOT = Path(__file__).resolve().parent.parent   # the repo root (one up from tools/)

SSH_OPTS = [
    "-o", "ConnectTimeout=8",               # give up on a dead address (3 s was too short when the
                                            # board's WiFi is waking up: the first try timed out)
    "-o", "BatchMode=yes",                  # never hang on a password prompt
    "-o", "HostKeyAlias=unoq",              # same board identity for IP and .local
    "-o", "StrictHostKeyChecking=accept-new",  # trust the board on first connect
]


def run(cmd, check=True, timeout=None, quiet=False):
    if not quiet:
        print("$", " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, check=check, timeout=timeout,
                              capture_output=quiet)
    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"  (no answer after {timeout}s, moving on)", flush=True)
        return None


def find_adb_board():
    """Return ('adb', None) if exactly one board answers over USB, else None."""
    if not shutil.which("adb"):
        return None
    print("Trying USB (adb) ...", end=" ", flush=True)
    r = run(["adb", "devices"], check=False, timeout=10, quiet=True)
    if r is not None and r.returncode == 0:
        lines = r.stdout.decode().splitlines()[1:]
        devices = [l.split()[0] for l in lines if l.strip().endswith("device")]
        if devices:
            print("connected")
            return ("adb", None)
    print("no device")
    return None


def find_board():
    """Return ('adb', None) or ('ssh', 'user@host') for the first transport that answers."""
    board = find_adb_board()
    if board:
        return board
    for host in HOSTS:
        print(f"Trying {host} ...", end=" ", flush=True)
        r = run(["ssh", *SSH_OPTS, f"{USER}@{host}", "true"],
                check=False, timeout=10, quiet=True)
        if r is not None and r.returncode == 0:
            print("connected")
            return ("ssh", f"{USER}@{host}")
        print("no answer")
    sys.exit("Couldn't reach the board over USB (adb) or at any address in HOSTS. "
             "Check that it's on, plugged in or on the same network, and that "
             "the IP in board_secrets.py is current.")


def remote(board, command, tty=False, **kwargs):
    """Run a shell command on the board."""
    kind, target = board
    if kind == "adb":
        return run(["adb", "shell", *(["-t"] if tty else []), command], **kwargs)
    flags = ["-t"] if tty else ["-n"]
    return run(["ssh", *SSH_OPTS, *flags, target, command], **kwargs)


def board_ip(board):
    """Best guess at the board's LAN IP, or None if it isn't on a network."""
    kind, target = board
    if kind == "ssh":
        return target.split("@")[-1]
    r = remote(board, "ip -4 route get 1.1.1.1", check=False, quiet=True, timeout=10)
    if r is not None and r.returncode == 0:
        words = r.stdout.decode().split()
        if "src" in words:
            return words[words.index("src") + 1]
    return None


def web_url(board):
    """URL for the app's web page (port 7000 on the board), or None."""
    kind, _ = board
    ip = board_ip(board)
    if ip:
        return f"http://{ip}:7000"
    if kind == "adb":
        # Not on a network: tunnel over USB. Local port is 7007 because macOS
        # uses 7000 for AirPlay.
        r = run(["adb", "forward", f"tcp:{WEB_LOCAL_PORT}", "tcp:7000"],
                check=False, quiet=True, timeout=10)
        if r is not None and r.returncode == 0:
            return f"http://localhost:{WEB_LOCAL_PORT} (via USB)"
    return None


def find_app_dir(arg):
    """The App folder for a file path inside it, or for an app's folder name."""
    p = Path(arg)
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        d = p if p.is_dir() else p.parent
        for cand in (d, *d.parents):
            if (cand / "app.yaml").exists():
                return cand
            if cand == ROOT:
                break
        return None
    # Just a name, e.g. "minifig-light-tracker": search the repo (skipping hidden folders and .venv).
    matches = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [n for n in dirnames if not n.startswith(".") and n not in ("node_modules", "__pycache__")]
        if "app.yaml" in filenames and Path(dirpath).name == arg:
            matches.append(Path(dirpath))
    if len(matches) > 1:
        sys.exit(f"More than one app is named '{arg}': "
                 + ", ".join(str(m.relative_to(ROOT)) for m in matches)
                 + ". Pass a file path inside the one you want.")
    return matches[0] if matches else None


def push(board, items, remote_dir):
    """Copy local files/folders into remote_dir on the board."""
    kind, target = board
    if kind == "adb":
        for item in items:
            run(["adb", "push", item, f"{remote_dir}/"])
    else:
        run(["scp", *SSH_OPTS, "-r", "-q", *items, f"{target}:{remote_dir}/"])


def main():
    # python deploy.py --cmd "some command"  -> run it on the board
    if len(sys.argv) >= 2 and sys.argv[1] == "--cmd":
        if len(sys.argv) < 3:
            sys.exit('Usage: python deploy.py --cmd "command to run on the board"')
        board = find_board()
        r = remote(board, " ".join(sys.argv[2:]), check=False)
        sys.exit(r.returncode if r else 1)

    # python deploy.py --shell  -> interactive terminal on the board
    if len(sys.argv) >= 2 and sys.argv[1] == "--shell":
        board = find_board()
        kind, target = board
        cmd = ["adb", "shell"] if kind == "adb" else ["ssh", *SSH_OPTS, "-t", target]
        r = run(cmd, check=False)
        sys.exit(r.returncode if r else 1)

    # python deploy.py --python  -> Python prompt in the running app's container,
    # or on the board's own Linux side if no app is running.
    if len(sys.argv) >= 2 and sys.argv[1] == "--python":
        board = find_board()
        r = remote(board, "docker ps --format '{{.Names}}' | grep -- '-main-1$'",
                   check=False, quiet=True, timeout=15)
        running = r.stdout.decode().split() if r is not None and r.returncode == 0 else []
        if running:
            container = running[0]
            print(f"Python inside the running app '{container[:-len('-main-1')]}' "
                  "(its virtualenv, so arduino.app_utils imports work)", flush=True)
            # Use the app's own virtualenv so arduino.app_utils etc. can be imported.
            cmd = f"docker exec -it {container} /app/.cache/.venv/bin/python"
        else:
            print("No app is running; Python on the board's Linux side", flush=True)
            cmd = "python3"
        r = remote(board, cmd, tty=True, check=False)
        sys.exit(r.returncode if r else 1)

    if len(sys.argv) < 2 or not sys.argv[1]:
        sys.exit("Open a file inside the app you want to run, then try again.")

    # Apps can live anywhere in this repo (e.g. HW4/minifig-light-tracker), so walk up from
    # the open file to the nearest folder with an app.yaml. On the board every
    # app goes in ArduinoApps/<that folder's name>.
    app_dir = find_app_dir(sys.argv[1])
    if app_dir is None:
        sys.exit(f"'{sys.argv[1]}' isn't inside an App (no app.yaml above it). "
                 "Open a file inside an app folder.")
    app = app_dir.name
    print(f"App: {app_dir.relative_to(ROOT)}  ->  {REMOTE_DIR}/{app}", flush=True)

    board = find_board()
    remote_app = f"{REMOTE_DIR}/{app}"

    # The board runs one app at a time, so stop whichever app is running
    # (this one or another) before starting yours.
    r = remote(board, "docker ps --format '{{.Names}}' | grep -- '-main-1$'",
               check=False, quiet=True, timeout=15)
    running = [n[:-len("-main-1")] for n in r.stdout.decode().split()] if r and r.returncode == 0 else []
    for name in running:
        remote(board, f"arduino-app-cli app stop {REMOTE_DIR}/{name}", check=False, timeout=30)
    if app not in running:   # also covers an app stuck half-started
        remote(board, f"arduino-app-cli app stop {remote_app}", check=False, timeout=30)

    # Copy only your source files. Hidden folders like .cache hold the
    # Python environment App Lab builds on the board, so they're skipped.
    # scp overwrites changed files but does NOT delete files you removed
    # locally; delete those on the board by hand.
    skip = {"__pycache__", "node_modules"}
    items = [str(p) for p in sorted(app_dir.iterdir())
             if not p.name.startswith(".") and p.name not in skip]
    remote(board, f"mkdir -p {remote_app}")
    push(board, items, remote_app)

    # Start it
    remote(board, f"arduino-app-cli app start {remote_app}", tty=True)

    # The app's own print/log output only lives in the container logs, so show
    # the first few seconds of it here, plus where to find the web page.
    time.sleep(5)
    remote(board, f"arduino-app-cli app logs {remote_app} --tail 15", check=False)
    url = web_url(board)
    if url:
        print(f"\nIf the app has a web page, open {url}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"Command failed with exit code {e.returncode}")
