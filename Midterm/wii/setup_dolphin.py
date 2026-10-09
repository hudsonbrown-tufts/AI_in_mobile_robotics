"""Configure Dolphin to use the LEGO Wii Remote (run with Dolphin CLOSED).

  1. Enables Dolphin's DSU client and adds our server (127.0.0.1:26760),
     described as "LEGO Wii Remote" -- Dolphin names the device after that
     description, so it shows up as  DSUClient/0/LEGO Wii Remote.
  2. Installs profiles/LEGO Wii Remote.ini into Dolphin's Wiimote profiles.
  3. Applies that profile to Wii Remote 1 (emulated, no extension).

Every file it changes is backed up first as <name>.bak-<timestamp>.

The input names in the profile were read out of Dolphin 2609a's own
Dolphin.exe (its DSU inputs are: Pad N/S/E/W, Square, Cross, Circle,
Triangle, L1, R1, L2, R2, L3, R3, Share, Options, PS, Left/Right X/Y +/-,
Accel Up/Down/Left/Right/Forward/Backward, Gyro Pitch/Roll/Yaw ...).

Usage:
    python setup_dolphin.py                 # find the Dolphin user folder automatically
    python setup_dolphin.py --user-dir "C:\\path\\to\\Dolphin Emulator"
"""

import argparse
import configparser
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PROFILE_NAME = "LEGO Wii Remote"
PROFILE_SRC = HERE / "profiles" / f"{PROFILE_NAME}.ini"
DSU_DESCRIPTION = PROFILE_NAME
DSU_ADDRESS, DSU_PORT = "127.0.0.1", 26760


def find_user_dir():
    for p in (Path(os.environ.get("APPDATA", "")) / "Dolphin Emulator",
              Path.home() / "Documents" / "Dolphin Emulator"):
        if (p / "Config").is_dir():
            return p
    return None


def dolphin_running():
    try:
        import subprocess
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Dolphin.exe"], capture_output=True, text=True).stdout
        return "Dolphin.exe" in out
    except Exception:
        return False


def read_ini(path):
    ini = configparser.ConfigParser(interpolation=None, delimiters=("=",))
    ini.optionxform = str          # keep key case ("Buttons/A")
    if path.exists():
        ini.read(path, encoding="utf-8")
    return ini


def write_ini(ini, path):
    with open(path, "w", encoding="utf-8") as f:
        for section in ini.sections():
            f.write(f"[{section}]\n")
            for k, v in ini[section].items():
                f.write(f"{k} = {v}\n")


def backup(path, stamp):
    if path.exists():
        dest = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, dest)
        print(f"  backed up {path.name} -> {dest.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-dir", help="Dolphin user folder (contains Config/)")
    args = parser.parse_args()

    user = Path(args.user_dir) if args.user_dir else find_user_dir()
    if user is None or not (user / "Config").is_dir():
        sys.exit("Couldn't find Dolphin's user folder. Start Dolphin once, close it, or pass --user-dir.")
    if dolphin_running():
        sys.exit("Dolphin is running -- close it first (it rewrites its config files on exit).")
    config = user / "Config"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Dolphin user folder: {user}")

    # 1. DSU client
    dsu_path = config / "DSUClient.ini"
    backup(dsu_path, stamp)
    dsu = read_ini(dsu_path)
    if not dsu.has_section("Server"):
        dsu.add_section("Server")
    entry = f"{DSU_DESCRIPTION}:{DSU_ADDRESS}:{DSU_PORT};"
    others = [e + ";" for e in dsu["Server"].get("Entries", "").split(";")
              if e and not e.endswith(f":{DSU_ADDRESS}:{DSU_PORT}")]
    dsu["Server"]["Enabled"] = "True"
    dsu["Server"]["Entries"] = entry + "".join(others)
    write_ini(dsu, dsu_path)
    print(f"  DSU client enabled, server {entry}")

    # 2. Profile
    profile_dir = config / "Profiles" / "Wiimote"
    profile_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROFILE_SRC, profile_dir / PROFILE_SRC.name)
    print(f"  installed profile -> {profile_dir / PROFILE_SRC.name}")

    # 3. Wii Remote 1 = the profile
    wm_path = config / "WiimoteNew.ini"
    backup(wm_path, stamp)
    wm = read_ini(wm_path)
    profile = read_ini(PROFILE_SRC)["Profile"]
    if wm.has_section("Wiimote1"):
        wm.remove_section("Wiimote1")
    sections = ["Wiimote1"] + [s for s in wm.sections()]
    new = configparser.ConfigParser(interpolation=None, delimiters=("=",))
    new.optionxform = str
    for s in sections:
        new.add_section(s)
        src = profile if s == "Wiimote1" else wm[s]
        for k, v in src.items():
            new[s][k] = v
    write_ini(new, wm_path)
    print(f"  Wii Remote 1 now uses the '{PROFILE_NAME}' mapping (device {profile['Device']})")

    print("\nDone. Next: start  python wii_remote.py , then open Dolphin > Controllers > Wii Remote 1 >")
    print("Configure, and check the Device dropdown shows the DSUClient device.")


if __name__ == "__main__":
    main()
