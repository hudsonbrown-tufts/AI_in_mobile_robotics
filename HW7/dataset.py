"""Saving and loading HW7 recordings (laptop side).

A recording is data/<person>/<kind>_<YYYYmmdd_HHMMSS>.csv (columns seq, t, then
stepnet.COLUMNS in g and deg/s, one row per 50 Hz sample; the same format as
HW6) plus a .json next to it with what Hudson typed in:

    kind "hall"      one hallway length: {"steps", "distance_m", "hall_m"}
    kind "long"      a few minutes back and forth: {"steps", "lengths", "distance_m", "hall_m"}
                     (the final test; never trained on)
    kind "standing"  standing still: 0 steps, 0 m (no .json needed)

HW6's recordings of Hudson are reused too (config.HW6_EXTRA): standing as 0
steps, and walking as "walk_uncounted" (stride rate only, since nobody counted
its steps or measured its distance).
"""

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
import stepnet as S

CSV_HEADER = ["seq", "t"] + S.COLUMNS
KINDS = ("hall", "long", "standing")
HALLWAY_FILE = config.DATA_DIR / config.PERSON / "hallway.json"


def recording_path(kind, person=config.PERSON, data_dir=config.DATA_DIR):
    folder = Path(data_dir) / person
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.csv"


def save_recording(path, seqs, rows, meta=None):
    """seqs: sample sequence numbers (unwrapped ints); rows: (n, 12) array."""
    seqs = np.asarray(seqs)
    t = (seqs - seqs[0]) / S.SAMPLE_HZ
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for s, ti, row in zip(seqs, t, rows):
            w.writerow([int(s), f"{ti:.2f}"] + [f"{v:.4f}" for v in row])
    if meta is not None:
        save_meta(path, meta)


def save_meta(path, meta):
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def delete_recording(path):
    path = Path(path)
    for p in (path, path.with_suffix(".json")):
        if p.exists():
            p.unlink()


@dataclass
class Recording:
    path: Path
    kind: str                     # hall / long / standing / walk_uncounted
    seqs: np.ndarray              # unwrapped sample numbers
    rows: np.ndarray              # (n, 12)
    meta: dict = field(default_factory=dict)

    @property
    def name(self):
        return self.path.name

    @property
    def t(self):
        """Time of each row (s), holes included."""
        return (self.seqs - self.seqs[0]) / S.SAMPLE_HZ

    @property
    def seconds(self):
        return len(self.rows) / S.SAMPLE_HZ

    @property
    def steps(self):
        """Counted steps (None if nobody counted)."""
        if self.kind == "standing":
            return 0
        return self.meta.get("steps")

    @property
    def distance_m(self):
        if self.kind == "standing":
            return 0.0
        return self.meta.get("distance_m")

    def segments(self):
        """(start, end) row ranges with no hole bigger than MAX_GAP_ROWS."""
        breaks = np.where(np.diff(self.seqs) > config.MAX_GAP_ROWS)[0] + 1
        edges = [0] + list(breaks) + [len(self.rows)]
        return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def replies(self, n=6):
        """The recording cut into sketch-sized replies: (seq & 0xFFFF, rows)."""
        for i in range(0, len(self.rows), n):
            yield int(self.seqs[i]) % 65536, self.rows[i:i + n]


def load_recording(path, kind=None):
    path = Path(path)
    kind = kind or path.stem.split("_")[0]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if len(data) == 0:
        return None
    meta = {}
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return Recording(path=path, kind=kind, seqs=data[:, 0].astype(np.int64), rows=data[:, 2:], meta=meta)


def load_recordings(data_dir=config.DATA_DIR, include_hw6=True, log=None):
    """All recordings in data_dir (hall, long, standing) plus HW6's, sorted by name."""
    recs = []
    for path in sorted(Path(data_dir).glob("*/*.csv")):
        if path.stem.split("_")[0] not in KINDS:
            continue
        rec = load_recording(path)
        if rec is None:
            continue
        if rec.kind in ("hall", "long") and rec.steps is None:
            if log:
                log(f"  skipping {path.name}: no step count in its .json")
            continue
        recs.append(rec)
    if include_hw6 and config.HW6_DATA_DIR.exists():
        for pattern, kind in config.HW6_EXTRA.items():
            for path in sorted(config.HW6_DATA_DIR.glob(pattern)):
                rec = load_recording(path, kind)
                if rec is not None:
                    recs.append(rec)
    return recs


def load_hallway_m():
    try:
        return float(json.loads(HALLWAY_FILE.read_text(encoding="utf-8"))["hall_m"])
    except (FileNotFoundError, KeyError, ValueError):
        return None


def save_hallway_m(meters):
    HALLWAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HALLWAY_FILE.write_text(json.dumps({"hall_m": meters}), encoding="utf-8")


def summarize(recs, log=print):
    for kind in ("hall", "long", "standing", "walk_uncounted"):
        rs = [r for r in recs if r.kind == kind]
        if rs:
            steps = sum(r.steps or 0 for r in rs)
            log(f"  {kind:<15} {len(rs):2d} recordings, {sum(r.seconds for r in rs):5.0f} s"
                + (f", {steps} counted steps" if kind in ("hall", "long") else ""))
