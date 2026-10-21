"""Saving and loading labeled recordings (laptop side, shared by both parts).

A recording is data/<person>/<activity>_<YYYYmmdd_HHMMSS>.csv with columns
seq, t, then classifiers.COLUMNS (g and deg/s), one row per 50 Hz sample.
"""

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
import classifiers as C

CSV_HEADER = ["seq", "t"] + C.COLUMNS


def recording_path(person, activity, data_dir=config.DATA_DIR):
    folder = Path(data_dir) / person.strip().replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{activity}_{time.strftime('%Y%m%d_%H%M%S')}.csv"


def save_recording(path, seqs, rows):
    """seqs: sample sequence numbers (unwrapped ints); rows: (n, 12) array."""
    seqs = np.asarray(seqs)
    t = (seqs - seqs[0]) / C.SAMPLE_HZ
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for s, ti, row in zip(seqs, t, rows):
            w.writerow([int(s), f"{ti:.2f}"] + [f"{v:.4f}" for v in row])


@dataclass
class Recording:
    person: str
    label: str
    path: Path
    segments: list = field(default_factory=list)   # continuous (n, 12) pieces

    @property
    def seconds(self):
        return sum(len(s) for s in self.segments) / C.SAMPLE_HZ

    def windows(self, hop=C.HOP):
        """All decision windows the board would see: (k, WINDOW, 12)."""
        out = [seg[i:i + C.WINDOW] for seg in self.segments
               for i in range(0, len(seg) - C.WINDOW + 1, hop)]
        return np.stack(out) if out else np.zeros((0, C.WINDOW, len(C.COLUMNS)))


def load_recording(path):
    path = Path(path)
    label = path.stem.split("_")[0]
    if label not in C.STATES:
        return None
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if len(data) == 0:
        return None
    seq, rows = data[:, 0].astype(np.int64), data[:, 2:]
    trim = int(config.TRIM_S * C.SAMPLE_HZ)
    seq, rows = seq[trim:len(seq) - trim], rows[trim:len(rows) - trim]
    rec = Recording(person=path.parent.name, label=label, path=path)
    if len(rows) == 0:
        return rec
    breaks = np.where(np.diff(seq) > config.MAX_GAP_ROWS)[0] + 1
    rec.segments = [s for s in np.split(rows, breaks) if len(s)]
    return rec


def load_recordings(data_dir=config.DATA_DIR):
    recs = []
    for path in sorted(Path(data_dir).glob("*/*.csv")):
        rec = load_recording(path)
        if rec is not None and rec.segments:
            recs.append(rec)
    return recs


def summarize(recs, log=print):
    """Print seconds of data per person and activity."""
    people = sorted({r.person for r in recs})
    log(f"{'person':<14}" + "".join(f"{s:>10}" for s in C.STATES))
    for p in people:
        secs = {s: sum(r.seconds for r in recs if r.person == p and r.label == s) for s in C.STATES}
        log(f"{p:<14}" + "".join(f"{secs[s]:>9.0f}s" for s in C.STATES))


def stack_windows(recs, hop=C.HOP):
    """Windows from all recordings -> (windows, labels, persons, recording index)."""
    xs, labels, persons, rec_ids = [], [], [], []
    for i, r in enumerate(recs):
        w = r.windows(hop)
        xs.append(w)
        labels += [r.label] * len(w)
        persons += [r.person] * len(w)
        rec_ids += [i] * len(w)
    if not xs:
        return np.zeros((0, C.WINDOW, len(C.COLUMNS))), np.array([]), np.array([]), np.array([])
    return np.concatenate(xs), np.array(labels), np.array(persons), np.array(rec_ids)


def confusion(true, pred, classes=C.STATES, log=print):
    """Print a confusion matrix (rows = true, columns = predicted). Returns accuracy."""
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    cols = list(classes) + (["unsure"] if any(p is None for p in pred) else [])
    log(f"{'true/pred':<12}" + "".join(f"{c:>10}" for c in cols) + f"{'correct':>10}")
    for t in classes:
        mask = true == t
        if not mask.any():
            continue
        counts = [np.sum(pred[mask] == c) if c != "unsure" else np.sum([p is None for p in pred[mask]])
                  for c in cols]
        log(f"{t:<12}" + "".join(f"{n:>10}" for n in counts) + f"{counts[cols.index(t)] / mask.sum():>9.0%}")
    acc = float(np.mean(true == pred)) if len(true) else 0.0
    log(f"overall accuracy {acc:.1%} over {len(true)} windows")
    return acc
