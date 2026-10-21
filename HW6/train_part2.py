"""PART 2 one-time training: a small 1D CNN on all 6 IMU channels.

Reads every recording in data/, trains with augmentation (so it carries over to
people who weren't recorded), validates on held-out data, then trains the final
model on everything and exports it as plain JSON (part2_model.json). The board runs it with numpy
(classifiers.Part2NeuralNet). Before saving, this script checks that the numpy
version gives the same output as PyTorch.

Validation (printed, not used by the board):
  - >= 2 people: leave-one-person-out (train on everyone else, test on them)
  - 1 person, >= 2 recordings of every activity: hold out the newest recording of each
  - otherwise: hold out the last 25% of every recording

Normally this runs inside the GUI's calibration wizard (Shift+C in gui.py),
which also installs the result, keeps the old one in calibration_history/ and
deploys. Run from the command line it only writes --out:

    .venv/Scripts/python HW6/train_part2.py --out <file>
    .venv/Scripts/python HW6/train_part2.py --data HW6/testing/data_synthetic --out <file>
"""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

import config
import classifiers as C
import dataset

# Fixed input scales (not learned per user): accel / 2 g, gyro / 500 deg/s.
SCALES = [2.0, 2.0, 2.0, 500.0, 500.0, 500.0]

# Augmentation ranges
ROTATE_DEG = 20        # strap sits at a slightly different angle
AMPLITUDE = (0.8, 1.25)  # softer / harder movement
STRETCH = (0.8, 1.25)    # slower / faster cadence
NOISE_ACCEL_G, NOISE_GYRO_DPS = 0.02, 2.0

EPOCHS, PATIENCE, BATCH, LR, WEIGHT_DECAY = 120, 20, 64, 2e-3, 1e-4
SEED = 0


class Cancelled(Exception):
    """Raised inside training when should_stop() returns True."""


class ActivityCNN(nn.Module):
    """Must match classifiers.Part2NeuralNet (same layers, same names)."""

    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 16, 7)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.conv3 = nn.Conv1d(32, 32, 5)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):                       # x: (batch, 6, WINDOW)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        return self.fc(self.drop(x.mean(dim=2)))


# ---------------------------------------------------------------- data -------

def channels(seg):
    """(n, 12) rows -> (n, 6) the Part 2 channels."""
    return seg[:, [C.COL[c] for c in C.PART2_CHANNELS]]


def random_rotation(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(-ROTATE_DEG, ROTATE_DEG))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def augmented_window(seg6, rng):
    """A random 2 s window from a segment with all augmentations applied: (WINDOW, 6)."""
    stretch = rng.uniform(*STRETCH)
    src_len = min(len(seg6), int(round(C.WINDOW * stretch)))
    start = rng.integers(0, len(seg6) - src_len + 1)
    pos = np.linspace(0, src_len - 1, C.WINDOW)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, src_len - 1)
    frac = (pos - i0)[:, None]
    src = seg6[start:start + src_len]
    w = src[i0] * (1 - frac) + src[i1] * frac
    acc, gyro = w[:, :3], w[:, 3:]
    r = random_rotation(rng)
    acc, gyro = acc @ r.T, gyro @ r.T
    k = rng.uniform(*AMPLITUDE)
    mean = acc.mean(axis=0)
    acc = mean + k * (acc - mean)             # scale the movement, not gravity
    gyro = gyro * k
    acc = acc + rng.normal(0, NOISE_ACCEL_G, acc.shape)
    gyro = gyro + rng.normal(0, NOISE_GYRO_DPS, gyro.shape)
    return np.hstack([acc, gyro])


def to_input(windows6):
    """(k, WINDOW, 6) -> (k, 6, WINDOW) float32 scaled."""
    return (np.transpose(windows6, (0, 2, 1)) / np.array(SCALES)[None, :, None]).astype(np.float32)


class Pieces:
    """Continuous pieces of data, each with a label index."""

    def __init__(self):
        self.segs, self.labels = [], []

    def add(self, seg, label):
        if len(seg) >= C.WINDOW:
            self.segs.append(channels(seg))
            self.labels.append(C.STATES.index(label))

    def epoch(self, rng):
        """One epoch of augmented windows, about one per HOP of data."""
        xs, ys = [], []
        for seg, y in zip(self.segs, self.labels):
            for _ in range(max(1, (len(seg) - C.WINDOW) // C.HOP + 1)):
                xs.append(augmented_window(seg, rng))
                ys.append(y)
        return to_input(np.stack(xs)), np.array(ys)

    def plain_windows(self):
        """Un-augmented windows exactly as the board sees them."""
        xs, ys = [], []
        for seg, y in zip(self.segs, self.labels):
            for i in range(0, len(seg) - C.WINDOW + 1, C.HOP):
                xs.append(seg[i:i + C.WINDOW])
                ys.append(y)
        if not xs:
            return np.zeros((0, 6, C.WINDOW), np.float32), np.zeros(0, int)
        return to_input(np.stack(xs)), np.array(ys)

    def class_weights(self):
        counts = np.zeros(len(C.STATES))
        for seg, y in zip(self.segs, self.labels):
            counts[y] += len(seg)
        counts = np.maximum(counts, 1)
        return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def split(recs):
    """-> list of (name, train Pieces, val Pieces) folds and a description."""
    people = sorted({r.person for r in recs})
    folds = []
    if len(people) >= 2:
        for p in people:
            tr, va = Pieces(), Pieces()
            for r in recs:
                for seg in r.segments:
                    (va if r.person == p else tr).add(seg, r.label)
            folds.append((f"held out {p}", tr, va))
        return folds, "leave-one-person-out"
    per_label = {s: [r for r in recs if r.label == s] for s in C.STATES}
    tr, va = Pieces(), Pieces()
    if all(len(v) >= 2 for v in per_label.values()):
        for s, rs in per_label.items():
            rs = sorted(rs, key=lambda r: r.path.name)
            for r in rs:
                for seg in r.segments:
                    (va if r is rs[-1] else tr).add(seg, s)
        return [("held out newest recordings", tr, va)], "newest recording of each activity held out"
    for r in recs:
        for seg in r.segments:
            cut = int(len(seg) * 0.75)
            tr.add(seg[:cut - C.WINDOW // 2], r.label)       # gap so train/val windows don't overlap
            va.add(seg[cut:], r.label)
    return [("held out last 25%", tr, va)], "last 25% of each recording held out"


# ---------------------------------------------------------------- training ---

def train(tr, va=None, epochs=EPOCHS, seed=SEED, verbose=True, log=print, should_stop=None):
    """Train a model. With `va`, early-stops on its loss. -> (model, best epoch, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActivityCNN(len(C.STATES))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=tr.class_weights())
    if va is not None:
        xv, yv = va.plain_windows()
        xv, yv = torch.from_numpy(xv), torch.from_numpy(yv)
    best = (float("inf"), 0, None)
    for epoch in range(1, epochs + 1):
        if should_stop and should_stop():
            raise Cancelled()
        model.train()
        x, y = tr.epoch(rng)
        perm = rng.permutation(len(y))
        x, y = torch.from_numpy(x[perm]), torch.from_numpy(y[perm])
        total = 0.0
        for i in range(0, len(y), BATCH):
            opt.zero_grad()
            loss = loss_fn(model(x[i:i + BATCH]), y[i:i + BATCH])
            loss.backward()
            opt.step()
            total += loss.item() * len(y[i:i + BATCH])
        if va is None or len(yv) == 0:
            continue
        model.eval()
        with torch.no_grad():
            out = model(xv)
            vloss = nn.functional.cross_entropy(out, yv).item()
            vacc = (out.argmax(1) == yv).float().mean().item()
        if vloss < best[0]:
            best = (vloss, epoch, {k: v.clone() for k, v in model.state_dict().items()})
        if verbose and (epoch % 10 == 0 or epoch == 1):
            log(f"  epoch {epoch:3d}  train loss {total / len(y):.3f}  val loss {vloss:.3f}  val acc {vacc:.1%}")
        if epoch - best[1] >= PATIENCE:
            break
    if best[2] is not None:
        model.load_state_dict(best[2])
    model.eval()
    return model, (best[1] or epochs)


def evaluate(model, pieces):
    x, y = pieces.plain_windows()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x)), 1).numpy()
    pred = [C.STATES[i] if p.max() >= C.PART2_MIN_CONFIDENCE else None
            for i, p in zip(probs.argmax(1), probs)]
    return [C.STATES[i] for i in y], pred


def export(model, info):
    weights = {k: v.detach().cpu().numpy().astype(float).round(7).tolist()
               for k, v in model.state_dict().items()}
    return {"classes": C.STATES, "scales": SCALES, "window": C.WINDOW, "sample_hz": C.SAMPLE_HZ,
            **info, "weights": weights}


def check_numpy_matches(model, model_json, pieces, log=print):
    net = C.Part2NeuralNet(model_json)
    x, _ = pieces.plain_windows()
    x = x[:50]
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(x)), 1).numpy()
    ours = np.stack([net.probabilities_from_input(xi.astype(np.float64)) for xi in x])
    diff = float(np.abs(ref - ours).max())
    log(f"numpy vs PyTorch: max probability difference {diff:.2e} on {len(x)} windows")
    if diff > 1e-4:
        raise RuntimeError("The numpy forward pass doesn't match PyTorch; not saving.")


def train_model(recs, log=print, verbose=True, should_stop=None):
    """Recordings -> the part2_model.json contents (also used by the GUI)."""
    torch.set_num_threads(4)
    dataset.summarize(recs, log)
    missing = [s for s in C.STATES if not any(r.label == s for r in recs)]
    if missing:
        raise ValueError(f"No recordings for {missing}. Record them first.")

    folds, how = split(recs)
    log(f"\nValidation: {how}")
    all_true, all_pred, best_epochs = [], [], []
    for name, tr, va in folds:
        log(f"-- {name}: {len(tr.plain_windows()[1])} train / {len(va.plain_windows()[1])} val windows")
        model, best_epoch = train(tr, va, verbose=verbose, log=log, should_stop=should_stop)
        true, pred = evaluate(model, va)
        acc = np.mean(np.array(true, dtype=object) == np.array(pred, dtype=object))
        log(f"   best epoch {best_epoch}, accuracy {acc:.1%}")
        all_true += true
        all_pred += pred
        best_epochs.append(best_epoch)
    log("\nValidation windows (before smoothing; 'unsure' = confidence below "
        f"{C.PART2_MIN_CONFIDENCE:.0%}, keeps the previous state):")
    val_acc = dataset.confusion(all_true, all_pred, log=log)

    final_epochs = int(np.median(best_epochs))
    log(f"\nFinal model: all data, {final_epochs} epochs")
    everything = Pieces()
    for r in recs:
        for seg in r.segments:
            everything.add(seg, r.label)
    model, _ = train(everything, None, epochs=final_epochs, verbose=False, log=log, should_stop=should_stop)
    info = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "validation": how,
            "validation_accuracy": round(float(val_acc), 4), "epochs": final_epochs,
            "people": sorted({r.person for r in recs}),
            "parameters": int(sum(p.numel() for p in model.parameters()))}
    model_json = export(model, info)
    check_numpy_matches(model, model_json, everything, log=log)
    return model_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--out", required=True, help="where to write the result (install it with the GUI)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    try:
        model_json = train_model(dataset.load_recordings(args.data), verbose=not args.quiet)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"{exc} ({args.data})")
    with open(args.out, "w") as f:
        json.dump(model_json, f, separators=(",", ":"))
    print(f"Saved {args.out} ({model_json['parameters']} weights). "
          "To use it, calibrate through the GUI wizard (Shift+C).")


if __name__ == "__main__":
    main()
