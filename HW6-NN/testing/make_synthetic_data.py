"""Write synthetic recordings to HW6/testing/data_synthetic/ (3 fake people).

Only for testing the code; the real calibration uses HW6/data/.

    .venv/Scripts/python HW6/testing/make_synthetic_data.py
"""

import shutil
from pathlib import Path

import numpy as np

import synth
import classifiers as C
import dataset

OUT = Path(__file__).resolve().parent / "data_synthetic"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    seed = 0
    for p, person_seed in [("sim_ana", 1), ("sim_ben", 2), ("sim_cy", 3)]:
        person = synth.make_person(person_seed)
        for label in C.STATES:
            for take, secs in enumerate([40, 20]):
                seed += 1
                rows = synth.recording_rows(label, secs, person, seed)
                seqs = np.arange(len(rows)) + 1000 * seed
                if take == 1:                     # a dropped stretch, to exercise gap splitting
                    seqs[len(seqs) // 2:] += 30
                folder = OUT / p
                folder.mkdir(parents=True, exist_ok=True)
                dataset.save_recording(folder / f"{label}_2026010{take + 1}_000000.csv", seqs, rows)
    print(f"Wrote synthetic recordings to {OUT}")


if __name__ == "__main__":
    main()
