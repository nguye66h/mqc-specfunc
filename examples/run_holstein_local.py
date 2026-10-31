#!/usr/bin/env python3
"""Run one short Holstein mean-field Ehrenfest trajectory on this machine.

Writes trajectory-averaged momentum-space amplitudes to
examples/output/mfe-holstein.txt (or to the directory given as the first
argument). Launch it from anywhere; the script itself selects the
repository root. A converged spectrum needs many trajectories. This run
checks that the local path works.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(__file__).resolve().parent / "holstein_mfe.txt"
DEFAULT_OUT = Path(__file__).resolve().parent / "output"
# Model/holstein.py prepares the carrier at this momentum index.
INITIAL_K = 2


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "serial.py"), str(INPUT), str(out)]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)

    result = out / "mfe-holstein.txt"
    data = np.loadtxt(result, dtype=complex)
    amplitudes = data[:, 1:]
    prepared = amplitudes[0, INITIAL_K]
    print(f"Wrote {result}")
    print(f"Stored samples: {data.shape[0]}")
    print(f"Time grid: {data[0, 0].real} to {data[-1, 0].real}")
    print(f"Amplitude at t=0, momentum index {INITIAL_K}: {prepared}")
    print(
        f"Absolute amplitude at the last stored time, same momentum: "
        f"{abs(amplitudes[-1, INITIAL_K])}"
    )
    if abs(prepared - 1) > 1e-6:
        raise SystemExit(
            "The prepared momentum amplitude at t=0 was not 1. "
            "The Holstein initial state or the output layout has changed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
