"""Local smoke tests for the spectral-function drivers.

The LiF check is one electronic step on the included 5x5x5 parameter file.
It is a startup check, not a production spectrum.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOLSTEIN_STATES = 10
INITIAL_K = 2


def run_cmd(args, cwd, env=None):
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {args}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def holstein_input(method, nsteps, seed):
    return textwrap.dedent(
        f"""\
        Model                = holstein
        Method               = {method}
        System               = pc
        $dtN                 = 0.05
        $dtE                 = 0.01
        $NTraj               = 1
        $NSteps              = {nsteps}
        $nskip               = 1
        $SEED                = {seed}
        """
    )


class SmokeTest(unittest.TestCase):
    def test_runtime_dependencies_import(self):
        import numba
        import scipy

        self.assertTrue(callable(numba.jit))
        self.assertTrue(hasattr(scipy, "__version__"))

    def test_lif_parameter_file_matches_the_model(self):
        # Shapes follow Model/lif.py: a 5x5x5 cell, 3 hole bands, 6 phonon branches.
        data = np.load(ROOT / "Model" / "LiF_g_nijkq_N5_bands_3_4_5.npz")
        self.assertEqual(data["e_ik"].shape, (3, 125))
        self.assertEqual(data["w_nq"].shape, (6, 125))
        self.assertEqual(data["g_nijkq"].shape, (6, 3, 3, 125, 125))

    def test_local_example_prepares_one_momentum(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = run_cmd(
                [sys.executable, str(ROOT / "examples" / "run_holstein_local.py"), tmp],
                cwd=ROOT,
            )
            self.assertIn("mfe-holstein.txt", completed.stdout)
            data = np.loadtxt(Path(tmp) / "mfe-holstein.txt", dtype=complex)

        self.assertEqual(data.shape, (20, HOLSTEIN_STATES + 1))
        self.assertTrue(np.all(np.isfinite(data)))
        np.testing.assert_allclose(data[:, 0].real, np.arange(20) * 0.05)
        amplitudes = data[0, 1:]
        self.assertAlmostEqual(amplitudes[INITIAL_K], 1 + 0j, places=6)
        others = np.delete(amplitudes, INITIAL_K)
        self.assertTrue(np.all(np.abs(others) < 1e-8))
        self.assertAlmostEqual(np.sum(np.abs(amplitudes) ** 2), 1.0, places=6)

    def test_run_py_local_leaves_scratch_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scratch = tmp_path / "scratch"
            out = tmp_path / "out"
            input_path = tmp_path / "local.txt"
            input_path.write_text(holstein_input("mfe", nsteps=2, seed=1))
            env = os.environ.copy()
            env["SCRATCH"] = str(scratch)
            run_cmd(
                [sys.executable, str(ROOT / "run.py"), str(input_path), str(out)],
                cwd=ROOT,
                env=env,
            )
            data = np.loadtxt(out / "mfe-holstein.txt", dtype=complex)
            self.assertEqual(data.shape, (2, HOLSTEIN_STATES + 1))
            self.assertFalse((scratch / "NAMD").exists())

    def test_cluster_copy_is_preferred_over_a_workspace_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fold = Path(tmp) / "run"
            copied = fold / "examples"
            copied.mkdir(parents=True)
            (copied / "holstein_mfe.txt").write_text(
                holstein_input("mash", nsteps=2, seed=3)
            )
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "serial.py"),
                    "examples/holstein_mfe.txt",
                    str(fold),
                ],
                cwd=ROOT,
            )
            self.assertTrue((fold / "mash-holstein.txt").is_file())
            self.assertFalse((fold / "mfe-holstein.txt").exists())

    def test_mash_trajectory_is_finite_and_repeatable(self):
        def propagate(destination, seed):
            with tempfile.TemporaryDirectory() as tmp:
                input_path = Path(tmp) / "mash.txt"
                input_path.write_text(holstein_input("mash", nsteps=8, seed=seed))
                run_cmd(
                    [
                        sys.executable,
                        str(ROOT / "serial.py"),
                        str(input_path),
                        str(destination),
                    ],
                    cwd=ROOT,
                )
            return np.loadtxt(destination / "mash-holstein.txt", dtype=complex)

        with tempfile.TemporaryDirectory() as tmp:
            first_dir = Path(tmp) / "a"
            second_dir = Path(tmp) / "b"
            first = propagate(first_dir, seed=11)
            second = propagate(second_dir, seed=11)

        self.assertEqual(first.shape, (8, HOLSTEIN_STATES + 1))
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertGreater(np.max(np.abs(first[:, 1:])), 0)
        np.testing.assert_allclose(first, second)

    def test_lif_one_electronic_step_starts_at_one(self):
        # dtN == dtE gives a single electronic update. The saved sample is the
        # correlation before that update, which is 1 on the prepared state.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "lif.txt"
            input_path.write_text(
                textwrap.dedent(
                    """\
                    Model                = lif
                    Method               = mfe_kspace
                    System               = pc
                    $NSteps              = 1
                    $NTraj               = 1
                    $dtN                 = 1
                    $dtE                 = 1
                    $nskip               = 1
                    $SEED                = 1
                    """
                )
            )
            out = tmp_path / "out"
            run_cmd(
                [sys.executable, str(ROOT / "serial.py"), str(input_path), str(out)],
                cwd=ROOT,
            )
            data = np.atleast_2d(
                np.loadtxt(out / "mfe_kspace-lif.txt", dtype=complex)
            )

        self.assertEqual(data.shape, (1, 2))
        self.assertAlmostEqual(data[0, 0].real, 0.0, places=8)
        self.assertAlmostEqual(data[0, 1], 1 + 0j, places=6)

    def test_avg_means_complex_trajectory_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fold = Path(tmp)
            rows_a = [
                (0.0, 1 + 0j, 0 + 1j),
                (0.05, 0.5 + 0.25j, -0.25 + 0j),
            ]
            rows_b = [
                (0.0, 0 + 0j, 0 + 0j),
                (0.05, -0.5 + 0.25j, 0.25 + 1j),
            ]
            for name, rows in (("demo-0.txt", rows_a), ("demo-1.txt", rows_b)):
                with open(fold / name, "w") as handle:
                    for row in rows:
                        handle.write("\t".join(str(value) for value in row) + "\n")
            run_cmd([sys.executable, str(ROOT / "avg.py"), str(fold)], cwd=ROOT)
            averaged = np.loadtxt(fold / "demo.txt", dtype=complex)

        expected = np.array(
            [
                [0.0, 0.5 + 0j, 0 + 0.5j],
                [0.05, 0 + 0.25j, 0 + 0.5j],
            ],
            dtype=complex,
        )
        np.testing.assert_allclose(averaged, expected)


if __name__ == "__main__":
    unittest.main()
