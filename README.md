# mqc-specfunc

Mixed quantum-classical propagation of the time-domain signal that enters a polaron spectral function. The carrier is treated quantum mechanically and the phonons are classical. The spectral function A(k, ω) is the Fourier transform of that signal; this repository writes the signal, and the definition used in the accompanying paper is given below.

The methods and models are:

| Model | File | What it is | Methods |
| --- | --- | --- | --- |
| Holstein | `Model/holstein.py` | One-dimensional periodic Holstein chain. Defaults: 10 sites, J = 1, g = 1, ω = 1, T = 0, carrier prepared at momentum index 2. | `mfe`, `mash` |
| LiF | `Model/lif.py` | Hole polaron on a 5×5×5 cell with 3 valence bands. Electron-phonon parameters are read from `Model/LiF_g_nijkq_N5_bands_3_4_5.npz`. | `mfe_kspace` |

`mfe` is mean-field Ehrenfest. `mash` is the multi-state mapping approach to surface hopping (MASH). `mfe_kspace` is mean-field Ehrenfest in the band/momentum basis used for LiF. Use the pairs in the table. The other combinations call model functions with a different signature and stop.

## Author

Haimi Nguyen, ORCID [0000-0003-1685-445X](https://orcid.org/0000-0003-1685-445X).

## Citation

H. Nguyen, A. Mandal, A. Mahajan, and D. R. Reichman, "Mixed quantum-classical methods for polaron spectral functions," *J. Chem. Phys.* **163**, 114105 (2025). [https://doi.org/10.1063/5.0281529](https://doi.org/10.1063/5.0281529)

Preprint: [arXiv:2505.13365](https://arxiv.org/abs/2505.13365).

`CITATION.cff` carries the same record. The published calculations used 10,000 trajectories for Holstein MFE, 30,000 for Holstein MASH, and 200 for LiF MFE. The example below is one short trajectory and is only a functionality check.

## Credit

The job driver (`run.py`, `serial.py`, `avg.py`) and the MFE and MASH propagators are based on the SemiClassical-NAMD code from Arkajit Mandal's group:

[https://github.com/mandalgrouptamu/SemiClassical-NAMD](https://github.com/mandalgrouptamu/SemiClassical-NAMD)

This repository adds the Holstein and LiF spectral-function models, records momentum-space amplitudes (and, for LiF, the correlation sample), and adapts submission to SLURM. The MASH implementation follows J. E. Runeson and D. E. Manolopoulos, *J. Chem. Phys.* **159**, 094115 (2023), [https://doi.org/10.1063/5.0158147](https://doi.org/10.1063/5.0158147).

The repository is released under the MIT license. See `LICENSE`.

## Requirements

Python 3. The GitHub Actions smoke test runs on Python 3.12. Dependencies are NumPy, Numba, and SciPy. Numba is imported by the LiF model and by `mfe_kspace`. SciPy is what Numba uses for the jitted matrix-vector products in that propagator. The Holstein example uses NumPy.

```bash
python -m pip install -r requirements.txt
```

Run the commands below from the repository root so the model and method imports resolve.

## Local example

This Holstein mean-field run stays on the local machine. It uses one trajectory, 20 nuclear steps, and the fixed seed in `examples/holstein_mfe.txt`.

```bash
python examples/run_holstein_local.py
```

The script writes `examples/output/mfe-holstein.txt` and prints the prepared-momentum amplitude. At t = 0 that amplitude is 1. An output directory can be passed as the first argument.

The same input can be launched through the driver:

```bash
python run.py examples/holstein_mfe.txt examples/output
```

`System = pc` (or `serial`) selects this local path. `run.py` then calls `serial.py` with the same Python interpreter.

To run MASH instead, copy the example input, set `Method = mash`, and call `serial.py` with that file. MASH reads the same Holstein model.

## Input file

`input.txt` is the cluster-oriented default (`System = slurm`, Holstein, MASH). Each keyword is the first line that contains that word, so keep `Model`, `Method`, `System`, and `Cpus` on their own lines and ahead of any comment that repeats them.

| Key | Role |
| --- | --- |
| `Model` | `holstein` or `lif` |
| `Method` | `mfe`, `mash`, or `mfe_kspace` |
| `System` | `pc` or `serial` runs locally. `slurm` submits the batch script. |
| `Cpus` | Number of independent SLURM tasks. Each task runs `NTraj` trajectories. |

Holstein Hamiltonian parameters are the class attributes in `Model/holstein.py`: `NStates`, `J`, `g`, `w`, and `T`. Edit that file to change them. The on-site coupling implemented there is `g * sqrt(2 * w**3) * R` on each site, the bare phonon force is `w**2 * R`, and the initial `R` and `P` are drawn from the Wigner distribution of the harmonic bath.

A line whose name starts with `$` is assigned onto the run object. The propagators read these overrides:

- `NSteps`, `NTraj`, `nskip`, `dtN`, `M`, `initState`, `SEED`
- `dtE` for mean-field Ehrenfest
- `mo` and `mk` for MASH

`$NStates`, `$J`, `$g`, `$w`, and `$T` are stored on that object, while the Holstein energy and force functions continue to read the class values. Change those in `Model/holstein.py`.

`nskip` subsamples the Holstein MFE and MASH output. The LiF driver records every electronic step; keep its `nskip` at 1.

`SEED` fixes NumPy's random stream for the phonon sampling and, in MASH, the mapping variables. The published ensembles leave the seed unset so successive tasks differ.

## Output and the spectral function

For Holstein `mfe` and `mash`, `serial.py` writes

```text
{folder}/{method}-{model}.txt
```

and, when a task id is passed as the third argument, `{method}-{model}-{id}.txt`.

Each row is a time `t = 0, nskip * dtN, 2 * nskip * dtN, ...` followed by one complex value per momentum index `k = 2π j / N` with `j = 0, ..., N - 1`. The values are the trajectory average of the momentum-space amplitude. For mean-field Ehrenfest the column of the prepared momentum is the single-trajectory contribution to the correlation function C_k(t). MASH writes its coefficient estimator instead of the bare amplitude. Times and amplitudes are in the model's units (ℏ = 1).

For LiF, `mfe_kspace` writes one complex column, sampled every electronic step `dtE`. That sample is `corr` in `Method/mfe_kspace.py`: the electronic coefficient at `k_ind_of_c`, or its projection by `U` when `U` is set.

Average task files with a shared prefix:

```bash
python avg.py output
```

`avg.py` groups `name-*.txt` (the last `-` separates the task id) and writes `name.txt`, the arithmetic mean of the complex columns.

In the paper the retarded Green's function and the spectral function are

```text
G(k, t) = -i θ(t) C_k(t)

A(k, ω) = -(1/π) Im ∫_0^∞ dt exp(i ω t) exp(-γ t) G(k, t)
```

with broadening γ. The paper uses γ = 0.05 unless noted otherwise. This repository stops at C_k(t).

## Optional SLURM

`submit.sh` is a NERSC Perlmutter-style script: shared CPU QOS, a `cpu` constraint, `$SCRATCH`, and `module load conda`. Another site needs its own account, QOS, modules, and paths.

Before submitting:

1. Put this repository where the batch script can see it. The script calls `$HOME/NAMD/serial.py`.
2. Edit `submit.sh`: replace `user@domain.com`, and replace the conda environment name `pyenv` if needed. Memory is 1 GB and the wall time is 24 hours.
3. Set `Model`, `Method`, `System = slurm`, and `Cpus` in `input.txt`. Set `$NTraj` and `$NSteps` there as well. Adjust the Holstein class, or the LiF class, for Hamiltonian parameters.
4. From a login node where `$SCRATCH` is set, run `python run.py input.txt my-run`.

`run.py` copies `input.txt`, the selected model, and the selected method into `$SCRATCH/NAMD/my-run` (and the LiF `.npz` when `Model = lif`), then submits one `sbatch` job per CPU. Each job runs `serial.py` with that CPU index as the task id. Output stays under `$SCRATCH/NAMD/my-run/`. Average it with `python avg.py` pointed at that directory.

`System = htcondor` copies those same files and then returns. HTCondor submission is the upstream `condor.sub` script; this repository only includes `submit.sh`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The smoke test runs the local Holstein example, a short MASH trajectory, `run.py` on a local input, and `avg.py`. It also takes one LiF electronic step to confirm that the k-space driver starts. That step is a startup check. A production LiF grid is much longer: the model defaults are `dtN = 250`, `dtE = 1`, and a 375-state Hamiltonian.
