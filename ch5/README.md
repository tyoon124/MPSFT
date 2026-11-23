# Chapter 5. Molecular Simulation Analyses

This folder contains the Python scripts, LAMMPS input files, and reduced data files for the worked examples of Chapter 5 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| Folder | Example | Figure |
|---|---|---|
| `Example5.1/` | Radial distribution function, coordination number, and potential of mean force of the Lennard-Jones (LJ) fluid | Fig. 5.1 |
| `Example5.2/` | Local (Voronoi) density distributions of the LJ fluid | Fig. 5.2A, B |
| `Example5.3/` | Gas-like/liquid-like classification and percolation of the LJ fluid | Fig. 5.3 |
| `Example5.4/` | Hydrogen-bond network and tetrahedral order of TIP4P/2005 water | Fig. 5.4 |
| `Example5.5/` | Response functions near the critical point: fluctuation versus finite difference | Fig. 5.5 |
| `Example5.6/` | Excess chemical potential and excess entropy: thermodynamic route versus Widom insertion | Fig. 5.6 |
| `Example5.7/` | Self-diffusion coefficient and shear viscosity: Green–Kubo versus Einstein | Fig. 5.7 |

All LJ quantities are in reduced units (σ, ε, m, k_B = 1).

## Data in this repository

The full trajectories are too large for GitHub (several GB). The folders contain only the files that the scripts need, and each script recomputes a missing file from the trajectory if you have run the LAMMPS inputs yourself.

| Example | Provided instead of the full trajectory |
|---|---|
| 5.1 | `rho*/rdf_rho*.dat`: g(r) accumulated on the fly by LAMMPS |
| 5.2 | `rho0.10/volume.npy`, `rho0.70/volume.npy`: Voronoi cell volumes of every particle in every frame (float32) |
| 5.3 | `rho*/PiGas.npy`: gas-like (1) / liquid-like (0) label of every particle in every frame; `rho0.20`–`rho0.35/traj.npz`: every 10th frame (201 frames, float32), which is what the percolation analysis uses |
| 5.4 | `*/traj.npz`: every 10th frame (251 of 2501 frames, float32); `hbstat.npy`, `tetrahedral.npy`: results for these frames |
| 5.5 | `*/run_series.dat`: volume and enthalpy time series (complete) |
| 5.6 | `rho_*/traj.lammpstrj`, `rho_*/avg.txt` (complete); `Example5.6_results.txt`: Widom insertion results |
| 5.7 | `*/corr.npz`: mean-square displacement, velocity autocorrelation function, and stress autocorrelation function of each run |

All results agree with the book, except Example 5.4: there the averages come from 251 instead of 2501 frames and differ from the values quoted in the book by less than 0.01.

## Requirements

The scripts were tested with Python 3.10 and LAMMPS (22 Jul 2025, Update 3).

```bash
pip install numpy scipy matplotlib pandas seaborn freud-analysis pillow teqp
```

| Example | Packages |
|---|---|
| 5.1 | NumPy, SciPy, Matplotlib |
| 5.2 | NumPy, SciPy, Matplotlib, pandas, seaborn, freud, Pillow |
| 5.3 | NumPy, SciPy, Matplotlib, freud, Pillow |
| 5.4 | NumPy, SciPy, Matplotlib |
| 5.5 | NumPy, SciPy, Matplotlib, teqp |
| 5.6 | NumPy, SciPy, Matplotlib, teqp |
| 5.7 | NumPy, Matplotlib |

## How to run

The scripts take no command-line arguments. Edit the settings block at the top of a script if needed, and run it from its own folder:

```bash
cd Example5.1
python Example5.1.py
```

With the files provided here, every script finishes in less than a minute. Each script saves its figure as `figure5.X.pdf` in this chapter folder, except Example 5.2, which saves `Example5.2.pdf` in its own folder.

To repeat the simulations, run each LAMMPS input inside its own subfolder (for example, `cd Example5.1/rho0.30` and `lmp -in in.lj-T1.40-d0.30`). The runs are independent and can be run in parallel. Then delete the provided data file that you want to recompute (for example, `corr.npz` in Example 5.7) and run the Python script.

## Examples

### Example 5.1. Radial distribution function of the LJ fluid

NVT molecular dynamics of N = 1000 LJ atoms on the isotherm T* = 1.40 at 13 densities, 0.10 ≤ ρ* ≤ 0.70. The potential is truncated at 3σ, with tail corrections added to the energy and pressure. After 2.5 × 10⁶ equilibration steps, a 2 × 10⁷-step production run (Δt* = 0.002) accumulates g(r) on the fly every 10³ steps in bins of 0.01σ.

- Inputs: `rho*/in.lj-T1.40-d*`. The same runs also write the dump files (`traj_rho*.lammpstrj`, 2001 frames) used in Examples 5.2 and 5.3.
- The script reads `rdf_rho*.dat`, smooths g(r) with a Savitzky–Golay filter to locate its first minimum, and integrates the running coordination number. It also computes the potential of mean force W(r) = −k_BT ln g(r).
- Settings: `RHOS_GR` (densities of panels A and B), `RHOS_PMF` (densities of panel C), `R_MAX`.
- Output: `figure5.1.pdf`. The console prints r_min and N_c for each density.

### Example 5.2. Local density distributions of the LJ fluid

The local density of each atom is the inverse of its Voronoi cell volume. For ρ* = 0.1 the distribution is fitted with an inverse-gamma distribution, and for ρ* = 0.7 with a Gaussian, both by maximum likelihood (`scipy.stats.invgamma.fit` with the location fixed at zero, `scipy.stats.norm.fit`).

- Data: `rho0.10/volume.npy`, `rho0.70/volume.npy`. If missing, the script computes them with `freud.locality.Voronoi` from `traj.npz` or from the dump file of Example 5.1 placed in the same subfolder.
- Output: `Example5.2.pdf` (panels A and B of Fig. 5.2). The console prints the fitted parameters (shape α ≃ 5.26 and scale β ≃ 0.526 for ρ* = 0.1; mean ≃ 0.712 and standard deviation ≃ 0.092 for ρ* = 0.7).

### Example 5.3. Percolation of gas-like and liquid-like particles

Each atom is labeled gas-like or liquid-like by comparing the mean Voronoi local density of its first and second neighbor shells with ρ*_crit = 0.316. The gas-like fraction π^g is fitted with an inverse sigmoid, and the tangent at π^g = 1/2 gives ρ_pb and ρ_pa. The percolation of each class is then tested at ρ* = 0.20–0.35 with Stoll's cluster-counting algorithm, which is implemented in the script (Section 3).

- Data: `rho*/PiGas.npy` (labels for all 2001 frames) and `rho0.20`–`rho0.35/traj.npz` (every 10th frame). The labels are recomputed only for a density whose `PiGas.npy` is missing, which needs the full trajectory.
- Settings: `Dcrit` (classification threshold), `PERC_STRIDE` (frame interval of the percolation analysis), `PERC_CRITERION` (`'any'`: spanning along at least one axis).
- Output: `figure5.3.pdf`. The console prints the percolation probabilities P_inf of both classes. The snapshot images `lj-*.png` are drawn into the figure.
- Run time: about 30 s.

### Example 5.4. Hydrogen-bond network and tetrahedral order of water

NpT molecular dynamics of 1024 TIP4P/2005 molecules at 15 MPa and 298, 423, 453, 473, 503, and 523 K (PPPM electrostatics, 5 ns production). A hydrogen bond is counted when R(O–O) ≤ 3.3 Å, r(H···O) ≤ 2.5 Å, and the O–H···O angle is at least 95.7°. The tetrahedral order parameter q is computed from the four nearest oxygen atoms.

- Inputs: `*/in.water` with `*/data.initial`.
- Data: `*/traj.npz` (every 10th frame), `*/hbstat.npy` and `*/tetrahedral.npy` (results for these frames). The two result files are recomputed only if missing, which takes about 1.2 s per frame.
- Output: `figure5.4.pdf`.

### Example 5.5. Response functions near the critical point

NpT molecular dynamics of the LJ fluid along the isobar p* = 0.15 (≃ 1.17 p*_crit), with N = 500 at nine temperatures (T* = 1.28–1.50) and N = 2000 at four temperatures (T* = 1.36–1.42). The potential is truncated at 5σ with tail corrections. C_p, κ_T, and α_p are obtained from the fluctuations of V and H = E + pV (with delete-one-block jackknife errors) and from central finite differences with ΔT* = 0.02. The reference curves come from the EoS of Thol et al. (2016), evaluated with teqp inside the script.

- Inputs: `N*_T*_p0.15/in.run`. Data: `N*_T*_p0.15/run_series.dat`.
- Settings: `DATASETS` (system sizes, jackknife blocks, finite-difference temperatures), `DT_FD`, `REF_T_MIN`, `REF_T_MAX`.
- Output: `figure5.5.pdf` and `Example5.5_summary.csv`. The console prints a table with sampling diagnostics (volume correlation time, run length).

### Example 5.6. Excess chemical potential and excess entropy

NVT molecular dynamics of N = 500 LJ atoms on the isotherm T* = 1.40 at 11 densities (0.05 ≤ ρ* ≤ 0.80). The potential is truncated at 4σ with tail corrections. The thermodynamic route integrates (Z − 1)/ρ from the second virial coefficient at ρ = 0 with a weighted smoothing spline, and gives μ_ex and s⁺ = −s_ex/k_B. Widom insertion uses 5000 test particles per saved configuration.

- Inputs: `rho_*/in.lj`. Data: `rho_*/avg.txt` (Z and U/N), `rho_*/traj.lammpstrj` (configurations for Widom insertion).
- `Example5.6_results.txt` holds the Widom results. Delete it to rerun the insertions (several minutes).
- Settings: `SMOOTH` (spline smoothing), `N_INSERT`, `RC`, `RHO_WIDOM_MAX`.
- Output: `figure5.6.pdf` and `Example5.6_thermo_route.txt`.

### Example 5.7. Self-diffusion coefficient and shear viscosity

Equilibrium MD of the LJ fluid at ρ* = 0.316 with a 2.5σ cutoff and no tail corrections. This is the truncated-and-shifted LJ fluid, whose critical temperature is T*_crit ≃ 1.08. Each run is equilibrated in NVT and sampled in NVE (2 × 10⁵ steps, Δt* = 0.005). There are three replicas per state point: N = 256, 500, 1000, and 2000 at T* = 1.40, and N = 1000 at six temperatures between T* = 1.35 and 2.50.

- D is computed from the mean-square displacement (Einstein) and from the velocity autocorrelation function (Green–Kubo).
- η is computed from the Green–Kubo integral of five stress components (three off-diagonal elements and two normal-stress differences).
- Panel A plots D_L against 1/L with a linear fit; its slope is compared with the Yeh–Hummer slope.
- Inputs: `N*_T*_r*/in.lj_transport`. Data: `N*_T*_r*/corr.npz`. A run whose `corr.npz` is missing is analyzed from `run.P.dat` and `run.trj` (about 4 s per run), and its `corr.npz` is then written.
- Settings: `FIT_MSD` (time window of the MSD fit), `T_VACF` (upper limit of the VACF integral), `PLATEAU` (averaging window of the Green–Kubo running integral), `XI` (Yeh–Hummer constant).
- Output: `figure5.7.pdf`, `A_finite_size.dat` (panel A), and `BC_vs_T.dat` (panels B and C).

## References

- J. J. Potoff, A. Z. Panagiotopoulos, *J. Chem. Phys.* **109**, 10914 (1998). (critical point of the LJ fluid)
- J. Vrabec, G. K. Kedia, G. Fuchs, H. Hasse, *Mol. Phys.* **104**, 1509 (2006). (truncated-and-shifted LJ fluid)
- M. Thol et al., *J. Phys. Chem. Ref. Data* **45**, 023101 (2016). (LJ EoS)
- I. H. Bell, U. K. Deiters, A. M. M. Leal, *Ind. Eng. Chem. Res.* **61**, 6010 (2022). (teqp)
- A. P. Thompson et al., *Comput. Phys. Commun.* **271**, 108171 (2022). (LAMMPS)
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020). (freud)
- E. Stoll, *Comput. Phys. Commun.* **109**, 1 (1998). (cluster counting algorithm)
- T. J. Yoon, M. Y. Ha, W. B. Lee, Y.-W. Lee, *J. Chem. Phys.* **149**, 014502 (2018). (Widom delta)
- J. L. F. Abascal, C. Vega, *J. Chem. Phys.* **123**, 234505 (2005). (TIP4P/2005)
- I.-C. Yeh, G. Hummer, *J. Phys. Chem. B* **108**, 15873 (2004). (finite-size correction of D)
