# Chapter 7. Reactions in Supercritical Fluids

This folder contains the Python scripts, LAMMPS input files, and LAMMPS output files for the worked examples of Chapter 7 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| Folder | Example | Figure |
|---|---|---|
| `Example7.1/` | ReaxFF molecular dynamics of the OH-initiated oxidation of methanol at four densities | Fig. 7.1 |
| `Example7.2/` | Activation volume and apparent activation energy from the compressibility of the solvent (CO2, water) | Figs. 7.2 and 7.3 |
| `Example7.3/` | Solvent clustering around an attractive solute in a near-critical Lennard-Jones (LJ) fluid | Fig. 7.4 |
| `Example7.4/` | Piston effect in a closed cell of near-critical CO2 | Fig. 7.5 |

LJ quantities (Example 7.3) are in reduced units (σ, ε, m, k_B = 1).

## Data in this repository

Examples 7.2 and 7.4 need no data files.

- **Example 7.1.** The trajectories (`traj.lammpstrj`, 16.5 MB per density) are not included. Instead, each density folder holds `counts.dat`, the numbers of intact CH3OH molecules, OH radicals, H2O molecules, and free H atoms in every frame (0.05 ps apart). The script computed these counts from the full trajectories. Its first line gives the box volume and the fraction of nearly empty cells in the last frame. When `traj.lammpstrj` is present (for example, after you repeat a run), the script counts the species itself and rewrites `counts.dat`. With `counts.dat` alone, the script gives the same results as with the trajectories.
- **Example 7.3.** The LAMMPS output files that the script reads are complete (no reduction): the solvent–solute radial distribution function in 30 blocks (`rdf.dat`) and the pressure (`pressure.dat`), together with the LAMMPS log files.

## Requirements

The scripts were tested with Python 3.10 and LAMMPS (22 Jul 2025, Update 3; REAXFF package for Example 7.1).

```bash
pip install numpy scipy matplotlib CoolProp
```

| Example | Packages |
|---|---|
| 7.1 | NumPy, SciPy, Matplotlib |
| 7.2 | NumPy, SciPy, Matplotlib, CoolProp |
| 7.3 | NumPy, SciPy, Matplotlib |
| 7.4 | NumPy, SciPy, Matplotlib, CoolProp |

CoolProp provides the properties of CO2 (Span–Wagner) and water (IAPWS-95). The near-critical results of Example 7.4 depend slightly on the CoolProp version: with CoolProp 8.1, the 90 % time at T_crit + 0.5 K is 22.5 s.

## How to run

The scripts take no command-line arguments. Edit the settings block at the top of a script if needed, and run it from its own folder:

```bash
cd Example7.3
python Example7.3.py
```

Each script saves its figure (`figure7.X.pdf`) in its own folder. Run times: about 20 s for Example 7.1, 10 s for Example 7.2, a few seconds for Example 7.3, and about 30 s for Example 7.4.

To repeat the simulations, run each LAMMPS input inside its own subfolder (for example, `cd Example7.1/rho0.05` and `lmp -in in.reax`). Each input is self-contained and the runs are independent. On eight cores, they take 4–12 min (Example 7.1) and 27–55 min (Example 7.3).

## Examples

### Example 7.1. ReaxFF oxidation of methanol at several densities

32 CH3OH, 64 OH, and 200 H2O (920 atoms) at 2500 K and 0.05, 0.10, 0.20, and 0.30 g cm^-3, with the ReaxFF parametrization of Chenoweth et al. (`ffield.reax.cho`, distributed with LAMMPS). The run lasts 30 ps (0.25 fs steps).

- The species are identified from interatomic distances (C–H < 1.35 Å, C–O < 1.75 Å, O–H < 1.15 Å).
- The script fits a first-order decay to the number of intact methanol molecules from 1 ps on. The error comes from a moving-block bootstrap (5 ps blocks).
- The script also fits the two halves of the window separately and a decay with a plateau, and fits a bimolecular constant with the number of OH radicals of each frame.
- Output: `figure7.1.pdf`. The console prints k, the alternative fits, pressures, and local density exponents.

### Example 7.2. Activation volume and apparent activation energy

The script evaluates the compressibility link ΔV‡ = −κ_T RT (1 + ρΔc) for scCO2 at 313.15 K and fits the Diels–Alder rate constants of Paulaitis and Alexander (Pure Appl. Chem. 1987; digitized data embedded in the script). It also computes E_app = E_v − m RT²α_p along isobars of CO2 and water (E_v = 40 kJ mol^-1, m = 2).

- The water minima are very narrow, so the water isobars use 20 001 temperature points (`N_T_H2O`).
- Settings: `T1`, `DC_VALUES`, `FIT_MODE`, `E_V`, `M_RHO`, `CO2_ISOBARS`, `H2O_ISOBARS`.
- Output: `figure7.2.pdf` and `figure7.3.pdf`.

### Example 7.3. Solvent clustering around an attractive solute

3997 LJ solvent atoms and 3 solute atoms (σ2 = 1.2σ, ε12 = 1.5ε) at ρ* = 0.32 and T* = 1.30–2.00, with a cutoff of 4σ.

- The script computes the first-shell coordination number, the local density enhancement, and the excess number N_exc(R) from the block-averaged g12. It also applies the finite-size correction of Ganguly and van der Vegt.
- Panels A and B of the figure show a smooth fit of g12 with the form of Matteoli and Mansoori. All printed numbers come from the simulated g12.
- Output: `figure7.4.pdf`.

### Example 7.4. Piston effect

The script solves the one-dimensional energy equation of a closed CO2 layer (half-width 5 mm) with a uniform, time-dependent pressure. The layer starts at T_crit + 0.5 K, T_crit + 5 K (critical density), or 290 K and 10 MPa, and the wall temperature steps up by 0.01 K. Each case is solved with and without the piston-effect term.

- Output: `figure7.5.pdf`. The console prints t_D, Γ, the scaling estimate of t_PE, the 90 % times, and the pressure rise.
