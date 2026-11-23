# Chapter 6. Interfacial Phenomena, Wetting, and Confinement

This folder contains the Python scripts, LAMMPS input files, and LAMMPS output files for the worked examples of Chapter 6 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| Folder | Example | Figure |
|---|---|---|
| `Example6.1/` | Gibbs adsorption at the CO2/n-hexane interface (squared-gradient theory, PR EoS) | Fig. 6.1 |
| `Example6.2/` | Interfacial tension of the Lennard-Jones (LJ) vapor–liquid interface from direct coexistence | Fig. 6.2 |
| `Example6.3/` | Squared-gradient theory of the CO2 vapor–liquid interface (PR EoS) | Fig. 6.3 |
| `Example6.4/` | Attractive versus repulsive walls in a slit pore (LJ fluid) | Fig. 6.5 |

All LJ quantities are in reduced units (σ, ε, m, k_B = 1).

## Data in this repository

Examples 6.1 and 6.3 need no data files. For Examples 6.2 and 6.4, the LAMMPS output files that the scripts read are complete (no reduction): the density profiles along the slab axis, the pressure tensor (Example 6.2), and the density profiles across the pore (Example 6.4), together with the LAMMPS log files. All results agree with the book.

## Requirements

The scripts were tested with Python 3.10 and LAMMPS (22 Jul 2025, Update 3).

```bash
pip install numpy scipy matplotlib CoolProp
```

| Example | Packages |
|---|---|
| 6.1 | NumPy, SciPy, Matplotlib, CoolProp |
| 6.2 | NumPy, SciPy, Matplotlib |
| 6.3 | NumPy, SciPy, Matplotlib, CoolProp |
| 6.4 | NumPy, SciPy, Matplotlib |

CoolProp is used only for the reference surface tensions (Mulero et al. correlation) and saturated densities.

## How to run

The scripts take no command-line arguments. Edit the settings block at the top of a script if needed, and run it from its own folder:

```bash
cd Example6.2
python Example6.2.py
```

Each script saves its figure (`figure6.X.pdf`) in its own folder. Run times: about 25 s for Example 6.1 and a few seconds for the others. Example 6.4 also reads the bulk reference from `../Example6.2/`, so keep both folders side by side.

To repeat the simulations, run each LAMMPS input inside its own subfolder (for example, `cd Example6.2/T0.90` and `lmp -in in.slab`). Each input is self-contained and the runs are independent. They take 22–34 min (Example 6.2) and 3–8 min (Example 6.4) on four to eight cores.

## Examples

### Example 6.1. Gibbs adsorption at the CO2/n-hexane interface

Squared-gradient theory for the binary vapor–liquid interface with the Peng–Robinson EoS (k_ij = 0.11), constant influence parameters, and a geometric-mean cross parameter, which reduces the density path to a one-dimensional problem. κ(CO2) is taken from Example 6.3; κ(n-hexane) is fitted to its surface tension at 313.15 K.

- The bulk VLE is followed from 0.5 MPa in steps of 0.5 MPa. The mixture critical pressure is located by bisection, and eight more points approach it.
- The relative adsorption Γ_CO2(C6) is computed from the density profiles and from the Gibbs adsorption equation, −dγ/dμ_CO2 (finite differences along the coexistence curve).
- Settings: `TEMPERATURES`, `P_START`, `P_STEP`, `N_APPROACH`, `KIJ`, `PROFILE_CASE`.
- Output: `figure6.1.pdf`. The console prints γ and both adsorption values at every pressure.

### Example 6.2. LJ vapor–liquid interface from direct coexistence

NVT molecular dynamics (Nosé–Hoover) of N = 1250 LJ atoms in a 40.26σ × 8.75σ × 8.75σ box with a liquid slab normal to x, at T* = 0.90, 1.00, 1.10, 1.15, and 1.20. The potential is truncated and shifted at 4σ without tail corrections. Each run has 10⁶ steps of equilibration and 2 × 10⁶ steps of production (Δt* = 0.005).

- Inputs: `T*/in.slab`. Data: `T*/profile.dat` (density profile along x, bins of 0.5σ, 200 blocks), `T*/pressure.dat` (pressure tensor every 100 steps).
- The profile of each block is re-centered on the slab and fitted with the tanh profile (two interfaces and periodic images). γ follows from the pressure tensor, with the standard error from ten blocks. T*_crit and ρ*_crit follow from the coexistence curve (β = 0.326, rectilinear diameter).
- Output: `figure6.2.pdf`. The console prints the coexisting densities, γ, the 10–90 % width, the critical point (T*_crit = 1.243, ρ*_crit = 0.320), and the fits of γ = γ₀t^1.26 and γ₀t^1.5.

### Example 6.3. Squared-gradient theory of the CO2 vapor–liquid interface

Peng–Robinson EoS for CO2 (T_crit = 304.13 K, p_crit = 7.377 MPa, ω = 0.2239) with one temperature-independent influence parameter fitted at T/T_crit = 0.90 (κ = 1.67 × 10⁻²⁰ J m⁵ mol⁻²). The coexistence densities come from the Maxwell construction, and γ and the density profile come from quadrature.

- Settings: `TR_PROFILES`, `TR_FIT`, `TR_SCAN`, `T_WINDOW` (window of the exponent fits).
- Output: `figure6.3.pdf`. The console prints γ, the reference γ, the 10–90 % width, and the exponents in 10⁻³ ≤ t ≤ 10⁻² (1.51 for the theory, 1.25 for the reference).

### Example 6.4. Attractive versus repulsive walls in a slit pore

The slab of Example 6.2 confined between two 9-3 walls at z = 0 and z = H = 7σ (N = 600, same cross-section, potential, and run schedule). There are three walls: repulsive (ε_w = 1, cut at the minimum, 0.8579σ), attractive (ε_w = 1, cutoff 4σ), and strongly attractive (ε_w = 3, cutoff 4σ). Each wall is run at T* = 0.80, 0.90, 0.95, 1.00, 1.05, and 1.10.

- Inputs: `<wall>/T*/in.pore`. Data: `<wall>/T*/profile_x.dat` (along the pore axis), `profile_z_liquid.dat` and `profile_z_vapor.dat` (across the pore, bins of 0.1σ).
- Densities in the pore are given per accessible volume (factor H/(H − σ)). The bulk reference is `../Example6.2/`.
- Output: `figure6.5.pdf`. The console prints the apparent critical temperatures and densities and their shifts relative to the bulk.

## References

- J. W. Cahn, J. E. Hilliard, *J. Chem. Phys.* **28**, 258 (1958). (squared-gradient theory)
- D.-Y. Peng, D. B. Robinson, *Ind. Eng. Chem. Fundam.* **15**, 59 (1976). (PR EoS)
- A. Mulero, I. Cachadiña, M. I. Parra, *J. Phys. Chem. Ref. Data* **41**, 043105 (2012). (surface-tension correlations)
- I. H. Bell, J. Wronski, S. Quoilin, V. Lemort, *Ind. Eng. Chem. Res.* **53**, 2498 (2014). (CoolProp)
- J. G. Kirkwood, F. P. Buff, *J. Chem. Phys.* **17**, 338 (1949). (pressure-tensor route to γ)
- M. E. Fisher, H. Nakanishi, *J. Chem. Phys.* **75**, 5857 (1981). (critical-point shift in slit pores)
- A. P. Thompson et al., *Comput. Phys. Commun.* **271**, 108171 (2022). (LAMMPS)
