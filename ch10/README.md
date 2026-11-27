# Chapter 10. Particle Formation from Supercritical Solutions

This folder contains the Python scripts for the worked examples and the calculated figures of Chapter 10 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| Folder | Content | Figure |
|---|---|---|
| `Figure10.2/` | Solubility of a model solute in a CO2-expanded liquid (ternary regular-solution model) | Fig. 10.2 |
| `Example10.1/` | Isentropic expansion of CO2 in a RESS nozzle | Fig. 10.3 |
| `Example10.2/` | Classical nucleation theory for a RESS solute | Fig. 10.4A,B |
| `Example10.3/` | Population balance (moment method) of a RESS aerosol, with and without coagulation | Fig. 10.4 |
| `Example10.4/` | Jet break-up and dynamic surface tension vanishing in SAS: the model of Marra et al. | Fig. 10.5 |
| `Example10.5/` | Feed-solvent composition (acetone/DMSO) and the cross-over | Fig. 10.6 |

## Data in this repository

- **Examples 10.4 and 10.5.** `Example10.4_data.npz` and `Example10.5_data.npz` hold all arrays drawn in Figs. 10.5 and 10.6. With `RECOMPUTE = False` in the settings block, the scripts redraw the figure from these files without repeating the calculation. With `RECOMPUTE = True` (the default) they repeat the calculation and rewrite the files.
- **Example 10.4.** The calibration data (break-up and vanishing times at 313 K, liquid properties at 293 K, and the two experiments at 150 bar) are those of Marra, De Marco, and Reverchon, Chem. Eng. Sci. 71, 39 (2012), and are in the settings block. The PR binary parameters of CO2–DMSO are those of Rajasingam et al., J. Supercrit. Fluids 31, 227 (2004).

No other data files are needed.

## Requirements

The scripts were tested with Python 3.10.

```bash
pip install numpy scipy matplotlib CoolProp phasepy
```

| Script | Packages |
|---|---|
| Figure10.2 | NumPy, Matplotlib, CoolProp |
| Example 10.1 | NumPy, Matplotlib, CoolProp |
| Example 10.2 | NumPy, Matplotlib |
| Example 10.3 | NumPy, SciPy, Matplotlib |
| Example 10.4 | NumPy, SciPy, Matplotlib, phasepy, CoolProp |
| Example 10.5 | NumPy, SciPy, Matplotlib, phasepy, CoolProp |

CoolProp provides the properties of CO2 (Span–Wagner; viscosity only in Examples 10.4 and 10.5). The results were checked with CoolProp 8.1 and phasepy 0.0.56.

## How to run

The scripts take no command-line arguments. Edit the settings block at the top of a script if needed, and run it from its own folder:

```bash
cd Example10.4
python Example10.4.py
```

Each script saves its figure (`figure10.X.pdf`) in its own folder; Example 10.2 saves panels A and B alone (`figure10.4AB.pdf`). Run times: about 10 s for Example 10.1 and Example 10.4, 20 s for Example 10.5, and a few seconds for the others.

## Examples

### Figure 10.2. Solubility in a CO2-expanded liquid

The script evaluates the ternary regular-solution model (Eq. 10.1) for a solute of molar volume 250 cm^3 mol^-1 at 313 K. The solubility parameters of the solvents are Hansen total parameters; that of CO2 follows from Giddings' density correlation with Span–Wagner densities.

- Output: `figure10.2.pdf`. The console prints the relative solubility at CO2 volume fractions of 0.5 and 0.8.

### Example 10.1. Isentropic RESS nozzle expansion

The script follows pure CO2 from 35 MPa along isentropes at 313, 333, and 353 K to 0.6 MPa (equilibrium homogeneous flow, no friction or heat exchange). The velocity follows from u = [2(h0 − h)]^1/2, and the throat is the maximum of the mass flux G = ρu.

- Output: `figure10.3.pdf`. The console prints the throat state, the entry into the two-phase region, and the vapor fraction at the end.

### Example 10.2. Classical nucleation theory

Barrier, critical nucleus, and nucleation rate (Eqs. 10.12 and 10.13) for a solute with the molecular volume of naphthalene at 300 K; the interfacial tensions and the pre-exponential factor are illustrative.

- Output: `figure10.4AB.pdf`. The console prints r*, n*, ΔG*/kT, and J at S = 50, 100, and 1000, and the critical supersaturations.

### Example 10.3. Population balance of a RESS aerosol

The script advances the first three moments of the particle-volume distribution (Eqs. 10.9–10.11) in a fluid parcel quenched to S0 = 2000, with nucleation, free-molecular growth (sticking coefficient 0.1), and free-molecular Brownian coagulation, with and without coagulation. It also draws panels A and B with the parameters of Example 10.2.

- Output: `figure10.4.pdf`. The console prints the number density, geometric mean diameter, geometric standard deviation, and supersaturation at 0.1, 0.3, 1, and 10 ms.

### Example 10.4. Jet break-up and dynamic surface tension vanishing in SAS

The script codes the calculation of Marra et al. for yttrium acetate in DMSO (100 μm nozzle, 1 mL min^-1): Peng–Robinson EoS (phasepy) with two binary parameters, the Sterling–Sleicher dispersion relation for the break-up time, and the square-gradient interfacial tension on the self-similar interdiffusion profile for the vanishing time. Four inputs that the paper does not report are calibrated on its results at 313 K; the model is then applied at 333 K.

- Settings: `RECOMPUTE`, `T_K`, `T_CAL`, `P_GRID`, `C_MAP`, `SIGMA_ZERO`.
- Output: `figure10.5.pdf` and `Example10.4_data.npz`. The console prints the calibration, the check against the paper at 313 K, the two experiments, and the cross-over pressures at 333 K.

### Example 10.5. Acetone/DMSO feeds

The model and calibration of Example 10.4, applied at 333 K to feeds in which part of the DMSO is replaced by acetone (no solute). The ternary is followed along the dilution line of the feed, and below the mixture critical point of the feed the interface does not vanish, so the cross-over is max(p_x, p_MCP).

- Output: `figure10.6.pdf` and `Example10.5_data.npz`. The console prints the feed properties, the MCP and the cross-over for each feed, and the two times at 140, 180, and 220 bar.
