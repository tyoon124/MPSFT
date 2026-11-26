# Chapter 9. Supercritical Fluid Extraction and Fractionation

This folder contains the Python scripts for the worked examples and the calculated figures of Chapter 9 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| Folder | Content | Figure |
|---|---|---|
| `Figure9.1/` | Schematic pressure–temperature projections of the six van Konynenburg–Scott types | Fig. 9.1 |
| `Example9.1/` | Retrograde solubility of naphthalene in CO2 from the Peng–Robinson EoS | Fig. 9.2 |
| `Example9.2/` | Chrastil correlation of two solid solubilities and the separation-factor map | Fig. 9.3 |
| `Example9.3/` | Hansen solubility parameters of CO2 as functions of pressure and temperature | Fig. 9.4 |
| `Figure9.5/` | Periods of an extraction curve and the shrinking-core particle (schematic) | Fig. 9.5 |
| `Example9.4/` | Extraction curve of a packed bed with the broken-and-intact-cells (BIC) model | Fig. 9.6 |
| `Figure9.7/` | State table and duties of the CO2 loop of a two-separator extraction plant | Fig. 9.7 |

## Data in this repository

No separate data files are needed. The measured solubilities are in the settings blocks of the scripts:

- **Example 9.1 and 9.2 (naphthalene).** Sako, Yamane, Negishi, and Sato, Sekiyu Gakkaishi 37, 321 (1994): 38 points at 308, 318, 328, and 338 K.
- **Example 9.2 (glycyrrhizin).** Jia, Zabihi, Gao, and Zhao, J. Chem. Eng. Data 60, 1744 (2015), from the NIST ThermoML archive: 15 points at 308, 318, and 328 K.

## Requirements

The scripts were tested with Python 3.10.

```bash
pip install numpy scipy matplotlib CoolProp phasepy cairosvg pillow
```

| Script | Packages |
|---|---|
| Figure9.1 | NumPy, Matplotlib |
| Example 9.1 | NumPy, SciPy, Matplotlib, phasepy, CoolProp |
| Example 9.2 | NumPy, Matplotlib, CoolProp |
| Example 9.3 | NumPy, Matplotlib, CoolProp |
| Figure9.5 | NumPy, Matplotlib, cairosvg, Pillow |
| Example 9.4 | NumPy, SciPy, Matplotlib, CoolProp (Numba optional, for speed) |
| Figure9.7 | CoolProp |

CoolProp provides the properties of CO2 (Span–Wagner). The results were checked with CoolProp 8.1 and phasepy 0.0.56.

## How to run

The scripts take no command-line arguments. Edit the settings block at the top of a script if needed, and run it from its own folder:

```bash
cd Example9.4
python Example9.4.py
```

Each script saves its figure (`figure9.X.pdf`) in its own folder; `Figure9.7.py` only prints the state table and the duties (the diagram is drawn in `PFD.svg`). Run times: about 30 s for Example 9.1, 75 s for Example 9.4, and a few seconds for the others.

## Examples

### Example 9.1. Retrograde solubility of naphthalene

The script solves the solid–fluid equilibrium y2 = (p_sub/p) exp[v_s(p − p_sub)/RT]/φ2 with φ2 from the Peng–Robinson EoS of phasepy. The cross covolume is made quadratic with a second binary parameter l12.

- Settings: `K12` (a number, a (k0, k1) pair for a linear temperature dependence, or `None` for the unconstrained regression of k12 and l12), `L12`, `EXCLUDE`.
- Output: `figure9.2.pdf`. The console prints the fit, the AARD per isotherm, the crossover pressures, and the enhancement factor at 5–20 MPa.

### Example 9.2. Chrastil correlation and separation factor

The script fits ln c = k ln ρ1 + a/T + b to both data sets and maps α_AB = y_A/y_B on 308–328 K and 10–20 MPa.

- Output: `figure9.3.pdf`. The console prints k, a, b, the AARD, ΔH = −aR, and α_AB on the grid.

### Example 9.3. Hansen solubility parameters of CO2

The script scales the reference components (15.6, 5.2, 5.8 MPa^1/2 at 298.15 K and 39.13 cm^3 mol^-1) with the molar volume (Williams, Rubin, and Edwards) and compares the total with δ² = −u_res/V from the EoS.

- Output: `figure9.4.pdf`.

### Example 9.4. BIC model of a packed bed

The script integrates the broken-and-intact-cells balances (80 cells, RK4, time step 3 s) for a 0.30 m bed of milled solid in CO2 at 313 K and 30 MPa. The fluid-side coefficient comes from entropy scaling of the viscosity and self-diffusion of CO2 and the Wakao–Kaguei correlation; r and K_s are illustrative.

- Settings: bed geometry, `G`, `X0`, `R_REF`, `KS_REF`, `Y_STAR`, the particle-size distribution (median, geometric standard deviation, number of classes), `N_CELLS`, `DT`.
- Output: `figure9.6.pdf`. The console prints the transport chain (Re, Sc, Sh, k_f, NTU), the end of the constant-rate period, the shrinking-core fit, and the effect of the particle size.

### Figure 9.7. CO2 loop of an extraction plant

Pure CO2 (CoolProp); isenthalpic valves, no pressure drop, pump efficiency 0.70. The script prints the eight states and the pump work and heat duties in kJ per kg of circulating CO2.
