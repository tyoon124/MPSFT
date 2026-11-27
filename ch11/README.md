# Chapter 11. Nanomaterial and Thin-Film Synthesis

This folder contains the Python scripts for the worked examples of Chapter 11 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| Folder | Example | Figure |
|---|---|---|
| `Example11.1/` | Step coverage of supercritical fluid deposition (SFD) and chemical vapor deposition (CVD) in a trench | Fig. 11.4 |
| `Example11.2/` | Drying paths of an ethanol-filled gel and the two-phase region of CO2 + ethanol | Fig. 11.6 |

Neither example needs data files.

## Requirements

The scripts were tested with Python 3.10.

```bash
pip install numpy scipy matplotlib CoolProp phasepy
```

| Example | Packages |
|---|---|
| 11.1 | NumPy, SciPy, Matplotlib, CoolProp |
| 11.2 | NumPy, Matplotlib, CoolProp, phasepy |

CoolProp provides the properties of CO2 (Span–Wagner). phasepy provides the Peng–Robinson equation of state with the Wong–Sandler mixing rule and the NRTL model.

## How to run

The scripts take no command-line arguments. Edit the settings block at the top of a script if needed, and run it from its own folder:

```bash
cd Example11.1
python Example11.1.py
```

Each script saves its figure (`figure11.X.pdf`) in its own folder. Run times: about 15 s for Example 11.1 and about 10 s for Example 11.2.

## Examples

### Example 11.1. Why does zero-order deposition fill deep features?

The script compares SFD in CO2 (473 K, 15 MPa, precursor concentration 5 mol m^-3, Langmuir kinetics) with low-pressure CVD (450 K, 133 Pa, first-order kinetics) at the same growth rate at the trench opening (30 nm min^-1 of Cu).

- Transport: Chapman–Enskog diffusivities with the Enskog dense-fluid factor for CO2 and for Cu(hfac)2 in CO2; Knudsen diffusion for the CVD gas.
- Step coverage from the steady diffusion–reaction balance in a trench of width 10 nm, and film profiles in a trench of aspect ratio 40 as the trench narrows.
- The precursor concentration, the Langmuir constant, and the CVD conditions are illustrative.
- Output: `figure11.4.pdf`. The console prints the transport properties, Knudsen numbers, the aspect ratios at half coverage, and the film thicknesses.

### Example 11.2. Staying out of the two-phase region

- Panel A: p–T diagram of CO2 with the saturation line, the Widom line, an evaporation path, and the CO2 drying route. The inset gives the critical temperature of CO2 in a slit pore from the Fisher–Nakanishi relation, with the amplitude calibrated on Example 6.4.
- Panel B: p–x–y diagram of CO2 + ethanol at 313 K and 343 K with the binary parameters of Kim et al., J. Chem. Eng. Data 71 (2026) 1758. Near the mixture critical point, the bubble-point curve is traced by pseudo-arclength continuation of the equal-fugacity equations.
- Two depressurization paths from 12 MPa are compared: at constant composition z = 0.6, and after flushing with CO2.
- Output: `figure11.6.pdf`. The console prints the confined critical temperatures, the mixture critical points, and the pressure at which the constant-z path enters the two-phase region.
