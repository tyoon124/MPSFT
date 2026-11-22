# Chapter 2. Phase Behavior and Thermodynamic Anomalies

This folder contains the Python scripts and data files for the worked examples of Chapter 2 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| File | Description |
|---|---|
| `Example2.1.py` | Example 2.1: Coexistence curves and p–ρ isotherms of CO2 and water (Fig. 2.1) |
| `Example2.2.py` | Example 2.2: Vapor–liquid equilibrium of CO2–ethanol with the PR EoS and the Wong–Sandler/NRTL mixing rule (Fig. 2.3) |
| `Example2.3.py` | Example 2.3: Vapor–liquid equilibrium of CO2–n-hexane with the SAFT-γ Mie and PR EoSs (Fig. 2.4) |
| `Example2.4.py` | Example 2.4: Classical and crossover PR EoSs near the critical point of CO2 (Fig. 2.5) |
| `c2oh.xlsx` | Literature vapor–liquid equilibrium data for CO2 (1) + ethanol (2), used in Example 2.2 |
| `c6h14.xlsx` | Literature vapor–liquid equilibrium data for CO2 (1) + n-hexane (2), used in Example 2.3 |

## Requirements

The scripts were tested with Python 3.10.

```bash
pip install numpy scipy matplotlib seaborn pandas openpyxl CoolProp phasepy sgtpy
```

| Example | Packages |
|---|---|
| 2.1 | NumPy, Matplotlib, seaborn, CoolProp |
| 2.2 | NumPy, SciPy, pandas, openpyxl, Matplotlib, seaborn, [phasepy](https://github.com/gustavochm/phasepy) |
| 2.3 | NumPy, pandas, openpyxl, Matplotlib, seaborn, [SGTPy](https://github.com/gustavochm/sgtpy), phasepy |
| 2.4 | NumPy, SciPy, Matplotlib, CoolProp |

## How to run

The scripts take no command-line arguments. Keep the data files in the same folder as the scripts, edit the settings in the script if needed, and run it:

```bash
python Example2.1.py
```

Each script saves its figure as a PDF file in the same folder. Examples 2.1 and 2.4 finish in under a minute. Examples 2.2 and 2.3 take a few minutes because they trace the phase envelopes point by point.

## Examples

### Example 2.1. Coexistence curves and p–ρ isotherms of CO2 and water

The vapor–liquid coexistence curves and several isotherms of CO2 (panel A) and water (panel B) are calculated with CoolProp on the pressure–density plane. The backends are the Span–Wagner EoS for CO2 and the IAPWS-95 formulation for water.

- Settings: the `FLUIDS` dictionary sets the isotherms (in °C), the axis ranges, and the label positions of each panel.
- Output: `figure2.1.pdf`
- Step 1 of the script installs CoolProp with pip. Skip it if CoolProp is already installed.

### Example 2.2. CO2–ethanol with the PR EoS and the Wong–Sandler/NRTL mixing rule

The vapor–liquid equilibrium of CO2 (1) + ethanol (2) is calculated with phasepy, using the PR EoS with the Wong–Sandler mixing rule and the NRTL model (k12 = 0.40, α12 = 0.30, Δg12 = 526.09 K, Δg21 = −61.72 K).

- Panel A: p–x–y isotherms from 313 to 373 K. Each isotherm is traced by pseudo-arclength continuation in (x1, y1, p), so the bubble and dew curves close at the mixture critical point.
- Panel B: phase envelopes at constant composition (z1 = 0.1, 0.3, 0.5, 0.7, 0.9) from 270 to 510 K, with the vapor-pressure curves of the pure components. The critical points of the envelopes move continuously from that of CO2 to that of ethanol (van Konynenburg–Scott Type I).
- Panel C: relative deviation of the bubble pressure, %Δp = 100(p_lit − p_cal)/p_lit, at the measured temperature and liquid composition.
- Panel D: parity plot of the ethanol mole fraction in the vapor, y2.
- Settings: `TinKs` (isotherms of panel A), `T_MIN` and `T_MAX` (temperature range of the data compared in panels C and D), `Z_ENV` and `T_env_grid` (envelopes of panel B).
- Input: `c2oh.xlsx`
- Output: `figure2.3.pdf`

### Example 2.3. CO2–n-hexane with the SAFT-γ Mie and PR EoSs

The vapor–liquid equilibrium of CO2 (1) + n-hexane (2) at 298, 313, 353, and 393 K is calculated with the SAFT-γ Mie EoS (SGTPy, with published group parameters for CO2, CH3, and CH2) and compared with the PR EoS without a binary interaction parameter (phasepy).

- Panel A: p–x–y isotherms. SAFT-γ Mie results are dashed lines, and PR results are solid lines.
- Panel B: densities of the coexisting phases.
- Panel C: relative deviation of the bubble pressure of both EoSs at the measured temperature and liquid composition.
- Panel D: log-log parity plot of the n-hexane mole fraction in the vapor, y2.
- Settings: `TinKs` (isotherms), `T_TOL` (tolerance for matching the data to an isotherm), `EXP_FILE` (data file). If the data file is missing, only the EoS results are drawn.
- Input: `c6h14.xlsx`
- Output: `figure2.4.pdf`

### Example 2.4. Classical and crossover PR EoSs near the critical point of CO2

The classical PR EoS is compared with the crossover PR EoS of Feyzi et al., which applies Kiselev's renormalization procedure to the PR EoS. The Span–Wagner EoS (through CoolProp) is the reference.

- Panel A: coexistence curve, with the effective exponent of ρ^l − ρ^g for 3×10⁻⁴ < |t| < 10⁻².
- Panel B: isobaric heat capacity C_p along the critical isobar of each model.
- The console prints the critical constants of each model, the deviations of the saturated properties from the reference EoS, the effective exponents, and a table of C_p.
- Output: `figure2.5.pdf`

## Data files

`c2oh.xlsx` and `c6h14.xlsx` list the measured bubble- and dew-point data compiled from the literature. Both files have these columns:

| Column | Meaning |
|---|---|
| `x1` | CO2 mole fraction in the liquid |
| `y1` | CO2 mole fraction in the vapor |
| `T(K)` | Temperature, K |
| `P(bar)` | Pressure, bar |
| `Reference` | Source of the data point |

`c6h14.xlsx` also has a `KDB ID` column. Please cite the original sources in the `Reference` column if you use the data.

## References

- I. H. Bell, J. Wronski, S. Quoilin, V. Lemort, *Ind. Eng. Chem. Res.* **53**, 2498 (2014). (CoolProp)
- G. Chaparro, A. Mejía, *J. Comput. Chem.* **41**, 2504 (2020). (phasepy)
- A. Mejía, E. A. Müller, G. Chaparro Maldonado, *J. Chem. Inf. Model.* **61**, 1244 (2021). (SGTPy)
- R. Span, W. Wagner, *J. Phys. Chem. Ref. Data* **25**, 1509 (1996).
- W. Wagner, A. Pruß, *J. Phys. Chem. Ref. Data* **31**, 387 (2002).
- D.-Y. Peng, D. B. Robinson, *Ind. Eng. Chem. Fundam.* **15**, 59 (1976).
- D. S. H. Wong, S. I. Sandler, *AIChE J.* **38**, 671 (1992).
- H. Renon, J. M. Prausnitz, *AIChE J.* **14**, 135 (1968).
- B. Kim, D. Yoo, S. Gee, T. J. Yoon, *J. Chem. Eng. Data* **71**, 1758 (2026). (PR/NRTL parameters for CO2–ethanol)
- S. Dufal et al., *J. Chem. Eng. Data* **59**, 3272 (2014). (SAFT-γ Mie)
- F. Feyzi, M. Seydi, F. Alavi, *Fluid Phase Equilib.* **293**, 251 (2010).
- S. B. Kiselev, *Fluid Phase Equilib.* **147**, 7 (1998).
