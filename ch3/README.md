# Chapter 3. Transport and Dynamic Anomalies

This folder contains the Python scripts for the worked examples of Chapter 3 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| File | Description |
|---|---|
| `Example3.1.py` | Example 3.1: Shear viscosity and thermal conductivity of CO2 far from the critical point (Fig. 3.1) |
| `Example3.2.py` | Example 3.2: Critical enhancement of the viscosity and thermal conductivity of CO2 (Figs. 3.2 and 3.3) |
| `Example3.3.py` | Example 3.3: Rosenfeld residual-entropy scaling of the viscosity of CO2, CH4, and H2O (Fig. 3.4) |

## Requirements

The scripts were tested with Python 3.10.

```bash
pip install numpy scipy matplotlib seaborn CoolProp
```

| Example | Packages |
|---|---|
| 3.1 | NumPy, Matplotlib, seaborn, CoolProp |
| 3.2 | NumPy, SciPy, Matplotlib, seaborn, CoolProp |
| 3.3 | NumPy, SciPy, Matplotlib, CoolProp |

## How to run

The scripts take no command-line arguments. Edit the settings in the script if needed and run it:

```bash
python Example3.1.py
```

Each script saves its figure(s) as PDF files in the same folder. All three examples finish in under a minute.

## Examples

### Example 3.1. Viscosity and thermal conductivity of CO2 far from the critical point

The shear viscosity η and the thermal conductivity λ of CO2 are calculated with CoolProp from 220 to 800 K along ten isobars between 4 and 100 MPa. CoolProp uses the Span–Wagner EoS, the Laesecke–Muzny viscosity correlation, and the Huber et al. thermal conductivity correlation. All three correlations split the property into a zero-density term and a density-dependent excess term.

- Panel A: η. Panel B: λ. The dashed vertical line marks the critical temperature.
- Settings: `TinKs` (temperature grid), `PinMPas` (isobars, MPa).
- Output: `figure3.1.pdf`

### Example 3.2. Critical enhancement of the transport properties of CO2

The viscosity and thermal conductivity of CO2 are written as the sum of a dilute-gas term, an excess term, and a critical enhancement, following the reference correlation of Vesovic et al. (1990). The critical enhancement is computed with the mode-coupling crossover model of Olchowy and Sengers. The thermodynamic properties come from the Span–Wagner EoS in CoolProp. The script registers a copy of the CO2 model without its three non-analytic terms (`CO2_noNA`), because those terms produce spurious double peaks in the enhancement within a few kelvin of the critical temperature.

- The console prints a check table that compares the calculated λ and η with reference values at seven state points, with the deviations in percent.
- Fig. 3.2 (viscosity) and Fig. 3.3 (thermal conductivity) have the same layout:
  - Panel A: the property versus density along six isotherms (304.65–348.15 K), with the background (dilute-gas + excess) at the lowest and highest temperatures.
  - Panel B: the critical enhancement along the same isotherms.
  - Panel C: the three-term decomposition along the 7.5 MPa isobar.
- Settings: `TinKs` (isotherms), `rhos_plot` (density grid, kg m⁻³), `PinPa` (isobar of panel C, Pa), `TinKs_iso` (temperature grid of panel C).
- Output: `figure3.2.pdf`, `figure3.3.pdf`

### Example 3.3. Residual-entropy scaling of the viscosity of CO2, CH4, and H2O

The Rosenfeld reduced viscosity, η̃ = η / [ρ^(2/3) (m k_B T)^(1/2)] with ρ the number density, is plotted against the reduced residual entropy −s^res/k_B for CO2, CH4, and H2O. Both quantities are calculated with CoolProp along isotherms T_r = T/T_crit = 1.1–2.0, for ρ/ρ_crit = 0.1–2.0. The master curve

η̃ = a (−s^res/k_B)^(−2/3) + b (−s^res/k_B)^c exp(d s^res/k_B)

is fitted to the data of each fluid with `scipy.optimize.curve_fit`.

- Panels A–C: CO2, CH4, and H2O. The isotherms are colored by T_r (color bar), and the fitted master curve is the dashed line.
- Settings: `fluids` (CoolProp fluid names), `TrRange` (reduced temperatures), `DrRange` (reduced densities).
- Output: `figure3.4.pdf`

## References

- I. H. Bell, J. Wronski, S. Quoilin, V. Lemort, *Ind. Eng. Chem. Res.* **53**, 2498 (2014). (CoolProp)
- R. Span, W. Wagner, *J. Phys. Chem. Ref. Data* **25**, 1509 (1996). (CO2 EoS)
- A. Laesecke, C. D. Muzny, *J. Phys. Chem. Ref. Data* **46**, 013107 (2017). (CO2 viscosity)
- M. L. Huber, E. A. Sykioti, M. J. Assael, R. A. Perkins, *J. Phys. Chem. Ref. Data* **45**, 013102 (2016). (CO2 thermal conductivity)
- V. Vesovic, W. A. Wakeham, G. A. Olchowy, J. V. Sengers, J. T. R. Watson, J. Millat, *J. Phys. Chem. Ref. Data* **19**, 763 (1990).
- G. A. Olchowy, J. V. Sengers, *Int. J. Thermophys.* **10**, 417 (1989).
- Y. Rosenfeld, *J. Phys.: Condens. Matter* **11**, 5415 (1999).
- I. H. Bell, R. Messerly, M. Thol, L. Costigliola, J. C. Dyre, *J. Phys. Chem. B* **123**, 6345 (2019). (form of the master curve)
