# Chapter 8. Industrial Milestones: Historical Landmarks and Key Applications

This folder contains the Python scripts for the worked examples of Chapter 8 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| Folder | Example | Figure |
|---|---|---|
| `Example8.1/` | Solvent recovery by density collapse in a ROSE-type deasphalting loop (propane, n-butane, n-pentane) | Fig. 8.2 |
| `Example8.2/` | Position of ten industrial processes relative to the critical point (compressibility number κ_T p) | Fig. 8.3 |

Fig. 8.1 (timeline of the milestones) is a drawing and has no script.

## Data in this repository

Neither example needs data files. All properties come from CoolProp, and the operating states of Example 8.2 are entered in the settings block of the script, with their sources.

## Requirements

The scripts were tested with Python 3.10 and CoolProp 8.1.

```bash
pip install numpy matplotlib CoolProp
```

CoolProp provides the properties of CO2 (Span–Wagner), water (IAPWS-95), propane, n-butane, and n-pentane.

## How to run

The scripts take no command-line arguments. Edit the settings block at the top of a script if needed, and run it from its own folder:

```bash
cd Example8.1
python Example8.1.py
```

Each script saves its figure (`figure8.X.pdf`) in its own folder. Run times: about 10 s for Example 8.1 and 20 s for Example 8.2.

## Examples

### Example 8.1. Solvent recovery by density collapse

The extract is heated at p = 1.2 p_crit from the extractor state (0.94 T_crit) to the separator state (1.02 T_crit), and the lean solvent is cooled back. This supercritical swing, with and without a counter-current exchanger (ΔT_min = 10 K), is compared with evaporative recovery (flash to the saturation pressure at 313 K, condensation, pumping, and reheat).

- Settings: `SOLVENTS`, `P_RED`, `TR_EXT`, `TR_SEP`, `DT_MIN`, `T_COND`, and `P_SCAN` (pressures of the propane curves in panel A).
- Output: `figure8.2.pdf`. The console prints the densities, the swing duty with and without heat recovery, the flash and reheat duties, the heat of vaporization, and the pump work for each solvent.

### Example 8.2. Operating states relative to the critical point

The script converts the operating state of each process (temperature, pressure range, and working fluid) to T/T_crit, p/p_crit, and ρ/ρ_crit, and evaluates κ_T p at the geometric-mean pressure and at the ends of the range. It also maps κ_T p of CO2 over 0.9 ≤ T/T_crit ≤ 1.6 and 0.3 ≤ p/p_crit ≤ 4, and locates the Widom lines of CO2 and water as the maxima of κ_T along isobars.

- Settings: `PROCESSES` (label, fluid, T, pressure range, and source of each state), `TR_RANGE`, `PR_RANGE`, `N_T`, `N_P`.
- Output: `figure8.3.pdf`. The console prints T/T_crit, p/p_crit, ρ/ρ_crit, and κ_T p of every process.
