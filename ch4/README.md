# Chapter 4. Molecular Simulations: Theory and Practices

This folder contains the input files for the worked example of Chapter 4 of

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

| File | Description |
|---|---|
| `Example4.4/equilibration/towhee_input` | Example 4.4: Towhee input for the equilibration run (Gibbs ensemble, argon, 113.15 K) |
| `Example4.4/equilibration/towhee_ff_Rowley1975` | Lennard-Jones force-field file for argon (Rowley et al. 1975) |
| `Example4.4/production/towhee_input` | Example 4.4: Towhee input for the production run |
| `Example4.4/production/towhee_ff_Rowley1975` | Same force-field file, so that each run folder is self-contained |

There is no Python script for this chapter. The example is run with MCCCS Towhee, and the results are read from the Towhee output files.

## Requirements

- [MCCCS Towhee](https://towhee.sourceforge.net/). The inputs were tested with version 8.2.3.

## How to run

Towhee reads `towhee_input` and the force-field file from the folder it is started in.

1. Equilibration:

   ```bash
   cd Example4.4/equilibration
   towhee
   ```

2. Production: copy the final configuration of the equilibration run into the production folder as `towhee_initial`, then start Towhee there.

   ```bash
   cp ../equilibration/towhee_final ../production/towhee_initial
   cd ../production
   towhee
   ```

Each run is 200 000 Monte Carlo cycles of 500 attempted moves and takes from tens of minutes to a few hours on a single core, depending on the machine.

## Example 4.4. Gibbs-ensemble Monte Carlo simulation of argon vapor–liquid equilibrium

Two simulation boxes exchange volume and particles at fixed total volume and number of atoms, and they converge to the coexisting liquid and vapor phases without an explicit interface.

### Main keywords of `towhee_input`

Towhee reads the input as pairs of lines: a keyword, followed by its value(s).

| Keyword | Value | Meaning |
|---|---|---|
| `ensemble`, `numboxes` | `'nvt'`, 2 | Gibbs ensemble (two boxes, constant total N, V, T) |
| `temperature` | 113.1525 | T in K (0.75 × 150.87 K) |
| `nmolectyp` | 500 | Number of argon atoms |
| `stepstyle`, `nstep` | `'cycles'`, 200000 | Run length in cycles (one cycle = 500 attempted moves) |
| `printfreq`, `blocksize` | 1000, 2000 | Print interval and block length (cycles) for the block averages |
| `pressure_virial_freq` | 20 | Interval (cycles) for the virial pressure |
| `trmaxdispfreq`, `volmaxdispfreq` | 100, 100 | Interval (cycles) for adjusting the maximum translation and volume steps |
| `linit` | `.true.` (equilibration), `.false.` (production) | Build a new configuration, or read `towhee_initial` |
| `initmol`, `initlattice`, `hmatrix` | 400 and 100 atoms; simple cubic; 27.9 Å and 54.7 Å | Initial liquid-like and vapor-like boxes (about 1.22 and 0.04 g cm⁻³) |
| `pmvol` | 0.002 | Volume exchange between the boxes |
| `pm2boxcbswap` | 0.05 | Configurational-bias particle swap between the boxes |
| `pmtracm` | 0.55 (equilibration), 1.0 (production) | Center-of-mass translation |
| `pmrotate` | 1.0 (equilibration only) | Rotation (no effect for single-site argon) |
| `tatrac`, `tavol` | 0.5, 0.5 | Target acceptance ratios for translation and volume moves |
| `ff_filename`, `classical_potential` | `towhee_ff_Rowley1975`, `'Lennard-Jones'` | Force field: σ = 3.405 Å, ε/k_B = 119.8 K |
| `rcut`, `ltailc` | 12.0 Å, `.true.` | Cutoff (about 3.5σ) with analytic tail corrections |

The move probabilities are **cumulative**: the actual frequency of each move is the difference from the previous entry.

| Move | Equilibration | Production |
|---|---|---|
| Volume exchange | 0.2 % | 0.2 % |
| CBMC particle swap | 4.8 % | 4.8 % |
| Translation | 50 % | 95 % |
| Rotation | 45 % | — |

### Output

- `towhee_output`: running energies, box densities, acceptance ratios, and the block averages (one block = 2000 cycles). Check in the equilibration run that the energies and densities have stopped drifting.
- `towhee_vlcc` (production run): one line with T, the vapor and liquid densities (g cm⁻³), and the vapor- and liquid-box pressures (kPa), each followed by its uncertainty.
- `towhee_final`: final configuration, used to start the next run.
- `towhee_citations`: references for the methods used.

For comparison, the author's production run at 113.15 K gave ρ_g = 0.034 ± 0.003 g cm⁻³, ρ_l = 1.227 ± 0.006 g cm⁻³ and p_sat = 0.70 ± 0.05 MPa. The reference EoS of argon gives 0.040 g cm⁻³, 1.219 g cm⁻³ and 0.81 MPa (CoolProp).

### Other temperatures

To trace the coexistence curve, repeat both runs with `temperature` and the two `hmatrix` box edges changed as below. All other settings stay the same. The last three columns are the author's production results (`towhee_vlcc`).

| T/T_crit | T (K) | Liquid box (Å) | Vapor box (Å) | ρ_g (g cm⁻³) | ρ_l (g cm⁻³) | p_sat (MPa) |
|---|---|---|---|---|---|---|
| 0.70 | 105.609 | 27.5 | 64.2 | 0.021 | 1.280 | 0.42 |
| 0.75 | 113.1525 | 27.9 | 54.7 | 0.034 | 1.227 | 0.70 |
| 0.80 | 120.697 | 28.4 | 47.3 | 0.052 | 1.172 | 1.10 |
| 0.85 | 128.2395 | 29.0 | 41.3 | 0.077 | 1.109 | 1.62 |
| 0.90 | 135.783 | 29.8 | 36.1 | 0.112 | 1.037 | 2.29 |
| 0.95 | 143.3265 | 29.8 | 36.1 | 0.163 | 0.953 | 3.16 |

T_crit and ρ_crit can then be estimated from these points with the scaling law and the law of rectilinear diameters (Chapter 4, Sec. 4.6.1).

## References

- M. G. Martin, *Mol. Simul.* **39**, 1212 (2013). (MCCCS Towhee)
- A. Z. Panagiotopoulos, *Mol. Phys.* **61**, 813 (1987). (Gibbs ensemble)
- L. A. Rowley, D. Nicholson, N. G. Parsonage, *J. Comput. Phys.* **17**, 401 (1975). (argon Lennard-Jones parameters)
- I. H. Bell, J. Wronski, S. Quoilin, V. Lemort, *Ind. Eng. Chem. Res.* **53**, 2498 (2014). (CoolProp)
