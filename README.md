# MPSFT: Worked Examples

Python scripts and simulation input files for the worked examples in

> Tae Jun Yoon, *Molecular Physics of Supercritical Fluid Technologies: Fundamental Principles for Industrial Applications* (forthcoming).

The examples illustrate the physics discussed in the book. They are deliberately small: a few state points, short simulations, and scripts that run on a desktop computer. They are not meant to produce research-grade data, but they can be adapted to other fluids, mixtures, and conditions.

## Repository layout

```
MPSFT/
├── ch2/
│   ├── Example2.1.py
│   ├── Example2.2.py
│   ├── c2oh.xlsx            # literature data used by Example 2.2
│   └── ...
├── ch5/
│   ├── Example5.6/
│   │   ├── Example5.6.py    # analysis and plotting
│   │   ├── rho_0.050/in.lj  # one LAMMPS input per state point
│   │   └── ...
│   └── ...
└── ...
```

- One folder per chapter (`ch2` to `ch15`). Chapter 1 has no worked examples.
- Each example is a single script named after its number in the book (`Example2.2.py` is Example 2.2).
- Simulation examples (Chapters 4 to 7) have their own subfolder. Each state point has one self-contained input file in its own subfolder (for example, `T1.00/in.slab`). The Python script in the example folder only analyzes and plots the simulation output.
- Scripts that only draw schematic figures are not included.

## Requirements

The scripts were tested with Python 3.10. Most need only NumPy, SciPy, Matplotlib, and CoolProp:

```bash
pip install numpy scipy matplotlib CoolProp
```

Some examples need additional packages:

| Package | pip name | Used in |
|---|---|---|
| [phasepy](https://github.com/gustavochm/phasepy) | `phasepy` | Examples 2.2, 2.3, 9.1, 10.4, 10.5, 11.2, 12.4 |
| [SGTPy](https://github.com/gustavochm/sgtpy) | `sgtpy` | Example 2.3 |
| pandas, openpyxl | `pandas openpyxl` | Examples 2.2, 2.3, 5.2 |
| seaborn | `seaborn` | Examples 2.1 to 2.3, 3.1, 3.2, 5.2 |
| freud, Pillow | `freud-analysis pillow` | Examples 5.2, 5.3 |
| teqp | `teqp` | Example 5.6 |
| Numba | `numba` | Example 9.4 |
| iapws | `iapws` | Examples 12.1, 12.2, 13.1 |
| pyCHNOSZ | `pychnosz` | Example 13.4 (also needs R with the CHNOSZ package) |

To install everything at once:

```bash
pip install numpy scipy matplotlib CoolProp phasepy sgtpy pandas openpyxl seaborn freud-analysis pillow teqp numba iapws pychnosz
```

The simulation examples also need external programs:

- [LAMMPS](https://www.lammps.org/) for Chapters 5 to 7. Example 7.1 needs a LAMMPS build with the REAXFF package.
- [MCCCS Towhee](https://towhee.sourceforge.net/) for the Gibbs-ensemble Monte Carlo example of Chapter 4.

## How to run

The scripts take no command-line arguments. Open a script, edit the settings block at the top if needed (temperatures, compositions, number of points, and so on), and run it from its own folder:

```bash
cd ch2
python Example2.1.py
```

Each script begins with a header that states the example, followed by numbered blocks that walk through the calculation step by step. Most scripts finish in seconds. The phase-envelope calculations in Examples 2.2 and 2.3 take a few minutes.

For the simulation examples:

1. Run each input file with LAMMPS (or Towhee) inside its own subfolder, for example `lmp -in in.slab`. The runs are independent, so they can be sent to a workstation or cluster and run in parallel.
2. Run the Python script in the example folder. It reads the output files from the subfolders and produces the figure.

## Examples

### Chapter 2. Phase Behavior and Thermodynamic Anomalies

| Example | Topic | Notes |
|---|---|---|
| 2.1 | Vapor–liquid coexistence curves and p–ρ isotherms of CO2 and water | CoolProp (Span–Wagner and IAPWS-95) |
| 2.2 | Vapor–liquid equilibrium of CO2–ethanol with the PR EoS and the Wong–Sandler/NRTL mixing rule | phasepy; pseudo-arclength continuation; data in `c2oh.xlsx` |
| 2.3 | CO2–n-hexane with the SAFT-γ Mie EoS compared with the PR EoS | SGTPy, phasepy; data in `c6h14.xlsx` |
| 2.4 | Crossover PR EoS versus the classical PR EoS near the critical point of CO2 | CoolProp reference EoS |

### Chapter 3. Transport and Dynamic Anomalies

| Example | Topic | Notes |
|---|---|---|
| 3.1 | Shear viscosity and thermal conductivity of CO2 far from the critical point | CoolProp |
| 3.2 | Critical enhancement of the viscosity and thermal conductivity of CO2 | CoolProp |
| 3.3 | Excess-entropy (Rosenfeld) scaling of the viscosity of CO2, CH4, and H2O | CoolProp |

### Chapter 4. Molecular Simulations: Theory and Practices

| Example | Topic | Notes |
|---|---|---|
| 4.4 | Gibbs-ensemble Monte Carlo simulation of argon vapor–liquid equilibrium | Towhee input files |

### Chapter 5. Molecular Simulation Analyses

| Example | Topic | Notes |
|---|---|---|
| 5.1 | Radial distribution function of the Lennard-Jones fluid | LAMMPS |
| 5.2 | Local density distributions of the Lennard-Jones fluid from Voronoi tessellation | LAMMPS output from Example 5.1; freud |
| 5.3 | Percolation transition in the supercritical Lennard-Jones fluid | LAMMPS output from Example 5.1; freud |
| 5.4 | Hydrogen-bond network and tetrahedral order of TIP4P/2005 water | LAMMPS |
| 5.5 | Response functions near the critical point: fluctuation versus finite difference | LAMMPS |
| 5.6 | Excess chemical potential and entropy: thermodynamic route versus Widom insertion | LAMMPS; teqp |
| 5.7 | Self-diffusion coefficient and shear viscosity: Green–Kubo versus Einstein | LAMMPS |

### Chapter 6. Interfacial Phenomena, Wetting, and Confinement

| Example | Topic | Notes |
|---|---|---|
| 6.1 | Gibbs adsorption at the CO2/n-hexane interface | Squared-gradient theory with the PR EoS |
| 6.2 | Interfacial tension of the Lennard-Jones vapor–liquid interface from direct coexistence | LAMMPS (`in.slab`) |
| 6.3 | Squared-gradient theory of the CO2 vapor–liquid interface | PR EoS |
| 6.4 | Attractive versus repulsive walls in a slit pore | LAMMPS (`in.pore`) |

### Chapter 7. Reactions in Supercritical Fluids

| Example | Topic | Notes |
|---|---|---|
| 7.1 | ReaxFF molecular dynamics of radical-initiated methanol oxidation at several densities | LAMMPS with REAXFF (`in.reax`) |
| 7.2 | Activation volume and apparent activation energy from the compressibility of the solvent | CoolProp |
| 7.3 | Solvent clustering around an attractive solute in a near-critical Lennard-Jones fluid | LAMMPS (`in.solute`) |
| 7.4 | Piston effect in a closed cell of near-critical CO2 | CoolProp |

### Chapter 8. Industrial Milestones: Historical Landmarks and Key Applications

| Example | Topic | Notes |
|---|---|---|
| 8.1 | Solvent recovery by density collapse in a ROSE-type deasphalting loop | CoolProp |
| 8.2 | Position of the industrial milestones relative to the critical point | CoolProp |

### Chapter 9. Supercritical Fluid Extraction and Fractionation

| Example | Topic | Notes |
|---|---|---|
| 9.1 | Retrograde solubility of naphthalene in supercritical CO2 from the PR EoS | phasepy, CoolProp |
| 9.2 | Chrastil correlation of two solid solubilities and the selectivity map | CoolProp |
| 9.3 | Hansen solubility parameters of CO2 as functions of pressure and temperature | CoolProp |
| 9.4 | Extraction curve of a packed bed with the broken-and-intact-cells model | CoolProp, Numba |

### Chapter 10. Particle Formation from Supercritical Solutions

| Example | Topic | Notes |
|---|---|---|
| 10.1 | Isentropic RESS nozzle expansion of CO2 | CoolProp |
| 10.2 | Classical nucleation theory for a RESS solute | |
| 10.3 | Population balance of a RESS aerosol | |
| 10.4 | Jet break-up and dynamic surface tension vanishing in SAS (model of Marra et al.) | phasepy, CoolProp; `Example10.4_data.npz` |
| 10.5 | Feed-solvent composition and the crossover: acetone/DMSO mixtures | phasepy, CoolProp; `Example10.5_data.npz` |

### Chapter 11. Nanomaterial and Thin-Film Synthesis

| Example | Topic | Notes |
|---|---|---|
| 11.1 | Why zero-order deposition fills deep features | CoolProp |
| 11.2 | Staying out of the two-phase region during supercritical drying | phasepy, CoolProp |

### Chapter 12. Supercritical Water and Hydrothermal Processes

| Example | Topic | Notes |
|---|---|---|
| 12.1 | Water across the pseudocritical line | CoolProp, iapws |
| 12.2 | Born solvation and the pressure sensitivity of ion association | CoolProp, iapws |
| 12.3 | The pseudocritical pinch of an SCWO recuperator | CoolProp |
| 12.4 | Equilibrium gas composition of glucose SCWG | phasepy |

### Chapter 13. Electrochemistry in Sub- and Supercritical Fluids

| Example | Topic | Notes |
|---|---|---|
| 13.1 | Ion association and the conductivity maximum of dilute NaCl | CoolProp, iapws |
| 13.2 | Diffusion-limited currents in hot compressed water | CoolProp |
| 13.3 | Reversible and thermoneutral voltages in hot compressed water | CoolProp |
| 13.4 | Pourbaix diagrams of cobalt from ambient to supercritical water | pyCHNOSZ |
| 13.5 | Surface reaction versus product-layer diffusion in hot leaching of LiCoO2 | CoolProp |

### Chapter 14. Process Integration, Economic Metrics, and Sustainability Footprints

| Example | Topic | Notes |
|---|---|---|
| 14.1 | Energy balance of the CO2 solvent loop | CoolProp |
| 14.2 | Heat integration and exergy destruction in the solvent loop | CoolProp |
| 14.3 | Cost of manufacturing and carbon footprint of a batch extraction plant | |

### Chapter 15. Regulatory, Safety, and Hazard Mitigation in Supercritical Fluid Processes

| Example | Topic | Notes |
|---|---|---|
| 15.1 | Stored energy of supercritical CO2 and water and the overpressure of a vessel burst | CoolProp |
| 15.2 | Adiabatic temperature rise and feed upsets in an SCWO reactor | CoolProp |
| 15.3 | Discharge of supercritical CO2 through a relief nozzle | CoolProp |
| 15.4 | LOPA and SIL verification for extractor over-pressure | |

## Citation

If you use these scripts, please cite the book:

```
T. J. Yoon, Molecular Physics of Supercritical Fluid Technologies:
Fundamental Principles for Industrial Applications (forthcoming).
```

Please also cite the packages and reference EoSs used in each example, as listed in the corresponding chapter.

## License

<!-- Choose a license (for example, MIT) and add a LICENSE file. -->

## Contact

Tae Jun Yoon, Seoul National University. Please report problems or questions through the [Issues](https://github.com/tyoon124/MPSFT/issues) page.
