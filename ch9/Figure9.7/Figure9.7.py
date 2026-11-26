"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Molecular Physics of Supercritical Fluid Technologies
@author: Tae Jun Yoon

Chapter 9 Supercritical Fluid Extraction and Fractionation
Figure 9.7
Process flow diagram of a generic single-train supercritical CO2 extraction plant with a two-stage
separator cascade: high-pressure liquid CO2 pump P1, preheater E1, packed-bed extraction vessel
(feedstock loaded in batch or by a carousel of vessels), pressure let-down valves V1 and V2, heaters
E2 and E3 that return the stream to the temperature of the separator, first- and second-stage
separators S1 and S2 operated at successively lower pressure for selective fractionation of
co-extracted compounds, and a condenser E4 that returns the low-pressure CO2 to the pump suction for
solvent recycle. The red circles number the states of the CO2, which are listed in the table below
the diagram (pure CO2, CoolProp; isenthalpic valves, no pressure drop, pump efficiency 0.70); state
6 is the feed to S1 and its overhead. The blue values are the pump work w and the heat duties q of
the exchangers in kJ per kg of circulating CO2 (q < 0: heat removed).

(pure CO2, CoolProp / Span-Wagner) and the specific duties per kg of circulating CO2: the numbers of
the state table and of the labels of Fig. 9.7. The diagram itself is drawn separately (PFD.svg);
this script only calculates and prints the numbers.
No command-line arguments: edit the SETTINGS block below and run   python Figure9.7.py
Streams: 1 condenser outlet (subcooled liquid) -> pump -> 2 -> preheater E1 -> 3 -> extractor -> 4
-> let-down valve V1 -> 5 -> heater E2 -> 6 -> separator S1 (extract 1) -> overhead 6 -> valve V2 ->
7 -> heater E3 -> 8 -> separator S2 (extract 2) -> overhead 8 -> condenser E4 -> 1.  The valves are
isenthalpic, heat exchangers and separators have no pressure drop, the pump has the isentropic
efficiency ETA_PUMP, and the dissolved extract is neglected (its mass fraction in the fluid is of
the order of 1 %): the numbers are the CO2 loop, not a simulation of a specific feedstock.
Requires: CoolProp.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Import packages
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from CoolProp.CoolProp import PropsSI

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Settings
Edit the parameters below to change the calculation.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ============================== SETTINGS ====================================
T_COND, P_LOW = 288.15, 5.5e6        # K, Pa, condenser outlet (subcooled liquid) and pressure of the low-pressure side
P_EXT, T_EXT  = 30e6, 323.15         # Pa, K, extraction
P_S1, T_S1    = 12e6, 323.15         # Pa, K, first separator
P_S2, T_S2    = 5.5e6, 303.15        # Pa, K, second separator
ETA_PUMP      = 0.70                 # isentropic efficiency of the pump

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Calculate the results
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ============================================================================
f = lambda out, a, va, b, vb: PropsSI(out, a, va, b, vb, "CO2")
h = lambda T, p: f("Hmass", "T", T, "P", p)
st = {}                                                                      # state: (T, p, h)
st[1] = (T_COND, P_LOW, h(T_COND, P_LOW))
h2s = f("Hmass", "Smass", f("Smass", "T", T_COND, "P", P_LOW), "P", P_EXT)
h2 = st[1][2] + (h2s - st[1][2]) / ETA_PUMP
st[2] = (f("T", "Hmass", h2, "P", P_EXT), P_EXT, h2)
st[3] = (T_EXT, P_EXT, h(T_EXT, P_EXT))
st[4] = st[3]
st[5] = (f("T", "Hmass", st[4][2], "P", P_S1), P_S1, st[4][2])
st[6] = (T_S1, P_S1, h(T_S1, P_S1))
h7 = st[6][2]
st[7] = (f("T", "Hmass", h7, "P", P_S2), P_S2, h7)
st[8] = (T_S2, P_S2, h(T_S2, P_S2))
dens = lambda k: f("Dmass", "T", st[k][0], "P", st[k][1]) if k not in (7,) else None
w_pump = st[2][2] - st[1][2]
q1 = st[3][2] - st[2][2]
q2 = st[6][2] - st[5][2]
q3 = st[8][2] - st[7][2]
qc = st[8][2] - st[1][2]
x7 = f("Q", "Hmass", h7, "P", P_S2)
print("state   T/K     p/MPa   rho/(kg/m3)   h/(kJ/kg)")
rows = []
for k in range(1, 9):
    T, p, hh = st[k]
    if k == 7:
        rho = 1 / ((1 - x7) / f("Dmass", "P", p, "Q", 0) + x7 / f("Dmass", "P", p, "Q", 1))
        ph = f"2-phase, vapor fraction {x7:.2f}"
    else:
        rho = f("Dmass", "T", T, "P", p)
        ph = "liquid" if k == 1 else "fluid"
    rows.append((k, T, p / 1e6, rho, hh / 1e3, ph))
    print(f"{k:4d} {T:8.2f} {p / 1e6:8.2f} {rho:10.1f} {hh / 1e3:12.1f}   {ph}")
print(f"pump work {w_pump / 1e3:.1f} kJ/kg; E1 {q1 / 1e3:.1f}; E2 {q2 / 1e3:.1f}; E3 {q3 / 1e3:.1f}; condenser {qc / 1e3:.1f} kJ/kg (heat removed)")
