"""
Freeze-drying simulation of mixed vegetable cubes (spinach / cabbage / broccoli)
================================================================================
Thermodynamic model + simple empirical quality model.

IMPORTANT: every number marked [ASSUMED] is a typical/approximate literature-style
value chosen for this student project, NOT a measurement. Results are MODEL
ESTIMATES. Change the values in the DATA section to see how the results move.

Stages modelled
  1. Primary drying   : ice sublimates. Heat balance + vapour-transport balance
                        give the product (ice-front) temperature Tp.
  2. Secondary drying : bound water is desorbed (first-order kinetics).
  3. Quality          : structure, rehydration and vitamin C retention ->
                        overall retention (%).

Run:  python3 freeze_drying_simulation.py
Output: tables in the terminal, PNG graphs and CSV files in ./results
"""

import math
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# DATA  (component properties)
# =============================================================================
VEG = {
    #                    water    Tc(C)   VitC    blanch  RI_max  Rp slope
    #                    fraction collapse mg/100g VitC    rehyd.  (A)
    "spinach":  dict(w=0.914, Tc=-26.0, vitc=28.1, blanch=0.80, ri=0.88, A=3.0e6),
    "cabbage":  dict(w=0.922, Tc=-28.0, vitc=36.6, blanch=0.85, ri=0.92, A=3.5e6),
    "broccoli": dict(w=0.893, Tc=-24.0, vitc=89.2, blanch=0.80, ri=0.85, A=5.0e6),
}
# w, vitc  : approx. USDA composition of raw vegetables
# Tc       : [ASSUMED] collapse temperature of the frozen matrix
# blanch   : [ASSUMED] vitamin C retained after blanching
# ri       : [ASSUMED] best-case rehydration index (fraction of removed water regained)
# A        : [ASSUMED] growth of dried-layer resistance with dried thickness
NAMES = ["spinach", "cabbage", "broccoli"]

# =============================================================================
# CONSTANTS AND PROCESS PARAMETERS
# =============================================================================
DH_SUB = 2.83e6          # J/kg   latent heat of sublimation of ice
LF = 3.34e5              # J/kg   latent heat of fusion
CP_W, CP_ICE = 4186.0, 2100.0   # J/kg.K
RHO_CUBE = 950.0         # kg/m3  [ASSUMED] density of a compressed fresh cube
R0 = 4.0e4               # Pa.m2.s/kg  [ASSUMED] base mass-transfer resistance
B_RP = 50.0              # 1/m    [ASSUMED] saturation of resistance with thickness
MIX_PACKING_PENALTY = 0.20   # [ASSUMED] extra resistance for heterogeneous mixes
KC, KP, KD = 6.0, 0.25, 0.03  # [ASSUMED] Kv = KC + KP*Pc/(1+KD*Pc)  (W/m2.K, Pc in Pa)

CUBE_THICKNESS = 0.010   # m   (10 mm)
T_FREEZE = -40.0         # C   freezing temperature
T_SEC = 40.0             # C   secondary drying temperature
T_SEC_HOURS = 6.0        # h   secondary drying time

# Secondary drying / isotherm  [ASSUMED]
X_BOUND0 = 0.30          # kg water / kg dry solids left after primary drying
X_EQ = 0.04              # equilibrium moisture at secondary drying conditions
K_SEC_REF = 0.5          # 1/h at 40 C
EA_SEC = 40e3            # J/mol
OSWIN_C, OSWIN_N = 0.10, 0.5   # X = C*(aw/(1-aw))^N

# Vitamin C degradation (first order, Arrhenius)  [ASSUMED]
K_VC_REF, T_VC_REF, EA_VC = 0.01, 333.15, 50e3   # 1/h at 60 C
VC_COLLAPSE_EFFECT = 0.30   # fraction of VitC retention that depends on structure
COMPRESSION_RI_FACTOR = 0.95  # [ASSUMED] cells damaged slightly by compression

# Overall retention weights
W_STRUCT, W_REHYD, W_VITC = 0.40, 0.35, 0.25

# Acceptance criteria
AW_LIMIT = 0.60          # microbial safety
RETENTION_LIMIT = 80.0   # % "beneficial"
TIME_LIMIT_H = 36.0      # h   practical drying time (my criterion, adjustable)

# Energy model [ASSUMED]
BATCH_MASS_KG = 20.0     # kg fresh product per batch
P_PUMP_KW = 1.0          # vacuum pump power
CARNOT_EFF = 0.40        # real COP / Carnot COP
T_COND_C = -50.0         # condenser temperature
T_AMB_C = 30.0           # heat-rejection temperature
T_INITIAL_C = 25.0       # fresh crop temperature
R_GAS = 8.314


# =============================================================================
# THERMODYNAMIC FUNCTIONS
# =============================================================================
def p_ice(T_K):
    """Saturation vapour pressure over ice (Pa), Murphy & Koop (2005)."""
    return math.exp(9.550426 - 5723.265 / T_K + 3.53068 * math.log(T_K)
                    - 0.00728332 * T_K)


def kv(Pc):
    """Shelf-to-product heat transfer coefficient (W/m2.K), rises with pressure."""
    return KC + KP * Pc / (1 + KD * Pc)


def mix_props(x):
    """Mass-fraction weighted properties of the cube. x = (spinach, cabbage, broccoli)."""
    w = sum(xi * VEG[n]["w"] for xi, n in zip(x, NAMES))
    A = sum(xi * VEG[n]["A"] for xi, n in zip(x, NAMES))
    hetero = 1.0 - sum(xi ** 2 for xi in x)          # 0 for pure, 0.67 for 33/33/33
    packing = 1.0 + MIX_PACKING_PENALTY * hetero
    return w, A, packing


def solve_Tp(Ts_K, Pc, Rp, Kv):
    """Solve heat balance  Kv(Ts-Tp) = dHs*(Pice(Tp)-Pc)/Rp  for Tp by bisection."""
    f = lambda T: Kv * (Ts_K - T) - DH_SUB * (p_ice(T) - Pc) / Rp
    lo, hi = 200.0, 273.15
    if f(hi) > 0:                 # balance only reached above 0 C -> ice melts
        return None, "melt"
    if p_ice(Ts_K) <= Pc:         # shelf too cold for this pressure -> no sublimation
        return None, "stall"
    lo = max(lo, 200.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), "ok"


def primary_drying(w, A, packing, Ts_C, Pc, L=CUBE_THICKNESS, n=150):
    """Integrate the sublimation front through the cube thickness."""
    Ts_K = Ts_C + 273.15
    Kv = kv(Pc)
    ice_per_vol = RHO_CUBE * w               # kg ice / m3
    dl = L / n
    t, Tp_max, Tp_sum = 0.0, -1e9, 0.0
    for i in range(n):
        l = (i + 0.5) * dl
        Rp = (R0 + A * l / (1 + B_RP * l)) * packing
        Tp, status = solve_Tp(Ts_K, Pc, Rp, Kv)
        if status != "ok":
            return dict(status=status, time_h=float("inf"), Tp_max=None, Tp_avg=None)
        flux = (p_ice(Tp) - Pc) / Rp        # kg/m2.s
        t += ice_per_vol * dl / flux
        Tp_max = max(Tp_max, Tp - 273.15)
        Tp_sum += Tp - 273.15
    return dict(status="ok", time_h=t / 3600.0, Tp_max=Tp_max, Tp_avg=Tp_sum / n)


def secondary_drying(T_C=T_SEC, hours=T_SEC_HOURS):
    """First-order desorption. Returns final moisture (dry basis, wet basis) and aw."""
    T = T_C + 273.15
    k = K_SEC_REF * math.exp(-EA_SEC / R_GAS * (1 / T - 1 / 313.15))
    X = X_EQ + (X_BOUND0 - X_EQ) * math.exp(-k * hours)
    xwb = X / (1 + X)
    r = (X / OSWIN_C) ** (1 / OSWIN_N)       # = aw/(1-aw)
    aw = r / (1 + r)
    return X, xwb * 100.0, aw


def structure_score(Tp_max, Tc):
    """Logistic collapse model: 50 % retained when Tp_max = Tc, 2 K width."""
    return 1.0 / (1.0 + math.exp((Tp_max - Tc) / 2.0))


def vitc_retention(S, name):
    k = K_VC_REF * math.exp(-EA_VC / R_GAS * (1 / (T_SEC + 273.15) - 1 / T_VC_REF))
    thermal = math.exp(-k * T_SEC_HOURS)
    return VEG[name]["blanch"] * thermal * ((1 - VC_COLLAPSE_EFFECT) + VC_COLLAPSE_EFFECT * S)


def energy_per_kg_dry(w, time_total_h, Ts_C):
    """Electrical energy (kWh) per kg of dried product -- estimate."""
    m_w = w * BATCH_MASS_KG
    m_dry = BATCH_MASS_KG - m_w * 0.97       # ~97 % of water removed
    cop = lambda Tcold, Thot: CARNOT_EFF * (Tcold + 273.15) / (Thot - Tcold)
    # freezing: cool water from T_initial to 0, freeze, cool ice to T_FREEZE
    q_freeze = m_w * (CP_W * T_INITIAL_C + LF + CP_ICE * (-T_FREEZE))
    e_freeze = q_freeze / cop(T_FREEZE, T_AMB_C) / 3.6e6
    # sublimation heat is supplied by heater (eta=1) and removed again by condenser
    q_sub = m_w * DH_SUB
    e_heat = q_sub / 3.6e6
    e_cond = q_sub / cop(T_COND_C, T_AMB_C) / 3.6e6
    e_pump = P_PUMP_KW * time_total_h
    return (e_freeze + e_heat + e_cond + e_pump) / m_dry


# =============================================================================
# FULL SIMULATION OF ONE CUBE
# =============================================================================
def simulate(x, Pc, Ts_C):
    """x = (spinach, cabbage, broccoli) mass fractions (sum = 1). Pc in Pa."""
    w, A, packing = mix_props(x)
    pd = primary_drying(w, A, packing, Ts_C, Pc)
    X, xwb, aw = secondary_drying()
    res = dict(x=x, Pc=Pc, Ts=Ts_C, status=pd["status"], water=w * 100,
               time_h=pd["time_h"], Tp_max=pd["Tp_max"], Tp_avg=pd["Tp_avg"],
               moisture_wb=xwb, aw=aw)
    if pd["status"] != "ok":
        res.update(structure=0.0, rehyd=0.0, vitc=0.0, retention=0.0,
                   energy=float("nan"), verdict="FAIL (" + pd["status"] + ")")
        return res
    S = rehyd = vc = 0.0
    for xi, n in zip(x, NAMES):
        if xi == 0:
            continue
        Si = structure_score(pd["Tp_max"], VEG[n]["Tc"])
        S += xi * Si
        rehyd += xi * VEG[n]["ri"] * COMPRESSION_RI_FACTOR * (0.4 + 0.6 * Si)
        vc += xi * vitc_retention(Si, n)
    ret = 100 * (W_STRUCT * S + W_REHYD * rehyd + W_VITC * vc)
    total_h = pd["time_h"] + T_SEC_HOURS
    ok = aw <= AW_LIMIT and ret >= RETENTION_LIMIT and pd["time_h"] <= TIME_LIMIT_H
    res.update(structure=100 * S, rehyd=100 * rehyd, vitc=100 * vc, retention=ret,
               energy=energy_per_kg_dry(w, total_h, Ts_C),
               verdict="EDIBLE & BENEFICIAL" if ok else "NOT ACCEPTABLE")
    return res


# =============================================================================
# EXPERIMENT 2 : composition study at fixed process conditions
# =============================================================================
BASE_PC = 20.0     # Pa  (0.2 mbar)
BASE_TS = 20.0     # C   shelf temperature in primary drying

def experiment_2():
    comps = [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2)]
    labels = ["100/0/0", "50/50/0", "33/33/33", "60/20/20"]
    print("\n" + "=" * 100)
    print(f"EXPERIMENT 2a : composition (spinach/cabbage/broccoli %)  at Pc = {BASE_PC} Pa, "
          f"Ts = {BASE_TS} C")
    print("=" * 100)
    hdr = ("Mix", "Water%", "Tp_max C", "t_prim h", "Moist% wb", "aw", "Struct%",
           "Rehyd%", "VitC%", "OVERALL%", "Verdict")
    print(("{:<10}" + "{:>10}" * 9 + "  {}").format(*hdr))
    rows = []
    for c, lab in zip(comps, labels):
        r = simulate(c, BASE_PC, BASE_TS)
        rows.append((lab, r))
        print(("{:<10}" + "{:>10.1f}" * 3 + "{:>10.2f}{:>10.3f}" + "{:>10.1f}" * 4 + "  {}").format(
            lab, r["water"], r["Tp_max"], r["time_h"], r["moisture_wb"], r["aw"],
            r["structure"], r["rehyd"], r["vitc"], r["retention"], r["verdict"]))
    with open(f"{OUT_DIR}/experiment2_table.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mix_spin_cab_broc", "water_%", "Tp_max_C", "primary_time_h",
                     "final_moisture_%wb", "aw", "structure_%", "rehydration_%",
                     "vitaminC_%", "overall_retention_%", "verdict"])
        for lab, r in rows:
            wr.writerow([lab, round(r["water"], 1), round(r["Tp_max"], 2),
                         round(r["time_h"], 2), round(r["moisture_wb"], 2),
                         round(r["aw"], 3), round(r["structure"], 1),
                         round(r["rehyd"], 1), round(r["vitc"], 1),
                         round(r["retention"], 1), r["verdict"]])

    # ---- 2b : curves. x-axis = spinach %, one curve per broccoli %, cabbage = rest
    print("\nEXPERIMENT 2b : overall retention (%) vs spinach %, curves = broccoli %"
          " (cabbage = remainder)")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    broc_levels = [0, 20, 40, 60, 80]
    csv_rows = []
    for b in broc_levels:
        xs, ys = [], []
        for s in range(0, 100 - b + 1, 5):
            c = (s / 100, (100 - s - b) / 100, b / 100)
            r = simulate(c, BASE_PC, BASE_TS)
            xs.append(s)
            ys.append(r["retention"])
            csv_rows.append([b, s, 100 - s - b, round(r["retention"], 2)])
        ax.plot(xs, ys, marker="o", ms=4, label=f"Broccoli {b}%")
        print(f"  broccoli {b:>2}% : " + ", ".join(f"{y:.1f}" for y in ys[:: max(1, len(ys) // 5)]))
    ax.axhline(RETENTION_LIMIT, color="grey", ls="--", lw=1)
    ax.text(1, RETENTION_LIMIT + 0.3, "80 % acceptance limit", fontsize=8, color="grey")
    ax.set_xlabel("Spinach in cube (%)")
    ax.set_ylabel("Overall retention / efficiency (%)")
    ax.set_title(f"Experiment 2: effect of composition  (Pc = {BASE_PC} Pa, Ts = {BASE_TS} °C)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/experiment2_curves.png", dpi=200)
    plt.close(fig)
    with open(f"{OUT_DIR}/experiment2_curves.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["broccoli_%", "spinach_%", "cabbage_%", "overall_retention_%"])
        wr.writerows(csv_rows)


# =============================================================================
# EXPERIMENT 3 : process sweep (pressure & shelf temperature) for a 33/33/33 cube
# =============================================================================
def experiment_3():
    mix = (1 / 3, 1 / 3, 1 / 3)
    Pcs = np.array([2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300], dtype=float)  # Pa
    Tss = np.arange(-10, 61, 10, dtype=float)                                      # C
    grid = {}
    for P in Pcs:
        for T in Tss:
            grid[(P, T)] = simulate(mix, P, T)

    print("\n" + "=" * 100)
    print("EXPERIMENT 3 : process sweep, 33/33/33 cube")
    print("=" * 100)
    print("\nOverall retention (%)   rows = chamber pressure (Pa), columns = shelf temp (C)")
    print("{:>8}".format("Pc\\Ts") + "".join(f"{int(T):>8}" for T in Tss))
    for P in Pcs:
        print(f"{P:>8.0f}" + "".join(f"{grid[(P, T)]['retention']:>8.1f}" for T in Tss))

    print("\nPrimary drying time (h)   ('inf' = no drying / melted)")
    print("{:>8}".format("Pc\\Ts") + "".join(f"{int(T):>8}" for T in Tss))
    for P in Pcs:
        print(f"{P:>8.0f}" + "".join(
            f"{grid[(P, T)]['time_h']:>8.1f}" if grid[(P, T)]['time_h'] != float('inf')
            else f"{'inf':>8}" for T in Tss))

    print("\nMax product temperature Tp_max (C)  (collapse limit of weakest component: "
          f"{min(v['Tc'] for v in VEG.values()):.0f} C)")
    print("{:>8}".format("Pc\\Ts") + "".join(f"{int(T):>8}" for T in Tss))
    for P in Pcs:
        print(f"{P:>8.0f}" + "".join(
            f"{grid[(P, T)]['Tp_max']:>8.1f}" if grid[(P, T)]['Tp_max'] is not None
            else f"{'--':>8}" for T in Tss))

    print("\nEnergy (kWh per kg dried product)")
    print("{:>8}".format("Pc\\Ts") + "".join(f"{int(T):>8}" for T in Tss))
    for P in Pcs:
        print(f"{P:>8.0f}" + "".join(
            f"{grid[(P, T)]['energy']:>8.0f}" if not math.isnan(grid[(P, T)]['energy'])
            else f"{'--':>8}" for T in Tss))

    # best acceptable setting = lowest energy among acceptable ones
    acc = [(k, v) for k, v in grid.items() if v["verdict"] == "EDIBLE & BENEFICIAL"]
    print("\nAcceptable settings (aw<=0.6, retention>=80 pct, primary time<=%.0f h): %d of %d"
          % (TIME_LIMIT_H, len(acc), len(grid)))
    if acc:
        (P, T), v = min(acc, key=lambda kv: kv[1]["energy"])
        print(f"Lowest-energy acceptable setting : Pc = {P:.0f} Pa, Ts = {T:.0f} C -> "
              f"retention {v['retention']:.1f} %, time {v['time_h']:.1f} h, "
              f"energy {v['energy']:.0f} kWh/kg")
        (P2, T2), v2 = max(acc, key=lambda kv: kv[1]["retention"])
        print(f"Highest-retention setting        : Pc = {P2:.0f} Pa, Ts = {T2:.0f} C -> "
              f"retention {v2['retention']:.1f} %, time {v2['time_h']:.1f} h, "
              f"energy {v2['energy']:.0f} kWh/kg")

    # save CSV
    with open(f"{OUT_DIR}/experiment3_grid.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Pc_Pa", "Ts_C", "status", "Tp_max_C", "primary_time_h", "retention_%",
                     "structure_%", "rehydration_%", "vitaminC_%", "energy_kWh_per_kg", "verdict"])
        for (P, T), v in grid.items():
            wr.writerow([P, T, v["status"],
                         None if v["Tp_max"] is None else round(v["Tp_max"], 2),
                         None if v["time_h"] == float("inf") else round(v["time_h"], 2),
                         round(v["retention"], 2), round(v["structure"], 2),
                         round(v["rehyd"], 2), round(v["vitc"], 2),
                         None if math.isnan(v["energy"]) else round(v["energy"], 1),
                         v["verdict"]])

    # ---- Graph 3a : heat map of overall retention
    Z = np.array([[grid[(P, T)]["retention"] for T in Tss] for P in Pcs])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(Tss)))
    ax.set_xticklabels([int(t) for t in Tss])
    ax.set_yticks(range(len(Pcs)))
    ax.set_yticklabels([int(p) for p in Pcs])
    for i in range(len(Pcs)):
        for j in range(len(Tss)):
            ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Shelf temperature during primary drying (°C)")
    ax.set_ylabel("Chamber pressure (Pa)")
    ax.set_title("Experiment 3: overall retention (%) for 33/33/33 cube")
    fig.colorbar(im, label="Overall retention (%)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/experiment3_heatmap.png", dpi=200)
    plt.close(fig)

    # ---- Graph 3b : retention vs pressure, one curve per shelf temperature
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for T in [0, 20, 40, 60]:
        axs[0].plot(Pcs, [grid[(P, float(T))]["retention"] for P in Pcs], marker="o",
                    ms=4, label=f"Ts = {T} °C")
    axs[0].set_xscale("log")
    axs[0].axhline(RETENTION_LIMIT, color="grey", ls="--", lw=1)
    axs[0].set_xlabel("Chamber pressure (Pa)")
    axs[0].set_ylabel("Overall retention (%)")
    axs[0].set_title("Retention vs chamber pressure")
    axs[0].grid(alpha=0.3)
    axs[0].legend()
    # ---- Graph 3c : retention vs shelf temperature, one curve per pressure
    for P in [5, 20, 50, 120]:
        axs[1].plot(Tss, [grid[(float(P), T)]["retention"] for T in Tss], marker="o",
                    ms=4, label=f"Pc = {P} Pa")
    axs[1].axhline(RETENTION_LIMIT, color="grey", ls="--", lw=1)
    axs[1].set_xlabel("Shelf temperature (°C)")
    axs[1].set_ylabel("Overall retention (%)")
    axs[1].set_title("Retention vs shelf temperature")
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/experiment3_retention_curves.png", dpi=200)
    plt.close(fig)

    # ---- Graph 3d : drying time and energy vs shelf temperature
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for P in [5, 20, 50, 120]:
        t = [grid[(float(P), T)]["time_h"] for T in Tss]
        e = [grid[(float(P), T)]["energy"] for T in Tss]
        t = [np.nan if v == float("inf") else v for v in t]
        axs[0].plot(Tss, t, marker="o", ms=4, label=f"Pc = {P} Pa")
        axs[1].plot(Tss, e, marker="o", ms=4, label=f"Pc = {P} Pa")
    axs[0].axhline(TIME_LIMIT_H, color="grey", ls="--", lw=1)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Shelf temperature (°C)")
    axs[0].set_ylabel("Primary drying time (h)")
    axs[0].set_title("Drying time vs shelf temperature")
    axs[1].set_xlabel("Shelf temperature (°C)")
    axs[1].set_ylabel("Energy (kWh per kg dried product)")
    axs[1].set_title("Energy use vs shelf temperature")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/experiment3_time_energy.png", dpi=200)
    plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    print("Freeze-drying model  -  cube thickness %.0f mm, freeze at %.0f C, "
          "secondary drying %.0f C for %.0f h" % (CUBE_THICKNESS * 1000, T_FREEZE,
                                                   T_SEC, T_SEC_HOURS))
    experiment_2()
    experiment_3()
    print("\nGraphs and CSV files saved in ./%s/" % OUT_DIR)
