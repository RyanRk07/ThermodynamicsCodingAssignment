#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OTTO CYCLE SIMULATOR
BAMEE203L - Thermodynamics and Thermal Systems | Assignment II | Set E | Question 1
=====================================================================================

Air-standard Otto cycle:
    1-2  isentropic compression
    2-3  constant-volume heat addition
    3-4  isentropic expansion
    4-1  constant-volume heat rejection

Assignment case (Ryan Roshan Kuruvilla, 25BME1227):
    r = 8, P1 = 1 bar, T1 = 300 K, and the temperature at the end of the
    isentropic expansion is T4 = 800 K.  The heat addition
    follows from T4:   T3 = T4 * r^(g-1),   q_in = cv * (T3 - T2).


"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    TK_OK = True
except Exception:
    TK_OK = False

# =====================================================================================
#  1.  THERMODYNAMIC MODEL
# =====================================================================================

CV_TEXTBOOK = 0.718  # kJ/kg.K  (air at room temperature)


@dataclass(frozen=True)
class Params:
    r: float = 8.0  # compression ratio
    P1: float = 100.0  # kPa  (1 bar)
    T1: float = 300.0  # K
    gamma: float = 1.4
    R: float = 0.287  # kJ/kg.K
    mode: str = "T4"  # "q" | "T3" | "T4"
    value: float = 800.0  # value of mode quantity
    textbook_cv: bool = True  # True: cv = 0.718 ; False: cv = R/(gamma-1)


ASSIGNMENT = Params(r=8, P1=100, T1=300, gamma=1.4, R=0.287, mode="T4", value=800.0)


@dataclass
class Result:
    p: Params
    cv: float
    T: np.ndarray  # [T1, T2, T3, T4]  (K)
    P: np.ndarray  # [P1, P2, P3, P4]  (kPa)
    v: np.ndarray  # [v1, v2, v3, v4]  (m^3/kg)
    s: np.ndarray  # s - s1  (kJ/kg.K)
    q_in: float
    q_out: float
    w_comp: float
    w_exp: float
    w_net: float
    eta: float
    mep: float  # kPa

    @property
    def Tmax(self):
        return self.T[2]

    @property
    def Pmax(self):
        return self.P[2]


def solve_otto(p: Params) -> Result:
    """Solve the air-standard Otto cycle. Raises ValueError for non-physical input."""
    r, g, R = p.r, p.gamma, p.R
    if r <= 1:
        raise ValueError("Compression ratio must be greater than 1.")
    if g <= 1:
        raise ValueError("gamma must be greater than 1.")
    if p.P1 <= 0 or p.T1 <= 0 or R <= 0:
        raise ValueError("P1, T1 and R must be positive.")

    cv = CV_TEXTBOOK if p.textbook_cv else R / (g - 1.0)
    k = r ** (g - 1.0)
    T1, P1 = p.T1, p.P1
    v1 = R * T1 / P1
    v2 = v1 / r
    T2 = T1 * k
    P2 = P1 * r ** g

    if p.mode == "q":
        if p.value <= 0:
            raise ValueError("Heat addition must be positive.")
        q_in = p.value
        T3 = T2 + q_in / cv
    elif p.mode == "T3":
        T3 = p.value
        if T3 <= T2:
            raise ValueError(f"T3 must exceed T2 = {T2:.1f} K.")
        q_in = cv * (T3 - T2)
    elif p.mode == "T4":
        if p.value <= T1:
            raise ValueError(f"T4 must exceed T1 = {T1:.1f} K.")
        T3 = p.value * k
        q_in = cv * (T3 - T2)
    else:
        raise ValueError(f"Unknown mode {p.mode!r}")

    T4 = T3 / k
    P3 = P2 * T3 / T2
    P4 = P3 / r ** g
    q_out = cv * (T4 - T1)
    w_comp = cv * (T2 - T1)
    w_exp = cv * (T3 - T4)
    w_net = q_in - q_out
    eta = w_net / q_in
    mep = w_net / (v1 - v2)

    cvs = R / (g - 1.0)
    s3 = cvs * np.log(T3 / T2)
    return Result(
        p=p, cv=cv,
        T=np.array([T1, T2, T3, T4]), P=np.array([P1, P2, P3, P4]),
        v=np.array([v1, v2, v2, v1]), s=np.array([0.0, 0.0, s3, s3]),
        q_in=q_in, q_out=q_out, w_comp=w_comp, w_exp=w_exp, w_net=w_net, eta=eta, mep=mep,
    )


def cycle_curves(res: Result, n_stroke: int = 120, n_iso: int = 40):
    p = res.p
    g, R = p.gamma, p.R
    cvs = R / (g - 1.0)
    (T1, T2, T3, T4), (P1, P2, P3, P4) = res.T, res.P
    v1, v2 = res.v[0], res.v[1]
    s3 = res.s[2]
    out = []

    v = np.linspace(v1, v2, n_stroke)
    out.append(dict(v=v, P=P1 * (v1 / v) ** g, T=T1 * (v1 / v) ** (g - 1), s=np.zeros(n_stroke)))

    T = np.linspace(T2, T3, n_iso)
    out.append(dict(v=np.full(n_iso, v2), P=P2 * T / T2, T=T, s=cvs * np.log(T / T2)))

    v = np.linspace(v2, v1, n_stroke)
    out.append(dict(v=v, P=P3 * (v2 / v) ** g, T=T3 * (v2 / v) ** (g - 1), s=np.full(n_stroke, s3)))

    T = np.linspace(T4, T1, n_iso)
    out.append(dict(v=np.full(n_iso, v1), P=P4 * T / T4, T=T, s=s3 + cvs * np.log(T / T4)))
    return out


def engine_metrics(res: Result, disp_cc: float, rpm: float):
    r = res.p.r
    Vd = disp_cc * 1e-6
    V1 = Vd * r / (r - 1.0)
    m = res.P[0] * V1 / (res.p.R * res.T[0])
    W = m * res.w_net
    power = W * rpm / 120.0
    return dict(m_g=m * 1e3, W_J=W * 1e3, P_kW=power, V1_cc=V1 * 1e6, Vc_cc=V1 * 1e6 / r)


HOLD_LABELS = {
    "q_in (fixed heat addition)": "q",
    "T3 (fixed peak temperature)": "T3",
    "T4 (fixed end-expansion temperature)": "T4",
}
HOLD_NAME = {"q": "q_in", "T3": "T3", "T4": "T4"}


def hold_value(base: Result, hold: str) -> float:
    return {"q": base.q_in, "T3": base.T[2], "T4": base.T[3]}[hold]


def derive(base: Result, hold: str, **override) -> Result:
    return solve_otto(replace(base.p, mode=hold, value=hold_value(base, hold), **override))


def apply_x(base: Result, hold: str, key: str, x: float) -> Result:
    if key == "q":
        return solve_otto(replace(base.p, mode="q", value=x))
    if key == "P1":
        return derive(base, hold, P1=x * 100.0)
    return derive(base, hold, **{key: x})


def sweep_series(base, hold, key, xs, yfun):
    ys = []
    for x in xs:
        try:
            ys.append(yfun(apply_x(base, hold, key, x)))
        except ValueError:
            ys.append(np.nan)
    return np.array(ys)


SWEEP_X = {
    "Compression ratio r": ("r", 2.0, 20.0),
    "Heat addition q_in (kJ/kg)": ("q", 100.0, 2500.0),
    "Inlet temperature T1 (K)": ("T1", 250.0, 450.0),
    "Inlet pressure P1 (bar)": ("P1", 0.3, 3.0),
    "Specific-heat ratio gamma": ("gamma", 1.2, 1.67),
}
SWEEP_Y = {
    "Thermal efficiency (%)": lambda x: x.eta * 100,
    "Net work w_net (kJ/kg)": lambda x: x.w_net,
    "Mean effective pressure (bar)": lambda x: x.mep / 100,
    "Max pressure P3 (bar)": lambda x: x.P[2] / 100,
    "Max temperature T3 (K)": lambda x: x.T[2],
    "T2 - end of compression (K)": lambda x: x.T[1],
    "T4 - end of expansion (K)": lambda x: x.T[3],
    "Heat rejected q_out (kJ/kg)": lambda x: x.q_out,
    "Heat added q_in (kJ/kg)": lambda x: x.q_in,
}

# =====================================================================================
#  2.  PLOTTING
# =====================================================================================

SEG_COLORS = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]
SEG_SHORT = ["1-2 compression", "2-3 heat add", "3-4 expansion", "4-1 heat rej"]
SEG_STROKE = ["COMPRESSION stroke", "COMBUSTION / heat addition", "POWER stroke (expansion)",
              "EXHAUST / heat rejection"]


def draw_pv(ax, res, curves, vscale=1.0, log=False, shade=True, xlabel=None, legend=True):
    for k, c in enumerate(curves):
        ax.plot(c["v"] * vscale, c["P"] / 100, color=SEG_COLORS[k], lw=2, label=SEG_SHORT[k])
    vv = np.concatenate([c["v"] for c in curves]) * vscale
    PP = np.concatenate([c["P"] for c in curves]) / 100
    if shade:
        ax.fill(vv, PP, color="#1f77b4", alpha=0.10)
        v1, v2 = res.v[0] * vscale, res.v[1] * vscale
        ax.hlines(res.mep / 100, v2, v1, colors="k", linestyles="--", lw=1)
        ax.text(v1, res.mep / 100, f" MEP={res.mep / 100:.2f} bar", va="bottom", ha="right", fontsize=8)
    ax.plot(res.v * vscale, res.P / 100, "ko", ms=4, zorder=5)
    offs = {0: (5, -10), 1: (-20, -10), 2: (-25, 2), 3: (5, 2)}
    for i in range(4):
        ax.annotate(f"{i + 1}\n{res.P[i] / 100:.1f}b", (res.v[i] * vscale, res.P[i] / 100),
                    xytext=offs[i], textcoords="offset points", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    ax.set_xlabel(xlabel or "v (m³/kg)", fontsize=8)
    ax.set_ylabel("P (bar)", fontsize=8)
    ax.set_title("P-v Diagram", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=7, loc="upper right")


def draw_ts(ax, res, curves, shade=True):
    for k, c in enumerate(curves):
        ax.plot(c["s"], c["T"], color=SEG_COLORS[k], lw=2)
    if shade:
        ax.fill(np.concatenate([c["s"] for c in curves]), np.concatenate([c["T"] for c in curves]),
                color="#d62728", alpha=0.08)
    ax.plot(res.s, res.T, "ko", ms=4, zorder=5)
    offs = {0: (-20, -10), 1: (-20, 0), 2: (5, 2), 3: (5, -4)}
    for i in range(4):
        ax.annotate(f"{i + 1}\n{res.T[i]:.0f}K", (res.s[i], res.T[i]), xytext=offs[i],
                    textcoords="offset points", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    ax.set_xlabel("s - s₁ (kJ/kg·K)", fontsize=8)
    ax.set_ylabel("T (K)", fontsize=8)
    ax.set_title("T-s Diagram", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.margins(x=0.25, y=0.1)
    ax.grid(True, alpha=0.3)


def hold_description(base: Result, hold: str) -> str:
    val = hold_value(base, hold)
    unit = "kJ/kg" if hold == "q" else "K"
    return f"{HOLD_NAME[hold]} = {val:.1f} {unit}"


def build_compare_figure(fig, base: Result, ratios, hold: str):
    fig.clear()
    gs = GridSpec(2, 3, figure=fig)
    ax_pv, ax_ts = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_eta, ax_mep = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax_pmax, ax_tmax = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    cmap = matplotlib.colormaps["tab10"]
    results = []
    for i, r in enumerate(ratios):
        try:
            res = derive(base, hold, r=r)
        except ValueError:
            continue
        results.append(res)
        col = cmap(i % 10)
        cur = cycle_curves(res)
        ax_pv.plot(np.concatenate([c["v"] for c in cur]), np.concatenate([c["P"] for c in cur]) / 100,
                   color=col, lw=1.5, label=f"r = {r:g}")
        ax_ts.plot(np.concatenate([c["s"] for c in cur]), np.concatenate([c["T"] for c in cur]),
                   color=col, lw=1.5, label=f"r = {r:g}")
        ax_pv.plot(res.v[2], res.P[2] / 100, "o", color=col, ms=4)
        ax_ts.plot(res.s[2], res.T[2], "o", color=col, ms=4)

    ax_pv.set_title("P-v Diagrams", fontsize=9, fontweight="bold")
    ax_ts.set_title("T-s Diagrams", fontsize=9, fontweight="bold")
    for a in (ax_pv, ax_ts):
        a.tick_params(labelsize=8)
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7)

    rr = np.linspace(2, 20, 100)
    series = [
        (ax_eta, "Efficiency", "η (%)", lambda x: x.eta * 100),
        (ax_mep, "MEP", "MEP (bar)", lambda x: x.mep / 100),
        (ax_pmax, "Max Pressure P₃", "P₃ (bar)", lambda x: x.P[2] / 100),
        (ax_tmax, "Max Temp T₃", "T₃ (K)", lambda x: x.T[2]),
    ]
    for ax, title, ylabel, f in series:
        ax.plot(rr, sweep_series(base, hold, "r", rr, f), color="#888", lw=1.2)
        for i, res in enumerate(results):
            y = f(res)
            ax.plot(res.p.r, y, "o", ms=5, color=cmap(matplotlib_idx(ratios, res.p.r) % 10))
            ax.annotate(f"{y:.1f}" if y >= 10 else f"{y:.2f}", (res.p.r, y), xytext=(3, -8),
                        textcoords="offset points", fontsize=7)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("compression ratio r", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Compression Ratio Comparison ({hold_description(base, hold)})", fontsize=10, fontweight="bold")
    return results


def matplotlib_idx(ratios, r):
    for i, x in enumerate(ratios):
        if abs(x - r) < 1e-9:
            return i
    return 0


def build_sweep_figure(fig, base, hold, xlabel, xkey, xmin, xmax, ylabel, yfun):
    fig.clear()
    ax = fig.add_subplot(111)
    xs = np.linspace(xmin, xmax, 160)
    ys = sweep_series(base, hold, xkey, xs, yfun)
    ax.plot(xs, ys, lw=2, color="#1f77b4")
    xb = {"r": base.p.r, "q": base.q_in, "T1": base.p.T1, "P1": base.p.P1 / 100, "gamma": base.p.gamma}[xkey]
    if xmin <= xb <= xmax:
        yb = yfun(base)
        ax.plot(xb, yb, "*", ms=12, color="#d62728", zorder=5, label=f"Current ({xb:.4g}, {yb:.4g})")
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"{ylabel} vs {xlabel}", fontsize=10, fontweight="bold")


# =====================================================================================
#  3.  TEXT OUTPUT
# =====================================================================================

def working_text(res: Result) -> str:
    p = res.p
    r, g, R, cv = p.r, p.gamma, p.R, res.cv
    T1, T2, T3, T4 = res.T
    P1, P2, P3, P4 = res.P / 100.0
    v1, v2 = res.v[0], res.v[1]
    k = r ** (g - 1)
    L = []
    A = L.append
    A("AIR-STANDARD OTTO CYCLE - WORKED SOLUTION")
    A("=" * 60)
    A("GIVEN")
    A(f"  r = {r:g}   P1 = {P1:.3f} bar ({p.P1:.1f} kPa)   T1 = {T1:.1f} K")
    A(f"  gamma = {g:g}   R = {R:g} kJ/kg.K   cv = {cv:.4f} kJ/kg.K")
    if p.mode == "T4":
        A(f"  End-expansion temperature T4 = {T4:.1f} K")
    elif p.mode == "T3":
        A(f"  Peak temperature T3 = {T3:.1f} K")
    else:
        A(f"  Heat addition q_in = {res.q_in:.2f} kJ/kg")
    A("")
    A("STEP 1  Specific volumes")
    A(f"  v1 = R T1 / P1 = {v1:.4f} m^3/kg")
    A(f"  v2 = v3 = v1 / r = {v2:.4f} m^3/kg")
    A("")
    A("STEP 2  1-2 Isentropic compression")
    A(f"  T2 = T1 r^(g-1) = {T2:.2f} K")
    A(f"  P2 = P1 r^g     = {P2:.3f} bar")
    A("")
    A("STEP 3  2-3 Constant-volume heat addition")
    if p.mode == "T4":
        A(f"  T3 = T4 r^(g-1) = {T3:.2f} K")
        A(f"  q_in = cv (T3 - T2) = {res.q_in:.2f} kJ/kg")
    elif p.mode == "T3":
        A(f"  q_in = cv (T3 - T2) = {res.q_in:.2f} kJ/kg")
    else:
        A(f"  T3 = T2 + q_in / cv = {T3:.2f} K")
    A(f"  P3 = P2 T3 / T2 = {P3:.3f} bar   <-- MAXIMUM PRESSURE")
    A(f"  T3 = {T3:.2f} K                 <-- MAXIMUM TEMPERATURE")
    A("")
    A("STEP 4  3-4 Isentropic expansion")
    A(f"  T4 = T3 / r^(g-1) = {T4:.2f} K")
    A(f"  P4 = P3 / r^g     = {P4:.3f} bar")
    A("")
    A("STEP 5  Work and efficiency")
    A(f"  q_out = cv (T4 - T1) = {res.q_out:.2f} kJ/kg")
    A(f"  w_net = q_in - q_out = {res.w_net:.2f} kJ/kg")
    A(f"  eta   = 1 - 1/r^(g-1) = {res.eta * 100:.2f} %")
    A("")
    A("STEP 6  Mean effective pressure")
    A(f"  MEP = w_net / (v1 - v2) = {res.mep / 100:.3f} bar")
    A("")
    A("=" * 60)
    A("SUMMARY RESULTS")
    A(f"  q_in  = {res.q_in:.1f} kJ/kg    | T_max = {T3:.1f} K")
    A(f"  w_net = {res.w_net:.1f} kJ/kg    | P_max = {P3:.2f} bar")
    A(f"  eta   = {res.eta * 100:.2f} %     | MEP   = {res.mep / 100:.3f} bar")
    return "\n".join(L)


def comparison_rows(base: Result, ratios, hold="q"):
    rows = []
    for r in ratios:
        try:
            x = derive(base, hold, r=r)
        except ValueError:
            continue
        rows.append(x)
    return rows


def print_comparison(rows):
    hdr = f"{'r':>5} {'T2(K)':>8} {'P2(bar)':>8} {'T3(K)':>8} {'P3(bar)':>8} {'T4(K)':>8} {'P4(bar)':>8} " \
          f"{'q_in':>8} {'w_net':>8} {'eta(%)':>7} {'MEP(bar)':>9}"
    print(hdr)
    for x in rows:
        print(f"{x.p.r:5g} {x.T[1]:8.1f} {x.P[1] / 100:8.2f} {x.T[2]:8.1f} {x.P[2] / 100:8.2f} {x.T[3]:8.1f} "
              f"{x.P[3] / 100:8.3f} {x.q_in:8.1f} {x.w_net:8.1f} {x.eta * 100:7.2f} {x.mep / 100:9.3f}")


# =====================================================================================
#  4.  GUI
# =====================================================================================

MODE_CFG = {
    "q": ("Heat addition q_in", 50.0, 2500.0, 5.0, "{:.1f}", "kJ/kg"),
    "T4": ("T4 (end expansion)", 250.0, 1900.0, 5.0, "{:.1f}", "K"),
    "T3": ("T3 (peak temp)", 600.0, 3500.0, 10.0, "{:.1f}", "K"),
}


class StandardParamRow:
    """Straightforward input field with optional text label and unit."""

    def __init__(self, parent, row, label, lo, hi, step, init, fmt, on_change, unit=""):
        self.on_change = on_change
        self.lo, self.hi, self.fmt = lo, hi, fmt
        self._value = float(init)

        lbl = ttk.Label(parent, text=label, font=("TkDefaultFont", 8))
        lbl.grid(row=row, column=0, sticky="w", pady=2)

        self.var = tk.StringVar(value=fmt.format(init))
        self.ent = ttk.Entry(parent, textvariable=self.var, width=7, justify="right", font=("TkDefaultFont", 8))
        self.ent.grid(row=row, column=1, sticky="e", padx=2, pady=2)

        unit_lbl = ttk.Label(parent, text=unit, font=("TkDefaultFont", 8))
        unit_lbl.grid(row=row, column=2, sticky="w", pady=2)

        self.ent.bind("<Return>", self._on_entry)
        self.ent.bind("<FocusOut>", self._on_entry)

    def configure(self, label, lo, hi, step, init, fmt, unit=""):
        self.lo, self.hi, self.fmt = lo, hi, fmt
        self.set(init, notify=False)

    def get(self):
        return self._value

    def set(self, v, notify=True):
        self._value = float(v)
        self.var.set(self.fmt.format(self._value))
        if notify:
            self.on_change()

    def _on_entry(self, _e=None):
        try:
            v = float(self.var.get())
            v = min(max(v, self.lo), self.hi)
        except ValueError:
            self.var.set(self.fmt.format(self._value))
            return
        if abs(v - self._value) > 1e-12:
            self.set(v)
        else:
            self.var.set(self.fmt.format(self._value))


class ScrollableFrame(ttk.Frame):
    """Clean vertical scrollable frame container."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class App:
    TRAIL = 20

    def __init__(self, root):
        self.root = root
        root.title("Otto Cycle Simulator - Assignment II")

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = min(1200, int(sw * 0.82))
        h = min(720, int(sh * 0.82))
        root.geometry(f"{w}x{h}")
        root.minsize(800, 500)

        self.res = None
        self.dirty = set(TAB_NAMES)
        self._pending = None
        self._bg = None
        self._loading = False
        self.idx = 0.0
        self._last = time.perf_counter()

        self.mode = tk.StringVar(value="T4")
        self.cv_var = tk.BooleanVar(value=True)
        self.playing = tk.BooleanVar(value=True)
        self.opt_shade = tk.BooleanVar(value=True)
        self.opt_log = tk.BooleanVar(value=False)
        self.opt_abs = tk.BooleanVar(value=False)
        self.hold_var = tk.StringVar(value=list(HOLD_LABELS)[0])
        self.status = tk.StringVar(value="")

        self._build_ui()
        self.load_params(ASSIGNMENT)
        self.root.after(30, self._tick)

    def _build_ui(self):
        root = self.root
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # Left Column: Sidebar with inputs
        left_wrapper = ScrollableFrame(root)
        left_wrapper.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        left = left_wrapper.scrollable_frame

        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.status_lbl = ttk.Label(root, textvariable=self.status, anchor="w", padding=(4, 1),
                                    font=("TkDefaultFont", 8))
        self.status_lbl.grid(row=1, column=0, columnspan=2, sticky="ew")

        # ---- LEFT PANEL ----

        # Scenario Frame (Contains Assignment and Restored Sandbox Mode)
        f_scn = ttk.LabelFrame(left, text="Presets", padding=6)
        f_scn.pack(fill="x", pady=2)
        ttk.Button(f_scn, text="Set E Assignment Case", command=lambda: self.load_params(ASSIGNMENT)).pack(fill="x",
                                                                                                           pady=1)
        ttk.Button(f_scn, text="Sandbox (Free Exploration)",
                   command=lambda: self.load_params(Params(r=9.0, mode="q", value=1000.0))).pack(fill="x", pady=1)
        self.scn_var = tk.StringVar()
        ttk.Label(f_scn, textvariable=self.scn_var, foreground="#0a6b2b", font=("TkDefaultFont", 8, "bold")).pack(
            anchor="w", pady=(2, 0))

        # Mode Selection
        f_mode = ttk.LabelFrame(left, text="Heat Input Mode", padding=6)
        f_mode.pack(fill="x", pady=2)
        for key, txt in (("T4", "T4 (end expansion)"), ("q", "q_in (heat added)"), ("T3", "T3 (peak temp)")):
            ttk.Radiobutton(f_mode, text=txt, value=key, variable=self.mode, command=self._mode_changed).pack(
                anchor="w")

        # Parameters Frame
        f_params = ttk.LabelFrame(left, text="Parameters", padding=6)
        f_params.pack(fill="x", pady=2)
        ch = self.refresh

        self.r_row = StandardParamRow(f_params, 0, "Compression ratio r", 2, 20, 0.1, 8, "{:.1f}", ch)
        self.val_row = StandardParamRow(f_params, 1, "T4", 250, 1900, 5, 800, "{:.1f}", ch, "K")
        self.P1_row = StandardParamRow(f_params, 2, "Inlet Pressure P1", 0.3, 3.0, 0.05, 1.0, "{:.2f}", ch, "bar")
        self.T1_row = StandardParamRow(f_params, 3, "Inlet Temp T1", 250, 450, 1, 300, "{:.0f}", ch, "K")
        self.g_row = StandardParamRow(f_params, 4, "Specific heat ratio γ", 1.2, 1.67, 0.01, 1.4, "{:.2f}", ch)
        self.R_row = StandardParamRow(f_params, 5, "Gas Constant R", 0.2, 0.5, 0.001, 0.287, "{:.3f}", ch, "kJ/kgK")
        ttk.Checkbutton(f_params, text="Textbook cv = 0.718", variable=self.cv_var, command=self.refresh).grid(row=6,
                                                                                                               column=0,
                                                                                                               columnspan=3,
                                                                                                               sticky="w",
                                                                                                               pady=2)

        # Key Results
        f_res = ttk.LabelFrame(left, text="Results", padding=6)
        f_res.pack(fill="x", pady=2)
        self.kv = {}
        items = [("T3 (max T)", "K"), ("P3 (max P)", "bar"), ("η (Efficiency)", "%"), ("MEP", "bar"),
                 ("w_net", "kJ/kg"), ("q_in", "kJ/kg"), ("q_out", "kJ/kg")]
        for i, (name, unit) in enumerate(items):
            ttk.Label(f_res, text=name, font=("TkDefaultFont", 8)).grid(row=i, column=0, sticky="w")
            self.kv[name] = tk.StringVar()
            font = ("TkDefaultFont", 8, "bold") if i < 4 else ("TkDefaultFont", 8)
            ttk.Label(f_res, textvariable=self.kv[name], font=font).grid(row=i, column=1, sticky="e", padx=(8, 0))
        f_res.columnconfigure(1, weight=1)

        # Options
        f_eng = ttk.LabelFrame(left, text="Options", padding=6)
        f_eng.pack(fill="x", pady=2)
        self.disp_row = StandardParamRow(f_eng, 0, "Displacement", 50, 2000, 10, 500, "{:.0f}", self.refresh, "cc")
        self.rpm_row = StandardParamRow(f_eng, 1, "Speed", 500, 12000, 100, 3000, "{:.0f}", self.refresh, "rpm")
        ttk.Checkbutton(f_eng, text="Shade net work & MEP", variable=self.opt_shade, command=self._view_changed).grid(
            row=2, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(f_eng, text="Log-log P-v axes", variable=self.opt_log, command=self._view_changed).grid(row=3,
                                                                                                                column=0,
                                                                                                                columnspan=3,
                                                                                                                sticky="w")
        ttk.Checkbutton(f_eng, text="Abs volume (cm³)", variable=self.opt_abs, command=self._view_changed).grid(row=4,
                                                                                                                column=0,
                                                                                                                columnspan=3,
                                                                                                                sticky="w")

        # Export Actions
        f_btn = ttk.Frame(left)
        f_btn.pack(fill="x", pady=4)
        ttk.Button(f_btn, text="Export CSV", command=self.export_csv).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(f_btn, text="Save Plot", command=self.save_plot).pack(side="left", expand=True, fill="x", padx=1)

        # ---- RIGHT PANEL ----
        self.nb = ttk.Notebook(right)
        self.nb.grid(row=0, column=0, sticky="nsew")
        self.tab_cycle, self.tab_results = ttk.Frame(self.nb), ttk.Frame(self.nb)
        self.tab_compare, self.tab_sweep = ttk.Frame(self.nb), ttk.Frame(self.nb)

        self.nb.add(self.tab_cycle, text=" Live Cycle ")
        self.nb.add(self.tab_results, text=" Solution & Working ")
        self.nb.add(self.tab_compare, text=" Compare Ratios ")
        self.nb.add(self.tab_sweep, text=" Parameter Sweep ")
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._render_current())

        self._build_cycle_tab()
        self._build_results_tab()
        self._build_compare_tab()
        self._build_sweep_tab()

    def _build_cycle_tab(self):
        t = self.tab_cycle
        self.fig_cycle = Figure(figsize=(7, 4.5), dpi=95, layout="constrained")
        gs = GridSpec(2, 2, figure=self.fig_cycle, width_ratios=[1.2, 1])
        self.ax_pv = self.fig_cycle.add_subplot(gs[:, 0])
        self.ax_cyl = self.fig_cycle.add_subplot(gs[0, 1])
        self.ax_ts = self.fig_cycle.add_subplot(gs[1, 1])

        self.canvas_cycle = FigureCanvasTkAgg(self.fig_cycle, master=t)

        bar = ttk.Frame(t, padding=2)
        bar.pack(side="bottom", fill="x")
        self.play_btn = ttk.Button(bar, text="Pause", width=6, command=self._toggle_play)
        self.play_btn.pack(side="left", padx=2)

        ttk.Label(bar, text="Speed", font=("TkDefaultFont", 8)).pack(side="left")
        self.speed = tk.DoubleVar(value=0.35)
        ttk.Scale(bar, from_=0.05, to=3.0, variable=self.speed, length=80).pack(side="left", padx=2)

        ttk.Label(bar, text="Scrub", font=("TkDefaultFont", 8)).pack(side="left")
        self.scrub = ttk.Scale(bar, from_=0, to=1, length=120, command=self._on_scrub)
        self.scrub.pack(side="left", padx=2)

        self.readout = tk.StringVar()
        ttk.Label(bar, textvariable=self.readout, font=("TkFixedFont", 8)).pack(side="right", padx=4)

        tb = NavigationToolbar2Tk(self.canvas_cycle, t, pack_toolbar=False)
        tb.pack(side="bottom", fill="x")

        self.canvas_cycle.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas_cycle.mpl_connect("draw_event", self._on_draw)

    def _build_results_tab(self):
        t = self.tab_results
        cols = ("state", "process", "P", "T", "v", "s")
        self.tree = ttk.Treeview(t, columns=cols, show="headings", height=4)
        for c, h, w in (("state", "State", 45), ("process", "Process", 200), ("P", "P (bar)", 80),
                        ("T", "T (K)", 80), ("v", "v (m³/kg)", 90), ("s", "s-s1", 90)):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="x", padx=4, pady=4)

        self.txt = tk.Text(t, font=("TkFixedFont", 8), wrap="none")
        sb = ttk.Scrollbar(t, command=self.txt.yview)
        self.txt.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 4))

    def _hold_combo(self, parent):
        cb = ttk.Combobox(parent, textvariable=self.hold_var, values=list(HOLD_LABELS), state="readonly", width=30,
                          font=("TkDefaultFont", 8))
        cb.bind("<<ComboboxSelected>>", lambda e: self._mark_dirty("compare", "sweep"))
        return cb

    def _build_compare_tab(self):
        t = self.tab_compare
        bar = ttk.Frame(t, padding=4)
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="r values:", font=("TkDefaultFont", 8)).pack(side="left")
        self.ratios_var = tk.StringVar(value="8, 10, 12")
        e = ttk.Entry(bar, textvariable=self.ratios_var, width=12, font=("TkDefaultFont", 8))
        e.pack(side="left", padx=2)
        e.bind("<Return>", lambda ev: self._mark_dirty("compare"))
        ttk.Button(bar, text="Plot", command=lambda: self._mark_dirty("compare")).pack(side="left", padx=2)
        ttk.Label(bar, text=" Hold:", font=("TkDefaultFont", 8)).pack(side="left")
        self._hold_combo(bar).pack(side="left", padx=2)

        cols = ("r", "T2", "P2", "T3", "P3", "T4", "P4", "q", "w", "eta", "mep")
        heads = ("r", "T2(K)", "P2(b)", "T3(K)", "P3(b)", "T4(K)", "P4(b)", "q_in", "w_net", "η(%)", "MEP(b)")
        self.cmp_tree = ttk.Treeview(t, columns=cols, show="headings", height=4)
        for c, h in zip(cols, heads):
            self.cmp_tree.heading(c, text=h)
            self.cmp_tree.column(c, width=65, anchor="center")
        self.cmp_tree.pack(side="bottom", fill="x", padx=4, pady=4)

        self.fig_cmp = Figure(figsize=(7, 4.5), dpi=95, layout="constrained")
        self.canvas_cmp = FigureCanvasTkAgg(self.fig_cmp, master=t)
        tb = NavigationToolbar2Tk(self.canvas_cmp, t, pack_toolbar=False)
        tb.pack(side="bottom", fill="x")
        self.canvas_cmp.get_tk_widget().pack(side="top", fill="both", expand=True)

    def _build_sweep_tab(self):
        t = self.tab_sweep
        bar = ttk.Frame(t, padding=4)
        bar.pack(side="top", fill="x")
        self.sx_var = tk.StringVar(value=list(SWEEP_X)[0])
        self.sy_var = tk.StringVar(value=list(SWEEP_Y)[0])
        self.smin, self.smax = tk.StringVar(value="2"), tk.StringVar(value="20")

        ttk.Label(bar, text="X:", font=("TkDefaultFont", 8)).pack(side="left")
        cx = ttk.Combobox(bar, textvariable=self.sx_var, values=list(SWEEP_X), state="readonly", width=20,
                          font=("TkDefaultFont", 8))
        cx.pack(side="left", padx=2)
        cx.bind("<<ComboboxSelected>>", self._sweep_x_changed)

        ttk.Label(bar, text="Range:", font=("TkDefaultFont", 8)).pack(side="left")
        ttk.Entry(bar, textvariable=self.smin, width=5, font=("TkDefaultFont", 8)).pack(side="left", padx=1)
        ttk.Label(bar, text="-", font=("TkDefaultFont", 8)).pack(side="left")
        ttk.Entry(bar, textvariable=self.smax, width=5, font=("TkDefaultFont", 8)).pack(side="left", padx=1)

        ttk.Label(bar, text=" Y:", font=("TkDefaultFont", 8)).pack(side="left")
        cy = ttk.Combobox(bar, textvariable=self.sy_var, values=list(SWEEP_Y), state="readonly", width=22,
                          font=("TkDefaultFont", 8))
        cy.pack(side="left", padx=2)
        cy.bind("<<ComboboxSelected>>", lambda e: self._mark_dirty("sweep"))

        ttk.Button(bar, text="Plot", command=lambda: self._mark_dirty("sweep")).pack(side="left", padx=2)

        bar2 = ttk.Frame(t, padding=(4, 0))
        bar2.pack(side="top", fill="x")
        ttk.Label(bar2, text="Hold:", font=("TkDefaultFont", 8)).pack(side="left")
        self._hold_combo(bar2).pack(side="left", padx=2)

        self.fig_sw = Figure(figsize=(7, 4.5), dpi=95, layout="constrained")
        self.canvas_sw = FigureCanvasTkAgg(self.fig_sw, master=t)
        tb = NavigationToolbar2Tk(self.canvas_sw, t, pack_toolbar=False)
        tb.pack(side="bottom", fill="x")
        self.canvas_sw.get_tk_widget().pack(side="top", fill="both", expand=True)

    # ------------------------------------------------------------------ state handling
    def current_params(self) -> Params:
        return Params(r=self.r_row.get(), P1=self.P1_row.get() * 100.0, T1=self.T1_row.get(),
                      gamma=self.g_row.get(), R=self.R_row.get(), mode=self.mode.get(),
                      value=self.val_row.get(), textbook_cv=self.cv_var.get())

    def load_params(self, p: Params):
        self._loading = True
        self.mode.set(p.mode)
        lab, lo, hi, st, fmt, unit = MODE_CFG[p.mode]
        self.val_row.configure(lab, lo, hi, st, p.value, fmt, unit)
        self.r_row.set(p.r, notify=False)
        self.P1_row.set(p.P1 / 100.0, notify=False)
        self.T1_row.set(p.T1, notify=False)
        self.g_row.set(p.gamma, notify=False)
        self.R_row.set(p.R, notify=False)
        self.cv_var.set(p.textbook_cv)
        self._loading = False
        self.refresh()

    def _mode_changed(self):
        new = self.mode.get()
        val = self.val_row.get()
        if self.res is not None:
            val = {"q": self.res.q_in, "T3": self.res.T[2], "T4": self.res.T[3]}[new]
        lab, lo, hi, st, fmt, unit = MODE_CFG[new]
        self._loading = True
        self.val_row.configure(lab, lo, hi, st, round(val, 3), fmt, unit)
        self._loading = False
        self.refresh()

    def refresh(self):
        if self._loading:
            return
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(25, self._do_refresh)

    def _do_refresh(self):
        self._pending = None
        try:
            res = solve_otto(self.current_params())
        except ValueError as e:
            self.status.set("Warning: " + str(e))
            self.status_lbl.config(foreground="#b00020")
            return
        self.res = res
        self.status.set("Cycle solved successfully.")
        self.status_lbl.config(foreground="#0a6b2b")
        p = res.p
        is_assign = (abs(p.r - 8) < 1e-9 and abs(p.P1 - 100) < 1e-9 and abs(p.T1 - 300) < 1e-9
                     and p.mode == "T4" and abs(p.value - 800) < 1e-9 and abs(p.gamma - 1.4) < 1e-9
                     and abs(p.R - 0.287) < 1e-9)
        self.scn_var.set("✔ Assignment Case (Set E)" if is_assign else "Custom Scenario")
        self._update_key_results()
        self._update_results_tab()
        self._mark_dirty(*TAB_NAMES)

    def _mark_dirty(self, *names):
        self.dirty.update(names)
        self._render_current()

    def _render_current(self):
        if self.res is None:
            return
        name = TAB_NAMES[self.nb.index(self.nb.select())]
        if name in self.dirty:
            self.dirty.discard(name)
            getattr(self, "_draw_" + name)()
        elif name == "cycle":
            self.canvas_cycle.draw_idle()

    # ------------------------------------------------------------------ text/table updates
    def _update_key_results(self):
        r = self.res
        k = self.kv
        k["T3 (max T)"].set(f"{r.T[2]:.1f} K")
        k["P3 (max P)"].set(f"{r.P[2] / 100:.2f} bar")
        k["η (Efficiency)"].set(f"{r.eta * 100:.2f} %")
        k["MEP"].set(f"{r.mep / 100:.3f} bar")
        k["w_net"].set(f"{r.w_net:.1f} kJ/kg")
        k["q_in"].set(f"{r.q_in:.1f} kJ/kg")
        k["q_out"].set(f"{r.q_out:.1f} kJ/kg")

    def _update_results_tab(self):
        r = self.res
        self.tree.delete(*self.tree.get_children())
        procs = ["Start compression", "Isentropic comp 1→2", "Heat addition 2→3", "Isentropic exp 3→4"]
        for i in range(4):
            self.tree.insert("", "end", values=(i + 1, procs[i], f"{r.P[i] / 100:.3f}", f"{r.T[i]:.2f}",
                                                f"{r.v[i]:.4f}", f"{r.s[i]:.4f}"))
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", working_text(r))
        self.txt.config(state="disabled")

    # ------------------------------------------------------------------ drawing
    def _view_changed(self):
        self._mark_dirty("cycle")

    def _draw_cycle(self):
        res = self.res
        self._bg = None
        for ax in (self.ax_pv, self.ax_cyl, self.ax_ts):
            ax.clear()
        self.curves = cycle_curves(res)
        self.path = {k: np.concatenate([c[k] for c in self.curves]) for k in "vPTs"}
        self.seg = np.concatenate([np.full(len(c["v"]), i) for i, c in enumerate(self.curves)])
        self.N = len(self.seg)
        self.vscale = 1.0
        xlabel = None
        if self.opt_abs.get():
            m = engine_metrics(res, self.disp_row.get(), self.rpm_row.get())
            self.vscale = m["V1_cc"] / res.v[0]
            xlabel = "Volume V (cm³)"
        draw_pv(self.ax_pv, res, self.curves, self.vscale, self.opt_log.get(), self.opt_shade.get(), xlabel)
        draw_ts(self.ax_ts, res, self.curves, self.opt_shade.get())
        self._setup_cylinder()
        self.pv_trail, = self.ax_pv.plot([], [], "-", lw=3, color="k", alpha=0.55, animated=True)
        self.pv_dot, = self.ax_pv.plot([], [], "o", ms=8, mfc="gold", mec="k", mew=1, zorder=20, animated=True)
        self.ts_trail, = self.ax_ts.plot([], [], "-", lw=3, color="k", alpha=0.55, animated=True)
        self.ts_dot, = self.ax_ts.plot([], [], "o", ms=8, mfc="gold", mec="k", mew=1, zorder=20, animated=True)
        self.anim_artists = [self.pv_trail, self.pv_dot, self.ts_trail, self.ts_dot, self.gas, self.piston,
                             self.rod, self.spark, self.stroke_txt]
        self.canvas_cycle.draw_idle()

    def _setup_cylinder(self):
        ax = self.ax_cyl
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.3, 1.25)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        ax.set_title("Cylinder Animation", fontsize=10, fontweight="bold")
        kw = dict(color="k", lw=2)
        ax.plot([0.2, 0.2], [-0.05, 1.0], **kw)
        ax.plot([0.8, 0.8], [-0.05, 1.0], **kw)
        ax.plot([0.2, 0.8], [1.0, 1.0], **kw)
        ax.add_patch(Rectangle((0.47, 1.0), 0.06, 0.07, fc="#999", ec="k"))
        self.gas = Rectangle((0.2, 0.9), 0.6, 0.1, fc="#aaa", ec="none", animated=True)
        self.piston = Rectangle((0.2, 0.8), 0.6, 0.08, fc="#444", ec="k", animated=True)
        self.rod = Rectangle((0.47, -0.2), 0.06, 0.7, fc="#888", ec="k", animated=True)
        for a in (self.gas, self.rod, self.piston):
            ax.add_patch(a)
        self.spark, = ax.plot([0.5], [1.14], marker="*", ms=18, mfc="gold", mec="darkorange", ls="", animated=True)
        self.stroke_txt = ax.text(0.5, -0.22, "", ha="center", va="top", fontsize=8, fontweight="bold", animated=True)
        self.cmap = matplotlib.colormaps["coolwarm"]
        self.norm = Normalize(vmin=self.res.T[0], vmax=self.res.T[2])

    def _on_draw(self, _event):
        if not hasattr(self, "anim_artists"):
            return
        self._bg = self.canvas_cycle.copy_from_bbox(self.fig_cycle.bbox)
        self._apply_frame()
        for a in self.anim_artists:
            self.fig_cycle.draw_artist(a)

    def _apply_frame(self):
        res = self.res
        i = int(self.idx) % self.N
        v, P, T, s = (self.path[k][i] for k in "vPTs")
        sg = int(self.seg[i])
        ids = (i - np.arange(self.TRAIL, -1, -1)) % self.N
        x = self.path["v"][ids] * self.vscale
        self.pv_trail.set_data(x, self.path["P"][ids] / 100)
        self.pv_dot.set_data([v * self.vscale], [P / 100])
        self.ts_trail.set_data(self.path["s"][ids], self.path["T"][ids])
        self.ts_dot.set_data([s], [T])
        h = (v / res.v[0]) * 0.88
        self.gas.set_y(1.0 - h)
        self.gas.set_height(h)
        self.gas.set_facecolor(self.cmap(self.norm(T)))
        self.piston.set_y(1.0 - h - 0.08)
        self.rod.set_height(1.0 - h - 0.08 + 0.2)
        self.spark.set_visible(sg == 1)
        self.stroke_txt.set_text(SEG_STROKE[sg])
        self.readout.set(f"P={P / 100:.2f}b  T={T:.0f}K  v={v:.4f}")

    def _tick(self):
        now = time.perf_counter()
        dt = min(now - self._last, 0.2)
        self._last = now
        if (self.playing.get() and self.res is not None and self._bg is not None
                and self.nb.index(self.nb.select()) == 0):
            self.idx = (self.idx + self.speed.get() * self.N * dt) % self.N
            self._blit_frame()
            self._set_scrub_silent(self.idx / self.N)
        self.root.after(30, self._tick)

    def _blit_frame(self):
        if self._bg is None:
            return
        self.canvas_cycle.restore_region(self._bg)
        self._apply_frame()
        for a in self.anim_artists:
            self.fig_cycle.draw_artist(a)
        self.canvas_cycle.blit(self.fig_cycle.bbox)

    def _set_scrub_silent(self, frac):
        self._scrub_busy = True
        self.scrub.set(frac)
        self._scrub_busy = False

    def _on_scrub(self, val):
        if getattr(self, "_scrub_busy", False) or self.res is None:
            return
        self.idx = float(val) * (self.N - 1)
        if self.playing.get():
            self._toggle_play()
        self._blit_frame()

    def _toggle_play(self):
        self.playing.set(not self.playing.get())
        self.play_btn.config(text="Pause" if self.playing.get() else "Play")
        self._last = time.perf_counter()

    def _draw_results(self):
        pass

    def _ratios(self):
        out = []
        for tok in self.ratios_var.get().replace(";", ",").replace(" ", ",").split(","):
            if tok.strip():
                try:
                    r = float(tok)
                    if r > 1:
                        out.append(r)
                except ValueError:
                    pass
        return out[:6] or [self.res.p.r]

    def _draw_compare(self):
        hold = HOLD_LABELS[self.hold_var.get()]
        rows = build_compare_figure(self.fig_cmp, self.res, self._ratios(), hold)
        self.canvas_cmp.draw_idle()
        self.cmp_tree.delete(*self.cmp_tree.get_children())
        for x in rows:
            self.cmp_tree.insert("", "end",
                                 values=(f"{x.p.r:g}", f"{x.T[1]:.1f}", f"{x.P[1] / 100:.2f}", f"{x.T[2]:.1f}",
                                         f"{x.P[2] / 100:.2f}", f"{x.T[3]:.1f}", f"{x.P[3] / 100:.3f}",
                                         f"{x.q_in:.1f}", f"{x.w_net:.1f}", f"{x.eta * 100:.2f}",
                                         f"{x.mep / 100:.3f}"))

    def _sweep_x_changed(self, _e=None):
        _, lo, hi = SWEEP_X[self.sx_var.get()]
        self.smin.set(f"{lo:g}")
        self.smax.set(f"{hi:g}")
        self._mark_dirty("sweep")

    def _draw_sweep(self):
        try:
            lo, hi = float(self.smin.get()), float(self.smax.get())
            if hi <= lo:
                raise ValueError
        except ValueError:
            self.status.set("Warning: min < max required for sweep.")
            self.status_lbl.config(foreground="#b00020")
            return
        key = SWEEP_X[self.sx_var.get()][0]
        build_sweep_figure(self.fig_sw, self.res, HOLD_LABELS[self.hold_var.get()], self.sx_var.get(), key, lo, hi,
                           self.sy_var.get(), SWEEP_Y[self.sy_var.get()])
        self.canvas_sw.draw_idle()

    # ------------------------------------------------------------------ export
    def export_csv(self):
        if self.res is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="otto_results.csv")
        if not path:
            return
        r = self.res
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Otto cycle results"])
            w.writerow(
                ["r", r.p.r, "P1 (bar)", r.p.P1 / 100, "T1 (K)", r.p.T1, "gamma", r.p.gamma, "R", r.p.R, "cv", r.cv])
            w.writerow([])
            w.writerow(["State", "P (bar)", "T (K)", "v (m3/kg)", "s-s1 (kJ/kgK)"])
            for i in range(4):
                w.writerow([i + 1, f"{r.P[i] / 100:.4f}", f"{r.T[i]:.3f}", f"{r.v[i]:.5f}", f"{r.s[i]:.5f}"])
            w.writerow([])
            for name, val in (("q_in (kJ/kg)", r.q_in), ("q_out (kJ/kg)", r.q_out), ("w_comp (kJ/kg)", r.w_comp),
                              ("w_exp (kJ/kg)", r.w_exp), ("w_net (kJ/kg)", r.w_net), ("efficiency", r.eta),
                              ("MEP (bar)", r.mep / 100)):
                w.writerow([name, f"{val:.5f}"])
            w.writerow([])
            w.writerow(["Comparison", HOLD_NAME[HOLD_LABELS[self.hold_var.get()]] + " held constant"])
            w.writerow(
                ["r", "T2", "P2 (bar)", "T3", "P3 (bar)", "T4", "P4 (bar)", "q_in", "w_net", "eta (%)", "MEP (bar)"])
            for x in comparison_rows(r, self._ratios(), HOLD_LABELS[self.hold_var.get()]):
                w.writerow([x.p.r, f"{x.T[1]:.2f}", f"{x.P[1] / 100:.3f}", f"{x.T[2]:.2f}", f"{x.P[2] / 100:.3f}",
                            f"{x.T[3]:.2f}", f"{x.P[3] / 100:.3f}", f"{x.q_in:.2f}", f"{x.w_net:.2f}",
                            f"{x.eta * 100:.3f}", f"{x.mep / 100:.4f}"])
        self.status.set(f"Saved {path}")

    def save_plot(self):
        fig = {0: self.fig_cycle, 2: self.fig_cmp, 3: self.fig_sw}.get(self.nb.index(self.nb.select()))
        if fig is None:
            messagebox.showinfo("Save plot", "Switch to a plot tab first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            fig.savefig(path, dpi=200)
            self.status.set(f"Saved {path}")


TAB_NAMES = ["cycle", "results", "compare", "sweep"]


# =====================================================================================
#  5.  COMMAND-LINE MODES
# =====================================================================================

def cli_solve():
    res = solve_otto(ASSIGNMENT)
    print(working_text(res))
    print("\nEffect of compression ratio:\n")
    print_comparison(comparison_rows(res, [8, 10, 12], "q"))


def cli_report(outdir="."):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    os.makedirs(outdir, exist_ok=True)
    res = solve_otto(ASSIGNMENT)
    cur = cycle_curves(res)
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    draw_pv(ax1, res, cur)
    draw_ts(ax2, res, cur)
    fig.suptitle("Air-Standard Otto Cycle (r = 8, P1 = 1 bar, T1 = 300 K, T4 = 800 K)")
    p1 = os.path.join(outdir, "otto_cycle_r8.png")
    fig.savefig(p1, dpi=200)

    fig = Figure(figsize=(11, 6), layout="constrained")
    FigureCanvasAgg(fig)
    build_compare_figure(fig, res, [8, 10, 12], "q")
    p2 = os.path.join(outdir, "otto_compare_r8_10_12.png")
    fig.savefig(p2, dpi=200)
    print("Saved:", p1)
    print("Saved:", p2)


def main():
    ap = argparse.ArgumentParser(description="Otto Cycle Simulator")
    ap.add_argument("--solve", action="store_true", help="print worked solution")
    ap.add_argument("--report", action="store_true", help="save PNG plots")
    ap.add_argument("--outdir", default=".", help="output folder")
    a = ap.parse_args()
    if a.solve:
        cli_solve()
    elif a.report:
        cli_report(a.outdir)
    else:
        if not TK_OK:
            sys.exit("tkinter unavailable. Install python3-tk or run with --solve / --report.")
        root = tk.Tk()
        App(root)
        root.mainloop()


main()