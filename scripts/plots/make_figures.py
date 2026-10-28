"""Figures for the README from the measured data in results/.

    python scripts/plots/make_figures.py          # writes docs/img/*.png

Needs numpy, matplotlib and ase (for the Si geometry, rebuilt exactly as scripts/mace_ab.py built it).
"""
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AB = os.path.join(ROOT, "results", "ab")
IMG = os.path.join(ROOT, "docs", "img")
os.makedirs(IMG, exist_ok=True)

BG, FG, GRID = "#0f1117", "#e6e6e6", "#2a2f3a"
C = {"b70": "#2ec4ff", "a5000": "#76b900", "a5000cueq": "#b9e36a", "cpu-epyc": "#ff9f43"}
LABEL = {"b70": "Intel Arc Pro B70", "a5000": "NVIDIA RTX A5000", "a5000cueq": "RTX A5000 + cuEquivariance",
         "cpu-epyc": "EPYC 7742, 128 threads"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": GRID,
                     "grid.color": GRID, "font.size": 11, "axes.titleweight": "bold"})


def forces(tag, system, dtype):
    return np.load(os.path.join(AB, f"{tag}_{system}_{dtype}.npz"))["forces"]


# ---------------------------------------------------------------- 1. force parity
def fig_force_parity():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dtype, title in zip(axes, ["float32", "float64"], ["float32 (production MD dtype)", "float64"]):
        fa, fb = forces("a5000", "Si512", dtype), forces("b70", "Si512", dtype)
        d = (fb - fa).ravel()
        rel = np.sqrt((d ** 2).sum() / (fa ** 2).sum())
        ax.scatter(fa.ravel(), fb.ravel(), s=6, c=C["b70"], alpha=0.6, linewidths=0)
        lim = np.abs(fa).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1, zorder=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("force component, RTX A5000 (eV/Å)"); ax.set_ylabel("force component, Arc Pro B70 (eV/Å)")
        ax.set_title(title); ax.grid(True, lw=0.4)
        ins = ax.inset_axes([0.58, 0.08, 0.38, 0.3])
        scale = 1e-6 if dtype == "float32" else 1e-15
        ins.hist(d / scale, bins=40, color=C["b70"], alpha=0.9)
        ins.set_xlabel(f"B70 − A5000  (×{scale:g} eV/Å)", fontsize=8); ins.tick_params(labelsize=7)
        ins.set_yticks([]); ins.set_facecolor("#161a23")
        ax.text(0.04, 0.95, f"Si512, MACE-MP-0 small\n1536 force components\nrel. RMS difference {rel:.1e}",
                transform=ax.transAxes, va="top", fontsize=9.5, bbox=dict(fc="#161a23", ec=GRID))
    fig.suptitle("Same forces on an Intel Arc and an NVIDIA GPU, down to the dtype floor", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "force_parity.png"), dpi=160); plt.close(fig)


# ---------------------------------------------------------------- 2. speed and energy
def fig_speed_energy():
    meta = {t: json.load(open(os.path.join(AB, f"{t}_meta.json")))["runs"] for t in ["cpu-epyc", "a5000", "a5000cueq", "b70"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.2, 1, 1.2]})
    ax = axes[0]; systems = ["Si64", "Si512", "Si1728"]; tags = ["cpu-epyc", "a5000", "a5000cueq", "b70"]
    w = 0.2; x = np.arange(len(systems))
    for i, t in enumerate(tags):
        v = [meta[t].get(f"{s}_float32", {}).get("median_ms", np.nan) for s in systems]
        ax.bar(x + (i - 1.5) * w, v, w, color=C[t], label=LABEL[t])
        for xi, vi in zip(x + (i - 1.5) * w, v):
            if np.isfinite(vi): ax.text(xi, vi * 1.08, f"{vi:.0f}", ha="center", fontsize=7.5)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{s} atoms" for s in [64, 512, 1728]])
    ax.set_ylabel("ms per force call (median of 10)"); ax.set_title("fp32 force call time"); ax.grid(True, axis="y", lw=0.4)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.2)
    ax = axes[1]
    v = [meta["cpu-epyc"]["Si512_float64"]["median_ms"], meta["a5000"]["Si512_float64"]["median_ms"], meta["b70"]["Si512_float64"]["median_ms"]]
    ax.bar(range(3), v, color=[C["cpu-epyc"], C["a5000"], C["b70"]])
    for i, vi in enumerate(v): ax.text(i, vi + 10, f"{vi:.0f} ms", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(["EPYC", "A5000", "B70"]); ax.set_ylim(0, max(v) * 1.2)
    ax.set_title("fp64, Si512"); ax.set_ylabel("ms per force call"); ax.grid(True, axis="y", lw=0.4)
    ax.text(2, v[2] + 60, "2.1× faster\nthan the A5000", ha="center", fontsize=9, color=C["b70"])
    ax = axes[2]
    j = [("A5000, e3nn", 9.1, C["a5000"]), ("A5000 + cuEq", 4.8, C["a5000cueq"]), ("B70, e3nn", 7.3, C["b70"])]
    ax.bar(range(3), [x[1] for x in j], color=[x[2] for x in j])
    for i, (n, vi, _) in enumerate(j): ax.text(i, vi + 0.2, f"{vi:.1f} J", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([x[0] for x in j], fontsize=8.5); ax.set_ylim(0, 11)
    ax.set_title("board energy per call, Si512"); ax.set_ylabel("J per call"); ax.grid(True, axis="y", lw=0.4)
    fig.suptitle("MACE-MP-0 small on diamond-Si supercells: one Arc Pro B70 against one RTX A5000 and a 128-thread EPYC", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "speed_energy.png"), dpi=160); plt.close(fig)


# ---------------------------------------------------------------- 3. GROMACS NVT traces
def read_xvg(path):
    rows = [list(map(float, l.split())) for l in open(path) if l and l[0] not in "#@"]
    return np.array(rows)


def fig_gromacs_nvt():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for d, t, lab in [("case-si512-b70", "b70", "Arc Pro B70 (NNPot on XPU), 2.93 ns/day"),
                      ("case-si512-a5000", "a5000", "RTX A5000 (NNPot on CUDA), 2.84 ns/day"),
                      ("case-si512-cpu", "cpu-epyc", "CPU (NNPot on EPYC)")]:
        p = os.path.join(ROOT, "results", "gmx", d, "temp.xvg")
        if not os.path.exists(p): continue
        a = read_xvg(p); ax.plot(a[:, 0], a[:, 1], color=C[t], lw=1.4, label=f"{lab}: {a[10:,1].mean():.1f} ± {a[10:,1].std():.1f} K")
    ax.axhline(300, color=GRID, lw=1); ax.set_xlabel("time (ps)"); ax.set_ylabel("temperature (K)")
    ax.set_title("GROMACS 2026 mdrun: Si512, MACE-MP-0 through NNPot, v-rescale 300 K, 1 fs", fontsize=11.5)
    ax.grid(True, lw=0.4); ax.legend(fontsize=9, framealpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "gromacs_nvt.png"), dpi=160); plt.close(fig)


# ---------------------------------------------------------------- 4. rendered Si512 cell
def render_atoms(ax, pos, cell, colors, radius=0.40, elev=12, azim=38, bond_cut=2.6):
    """Sphere-impostor rendering: depth-sorted discs with a highlight, in a matplotlib axes."""
    e, a = np.radians(elev), np.radians(azim)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    R = Rx @ Rz
    p = (pos - cell.diagonal() / 2) @ R.T
    order = np.argsort(p[:, 2])
    depth = (p[:, 2] - p[:, 2].min()) / np.ptp(p[:, 2])
    # cell edges
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * cell.diagonal() - cell.diagonal() / 2
    cp = corners @ R.T
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(corners[i] - corners[j]) > 1e-6) == 1:
                ax.plot(cp[[i, j], 0], cp[[i, j], 1], color="#3b4252", lw=0.8, zorder=0)
    # bonds (within the cell only; periodic images are not drawn)
    from scipy.spatial import cKDTree
    pairs = cKDTree(pos).query_pairs(bond_cut)
    for i, j in pairs:
        ax.plot(p[[i, j], 0], p[[i, j], 1], color="#55606f", lw=1.1, zorder=0.5, solid_capstyle="round")
    s = (radius * 72 / ax.figure.dpi) ** 2 * 2200
    for i in order:
        shade = 0.55 + 0.45 * depth[i]
        c = np.array(colors[i]) * shade
        ax.scatter(p[i, 0], p[i, 1], s=s, color=c, edgecolors="#000000", linewidths=0.4, zorder=1 + depth[i])
        ax.scatter(p[i, 0] - radius * 0.3, p[i, 1] + radius * 0.3, s=s * 0.12, color=np.minimum(c + 0.45, 1), linewidths=0, zorder=1 + depth[i] + 1e-3)
    ax.set_aspect("equal"); ax.axis("off")


def fig_si512_render():
    from ase.build import bulk
    s = bulk("Si", "diamond", a=5.43, cubic=True).repeat((4, 4, 4))
    s.rattle(stdev=0.05, seed=7)  # exactly as scripts/mace_ab.py builds the comparison geometry
    fb, fa = forces("b70", "Si512", "float32"), forces("a5000", "Si512", "float32")
    fmag = np.linalg.norm(fb, axis=1); dmag = np.linalg.norm(fb - fa, axis=1) / 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    cm1 = LinearSegmentedColormap.from_list("f", ["#1b2a49", "#2ec4ff", "#ffffff"])
    cm2 = LinearSegmentedColormap.from_list("d", ["#2b2b2b", "#ff9f43", "#ffffff"])
    for ax, val, cm, title, unit in [(axes[0], fmag, cm1, "force magnitude on each atom (Arc Pro B70)", "eV/Å"),
                                     (axes[1], dmag, cm2, "difference between the Arc and the A5000, per atom", "×10⁻⁶ eV/Å")]:
        n = (val - val.min()) / np.ptp(val)
        render_atoms(ax, s.positions, s.cell.array, cm(n)[:, :3])
        ax.set_title(title, fontsize=11)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(val.min(), val.max()))
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01); cb.set_label(unit); cb.outline.set_edgecolor(GRID)
    fig.suptitle("512 silicon atoms, rattled 0.05 Å: MACE-MP-0 forces on the Intel Arc, and how far they are from the NVIDIA card",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "si512_forces.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_force_parity(); fig_speed_energy(); fig_gromacs_nvt(); fig_si512_render()
    print("wrote", sorted(os.listdir(IMG)))
