"""Render an extxyz trajectory from scripts/md_visual.py into an animated GIF and a poster frame.

    python scripts/plots/render_traj.py results/visual/water64_300K.extxyz docs/img/water64_300K.gif --title "..."
"""
import argparse, io, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from ase.io import read
from ase.data import covalent_radii
from scipy.spatial import cKDTree

BG = "#0f1117"
COLOR = {"Si": (0.62, 0.69, 0.82), "O": (0.93, 0.25, 0.20), "H": (0.92, 0.92, 0.92)}
RADIUS = {"Si": 0.42, "O": 0.34, "H": 0.20}


def project(pos, cell, elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    R = Rx @ Rz
    return (pos - cell.diagonal() / 2) @ R.T, R


def wrapped_positions(atoms):
    """Wrap into the cell; water molecules are wrapped as units (H follow their O) so no molecule is cut."""
    pos = atoms.positions.copy(); L = atoms.cell.array.diagonal(); sym = atoms.get_chemical_symbols()
    if "O" in sym and "H" in sym:
        o = [i for i, z in enumerate(sym) if z == "O"]; h = np.array([i for i, z in enumerate(sym) if z == "H"])
        for i in o:
            d = pos[h] - pos[i]; d -= np.round(d / L) * L
            mine = h[np.linalg.norm(d, axis=1) < 1.3]
            pos[i] -= np.floor(pos[i] / L) * L
            pos[mine] = pos[i] + d[np.linalg.norm(d, axis=1) < 1.3]
        return pos
    return pos - np.floor(pos / L) * L


def draw(ax, atoms, elev, azim, bonds=True, lim=None):
    pos = wrapped_positions(atoms); cell = atoms.cell.array; sym = atoms.get_chemical_symbols()
    p, R = project(pos, cell, elev, azim)
    depth = (p[:, 2] - p[:, 2].min()) / max(np.ptp(p[:, 2]), 1e-9)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * cell.diagonal() - cell.diagonal() / 2
    cp = corners @ R.T
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(corners[i] - corners[j]) > 1e-6) == 1:
                ax.plot(cp[[i, j], 0], cp[[i, j], 1], color="#3b4252", lw=0.8, zorder=0)
    if bonds:
        cut = {("Si", "Si"): 2.6, ("O", "H"): 1.25, ("H", "O"): 1.25}
        tree = cKDTree(pos)
        for i, j in tree.query_pairs(2.6):
            c = cut.get((sym[i], sym[j]))
            if c and np.linalg.norm(pos[i] - pos[j]) < c:
                ax.plot(p[[i, j], 0], p[[i, j], 1], color="#6b7280", lw=1.0, zorder=0.5, solid_capstyle="round")
    s0 = (72 / ax.figure.dpi) ** 2 * 2600
    for i in np.argsort(p[:, 2]):
        r = RADIUS[sym[i]]; shade = 0.55 + 0.45 * depth[i]
        c = np.array(COLOR[sym[i]]) * shade; s = s0 * r * r
        ax.scatter(p[i, 0], p[i, 1], s=s, color=c, edgecolors="#000000", linewidths=0.4, zorder=1 + depth[i])
        ax.scatter(p[i, 0] - r * 0.3, p[i, 1] + r * 0.3, s=s * 0.12, color=np.minimum(c + 0.45, 1), linewidths=0, zorder=1 + depth[i] + 1e-3)
    if lim is None:
        lim = np.abs(cp).max() * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    return lim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--title", default=""); ap.add_argument("--elev", type=float, default=12); ap.add_argument("--azim", type=float, default=38)
    ap.add_argument("--fs-per-frame", type=float, default=20, help="MD time between saved frames")
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--size", type=float, default=6.0)
    ap.add_argument("--spin", type=float, default=0.0, help="degrees of azimuth per frame")
    ap.add_argument("--poster", default=None, help="also write this PNG of one frame at higher dpi")
    ap.add_argument("--poster-frame", type=int, default=0, help="frame index for the poster (after striding)")
    a = ap.parse_args()
    frames = read(a.traj, ":")[:: a.stride]
    images = []; lim = None
    for k, atoms in enumerate(frames):
        fig, ax = plt.subplots(figsize=(a.size, a.size), facecolor=BG); ax.set_facecolor(BG)
        lim = draw(ax, atoms, a.elev, a.azim + a.spin * k, lim=lim)
        t_ps = k * a.stride * a.fs_per_frame / 1000
        ax.text(0.02, 0.97, a.title, transform=ax.transAxes, color="#e6e6e6", fontsize=10, va="top")
        ax.text(0.02, 0.03, f"t = {t_ps:5.2f} ps", transform=ax.transAxes, color="#9aa4b2", fontsize=10, family="monospace")
        buf = io.BytesIO(); fig.savefig(buf, dpi=100, facecolor=BG); plt.close(fig)
        buf.seek(0); images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
    images[0].save(a.out, save_all=True, append_images=images[1:], duration=60, loop=0, optimize=True)
    print("wrote", a.out, f"{len(images)} frames, {os.path.getsize(a.out)/1e6:.1f} MB")
    if a.poster:
        fig, ax = plt.subplots(figsize=(a.size * 1.3, a.size * 1.3), facecolor=BG); ax.set_facecolor(BG)
        draw(ax, frames[a.poster_frame], a.elev, a.azim + a.spin * a.poster_frame, lim=lim)
        ax.text(0.02, 0.97, a.title, transform=ax.transAxes, color="#e6e6e6", fontsize=11, va="top")
        fig.savefig(a.poster, dpi=150, facecolor=BG); plt.close(fig); print("wrote", a.poster)


if __name__ == "__main__":
    main()
