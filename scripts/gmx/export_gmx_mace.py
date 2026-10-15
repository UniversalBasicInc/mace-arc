# Export MACE-MP-0 (small) as a TorchScript model with the GROMACS NNPot calling convention:
#   forward(positions[nm], atomic_numbers, pairs (N_pairs,2 half list), shifts (N_pairs,3 nm), box, pbc) -> energy [kJ/mol]
# GROMACS computes forces as -dE/dpositions by autograd (torchmodel.cpp), so only energy is returned.
# Pair-list convention follows lmuellender/gmx-nnpot-tools models/gmx_mace.py (the NNPot authors):
# GROMACS hands a half list; MACE wants both directions, with shifts negated for the mirrored edges.
#   python export_gmx_mace.py --out mace_mp_small_gmx.pt [--device cpu]
import argparse, torch
from typing import Optional
from mace.calculators import mace_mp
from e3nn.util.jit import script

p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--device", default="cpu")
p.add_argument("--model", default="small"); a = p.parse_args()

class GmxMACE(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.model = raw.to(torch.float32)
        for q in self.model.parameters(): q.requires_grad = False
        zs = raw.atomic_numbers.tolist()
        zmap = torch.full((120,), -1, dtype=torch.long)
        for i, z in enumerate(zs): zmap[z] = i
        self.register_buffer("zmap", zmap)                     # Z -> one-hot column, no Python loop in TorchScript
        self.register_buffer("r_max", raw.r_max.clone())
        self.n_species = len(zs)
        self.length_conversion = 10.0                          # nm (GROMACS) -> A (MACE)
        self.energy_conversion = 96.48533212                   # eV (MACE) -> kJ/mol (GROMACS), CODATA 2018
    def forward(self, positions, atomic_numbers, pairs, shifts,
                cell: Optional[torch.Tensor] = None, pbc: Optional[torch.Tensor] = None):
        dev = positions.device
        pos = positions * self.length_conversion
        n = pos.shape[0]
        if cell is None: cell = torch.zeros(3, 3, device=dev, dtype=pos.dtype)
        else: cell = cell * self.length_conversion
        if pbc is None: pbc = torch.tensor([True, True, True], device=dev)
        edge_index = torch.cat([pairs, pairs[:, [1, 0]]], dim=0).t().to(torch.int64)
        shifts_A = torch.cat([-shifts, shifts], dim=0) * self.length_conversion
        idx = self.zmap[atomic_numbers.to(torch.int64)]
        node_attrs = torch.zeros(n, self.n_species, device=dev, dtype=pos.dtype)
        node_attrs[torch.arange(n, device=dev), idx] = 1.0
        data = {"ptr": torch.tensor([0, n], dtype=torch.int64, device=dev),
                "batch": torch.zeros(n, dtype=torch.int64, device=dev),
                "node_attrs": node_attrs, "positions": pos, "edge_index": edge_index,
                "shifts": shifts_A, "cell": cell, "pbc": pbc}
        out = self.model(data, training=False, compute_force=False, compute_virials=False,
                         compute_stress=False, compute_displacement=False)
        e = out["energy"]
        if e is None: e = torch.zeros(1, device=dev)
        return e * self.energy_conversion

raw = mace_mp(model=a.model, device=a.device, default_dtype="float32", return_raw_model=True)
m = GmxMACE(raw).to(a.device).eval()
print("r_max (A):", float(m.r_max), "-> pair-cutoff (nm):", float(m.r_max) / 10, "| species:", m.n_species)
scripted = script(m); scripted.save(a.out)
print("saved", a.out, "| torch", torch.__version__)
