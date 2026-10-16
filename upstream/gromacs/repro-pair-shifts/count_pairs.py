import torch
lines = open("conf.gro").read().splitlines()
n = int(lines[1]); box = [float(x) for x in lines[2+n].split()[:3]]
pos = torch.tensor([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in lines[2:2+n]], dtype=torch.float64)
box = torch.tensor(box, dtype=torch.float64)
d = pos[:, None, :] - pos[None, :, :]
d -= torch.round(d / box) * box          # minimum image (orthorhombic box, cutoff < half box)
r = d.norm(dim=-1)
within = (r < 0.6) & ~torch.eye(n, dtype=torch.bool)
print("atoms", n, "box", box.tolist(), "ordered pairs i!=j within 0.6 nm:", int(within.sum()), "unordered:", int(within.sum())//2)
