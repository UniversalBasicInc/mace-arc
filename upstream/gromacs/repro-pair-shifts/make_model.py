# A minimal TorchScript model that only takes positions, atomic numbers and the periodic pair shifts,
# and reports how many shift rows it received as the "energy" (kJ/mol). Forces come from autograd and are zero.
import torch
class ShiftCounter(torch.nn.Module):
    def forward(self, positions: torch.Tensor, atom_numbers: torch.Tensor, pair_shifts: torch.Tensor) -> torch.Tensor:
        n_shifts = float(pair_shifts.shape[0])
        # keep a graph to positions so GROMACS can take the gradient for the forces
        return n_shifts + 0.0 * positions.sum() + 1e-6 * pair_shifts.abs().sum()
m = torch.jit.script(ShiftCounter())
m.save("shift_counter.pt")
print(m(torch.zeros(3,3,requires_grad=True), torch.zeros(3,dtype=torch.int32), torch.ones(7,3)))
