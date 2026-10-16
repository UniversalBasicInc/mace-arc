# Same idea with atom-pairs present: energy = number of pairs + 1000 * number of shift rows, so one number shows both.
import torch
class PairShiftCounter(torch.nn.Module):
    def forward(self, positions: torch.Tensor, atom_numbers: torch.Tensor, atom_pairs: torch.Tensor, pair_shifts: torch.Tensor) -> torch.Tensor:
        return float(atom_pairs.shape[0]) + 1000.0 * float(pair_shifts.shape[0]) + 0.0 * positions.sum()
torch.jit.script(PairShiftCounter()).save("pair_shift_counter.pt")
