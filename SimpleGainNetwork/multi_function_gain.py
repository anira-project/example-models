"""Export a single .pte carrying three named methods (forward / gain2 / gain4).

Exercises anira's model_function support for the ExecuTorch backend: the same
program is loaded three times with different method names, and the output gain
proves which graph ran.  Shapes: [1, 1, 64] float32 in and out.
"""

import os
import torch
from executorch.exir import to_edge


class Gain(torch.nn.Module):
    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


def main() -> None:
    x = (torch.zeros(1, 1, 64),)
    methods = {
        "forward": torch.export.export(Gain(1.0), x),
        "gain2":   torch.export.export(Gain(2.0), x),
        "gain4":   torch.export.export(Gain(4.0), x),
    }
    program = to_edge(methods).to_executorch()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                       "simple_gain_network_multifunction.pte")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(program.buffer)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
