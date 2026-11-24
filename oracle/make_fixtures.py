from pathlib import Path

import torch

from oracle.oracle import torch_sdpa


BASE_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BASE_DIR / "fixtures"


def get_tensors():
    lengths = [1, 100, 1024, 4097, 8192]
    H, D = 32, 128

    for length in lengths:
        folder = FIXTURES_DIR / f"len_{length}"
        folder.mkdir(parents=True, exist_ok=True)

        q = torch.randn(H, D, dtype=torch.float16)
        K = torch.randn(H, length, D, dtype=torch.float16)
        V = torch.randn(H, length, D, dtype=torch.float16)

        expected = torch_sdpa(q, K, V)

        tensors = {
            "q.bin": q,
            "k.bin": K,
            "v.bin": V,
            "expected.bin": expected,
        }

        for filename, tensor in tensors.items():
            tensor.detach().cpu().numpy().tofile(folder / filename)


if __name__ == "__main__":
    get_tensors()