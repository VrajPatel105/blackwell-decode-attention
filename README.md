# blackwell-decode-attention

A CUDA C++ decode-attention kernel: one query token attending over a cached K/V.
Built around an INT8 paged KV layout (per-block scales, dequantized in-kernel).
The layout comes from my paged inference engine project; this repo has no
dependency on it and runs on synthetic tensors in that layout because I wanted to keep the projects seperated.