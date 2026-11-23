Constants: H = 32 heads, D = 128 head_dim. Both are fixed at compile time. L (cached sequence length) is a runtime parameter and can be any value from 1 up.
Scope: one sequence, one new query token, and no GQA (K/V heads equal query heads). There is no mask, because the single query attends to all L cached tokens.
Inputs (all fp16, row-major, contiguous, D fastest):
q: [H, D]
K: [H, L, D]
V: [H, L, D]
Output: o, fp16, [H, D].
Math:o[h] = softmax(q[h] x K[h]^T / root(D)) x V[h]. Accumulate in fp32 internally and convert to fp16 only when writing o.