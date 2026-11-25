#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using kernelFn = void (*)(const half* q, const half* K, const half* V, half* o, int num_heads, int seq_len, cudaStream_t s);

void decode_01_single_block_fp16 (const half*, const half*, const half*, half*,  int, int, cudaStream_t);

struct KernelEntry { int id; const char* name; kernelFn fn; };

inline const KernelEntry KERNELS[] = {
  {1, "01_single_block_fp16", decode_01_single_block_fp16}
};
inline constexpr int NUM_KERNELS = sizeof(KERNELS) / sizeof(KERNELS[0]);