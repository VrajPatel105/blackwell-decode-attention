#include "kernels.cuh"
#include "utils.cuh"
#include <cuda_fp16.h>

void decode_01_single_block_fp16(const half* q, const half* K, const half* V,
                                 half* o, int num_heads, int seq_len,
                                 cudaStream_t stream);