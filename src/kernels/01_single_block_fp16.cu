#include "kernels.cuh"
#include "utils.cuh"
#include <cuda_fp16.h>


namespace{
    
    __global__ void decode_01(const half* q, const half* K, const half* V, half* o, size_t num_heads, size_t seq_len){
    
    }
} // namespace

void decode_01_single_block_fp16(const half* q, const half* K, const half* V, half* o, int num_heads, int seq_len, cudaStream_t stream){

    dim3 grid(num_heads);
    dim3 block(128);
    decode_01<<<grid, block, 0, stream>>>(q, K, V, o, num_heads, seq_len);
    CUDA_CHECK(cudaGetLastError());

}