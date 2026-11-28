#include "kernels.cuh"
#include "utils.cuh"
#include <cuda_fp16.h>


namespace{

constexpr int NUM_THREADS = 256;
constexpr int ELE_PER_THREAD = 32; // total number of elements covered by one thread in that row
constexpr int d_k = 128; // we are taking this as a constnat for this entire project since the fixtures follow d_k = 128
    
    __global__ void decode_01(const half* q, const half* K, const half* V, half* o, size_t num_heads, size_t seq_len){
    
        // we will use full mha. attention = Softmax ((q . k.T ) / root(d_k)) . V
        // I will partition this into three different execution layers.
        // first one will be writing the q . k.T multiplication

        // q is [32, 128]
        // K is [32, 100, 128] // considering that the length is 100 for now to keep it simple
        // V is [32, 100, 128]
        // o will be [32, 128]

        // one block takes in one head and per block there are threads.
        // here we are using grid(num_heads) so that means we have 32 thread blocks and block(256) means we have 256 threads per block

        // part 1 . 

        size_t bx = blockIdx.x;
        size_t tx = threadIdx.x;


        // since each head is for each block, we can write the code as in that we are inside one block so the shapes are : 
        // q = [128], K = [100, 128], V = [100, 128]

        size_t rows_per_phase = NUM_THREADS / (d_k / ELE_PER_THREAD); // 256 / (128 / 32) = 256 / 4 = 64
        size_t phases = ceil_div(seq_len, rows_per_phase); // 100 / 64  = 1 phases (threads / elements per thread)
        for(int ph = 0; ph < phases; ph++){

            size_t q_offset = bx * d_k;
            size_t k_v_offset = bx * d_k * seq_len;

            const int row = (k_v_offset + ph * ELE_PER_THREAD * NUM_THREADS + tx * ELE_PER_THREAD) / d_k;

            


            // now we load the q values in the register themselves.


        }



    }
} // namespace

void decode_01_single_block_fp16(const half* q, const half* K, const half* V, half* o, int num_heads, int seq_len, cudaStream_t stream){

    dim3 grid(num_heads);
    dim3 block(NUM_THREADS);
    decode_01<<<grid, block, 0, stream>>>(q, K, V, o, num_heads, seq_len);
    CUDA_CHECK(cudaGetLastError());

}