#include "kernels.cuh"
#include "utils.cuh"
#include <cuda_fp16.h>
#include <cmath>


namespace{

#define FULL_MASK 0xffffffff
constexpr int NUM_THREADS = 256;
constexpr int ELE_PER_THREAD = 32; // total number of elements covered by one thread in that row
constexpr int THREADS_PER_WARP = 32;
constexpr int d_k = 128; // we are taking this as a constnat for this entire project since the fixtures follow d_k = 128

    __global__ void decode_01(const half* q, const half* K, const half* V, half* o, size_t num_heads, size_t seq_len){

        
        const float scale = rsqrtf((float)d_k); // for 1 / root d_k
        // define the shared mem
        extern __shared__ float scores[];
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
        
        size_t q_offset = bx * d_k;
        size_t k_v_offset = bx * d_k * seq_len;



        // load the q values into registeres since the q val does not change across phases since it's 1d [128];
        size_t col_start = tx * ELE_PER_THREAD % d_k;
        float q_value_register[ELE_PER_THREAD];

        for(int q_ele = 0; q_ele < ELE_PER_THREAD; q_ele++){
            q_value_register[q_ele] = __half2float(q[q_offset + col_start + q_ele]);
        }

        for(int ph = 0; ph < phases; ph++){

            const size_t row = ph * ELE_PER_THREAD * NUM_THREADS + tx * ELE_PER_THREAD;

            // the main accumulator loop for k and v
            size_t k_address = k_v_offset + row;

            float accu = 0.0f;
            for(size_t k = 0; k < ELE_PER_THREAD; k++){
                if(k_address + k < (k_v_offset + d_k * seq_len)){
                    accu += q_value_register[k] * __half2float(K[k_address + k]);
                }
            }

            for(int offset = 2; offset > 0; offset--){
                
                accu += __shfl_down_sync(FULL_MASK, accu, offset);
            
            }

            // now there are 256 threads per phase and per warp 32. each warp now has total of 8 score values (8 rows) cuz we had 4 as the warp shuffle reduction
            // so every thread at index 0, 4, 8, ... has the final acc for the row 0, 1, 2 ....
            size_t row_start_within_head = ph * rows_per_phase + tx / (d_k / ELE_PER_THREAD);

            if(tx % (d_k / ELE_PER_THREAD) == 0 && row_start_within_head < seq_len){
                scores[row_start_within_head] = accu * scale;
            }


        }

        __syncthreads();

        // now the softmax
        // we will follow four steps:
        // 1. get the max (will use one thread per element side by side to read from the memory and compare and finally use thread shuffling to get the max)
        // 2. calculate the numerator
        // 3. calculate the denominator 
        // final ouput

        // step 1 

        // the shared memory have the scores (our indexing will make sure to avoid shared mem bank conflicts which is why isntead of going with contigous memory per thread, we went with this one !)

        size_t total_iterations = ceil_div(seq_len, NUM_THREADS);
        float m = -INFINITY;
        for(size_t iteration = 0; iteration < total_iterations; iteration++){
            
            if( (iteration * NUM_THREADS + tx) < seq_len){
                float new_m = scores[ iteration * NUM_THREADS + tx];
                if(new_m > m){
                    m = new_m;
                }
            }
            
        }

        // now we will write the lane shuffle. now all those 256 threads have the local_max in there. 
        for(int offset = 16; offset > 0; offset /= 2){
            float new_m = __shfl_down_sync(FULL_MASK, m, offset);
            if(new_m > m){
                m = new_m;
            }
        }


        // now per warp. here there are 8 warps. each warps first thread has the max. since these are spreaded across different warps, we cant do the thread shuffling here. so we have another shared mem for that.
        // so every thread that is multiple of 8 
        size_t warp_max_smem_index = seq_len + tx / THREADS_PER_WARP;

        if((tx % THREADS_PER_WARP) == 0 && warp_max_smem_index < ( seq_len + THREADS_PER_WARP)){
            scores[warp_max_smem_index] = m;
        }

        __syncthreads();

        for(size_t idx = 0; idx < (NUM_THREADS / THREADS_PER_WARP); idx++){

            float new_m = scores[seq_len + idx];
            if(new_m > m){
                m = new_m;
            }
            
        }

        __syncthreads();


        // part 2 : the numerator = exponent(scores[p] - m)
        // part 3 as well. we can just combine part 2 and 3 together. 
        float running_sum = 0.0f; // denominator
        for(size_t iteration = 0; iteration < total_iterations; iteration++){

            if( (iteration * NUM_THREADS + tx ) < seq_len){
                float new_score = expf(scores[iteration * NUM_THREADS + tx] - m);
                scores[iteration * NUM_THREADS + tx ] = new_score;
                running_sum += new_score;
            }

        }

        // now each thread has running_sum but it's still partial. so we again apply the same thread shuffling.
        
        for(int offset = 16; offset > 0; offset /= 2){
            running_sum += __shfl_down_sync(FULL_MASK, running_sum, offset);
        }

        size_t running_sum_smem_index = seq_len + tx / THREADS_PER_WARP; // we will override the previous shared mem 

        if((tx % THREADS_PER_WARP) == 0 && running_sum_smem_index < ( seq_len + THREADS_PER_WARP)){
            scores[running_sum_smem_index] = running_sum;
        }

        __syncthreads();

        float final_denominator_scalar = 0;
        for(size_t idx = 0; idx < (NUM_THREADS / THREADS_PER_WARP); idx++){
            final_denominator_scalar += scores[seq_len + idx];
        }

        __syncthreads();


    }
} // namespace

void decode_01_single_block_fp16(const half* q, const half* K, const half* V, half* o, int num_heads, int seq_len, cudaStream_t stream){

    dim3 grid(num_heads);
    dim3 block(NUM_THREADS);
    size_t scores_smem_bytes = sizeof(float) * seq_len; // for scores shared mem
    size_t max_per_warp_bytes = sizeof(float) * (NUM_THREADS / THREADS_PER_WARP);
    size_t shmem_bytes = scores_smem_bytes + max_per_warp_bytes;
    decode_01<<<grid, block, shmem_bytes, stream>>>(q, K, V, o, num_heads, seq_len);
    CUDA_CHECK(cudaGetLastError());

}