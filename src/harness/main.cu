// Load the fixture (this is in io.cu)
// Allocate GPU buffers
// Copy q, K, V to the GPU
// Poison the output
// Launch the kernel
// Copy the output back
// Compare
// Free


#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <io.cuh>
#include <kernels.cuh>
#include <utils.cuh>
#include <compare.cuh>
using namespace std;

int main(){

    vector<string> folder_vec = {"len_1", "len_100", "len_1024", "len_4097", "len_8192"};

    for (const auto& folder_name : folder_vec) {

        Fixture f = load_fixture("fixtures/" + folder_name);
        
        half *d_q, *d_K, *d_V, *d_o;
        vector<half> h_out(f.expected.size()); 

        CUDA_CHECK(cudaMalloc((void**)&d_q, f.q.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_K, f.K.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_V, f.V.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_o, f.expected.size() * sizeof(half)));

        CUDA_CHECK(cudaMemcpy(d_q, f.q.data(), f.q.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, f.K.data(), f.K.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, f.V.data(), f.V.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_o, 0xFF, f.expected.size() * sizeof(half)));

        decode_01_single_block_fp16(d_q, d_K, d_V, d_o, f.H, f.L, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_o, f.expected.size() * sizeof(half), cudaMemcpyDeviceToHost));

        CheckResult r = compare(h_out, f.expected);
        cout << folder_name << ": " << (r.pass ? "PASS" : "FAIL") << "  max_err=" << r.max_abs_err << "  nan=" << r.num_nan << endl;

        cudaFree(d_q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_o);
    }   

}