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
#include <format>
using namespace std;

int main(){

    vector<string> folder_vec = {"len_1", "len_100", "len_1024", "len_4097", "len_8192"};

    for (const auto& folder_name : folder_vec) {

        Fixture f = load_fixture("fixtures/" + folder_name);

        half *d_q, *d_K, *d_V;
        float *d_expected;

        if(cudaMalloc((void**)&d_q, f.q.size() * sizeof(half)) != cudaSuccess){
            cout << "Unable to allocate memory for d_q" << endl;
        }
        if(cudaMalloc((void**)&d_K, f.K.size() * sizeof(half)) != cudaSuccess){
            cout << "Unable to allocate memory for d_K" << endl;
        }
        if(cudaMalloc((void**)&d_V, f.V.size() * sizeof(half)) != cudaSuccess){
            cout << "Unable to allocate memory for d_V" << endl;
        }
        if(cudaMalloc(&d_expected, f.expected.size() * sizeof(float)) != cudaSuccess){
            cout << "Unable to allocate memory for d_expected" << endl;
        }

        cudaMemcpy(d_q, f.q.data(), f.q.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, f.K.data(), f.K.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, f.V.data(), f.V.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_expected, f.expected.data(), f.expected.size() * sizeof(float), cudaMemcpyHostToDevice);

        


        cudaFree(d_q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_expected);
    }   

}