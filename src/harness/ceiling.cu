#include "bench.cuh"
#include "utils.cuh"

// Same method as the blackwell-gemm ceilings program: a 1 GiB device-to-device
// copy, counting both the read and the write.
double measure_bandwidth_ceiling() {
    const size_t bytes = 1ULL << 30;
    void *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemset(d_src, 0, bytes));

    const float copy_ms = time_kernels([&]() {
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, 0));
    }, 3, 20);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    return 2.0 * bytes / (copy_ms / 1e3) / 1e9;
}
