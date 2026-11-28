#include "bench.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace std;

static float find_median(vector<float> v) {
    sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 != 0) return v[n / 2];
    return (v[(n - 1) / 2] + v[n / 2]) / 2.0f;
}

float time_kernels(const function<void()>& launch, const int warmup, const int iterations,
                   const function<void()>& before_each) {

    if (iterations < 1 || warmup < 0) {
        fprintf(stderr, "time_kernels: invalid arguments (warmup=%d, iterations=%d) at %s:%d\n",
                warmup, iterations, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < warmup; i++) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> ms_per_launch;
    ms_per_launch.reserve(iterations);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iterations; i++) {

        if (before_each) before_each();   // not timed: it runs before the start event

        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms_per_launch.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return find_median(ms_per_launch);
}
