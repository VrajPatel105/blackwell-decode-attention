#pragma once
#include <functional>

// Median time (ms) of one launch, measured with CUDA events around launch() only.
// before_each (optional) runs before the start event on every timed iteration and is
// not timed. Use it to flush L2 so a kernel that re-reads the same K/V is not
// measured out of cache.
float time_kernels(const std::function<void()>& launch, int warmup, int iterations,
                   const std::function<void()>& before_each = nullptr);

// Measured device-to-device copy bandwidth in GB/s (read + write counted).
// This is the ceiling for a memory-bound kernel on the GPU it runs on.
double measure_bandwidth_ceiling();
