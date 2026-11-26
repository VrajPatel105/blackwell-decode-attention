#pragma once
#include <cuda_fp16.h>
#include <vector>

struct CheckResult {
    double max_abs_err;
    int    num_nan;
    bool   pass;
};

CheckResult compare(const std::vector<half>& out,
                    const std::vector<float>& ref,
                    double atol = 2e-3, double rtol = 2e-3);