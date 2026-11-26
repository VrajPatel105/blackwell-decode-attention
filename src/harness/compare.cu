// compare the two expected  tensors

#include "compare.cuh"
#include <cmath>
using namespace std;

CheckResult compare(const vector<half>& out,
                    const vector<float>& ref,
                    double atol, double rtol) {
    CheckResult r{0.0, 0, true};
    if (out.size() != ref.size()) { r.pass = false; return r; }

    for (size_t i = 0; i < ref.size(); i++) {
        double got  = __half2float(out[i]);
        double want = ref[i];
        if (isnan(got)) { r.num_nan++; r.pass = false; continue; }

        double err = fabs(got - want);
        if (err > r.max_abs_err) r.max_abs_err = err;
        if (err > atol + rtol * fabs(want)) r.pass = false;
    }
    return r;
}