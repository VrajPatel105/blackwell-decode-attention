// Harness for the decode-attention kernels.
//
//   decode                       correctness only, every kernel, every length (as before)
//   decode 1                     correctness only, kernel id 1
//   decode --bench               correctness, then timing of every kernel that passes
//   decode --bench --len 8192    one length only
//   decode --csv                 --bench, machine-readable rows on stdout (progress on stderr)
//   decode 1 --profile 8192      one launch only, for ncu / nsys
//   decode --bench --ceiling 779 use this bandwidth ceiling (GB/s) instead of measuring it
//
// For each length: load the fixture, run the kernel on a NaN-poisoned output, compare
// against the PyTorch oracle, and only if it passes, time it.
// Timing flushes L2 before every launch, so the number is DRAM bandwidth and not cache.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include "io.cuh"
#include "kernels.cuh"
#include "utils.cuh"
#include "compare.cuh"
#include "bench.cuh"
using namespace std;

static const vector<string> LENGTHS = {"1", "100", "1024", "4097", "8192"};

struct Args {
    int    kernel_id   = -1;     // -1 = all
    bool   bench       = false;
    bool   csv         = false;
    int    len         = 0;      // 0 = all lengths
    int    profile_len = 0;      // > 0 = single launch at this length
    double ceiling     = 0.0;    // GB/s, 0 = measure
};

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s [kernel_id|all] [--bench] [--csv] [--len L] [--profile L] [--ceiling GBps]\n"
        "  %s                        correctness only\n"
        "  %s --bench                correctness + timing\n"
        "  %s --csv                  timing as CSV on stdout\n"
        "  %s 1 --profile 8192       one launch of kernel 1, for ncu\n",
        prog, prog, prog, prog, prog);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* s = argv[i];
        if (strcmp(s, "--bench") == 0) {
            a.bench = true;
        } else if (strcmp(s, "--csv") == 0) {
            a.csv = true;
            a.bench = true;
        } else if (strcmp(s, "--len") == 0 && i + 1 < argc) {
            a.len = atoi(argv[++i]);
        } else if (strcmp(s, "--profile") == 0 && i + 1 < argc) {
            a.profile_len = atoi(argv[++i]);
        } else if (strcmp(s, "--ceiling") == 0 && i + 1 < argc) {
            a.ceiling = atof(argv[++i]);
        } else if (strcmp(s, "all") == 0) {
            a.kernel_id = -1;
        } else if (s[0] >= '0' && s[0] <= '9') {
            a.kernel_id = atoi(s);
        } else {
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (a.profile_len > 0 && a.kernel_id < 0) {
        fprintf(stderr, "--profile needs a specific kernel id\n");
        exit(EXIT_FAILURE);
    }
    return a;
}

static int choose_iterations(float one_launch_ms) {
    constexpr float TARGET_MS = 200.0f;
    constexpr int   MIN_ITERS = 5;
    constexpr int   MAX_ITERS = 10000;
    if (one_launch_ms <= 0.0f) return MAX_ITERS;
    int iters = int(TARGET_MS / one_launch_ms);
    return max(MIN_ITERS, min(MAX_ITERS, iters));
}

static constexpr int WARMUP = 5;

static void print_header(bool csv) {
    if (csv) {
        printf("id,name,L,H,D,pass,max_abs_err,num_nan,ms,bytes,gb_per_s,ceiling_gb_per_s,pct_ceiling\n");
    } else {
        printf("%-8s  %-22s  %-5s  %11s  %10s  %9s  %10s\n",
               "length", "kernel", "check", "max_abs_err", "ms", "GB/s", "% ceiling");
    }
}

// timing columns are blank / "-" when the kernel failed its check
static void print_row(bool csv, const KernelEntry& e, const Fixture& f, const CheckResult& r,
                      float ms, double bytes, double gbps, double ceiling, double pct) {
    if (csv) {
        if (r.pass)
            printf("%d,%s,%d,%d,%d,1,%.6e,%d,%.6f,%.0f,%.2f,%.2f,%.2f\n",
                   e.id, e.name, f.L, f.H, f.D, r.max_abs_err, r.num_nan, ms, bytes, gbps, ceiling, pct);
        else
            printf("%d,%s,%d,%d,%d,0,%.6e,%d,,,,%.2f,\n",
                   e.id, e.name, f.L, f.H, f.D, r.max_abs_err, r.num_nan, ceiling);
        return;
    }
    char len[16];
    snprintf(len, sizeof(len), "%d", f.L);
    if (r.pass)
        printf("%-8s  %-22s  %-5s  %11.2e  %10.5f  %9.1f  %10.2f\n",
               len, e.name, "PASS", r.max_abs_err, ms, gbps, pct);
    else
        printf("%-8s  %-22s  %-5s  %11.2e  %10s  %9s  %10s\n",
               len, e.name, "FAIL", r.max_abs_err, "-", "-", "-");
    fflush(stdout);
}

int main(int argc, char** argv) {

    const Args args = parse_args(argc, argv);

    vector<string> lens = LENGTHS;
    if (args.len > 0)         lens = {to_string(args.len)};
    if (args.profile_len > 0) lens = {to_string(args.profile_len)};

    // profile mode: exactly one launch, nothing else (for ncu)
    if (args.profile_len > 0) {
        const KernelEntry* entry = nullptr;
        for (int i = 0; i < NUM_KERNELS; ++i)
            if (KERNELS[i].id == args.kernel_id) entry = &KERNELS[i];
        if (!entry) {
            fprintf(stderr, "no kernel with id %d\n", args.kernel_id);
            return 1;
        }

        Fixture f = load_fixture("fixtures/len_" + lens[0]);
        half *d_q, *d_K, *d_V, *d_o;
        CUDA_CHECK(cudaMalloc((void**)&d_q, f.q.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_K, f.K.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_V, f.V.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_o, f.expected.size() * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_q, f.q.data(), f.q.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, f.K.data(), f.K.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, f.V.data(), f.V.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_o, 0xFF, f.expected.size() * sizeof(half)));

        fprintf(stderr, "profiling %s at L=%d (single launch)\n", entry->name, f.L);
        entry->fn(d_q, d_K, d_V, d_o, f.H, f.L, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_o);
        return 0;
    }

    // bandwidth ceiling and L2 flush buffer (bench mode only)
    double ceiling = args.ceiling;
    void*  d_flush = nullptr;
    size_t flush_bytes = 0;
    if (args.bench) {
        if (ceiling <= 0.0) {
            ceiling = measure_bandwidth_ceiling();
            fprintf(stderr, "bandwidth ceiling: %.1f GB/s (measured, device-to-device copy)\n", ceiling);
        } else {
            fprintf(stderr, "bandwidth ceiling: %.1f GB/s (given)\n", ceiling);
        }

        int dev = 0, l2_bytes = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev));
        flush_bytes = max<size_t>(2 * size_t(l2_bytes), size_t(64) << 20);
        CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
        fprintf(stderr, "L2 flush buffer: %.0f MiB (L2 is %.0f MiB)\n",
                flush_bytes / 1048576.0, l2_bytes / 1048576.0);

        print_header(args.csv);
    }

    bool any_fail = false;

    for (const string& len : lens) {

        Fixture f = load_fixture("fixtures/len_" + len);

        half *d_q, *d_K, *d_V, *d_o;
        vector<half> h_out(f.expected.size());

        CUDA_CHECK(cudaMalloc((void**)&d_q, f.q.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_K, f.K.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_V, f.V.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_o, f.expected.size() * sizeof(half)));

        CUDA_CHECK(cudaMemcpy(d_q, f.q.data(), f.q.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, f.K.data(), f.K.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, f.V.data(), f.V.size() * sizeof(half), cudaMemcpyHostToDevice));

        for (int i = 0; i < NUM_KERNELS; ++i) {
            const KernelEntry& e = KERNELS[i];
            if (args.kernel_id >= 0 && e.id != args.kernel_id) continue;

            auto run_kernel = [&]() {
                e.fn(d_q, d_K, d_V, d_o, f.H, f.L, 0);
            };

            // poison the output, run once, check against the oracle
            CUDA_CHECK(cudaMemset(d_o, 0xFF, f.expected.size() * sizeof(half)));
            run_kernel();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_o, f.expected.size() * sizeof(half), cudaMemcpyDeviceToHost));

            const CheckResult r = compare(h_out, f.expected);
            if (!r.pass) any_fail = true;

            if (!args.bench) {
                printf("len_%s: %s  max_err=%g  nan=%d  [%s]\n",
                       len.c_str(), r.pass ? "PASS" : "FAIL", r.max_abs_err, r.num_nan, e.name);
                continue;
            }

            if (!r.pass) {
                print_row(args.csv, e, f, r, 0.0f, 0.0, 0.0, ceiling, 0.0);
                continue;
            }

            // bytes the kernel has to move: K and V once each, plus q in and o out
            const double bytes = 2.0 * f.H * f.L * f.D * sizeof(half)
                               + 2.0 * f.H * f.D * sizeof(half);

            auto flush_l2 = [&]() {
                CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_bytes, 0));
            };

            const float est_ms = time_kernels(run_kernel, 1, 1, flush_l2);
            const float ms = time_kernels(run_kernel, WARMUP, choose_iterations(est_ms), flush_l2);
            const double gbps = bytes / (ms / 1e3) / 1e9;
            const double pct  = gbps / ceiling * 100.0;

            print_row(args.csv, e, f, r, ms, bytes, gbps, ceiling, pct);
        }

        cudaFree(d_q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_o);
    }

    if (d_flush) cudaFree(d_flush);
    return any_fail ? 1 : 0;
}
