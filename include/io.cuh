// headers for io  (this contains the struct signature and  the func signature)
#pragma once
#include <cuda_fp16.h>
#include <string>
#include <vector>
using namespace std;

struct Fixture{     
                    int H, D, L;
                    vector<half> q, K, V;
                    vector<float> expected;
                };

Fixture load_fixture(const string& folder);