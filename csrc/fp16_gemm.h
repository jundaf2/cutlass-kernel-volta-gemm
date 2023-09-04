#pragma once

#include <cuda.h>
#include <vector>

namespace volta {
    struct Gemm_params {
        using index_t = uint32_t;
        // The matrices.
        void *__restrict__ a_ptr;
        void *__restrict__ b_ptr;
        void *__restrict__ c_ptr;

        index_t m;
        index_t n;
        index_t k;
    };

    void run_gemm_fp16(Gemm_params &params, cudaStream_t stream);
}