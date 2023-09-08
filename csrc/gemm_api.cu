#include <cutlass/numeric_types.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "fp16_gemm.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

at::Tensor gemm_fp16(const at::Tensor &A,     
        const at::Tensor &B  
){
    auto dprops = at::cuda::getCurrentDeviceProperties();
    const auto a_shape = A.sizes();
    const auto b_shape = B.sizes();

    bool is_sm70 = dprops->major == 7 && dprops->minor == 0;
    bool is_half = (A.dtype() == torch::kFloat16) && (B.dtype() == torch::kFloat16);
    bool is_cuda = A.is_cuda() && B.is_cuda();
    bool is_contiguous = (A.stride(-1) == 1) && (B.stride(-1) == 1);
    bool is_multiplicable = a_shape[1] == b_shape[0];
    

    TORCH_CHECK(is_sm70, "Volta FP16 GEMM supports Volta GPU (Tesla V100, GeForce GTX TITAN V and Quadro GV100) only.");
    TORCH_CHECK(is_half, "Volta FP16 GEMM supports fp16 data type only.");
    TORCH_CHECK(is_cuda, "Volta FP16 GEMM supports inputs on cuda device only.");
    TORCH_CHECK(is_contiguous, "Volta FP16 GEMM supports input with contiguous last dimension only.");
    TORCH_CHECK(is_multiplicable, "Volta FP16 GEMM supports [m,k]x[k,n] only.");

    const auto m = a_shape[0];
    const auto n = b_shape[1];
    const auto k = b_shape[0];

    /*** may require padding A and B a little bit ***/

    at::cuda::CUDAGuard device_guard{(char) A.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    volta::Gemm_params params;
    params.M = m;
    params.N = n;
    params.K = k;

    at::Tensor C = torch::empty({m, n}, A.options());
    
    params.a_ptr = A.data_ptr();
    params.b_ptr = B.data_ptr();
    params.c_ptr = C.data_ptr();

    volta::run_gemm_fp16(params, stream);    

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "backend for volta_cutlass_gemm",
    m.def("gemm_fp16", &gemm_fp16, "gemm using volta fp16 tensor core based on cutlass");
}