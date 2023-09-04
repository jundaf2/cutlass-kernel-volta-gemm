#include <cutlass/numeric_types.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


at::Tensor
gemm_fp16(const at::Tensor &A,     
        const at::Tensor &B,        
        at::Tensor &C,
){
    at::Tensor C;
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc("volta_cutlass_gemm"),
    m.def("gemm_fp16", &gemm_fp16, "gemm using volta fp16 tensor core based on cutlass");
}