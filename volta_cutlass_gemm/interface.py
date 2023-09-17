from cuda_volta_cutlass_gemm import gemm_fp16 as gemm_fp16_

def gemm_fp16(A, B, backend="cute"):
    return gemm_fp16_(A, B, backend)
