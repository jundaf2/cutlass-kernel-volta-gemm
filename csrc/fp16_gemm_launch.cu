#include <cuda_device_runtime_api.h>
#include <iostream>
#include "fp16_gemm_kernel.cuh"
#include "fp16_gemm.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/util/device_memory.h"
#include "cutlass/gemm/device/gemm.h"
#include "helper.h"
#include "typeinfo"

namespace volta {

// The code section below describes datatype for input, output matrices and computation between elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations

// The code section below describes matrix layout of input and output matrices. Row Major for Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;
// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;
// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- this is the number of elements per vectorized memory access. For half precision, it's 8 elements. This becomes the vector width of math instructions in epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function


// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; 
// Number of pipelines you want to use
constexpr int NumStages = 2;


using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;



template<bool DoTime>
void launch_matmul_kernel(Gemm_params &params, cudaStream_t stream) {
    GpuTimer timer;
    if constexpr (DoTime) timer.start();

    // method 1: Launch initialized CUTLASS kernel
    if (params.backend == Backend::CUTLASS) {
        // Create a tuple of problem size for matrix multiplication
        cutlass::gemm::GemmCoord problem_size(params.M, params.N, params.K);

        typename cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a(reinterpret_cast<ElementInputA* >(params.a_ptr), params.K);
        typename cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b(reinterpret_cast<ElementInputB* >(params.b_ptr), params.N);
        typename cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c(reinterpret_cast<ElementOutput* >(params.c_ptr), params.N);

        // Initialize alpha and beta for dot product computation
        ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
        ElementComputeEpilogue beta = ElementComputeEpilogue(0);

        // Split K dimension into 1 partitions
        int split_k_slices = 1;

        // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch instantiated CUTLASS kernel
        typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            tensor_a,  // <- reference to matrix A on device
                                            tensor_b,  // <- reference to matrix B on device
                                            tensor_c,  // <- reference to matrix C on device
                                            tensor_c,  // <- reference to matrix D on device
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor

        // Instantiate CUTLASS kernel depending on templates
        Gemm gemm_op;
        // Check the problem size is supported or not 
        cutlass::Status status = gemm_op.can_implement(arguments);
        CUTLASS_CHECK(status);

        status = gemm_op(arguments);
        CUTLASS_CHECK(status);
    }
    // method 2: Launch kernel written by WMMA API
    else if (params.backend == Backend::WMMA)
    {
        constexpr int BM = 128;
        constexpr int BN = 128;
        constexpr int BK = 64;
        constexpr int WM = 32;
        constexpr int WN = 32;

        int BX = (params.N + BN - 1) / BN;
        int BY = (params.M + BM - 1) / BM;

        dim3 dimGrid(BX, BY);
        dim3 dimBlock((BM/WM)*(BN/WN)*32);
        wmma_kernel::gemm<BM,BN,BK,WM,WN><<<dimGrid, dimBlock, 0, stream>>>(reinterpret_cast<__half* >(params.a_ptr), reinterpret_cast<__half* >(params.b_ptr), reinterpret_cast<__half* >(params.c_ptr), params.M, params.N, params.K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());
    }
    // method 3: Launch kernel written in CUTE
    else if (params.backend == Backend::CUTE)
    {  
        using namespace cute;
        cute_kernel::Gemm_traits gemm_traits(params.M, params.N, params.K);
        auto dimGrid = gemm_traits.get_grid_shape();
        auto dimBlock = gemm_traits.get_block_shape();
        std::cout << "dimGrid " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;
        std::cout << "dimBlock " << dimBlock.x << " " << dimBlock.y << " " << dimBlock.z << std::endl;
        cute_kernel::gemm<<<dimGrid, dimBlock, 0, stream>>>(gemm_traits);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());
    }

    if constexpr (DoTime){
        timer.stop();
        double runtime = timer.elapsed_millis();
        double gflops = 2 * double(params.M) * double(params.N) * double(params.K) / 1e9 / runtime; // Two flops per multiply-add

        std::cout << "Volta Half GEMM" << ":\n";
        std::cout << "Runtime: " << runtime << " ms\n";
        std::cout << "TFLOPs: " << gflops  << "\n";
    }
}
    
void run_gemm_fp16(Gemm_params &params, cudaStream_t stream) {
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
        std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

        // Returning zero when built on older Toolkits so tests pass. The actions of this SDK example are no-op.
        return;
    }

    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    if (props.major != 7) {
        std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75." << std::endl;     
        return;   
    }

    launch_matmul_kernel<true>(params, stream);
}
}