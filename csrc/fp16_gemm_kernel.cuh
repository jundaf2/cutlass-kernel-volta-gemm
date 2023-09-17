#pragma once
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>
#include "cutlass/numeric_conversion.h"
#include "cute/tensor.hpp"
#include <mma.h>
#include <cuda_fp16.h>

namespace volta{


namespace cute_kernel {
using namespace cute;
#define CUTE_PRINT(VAR) \
  do { \
    print(#VAR); \
    print("   : "); \
    print(VAR); \
    print("\n"); \
  } while(0)

// all matrix are assumed row-major
struct Gemm_traits {
  using index_t = uint32_t;

  using BlockTileShape = Shape<_64, _64, _64>;
  using WarpTileShape  = Shape<_32, _32, _32>;
  
  using MMA_Atom_Arch = MMA_Atom<cute::SM70_8x8x4_F32F16F16F32_TN>;

  using MMAThrLayout = Layout<Shape<_8,_8,_1>>; // [1, warp tile size] = _M/2*16, _N/2*16, _K*4   [2, how many warps] = _M*_N*_K/((16/8)*(16/8)*(4/4)) = (_M/2) * (_N/2) * _K
  using MMAValLayout = Layout<Shape<_1,_1,_16>>; // warp register tile repeat how many times = _M*_N*_K
  using TiledMMA = TiledMMA<MMA_Atom_Arch, MMAThrLayout, MMAValLayout>; //  , Permutations

  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMMA{});

  using ElementA  = cutlass::half_t;
  const ElementA* ptr_A = nullptr;

  using ElementB  = cutlass::half_t;
  const ElementB* ptr_B = nullptr;

  using ElementAccumulator = float;

  using ElementC = cutlass::half_t;
  ElementC* ptr_C = nullptr;

  using ProblemShape = Shape<index_t,index_t,index_t,index_t>;
  ProblemShape problem_shape;

  CUTE_HOST_DEVICE
  Gemm_traits(index_t m, index_t n, index_t k){
    problem_shape = cute::make_shape(m, n, k, 1);
  }

  CUTE_HOST_DEVICE
  dim3
  get_grid_shape() {
    int batch_count = 1;
    if constexpr (rank(ProblemShape{}) == 4) {
      batch_count = cute::size<3>(problem_shape);
    }

    return dim3(
      cute::size(cute::ceil_div(cute::shape<0>(problem_shape), cute::shape<0>(BlockTileShape{}))),
      cute::size(cute::ceil_div(cute::shape<1>(problem_shape), cute::shape<1>(BlockTileShape{}))),
      batch_count
    );
  }

  CUTE_HOST_DEVICE
  dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }
};

template<typename Gemm_traits>
__global__ void gemm(const Gemm_traits gemm_traits)
{
    using __ = decltype(_);

    using ElementA = typename Gemm_traits::ElementA;
    using ElementB = typename Gemm_traits::ElementB;
    using ElementAccumulator = typename Gemm_traits::ElementAccumulator;
    using ElementC = typename Gemm_traits::ElementC;

    using TiledMMA = typename Gemm_traits::TiledMMA;
    using BlockTileShape = typename Gemm_traits::BlockTileShape;
    using WarpTileShape = typename Gemm_traits::WarpTileShape;

    // Smem Level Atom (Row Major)
    using SmemLayoutAtomA = Layout<Shape<_16,_32>, Stride<_32,_1>>; // Int<128 + 4>
    using SmemLayoutAtomB = Layout<Shape<_16,_32>, Stride<_32,_1>>;
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(shape<0>(BlockTileShape{}), shape<2>(BlockTileShape{})))); // 8, 32 -> 64, 64
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(shape<1>(BlockTileShape{}), shape<2>(BlockTileShape{})))); // 8 ,32 -> 64, 64
    using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
    using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
    using SmemTiledCopyA = decltype(make_tiled_copy_A(SmemCopyAtomA{}, TiledMMA{}));
    using SmemTiledCopyB = decltype(make_tiled_copy_B(SmemCopyAtomB{}, TiledMMA{}));

    // Gmem Level Atom (Col Major)
    using GmemLayoutAtomA = Layout<Shape<_16,_32>, Stride<_32,_1>>; // Gemm_traits::MaxThreadsPerBlock / Int<32>::value>
    using GmemLayoutAtomB = Layout<Shape<_16,_32>, Stride<_32,_1>>; // Gemm_traits::MaxThreadsPerBlock / Int<32>::value>
    using GmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
    using GmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
    using GmemTiledCopyA = decltype(make_tiled_copy(GmemCopyAtomA{}, GmemLayoutAtomA{}, Layout<Shape<_1,_1>>{}));
    using GmemTiledCopyB = decltype(make_tiled_copy(GmemCopyAtomB{}, GmemLayoutAtomB{}, Layout<Shape<_1,_1>>{}));
    using GmemTiledCopyC_R2G = Copy_Atom<DefaultCopy, ElementC> ; // Copy_Atom<DefaultCopy, ElementC>;

    __shared__ ElementA smemA[12288]; // cute::cosize_v<SmemLayoutA>
    __shared__ ElementB smemB[12288]; // cute::cosize_v<SmemLayoutB>

    const int thread_idx = threadIdx.x;
    auto problem_shape_MNKL = append<4>(gemm_traits.problem_shape, 1);
    auto blk_shape = BlockTileShape{}; 
    auto [m_coord, n_coord, l_coord] = blockIdx;
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);  

    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL); 

    // the input matrix A B C with its original layout
    Tensor A_mkl = make_tensor(make_gmem_ptr(gemm_traits.ptr_A), make_shape(M,K,L), make_stride(K, 1, M*K));
    Tensor B_nkl = make_tensor(make_gmem_ptr(gemm_traits.ptr_B), make_shape(N,K,L), make_stride(N, 1, N*K));
    Tensor C_mnl = make_tensor(make_gmem_ptr(gemm_traits.ptr_C), make_shape(M,N,L), make_stride(N, 1, M*N));

    Tensor A_mk = A_mkl(_,_,l_coord);                                                                        // (m,k)
    Tensor B_nk = B_nkl(_,_,l_coord);                                                                        // (n,k)
    Tensor C_mn = C_mnl(_,_,l_coord);                                                                        // (n,k)

    // Slice to get the tiles this thread block is responsible for
    Tensor gA = local_tile(A_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1,__,_1>{});           // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(B_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step<__,_1,_1>{});           // (BLK_N,BLK_K,k)                                          
    Tensor gC = local_tile(C_mn, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1,_1,__>{});           // (BLK_M,BLK_N)

    Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), SmemLayoutB{});   // (BLK_N,BLK_K,PIPE)


    GmemTiledCopyA gmem_tiled_copy_a;
    GmemTiledCopyB gmem_tiled_copy_b;
    auto gmem_thr_copy_a = gmem_tiled_copy_a.get_thread_slice(thread_idx);
    auto gmem_thr_copy_b = gmem_tiled_copy_b.get_thread_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_a.partition_S(gA);                        // (ACPY,ACPY_M,ACPY_K,k)         
    Tensor tAsA = gmem_thr_copy_a.partition_D(sA);                        // (ACPY,ACPY_M,ACPY_K)      
    Tensor tArA = make_fragment_like(tAsA);                               // (ACPY,ACPY_M,ACPY_K), for copy from gmem -> rmem -> smem

    Tensor tBgB = gmem_thr_copy_b.partition_S(gB);                        // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_b.partition_D(sB);                        // (BCPY,BCPY_N,BCPY_K)
    Tensor tBrB = make_fragment_like(tBsB);                               // (BCPY,BCPY_N,BCPY_K), for copy from gmem -> rmem -> smem

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);       // (MMA,MMA_M,MMA_K)                
    Tensor tCrB = thr_mma.partition_fragment_B(sB);       // (MMA,MMA_N,MMA_K)               
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(BlockTileShape{}));
    clear(accum);

    SmemTiledCopyA smem_tiled_copy_a;
    auto smem_thr_copy_A       = smem_tiled_copy_a.get_thread_slice(thread_idx);
    Tensor tCsA           = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

    SmemTiledCopyB smem_tiled_copy_b;
    auto smem_thr_copy_B       = smem_tiled_copy_b.get_thread_slice(thread_idx);
    Tensor tCsB           = smem_thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);


    // k-tile num
    auto K_BLOCK_MAX = size<2>(tCrA);
    int  k_loop_num = size<2>(gA);
    int k_loop_count = 0;

    if (thread_idx == 0 && m_coord == 0 && n_coord == 0) {
      CUTE_PRINT(M);
      CUTE_PRINT(N);
      CUTE_PRINT(K);
      CUTE_PRINT(L);

      CUTE_PRINT(int(cute::cosize_t<SmemLayoutA>::value));
      CUTE_PRINT(int(cute::cosize_t<SmemLayoutB>::value));

      CUTE_PRINT(gA.layout());
      CUTE_PRINT(gB.layout());
      CUTE_PRINT(gC.layout());
      CUTE_PRINT(sA.layout());
      CUTE_PRINT(sB.layout());
      
      CUTE_PRINT(tAgA.layout());
      CUTE_PRINT(tAsA.layout());
      CUTE_PRINT(tBgB.layout());
      CUTE_PRINT(tBsB.layout());
      CUTE_PRINT(tArA.layout());
      CUTE_PRINT(tBrB.layout());
      CUTE_PRINT(tCrA.layout());
      CUTE_PRINT(tCrB.layout());
      CUTE_PRINT(accum.layout());

      CUTE_PRINT(k_loop_num); // 128/32 = 4
      CUTE_PRINT(K_BLOCK_MAX); // 32/4 = 8

      CUTE_PRINT(tCsA.layout());
      CUTE_PRINT(tCsB.layout());
      CUTE_PRINT(tCrA_copy_view.layout());
      CUTE_PRINT(tCrB_copy_view.layout());
    }


    // auto k_tile_iter=make_coord_iterator(shape<2>(gA))
    
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_loop_count < k_loop_num; k_loop_count++) // k_loop_num
    {
      // Pipeline the outer products with a static for loop
      CUTLASS_PRAGMA_UNROLL
      for(int k_block = 0; k_block < K_BLOCK_MAX; k_block++) // K_BLOCK_MAX
      {
        
        if (k_block == 0) 
        {
          // Copy gmem to rmem 
          cute::copy(gmem_tiled_copy_a, tAgA(_,_,_,k_loop_count), tArA);
          cute::copy(gmem_tiled_copy_b, tBgB(_,_,_,k_loop_count), tBrB);

          // Copy rmem to smem, illegal memory access
          cute::copy(tArA, tAsA); 
          cute::copy(tBrB, tBsB);
          __syncthreads();
        }

        // Load A, B smem->rmem for k
        cute::copy(smem_tiled_copy_a, tCsA(_,_,k_block), tCrA_copy_view(_,_,k_block));
        cute::copy(smem_tiled_copy_b, tCsB(_,_,k_block), tCrB_copy_view(_,_,k_block));

        // Thread-level register gemm for k
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
      }
    }


    // constexpr int numel = decltype(size(accum))::value;
    // cutlass::NumericArrayConverter<ElementC, ElementAccumulator, numel> convert_op;
    // auto frag = convert_op(*reinterpret_cast<const cutlass::Array<ElementAccumulator, numel> *>(accum.data()));
    // Tensor rC = make_tensor(make_rmem_ptr<ElementC>(&frag), accum.layout());

        

    // auto thr_copy_C_r2g = make_tiled_copy_C(GmemTiledCopyC_R2G{}, tiled_mma).get_thread_slice(thread_idx);
    // Tensor tCrC = thr_copy_C_r2g.retile_S(rC); // rC; //                // ((Atom,AtomNum), MMA_M, MMA_N)                 
    
    // Tensor gCt = local_tile(gC, make_shape(shape<0>(BlockTileShape{}), shape<1>(BlockTileShape{})), _);  
    // Tensor tCgC = thr_copy_C_r2g.partition_D(gCt);
    // Tensor cC   = make_identity_tensor(make_shape(size<0>(gC),size<1>(gC)));       
    // Tensor cCt  = local_tile(cC, make_shape(shape<0>(BlockTileShape{}), shape<1>(BlockTileShape{})), _);                         
    // Tensor tCcC = thr_copy_C_r2g.partition_D(cCt);


    // // Compute tile residues for write Predication
    // auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
    // auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
    // auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
    // auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);    


    // CUTLASS_PRAGMA_UNROLL
    // for (int step_m = 0; step_m < size<2>(cCt); ++step_m)
    // {
    //   CUTLASS_PRAGMA_UNROLL
    //   for (int step_n = 0; step_n < size<3>(cCt); ++step_n)
    //   {
    //     // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    //     Tensor tCgCmn = tCgC(_,_,_,step_m,step_n);
    //     Tensor tCcCmn = tCcC(_,_,_,step_m,step_n);
    //     CUTLASS_PRAGMA_UNROLL
    //     for (int m = 0; m < size<1>(tCgCmn); ++m) 
    //     {
    //       CUTLASS_PRAGMA_UNROLL
    //       for (int n = 0; n < size<2>(tCgCmn); ++n) 
    //       {
    //         // Predication
    //         if (get<0>(tCcCmn(0,m,n)) < get<0>(residue_mnk) && get<1>(tCcCmn(0,m,n)) < get<1>(residue_mnk)) 
    //         {
    //           cute::copy(tCrC(_,m,n), tCgCmn(_,m,n)); // thr_copy_C_r2g, 
    //         }
    //       }
    //     }
    //   }
    // }
}

}

namespace wmma_kernel{

using namespace nvcuda;
using half_t = __half;
using index_t = unsigned int;
typedef int4 copy_t;

template <int BM, int BN, int BK, int WM, int WN>
__global__ void gemm(
    half_t * __restrict__ A, half_t * __restrict__ B, half_t * __restrict__ C,
    const index_t M, const index_t N, const index_t K) {

    const index_t block_n = blockIdx.x;
    const index_t block_m = blockIdx.y;
    const index_t thread_id = threadIdx.x;
    const index_t lane_id = thread_id % 32; 
    const index_t warp_id = thread_id / 32;

    constexpr index_t APAD = 8;
    constexpr index_t BPAD = 8;
    constexpr index_t TC = 16;
    constexpr index_t LDST_ELT_NUM = 8; 

    __shared__ half_t smem_a[BM][BK + APAD]; // [128, 32]
    __shared__ half_t smem_b[BK][BN + BPAD]; // [32, 128]

    constexpr index_t kNWarps = (BM/WM)*(BN/WN);
    constexpr index_t kThreadsPerBlock = kNWarps*32;
    constexpr index_t WARP_M_LOOP = WM / TC;
    constexpr index_t WARP_N_LOOP = WN / TC;
    constexpr index_t WARP_K_LOOP = BK / TC;

    wmma::fragment<wmma::accumulator, TC, TC, TC, float> frag_acc[WARP_M_LOOP][WARP_N_LOOP];

    #pragma unroll
    for (index_t m = 0; m < WARP_M_LOOP; m++) {
        #pragma unroll
        for (index_t n = 0; n < WARP_N_LOOP; n++) {
            wmma::fill_fragment(frag_acc[m][n], 0.0);
        }
    }

    // A copy thread Layout = [128, 4], [64,  8]
    constexpr index_t THR_SHAPE_A_K = BK/LDST_ELT_NUM; 
    constexpr index_t THR_SHAPE_A_M = kThreadsPerBlock/THR_SHAPE_A_K; 
    const index_t thr_layout_a_m = thread_id / THR_SHAPE_A_K;
    const index_t thr_layout_a_k =  thread_id % THR_SHAPE_A_K;

    // B copy thread Layout = [32, 16], [32, 16]
    constexpr index_t THR_SHAPE_B_N = BN/LDST_ELT_NUM;
    constexpr index_t THR_SHAPE_B_K = kThreadsPerBlock/THR_SHAPE_B_N;
    const index_t thr_layout_b_k = thread_id / THR_SHAPE_B_N;
    const index_t thr_layout_b_n = thread_id % THR_SHAPE_B_N;

    constexpr index_t THR_STRIDE_A_M = BM/THR_SHAPE_A_M;
    constexpr index_t THR_STRIDE_A_K = BK/THR_SHAPE_A_K;
    constexpr index_t THR_STRIDE_B_N = BN/THR_SHAPE_B_N;
    constexpr index_t THR_STRIDE_B_K = BK/THR_SHAPE_B_K;

    const index_t smem_thr_store_a_m = thr_layout_a_m * THR_STRIDE_A_M;
    const index_t smem_thr_store_a_k = thr_layout_a_k * THR_STRIDE_A_K;
    const index_t smem_thr_store_b_k = thr_layout_b_k * THR_STRIDE_B_K;
    const index_t smem_thr_store_b_n = thr_layout_b_n * THR_STRIDE_B_N;

    const index_t gmem_local_tile_m = block_m * BM;
    const index_t gmem_local_tile_n = block_n * BN;

    index_t gmem_thr_load_c_m = (gmem_local_tile_m + smem_thr_store_a_m) * K + smem_thr_store_a_k;
    index_t gmem_thr_load_c_n = smem_thr_store_b_k * N + gmem_local_tile_n + smem_thr_store_b_n;

    index_t warp_m = warp_id / (BM/WM);
    index_t warp_n = warp_id % (BM/WM);

    #pragma nounroll
    for (index_t k_loop = 0; k_loop < K / BK; k_loop++) {
        #pragma unroll
        for(index_t smem_a_load_m = 0; smem_a_load_m < THR_STRIDE_A_M; smem_a_load_m++) {
            *reinterpret_cast<copy_t*>(&smem_a[smem_thr_store_a_m + smem_a_load_m][smem_thr_store_a_k]) = *reinterpret_cast<copy_t*>(&A[gmem_thr_load_c_m + smem_a_load_m*K]);
        }
        
        
        #pragma unroll
        for(index_t smem_b_load_k = 0; smem_b_load_k < THR_STRIDE_B_K; smem_b_load_k++) {
            *reinterpret_cast<copy_t*>(&smem_b[smem_thr_store_b_k + smem_b_load_k][smem_thr_store_b_n]) = *reinterpret_cast<copy_t*>(&B[gmem_thr_load_c_n + smem_b_load_k*N]);
        }


        gmem_thr_load_c_m += BK;
        gmem_thr_load_c_n += BK * N;

        __syncthreads();

        #pragma unroll
        for(index_t k = 0; k < WARP_K_LOOP; k++){
          wmma::fragment<wmma::matrix_a, TC, TC, TC, half_t, wmma::row_major> frag_a[WARP_M_LOOP];
          wmma::fragment<wmma::matrix_b, TC, TC, TC, half_t, wmma::row_major> frag_b[WARP_N_LOOP];

          #pragma unroll
          for (index_t m = 0; m < WARP_M_LOOP; m++) {
              wmma::load_matrix_sync(frag_a[m], &smem_a[warp_m * WM + m*TC][k*TC], BK + APAD);
              #pragma unroll
              for (index_t n = 0; n < WARP_N_LOOP; n++) {
                wmma::load_matrix_sync(frag_b[n], &smem_b[k*TC][warp_n * WN + n*TC], BN + BPAD);
                wmma::mma_sync(frag_acc[m][n], frag_a[m], frag_b[n], frag_acc[m][n]);
              }
          }
        }
        __syncthreads();
    }

    const index_t gmem_thr_store_c_m = block_m * BM + warp_m * WM;
    const index_t gmem_thr_store_c_n = block_n * BN + warp_n * WN;
    index_t gmem_thr_store_c = gmem_thr_store_c_m * N + gmem_thr_store_c_n; 


    wmma::fragment<wmma::accumulator, TC, TC, TC, half_t> frag_c[WARP_M_LOOP][WARP_N_LOOP];

    #pragma unroll
    for (index_t m = 0; m < WARP_M_LOOP; m++) {
        #pragma unroll
        for (index_t n = 0; n < WARP_N_LOOP; n++) {
            // volta, fp32 src reg -> fp16 dst reg
            #pragma unroll
            for (int t = 0; t < frag_acc[m][n].num_elements / 2; t++) { 
              index_t tt = (lane_id & 2) + ((t & 2) << 1) + (t & 1);
              frag_c[m][n].x[tt] = __float2half_rn(frag_acc[m][n].x[tt]);
            }

            #pragma unroll
            for (int t = 0; t < frag_acc[m][n].num_elements / 2; t++) { 
              index_t tt = ((lane_id & 2) ^ 2) + ((t & 2) << 1) + (t & 1);
              frag_c[m][n].x[tt] = __shfl_xor_sync(0xffffffff, __float2half_rn(frag_acc[m][n].x[tt]), 0x0002);
            }
            
            wmma::store_matrix_sync(&C[gmem_thr_store_c + m * TC * N + n * TC], frag_c[m][n], N, wmma::mem_row_major);
        }
    }
}


}

}