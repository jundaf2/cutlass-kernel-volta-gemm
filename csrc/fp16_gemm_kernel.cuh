#pragma once
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>


namespace volta{


namespace kernel {

template<typename Gemm_traits>
__global__ void gemm(const Gemm_traits &gemm_traits, const typename Gemm_traits::Arguments& args)
{
    using cute::Tensor,cute::Stride,cute::Layout,cute::Step,cute::Shape;
    using cute::Copy_Atom,cute::MMA_Atom;
    using cute::UniversalCopy,cute::DefaultCopy;
    using cute::make_tiled_copy,cute::make_tiled_copy_A,cute::make_tiled_copy_B,cute::make_tiled_copy_C;
    using cute::get,cute::size,cute::append,cute::shape,cute::clear,cute::take;
    using cute::Int;
    using cute::_,cute::_0,cute::_1,cute::_2,cute::_4,cute::_8,cute::_16,cute::_32,cute::_64,cute::_128,cute::_256;
    using cute::make_coord,cute::make_shape,cute::make_tuple,cute::make_fragment_like,cute::make_identity_tensor,cute::make_smem_ptr,cute::make_tensor,cute::make_coord_iterator,cute::make_int_sequence;
    using cute::tile_to_shape,cute::local_tile;
    using cute::for_each;
    using cute::partition_fragment_C;
    
    using index_t = uint32_t;
    using ElementA = typename Gemm_traits::ElementA;
    using ElementB = typename Gemm_traits::ElementB;
    using ElementAccumulator = typename Gemm_traits::ElementAccumulator;
    using ElementC = typename Gemm_traits::ElementC;

    using GmemTiledCopyA = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<ElementA>, ElementA>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}, Layout<Shape<_1, _1>>{}));
    using GmemTiledCopyB = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<ElementB>, ElementB>{}, Layout<Shape<_32, _8>, Stride< _8, _1>>{}, Layout<Shape<_1, _1>>{}));
    using GmemTiledCopyC_R2G = Copy_Atom<DefaultCopy, ElementC>;  // using GmemTiledCopyC =  decltype(make_tiled_copy(Copy_Atom<UniversalCopy<ElementA>, ElementA>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}, Layout<Shape<_1, _1>>{}));

    using SmemLayoutAtomA = decltype(Layout<Shape<_128, _8>, Stride<_1, Int<128 + 4>>>{});
    using SmemLayoutAtomB = decltype(Layout<Shape<_128, _8>, Stride<_1, _128>>{});
    using TileShape = Gemm_traits::ThreadblockShape;
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));
    using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{}))));

    using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
    using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
    using SmemCopyAtomC_R2S = Copy_Atom<DefaultCopy, ElementAccumulator>;
    using SmemTiledCopyS2R = TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>, Layout<Shape<_128,_8>, Stride<_8,_1>>, Shape<_128,_8>>;

    typename Gemm_traits::EpilogueOutputOp::Params epilogue_op = typename Gemm_traits::EpilogueOutputOp::Params();

    using MMA_Atom_Arch = MMA_Atom<cute::SM70_8x8x4_F32F16F16F32_TT>;
    using ValLayoutMNK = Layout<Shape<_1, _1, _2>>;
    using TiledMma = TiledMMA<typename MMA_Atom_Arch, Layout<Shape<_16, _16, _1>>, typename ValLayoutMNK>;
    using TransformA = cutlass::ComplexTransform::kNone;
    using TransformB = cutlass::ComplexTransform::kNone;

    __shared__ ElementA smemA[cute::cosize_v<SmemLayoutA>];
    __shared__ ElementB smemB[cute::cosize_v<SmemLayoutB>];
    __shared__ ElementC smemC[cute::cosize_v<SmemLayoutC>];

    const int thread_idx = threadIdx.x;

    auto problem_shape_MNKL = append<4>(args.problem_size, Int<1>{});
    auto blk_shape = TileShape{}; 
    auto [m_coord, n_coord, l_coord] = blockIdx;
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);  

    Tensor A_mkl = args.ref_A;
    Tensor B_nkl = args.ref_B;
    Tensor C_mnl = args.ref_C;
    Tensor D_mnl = args.ref_D;

    // Get batch slice
    Tensor A_mk = A_mkl(_,_,l_coord);                                                                        // (m,k)
    Tensor B_nk = B_nkl(_,_,l_coord);                                                                        // (n,k)

    // Slice to get the tiles this thread block is responsible for
    Tensor gA = local_tile(A_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1,_,_1>{});           // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(B_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_,_1,_1>{});           // (BLK_N,BLK_K,k)

    Tensor gC_mnl = local_tile(C_mnl, blk_shape, make_coord(_,_,_), Step<_1,_1,_>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(D_mnl, blk_shape, make_coord(_,_,_), Step<_1,_1,_>{});      // (BLK_M,BLK_N,m,n,l)
    
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)

    Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{}); 
    Tensor sB = make_tensor(make_smem_ptr(smemB), SmemLayoutB{}); 
    Tensor sC = make_tensor(make_smem_ptr(smemC), SmemLayoutC{});

    GmemTiledCopyA gmem_tiled_copy_a;
    GmemTiledCopyB gmem_tiled_copy_b;
    auto gmem_thr_copy_a = gmem_tiled_copy_a.get_slice(thread_idx);
    auto gmem_thr_copy_b = gmem_tiled_copy_b.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_a.partition_S(gA);                                
    Tensor tAsA = gmem_thr_copy_a.partition_D(sA);                                
    Tensor tBgB = gmem_thr_copy_b.partition_S(gB); 
    Tensor tBsB = gmem_thr_copy_b.partition_D(sB);

    Tensor tArA = make_fragment_like(tAsA);
    Tensor tBrB = make_fragment_like(tBsB);

    auto k_tile_iter  = make_coord_iterator(shape<2>(gA));
    int  k_tile_count = size<2>(gA);

    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL); 
    // Compute tile residues for predication
    auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
    auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
    auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);     

    //
    // PREDICATES
    //

    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tAcA = gmem_thr_copy_a.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = gmem_thr_copy_b.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m,0) = get<0>(tAcA(0,m,0)) < get<0>(residue_mnk);  // blk_m coord < residue_m
    }
    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = get<0>(tBcB(0,n,0)) < get<1>(residue_mnk);  // blk_n coord < residue_n
    }

    //
    // PREFETCH
    //

    // Clear the rmem tiles to account for predicated off loads
    clear(tArA);
    clear(tBrB);

    // Start async loads for 0th k-tile, where we take care of the k residue
    {
      Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tArA); ++k) {
        if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gA shifted)
          cute::copy_if(gmem_tiled_copy_a, tApA(_,k), tAgAk(_,_,k), tArA(_,_,k));
        }
      }
      Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBrB); ++k) {
        if (get<1>(tBcB(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gB shifted)
          cute::copy_if(gmem_tiled_copy_b, tBpB(_,k), tBgBk(_,_,k), tBrB(_,_,k));
        }
      }
      ++k_tile_iter;
      --k_tile_count;
    }

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);                       
    Tensor tCrB = thr_mma.partition_fragment_B(sB);                      
    
    Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(TileShape{}));
    clear(accum);

  
    auto thr_copy_A       = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCsA           = thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);

    auto thr_copy_B       = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCsB           = thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    auto thr_copy_C_r2s = make_tiled_copy_C(SmemCopyAtomC_R2S{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCaC = thr_copy_C_r2s.retile_S(accum);                                       
    Tensor tCsC = thr_copy_C_r2s.partition_D(sC);                                                             

    //
    // Prologue
    //

    // Copy rmem to smem
    cute::copy(tArA, tAsA);
    cute::copy(tBrB, tBsB);
    // Clear accumulators
    __syncthreads();

    // Load A, B smem->rmem for k=0
    cute::copy(tCsA(_,_,0), tCrA_copy_view(_,_,0));
    cute::copy(tCsB(_,_,0), tCrB_copy_view(_,_,0));
    //
    // Mainloop
    //

    // Size of the k-tiles's outer product mode (k)
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -1)
    {
      // Pipeline the outer products with a static for loop
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) 
      {
        if (k_block == K_BLOCK_MAX - 1) 
        {
          __syncthreads();

          // Copy rmem to smem
          cute::copy(tArA, tAsA);
          cute::copy(tBrB, tBsB);
          __syncthreads();
        }

        // Load A, B smem->rmem for k+1
        int k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;    // static
        cute::copy(tCsA(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        cute::copy(tCsB(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        if (k_block == 0) 
        {
          if (k_tile_count <= 0) {
            clear(tApA);
            clear(tBpB);
          }
          cute::copy_if(gmem_tiled_copy_a, tApA, tAgA(_,_,_,*k_tile_iter), tArA);
          cute::copy_if(gmem_tiled_copy_b, tBpB, tBgB(_,_,_,*k_tile_iter), tBrB);
          ++k_tile_iter;
          --k_tile_count;
        }

        // transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});

        // Thread-level register gemm for k
        // disambiguate gemm (shared with the namespace name)
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
      });
    }


    // Tile gD and gC by the shape of SmemLayout first
    Tensor gCt = local_tile(gC, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})), _);  
    Tensor gDt = local_tile(gD, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})), _); 

    // Partition sC, gC, and gD for the output
    auto tD     = SmemTiledCopyS2R{}.get_thread_slice(thread_idx);
    Tensor tDsC = tD.partition_S(sC);
    Tensor tDgC = tD.partition_D(gCt);
    Tensor tDgD = tD.partition_D(gDt);

    // Allocate intermediate registers on the dst tensors
    Tensor tDrC = make_tensor<ElementAccumulator>(take<0,3>(shape(tDgC)));          
    Tensor tDrD = make_tensor<ElementC>(shape(tDrC));             

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD)));       
    Tensor cDt  = local_tile(cD, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})), _);                         
    Tensor tDcD = tD.partition_D(cDt);


    // For each tiling needed for SmemLayout to cover shape(gD)
    CUTLASS_PRAGMA_UNROLL
    for (int step_m = 0; step_m < size<2>(cDt); ++step_m)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int step_n = 0; step_n < size<3>(cDt); ++step_n)
      {
        // Step 1. Copy to SMEM
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
            int mma_m = step_m * size<1>(tCsC) + pipe_m;
            int mma_n = step_n * size<2>(tCsC) + pipe_n;
            
            cute::copy(thr_copy_C_r2s, tCaC(_,mma_m,mma_n), tCsC(_,pipe_m,pipe_n));
          }
        }

        // Step 2. Wait for SMEM writes to complete
        __syncthreads();

        // Step 3. Copy from SMEM into a fragment
        cute::copy(SmemTiledCopyS2R{}, tDsC, tDrC);

        // Step 4. Wait for SMEM reads to complete
        __syncthreads();

        Tensor tDgDmn = tDgD(_,_,_,step_m,step_n);
        Tensor tDcDmn = tDcD(_,_,_,step_m,step_n);

        // Step 5. Elementwise operation with conversion
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tDrC); ++i) {
          tDrD(i) = epilogue_op(tDrC(i));
        }

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(tDgDmn); ++m) 
        {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < size<2>(tDgDmn); ++n) 
          {
            // Predication
            if (get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk) &&
                get<1>(tDcDmn(0,m,n)) < get<1>(residue_mnk)) 
            {
              // Step 6. Copy to GMEM
              cute::copy(GmemTiledCopyC_R2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
            }
          }
        }
      }
    }





    // extern __shared__ char smem_[];
    // using Element = cutlass::half_t;
    

    

    // using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    // using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;

    // Gemm_traits::GemmKernel::DefaultMmaCore;


    // using ArchTag = typename Gemm_traits::ArchTag;

    

    // using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
    // using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));
    // __shared__ ElementA smemA[cute::cosize_v<SmemLayoutA>];
    // __shared__ ElementB smemB[cute::cosize_v<SmemLayoutB>];

    
    

    // using GmemTiledCopyA = GmemTiledCopyA_;
    // using GmemTiledCopyB = GmemTiledCopyB_;
    
    // using SmemCopyAtomA = SmemCopyAtomA_;
    // using SmemCopyAtomB = SmemCopyAtomB_;
    // using TransformA = TransformA_;
    // using TransformB = TransformB_;·
    

    // static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    // static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
    // static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

    // static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    // static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
    // static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");


    // cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    // cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    
    // static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    // static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
    // static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");
    // static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    // static_assert(rank(SmemLayoutA{}) == 2,
    //   "MainloopTwoStage must not have a smem shape with a pipeline mode.");
    // static_assert(rank(SmemLayoutB{}) == 2,
    //   "MainloopTwoStage must not have a smem shape with a pipeline mode.");

    // // Construct shared memory tiles

    // SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    

    // // Partition the copying of A and B tiles across the threads
    // GmemTiledCopyA gmem_tiled_copy_a;
    // GmemTiledCopyB gmem_tiled_copy_b;
    // auto gmem_thr_copy_a = gmem_tiled_copy_a.get_slice(thread_idx);
    // auto gmem_thr_copy_b = gmem_tiled_copy_b.get_slice(thread_idx);

    // Tensor tAgA = gmem_thr_copy_a.partition_S(gA);                                  // (ACPY,ACPY_M,ACPY_K,k)
    // Tensor tAsA = gmem_thr_copy_a.partition_D(sA);                                  // (ACPY,ACPY_M,ACPY_K)
    // Tensor tBgB = gmem_thr_copy_b.partition_S(gB);                                  // (BCPY,BCPY_N,BCPY_K,k)
    // Tensor tBsB = gmem_thr_copy_b.partition_D(sB);                                  // (BCPY,BCPY_N,BCPY_K)

    // // Allocate the register tiles for double buffering -- same shape as partitioned data
    // Tensor tArA = make_fragment_like(tAsA);                                    // (ACPY,ACPY_M,ACPY_K)
    // Tensor tBrB = make_fragment_like(tBsB);                                    // (BCPY,BCPY_N,BCPY_K)

    // // Tile MMA compute thread partitions and allocate accumulators
    // TiledMma tiled_mma;
    // auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    // Tensor tCrA  = thr_mma.partition_fragment_A(sA);                           // (MMA,MMA_M,MMA_K)
    // Tensor tCrB  = thr_mma.partition_fragment_B(sB);                           // (MMA,MMA_M,MMA_K)

    // CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    // CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    // CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    // CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    // CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    // //
    // // Copy Atom retiling
    // //

    // auto thr_copy_A       = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma).get_thread_slice(thread_idx);
    // Tensor tCsA           = thr_copy_A.partition_S(sA);
    // Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    // CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M

    // auto thr_copy_B       = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma).get_thread_slice(thread_idx);
    // Tensor tCsB           = thr_copy_B.partition_S(sB);
    // Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);
    // CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    // //
    // // Prologue
    // //

    // // Copy gmem to rmem for the first k_tile
    // cute::copy(gmem_tiled_copy_a, tAgA(_,_,_,*k_tile_iter), tArA);
    // cute::copy(gmem_tiled_copy_b, tBgB(_,_,_,*k_tile_iter), tBrB);
    // if (--k_tile_count > 0) ++k_tile_iter;
    // // Copy rmem to smem
    // cute::copy(tArA, tAsA);
    // cute::copy(tBrB, tBsB);
    // // Clear accumulators
    // __syncthreads();

    // // Load A, B smem->rmem for k=0
    // cute::copy(tCsA(_,_,0), tCrA_copy_view(_,_,0));
    // cute::copy(tCsB(_,_,0), tCrB_copy_view(_,_,0));
    // //
    // // Mainloop
    // //

    // // Size of the k-tiles's outer product mode (k)
    // auto K_BLOCK_MAX = size<2>(tCrA);

    // CUTLASS_PRAGMA_NO_UNROLL
    // while (k_tile_count > -1)
    // {
    //   // Pipeline the outer products with a static for loop
    //   for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) 
    //   {
    //     if (k_block == K_BLOCK_MAX - 1) 
    //     {
    //       __syncthreads();

    //       // Copy rmem to smem
    //       cute::copy(tArA, tAsA);
    //       cute::copy(tBrB, tBsB);
    //       __syncthreads();
    //     }

    //     // Load A, B smem->rmem for k+1
    //     int k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;     // static
    //     cute::copy(tCsA(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
    //     cute::copy(tCsB(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
    //     if (k_block == 0) 
    //     {
    //       // Copy gmem to rmem
    //       cute::copy(gmem_tiled_copy_a, tAgA(_,_,_,*k_tile_iter), tArA);
    //       cute::copy(gmem_tiled_copy_b, tBgB(_,_,_,*k_tile_iter), tBrB);
    //       if (--k_tile_count > 0) ++k_tile_iter;
    //     }

    //     // transform before compute
    //     cute::transform(tCrA(_,_,k_block), TransformA{});
    //     cute::transform(tCrB(_,_,k_block), TransformB{});

    //     // Thread-level register gemm for k
    //     // disambiguate gemm (shared with the namespace name)
    //     cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
    //   });
    // }
}

}

}