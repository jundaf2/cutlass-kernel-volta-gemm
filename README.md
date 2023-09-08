# cutlass-kernel-volta-gemm
This is a cutlass-based kernel-level GEMM for Volta architecture.

## Dependencies
- pytorch
- pytest

## Notes
First deploy blocks, then deploy warps.
- Block
    - BM = 64
    - BN = 64
- Warp 
    - WM = 16
    - WN = 16
NUM_WARPS = (BMxBN)/(WMxWN) = 16
NUM_THREADS_PER_CTA = WARP_SIZExNUM_WARPS = 32x16 = 512