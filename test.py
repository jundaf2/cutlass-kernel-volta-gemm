import torch
import pytest
import volta_cutlass_gemm
dtype = torch.float16
device = torch.device('cuda')


M = 1024
N = 2048
K = 512
A_ref = torch.randn([M,K], dtype=torch.float, device=device)
B_ref = torch.randn([K,N], dtype=torch.float, device=device)
A_half =  A_ref.to(dtype)
B_half = B_ref.to(dtype)
C_half = volta_cutlass_gemm.gemm_fp16(A_half, B_half,"cute")

# @pytest.mark.parametrize("M", [744,1400,1640,1760,1864,2552,4576,4752,6416,7304,7800,8056])
# @pytest.mark.parametrize("N", [240,1456,2528,3800,4352,4664,4816,5680,5952,6232,6856,6928])
# @pytest.mark.parametrize("K", [64,200,2088,2840,3360,3840,4176,4296,4736,5736,6032,6048])
# def test_matmulfp16(M,N,K): 
#     A_ref = torch.randn([M,K], dtype=torch.float, device=device)
#     B_ref = torch.randn([K,N], dtype=torch.float, device=device)
#     C_ref = torch.matmul(A_ref, B_ref)
#     C_ref_half = C_ref.to(dtype)

#     A_half =  A_ref.to(dtype)
#     B_half = B_ref.to(dtype)
#     C_half = volta_cutlass_gemm.gemm_fp16(A_half, B_half,"cute")

#     print("C_half", C_half)
#     print("C_ref_half", C_ref_half)

#     thhd = 1e-3
#     assert (C_half - C_ref_half).abs().max().item() <= thhd * min((C_half).abs().max().item(), (C_ref_half).abs().max().item()), "result amax diff: " + str((C_half - C_ref_half).abs().max().item()) + " allowed amax diff: " + str(thhd * min((C_half).abs().max().item(), (C_ref_half).abs().max().item()))


