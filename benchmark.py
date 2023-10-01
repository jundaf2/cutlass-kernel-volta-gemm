import torch
import pytest
import volta_cutlass_gemm
dtype = torch.float16
device = torch.device('cuda')

torch.set_printoptions(edgeitems=8,linewidth=1000,sci_mode=False)

def test_matmulfp16(M,N,K): 
    A_ref = torch.randn([M,K], dtype=torch.float, device=device)
    B_ref = torch.randn([K,N], dtype=torch.float, device=device)
    C_ref = torch.matmul(A_ref, B_ref)
    C_ref_half = C_ref.to(dtype)

    A_half =  A_ref.to(dtype)
    B_half = B_ref.to(dtype)
    C_half = volta_cutlass_gemm.gemm_fp16(A_half, B_half, "wmma")

    # print("C_half", C_half)
    # print("C_ref_half", C_ref_half)

    thhd = 1e-3
    assert (C_half - C_ref_half).abs().max().item() <= thhd * min((C_half).abs().max().item(), (C_ref_half).abs().max().item()), "result amax diff: " + str((C_half - C_ref_half).abs().max().item()) + " allowed amax diff: " + str(thhd * min((C_half).abs().max().item(), (C_ref_half).abs().max().item()))




M = 256
N = 256
K = 256
print("----------------------------")
print("(M,N,K):({},{},{})".format(M,N,K))
test_matmulfp16(M,N,K)

for M in [128,256,512,1024,2048,4096]:
    for N in [128,256,512,1024,2048,4096]:
        for K in [64,128,256,512,1024,2048,4096]:
            print("----------------------------")
            print("(M,N,K):({},{},{})".format(M,N,K))
            test_matmulfp16(M,N,K)