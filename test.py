import torch
import pytest

dtype = torch.float16
device = torch.device('cuda')


M = 1024
N = 2048
K = 512

@pytest.mark.parametrize("M", [512,1024,2048])
@pytest.mark.parametrize("N", [512,1024,2048])
@pytest.mark.parametrize("K", [512,1024,2048])
def test_matmulfp16(M,N,K): 
    A_ref = torch.randn([M,K], dtype=torch.float, device=device)
    B_ref = torch.randn([K,N], dtype=torch.float, device=device)
    C_ref = torch.matmul(A_ref, B_ref)
    C_ref_half = C_ref.to(dtype)

    A_half =  A_ref.to(dtype)
    B_half = B_ref.to(dtype)
    C_half = torch.matmul(A_half, B_half)

    thhd = 1e-3
    assert (C_half - C_ref_half).abs().max().item() <= thhd * min((C_half).abs().max().item(), (C_ref_half).abs().max().item()), "result amax diff: " + str((C_half - C_ref_half).abs().max().item()) + " allowed amax diff: " + str(thhd * min((C_half).abs().max().item(), (C_ref_half).abs().max().item()))


