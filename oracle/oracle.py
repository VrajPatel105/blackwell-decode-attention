import torch

def torch_sdpa(q, K, V):

    # converting the fp16 to fp32 tensors
    q = q.to(torch.float32)
    K = K.to(torch.float32)
    V = V.to(torch.float32)

    # q comes in [h, t] so we convert it to be like K and V -> [h,1,t]
    q = torch.unsqueeze(q, 1)

    # print(q.size())
    # print(K.size())
    # print(V.size())
    # torch.Size([2, 1, 3])
    # torch.Size([2, 3, 4])
    # torch.Size([2, 3, 4])

    output = torch.nn.functional.scaled_dot_product_attention(q, K, V, is_causal=False).squeeze(1)

    return output

# one more manual function to verify the torch_sdpa function's  output 

def attention_manual(q, K, V):

    q = q.to(torch.float32)
    K = K.to(torch.float32)
    V = V.to(torch.float32)

    q = torch.unsqueeze(q, 1)

    # extract D 
    D = K.shape[2]

    # the attention formula is softmax( Q @ k^T / 1/root(D) ) @ V

    k_transpose = torch.transpose(K, 1, 2)

    scores = (q @ k_transpose) * (D ** -0.5)
    weights = torch.softmax(scores, dim=2)

    output = weights @ V

    return output.squeeze(1)


if __name__ == "__main__":

    torch.manual_seed(0)
    H, L, D = 2, 16, 128
    q = torch.randn(H, D, dtype=torch.float16)
    K = torch.randn(H, L, D, dtype=torch.float16)
    V = torch.randn(H, L, D, dtype=torch.float16)

    a = torch_sdpa(q, K, V)
    b = attention_manual(q, K, V)
    assert a.shape == b.shape == (H, D)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-5), (a - b).abs().max()

    # degenerate case: with L=1 the softmax is 1, so the output is exactly V
    K1, V1 = K[:, :1], V[:, :1]
    assert torch.allclose(torch_sdpa(q, K1, V1), V1[:, 0].float(), atol=1e-5)

    print("oracle self-check passed")