import triton
import triton.language as tl


@triton.jit
def op(a1, b1, a2, b2):
    return a1 * a2, a2 * b1 + b2

@triton.jit
def ssm_load(Ks, A, B, C):
    "Helper for loading"
    a = tl.load(A + Ks)
    b = tl.load(B + Ks)
    c = tl.load(C + Ks)
    return a, b, c

@triton.jit
def ssm_scan(h1, h2, h2_0, dim: tl.constexpr = 0, rev: tl.constexpr = 0):
    n1, n2 = op(tl.zeros_like(h1)+1.0, h2_0, h1, h2)
    h1, h2 = tl.associative_scan((n1, n2), dim, op, reverse=rev)
    return h1, h2

@triton.jit
def select(X, mask, dim=-1):
    return tl.sum(X * mask, dim, 1)

@triton.jit
def pscan_tt(X, A, B, C, H_0, dH_0, dA, dB, dC, dY, Y, H, K: tl.constexpr, L: tl.constexpr, N: tl.constexpr, back: tl.constexpr = True):
    # L is the total length of the sequence.
    pid = tl.program_id(0)
    nH = tl.num_programs(0) # Number of blocks.
    Ks = tl.arange(0, K)[None, :]
    Ns = tl.arange(0, N)[:, None] # N x 1 - For each hidden.
    kid = pid * K
    h_span = Ns*nH + pid

    a, b, c = ssm_load(Ns * L + Ks + kid, A, B, C) # N x K
    x = tl.load(X + Ks + kid) # K
    h2_0 = tl.load(H_0 + nH*N + h_span) * (Ks==0)

    # Compute forward for all hidden.
    h1, h2 = ssm_scan(a, b * x, h2_0, dim=1)
    y = tl.sum(c * h2, 0)

    # Save
    tl.store(Y + Ks + kid, y[None, :])
    tl.store(H + 0 * nH*N + h_span, select(h1, Ks == (K-1)))
    tl.store(H + 1 * nH*N + h_span, select(h2, Ks == (K-1)))

    if not back: return

    
