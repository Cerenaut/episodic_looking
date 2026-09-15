import torch


def nonzero_bit_sequences(N, device=None):
    """
    Enumerates all the integers 1 .. 2^N - 1 for systematically evaluating action combinations.
    """
    ints = torch.arange(1, 2**N, device=device)

    # convert to bits
    bits = ((ints[:, None] >> torch.arange(N, device=device)) & 1).to(torch.long)

    return bits

def one_hot_bit_sequences(N, device=None):
    """
    Enumerates all one-hot bit sequences of N bits.
    """
    return torch.eye(N, device=device)

def get_sparse_signed_pairs_projection(
    input_size: int,
    hidden_size: int,
    device=None,
    dtype=torch.float32,
    eps: float = 1e-8,
    requires_grad: bool = False,
) -> torch.Tensor:

    W = torch.randn(
        hidden_size // 2, 
        input_size, 
        device=device, 
        dtype=dtype,
        requires_grad = requires_grad,
    )

    # Separate into +ve and -ve parts
    W_pos = torch.relu(W)
    W_neg = torch.relu(-W)

    projection = torch.cat([W_pos, W_neg], dim=0)
    projection = projection / (projection.norm(dim=1, keepdim=True) + eps)
    return projection

def get_ensemble_sparse_signed_pairs_projection(
    num_ensembles: int,
    input_size: int,
    hidden_size: int,
    device=None,
    dtype=torch.float32,
    eps: float = 1e-8,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Returns a projection tensor with shape [num_ensembles, memory_size, input_size]
    Each ensemble is independent (for ensemble hashing)
    """
    projections = []
    for _ in range(num_ensembles):
        projection = get_sparse_signed_pairs_projection(
            input_size = input_size,
            hidden_size = hidden_size,
            device = device,
            dtype = dtype,
            eps = eps,
            requires_grad = requires_grad,
        )
        projections.append(projection)

    # [num_ensembles, memory_size, input_size]
    projection_tensor = torch.stack(projections, dim=0) #.contiguous()
    return projection_tensor
