<<<PATCH 27>>>
<<<OLD>>>
    if mask.shape != times.shape:
        raise ValueError("mask須為(B,T)")
    if tau.shape != (h.shape[0],):
<<<NEW>>>
    if mask.shape != times.shape:
        raise ValueError("mask須為(B,T)")
    if mask.dtype != torch.bool:
        raise ValueError("mask須為torch.bool")
    if tau.shape != (h.shape[0],):
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
    h = torch.tensor([[[1.0]]], dtype=DTYPE)
    times = torch.tensor([[0.0]], dtype=DTYPE)
    mask = torch.tensor([[True]])
    tau = torch.tensor([0.0], dtype=DTYPE)
    for sigma, window in (
<<<NEW>>>
    h = torch.tensor([[[1.0]]], dtype=DTYPE)
    times = torch.tensor([[0.0]], dtype=DTYPE)
    mask = torch.tensor([[True]])
    tau = torch.tensor([0.0], dtype=DTYPE)
    expect_value_error(
        lambda: aligned_pool(
            h, times, torch.tensor([[1]], dtype=torch.int64), tau, 1.0, 1.0
        ),
        "mask須為torch.bool",
    )
    for sigma, window in (
<<<END>>>