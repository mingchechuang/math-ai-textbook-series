<<<PATCH 01>>>
<<<OLD>>>
$T_s$ 是時間位置數，$D_s$ 是每筆感測特徵數。$M^{(s)}_{bi}=1$ 表示該位置有效。本章使用位置級遮罩；若只有部分特徵缺失，應改用 $(B,T_s,D_s)$ 的特徵級遮罩。
<<<NEW>>>
$T_s$ 是時間位置數，$D_s$ 是每筆感測特徵數。$M^{(s)}_{bi}=1$ 表示該位置有效；數學上以二值表示，程式輸入則要求 `torch.bool`，其中 `True` 表示有效，整數 0／1 張量須先明確轉換。本章使用位置級遮罩；若只有部分特徵缺失，應改用 $(B,T_s,D_s)$ 的特徵級遮罩。
<<<END>>>
<<<PATCH 02>>>
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
<<<PATCH 03>>>
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