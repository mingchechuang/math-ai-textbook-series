<<<PATCH 01>>>
<<<OLD>>>
$M^{(s)}_{bi}=1$ 表示該位置有效。本章使用位置級遮罩；
<<<NEW>>>
$M^{(s)}_{bi}=1$ 表示該位置有效。數學上以二值表示；程式要求 `torch.bool`，其中 `True` 表示有效，整數遮罩須先驗證並轉換。本章使用位置級遮罩；
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    if mask.shape != times.shape:
        raise ValueError("mask須為(B,T)")
<<<NEW>>>
    if mask.shape != times.shape:
        raise ValueError("mask須為(B,T)")
    if mask.dtype != torch.bool:
        raise ValueError("mask須為torch.bool")
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    for sigma, window in (
        (float("nan"), 1.0), (float("inf"), 1.0), (0.0, 1.0),
<<<NEW>>>
    expect_value_error(
        lambda: aligned_pool(
            h, times, torch.tensor([[1]], dtype=torch.int64), tau, 1.0, 1.0
        ),
        "mask須為torch.bool",
    )
    for sigma, window in (
        (float("nan"), 1.0), (float("inf"), 1.0), (0.0, 1.0),
<<<END>>>