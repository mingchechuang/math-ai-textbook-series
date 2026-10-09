<<<PATCH 27>>>
<<<OLD>>>
    shifted_energy = 0.5 * left * right
    shifted_energy = torch.where(
        allowed,
        shifted_energy,
        torch.full_like(shifted_energy, float("inf")),
    )

    # 極遠位置的shifted_energy可為+inf，其exp(-inf)=0是合法極限。
    raw64 = torch.exp(-shifted_energy)
    raw64 = torch.where(allowed, raw64, torch.zeros_like(raw64))

    denom64 = raw64.sum(dim=1, keepdim=True)
    if torch.any(valid[:, None] & (denom64 <= 0)):
        raise FloatingPointError("穩定時間權重仍出現零分母")
<<<NEW>>>
    # 最近位置直接指定能量為零，避免left=0、right=inf時的0*inf。
    nearest = allowed & (abs_delta == safe_min)
    shifted_energy = torch.where(
        nearest, torch.zeros_like(left), 0.5 * left * right
    )
    shifted_energy = torch.where(
        allowed,
        shifted_energy,
        torch.full_like(shifted_energy, float("inf")),
    )
    if torch.any(allowed & torch.isnan(shifted_energy)):
        raise FloatingPointError("允許位置的時間能量為NaN")

    # 非最近位置可有+inf能量，其exp(-inf)=0是合法極限。
    raw64 = torch.exp(-shifted_energy)
    raw64 = torch.where(allowed, raw64, torch.zeros_like(raw64))

    denom64 = raw64.sum(dim=1, keepdim=True)
    if torch.any(valid[:, None] & (~torch.isfinite(denom64) | (denom64 <= 0))):
        raise FloatingPointError("穩定時間權重分母無效")
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
    test_extreme_finite_time_scale()

    h = torch.tensor([[[1.0]]], dtype=DTYPE)
<<<NEW>>>
    test_extreme_finite_time_scale()

    # 有限float64輸入下，最近位置的d+m可能溢位，但能量仍應為零。
    large_h = torch.tensor([[[2.0]]], dtype=DTYPE)
    large_t = torch.tensor([[1e308]], dtype=torch.float64)
    large_tau = torch.tensor([0.0], dtype=torch.float64)
    large_z, large_w, large_valid, _ = aligned_pool(
        large_h, large_t, torch.tensor([[True]]), large_tau,
        sigma=2.0, window=1.7e308,
    )
    if not bool(large_valid[0]) or not torch.isfinite(large_w).all():
        raise AssertionError("有限極限的最近位置權重無效")
    if not torch.allclose(large_w, torch.ones_like(large_w)):
        raise AssertionError("單一最近位置權重應為一")
    if not torch.allclose(large_z, large_h[:, 0, :]):
        raise AssertionError("有限極限的池化結果錯誤")

    h = torch.tensor([[[1.0]]], dtype=DTYPE)
<<<END>>>