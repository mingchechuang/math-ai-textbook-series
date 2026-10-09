<<<PATCH 01>>>
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
    # 最近位置的能量依定義為零，避免直接計算0 * inf。
    nearest = allowed & (abs_delta == safe_min)
    safe_left = torch.where(nearest, torch.zeros_like(left), left)
    safe_right = torch.where(nearest, torch.zeros_like(right), right)
    shifted_energy = 0.5 * safe_left * safe_right
    shifted_energy = torch.where(
        allowed, shifted_energy, torch.full_like(shifted_energy, float("inf"))
    )
    if torch.isnan(shifted_energy).any():
        raise FloatingPointError("時間權重能量出現NaN")

    # 非最近位置的+inf能量對應零權重。
    raw64 = torch.exp(-shifted_energy)
    denom64 = raw64.sum(dim=1, keepdim=True)
    if torch.any(valid[:, None] & (
        ~torch.isfinite(denom64) | (denom64 <= 0)
    )):
        raise FloatingPointError("穩定時間權重分母無效")
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    if not torch.allclose(pooled, torch.tensor([[2.0]])):
        raise AssertionError("極尖銳權重未選中最近位置")
<<<NEW>>>
    if not torch.allclose(pooled, torch.tensor([[2.0]])):
        raise AssertionError("極尖銳權重未選中最近位置")

    # 有限輸入使非最近位置的尺度溢位；最近位置仍應取得權重1。
    far_times = torch.tensor([[0.0, 1e300]], dtype=torch.float64)
    far_tau = torch.tensor([0.0], dtype=torch.float64)
    far_pool, far_weight, far_valid, far_count = aligned_pool(
        h, far_times, mask, far_tau, sigma=1e-300, window=1e301
    )
    if not bool(far_valid[0]) or int(far_count[0]) != 2:
        raise AssertionError("有限極端時間的允許遮罩錯誤")
    if not torch.isfinite(far_weight).all():
        raise AssertionError("有限極端時間的權重含非有限值")
    if not torch.allclose(far_weight, torch.tensor([[1.0, 0.0]])):
        raise AssertionError("有限極端時間的權重錯誤")
    if not torch.allclose(far_pool, torch.tensor([[2.0]])):
        raise AssertionError("有限極端時間的池化結果錯誤")
<<<END>>>