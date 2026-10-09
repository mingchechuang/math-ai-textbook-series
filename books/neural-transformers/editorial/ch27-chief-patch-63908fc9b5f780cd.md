<<<PATCH 01>>>
<<<OLD>>>
    if sigma <= 0 or window < 0:
        raise ValueError("sigma須為正且window須為非負")
<<<NEW>>>
    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma須為有限正純量")
    if not np.isscalar(window) or not np.isfinite(window) or window < 0:
        raise ValueError("window須為有限非負純量")
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    if temperature <= 0:
        raise ValueError("temperature須為正")
<<<NEW>>>
    if not np.isscalar(temperature) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature須為有限正純量")
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    all_missing = clone_batch(batch)
    all_missing["ms"][:] = False
    expect_value_error(
        lambda: model.loss(all_missing),
        "有效配對少於兩筆",
    )

    test_extreme_finite_time_scale()
    return report
<<<NEW>>>
    all_missing = clone_batch(batch)
    all_missing["ms"][:] = False
    expect_value_error(
        lambda: model.loss(all_missing),
        "有效配對少於兩筆",
    )

    # 只確認拋出ValueError，不依賴特定訊息文字。
    def expect_any_value_error(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("預期ValueError但未拋出")

    h = torch.zeros(1, 1, 1, dtype=DTYPE)
    times = torch.zeros(1, 1, dtype=DTYPE)
    mask = torch.ones(1, 1, dtype=torch.bool)
    tau = torch.zeros(1, dtype=DTYPE)
    for bad_sigma in (float("nan"), float("inf"), 0.0, -1.0):
        expect_any_value_error(
            lambda bs=bad_sigma: aligned_pool(
                h, times, mask, tau, bs, 1.0
            )
        )
    for bad_window in (float("nan"), float("inf"), -1.0):
        expect_any_value_error(
            lambda bw=bad_window: aligned_pool(
                h, times, mask, tau, 1.0, bw
            )
        )

    q = torch.ones(2, D, dtype=DTYPE) / np.sqrt(D)
    valid = torch.ones(2, dtype=torch.bool)
    for bad_temp in (float("nan"), float("inf"), 0.0, -1.0):
        expect_any_value_error(
            lambda bt=bad_temp: pair_loss(q, q, valid, valid, bt)
        )

    test_extreme_finite_time_scale()
    return report
<<<END>>>