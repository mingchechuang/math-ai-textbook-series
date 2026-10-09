<<<PATCH 27>>>
<<<OLD>>>
    if sigma <= 0 or window < 0:
        raise ValueError("sigma須為正且window須為非負")
<<<NEW>>>
    if not np.isfinite(sigma) or not np.isfinite(window) or sigma <= 0 or window < 0:
        raise ValueError("sigma須為有限正數且window須為有限非負數")
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
    if temperature <= 0:
        raise ValueError("temperature須為正")
<<<NEW>>>
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature須為有限正數")
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
    test_extreme_finite_time_scale()
    return report
<<<NEW>>>
    test_extreme_finite_time_scale()

    h = torch.tensor([[[1.0]]], dtype=DTYPE)
    times = torch.tensor([[0.0]], dtype=DTYPE)
    mask = torch.tensor([[True]])
    tau = torch.tensor([0.0], dtype=DTYPE)
    for sigma, window in (
        (float("nan"), 1.0), (float("inf"), 1.0), (0.0, 1.0),
        (1.0, float("nan")), (1.0, float("inf")), (1.0, -1.0),
    ):
        expect_value_error(
            lambda sigma=sigma, window=window: aligned_pool(
                h, times, mask, tau, sigma, window
            ),
            "sigma須為有限正數且window須為有限非負數",
        )

    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    valid = torch.tensor([True, True])
    for temperature in (float("nan"), float("inf"), 0.0, -1.0):
        expect_value_error(
            lambda temperature=temperature: pair_loss(
                a, a, valid, valid, temperature
            ),
            "temperature須為有限正數",
        )
    return report
<<<END>>>