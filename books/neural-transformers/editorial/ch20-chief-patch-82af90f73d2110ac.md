<<<PATCH 20>>>
<<<OLD>>>
    bad_logits = np.array([[np.nan, 0.0]])
    expect_value_error(
        lambda: (_ for _ in ()).throw(
            ValueError("logits 含非有限值")
        ) if not np.isfinite(bad_logits).all() else None,
        "非有限值",
    )
<<<NEW>>>
    # 注入 NaN 到輸出偏置，驗證模型自身的非有限 logits 檢查。
    saved_bout = model.p["bout"].copy()
    model.p["bout"][0] = np.nan
    expect_value_error(
        lambda: model.loss_and_grads(ids, targets, mask),
        "非有限值",
    )
    model.p["bout"] = saved_bout
<<<END>>>