<<<PATCH 24>>>
<<<OLD>>>
    # 明確拒絕不可表示的 PPL；mean_nll 仍可供呼叫端另行記錄。
    if not np.isfinite(mean_nll) or mean_nll > np.log(np.finfo(float).max):
        raise OverflowError("PPL not representable; record mean NLL instead.")
    perplexity = np.exp(mean_nll)
    return perplexity, mean_nll
<<<NEW>>>
    if not np.isfinite(mean_nll):
        raise OverflowError("Mean NLL is not finite.")
    # PPL 超出 float64 範圍時仍回傳可記錄的 Mean NLL。
    if mean_nll > np.log(np.finfo(float).max):
        return np.inf, float(mean_nll)
    perplexity = np.exp(mean_nll)
    return perplexity, mean_nll
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    assert select_threshold(np.array([.8, .6]), np.array([1, 0]),
                            [0, .8, 1], .5) == .8
<<<NEW>>>
    assert select_threshold(np.array([.8, .6]), np.array([1, 0]),
                            [0, .8, 1], .5) == .8
    vp = result["selection"]["validation_point"]
    assert vp[1:3] == (1, 0)
    assert np.isclose(vp[3], .5) and np.isclose(vp[4], 0)
    id_point = result["id", "all"]["selected_point"]
    ood_point = result["ood", "all"]["selected_point"]
    assert id_point[1:3] == (2, 0)
    assert np.isclose(id_point[3], 2 / 3) and np.isclose(id_point[4], 0)
    assert ood_point[1:3] == (1, 1)
    assert np.isclose(ood_point[3], 1 / 3) and np.isclose(ood_point[4], 1)
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    assert np.isfinite(compute_global_ppl(z, np.array([[0]]),
                                           np.array([[True]]))[0])
<<<NEW>>>
    assert np.isfinite(compute_global_ppl(z, np.array([[0]]),
                                           np.array([[True]]))[0])
    overflow_ppl, overflow_mean = compute_global_ppl(
        z, np.array([[1]]), np.array([[True]]))
    assert np.isinf(overflow_ppl) and np.isclose(overflow_mean, 1000.)
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
        macro_ece = np.mean(eces)
        return eces, macro_ece
    ```
<<<NEW>>>
        macro_ece = np.mean(eces)
        return eces, macro_ece

    # 以下是預期測試，未執行；沿用本章前述 expect_value_error。
    eces, macro = compute_classwise_ece(
        [[.8, .2], [.3, .7]], [0, 1], [0, .5, 1])
    assert len(eces) == 2 and np.isclose(macro, np.mean(eces))
    expect_value_error(compute_classwise_ece,
                       np.empty((0, 2)), np.empty((0,), dtype=int), [0, 1])
    expect_value_error(compute_classwise_ece,
                       np.empty((2, 0)), np.array([0, 0]), [0, 1])
    expect_value_error(compute_classwise_ece, [[.8, .3]], [0], [0, 1])
    expect_value_error(compute_classwise_ece, [[.8, .2]], [2], [0, 1])
    expect_value_error(compute_classwise_ece, [[np.nan, .2]], [0], [0, 1])
    ```
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    *   這表明模型對新異常模式缺乏泛化能力。
<<<NEW>>>
    *   這表示模型與告警群的 token 分布匹配較差；是否屬於語義泛化失敗，仍須控制詞彙、模板、長度、tokenizer、上下文與標註規則。
<<<END>>>