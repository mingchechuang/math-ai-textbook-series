<<<PATCH 01>>>
<<<OLD>>>
    mean_nll = total_nll / num_valid
    # 明確拒絕不可表示的 PPL；mean_nll 仍可供呼叫端另行記錄。
    if not np.isfinite(mean_nll) or mean_nll > np.log(np.finfo(float).max):
        raise OverflowError("PPL not representable; record mean NLL instead.")
<<<NEW>>>
    mean_nll = total_nll / num_valid
    # 拋出 OverflowError 時不會回傳 mean_nll；呼叫端若需記錄，須另設回傳介面。
    if not np.isfinite(mean_nll) or mean_nll > np.log(np.finfo(float).max):
        raise OverflowError("PPL not representable.")
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    assert select_threshold(np.array([.8, .6]), np.array([1, 0]),
                            [0, .8, 1], .5) == .8
<<<NEW>>>
    assert select_threshold(np.array([.8, .6]), np.array([1, 0]),
                            [0, .8, 1], .5) == .8
    assert result["selection"]["selected_tau"] == .8
    validation_point = result["selection"]["validation_point"]
    assert validation_point[1] == 1 and validation_point[2] == 0
    id_point = result["id", "all"]["selected_point"]
    ood_point = result["ood", "all"]["selected_point"]
    assert id_point[1] == 2 and id_point[2] == 0
    assert np.isclose(id_point[3], 2 / 3) and id_point[4] == 0
    assert ood_point[1] == 1 and ood_point[2] == 1
    assert np.isclose(ood_point[3], 1 / 3) and ood_point[4] == 1
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    assert np.isfinite(compute_global_ppl(z, np.array([[0]]),
                                           np.array([[True]]))[0])
<<<NEW>>>
    assert np.isfinite(compute_global_ppl(z, np.array([[0]]),
                                           np.array([[True]]))[0])
    # 極端有限 logits 下真標籤機率極小，PPL 應不可表示。
    expect_error(OverflowError, compute_global_ppl,
                 np.array([[[1000., 0.]]]), np.array([[1]]),
                 np.array([[True]]))
<<<END>>>