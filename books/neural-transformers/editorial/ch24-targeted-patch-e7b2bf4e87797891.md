<<<PATCH 01>>>
<<<OLD>>>
令箱貢獻 $Y_b$ 在 $N_b=0$ 時為 $0$，在 $N_b>0$ 時為 $\\frac{N_b}{N}|X_b|$。這使 $Y_b$ 在整個樣本空間均有定義，不必計算空箱中未定義的 $X_b$。只對 $n=1,\\ldots,N$ 求和：
<<<NEW>>>
定義在空箱時也有值的箱貢獻：
$$
Y_b=
\\begin{cases}
\\frac{N_b}{N}|\\text{Acc}_b-\\bar p_b|,&N_b>0,\\\\
0,&N_b=0.
\\end{cases}
$$
以下條件期望中的 $X_b=\\text{Acc}_b-\\bar p_b$ 僅在 $N_b>0$ 時使用，只對 $n=1,\\ldots,N$ 求和：
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
**Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau) = \\frac{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(\\hat{y}_i \\neq y_i)}{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(s(x_i) \\ge \\tau)} $$
<<<NEW>>>
定義接受指標 $a_i(\\tau)=\\mathbb{I}(s(x_i)\\ge\\tau)$。
*   **Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau)=\\frac{\\sum_{i=1}^{N}a_i(\\tau)\\mathbb{I}(\\hat{y}_i\\neq y_i)}{\\sum_{i=1}^{N}a_i(\\tau)} $$
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    mean_nll = total_nll / num_valid
    # 明確拒絕不可表示的 PPL；mean_nll 仍可供呼叫端另行記錄。
    if not np.isfinite(mean_nll) or mean_nll > np.log(np.finfo(float).max):
        raise OverflowError("PPL not representable; record mean NLL instead.")
    perplexity = np.exp(mean_nll)
    return perplexity, mean_nll
<<<NEW>>>
    mean_nll = total_nll / num_valid
    # 不可表示時拋錯；呼叫端若需保留 mean_nll，須另行計算或另設回傳介面。
    if not np.isfinite(mean_nll) or mean_nll > np.log(np.finfo(float).max):
        raise OverflowError("PPL not representable.")
    perplexity = np.exp(mean_nll)
    return perplexity, mean_nll
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")

def test_normal():
<<<NEW>>>
    expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")

def expect_error(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError("Expected exception")

def test_normal():
<<<END>>>
<<<PATCH 05>>>
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
<<<PATCH 06>>>
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