<<<PATCH 01>>>
<<<OLD>>>
注意力機制解決了序列模型中長距離依賴的問題。直觀上，每個 Query 向量 $q_i$ 會與所有 Key 向量 $k_j$ 計算相關性分數，並根據這些分數對 Value 向量 $v_j$ 進行加權求和。
<<<NEW>>>
注意力讓遠距位置有直接的計算路徑，可緩解部分序列模型逐步傳遞資訊的困難，但不保證模型一定學得長距依賴。每個 Query 向量 $q_i$ 與允許注意的 Key 向量 $k_j$ 計算分數，再依權重混合對應的 Value 向量 $v_j$。Query 與 Key 決定混合係數，Value 決定混合內容；即使兩個查詢的權重相同，其 Value 不同也可能產生不同輸出。反過來，若所有 Value 相同，重新分配總和為一的權重不會改變輸出。因此權重既不是最終預測貢獻的直接量度，也不是因果證據。
<<<END>>>

<<<PATCH 02>>>
<<<OLD>>>
*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*
<<<NEW>>>
*注意：梯度形狀必須與輸入 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*上列三維張量的轉置均指逐批次交換最後兩軸；NumPy 三維陣列的 `.T` 會反轉全部軸，所以程式使用 `np.swapaxes`。

可以用微分檢查整條反傳鏈，而不只靠形狀猜公式。固定一個批次，令上游梯度為 $G_Y$，寫成 $dL=\operatorname{tr}(G_Y^TdY)$。由 $dY=(dA)V+A(dV)$，分別收集 $dA$ 和 $dV$ 的係數，便得到 $G_A=G_YV^T$ 及 $G_V=A^TG_Y$。後一式會把多個查詢讀取同一鍵位置所造成的梯度相加。若損失已對批次取平均，平均因子應先包含於 $G_Y$，不應在每一步反傳重複除以批次數。

固定一個查詢列，記其 Softmax 權重為 $a_j$。微小分數變動滿足 $da_j=a_j(ds_j-\sum_l a_l ds_l)$。將它代入 $dL=\sum_jg_jda_j$，依各個 $ds_j$ 收集係數，得到 $g_{s,j}=a_j(g_j-\sum_l a_lg_l)$。沿鍵軸求和並利用 $\sum_j a_j=1$，可得 $\sum_jg_{s,j}=0$。這是因為對同一列分數同加常數不改變 Softmax；手算例題中近似小數的和為零，只是此結構性結論的數值核對。若某鍵在 Softmax 前被硬遮罩，該處權重為零，分數梯度亦為零；若整列都被遮罩，就沒有可歸一化的分布，必須明確拒絕。

最後，由 $dS=((dQ)K^T+Q(dK)^T)/\sqrt{d_k}$，代入 $dL=\operatorname{tr}(G_S^TdS)$，分別收集 $dQ$、$dK$，便得到 $G_Q=G_SK/\sqrt{d_k}$、$G_K=G_S^TQ/\sqrt{d_k}$。這些是輸入張量的梯度；若三路投影使用共享參數，參數梯度還須在各路回傳後累加。方差命題亦有邊界：訓練後分量可能相關，實際分數方差不一定為一。縮放是降低僅因維度增加造成權重過度尖銳的風險，而非保證任意模型都不會飽和。有限差分只核對指定輸入附近的實作，不能取代這些推導。
<<<END>>>

<<<PATCH 03>>>
<<<OLD>>>
    # 1. Shape 驗證
    if Q.shape[-1] != K.shape[-1]:
<<<NEW>>>
    # 1. Shape 驗證
    if not all(isinstance(x, np.ndarray) for x in (Q, K, V)):
        raise TypeError("Q, K, V must be NumPy arrays")
    if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
        raise ValueError("Q, K, V must have at least two dimensions")
    if Q.shape[-2] == 0 or V.shape[-1] == 0:
        raise ValueError("Tq and dv must be positive")
    if Q.shape[-1] != K.shape[-1]:
<<<END>>>

<<<PATCH 04>>>
<<<OLD>>>
    print("All tests passed.")
<<<NEW>>>
    # 非方形長度及合法的部分遮罩
    rng = np.random.default_rng(14)
    q = rng.standard_normal((1, 5, 2))
    k = rng.standard_normal((1, 3, 2))
    v = rng.standard_normal((1, 3, 4))
    allowed = np.array([[[True, False, True],
                         [False, True, True],
                         [True, True, False],
                         [True, False, False],
                         [False, True, False]]])
    y, c = scaled_dot_product_attention(q, k, v, allowed)
    assert y.shape == (1, 5, 4)
    assert c["attn_weights"].shape == (1, 5, 3)
    assert np.all(c["attn_weights"][~allowed] == 0)
    assert np.allclose(np.sum(c["attn_weights"], axis=-1), 1)
    gq, gk, gv = scaled_dot_product_attention_bwd(np.ones_like(y), c)
    assert (gq.shape, gk.shape, gv.shape) == (q.shape, k.shape, v.shape)
    assert max(finite_diff_check(q, k, v, allowed)) < 1e-5

    def rejects(call, error):
        try:
            call()
        except error:
            return
        raise AssertionError("expected rejection")

    rejects(lambda: scaled_dot_product_attention(q[0, 0], k, v), ValueError)
    rejects(lambda: scaled_dot_product_attention([1.0], k, v), TypeError)
    rejects(lambda: scaled_dot_product_attention(q, k[..., :1], v), ValueError)
    rejects(lambda: scaled_dot_product_attention(q, k, v[:, :2]), ValueError)
    rejects(lambda: scaled_dot_product_attention(
        q, k, v, np.ones((1, 5, 2), dtype=bool)), ValueError)
    rejects(lambda: scaled_dot_product_attention(
        q, k, v, np.ones((1, 5, 3))), TypeError)
    rejects(lambda: scaled_dot_product_attention(q * np.nan, k, v), ValueError)
    for bad_scale in (0, -1, np.nan):
        rejects(lambda s=bad_scale: scaled_dot_product_attention(
            q, k, v, scale_factor=s), ValueError)
    rejects(lambda: scaled_dot_product_attention_bwd(
        np.ones((1, 4, 4)), c), ValueError)
    print("All tests passed.")

if __name__ == "__main__":
    run_tests()
<<<END>>>

<<<PATCH 05>>>
<<<OLD>>>
3.  **程式**：修改 `scaled_dot_product_attention`，使其支持 `mask` 為浮點數矩陣（例如 0 或 -1e9），並驗證梯度。
4.  **整合**：假設 $T_q = 10, T_k = 10, d_k = 64$。若 $Q, K$ 分量为 $N(0,1)$，估計 $QK^T$ 元素的分佈。若不縮放，Softmax 輸出會呈現什麼特徵？
<<<NEW>>>
3.  **程式**：另加與分數同形狀的有限浮點 `attn_bias`，保留布林 `mask` 為硬遮罩；寫出輸入檢查及固定偏置時的梯度驗證。
4.  **整合**：假設 $T_q=10,T_k=10,d_k=64$，且 $Q,K$ 分量相互獨立、服從 $N(0,1)$。估計 $QK^T$ 單一元素的均值與方差，並區分傳向分數與 Value 的梯度。
5.  **反例**：令兩個鍵的 Value 相同，構造兩組不同的合法注意力權重，檢驗「較大權重必然代表較大最終預測貢獻」。
<<<END>>>