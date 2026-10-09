<<<PATCH 14>>>
<<<OLD>>>
*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*
<<<NEW>>>
*注意：梯度形狀必須與輸入 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*此處轉置只交換每個批次的最後兩軸；NumPy 三維陣列的 `.T` 會反轉全部軸，所以程式使用 `np.swapaxes`。

固定一個批次，以 $G_Y$ 記上游梯度，寫 $dL=\operatorname{tr}(G_Y^TdY)$。由 $dY=(dA)V+A(dV)$，分別收集微分係數，得到 $G_A=G_YV^T$、$G_V=A^TG_Y$。後一式沿查詢軸累加同一鍵的梯度；若損失取批次平均，其因子應已包含在 $G_Y$，不再重複除以批次數。

固定一個查詢列，Softmax 微分為 $da_j=a_j(ds_j-\sum_l a_l ds_l)$。代入 $dL=\sum_jg_jda_j$ 並收集各個 $ds_j$，得到 $g_{s,j}=a_j(g_j-\sum_l a_lg_l)$。因 $\sum_j a_j=1$，沿鍵軸求和可得 $\sum_jg_{s,j}=0$；這也反映了整列分數共同平移不改變 Softmax。手算例題的小數加總只是近似核對。硬遮罩位置在 Softmax 前被設為負無限，其權重與分數梯度均為零；全遮罩列沒有可歸一化的分布，應由前向拒絕。

最後把 $dS=((dQ)K^T+Q(dK)^T)/\sqrt{d_k}$ 代入 $dL=\operatorname{tr}(G_S^TdS)$，分別收集 $dQ,dK$，得到 $G_Q=G_SK/\sqrt{d_k}$ 與 $G_K=G_S^TQ/\sqrt{d_k}$。若投影權重被多條路徑共用，參數梯度仍須累加。方差命題的獨立性可能在訓練後不成立；縮放降低僅由維度增長造成權重過度尖銳的風險，不保證任意模型都不飽和。
<<<END>>>

<<<PATCH 14>>>
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

<<<PATCH 14>>>
<<<OLD>>>
    print("All tests passed.")
<<<NEW>>>
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
    assert np.allclose(c["attn_weights"].sum(axis=-1), 1)
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

<<<PATCH 14>>>
<<<OLD>>>
3.  **程式**：修改 `scaled_dot_product_attention`，使其支持 `mask` 為浮點數矩陣（例如 0 或 -1e9），並驗證梯度。
4.  **整合**：假設 $T_q = 10, T_k = 10, d_k = 64$。若 $Q, K$ 分量为 $N(0,1)$，估計 $QK^T$ 元素的分佈。若不縮放，Softmax 輸出會呈現什麼特徵？
<<<NEW>>>
3.  **程式**：另加與分數同形狀的有限浮點 `attn_bias`，保留布林 `mask` 為硬遮罩；寫出輸入檢查及固定偏置時的梯度驗證。
4.  **整合**：假設 $T_q=10,T_k=10,d_k=64$，且 $Q,K$ 分量相互獨立、服從 $N(0,1)$。估計 $QK^T$ 單一元素的均值與方差，區分傳向分數與 Value 的梯度。
5.  **反例**：兩個鍵的 Value 完全相同時，構造兩組不同的合法權重，檢驗「較大權重必然代表較大最終貢獻」。
<<<END>>>

<<<PATCH 14>>>
<<<OLD>>>
3.  **解**：
    程式碼修改：
    ```python
    if mask is not None:
        if not np.issubdtype(mask.dtype, np.floating):
             raise TypeError("Float mask expected")
        # 假設 mask 浮點數中，負值極小表示遮罩
        # 直接相加
        scores = scores + mask
        # 需確保 mask 中沒有使所有 logit 變 -inf 的情況
    ```
    梯度驗證邏輯不變。

4.  **解**：
    $QK^T$ 元素近似 $N(0, d_k) = N(0, 64)$。Std $= 8$。
    若不縮放，Softmax 輸入差異很大。相比縮放後，Softmax 分佈會更尖銳（Spiky），大部分權重趨近於 0，最大權重趨近於 1，導致梯度在大部分方向上為零，僅在最大分數對應的 Key 上有梯度。這會增加訓練的不穩定性。
<<<NEW>>>
3.  **解**：保留原布林硬遮罩及全遮罩拒絕，在函式簽名增加 `attn_bias=None`。算出 `scores` 後、處理布林遮罩前插入：
    ```python
    if attn_bias is not None:
        if not isinstance(attn_bias, np.ndarray):
            raise TypeError("bias must be a NumPy array")
        if attn_bias.shape != scores.shape:
            raise ValueError("bias shape mismatch")
        if not np.issubdtype(attn_bias.dtype, np.floating):
            raise TypeError("bias must be floating")
        if not np.all(np.isfinite(attn_bias)):
            raise ValueError("bias must be finite")
        scores = scores + attn_bias
    ```
    有限偏置不是硬遮罩。偏置固定時三路反傳公式不變；取非方形的有限輸入與非均勻偏置，每次擾動一個輸入元素，以同一偏置重算輸出總和，再以中央差分逐項核對全一上游梯度所得的三路梯度。若偏置本身可訓練，其梯度為加入偏置後分數的梯度。

4.  **解**：單一內積均值為零、方差為六十四，標準差為八；常態只是近似分布。未縮放時權重可能更尖銳。最大權重趨近一時，其自身 Softmax 導數 $a_m(1-a_m)$ 也趨近零，不能說僅最大鍵仍有分數梯度。傳向 $Q,K$ 的部分梯度可能變弱；$dV=A^TdY$ 則可能主要流向高權重位置，具體值取決於上游梯度。

5.  **解**：令兩個 Value 均為 $[2,3]$，權重分別取 $(0.9,0.1)$ 與 $(0.1,0.9)$。兩次輸出都為 $[2,3]$，故權重變化不必然改變輸出，更不能單由權重判定最終貢獻。
<<<END>>>