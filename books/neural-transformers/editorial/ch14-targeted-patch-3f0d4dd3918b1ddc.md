<<<PATCH 01>>>
<<<OLD>>>
*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*
<<<NEW>>>
*注意：梯度形狀必須與輸入 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*此處各三維張量的轉置均指逐批次交換最後兩軸；NumPy 三維陣列的 `.T` 會反轉所有軸，因此程式使用 `np.swapaxes`。

以微分核對反傳。固定一個批次，令上游梯度為 $G_Y$，採用 $dL=\operatorname{tr}(G_Y^TdY)$。因 $dY=(dA)V+A(dV)$，分別收集兩項的微分係數，可得 $G_A=G_YV^T$ 及 $G_V=A^TG_Y$。後一式沿查詢軸累加同一鍵位置的梯度；多個查詢讀取同一鍵時，不得只取其中一個。若損失先對批次取平均，平均因子應已包含在 $G_Y$，後續各路不再重複除以批次數。

固定一個查詢列，以 $a_j$ 表示其權重。由 Softmax 微分可得 $da_j=a_j(ds_j-\sum_l a_l ds_l)$。代入 $dL=\sum_jg_jda_j$，按各個 $ds_j$ 收集係數，得到 $g_{s,j}=a_j(g_j-\sum_l a_lg_l)$。沿鍵軸求和並利用 $\sum_j a_j=1$，即得 $\sum_jg_{s,j}=0$。這也符合 Softmax 對整列分數共同平移不變的性質；手算中四捨五入的小數和只是近似核對。布林硬遮罩先於 Softmax 把禁止位置設為負無限；只要該列有允許鍵，禁止位置的權重及其分數梯度均為零。若整列禁止，就沒有可歸一化的分布，必須在前向拒絕。

最後，固定縮放因子時，$dS=((dQ)K^T+Q(dK)^T)/\sqrt{d_k}$。代入 $dL=\operatorname{tr}(G_S^TdS)$，逐項收集 $dQ$、$dK$，得到 $G_Q=G_SK/\sqrt{d_k}$、$G_K=G_S^TQ/\sqrt{d_k}$。這些是輸入張量的梯度；若多條路徑共用投影參數，仍須在參數處累加各路梯度。有限差分只核對選定輸入附近的實作，不能代替上述鏈式法則證明。

方差命題亦有邊界：它要求相應乘積之間足夠獨立。訓練後的分量可能相關，實際分數方差未必為一。縮放降低僅由維度增加造成權重過度尖銳的風險，並非保證所有資料上的 Softmax 都不飽和；而且 Value 的梯度另有直接路徑，不能籠統說注意力各路梯度都會消失。
<<<END>>>

<<<PATCH 02>>>
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

<<<PATCH 03>>>
<<<OLD>>>
    print("All tests passed.")
<<<NEW>>>
    # 非方形序列、部分遮罩與三路梯度
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

<<<PATCH 04>>>
<<<OLD>>>
3.  **程式**：修改 `scaled_dot_product_attention`，使其支持 `mask` 為浮點數矩陣（例如 0 或 -1e9），並驗證梯度。
4.  **整合**：假設 $T_q = 10, T_k = 10, d_k = 64$。若 $Q, K$ 分量为 $N(0,1)$，估計 $QK^T$ 元素的分佈。若不縮放，Softmax 輸出會呈現什麼特徵？
<<<NEW>>>
3.  **程式**：另加與分數同形狀的有限浮點 `attn_bias`，保留布林 `mask` 作硬遮罩；寫出檢查與固定偏置時的梯度驗證。
4.  **整合**：假設 $T_q=10,T_k=10,d_k=64$，且 $Q,K$ 分量相互獨立、服從 $N(0,1)$。估計 $QK^T$ 單一元素的均值與方差，區分傳向分數與 Value 的梯度。
5.  **反例**：兩個鍵的 Value 完全相同時，構造兩組不同的合法權重，檢驗「權重越大，對最終輸出的貢獻必越大」。
<<<END>>>

<<<PATCH 05>>>
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
3.  **解**：保留原布林硬遮罩與全遮罩拒絕，在函式簽名增加 `attn_bias=None`。算出 `scores` 後、處理布林遮罩前插入：
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
    此偏置不是硬遮罩；有限的巨大負數不保證禁止位置權重精確為零。偏置固定時，三路反傳公式不變。驗證可選有限、非方形的 $Q,K,V$ 和非均勻偏置，每次只擾動一個輸入元素，以相同偏置重新算輸出元素總和，再以中央差分逐項核對全一上游梯度所得的 $dQ,dK,dV$；布林全遮罩拒絕仍須另測。若偏置本身可訓練，它的梯度是對加入偏置後分數的梯度。

4.  **解**：單一內積均值為零、方差為六十四，標準差為八；常態只是有限維近似，並非精確分布。未縮放時權重可能更尖銳。最大權重趨近一時，其自身 Softmax 導數 $a_m(1-a_m)$ 也趨近零，不能說僅最大鍵仍有分數梯度。傳向 $Q,K$ 的部分梯度可能變弱；$dV=A^TdY$ 則可能主要流向高權重位置，具體值取決於上游梯度。

5.  **解**：令兩個 Value 均為 $[2,3]$，兩組權重分別為 $(0.9,0.1)$ 與 $(0.1,0.9)$。兩組沿鍵軸的和均為一，兩次加權輸出亦均為 $[2,3]$。權重不同而輸出相同，故單憑權重不能判定最終貢獻。
<<<END>>>