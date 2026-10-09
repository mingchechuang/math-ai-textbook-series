<<<PATCH 14>>>
<<<OLD>>>
注意力機制解決了序列模型中長距離依賴的問題。直觀上，每個 Query 向量 $q_i$ 會與所有 Key 向量 $k_j$ 計算相關性分數，並根據這些分數對 Value 向量 $v_j$ 進行加權求和。

若直接計算內積 $q_i \cdot k_j$，當維度 $d_k$ 增大時，內積的數值範圍也會隨之擴大。這會導致 Softmax 函數進入飽和區（Saturation Region），使得梯度極小，模型難以學習。因此，我們需要一個縮放因子來規範分數的尺度。
<<<NEW>>>
注意力提供遠距位置之間的直接計算路徑，可緩解部分序列模型逐步傳遞資訊的困難，但不保證模型一定學得長距依賴。每個 Query 向量 $q_i$ 與允許注意的 Key 向量 $k_j$ 計算分數，再依分數所形成的權重混合對應的 Value 向量 $v_j$。Query 與 Key 決定混合係數，Value 決定混合的內容；若所有 Value 相同，改變權重而維持每列權重和為一，輸出仍然相同。因此權重不是最終預測貢獻的直接量度。

在下述獨立、零均值、單位方差假設下，未縮放內積的標準差隨維度的平方根增長。較分散的分數可能使 Softmax 權重過於尖銳，傳向分數的部分梯度因而變弱；這是縮放的動機，不是對所有資料或訓練狀態的保證。Value 的梯度還有不經過 Softmax Jacobian 的路徑，不能籠統地說所有梯度都會消失。這個區別在解讀手算梯度與習題中的反例時尤其重要。
<<<END>>>

<<<PATCH 14>>>
<<<OLD>>>
*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*
<<<NEW>>>
*注意：梯度形狀必須與輸入 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*三維張量公式中的轉置都指逐批次交換最後兩軸。NumPy 三維陣列的 `.T` 會反轉全部軸，因此程式使用 `np.swapaxes`。

為何反向公式不只是形狀上的猜測？固定一個批次，以上游梯度 $G_Y$ 寫 $dL=\operatorname{tr}(G_Y^TdY)$。由 $dY=(dA)V+A(dV)$，分別收集 $dA$ 與 $dV$ 的係數，即得到 $G_A=G_YV^T$、$G_V=A^TG_Y$。後一式沿查詢軸累加同一鍵位置收到的梯度；多個查詢使用同一鍵時，不可只保留最後一個查詢。若損失已對批次取平均，平均因子應包含於 $G_Y$，不應在後續每條路徑再次除以批次數。

固定一個查詢列，令其 Softmax 權重為 $a_j$。微分滿足 $da_j=a_j(ds_j-\sum_l a_l ds_l)$。將它代入 $dL=\sum_jg_jda_j$，依各個 $ds_j$ 收集係數，得到 $g_{s,j}=a_j(g_j-\sum_l a_lg_l)$。沿鍵軸相加並使用 $\sum_j a_j=1$，可得 $\sum_jg_{s,j}=0$。這是整列分數共同平移不改變 Softmax 的結果；手算例題的小數和為零只是近似核對。硬遮罩在 Softmax 前把禁止位置設為負無限，只要該列仍有允許鍵，禁止位置的權重與分數梯度都是零；全遮罩列則不存在可歸一化的分布，必須明確拒絕。

最後，由 $dS=((dQ)K^T+Q(dK)^T)/\sqrt{d_k}$，代入 $dL=\operatorname{tr}(G_S^TdS)$ 並分別收集 $dQ,dK$，得到 $G_Q=G_SK/\sqrt{d_k}$ 及 $G_K=G_S^TQ/\sqrt{d_k}$。這些是輸入張量的梯度；若投影參數被多條路徑共用，參數梯度仍須累加。有限差分只核對特定輸入附近的實作，不能取代上述鏈式法則的推導。訓練後分量也可能相關，因此方差命題不保證此時分數方差仍為一。
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
    # 非對稱長度、部分硬遮罩與三路梯度
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

## 習題解答

1.  **解**：
    $QK^T = 1(1) + 0(1) = 1$。
    Scale: $1/\sqrt{2} \approx 0.707$。
    Softmax($0.707$)：單元素 Softmax 恆為 1。
    Output: $1 \times V = 1 \times [1] = [1]$。
    *(註：單個 Key 的 Softmax 恆為 1，無論分數為何)*。

2.  **解**：
    若 $Var(q_i) = \sigma^2, Var(k_i) = \sigma^2$。
    $Var(q_i k_i) = \sigma^4$。
    $Var(q \cdot k) = d_k \sigma^4$。
    Std $= \sqrt{d_k} \sigma^2$。
    縮放因子應為 $\sqrt{d_k} \sigma^2$ 以使方差為 1。
    *(若 Q, K 初始化為 $N(0, 1/\sqrt{d_k})$，則 $\sigma^2 = 1/\sqrt{d_k}$，縮放因子為 $1$)*。

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
3.  **程式**：另加與分數同形狀的有限浮點 `attn_bias`，保留布林 `mask` 為硬遮罩；寫出檢查與固定偏置時的梯度驗證。
4.  **整合**：假設 $T_q=10,T_k=10,d_k=64$，且 $Q,K$ 分量相互獨立、服從 $N(0,1)$。估計 $QK^T$ 單一元素的均值與方差，區分傳向分數與 Value 的梯度。
5.  **反例**：兩個鍵的 Value 完全相同時，構造兩組不同的合法權重，檢驗「較大權重必然代表較大最終貢獻」。

## 習題解答

1.  **解**：內積為一，縮放後為 $1/\sqrt{2}$；單元素 Softmax 為一，故輸出為 $[1]$。只有一個允許鍵時，輸出不依賴該鍵的分數。

2.  **解**：獨立與零均值條件下，每項乘積方差為 $\sigma^4$，相加得 $d_k\sigma^4$；若希望縮放後方差為一，可除以 $\sqrt{d_k}\sigma^2$。此結論不保證訓練後的分量仍獨立。

3.  **解**：保留原布林硬遮罩與全遮罩拒絕，另在函式簽名增加 `attn_bias=None`。算出 `scores` 後、處理布林遮罩前插入：
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
    有限偏置不是硬遮罩：極小但有限的值也不保證禁止位置權重精確為零。偏置固定時三路反傳公式不變。取非方形的有限輸入與非均勻偏置，每次擾動其中一個輸入元素，以相同偏置重算輸出元素總和；其中央差分應逐項核對全一上游梯度所得的三路梯度。若偏置本身可訓練，其梯度為加入偏置後分數的梯度。

4.  **解**：單一內積均值為零、方差為六十四、標準差為八；常態只是有限維近似，並非精確分布。未縮放時權重可能更尖銳。最大權重趨近一時，其自身 Softmax 導數 $a_m(1-a_m)$ 也趨近零，不能說僅最大鍵仍有分數梯度。傳向 $Q,K$ 的部分梯度可能變弱；$dV=A^TdY$ 則可能主要流向高權重位置，具體值取決於上游梯度。

5.  **解**：令兩個 Value 均為 $[2,3]$，權重分別取 $(0.9,0.1)$ 與 $(0.1,0.9)$。兩次輸出均為 $[2,3]$。權重變化不必然改變輸出，不能單憑權重判定最終貢獻。
<<<END>>>