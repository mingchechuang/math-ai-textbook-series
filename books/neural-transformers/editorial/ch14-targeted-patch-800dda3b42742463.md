<<<PATCH 01>>>
<<<OLD>>>
注意力機制解決了序列模型中長距離依賴的問題。直觀上，每個 Query 向量 $q_i$ 會與所有 Key 向量 $k_j$ 計算相關性分數，並根據這些分數對 Value 向量 $v_j$ 進行加權求和。
<<<NEW>>>
注意力提供直接連結遠距位置的計算機制，緩解部分序列模型中資訊傳遞路徑過長的問題；它不保證模型必然學得長距依賴。直觀上，每個 Query 向量 $q_i$ 會與所有 Key 向量 $k_j$ 計算分數，並根據這些分數對 Value 向量 $v_j$ 進行加權求和。Query 與 Key 決定混合係數，Value 則決定被混合的內容：即使兩個位置獲得相同權重，若其 Value 不同，對輸出的作用也不同。反過來，若所有 Value 相同，改變權重而維持每列權重和為一，輸出仍不變。

這裡的「每個」只涵蓋允許注意的位置。自注意力通常令查詢與鍵來自同一序列，但兩者長度不必相等；例如一組較短的查詢讀取較長的記錄時，分數矩陣便是長方形。計算時應逐個查詢沿 Key 軸歸一化，不應因畫出的矩陣恰好是方形，就誤以為要沿 Query 軸求和。
<<<END>>>

<<<PATCH 02>>>
<<<OLD>>>
*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*
<<<NEW>>>
*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*以上帶轉置的三維張量，均指在各批次內交換最後兩軸；NumPy 對三維陣列使用 `.T` 會反轉所有軸，不能用它代替此處的轉置。

也可用微分核對整條反傳鏈。固定一個批次，令上游梯度為 $G_Y$，以矩陣內積寫成 $dL=\operatorname{tr}(G_Y^T dY)$。由 $dY=(dA)V+A(dV)$，依循跡的循環性分別收集 $dA$ 與 $dV$ 的係數，便得到 $G_A=G_YV^T$ 及 $G_V=A^TG_Y$。批次間若損失是求和，各批次照此獨立計算；若損失另取平均，上游 $G_Y$ 應已包含該平均因子，不可在每條梯度路徑再次除以批次數。

對固定的查詢列，記其機率為 $a$。微小的分數變動滿足 $da_j=a_j(ds_j-\sum_l a_l ds_l)$。將 $dL=\sum_j g_j da_j$ 代入，收集每個 $ds_j$ 的係數，得到 $g_{s,j}=a_j(g_j-\sum_l a_lg_l)$。因此 $\sum_jg_{s,j}=0$：對同一列所有分數加上相同常數不改變 Softmax，其梯度也必須與此共同平移方向正交。這是精確的結構性檢查，不只是手算時小數湊巧相消。若某位置在 Softmax 前被硬遮罩，其權重精確為零，上式給出該位置的分數梯度為零；只要該列仍有允許位置，其他位置仍依允許權重計算加權和。

最後，由 $dS=((dQ)K^T+Q(dK)^T)/\sqrt{d_k}$，把兩項各自代入 $dL=\operatorname{tr}(G_S^TdS)$，收集 $dQ$、$dK$，即得上列兩個梯度式。這也說明為何 $K$ 的梯度必須使用 $G_S^TQ$，而不是形狀相反的 $Q^TG_S$。這些式子計算的是輸入張量的梯度；若同一組投影權重產生多個輸入，還須在其使用處把權重梯度累加，不能只保留最後一條路徑。

縮放的方差證明也有邊界：它要求相應乘積之間足夠獨立，且所述分量具有指定的均值與方差。訓練後的 Query、Key 可能相關，實際分數方差不一定等於一；縮放是控制典型初始化尺度的設計，不是對所有資料、所有層都避免飽和的定理。有限差分能檢查特定輸入附近的實作，不能替代上述鏈式法則的推導。
<<<END>>>

<<<PATCH 03>>>
<<<OLD>>>
    # 1. Shape 驗證
    if Q.shape[-1] != K.shape[-1]:
<<<NEW>>>
    # 1. Shape 驗證
    if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
        raise ValueError("Q, K, V must have at least two dimensions")
    if Q.shape[-2] <= 0 or V.shape[-1] <= 0:
        raise ValueError("Tq and dv must be positive")
    if Q.shape[-1] != K.shape[-1]:
<<<END>>>

<<<PATCH 04>>>
<<<OLD>>>
    print("All tests passed.")
```

## 測試與預期結果

執行 `run_tests()` 時，預期所有 `assert` 通過，並輸出 "All tests passed."。
*   **測試 1**：驗證輸出形狀正確。
*   **測試 2**：驗證梯度形狀與參數一致。
*   **測試 3**：驗證有限差分誤差小於 $10^{-5}$。
*   **測試 4**：驗證全遮罩行拋出 `ValueError`。
*   **測試 5**：驗證不支援的前導軸廣播被拒絕。
<<<NEW>>>
    # 測試 6：非對稱長度、部分硬遮罩與三路梯度
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
    assert c['attn_weights'].shape == (1, 5, 3)
    assert np.all(c['attn_weights'][~allowed] == 0)
    assert np.allclose(c['attn_weights'].sum(axis=-1), 1)
    gq, gk, gv = scaled_dot_product_attention_bwd(np.ones_like(y), c)
    assert (gq.shape, gk.shape, gv.shape) == (q.shape, k.shape, v.shape)
    assert max(finite_diff_check(q, k, v, allowed)) < 1e-5

    # 測試 7：輸入契約的故障類別
    def rejects(call, error):
        try:
            call()
        except error:
            return
        raise AssertionError("expected rejection")
    rejects(lambda: scaled_dot_product_attention(q[0, 0], k, v), ValueError)
    rejects(lambda: scaled_dot_product_attention(q, k[..., :1], v), ValueError)
    rejects(lambda: scaled_dot_product_attention(q, k, v[:, :2]), ValueError)
    rejects(lambda: scaled_dot_product_attention(q, k, v,
             np.ones((1, 5, 2), dtype=bool)), ValueError)
    rejects(lambda: scaled_dot_product_attention(q, k, v,
             np.ones((1, 5, 3))), TypeError)
    rejects(lambda: scaled_dot_product_attention(q * np.nan, k, v), ValueError)
    for bad_scale in (0, -1, np.nan):
        rejects(lambda s=bad_scale: scaled_dot_product_attention(
            q, k, v, scale_factor=s), ValueError)
    rejects(lambda: scaled_dot_product_attention_bwd(
        np.ones((1, 4, 4)), c), ValueError)
    print("All tests passed.")

if __name__ == "__main__":
    run_tests()
```

## 測試與預期結果

執行上述程式時，預期所有斷言通過，最後才輸出 "All tests passed."；這不是本章已執行的紀錄。測試一至三分別檢查對稱長度的前向形狀、反向形狀與中央差分；測試四拒絕全遮罩列，測試五拒絕前導軸自動廣播。測試六使用五個查詢、三個鍵及不同的值維度，避免方形矩陣碰巧掩蓋轉置錯誤。它還檢查被遮罩位置的權重恰為零、每列沿最後一軸求和為一，並在相同遮罩下比較三路梯度與有限差分。測試七涵蓋低秩輸入、維度或長度不符、遮罩形狀與型別、非有限輸入、非法縮放和上游梯度形狀；被拒絕是這些輸入的預期結果，並非演算法計算失敗。

除故障測試外，這組非對稱資料採局部固定種子，便於重做同一組輸入；固定種子不保證不同 NumPy 版本或設備逐位相同。中央差分比較的是有限步長下的近似值，門檻並不是對所有輸入的數學保證。除目前以輸出元素總和為標量目標的檢查外，除錯時也可固定一個非均勻上游張量，改用輸出與該張量逐元素相乘後的總和做數值損失；解析反傳則傳入同一張量。若數值與解析梯度不符，先核對擾動前後遮罩是否固定、上游梯度是否對應同一損失，再分別檢查縮放、Softmax 的逐列求和，以及最後兩軸的轉置。
<<<END>>>

<<<PATCH 05>>>
<<<OLD>>>
3.  **程式**：修改 `scaled_dot_product_attention`，使其支持 `mask` 為浮點數矩陣（例如 0 或 -1e9），並驗證梯度。
4.  **整合**：假設 $T_q = 10, T_k = 10, d_k = 64$。若 $Q, K$ 分量为 $N(0,1)$，估計 $QK^T$ 元素的分佈。若不縮放，Softmax 輸出會呈現什麼特徵？
<<<NEW>>>
3.  **程式**：加入與分數同形狀的有限浮點 `attn_bias`，說明它與布林硬遮罩的差別，並以固定偏置核對 $Q,K,V$ 梯度。
4.  **整合**：假設 $T_q = 10, T_k = 10, d_k = 64$，且 $Q,K$ 分量獨立、服從 $N(0,1)$。估計 $QK^T$ 單一元素的均值與方差，說明不縮放時權重及不同梯度路徑可能出現的變化。
5.  **反例**：令同一查詢可注意兩個鍵，且兩個 Value 完全相同。構造兩組不同的注意力權重，計算輸出，檢驗「較大權重必定代表對最終輸出較大貢獻」這句話。
<<<END>>>