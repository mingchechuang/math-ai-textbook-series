<<<PATCH 01>>>
<<<OLD>>>
注意力機制解決了序列模型中長距離依賴的問題。直觀上，每個 Query 向量 $q_i$ 會與所有 Key 向量 $k_j$ 計算相關性分數，並根據這些分數對 Value 向量 $v_j$ 進行加權求和。
<<<NEW>>>
注意力提供遠距位置之間的直接計算路徑，緩解資訊傳遞路徑過長的問題，但不保證模型一定學得長距依賴。每個 Query 向量 $q_i$ 與允許注意的 Key 向量 $k_j$ 計算分數，再依分數對 Value 向量 $v_j$ 加權求和。Query 與 Key 決定混合係數，Value 決定混合的內容；兩者不可混為一談。即使鍵與查詢的序列長度不同，每個查詢仍各自沿全部允許的鍵歸一化，因此分數矩陣可以是長方形。若所有 Value 相同，無論允許位置之間如何重新分配權重，輸出都相同；這已提醒我們不能把權重直接當作最終預測貢獻。
<<<END>>>

<<<PATCH 02>>>
<<<OLD>>>
*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*
<<<NEW>>>
*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*三維張量公式中的轉置均指逐批次交換最後兩軸，而不是 NumPy 三維陣列的 `.T`，後者會反轉所有軸。

以下以微分補足反傳的理由。固定一個批次，設上游梯度為 $G_Y$，採 $dL=\operatorname{tr}(G_Y^TdY)$。因為 $dY=(dA)V+A(dV)$，分別收集兩項的微分係數，即得 $G_A=G_YV^T$ 與 $G_V=A^TG_Y$。這同時說明，同一個鍵位置被多個查詢使用時，它的 Value 梯度須沿查詢軸累加；矩陣乘法 $A^TG_Y$ 已完成此累加，不能只取其中一個查詢。若損失定義為批次平均，平均因子應先進入 $G_Y$，後續反傳不再重複除以批次數。

再固定一個查詢列，以 $a_j$ 記其 Softmax 權重。微分滿足 $da_j=a_j(ds_j-\sum_l a_l ds_l)$。將它代入 $dL=\sum_jg_jda_j$，按每個 $ds_j$ 收集係數，得到 $g_{s,j}=a_j(g_j-\sum_l a_lg_l)$。把這些係數沿鍵軸相加，利用 $\sum_j a_j=1$，可知 $\sum_jg_{s,j}=0$。這與對一列所有分數同加常數不改變 Softmax 相符，也是比小數近似更可靠的結構檢查。布林硬遮罩先於 Softmax 把禁止位置設為負無限；當該列至少有一個允許位置時，禁止位置的權重與其分數梯度均為零。全列禁止則沒有可定義的機率分布，不能套用此推導，而應由前向明確拒絕。

最後，固定縮放因子時有 $dS=((dQ)K^T+Q(dK)^T)/\sqrt{d_k}$。把它代入 $dL=\operatorname{tr}(G_S^TdS)$ 並分別收集 $dQ$、$dK$ 的係數，便得到 $G_Q=G_SK/\sqrt{d_k}$、$G_K=G_S^TQ/\sqrt{d_k}$。若 $Q,K,V$ 由共享參數產生，這裡只是三個輸入的梯度；還須把各路傳回共享參數的梯度相加。方差命題亦有適用範圍：訓練後的分量可能相關，實際方差未必等於命題中的值。縮放降低僅由維度增長造成分數過大的風險，並不保證所有資料上的 Softmax 都不飽和。
<<<END>>>

<<<PATCH 03>>>
<<<OLD>>>
    # 1. Shape 驗證
    if Q.shape[-1] != K.shape[-1]:
<<<NEW>>>
    # 1. Shape 驗證
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
    # 非方形長度、部分遮罩：分數為 (1, 5, 3)，輸出為 (1, 5, 4)
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
<<<END>>>

<<<PATCH 05>>>
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
3.  **程式**：另加與分數同形狀的有限浮點 `attn_bias`，保持原有布林 `mask` 為硬遮罩；寫出輸入檢查及偏置固定時的梯度驗證方式。
4.  **整合**：假設 $T_q=10,T_k=10,d_k=64$，且 $Q,K$ 分量相互獨立、服從 $N(0,1)$。估計 $QK^T$ 單一元素的均值與方差，並分辨不縮放時傳向分數與傳向 Value 的梯度。
5.  **反例**：兩個鍵的 Value 完全相同時，構造兩組不同的合法權重，檢驗「權重越大，對最終輸出的貢獻必越大」是否成立。

## 習題解答

1.  **解**：$QK^T=1(1)+0(1)=1$，縮放後為 $1/\sqrt{2}$。單元素 Softmax 為一，故輸出為 $[1]$。這個邊界例子也說明：只有一個允許的鍵時，輸出不依賴其分數。

2.  **解**：獨立與零均值條件下，每項乘積的方差為 $\sigma^4$，各項相加得 $d_k\sigma^4$；標準差為 $\sqrt{d_k}\sigma^2$。若希望縮放後的方差為一，可除以此標準差。這不等於聲稱任意訓練後的分量仍獨立。

3.  **解**：為避免把有限負數誤作禁止注意，應保留原有布林硬遮罩與全遮罩列拒絕。可在原函式增加參數 `attn_bias=None`；算出 `scores` 後、布林遮罩處理前，插入：
    ```python
    if attn_bias is not None:
        if attn_bias.shape != scores.shape:
            raise ValueError("bias shape mismatch")
        if not np.issubdtype(attn_bias.dtype, np.floating):
            raise TypeError("bias must be floating")
        if not np.all(np.isfinite(attn_bias)):
            raise ValueError("bias must be finite")
        scores = scores + attn_bias
    ```
    這是加入分數的偏置，不是硬遮罩；極小但有限的值也不能保證被禁止位置的權重恰為零。偏置固定時，反向傳播對 $Q,K,V$ 的公式不變。驗證時在每次擾動三個輸入之一後，用同一偏置重算輸出，取輸出元素總和的中央差分，與原函式在相同偏置下傳入全一上游梯度所得的三路梯度逐項比較。若偏置本身也可訓練，其梯度是遮罩前對分數的梯度，並須依偏置是否廣播而沿相應軸求和；此題限定偏置形狀與分數完全一致。

4.  **解**：單一內積的均值為零、方差為六十四，標準差為八；有限維內積的分布可用常態近似，但不應宣稱精確服從常態。未縮放時，與除以八的情形相比，分數往往更分散，權重可能較尖銳。當權重趨近獨熱分布時，Softmax 對最大位置的自身導數 $a_m(1-a_m)$ 也趨近零；不能說只有最大鍵仍有分數梯度。傳向 $Q,K$ 的梯度可能變弱，而 $dV=A^TdY$ 仍可能主要流向高權重位置；具體值取決於上游梯度與 Value。

5.  **解**：令兩個 Value 均為直向量 $[2,3]^T$。權重分別取 $(0.9,0.1)$ 與 $(0.1,0.9)$，兩次加權和皆為 $[2,3]^T$。兩組權重明顯不同，輸出卻相同；因此不能單靠注意力權重判定對最終輸出的貢獻，更不能據此推斷因果關係。
<<<END>>>