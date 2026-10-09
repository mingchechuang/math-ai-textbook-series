<<<PATCH 14>>>
<<<OLD>>>
    注意: 若輸入含 NaN 或全列為 -inf，會導致未定義行為或錯誤，需在上游檢查。
<<<NEW>>>
    含 NaN、+inf 或沿 axis 的一整列全為 -inf 時明確拋出
    ValueError；部分位置為 -inf、該列仍有有限值時允許計算。
<<<END>>>

<<<PATCH 14>>>
<<<OLD>>>
    if mask is not None:
        if mask.shape != scores.shape:
<<<NEW>>>
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be a NumPy array")
        if mask.shape != scores.shape:
<<<END>>>

<<<PATCH 14>>>
<<<OLD>>>
**幾何直覺**：
$QK^T$ 測量的是向量夾角的餘弦相似性（若向量已歸一化）或投影長度。縮放因子 $1/\sqrt{d_k}$ 確保當維度增加時，相似度量測的動態範圍不會無限擴大。

**養殖應用案例**：
假設我們用 Transformer 建模水產養殖的日誌序列。
*   **Input**：時間序列上的感測值（溫度、溶氧、pH）與操作日誌（投餌時間）。
*   **Q, K, V**：來自同一個 Embedding 層的不同投影。
*   **注意力意義**：
    *   Query 為「今天的魚群異常」。
    *   Keys 為「過去 7 天的所有事件」。
    *   若資料與訓練目標使投影學得相應關聯，模型可能給與「昨天溶氧驟降」較高的權重。注意，注意力權重本身不是因果證據，僅表示在當前模型表示下該 Token 對當前預測的貢獻較大。
*   **遮罩**：在預測「今天」的狀態時，必須遮罩「明天」及以後的日誌（Causal Mask），防止未來資訊洩漏。
<<<NEW>>>
**幾何直覺**：
兩向量均為單位向量時，內積才等於餘弦相似度；未歸一化的內積同時受範數和夾角影響。縮放調整分數尺度，並不將一般內積變成純角度相似度。

**養殖應用案例**：
假設以完全合成的養殖日誌作示意，而非依據真實現場資料或操作閾值。
*   **Input**：合成的溫度、溶氧、pH 記錄及操作文字。
*   **Q, K, V**：同一組輸入隱藏表示的不同投影；隱藏表示也可能來自前一層，不必直接來自詞元查表。
*   **注意力意義**：
    *   Query 代表模型對「今天的魚群異常」這個位置的表示。
    *   Keys 對應過去事件的位置。
    *   若投影學得某種關聯，「昨天溶氧驟降」的位置可能取得較高權重。這只表示該查詢對相應鍵／值位置分配了較大的混合係數；值向量、輸出投影及後續層仍會影響結果，不能單由權重判定最終貢獻、語義重要性或因果關係。
*   **遮罩**：預測今天時須在歸一化前遮罩明天及以後的日誌，防止未來資訊洩漏；此處沒有生成任何現場操作建議。
<<<END>>>

<<<PATCH 14>>>
<<<OLD>>>
1.  縮放因子 $\sqrt{d_k}$ 的必要性來自於避免 Softmax 梯度飽和（在獨立單位方差假設下）。
<<<NEW>>>
1.  在所列獨立、零均值、單位方差假設下，除以 $\sqrt{d_k}$ 可抵消內積標準差隨維度增加的尺度，降低 Softmax 過度尖銳的風險；它並非唯一必要的尺度，也不保證不會飽和。
<<<END>>>

<<<PATCH 14>>>
<<<OLD>>>
    有限偏置不是硬遮罩：極小但有限的值也不保證禁止位置權重精確為零。偏置固定時三路反傳公式不變。取非方形的有限輸入與非均勻偏置，每次擾動其中一個輸入元素，以相同偏置重算輸出元素總和；其中央差分應逐項核對全一上游梯度所得的三路梯度。若偏置本身可訓練，其梯度為加入偏置後分數的梯度。
<<<NEW>>>
    同時將前向函式簽名改為 `def scaled_dot_product_attention(Q, K, V, mask=None, scale_factor=None, attn_bias=None):`。有限偏置不是硬遮罩；極小但有限的值不保證位置權重精確為零。偏置固定時三路反傳公式不變。下列完整檢查入口在非方形案例中將同一偏置傳給初次前向與每次正負擾動的前向；前述驗證片段應插在主函式計算 `scores` 後、布林遮罩處理前：
    ```python
    def check_fixed_bias():
        rng = np.random.default_rng(14)
        q = rng.standard_normal((1, 2, 2))
        k = rng.standard_normal((1, 3, 2))
        v = rng.standard_normal((1, 3, 2))
        bias = np.array([[[0.2, -0.4, 0.1],
                          [0.3, 0.0, -0.2]]])
        y, cache = scaled_dot_product_attention(
            q, k, v, attn_bias=bias)
        assert y.shape == (1, 2, 2)
        upstream = np.array([[[1.0, -0.5], [0.2, 0.7]]])
        grads = scaled_dot_product_attention_bwd(upstream, cache)

        def loss():
            out, _ = scaled_dot_product_attention(
                q, k, v, attn_bias=bias)
            return np.sum(out * upstream)

        for array, analytic in zip((q, k, v), grads):
            for index in np.ndindex(array.shape):
                original = array[index]
                array[index] = original + 1e-5
                plus = loss()
                array[index] = original - 1e-5
                minus = loss()
                array[index] = original
                numeric = (plus - minus) / (2e-5)
                assert abs(numeric - analytic[index]) < 1e-5
    ```
    預期呼叫 `check_fixed_bias()` 時所有斷言成立；本章未聲稱已實際執行。固定偏置不是本題的求導目標；若它也可訓練且與分數同形狀，其梯度才是加入偏置後的分數梯度。若日後容許偏置廣播，尚須沿廣播軸求和回原形狀。
<<<END>>>