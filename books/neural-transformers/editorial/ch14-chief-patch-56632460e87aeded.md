<<<PATCH 01>>>
<<<OLD>>>
**幾何直覺**：
$QK^T$ 測量的是向量夾角的餘弦相似性（若向量已歸一化）或投影長度。縮放因子 $1/\sqrt{d_k}$ 確保當維度增加時，相似度量測的動態範圍不會無限擴大。
<<<NEW>>>
**幾何直覺**：
未歸一化內積同時受範數與夾角影響；兩向量均單位化時才等於餘弦相似度。縮放因子 $1/\sqrt{d_k}$ 確保當維度增加時，相似度量測的動態範圍不會無限擴大。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    *   若資料與訓練目標使投影學得相應關聯，模型可能給與「昨天溶氧驟降」較高的權重。注意，注意力權重本身不是因果證據，僅表示在當前模型表示下該 Token 對當前預測的貢獻較大。
<<<NEW>>>
    *   若資料與訓練目標使投影學得相應關聯，模型可能給與「昨天溶氧驟降」較高的權重。注意，注意力權重本身不是因果證據；它只表示該 query 對該 key/value 位置分配較大的混合權重，不能單由權重判定最終貢獻、語義重要性或因果關係。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
1.  **手算**：給定 $Q = [1, 0], K = [1, 1], V = [1]$（維度 $d_k=2, d_v=1$，單個 Query, 單個 Key）。計算 Attention 輸出。
2.  **理論**：若 $q, k$ 分量的方差為 $\sigma^2$ 而非 1（且獨立、零均值），縮放因子應為何？
3.  **程式**：修改 `scaled_dot_product_attention`，使其支持 `mask` 為浮點數矩陣（例如 0 或 -1e9），並驗證梯度。
4.  **整合**：假設 $T_q = 10, T_k = 10, d_k = 64$。若 $Q, K$ 分量为 $N(0,1)$，估計 $QK^T$ 元素的分佈。若不縮放，Softmax 輸出會呈現什麼特徵？
<<<NEW>>>
1.  **手算**：給定 $Q = [1, 0], K = [1, 1], V = [1]$（維度 $d_k=2, d_v=1$，單個 Query, 單個 Key）。計算 Attention 輸出。
2.  **理論**：若 $q, k$ 分量的方差為 $\sigma^2$ 而非 1（且獨立、零均值），縮放因子應為何？
3.  **程式**：修改 `scaled_dot_product_attention`，使其支持 additive logit bias（例如新增 `attn_bias` 參數，浮點數矩陣），並驗證梯度。注意：additive bias 不表示 hard mask；hard mask 仍由布林陣列負責。
4.  **整合**：假設 $T_q = 10, T_k = 10, d_k = 64$。若 $Q, K$ 分量为 $N(0,1)$，估計 $QK^T$ 元素的分佈。若不縮放，Softmax 輸出會呈現什麼特徵？
<<<END>>>
<<<PATCH 04>>>
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
3.  **解**：
    修改為支持 additive logit bias，例如新增 `attn_bias` 參數：
    ```python
    if attn_bias is not None:
        if not np.issubdtype(attn_bias.dtype, np.floating):
            raise TypeError("attn_bias must be floating dtype")
        if attn_bias.shape != scores.shape:
            raise ValueError("attn_bias shape must match scores shape")
        scores = scores + attn_bias
        # 注意：additive bias 不表示 hard mask，僅為有限 logit 偏移。
    ```
    梯度驗證邏輯不變。若需 hard mask，仍應傳入布林 `mask`，並確保在 Softmax 前將 False 設為 `-np.inf`。

4.  **解**：
    $QK^T$ 元素近似 $N(0, d_k) = N(0, 64)$。Std $= 8$。
    若不縮放，Softmax 輸入差異很大。相比縮放後，Softmax 分佈會更尖銳（Spiky），大部分權重趨近於 0，最大權重趨近於 1。此時 Softmax Jacobian 的許多分量可能變小，使傳向 logits、$Q$ 與 $K$ 的梯度在部分方向上變弱；精確梯度仍取決於上游 $dA$。這會增加訓練的不穩定性。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
1.  縮放因子 $\sqrt{d_k}$ 的必要性來自於避免 Softmax 梯度飽和（在獨立單位方差假設下）。
<<<NEW>>>
1.  在所列獨立與單位方差近似下，除以 $\sqrt{d_k}$ 抵消內積標準差隨維度增長的尺度，降低僅由維度造成 Softmax 過度尖銳的風險。
<<<END>>>