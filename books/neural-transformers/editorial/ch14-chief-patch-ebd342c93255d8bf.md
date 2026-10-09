<<<PATCH 01>>>
<<<OLD>>>
## AI、幾何與養殖案例

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
## AI、幾何與養殖案例

**幾何直覺**：
未歸一化內積同時受範數與夾角影響；兩向量均單位化時才等於餘弦相似度。縮放因子 $1/\sqrt{d_k}$ 確保當維度增加時，相似度量測的動態範圍不會無限擴大。

**養殖應用案例**：
假設我們用 Transformer 建模水產養殖的日誌序列。
*   **Input**：時間序列上的感測值（溫度、溶氧、pH）與操作日誌（投餌時間）。
*   **Q, K, V**：來自同一個 Embedding 層的不同投影。
*   **注意力意義**：
    *   Query 為「今天的魚群異常」。
    *   Keys 為「過去 7 天的所有事件」。
    *   若資料與訓練目標使投影學得相應關聯，模型可能給與「昨天溶氧驟降」較高的權重。注意，注意力權重本身不是因果證據；它只表示該 query 對該 key/value 位置分配較大的混合權重，不能單由權重判定最終貢獻、語義重要性或因果關係。
*   **遮罩**：在預測「今天」的狀態時，必須遮罩「明天」及以後的日誌（Causal Mask），防止未來資訊洩漏。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
### 推導：反向傳播（VJP）

設縮放後的分數 $S = \frac{QK^T}{\sqrt{d_k}}$。
設 Softmax 輸出 $A = \text{softmax}(S)$。
設注意力輸出 $Y = AV$。

損失函數 $L$ 對 $Y$ 的梯度為 $\frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times T_q \times d_v}$。

**1. 對 $V$ 的梯度**
$$
\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial Y}
$$
形狀：$(B, T_k, T_q) \times (B, T_q, d_v) \rightarrow (B, T_k, d_v)$。

**2. 對 $A$ 的梯度**
$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} V^T
$$
形狀：$(B, T_q, d_v) \times (B, d_v, T_k) \rightarrow (B, T_q, T_k)$。

**3. 對 $S$ 的梯度（Softmax VJP）**
令 $G_A = \frac{\partial L}{\partial A}$。Softmax 的 Jacobian-vector product 公式為：
$$
\frac{\partial L}{\partial S}_{:,j} = A_{:,j} \left( G_A_{:,j} - \sum_{l=1}^{T_k} A_{:,l} G_A_{:,l} \right)
$$
在矩陣形式下（注意廣播方向）：
$$
\frac{\partial L}{\partial S} = A \odot \left( G_A - \text{sum}(A \odot G_A, \text{axis}=T_k, \text{keepdims=True}) \right)
$$
其中 $\text{sum}$ 的結果形狀為 $(B, T_q, 1)$，需廣播回 $(B, T_q, T_k)$。

**4. 對 $Q$ 的梯度**
由 $S = \frac{1}{\sqrt{d_k}} Q K^T$，利用鏈式法則：
$$
\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial L}{\partial S} \right) K
$$
形狀：$(B, T_q, T_k) \times (B, T_k, d_k) \rightarrow (B, T_q, d_k)$。

**5. 對 $K$ 的梯度**
由 $S = \frac{1}{\sqrt{d_k}} Q K^T$，且 $K$ 在轉置位置：
$$
\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial L}{\partial S} \right)^T Q
$$
形狀：$(B, T_k, T_q) \times (B, T_q, d_k) \rightarrow (B, T_k, d_k)$。

*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*
<<<NEW>>>
### 推導：反向傳播（VJP）

設縮放後的分數 $S = \frac{QK^T}{\sqrt{d_k}}$。
設 Softmax 輸出 $A = \text{softmax}(S)$。
設注意力輸出 $Y = AV$。

損失函數 $L$ 對 $Y$ 的梯度為 $\frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times T_q \times d_v}$。

**Frobenius 內積與梯度推導**
矩陣微分可用 Frobenius 內積表示：$df = \langle G, dX \rangle_F = \text{tr}(G^T dX)$，其中 $G = \frac{\partial f}{\partial X}$。以下利用此形式推導各梯度。

**1. 對 $V$ 的梯度**
由 $Y = AV$，$\langle dY, \frac{\partial L}{\partial Y} \rangle = \langle A dV, \frac{\partial L}{\partial Y} \rangle = \langle dV, A^T \frac{\partial L}{\partial Y} \rangle$，故：
$$
\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial Y}
$$
此處 $A^T$ 表示對每個 batch 交換最後兩軸（即 $(B, T_q, T_k)$ 轉為 $(B, T_k, T_q)$），而非 NumPy 對所有軸反序。形狀：$(B, T_k, T_q) \times (B, T_q, d_v) \rightarrow (B, T_k, d_v)$。

**2. 對 $A$ 的梯度**
由 $Y = AV$，$\langle dY, \frac{\partial L}{\partial Y} \rangle = \langle dA V, \frac{\partial L}{\partial Y} \rangle = \langle dA, \frac{\partial L}{\partial Y} V^T \rangle$，故：
$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} V^T
$$
此處 $V^T$ 同樣表示逐 batch 交換最後兩軸。形狀：$(B, T_q, d_v) \times (B, d_v, T_k) \rightarrow (B, T_q, T_k)$。

**3. 對 $S$ 的梯度（Softmax VJP）**
令 $G_A = \frac{\partial L}{\partial A}$。Softmax 的 Jacobian-vector product 公式為：
$$
\frac{\partial L}{\partial S}_{:,j} = A_{:,j} \left( G_A_{:,j} - \sum_{l=1}^{T_k} A_{:,l} G_A_{:,l} \right)
$$
在矩陣形式下（注意廣播方向）：
$$
\frac{\partial L}{\partial S} = A \odot \left( G_A - \text{sum}(A \odot G_A, \text{axis}=T_k, \text{keepdims=True}) \right)
$$
其中 $\text{sum}$ 的結果形狀為 $(B, T_q, 1)$，需廣播回 $(B, T_q, T_k)$。

**證明每列和為零**：Softmax 對每列輸入加上任意常數不變，因此梯度必須與全 1 方向正交。即對每個 $i$，$\sum_{j} \frac{\partial L}{\partial S_{i,j}} = 0$。這可作為數值實作的結構性核對。

**4. 對 $Q$ 的梯度**
由 $S = \frac{1}{\sqrt{d_k}} Q K^T$，利用鏈式法則：
$$
\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial L}{\partial S} \right) K
$$
形狀：$(B, T_q, T_k) \times (B, T_k, d_k) \rightarrow (B, T_q, d_k)$。

**5. 對 $K$ 的梯度**
由 $S = \frac{1}{\sqrt{d_k}} Q K^T$，且 $K$ 在轉置位置：
$$
\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial L}{\partial S} \right)^T Q
$$
此處 $(dS)^T$ 表示逐 batch 交換最後兩軸（即 $(B, T_q, T_k)$ 轉為 $(B, T_k, T_q)$）。形狀：$(B, T_k, T_q) \times (B, T_q, d_k) \rightarrow (B, T_k, d_k)$。

*注意：梯度形狀必須與參數 $K$ 的形狀 $(B, T_k, d_k)$ 一致。*
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
3.  **程式**：修改 `scaled_dot_product_attention`，使其支持 `mask` 為浮點數矩陣（例如 0 或 -1e9），並驗證梯度。
4.  **整合**：假設 $T_q = 10, T_k = 10, d_k = 64$。若 $Q, K$ 分量为 $N(0,1)$，估計 $QK^T$ 元素的分佈。若不縮放，Softmax 輸出會呈現什麼特徵？
<<<NEW>>>
3.  **程式**：修改 `scaled_dot_product_attention`，使其支持 additive logit bias（例如 `attn_bias` 參數，浮點數矩陣），並驗證梯度。注意：additive bias 不表示 hard mask；hard mask 仍由布林陣列負責。
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
import numpy as np

def stable_softmax(x, axis=-1):
    """
    穩定 Softmax。
    輸入: x 形狀 (..., T)
    輸出: softmax(x) 形狀 (..., T)
    注意: 若輸入含 NaN 或全列為 -inf，會導致未定義行為或錯誤，需在上游檢查。
    """
    # 檢查 NaN 和 +inf
    if np.any(np.isnan(x)) or np.any(np.isposinf(x)):
        raise ValueError("softmax input contains NaN or +inf")
    
    # 檢查是否有全 -inf 行
    if np.any(np.all(np.isneginf(x), axis=axis)):
        raise ValueError("softmax row has no finite logit")
        
    # 減去最大值以確保數值穩定
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    # 檢查是否有全遮罩行 (sum_exp == 0)
    if np.any(sum_exp == 0):
        raise ValueError("Error: Detected fully masked row in softmax input.")
        
    return exp_x / sum_exp

def scaled_dot_product_attention(Q, K, V, mask=None, scale_factor=None):
    """
    Q: (..., Tq, dk)
    K: (..., Tk, dk)
    V: (..., Tk, dv)
    mask: (..., Tq, Tk) boolean, True=Allowed.
    scale_factor: 縮放因子，預設為 sqrt(dk)。
    """
    # 1. Shape 驗證
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have same last dimension (dk)")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must have same sequence length (Tk)")
    if Q.shape[:-2] != K.shape[:-2] or Q.shape[:-2] != V.shape[:-2]:
        raise ValueError("Q, K, V must have identical leading batch shapes")
    if Q.shape[-1] <= 0 or K.shape[-2] <= 0:
        raise ValueError("dk and Tk must be positive")
<<<NEW>>>
import numpy as np

def stable_softmax(x, axis=-1):
    """
    穩定 Softmax。
    輸入: x 形狀 (..., T)
    輸出: softmax(x) 形狀 (..., T)
    含 NaN、+inf 或全 -inf 的 reduction 行會明確拋出 ValueError；
    部分位置為 -inf、且至少有一個有限 logit 的行允許計算。
    """
    # 檢查 NaN 和 +inf
    if np.any(np.isnan(x)) or np.any(np.isposinf(x)):
        raise ValueError("softmax input contains NaN or +inf")
    
    # 檢查是否有全 -inf 行
    if np.any(np.all(np.isneginf(x), axis=axis)):
        raise ValueError("softmax row has no finite logit")
        
    # 減去最大值以確保數值穩定
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    # 防禦性檢查：正常情況下 sum_exp 不應為 0
    if np.any(sum_exp == 0):
        raise ValueError("Error: Detected fully masked row in softmax input.")
        
    return exp_x / sum_exp

def scaled_dot_product_attention(Q, K, V, mask=None, scale_factor=None):
    """
    Q: (..., Tq, dk)
    K: (..., Tk, dk)
    V: (..., Tk, dv)
    mask: (..., Tq, Tk) boolean, True=Allowed.
    scale_factor: 縮放因子，預設為 sqrt(dk)。
    """
    # 1. Shape 驗證
    if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
        raise ValueError("Q, K, V must have at least two dimensions")
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have same last dimension (dk)")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must have same sequence length (Tk)")
    if Q.shape[:-2] != K.shape[:-2] or Q.shape[:-2] != V.shape[:-2]:
        raise ValueError("Q, K, V must have identical leading batch shapes")
    if Q.shape[-1] <= 0 or K.shape[-2] <= 0:
        raise ValueError("dk and Tk must be positive")
<<<END>>>