# 第14章 縮放內積注意力與QKV

## 學習目標與先備知識

在本章結束時，讀者應能掌握縮放內積注意力（Scaled Dot-Product Attention, SDPA）的數學定義、幾何直覺與完整實作。具體目標包括：
1.  **數學推導**：理解 $QK^T$ 的形狀，並證明為何除以 $\sqrt{d_k}$ 能維持分數方差穩定。
2.  **反向傳播**：能手寫並實作 $Q, K, V$ 的完整梯度計算，特別是 Softmax VJP 的正確形式。
3.  **實作細節**：處理對稱與非對稱序列長度、布林遮罩（Mask）的邏輯，以及數值穩定性（Softmax 減去最大值）。
4.  **驗證方法**：使用有限差分（Finite Difference）驗證梯度正確性，並設計正常、邊界與故障測試。

先備知識需熟練 Volume I 的矩陣運算（$matmul$、轉置、廣播）與 Volume IV 的微分鏈式法則（VJP、Frobenius 內積）。本章假設讀者已理解「梯度必須與參數同 Shape」以及「廣播反向傳播需沿被廣播軸求和」的原則。本章程式實作採用「嚴密 Shape 檢查」策略，即 $Q, K, V$ 的前導批次軸必須完全相同，不支援 NumPy 的自動廣播，以避免反向傳播中因 Shape 不一致導致的梯度累加錯誤。

## 問題與直覺

注意力機制解決了序列模型中長距離依賴的問題。直觀上，每個 Query 向量 $q_i$ 會與所有 Key 向量 $k_j$ 計算相關性分數，並根據這些分數對 Value 向量 $v_j$ 進行加權求和。

若直接計算內積 $q_i \cdot k_j$，當維度 $d_k$ 增大時，內積的數值範圍也會隨之擴大。這會導致 Softmax 函數進入飽和區（Saturation Region），使得梯度極小，模型難以學習。因此，我們需要一個縮放因子來規範分數的尺度。

本卷約定：
*   **遮罩語義**：布林值 `True` 表示允許注意（Allowed），`False` 表示遮罩（Masked/Forbidden）。
*   **遮罩位置**：遮罩必須在 Softmax **之前** 施加（通常透過加上 $-\infty$ 實現）。
*   **全遮罩行**：若某 Query 對應的所有 Keys 均被遮罩，實作必須明確拒絕（拋出錯誤），不得讓 $NaN$ 靜默流入。
*   **Shape 約定**：$Q \in \mathbb{R}^{B \times T_q \times d_k}$，$K \in \mathbb{R}^{B \times T_k \times d_k}$，$V \in \mathbb{R}^{B \times T_k \times d_v}$。Softmax 沿著最後一個維度（Key 軸）進行歸一化。

## 定義、定理與推導

### 定義 14.1：縮放內積注意力

給定批次維度 $B$，查詢 $Q \in \mathbb{R}^{B \times T_q \times d_k}$，鍵 $K \in \mathbb{R}^{B \times T_k \times d_k}$，值 $V \in \mathbb{R}^{B \times T_k \times d_v}$。縮放內積注意力定義為：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
1.  $QK^T$ 的形状為 $(B, T_q, T_k)$。
2.  $\text{softmax}$ 沿著 $T_k$ 軸（最後一個軸）進行計算。
3.  輸出形狀為 $(B, T_q, d_v)$。

### 命題 14.1：縮放因子的方差假設

**命題**：假設 $q, k \in \mathbb{R}^{d_k}$ 的分量 $q_i, k_j$ 是獨立同分布的隨機變數，且均值為 0，方差為 1（即 $E[q_i]=0, \text{Var}(q_i)=1$）。則內積 $q \cdot k$ 的方差為 $d_k$。

**證明**：

內積定義為：
$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

根據方差的可加性（當隨機變數獨立時）：
$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)
$$

對於獨立隨機變數 $q_i$ 和 $k_i$，且均值為 0：
$$
\text{Var}(q_i k_i) = E[(q_i k_i)^2] - (E[q_i k_i])^2 = E[q_i^2]E[k_i^2] - 0 = 1 \cdot 1 = 1
$$

因此：
$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} 1 = d_k
$$

標準差為 $\sqrt{d_k}$。
若我們將內積除以 $\sqrt{d_k}$，則新變數的方差為：
$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(q \cdot k) = 1
$$

**結論**：在獨立、零均值、單位分量的假設下，除以 $\sqrt{d_k}$ 可將分數的方差維持為 1，防止隨維度增加而發散，從而降低僅因維度增加而導致 Softmax 過度飽和的風險。

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

## 逐步手算例題

### 例 14.1：前向計算（手算）

設 $B=1, T_q=2, T_k=3, d_k=2, d_v=2$。
$$
Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
$$

1.  **計算 $QK^T$**：
    $$
    QK^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}
    $$

2.  **縮放**：
    $\sqrt{d_k} = \sqrt{2} \approx 1.414$。
    $$
    S = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix} \approx \begin{bmatrix} 0.707 & 0.707 & 0 \\ 0.707 & 0 & 0.707 \end{bmatrix}
    $$

3.  **Softmax（沿 Key 軸，即列方向）**：
    *   Row 0: $[0.707, 0.707, 0]$。$e^{0.707} \approx 2.028$, $e^0=1$。Sum $\approx 5.056$。
        $A_0 \approx [0.401, 0.401, 0.198]$。
    *   Row 1: $[0.707, 0, 0.707]$。Sum $\approx 5.056$。
        $A_1 \approx [0.401, 0.198, 0.401]$。
    
    $$
    A \approx \begin{bmatrix} 0.401 & 0.401 & 0.198 \\ 0.401 & 0.198 & 0.401 \end{bmatrix}
    $$

4.  **計算 $Y = AV$**：
    $$
    Y \approx \begin{bmatrix} 0.401 & 0.401 & 0.198 \\ 0.401 & 0.198 & 0.401 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
    $$
    $Y_{0,0} = 0.401(1) + 0.401(0) + 0.198(1) = 0.599$
    $Y_{0,1} = 0.401(0) + 0.401(1) + 0.198(1) = 0.599$
    $Y_{1,0} = 0.401(1) + 0.198(0) + 0.401(1) = 0.802$
    $Y_{1,1} = 0.401(0) + 0.198(1) + 0.401(1) = 0.599$
    
    $$
    Y \approx \begin{bmatrix} 0.599 & 0.599 \\ 0.802 & 0.599 \end{bmatrix}
    $$

### 例 14.2：反向傳播（手算）

接上例，假設 $\frac{\partial L}{\partial Y} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$。

1.  **$\frac{\partial L}{\partial V}$**：
    $$
    \frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial Y} = \begin{bmatrix} 0.401 & 0.401 \\ 0.401 & 0.198 \\ 0.198 & 0.401 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.401 & 0.401 \\ 0.401 & 0.198 \\ 0.198 & 0.401 \end{bmatrix}
    $$

2.  **$\frac{\partial L}{\partial A}$**：
    $$
    \frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} V^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}
    $$

3.  **$\frac{\partial L}{\partial S}$**：
    計算加權和 $C_i = \sum_j A_{i,j} G_{A,i,j}$。
    *   Row 0: $C_0 = 0.401(1) + 0.401(0) + 0.198(1) = 0.599$。
    *   Row 1: $C_1 = 0.401(0) + 0.198(1) + 0.401(1) = 0.599$。
    
    $\frac{\partial L}{\partial S}_{i,j} = A_{i,j} (G_{A,i,j} - C_i)$
    
    Row 0:
    $j=0: 0.401(1 - 0.599) = 0.1608$
    $j=1: 0.401(0 - 0.599) = -0.2402$
    $j=2: 0.198(1 - 0.599) = 0.0794$
    
    Row 1:
    $j=0: 0.401(0 - 0.599) = -0.2402$
    $j=1: 0.198(1 - 0.599) = 0.0794$
    $j=2: 0.401(1 - 0.599) = 0.1608$
    
    $$
    \frac{\partial L}{\partial S} \approx \begin{bmatrix} 0.161 & -0.240 & 0.079 \\ -0.240 & 0.079 & 0.161 \end{bmatrix}
    $$
    *(驗證：每列和應為0。$0.161 - 0.240 + 0.079 = 0$。正確。)*

4.  **$\frac{\partial L}{\partial Q}$**：
    $$
    \frac{\partial L}{\partial Q} = \frac{1}{\sqrt{2}} \frac{\partial L}{\partial S} K \approx 0.707 \times \begin{bmatrix} 0.161 & -0.240 & 0.079 \\ -0.240 & 0.079 & 0.161 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}
    $$
    $$
    \approx 0.707 \times \begin{bmatrix} -0.079 & 0.240 \\ -0.161 & -0.079 \end{bmatrix} \approx \begin{bmatrix} -0.056 & 0.170 \\ -0.114 & -0.056 \end{bmatrix}
    $$

5.  **$\frac{\partial L}{\partial K}$**：
    $$
    \frac{\partial L}{\partial K} = \frac{1}{\sqrt{2}} \left( \frac{\partial L}{\partial S} \right)^T Q \approx 0.707 \times \begin{bmatrix} 0.161 & -0.240 \\ -0.240 & 0.079 \\ 0.079 & 0.161 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
    $$
    $$
    \approx 0.707 \times \begin{bmatrix} 0.161 & -0.240 \\ -0.240 & 0.079 \\ 0.079 & 0.161 \end{bmatrix} \approx \begin{bmatrix} 0.114 & -0.170 \\ -0.170 & 0.056 \\ 0.056 & 0.114 \end{bmatrix}
    $$
    *驗證：$\frac{\partial L}{\partial K}$ 的形狀為 $(3, 2)$，與 $K$ 的形狀 $(3, 2)$ 一致。*

## 實作與程式

以下提供自足的 NumPy 實作，包含前向、反向、遮罩處理與有限差分驗證。程式僅使用標準庫與 NumPy，不依賴 PyTorch 或外部訓練框架。

```python
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

    # 檢查輸入是否有限
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(K)) and np.all(np.isfinite(V))):
        raise ValueError("Q, K, V must contain only finite values")

    dk = Q.shape[-1]
    if scale_factor is None:
        scale_factor = np.sqrt(dk)
    if not np.isscalar(scale_factor):
        raise TypeError("scale_factor must be scalar")
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")

    # 2. 前向計算
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / scale_factor
    
    if mask is not None:
        if mask.shape != scores.shape:
            raise ValueError("Mask shape must match scores shape")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise TypeError("Mask must be boolean dtype")
        
        # 檢查全遮罩行
        row_sums = np.sum(mask, axis=-1)
        if np.any(row_sums == 0):
            raise ValueError("Error: Query has no allowed keys (fully masked row).")
            
        # 將 False (Masked) 設為 -inf
        # True -> 0, False -> -inf
        mask_float = np.where(mask, 0.0, -np.inf)
        scores = scores + mask_float

    attn_weights = stable_softmax(scores, axis=-1)
    output = np.matmul(attn_weights, V)
    
    # 保存中間結果供反向使用
    cache = {
        'Q': Q, 'K': K, 'V': V,
        'attn_weights': attn_weights,
        'scale_factor': scale_factor
    }
    return output, cache

def scaled_dot_product_attention_bwd(dY, cache):
    """
    dY: (..., Tq, dv)
    """
    Q = cache['Q']
    K = cache['K']
    V = cache['V']
    attn_weights = cache['attn_weights']
    scale_factor = cache['scale_factor']
    
    expected_dy_shape = attn_weights.shape[:-1] + (V.shape[-1],)
    if dY.shape != expected_dy_shape:
        raise ValueError("dY shape mismatch")

    # 1. dV
    dV = np.matmul(np.swapaxes(attn_weights, -1, -2), dY)
    
    # 2. dAttn
    dAttn = np.matmul(dY, np.swapaxes(V, -1, -2))
    
    # 3. dScores (Softmax VJP)
    # dScores = A * (dAttn - sum(A * dAttn, axis=key, keepdims=True))
    weighted_sum = np.sum(attn_weights * dAttn, axis=-1, keepdims=True)
    dScores = attn_weights * (dAttn - weighted_sum)
    
    # 4. dQ and dK
    # dQ = (1/scale) * dScores @ K
    dQ = np.matmul(dScores, K) / scale_factor
    # dK = (1/scale) * dScores.T @ Q
    # 注意: dScores 形狀 (..., Tq, Tk), Q 形狀 (..., Tq, dk)
    # dScores.T 形狀 (..., Tk, Tq)
    dK = np.matmul(np.swapaxes(dScores, -1, -2), Q) / scale_factor
    
    return dQ, dK, dV

def finite_diff_check(Q, K, V, mask=None, eps=1e-5):
    """
    有限差分驗證 Q, K, V 的梯度。
    返回三個誤差值。
    """
    # 確保使用 float64
    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    V = V.astype(np.float64)
    
    # 前向
    Y, cache = scaled_dot_product_attention(Q, K, V, mask)
    dY = np.ones_like(Y)
    dQ, dK, dV = scaled_dot_product_attention_bwd(dY, cache)
    
    def loss(q, k, v):
        y, _ = scaled_dot_product_attention(q, k, v, mask)
        return np.sum(y)
    
    def num_grad(param, bwd_grad, name):
        num_g = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            
            if name == 'Q':
                Q[idx] = old_val + eps
                y_plus = loss(Q, K, V)
                Q[idx] = old_val - eps
                y_minus = loss(Q, K, V)
                Q[idx] = old_val
            elif name == 'K':
                K[idx] = old_val + eps
                y_plus = loss(Q, K, V)
                K[idx] = old_val - eps
                y_minus = loss(Q, K, V)
                K[idx] = old_val
            elif name == 'V':
                V[idx] = old_val + eps
                y_plus = loss(Q, K, V)
                V[idx] = old_val - eps
                y_minus = loss(Q, K, V)
                V[idx] = old_val
            
            num_g[idx] = (y_plus - y_minus) / (2 * eps)
            it.iternext()
        return np.max(np.abs(num_g - bwd_grad))
    
    err_Q = num_grad(Q, dQ, 'Q')
    err_K = num_grad(K, dK, 'K')
    err_V = num_grad(V, dV, 'V')
    
    return err_Q, err_K, err_V

def run_tests():
    # 測試 1: 正常前向
    Q = np.random.randn(2, 3, 4).astype(np.float64)
    K = np.random.randn(2, 3, 4).astype(np.float64)
    V = np.random.randn(2, 3, 4).astype(np.float64)
    Y, cache = scaled_dot_product_attention(Q, K, V)
    assert Y.shape == (2, 3, 4)
    
    # 測試 2: 梯度驗證
    dY = np.random.randn(2, 3, 4).astype(np.float64)
    dQ, dK, dV = scaled_dot_product_attention_bwd(dY, cache)
    assert dQ.shape == Q.shape
    assert dK.shape == K.shape
    assert dV.shape == V.shape
    
    # 測試 3: 有限差分
    err_Q, err_K, err_V = finite_diff_check(Q, K, V)
    assert err_Q < 1e-5 and err_K < 1e-5 and err_V < 1e-5, f"Gradient check failed: {err_Q}, {err_K}, {err_V}"
    
    # 測試 4: 全遮罩拒絕
    Q_small = np.random.randn(1, 1, 2).astype(np.float64)
    K_small = np.random.randn(1, 2, 2).astype(np.float64)
    V_small = np.random.randn(1, 2, 2).astype(np.float64)
    mask_full = np.zeros((1, 1, 2), dtype=bool)
    try:
        scaled_dot_product_attention(Q_small, K_small, V_small, mask=mask_full)
        raise Exception("Should have raised ValueError")
    except ValueError as e:
        assert "fully masked" in str(e)
    
    # 測試 5: 廣播拒絕
    Q_b = np.random.randn(2, 3, 4).astype(np.float64)
    K_b = np.random.randn(1, 3, 4).astype(np.float64)
    V_b = np.random.randn(1, 3, 4).astype(np.float64)
    try:
        scaled_dot_product_attention(Q_b, K_b, V_b)
        raise Exception("Should have raised ValueError")
    except ValueError as e:
        assert "identical leading batch shapes" in str(e)
        
    print("All tests passed.")
```

## 測試與預期結果

執行 `run_tests()` 時，預期所有 `assert` 通過，並輸出 "All tests passed."。
*   **測試 1**：驗證輸出形狀正確。
*   **測試 2**：驗證梯度形狀與參數一致。
*   **測試 3**：驗證有限差分誤差小於 $10^{-5}$。
*   **測試 4**：驗證全遮罩行拋出 `ValueError`。
*   **測試 5**：驗證不支援的前導軸廣播被拒絕。

## 反例與常見陷阱

1.  **忘記縮放**：若 $d_k=512$，內積標準差約 22.6。Softmax 輸入差異過大，導致權重極端二值化，梯度極小，訓練收斂緩慢。
2.  **Softmax 軸錯誤**：錯誤地對 Query 軸進行歸一化。注意力權重必須對每個 Query 獨立歸一化。
3.  **遮罩語義混淆**：本卷 `True=Allowed`。若使用加性遮罩（加 -inf），必須確保在 Softmax **之前** 加上。若在 Softmax 之後乘以 0，會改變分母，導致權重分佈錯誤。
4.  **全遮罩行未處理**：若某 Query 對應的所有 Keys 都被遮罩，Softmax 輸入全為 -inf，導致 $0/0 = NaN$。實作必須檢測並拒絕。
5.  **$dK$ 轉置錯誤**：$dK$ 必須是 $dS^T Q$，形狀與 $K$ 相同。常見的錯誤是寫成 $Q^T dS$，這會導致 Shape 錯誤或結果轉置。

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

## 習題

1.  **手算**：給定 $Q = [1, 0], K = [1, 1], V = [1]$（維度 $d_k=2, d_v=1$，單個 Query, 單個 Key）。計算 Attention 輸出。
2.  **理論**：若 $q, k$ 分量的方差為 $\sigma^2$ 而非 1（且獨立、零均值），縮放因子應為何？
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

## 本章小結

本章介紹了縮放內積注意力的數學基礎、方差假設證明、完整的前向與反向傳播實作。重點在於：
1.  縮放因子 $\sqrt{d_k}$ 的必要性來自於避免 Softmax 梯度飽和（在獨立單位方差假設下）。
2.  Softmax 沿 Key 軸歸一化。
3.  遮罩必須在 Softmax 前施加，全遮罩行必須明確拒絕。
4.  反向傳播中 Softmax 的 VJP 公式是關鍵，$dK$ 的計算需注意轉置與 Shape 匹配。

## 參考來源

*   [N1] Vaswani et al., Attention Is All You Need. (縮放因子的原始來源)
*   [N3] PyTorch SDPA API. (參考 API 語義，非本程式執行證據)
*   [N4] NumPy Broadcasting Guide. (延伸閱讀，尚未逐條核對)