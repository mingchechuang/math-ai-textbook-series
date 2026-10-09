<<<PATCH 04>>>
<<<OLD>>>
**情形二：自回歸序列模型**
若樣本為序列 $(x_1, \dots, x_T)$，且模型假設條件分布 $p_\theta(x_t | x_{<t})$，則聯合似然由鏈式法則分解為：
$$ L(\theta) = \prod_{t=1}^T p_\theta(x_t | x_{<t}) $$
此時樣本並非無條件獨立，而是馬可夫性質下的條件獨立。對數似然為：
$$ \ell(\theta) = \sum_{t=1}^T \ln p_\theta(x_t | x_{<t}) $$

**總負對數似然（Total NLL）**定義為：
$$ \text{Total NLL}(\theta) = -\sum_{i=1}^N \ln p_\theta(x_i) \quad \text{(或} \sum_{t=1}^T \ln p_\theta(x_t | x_{<t}) \text{)} $$
**平均負對數似然（Mean NLL）**定義為：
$$ \text{Mean NLL}(\theta) = \frac{1}{N_{\text{valid}}} \text{Total NLL}(\theta) $$
其中 $N_{\text{valid}}$ 為有效樣本或 token 數量。最小化 Total NLL 或 Mean NLL 均等價於最大化似然（MLE）。
<<<NEW>>>
**情形二：自回歸序列模型**
若樣本為序列 $(x_1, \dots, x_T)$，且模型以先前 token 為條件預測下一個 token，即使用條件分布 $p_\theta(x_t | x_{<t})$，則聯合似然由機率鏈式法則分解為：
$$ L(\theta) = \prod_{t=1}^T p_\theta(x_t | x_{<t}) $$
此分解來自機率鏈式法則，不要求各 token 無條件獨立，也不額外假設固定階數的馬可夫性；一般 decoder-only Transformer 可以依賴完整可見上下文 $x_{<t}$，而非只依賴 $x_{t-1}$。對數似然為：
$$ \ell(\theta) = \sum_{t=1}^T \ln p_\theta(x_t | x_{<t}) $$

**分類問題的條件似然**：在監督分類中樣本為成對 $(x_i, y_i)$，模型輸出條件分布 $p_\theta(y_i \mid x_i)$，對數似然為 $\sum_{i=1}^N \ln p_\theta(y_i \mid x_i)$。當標籤以 one-hot 表示並對候選類別取負號時，即化為交叉熵，這正是 4.3 節推導的來源。

**總負對數似然（Total NLL）**分兩種情形定義：
$$ \text{Total NLL}(\theta) = -\sum_{i=1}^N \ln p_\theta(x_i) \quad \text{(i.i.d. 樣本)} $$
$$ \text{Total NLL}(\theta) = -\sum_{t=1}^T \ln p_\theta(x_t | x_{<t}) \quad \text{(自回歸序列)} $$
**平均負對數似然（Mean NLL）**定義為：
$$ \text{Mean NLL}(\theta) = \frac{1}{N_{\text{valid}}} \text{Total NLL}(\theta) $$
其中 $N_{\text{valid}}$ 為有效樣本或 token 數量。在有效集合預先固定且 $N_{\text{valid}}>0$ 時，Total NLL 與 Mean NLL 只差固定正比例常數，因此具有相同的最優參數，均等價於最大化似然（MLE）；若分母依賴 $\theta$（例如模型自行決定拒答或遮罩），則不保證等價。實作中必須區分三種分母：$N$（樣本數）、$N_{\text{valid}}$（有效樣本或 token 數）與 $N_{\text{token}}$（含 PAD 的總 token 數），並在文件中明示採用哪一種。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
在機器學習中，「預測得好」需要量化的標準。最直覺的想法是預測機率越高越好。然而，線性差值（如 $1-p$）在優化上存在問題：當 $p$ 接近 1 時，梯度變得很小，模型無法有效地區分「非常確定」與「完全確定」。此外，線性損失無法反映誤判的嚴重性差異：將機率從 0.1 提升到 0.2 的提升幅度，在資訊量意義上遠大於從 0.9 提升到 0.95。
<<<NEW>>>
在機器學習中，「預測得好」需要量化的標準。最直覺的想法是預測機率越高越好。然而，線性差值（如 $1-p$）有兩項缺點：其一，它不具有對數似然的乘積轉加總性，長序列聯合機率無法藉由逐項加總評估；其二，它對「將真實事件賦予接近零機率」的預測只給出有限懲罰，而 $-\ln p$ 會趨近無界，能強烈抑制過度自信的錯誤。此外，線性損失無法反映誤判的嚴重性差異：將機率從 0.1 提升到 0.2 的提升幅度，在資訊量意義上遠大於從 0.9 提升到 0.95。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    N, C = y_pred.shape
    ce_terms = np.zeros_like(y_pred, dtype=np.float64)
    
    if y_true.ndim == 1:
        # Index labels
        y_idx = np.asarray(y_true)
        if y_idx.shape != (N,):
<<<NEW>>>
    N, C = y_pred.shape
    ce_terms = np.zeros_like(y_pred, dtype=np.float64)
    
    y_raw = np.asarray(y_true)
    if y_raw.ndim == 1:
        # Index labels (Python list 亦可直接傳入)
        y_idx = y_raw
        if y_idx.shape != (N,):
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
def _as_float_array(arr, name):
    """將輸入轉換為 float64 陣列，確保後續運算精度。"""
    a = np.asarray(arr)
    if not np.issubdtype(a.dtype, np.number):
        raise TypeError(f"{name} must be numeric.")
    return a.astype(np.float64)

def validate_probability(p, name="p", axis=-1):
    """
    驗證輸入是否為合法的概率陣列。
    p: 2D array (N, C) or 1D (C,).
    """
    if p.ndim < 1:
        raise ValueError(f"{name} must be at least 1D.")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(p < 0):
        raise ValueError(f"{name} must be non-negative.")
    if np.any(p > 1.0 + 1e-12):
        raise ValueError(f"{name} must be <= 1.")
<<<NEW>>>
def _as_float_array(arr, name):
    """將輸入轉換為實數 float64 陣列，確保後續運算精度。"""
    a = np.asarray(arr)
    if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise TypeError(f"{name} must be a real numeric array.")
    return a.astype(np.float64)

def validate_probability(p, name="p", axis=-1):
    """
    驗證輸入是否為合法的概率陣列。
    p: 2D array (N, C) or 1D (C,).
    """
    if p.ndim < 1:
        raise ValueError(f"{name} must be at least 1D.")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(p < 0):
        raise ValueError(f"{name} must be non-negative.")
    if np.any(p > 1.0):
        raise ValueError(f"{name} must be <= 1.")
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
## 習題

1.  **手算**：計算 $P=[0.1, 0.2, 0.7]$ 和 $Q=[0.2, 0.3, 0.5]$ 的 KL 散度（bits）。
2.  **證明**：利用命題 4.1 與恆等式 $D_{KL}(P \| Q) = H(P, Q) - H(P)$，證明 $H(P, Q) \geq H(P)$，並分析等號成立的條件。
3.  **反例**：構造兩個 batch，其中 Batch A 有 1 個 token (NLL=10)，Batch B 有 99 個 token (NLL=0)。計算錯誤的「Perplexity 平均」與正確的「Token 加權 Perplexity」，並說明差異。
4.  **整合**：給定如下數據，計算總有效 NLL、平均 NLL (nats/bits) 及 Perplexity。
    *   Batch 1: 3 tokens. NLLs: $[1.0, 0.5, \text{PAD}]$. Valid Mask: $[1, 1, 0]$.
    *   Batch 2: 2 tokens. NLLs: $[2.0, \text{PAD}]$. Valid Mask: $[1, 0]$.
    *   請展示逐步計算過程。

## 習題解答

1.  **解答**：
    $D_{KL}(P \| Q) = 0.1\log_2(0.5) + 0.2\log_2(2/3) + 0.7\log_2(1.4)$
    $= 0.1(-1) + 0.2(-0.58496) + 0.7(0.48543)$
    $= -0.1 - 0.11699 + 0.33980 = 0.12281$ bits。

2.  **解答**：
    由恆等式 $H(P, Q) = H(P) + D_{KL}(P \| Q)$。
    根據命題 4.1，$D_{KL}(P \| Q) \geq 0$。
    因此 $H(P, Q) \geq H(P)$。
    等號成立當且僅當 $D_{KL}(P \| Q) = 0$，根據命題 4.1 的等號條件，這發生當且僅當 $P=Q$。

3.  **解答**：
    *   Batch A: NLL sum = 10, Count = 1. $PP_A = e^{10} \approx 22026.47$。
    *   Batch B: NLL sum = 0, Count = 99. $PP_B = e^0 = 1$。
    *   錯誤平均 PP: $(22026.47 + 1) / 2 \approx 11013.74$。
    *   正確 PP: Total NLL = 10, Total Count = 100. Mean NLL = 0.1. $PP_{correct} = e^{0.1} \approx 1.1052$。
    *   差異：錯誤方法被極端異常值主導，嚴重高估整體困惑度，無法反映大多數 token 的低損失。

4.  **解答**：
    *   **Batch 1 有效 NLL**: $1.0 + 0.5 = 1.5$。有效 token 數: 2。
    *   **Batch 2 有效 NLL**: $2.0$。有效 token 數: 1。
    *   **總有效 NLL**: $1.5 + 2.0 = 3.5$ nats。
    *   **總有效 token 數**: $2 + 1 = 3$。
    *   **平均 NLL (nats)**: $3.5 / 3 \approx 1.1667$ nats。
    *   **平均 NLL (bits)**: $1.1667 \times \log_2 e \approx 1.1667 \times 1.4427 \approx 1.6832$ bits。
    *   **Perplexity**: $\exp(1.1667) \approx 3.211$。
<<<NEW>>>
## 習題

1.  **手算**：計算 $P=[0.1, 0.2, 0.7]$ 和 $Q=[0.2, 0.3, 0.5]$ 的 KL 散度（bits）。
2.  **證明**：利用命題 4.1 與恆等式 $D_{KL}(P \| Q) = H(P, Q) - H(P)$，證明 $H(P, Q) \geq H(P)$，並分析等號成立的條件。
3.  **反例**：構造兩個 batch，其中 Batch A 有 1 個 token (NLL=10)，Batch B 有 99 個 token (NLL=0)。計算錯誤的「Perplexity 平均」與正確的「Token 加權 Perplexity」，並說明差異。
4.  **整合**：給定如下數據，計算總有效 NLL、平均 NLL (nats/bits) 及 Perplexity。
    *   Batch 1: 3 tokens. NLLs: $[1.0, 0.5, \text{PAD}]$. Valid Mask: $[1, 1, 0]$.
    *   Batch 2: 2 tokens. NLLs: $[2.0, \text{PAD}]$. Valid Mask: $[1, 0]$.
    *   請展示逐步計算過程。
5.  **程式**：擴充本章 `kl_divergence` 使其支援形狀 $(N, C)$ 的批次輸入並回傳形狀 $(N,)$ 的 nats 陣列。要求：(a) 沿最後類別軸 reduction；(b) 逐筆檢查支撐集，若某筆有 $P(x)>0$ 且 $Q(x)=0$ 則該筆為 `np.inf` 而不影響其他筆；(c) 禁止跨 batch 聚合（不可回傳單一純量）；(d) 對 `N=0` 或 `C=0` 拋出 `ValueError`。並為形狀、正常值、支撐集違規與拒絕跨 batch 各寫一個小型 assert 測試。

## 習題解答

1.  **解答**：
    $D_{KL}(P \| Q) = 0.1\log_2(0.5) + 0.2\log_2(2/3) + 0.7\log_2(1.4)$
    $= 0.1(-1) + 0.2(-0.58496) + 0.7(0.48543)$
    $= -0.1 - 0.11699 + 0.33980 = 0.12281$ bits。

2.  **解答**：
    由恆等式 $H(P, Q) = H(P) + D_{KL}(P \| Q)$。
    根據命題 4.1，$D_{KL}(P \| Q) \geq 0$。
    因此 $H(P, Q) \geq H(P)$。
    等號成立當且僅當 $D_{KL}(P \| Q) = 0$，根據命題 4.1 的等號條件，這發生當且僅當 $P=Q$。

3.  **解答**：
    *   Batch A: NLL sum = 10, Count = 1. $PP_A = e^{10} \approx 22026.47$。
    *   Batch B: NLL sum = 0, Count = 99. $PP_B = e^0 = 1$。
    *   錯誤平均 PP: $(22026.47 + 1) / 2 \approx 11013.74$。
    *   正確 PP: Total NLL = 10, Total Count = 100. Mean NLL = 0.1. $PP_{correct} = e^{0.1} \approx 1.1052$。
    *   差異：錯誤方法被極端異常值主導，嚴重高估整體困惑度，無法反映大多數 token 的低損失。

4.  **解答**：
    *   **Batch 1 有效 NLL**: $1.0 + 0.5 = 1.5$。有效 token 數: 2。
    *   **Batch 2 有效 NLL**: $2.0$。有效 token 數: 1。
    *   **總有效 NLL**: $1.5 + 2.0 = 3.5$ nats。
    *   **總有效 token 數**: $2 + 1 = 3$。
    *   **平均 NLL (nats)**: $3.5 / 3 \approx 1.1667$ nats。
    *   **平均 NLL (bits)**: $1.1667 \times \log_2 e \approx 1.1667 \times 1.4427 \approx 1.6832$ bits。
    *   **Perplexity**: $\exp(1.1667) \approx 3.211$。

5.  **解答**（參考實作）：
    ```python
    def kl_divergence_batch(p_true, q_model):
        """批次 KL(P||Q)，沿最後類別軸 reduction。"""
        p = _as_float_array(p_true, "p_true")
        q = _as_float_array(q_model, "q_model")
        if p.ndim != 2 or q.ndim != 2:
            raise ValueError("Expect 2D (N, C) arrays.")
        if p.shape != q.shape:
            raise ValueError("Shape mismatch.")
        N, C = p.shape
        if N == 0 or C == 0:
            raise ValueError("Empty batch or empty categories.")
        validate_probability(p, "p_true")
        validate_probability(q, "q_model")
        p_support = p > 0
        bad = np.any(p_support & (q == 0), axis=-1)
        terms = np.zeros_like(p, dtype=np.float64)
        terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
        out = np.sum(terms, axis=-1)
        out[bad] = np.inf
        return out

    # 測試
    P = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    Q = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])
    out = kl_divergence_batch(P, Q)
    assert out.shape == (2,), f"shape error: {out.shape}"
    assert np.isclose(out[0], 0.0), "first row should be 0."
    assert np.isinf(out[1]), "second row should be inf."
    try:
        kl_divergence_batch(np.zeros((0, 5)), np.zeros((0, 5)))
        assert False, "should reject empty batch"
    except ValueError:
        pass
    ```

    說明：`p_support` 逐元素判斷 $P>0$；`bad` 沿最後類別軸 reduce，只有真正違規的那一筆被設為 `inf`，符合「一筆違規不影響其他筆」的契約。函數回傳形狀 $(N,)$ 陣列，不做跨 batch 聚合，避免將不同分布的 KL 混為單一數字。`N=0` 或 `C=0` 直接拒絕；若上層需要空結果，應自行處理。
<<<END>>>