# 第04章 似然、資訊量與評估指標

## 學習目標與先備知識

本章的核心目標是建立模型評估的數學基礎。讀者需理解最大似然估計（MLE）如何從「證據」推導出「參數」，以及負對數似然（NLL）、交叉熵（Cross-Entropy, CE）、KL散度（Kullback-Leibler Divergence）與香農熵（Shannon Entropy）之間的精確數學關係。

在進入本章前，讀者應具備以下先備知識：
1.  **線性代數**：熟悉矩陣乘法、內積與轉置運算。
2.  **微積分**：理解偏導數與鏈式法則，特別是關於 $\log$ 與指數函數的微分。
3.  **基礎機率**：理解機率質量函數（PMF）的歸一化條件 $\sum_i p(x_i) = 1$ 以及支撐集（Support）的概念。
4.  **NumPy基礎**：能使用 `np.array` 進行簡單的矩陣運算與索引操作。

本章將嚴格區分「訓練誤差」與「真實能力」。我們不追求讓訓練誤差趨近於零，而是追求在驗證集與測試集上獲得穩定的泛化指標。所有計算均基於離散分類問題，以確保數值穩定性與理論清晰性。

## 問題與直覺

在機器學習中，我們常說模型預測得「好」，但「好」的定義是什麼？最直觀的想法是預測機率越高越好。然而，當機率為 0.99 與 0.999 時，直覺上的差異極小，但在數學優化上，我們需要一個可微、且能反映「誤判嚴重程度」的指標。

考慮一個二元分類問題：真實標籤是 $y=1$。
*   若模型預測 $p(y=1)=0.5$，這是一個完全不確定的預測。
*   若模型預測 $p(y=1)=0.9$，這是一個較好的預測。
*   若模型預測 $p(y=1)=0.99$，這是一個極好的預測。

直覺上，錯誤的「懲罰」應當隨著機率遠離 1 而指數級增加。這就是為什麼我們使用**對數似然**而非線性差值。$\log$ 函數具有將乘法變為加法的性質，使得在處理長序列或大量樣本時，聯合機率不會迅速下溢（underflow）為 0。

此外，資訊量（Information Content）提供了一個視角：一個事件越罕見，它發生時攜帶的資訊量越大。$-\log p(x)$ 衡量了觀察到 $x$ 時消除了多少不確定性。當 $p(x)$ 接近 1 時，$-\log p(x)$ 接近 0（幾乎沒有新資訊）；當 $p(x)$ 接近 0 時，$-\log p(x)$ 趨近於 $\infty$（巨大的資訊量或巨大的懲罰）。

## 定義、定理與推導

### 4.1 似然與最大似然估計

設隨機變數 $X$ 取自分布 $p_\theta(x)$，其中 $\theta$ 為參數。給定一個獨立同分佈（i.i.d.）樣本集 $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$，**似然函數**（Likelihood Function）定義為：
$$
L(\theta) = \mathcal{L}(\theta \mid \mathcal{D}) = \prod_{i=1}^{N} p_\theta(x_i)
$$
最大似然估計（MLE）尋求使似然函數最大化的參數 $\hat{\theta}_{MLE}$：
$$
\hat{\theta}_{MLE} = \arg\max_{\theta} \log L(\theta) = \arg\max_{\theta} \sum_{i=1}^{N} \log p_\theta(x_i)
$$
取對數是因為 $\log$ 是單調遞增函數，極值點不變，且將乘積轉為求和，便於優化。

**負對數似然（NLL）** 定義為：
$$
\text{NLL}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i)
$$
最小化 NLL 等價於最大化似然。

### 4.2 熵、交叉熵與 KL 散度

對於離散隨機變數 $X$，其**香農熵**（Shannon Entropy）定義為：
$$
H(P) = -\sum_{x \in \mathcal{X}} P(x) \log_b P(x)
$$
其中 $b$ 為對數底數（通常為 2 或 $e$）。若以 $e$ 為底，單位為 **nats**；若以 2 為底，單位為 **bits**。

**交叉熵**（Cross-Entropy）衡量的是使用分布 $Q$ 來編碼來自真實分布 $P$ 的隨機變數所需的平均長度：
$$
H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log_b Q(x)
$$
注意：若存在 $x$ 使得 $P(x) > 0$ 但 $Q(x) = 0$，則 $H(P, Q) = \infty$。

**KL散度**（Kullback-Leibler Divergence）衡量分布 $Q$ 相對於 $P$ 的「資訊損失」或「距離」：
$$
D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log_b \frac{P(x)}{Q(x)} = \sum_{x \in \mathcal{X}} P(x) (\log_b P(x) - \log_b Q(x))
$$
展開可得重要恆等式：
$$
D_{KL}(P \parallel Q) = H(P, Q) - H(P)
$$
**定理 4.1（KL散度的非負性）**
對於任何兩個分布 $P$ 和 $Q$（滿足 $P(x) > 0 \implies Q(x) > 0$），有 $D_{KL}(P \parallel Q) \geq 0$，且當且僅當 $P(x) = Q(x)$ 幾乎處處成立時，等號成立。

*證明：*
利用 Gibbs 不等式（Gibbs' Inequality）。考慮函數 $f(t) = t \log t$，其二階導數 $f''(t) = 1/t > 0$（對於 $t>0$），故 $f(t)$ 是嚴格凸函數。
根據 Jensen 不等式：
$$
D_{KL}(P \parallel Q) = -\sum_{x} P(x) \log \frac{Q(x)}{P(x)} = - \mathbb{E}_{x \sim P} \left[ \log \frac{Q(x)}{P(x)} \right]
$$
令 $Z = \frac{Q(x)}{P(x)}$，則
$$
D_{KL}(P \parallel Q) = - \mathbb{E} [\log Z] \geq - \log \mathbb{E}[Z]
$$
計算 $\mathbb{E}[Z]$：
$$
\mathbb{E}\left[\frac{Q(x)}{P(x)}\right] = \sum_{x} P(x) \frac{Q(x)}{P(x)} = \sum_{x} Q(x) = 1
$$
因此，
$$
D_{KL}(P \parallel Q) \geq - \log(1) = 0
$$
等號成立當且僅當 $Z$ 為常數，即 $\frac{Q(x)}{P(x)} = C$ 對所有 $x$ 成立。由於 $\sum Q(x) = 1$，故 $C=1$，即 $P(x) = Q(x)$。證畢。

### 4.3 困惑度（Perplexity）

在自然語言處理（NLP）或序列建模中，**困惑度**常用於評估語言模型。對於 $N$ 個 token 的序列，困惑度定義為：
$$
PP = \exp\left( \frac{1}{N} \sum_{i=1}^{N} -\log p(x_i \mid x_{<i}) \right) = \exp(\text{NLL})
$$
直覺解釋：困惑度相當於模型在每一步預測時，「有效」考慮的平均候選詞數量。若 $PP=100$，表示模型平均每步在 100 個詞之間感到同等程度的困惑。

## 逐步手算例題

### 例題 4.1：二元分類的 CE 與 NLL

假設我們有 3 個樣本，二分類標籤 $y \in \{0, 1\}$。模型預測的機率如下表：

| 樣本 ID | 真實標籤 $y$ | 預測 $P(y=1)$ | 預測 $P(y=0)$ |
| :--- | :---: | :---: | :---: |
| 1 | 1 | 0.9 | 0.1 |
| 2 | 0 | 0.1 | 0.9 |
| 3 | 1 | 0.6 | 0.4 |

**任務**：計算總 NLL（以 $\ln$ 為底，單位 nats）與平均 CE。

**步驟 1：計算每個樣本的 NLL**
對於樣本 $i$，NLL$_i = -\log P(y_i)$。

*   **樣本 1** ($y=1, p=0.9$):
    $$ -\log(0.9) \approx -(-0.10536) = 0.10536 $$
*   **樣本 2** ($y=0, p=0.9$):
    注意標籤是 0，所以我們看 $P(y=0)=0.9$。
    $$ -\log(0.9) \approx 0.10536 $$
*   **樣本 3** ($y=1, p=0.6$):
    $$ -\log(0.6) \approx -(-0.51083) = 0.51083 $$

**步驟 2：計算總 NLL 與平均 CE**
總 NLL (Sum) $= 0.10536 + 0.10536 + 0.51083 = 0.72155$
樣本數 $N=3$。
平均 CE (Mean) $= \frac{0.72155}{3} \approx 0.24052$ nats。

**步驟 3：轉換為 Bits**
若需 bits，除以 $\log_2(e) \approx 1.4427$。
$$ 0.24052 \times \log_2(e) \approx 0.3469 \text{ bits} $$

### 例題 4.2：KL 散度與零機率陷阱

設分布 $P = [0.5, 0.5, 0]$ 和 $Q = [0.5, 0.5, 0]$。
計算 $D_{KL}(P \parallel Q)$。

**手算：**
$$ D_{KL}(P \parallel Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)} $$
*   $i=1$: $0.5 \log(0.5/0.5) = 0.5 \log(1) = 0$
*   $i=2$: $0.5 \log(0.5/0.5) = 0.5 \log(1) = 0$
*   $i=3$: $0 \log(0/0)$。

**數學約定**：當 $P(x) = 0$ 時，項 $P(x) \log(P(x)/Q(x))$ 定義為 0，即使 $Q(x)=0$（只要 $P(x)=0$）。這是因為 $0 \cdot \infty$ 的極限在測度論意義下取 0（若 $P$ 絕不取該值，則該點不貢獻熵）。
因此，$D_{KL}(P \parallel Q) = 0$。

**反例**：
設 $P = [0.5, 0.5, 0]$ 和 $Q = [0.5, 0, 0.5]$。
*   $i=1$: $0$
*   $i=2$: $0.5 \log(0.5/0) \to \infty$
*   $i=3$: $0$
結果：$D_{KL}(P \parallel Q) = \infty$。
**陷阱**：在程式實作中，若直接計算 $\log(Q)$，當 $Q=0$ 時會產生 $-\infty$，導致 NaN 或 Inf。必須檢查支撐集：若 $P(x)>0$ 且 $Q(x)=0$，應直接返回 $+\infty$ 或根據業務邏輯進行拒絕/懲罰。

## 實作與程式

以下提供一個自足的 NumPy 實作，用於計算分類問題的 NLL、CE 和 KL 散度，並處理邊界情況。

```python
import numpy as np

def safe_log_probs(p):
    """
    計算 log probabilities.
    處理 p=0 的情況，返回 -inf。
    若 p 不在 [0, 1] 或 sum != 1 (tolerance 內)，拋出錯誤。
    """
    if np.any(p < -1e-12) or np.any(p > 1 + 1e-12):
        raise ValueError("Probabilities must be in [0, 1].")
    if np.abs(np.sum(p) - 1.0) > 1e-9:
        raise ValueError("Probabilities must sum to 1.")
    # Clip small negatives due to floating point errors
    p_clipped = np.clip(p, 1e-15, 1.0)
    return np.log(p_clipped)

def cross_entropy(y_true, y_pred, axis=-1):
    """
    計算交叉熵 H(P, Q).
    y_true: (N, C) one-hot 或 (N,) indices (此處假設 one-hot 以便展示)
    y_pred: (N, C) probabilities
    返回: (N,) 每個樣本的 CE
    """
    # 檢查 y_true 是否合法 one-hot
    if y_true.ndim == 1:
        # Convert indices to one-hot
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1.0
    else:
        y_true_onehot = y_true

    # 檢查 y_pred 總和
    sums = np.sum(y_pred, axis=axis)
    if np.any(np.abs(sums - 1.0) > 1e-9):
        raise ValueError("y_pred must sum to 1 along the last axis.")

    log_q = safe_log_probs(y_pred)
    
    # 關鍵：如果 P>0 且 Q=0，CE 應為 inf。
    # 在上述 safe_log_probs 中，我們 clip 了 0 到 1e-15，這是一種數值近似。
    # 更嚴格的檢查：
    mask = (y_true_onehot > 0) & (y_pred < 1e-15)
    if np.any(mask):
        return np.full(y_true.shape, np.inf)

    ce = -np.sum(y_true_onehot * log_q, axis=axis)
    return ce

def kl_divergence(p_true, q_model):
    """
    計算 KL(P || Q).
    p_true: (C,) 真實分布
    q_model: (C,) 模型分布
    """
    # 檢查支撐集
    if np.any((p_true > 1e-15) & (q_model < 1e-15)):
        return np.inf
    
    # 過濾 P>0 的項
    mask = p_true > 1e-15
    p_term = p_true[mask]
    q_term = q_model[mask]
    
    log_ratio = np.log(p_term) - np.log(q_term)
    kl = np.sum(p_term * log_ratio)
    return kl

def perplexity_from_nll(nll_per_token, total_tokens):
    """
    計算困惑度.
    nll_per_token: (N,) 每個 token 的 NLL
    total_tokens: int
    """
    if total_tokens == 0:
        raise ValueError("Total tokens must be > 0.")
    avg_nll = np.sum(nll_per_token) / total_tokens
    return np.exp(avg_nll)
```

### 程式測試策略

1.  **正常情況**：標準二分類或三分類，所有 $Q(x) > 0$。預期結果與手算一致。
2.  **邊界情況**：
    *   $Q(x) \approx 0$ 但 $P(x) > 0$：預期 CE 為 `inf` 或極大值。
    *   $P(x) = 0$ 但 $Q(x) = 0$：預期 KL 貢獻為 0。
3.  **故障情況**：
    *   $Q$ 總和不等於 1：應拋出 `ValueError`。
    *   輸入包含 NaN：`safe_log_probs` 應檢測並拋出異常。

## 測試與預期結果

我們設計以下測試用例來驗證實作與理論的一致性。

**測試 1：對稱分布**
$P=[0.5, 0.5], Q=[0.5, 0.5]$
*   預期 $H(P) = 1$ bit。
*   預期 $H(P, Q) = 1$ bit。
*   預期 $D_{KL}(P \parallel Q) = 0$。
*   *預期結果*：NumPy 計算結果應為 `[1.0]` (bits) 和 `0.0` (KL)，誤差在 $1e-10$ 內。

**測試 2：偏斜分布**
$P=[0.9, 0.1], Q=[0.5, 0.5]$
*   $H(P) = -(0.9 \log_2 0.9 + 0.1 \log_2 0.1) \approx 0.469$ bits.
*   $H(P, Q) = -(0.9 \log_2 0.5 + 0.1 \log_2 0.5) = 1.0$ bit.
*   $D_{KL} = 1.0 - 0.469 = 0.531$ bits.
*   *預期結果*：計算出的 KL 應接近 0.531。

**測試 3：零機率故障**
$P=[1.0, 0.0], Q=[0.0, 1.0]$
*   *預期結果*：`kl_divergence` 應返回 `inf`。`cross_entropy` 對應樣本應返回 `inf`。

**測試 4：困惑度**
NLL 序列為 $[0.693, 0.693]$ (即 $\ln 2$)，token 數 $N=2$。
*   $Avg NLL = 0.693$。
*   $PP = \exp(0.693) \approx 2.0$。
*   *預期結果*：返回 2.0。

## 反例與常見陷阱

1.  **使用 `np.log(q) + 1e-7` 代替安全處理**：
    添加固定的 epsilon（如 $10^{-7}$）會改變概率分布的幾何結構。當 $Q$ 接近 0 時，這會人為地限制 NLL 的最大值，導致模型低估風險。正確做法是區分「數值下溢」與「邏輯上的零機率」。若邏輯上 $Q$ 可以為 0，則必須定義 $-\log(0) = \infty$ 的處理機制（例如在損失函數中將其置為一個非常大的常數，或直接標記樣本為失敗）。
    
2.  **混淆 MLE 與 MAP**：
    MLE 尋找的是使數據可能性最大的參數。MAP（Maximum A Posteriori）考慮了先驗分布。在缺乏先驗資訊時，不應假設 MAP 優於 MLE。

3.  **對數底數混用**：
    在程式中，`np.log` 默認以 $e$ 為底（nats）。若業務需求是 bits，必須顯式轉換：`value * np.log(2)` 或 `np.log(value, 2)`。混用會導致困惑度解釋錯誤。

4.  **忽略標籤平滑（Label Smoothing）的影響**：
    如果使用標籤平滑（將 one-hot $1.0$ 改為 $1-\epsilon$，其他類改為 $\epsilon/C$），則真實分布 $P$ 變為均勻分佈的混合。此時 $H(P)$ 不再為 0，CE 的最優解也發生了變化。評估指標必須基於實際訓練所用的分布定義。

## AI、幾何與養殖案例

### AI 案例：語言模型的校準
在大型語言模型（LLM）中，模型輸出 softmax 機率。若模型對錯誤答案輸出 $P=0.99$，而對正確答案輸出 $P=0.01$，則 NLL 極高。這不僅反映預測錯誤，更反映模型的**校準（Calibration）**失敗。低 NLL 不僅意味著準確性高，也意味著模型對其自信度有正確的估計。

### 幾何視角
KL 散度 $D_{KL}(P \parallel Q)$ 在統計流形上對應於 Fisher 計量下的 geodesic 距離的近似（局部二階）。最小化 NLL 可以看作是在參數空間中尋找一個點，使得該點生成的分布 $Q_\theta$ 在 $L_1$ 或 $L_2$ 意義上（取決於損失函數形式）最接近經驗分布 $P_{emp}$。

### 養殖案例：感測數據的異常檢測
在養殖監控中，我們可能監測水溫。假設正常水溫分布為 $P \sim \mathcal{N}(25, 2^2)$。模型預測下一小時水溫分布 $Q$。
若模型預測 $Q$ 集中在 25 度，但實際發生 30 度（罕見事件），NLL 將很大。
*   **應用**：連續監控 NLL。若 NLL 持續高於歷史門檻值，觸發「異常檢測」告警。
*   **注意**：這裡的「標籤」是真實測得值（離散化後），而「預測」是模型分佈。若模型完全未見過 30 度（支撐集不包含），NLL 將趨近於無限大，這正是我們希望捕獲的「分布外（OOD）」信號。

## 習題

1.  **手算**：計算分布 $P=[0.1, 0.2, 0.7]$ 和 $Q=[0.2, 0.3, 0.5]$ 的 KL 散度（以 bits 為單位）。
2.  **證明**：證明 $H(P, Q) \geq H(P)$ 當且僅當 $D_{KL}(P \parallel Q) \geq 0$。
3.  **程式**：修改 `cross_entropy` 函數，支持 `labels` 為 index 形式 `(N,)` 而非 one-hot `(N, C)`，並確保當 `labels` 超出範圍時拋出索引錯誤。
4.  **整合**：給定 5 個樣本的 NLL 值：$[0.1, 0.2, 0.5, 0.8, 1.4]$，計算整體困惑度。

## 習題解答

1.  **解答**：
    $D_{KL}(P \parallel Q) = \sum P_i \log_2 (P_i / Q_i)$
    $i=1: 0.1 \log_2(0.5) = 0.1 \times (-1) = -0.1$
    $i=2: 0.2 \log_2(2/3) \approx 0.2 \times (-0.585) = -0.117$
    $i=3: 0.7 \log_2(1.4) \approx 0.7 \times 0.485 = 0.3395$
    總和 $= -0.1 - 0.117 + 0.3395 = 0.1225$ bits。

2.  **解答**：
    由定義 $D_{KL}(P \parallel Q) = H(P, Q) - H(P)$。
    移項得 $H(P, Q) = H(P) + D_{KL}(P \parallel Q)$。
    若 $D_{KL} \geq 0$，則 $H(P, Q) \geq H(P)$。
    反之，若 $H(P, Q) \geq H(P)$，則 $H(P, Q) - H(P) \geq 0$，即 $D_{KL} \geq 0$。
    等號成立當且僅當 $D_{KL} = 0$，即 $P=Q$。

3.  **程式解答**：
    ```python
    def cross_entropy_indices(y_true_idx, y_pred):
        if y_true_idx.ndim != 1:
            raise ValueError("y_true_idx must be 1D")
        if np.any(y_true_idx < 0) or np.any(y_true_idx >= y_pred.shape[1]):
            raise IndexError("Label index out of bounds")
        # Gather probabilities for the correct class
        probs = y_pred[np.arange(y_pred.shape[0]), y_true_idx]
        if np.any(probs <= 0):
            return np.where(probs <= 0, np.inf, -np.log(np.clip(probs, 1e-15, 1.0)))
        return -np.log(np.clip(probs, 1e-15, 1.0))
    ```

4.  **解答**：
    總 NLL $= 0.1 + 0.2 + 0.5 + 0.8 + 1.4 = 3.0$。
    樣本數 $N=5$。
    $Avg NLL = 3.0 / 5 = 0.6$。
    $PP = \exp(0.6) \approx 1.822$。

## 本章小結

本章建立了評估機率模型的核心數學工具。我們從最大似然估計出發，推導出負對數似然作為優化目標。接著，通過引入香農熵、交叉熵和 KL 散度，我們量化了模型分布與真實分布之間的差異。關鍵結論包括：
1.  NLL 最小化等價於 MLE。
2.  KL 散度非負，且僅在分布完全匹配時為零。
3.  處理零機率是評估指標實作中的關鍵挑戰，必須明確定義支撐集外部的行為。
4.  困惑度是 NLL 的指數形式，提供了直覺的「有效選擇數量」解釋。

這些概念不僅適用於分類問題，也是後續章節中訓練 Transformer 語言模型（通過 token 級的 NLL）和評估生成質量（通過 perplexity）的基石。

## 參考來源

1.  Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley. (熵與 KL 散度的經典定義與證明).
2.  Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (深度學習中的 MLE 與 Cross-Entropy 損失).
3.  Jelinek, F., Merchan, R. L., & Bahl, L. R. (1977). *Design of a Digital Speech Recognition System*. Bell System Technical Journal. (Perplexity 在語言模型評估中的早期應用).