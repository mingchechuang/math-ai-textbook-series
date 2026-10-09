# 第04章 似然、資訊量與評估指標

## 學習目標與先備知識

本章旨在建立評估機率模型的數學基礎與實作規範。讀者需理解最大似然估計（MLE）如何從數據證據推導參數，並掌握負對數似然（NLL）、交叉熵（Cross-Entropy, CE）、KL散度（Kullback-Leibler Divergence）與香農熵（Shannon Entropy）的嚴格數學定義及其在優化中的角色。

本章的核心契約是**精確性**。我們將區分「數學上的零機率」與「數值計算中的下溢」，明確定義當分布支撐集不一致時的行為（通常導致指標為無限大）。同時，我們將統一單位系統，區分 nats（自然對數）與 bits（以 2 為底），並糾正常見的單位轉換錯誤。

**先備知識**：
1.  **離散機率**：理解概率質量函數（PMF）的歸一化 $\sum p_i=1$ 及支撐集概念。
2.  **微積分**：掌握鏈式法則及 $\ln(x)$ 的導數。
3.  **NumPy**：熟悉陣列運算、軸（axis）指定及布林索引（Boolean Indexing）。
4.  **線性代數**：理解向量內積與矩陣轉置在梯度計算中的應用。

本章所有計算均基於離散分類問題或離散化的序列模型，以確保理論的清晰性與數值的穩定性。我們不追求訓練誤差的絕對最小化，而是關注指標的數學正確性與泛化能力的評估邏輯。

## 問題與直覺

在機器學習中，「預測得好」需要量化的標準。最直覺的想法是預測機率越高越好。然而，線性差值（如 $1-p$）在優化上存在問題：當 $p$ 接近 1 時，梯度變得很小，模型無法有效地區分「非常確定」與「完全確定」。

我們使用**對數似然**作為基礎，因為：
1.  **乘法變加法**：獨立事件的聯合機率是乘積，取對數後變為求和，便於處理長序列並避免數值下溢。
2.  **懲罰過度自信**：當真實標籤的機率 $p$ 趨近於 0 時，$-\ln p$ 無界增大。這意味著模型若對錯誤答案給予極低機率，會受到嚴厲懲罰；反之，若對正確答案給予高機率，懲罰趨近於 0。
3.  **資訊解釋**：$-\ln p(x)$ 代表觀察到 $x$ 時消除的不確定性（資訊量）。

**注意**：損失隨機率降低而「無界增大」，並非「指數級增加」。準確的描述是：損失是機率的對數反比。

## 定義、定理與推導

### 4.1 似然與負對數似然

設樣本集 $\mathcal{D} = \{x_1, \dots, x_N\}$ 來自分布 $p_\theta(x)$。**似然函數**為：
$$ L(\theta) = \prod_{i=1}^N p_\theta(x_i) $$
**對數似然**為：
$$ \ell(\theta) = \ln L(\theta) = \sum_{i=1}^N \ln p_\theta(x_i) $$
**總負對數似然（Total NLL）**定義為：
$$ \text{Total NLL}(\theta) = -\sum_{i=1}^N \ln p_\theta(x_i) $$
**平均負對數似然（Mean NLL）**定義為：
$$ \text{Mean NLL}(\theta) = -\frac{1}{N} \sum_{i=1}^N \ln p_\theta(x_i) $$
最小化 Total NLL 或 Mean NLL 均等價於最大化似然（MLE）。

### 4.2 熵、交叉熵與 KL 散度

**香農熵** $H(P)$ 衡量分布 $P$ 的內在不確定性：
$$ H(P) = -\sum_{x} P(x) \ln P(x) $$
**交叉熵** $H(P, Q)$ 衡量用分布 $Q$ 編碼來自 $P$ 的資料所需的平均長度：
$$ H(P, Q) = -\sum_{x} P(x) \ln Q(x) $$
**KL 散度** $D_{KL}(P \| Q)$ 衡量 $Q$ 與 $P$ 的差異：
$$ D_{KL}(P \| Q) = \sum_{x} P(x) \ln \frac{P(x)}{Q(x)} = H(P, Q) - H(P) $$

**支撐集規則**：
1. 若 $P(x) = 0$，則項 $P(x)\ln(P(x)/Q(x))$ 定義為 0（即使 $Q(x)=0$）。
2. 若 $P(x) > 0$ 且 $Q(x) = 0$，則 $D_{KL}(P \| Q) = \infty$。

### 4.3 定理：KL 散度的非負性

**命題 4.1**：對於任意兩個概率分布 $P, Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。

**證明**：
令 $S = \{x \mid P(x) > 0\}$ 為 $P$ 的支撐集。
$$ D_{KL}(P \| Q) = \sum_{x \in S} P(x) \ln \frac{P(x)}{Q(x)} = -\sum_{x \in S} P(x) \ln \frac{Q(x)}{P(x)} $$
利用 Jensen 不等式（$\ln$ 為凹函數）：
$$ -\sum_{x \in S} P(x) \ln \frac{Q(x)}{P(x)} \geq -\ln \left( \sum_{x \in S} P(x) \frac{Q(x)}{P(x)} \right) $$
計算期望項：
$$ \sum_{x \in S} P(x) \frac{Q(x)}{P(x)} = \sum_{x \in S} Q(x) $$
因為 $Q$ 是概率分布，$\sum_{x \in S} Q(x) \leq 1$。
故：
$$ D_{KL}(P \| Q) \geq -\ln \left( \sum_{x \in S} Q(x) \right) \geq -\ln(1) = 0 $$
**等號條件分析**：
1. Jensen 不等式取等號當且僅當 $\frac{Q(x)}{P(x)}$ 在 $S$ 上為常數，設為 $C$。即 $Q(x) = C P(x)$ 對所有 $x \in S$。
2. 第二個不等式取等號當且僅當 $\sum_{x \in S} Q(x) = 1$。這意味著 $Q$ 在 $S$ 的補集上質量為 0。
合併兩點：$Q(x) = C P(x)$ 且 $\sum_{S} C P(x) = C \cdot 1 = 1 \implies C=1$。
因此 $Q(x) = P(x)$ 對所有 $x \in S$。又在 $S$ 外 $P(x)=0$ 且 $Q(x)=0$，故全域 $P=Q$。證畢。

### 4.4 單位轉換：Nats 與 Bits

*   **Nats**：使用自然對數 $\ln$。單位：nats。
*   **Bits**：使用以 2 為底對數 $\log_2$。單位：bits。

轉換關係：
$$ \text{Value}_{\text{bits}} = \frac{\text{Value}_{\text{nats}}}{\ln 2} = \text{Value}_{\text{nats}} \cdot \log_2 e $$
注意：$\log_2 e \approx 1.4427$。若從 nats 轉 bits，應**除以** $\ln 2$（約 0.6931）或**乘以** $\log_2 e$（約 1.4427）。切勿混淆除法與乘法的方向。

### 4.5 困惑度（Perplexity）

對於序列模型，困惑度定義為有效 token 的平均 NLL 的指數：
$$ PP = \exp(\text{Mean NLL}_{\text{valid}}) = \exp\left( \frac{\sum_{t \in \text{valid}} -\ln p(x_t | x_{<t})}{|\text{valid}|} \right) $$
**直覺限制**：PP 等於「等效候選詞數量」僅在模型輸出均勻分布的特殊情況下精確成立。一般情況下，它是平均對數損失的指數，反映模型的整體預測困難度。

## 逐步手算例題

### 例題 4.1：二分類 CE 與單位轉換

假設 3 個樣本，真實標籤與預測機率如下：
1.  $y=1, p_1=0.9$
2.  $y=0, p_0=0.9$
3.  $y=1, p_1=0.6$

**步驟 1：計算每個樣本的 NLL (nats)**
*   $L_1 = -\ln(0.9) \approx 0.10536$
*   $L_2 = -\ln(0.9) \approx 0.10536$
*   $L_3 = -\ln(0.6) \approx 0.51083$

**步驟 2：計算總 NLL 與平均 NLL**
*   Total NLL $= 0.10536 + 0.10536 + 0.51083 = 0.72155$ nats
*   Mean NLL $= 0.72155 / 3 \approx 0.24052$ nats

**步驟 3：轉換為 Bits**
*   Mean NLL (bits) $= 0.24052 \times \log_2 e \approx 0.24052 \times 1.4427 \approx 0.3470$ bits
*   或 Mean NLL (bits) $= 0.24052 / \ln 2 \approx 0.24052 / 0.6931 \approx 0.3470$ bits

### 例題 4.2：KL 散度與零機率

**案例 A**：$P=[0.5, 0.5, 0], Q=[0.5, 0.5, 0]$
*   $D_{KL} = 0.5\ln(1) + 0.5\ln(1) + 0\ln(0/0) = 0 + 0 + 0 = 0$。
*   解釋：$P=Q$，無差異。

**案例 B**：$P=[0.5, 0.5, 0], Q=[0.5, 0, 0.5]$
*   $i=1: 0.5\ln(1) = 0$
*   $i=2: 0.5\ln(0.5/0) \to \infty$
*   $i=3: 0$
*   結果：$D_{KL} = \infty$。
*   解釋：$P$ 在索引 2 有質量，但 $Q$ 視為該事件不可能，資訊損失無限。

## 實作與程式

以下 NumPy 實作嚴格遵循數學定義，區分精確零與數值下溢，並處理形狀驗證。

```python
import numpy as np

def validate_probability(p, name="p"):
    """驗證輸入是否為合法的概率陣列。"""
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(p < -1e-12):
        raise ValueError(f"{name} must be non-negative.")
    if np.any(p > 1.0 + 1e-12):
        raise ValueError(f"{name} must be <= 1.")
    # 沿最後軸檢查歸一化
    sums = np.sum(p, axis=-1)
    if np.any(np.abs(sums - 1.0) > 1e-9):
        raise ValueError(f"{name} rows must sum to 1. Got sums: {sums}")

def safe_log_prob(p):
    """
    計算 log p。
    p=0 時返回 -inf。p<0 拋錯。
    """
    if np.any(p < 0):
        raise ValueError("Probability cannot be negative.")
    # 精確處理零
    out = np.full_like(p, -np.inf, dtype=float)
    pos_mask = p > 0
    out[pos_mask] = np.log(p[pos_mask])
    return out

def cross_entropy(y_true, y_pred):
    """
    計算交叉熵。
    y_true: (N, C) one-hot 或 (N,) index
    y_pred: (N, C) probability
    返回: (N,) CE per sample (nats)
    """
    # 驗證 y_pred
    validate_probability(y_pred, "y_pred")
    
    # 處理 y_true
    if y_true.ndim == 1:
        if y_true.dtype.kind != 'i':
            raise ValueError("y_true indices must be integer.")
        if np.any(y_true < 0) or np.any(y_true >= y_pred.shape[1]):
            raise IndexError("Label index out of bounds.")
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1.0
    else:
        y_true_onehot = y_true
        if y_true_onehot.shape != y_pred.shape:
            raise ValueError("Shape mismatch between y_true and y_pred.")
        validate_probability(y_true_onehot, "y_true")

    # 計算 log q
    log_q = safe_log_prob(y_pred)
    
    # CE = -sum(P * log Q)
    # 注意：若 P>0 且 Q=0，log_q 為 -inf，乘積為 inf，負號後為 inf。
    # 若 P=0，項為 0 * (-inf)，在 NumPy 中 0 * -inf = nan。
    # 因此必須先掩蔽 P=0 的位置。
    ce_terms = np.zeros_like(y_pred)
    mask = y_true_onehot > 0
    ce_terms[mask] = y_true_onehot[mask] * log_q[mask]
    
    # 檢查是否有 P>0 但 Q=0 的情況
    bad_mask = (y_true_onehot > 0) & (y_pred == 0)
    if np.any(bad_mask):
        # 逐個樣本檢查
        ce_per_sample = -np.sum(ce_terms, axis=-1)
        sample_bad = np.any(bad_mask, axis=-1)
        ce_per_sample[sample_bad] = np.inf
        return ce_per_sample

    ce_per_sample = -np.sum(ce_terms, axis=-1)
    return ce_per_sample

def kl_divergence(p_true, q_model):
    """
    計算 KL(P || Q)。
    p_true, q_model: (C,) 1D probability vectors.
    返回: float (nats) or inf.
    """
    # 驗證
    if p_true.shape != q_model.shape:
        raise ValueError("Shape mismatch.")
    validate_probability(p_true, "p_true")
    validate_probability(q_model, "q_model")
    
    # 支撐集檢查
    p_support = p_true > 0
    if np.any(p_support & (q_model == 0)):
        return np.inf

    # 計算
    terms = np.zeros_like(p_true)
    terms[p_support] = p_true[p_support] * np.log(p_true[p_support] / q_model[p_support])
    return float(np.sum(terms))

def perplexity(nll_values, valid_mask):
    """
    計算困惑度。
    nll_values: (N,) array of NLL per token.
    valid_mask: (N,) boolean mask for valid tokens.
    返回: float.
    """
    if nll_values.shape != valid_mask.shape:
        raise ValueError("Shape mismatch.")
    count = np.sum(valid_mask)
    if count == 0:
        raise ValueError("No valid tokens.")
    if not np.all(np.isfinite(nll_values[valid_mask])):
        # 若有效 token 包含 inf，PP 為 inf
        if np.any(nll_values[valid_mask] == np.inf):
            return np.inf
        raise ValueError("Non-finite NLL in valid tokens.")
    
    sum_nll = np.sum(nll_values[valid_mask])
    mean_nll = sum_nll / count
    return np.exp(mean_nll)
```

## 測試與預期結果

以下測試驗證程式的正確性。

1.  **正常 CE**：
    *   輸入：`y_pred=[[0.9, 0.1], [0.9, 0.1], [0.6, 0.4]]`, `y_true=[1, 0, 1]` (注意索引 1 對應 0.9, 索引 0 對應 0.1? 不，`y_pred` 第一列 `[0.9, 0.1]`，若 `y_true=1`，取第二個元素 0.1? 不對，通常 `y_pred` 的列是類別機率。若 `y_true=1`，取 `y_pred[:, 1]`。
    *   修正測試數據以符合例題 4.1：
        *   Sample 1: $y=1, p_1=0.9 \implies y\_pred[0] = [0.1, 0.9]$
        *   Sample 2: $y=0, p_0=0.9 \implies y\_pred[1] = [0.9, 0.1]$
        *   Sample 3: $y=1, p_1=0.6 \implies y\_pred[2] = [0.4, 0.6]$
    *   預期輸出：`[0.10536, 0.10536, 0.51083]`。

2.  **零機率 KL**：
    *   $P=[0.5, 0.5, 0], Q=[0.5, 0, 0.5]$。
    *   預期輸出：`np.inf`。

3.  **NaN 拒絕**：
    *   `y_pred` 包含 `np.nan`。
    *   預期：拋出 `ValueError`。

4.  **困惑度**：
    *   `nll=[0.693, 0.693]`, `mask=[True, True]`。
    *   預期：`exp(0.693) ≈ 2.0`。

## 反例與常見陷阱

1.  **錯誤的 Bits 轉換**：
    *   陷阱：將 nats 除以 $\log_2 e$ (≈1.44) 來得到 bits。
    *   正確：nats 除以 $\ln 2$ (≈0.69) 或乘以 $\log_2 e$ (≈1.44)。
    *   反例：$1$ nat $= 1/0.693 \approx 1.44$ bits。若錯誤除法，會得到 $0.69$ bits，嚴重低估不確定性。

2.  **Perplexity 的平均錯誤**：
    *   陷阱：計算每個 batch 的 perplexity，然後對 batch 的 perplexity 做算術平均。
    *   正確：必須將所有 batch 的 NLL 總和相加，除以**總有效 token 數**，再指數化。
    *   反例：Batch 1 (10 tokens, NLL sum 10, PP $e^1$), Batch 2 (100 tokens, NLL sum 100, PP $e^1$)。
        *   錯誤平均 PP：$(e^1 + e^1)/2 = e^1$。
        *   正確 PP：$\exp((10+100)/(10+100)) = e^1$。此例巧合相同，但若 NLL 不同，例如 B1 (NLL sum 5, 10 tok), B2 (NLL sum 50, 100 tok)。
        *   錯誤：$(e^{0.5} + e^{0.5})/2 = e^{0.5}$。
        *   正確：$\exp(55/110) = e^{0.5}$。
        *   更極端反例：B1 (NLL 10, 1 tok, PP $e^{10}$), B2 (NLL 10, 100 tok, PP $e^{0.1}$)。
        *   錯誤平均：$(e^{10} + e^{0.1})/2 \approx e^{10}/2$。
        *   正確：$\exp(20/101) \approx e^{0.2}$。錯誤估計會高估模型的困惑度。

3.  **零機率的 NumPy NaN**：
    *   陷阱：直接計算 `np.sum(p * np.log(q))`。若 $p=0, q=0$，則 $0 \times -\infty = \text{NaN}$。
    *   正確：必須使用掩蔽（masking），只計算 $p>0$ 的項。

## AI、幾何與養殖案例

### AI 案例：校準與 NLL
NLL 是適當的評分函數（Proper Scoring Rule）。在真分布期望下，NLL 最小化對應於學習真條件分布。然而，有限測試集上的低 NLL **不單獨證明**模型校準良好或具有高準確率。高 NLL 可能來自少數高信心錯誤。校準評估應使用 ECE（Expected Calibration Error）等專門指標。

### 幾何視角
在統計流形中，KL 散度的局部二階近似與 Fisher 資訊矩陣相關：
$$ D_{KL}(p_\theta \| p_{\theta+\delta}) \approx \frac{1}{2} \delta^T I(\theta) \delta $$
這表明 NLL 最小化是在 Fisher 計量下尋找最接近經驗分布的模型參數。注意：KL 不對稱，因此這不是一個對稱距離。

### 養殖案例：離散異常檢測
假設水溫分為 5 個箱（Bin）：[10-15, 15-20, 20-25, 25-30, 30-35]。
正常分布 $P=[0.01, 0.1, 0.6, 0.29, 0.0]$。
模型預測下一小時 $Q=[0.05, 0.1, 0.6, 0.25, 0.0]$。
若實際落入 Bin 5（30-35度），而 $P_5=0$，則該事件 NLL 為 $\infty$。
在實作中，若訓練資料從未出現 Bin 5，模型可能預測極小值而非 0。若預測 $Q_5=10^{-6}$，NLL $= -\ln(10^{-6}) \approx 13.8$。這是一個高風險信號，觸發人工審查，而非自動判定故障，因為有限資料無法證明該事件絕對不可能。

## 習題

1.  **手算**：計算 $P=[0.1, 0.2, 0.7]$ 和 $Q=[0.2, 0.3, 0.5]$ 的 KL 散度（bits）。
2.  **證明**：證明 $H(P, Q) \geq H(P)$ 當且僅當 $D_{KL}(P \| Q) \geq 0$。
3.  **反例**：構造兩個 batch，其中 Batch A 有 1 個 token (NLL=10)，Batch B 有 99 個 token (NLL=0)。計算錯誤的「Perplexity 平均」與正確的「Token 加權 Perplexity」，並說明差異。
4.  **程式**：修改 `cross_entropy`，使其支持 Soft Labels（即 $y\_true$ 可以是任意和為 1 的分布，用於知識蒸餾），並驗證輸入歸一化。

## 習題解答

1.  **解答**：
    $D_{KL}(P \| Q) = 0.1\log_2(0.5) + 0.2\log_2(2/3) + 0.7\log_2(1.4)$
    $= 0.1(-1) + 0.2(-0.58496) + 0.7(0.48543)$
    $= -0.1 - 0.11699 + 0.33980 = 0.12281$ bits。

2.  **解答**：
    由定義 $D_{KL}(P \| Q) = H(P, Q) - H(P)$。
    $H(P, Q) \geq H(P) \iff H(P, Q) - H(P) \geq 0 \iff D_{KL}(P \| Q) \geq 0$。
    由命題 4.1 知 $D_{KL} \geq 0$ 恆成立，故 $H(P, Q) \geq H(P)$ 恆成立，等號當且僅當 $P=Q$。

3.  **解答**：
    *   Batch A: NLL sum = 10, Count = 1. $PP_A = e^{10} \approx 22026$。
    *   Batch B: NLL sum = 0, Count = 99. $PP_B = e^0 = 1$。
    *   錯誤平均 PP: $(22026 + 1) / 2 \approx 11013.5$。
    *   正確 PP: Total NLL = 10, Total Count = 100. Mean NLL = 0.1. $PP_{correct} = e^{0.1} \approx 1.105$。
    *   差異：錯誤方法被極端異常值主導，嚴重高估整體困惑度。

4.  **程式解答**：
    ```python
    def cross_entropy_soft(y_true_soft, y_pred):
        if y_true_soft.shape != y_pred.shape:
            raise ValueError("Shape mismatch")
        validate_probability(y_true_soft, "y_true_soft")
        validate_probability(y_pred, "y_pred")
        
        log_q = safe_log_prob(y_pred)
        # 掩蔽 P=0 以避免 0 * -inf
        terms = np.where(y_true_soft > 0, y_true_soft * log_q, 0)
        ce = -np.sum(terms, axis=-1)
        
        # 檢查是否有 P>0 但 Q=0
        bad = np.any((y_true_soft > 0) & (y_pred == 0), axis=-1)
        ce[bad] = np.inf
        return ce
    ```

## 本章小結

本章確立了似然、資訊量與評估指標的嚴格數學基礎。我們證明 KL 散度非負，並明確處理了零機率與支撐集不一致的邊界情況。實作部分強調了 NumPy 中避免 NaN 的掩蔽技術，以及 nats/bits 的正確轉換。困惑度的計算必須基於全局有效 token 的 NLL 平均，而非批次 perplexity 的簡單平均。這些原則是後續 Transformer 訓練與評估的基石。

## 參考來源

1.  Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley. (KL 散度與熵的標準參考)
2.  Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (NLL 與梯度下降的應用)
3.  Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (Draft). (Perplexity 在語言模型中的定義)

*註：以上來源為標準學術參考，本章未執行外部程式驗證其內容，僅引用其定義作為數學依據。*