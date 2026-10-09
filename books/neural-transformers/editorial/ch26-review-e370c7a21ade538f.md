## 獨立審稿報告：第26章 量化、蒸餾與效率的證據

### 1. 數學定義與證明核對

**1.1 量化誤差界 (Proposition 26.1)**
*   **定義**：$s = \max|x| / q_{max}$。
*   **誤差界**：$|x - \hat{x}| \le s/2$。
*   **證明檢查**：
    *   $x_i/s \in [-q_{max}, q_{max}]$。
    *   $\text{round}(x_i/s)$ 產生整數 $m_i$。
    *   若 $x_i/s$ 在 $[-q_{max}, q_{max}]$ 內，$\text{round}$ 後的值不會超出此範圍（因為端點是整數，round會保持在端點或內部）。
    *   Clip 是恆等映射。
    *   $|x_i - s \cdot m_i| = s |x_i/s - m_i| \le s \cdot 0.5$。
    *   **結論**：證明邏輯正確。

**1.2 蒸餾梯度 (Proposition 26.2)**
*   **損失**：$\mathcal{L} = \text{KL}(p || q) = \sum p \log p - \sum p \log q$。
*   **梯度**：$\frac{\partial \mathcal{L}}{\partial z_{s,j}} = \frac{q_j - p_j}{T}$。
*   **證明檢查**：
    *   $\frac{\partial}{\partial z_j} \sum_k p_k \log q_k = \sum_k p_k \frac{1}{q_k} \frac{\partial q_k}{\partial z_j}$。
    *   $\frac{\partial q_k}{\partial z_j} = \frac{1}{T} q_k (\delta_{kj} - q_j)$。
    *   $\sum_k p_k \frac{1}{q_k} \frac{1}{T} q_k (\delta_{kj} - q_j) = \frac{1}{T} \sum_k p_k (\delta_{kj} - q_j) = \frac{1}{T} (p_j - q_j \sum p_k) = \frac{1}{T} (p_j - q_j)$。
    *   因為 KL 是 $-\sum p \log q$ 加上常數，梯度取負號：$-\frac{1}{T}(p_j - q_j) = \frac{1}{T}(q_j - p_j)$。
    *   **結論**：公式正確。

**1.3 手算例題**
*   **26.1 (量化)**：
    *   $x=[-1.2, 0.4, 0.9, -0.3]$。
    *   $s=1.2/3=0.4$。
    *   $x/s = [-3, 1, 2.25, -0.75]$。
    *   Round: $[-3, 1, 2, -1]$。
    *   Dequant: $[-1.2, 0.4, 0.8, -0.4]$。
    *   Error: $[0, 0, 0.1, 0.1]$。Max error $0.1 \le 0.2$。
    *   **結論**：計算正確。
*   **26.2 (蒸餾)**：
    *   $z_t=[2,1,0], z_s=[0,1,2], T=2$。
    *   $p \approx [0.506, 0.307, 0.186]$。
    *   $q \approx [0.186, 0.307, 0.506]$。
    *   Grad: $(q-p)/2 \approx [-0.16, 0, 0.16]$。
    *   **結論**：計算正確。

### 2. 程式碼與實作核對

*   **Shape/Broadcast**：
    *   `quantize_per_column`: `W` is `(Din, Dout)`. `max_abs` along `axis=0` is `(1, Dout)`. Correct.
    *   `distill_kl_and_grad`: `z_t, z_s` are `(N, C)`. `logp, logq` are `(N, C)`. `term` is `(N, C)`. `kl_per_sample` is `(N,)`. `grad` is `(N, C)`. Correct.
*   **Gradient Check**：
    *   `manual_grads`: `dW = X.T @ dz`. `dz` includes `1/N`. So `dW` is correct for the averaged loss.
    *   `fd_grad`: Central difference. Correct.
*   **NaN Handling**：
    *   `np.where(p > 0, p * (logp - logq), 0.0)`.
    *   **關鍵檢查**：NumPy 的 `np.where` 會先計算兩個分支。如果 `p=0`，`logp` 可能是 `-inf`（如果直接取 log）。但這裡 `logp` 是 `st - logsumexp(st)`。
    *   如果 `st` 是有限值，`logsumexp` 是有限值，`logp` 就是有限值。
    *   只有當 `st` 極大導致 `exp` 溢位時，`logsumexp` 可能不穩，但 `st - max` 會避免溢位。
    *   文檔中說明：「`logp` 與 `logq` 是直接以縮放 logits 減去 logsumexp 計算... 不會因此必然產生 NaN」。這是非常準確的 NumPy 數值分析。
    *   程式碼中檢查了 `np.isfinite(logp)` 和 `logq`，確保不會有 NaN/Inf 流入 `np.where` 導致意外。這是一個很好的防禦性編程習慣。

### 3. 習題與解答核對

*   **A1**: $x=[-3.0, 1.5, 0.05]$, $b=4, q_{max}=7$.
    *   $s=3/7 \approx 0.4286$.
    *   $x/s = [-7, 3.5, 0.116]$.
    *   Round(3.5) in NumPy is 4 (Banker's rounding).
    *   Dequant: $[-3.0, 4*0.4286, 0] = [-3.0, 1.7144, 0]$.
    *   Error: $[0, 1.5-1.7144, 0.05] = [0, -0.2144, 0.05]$.
    *   Max error $0.2144 \le 0.4286/2 = 0.2143$?
    *   Wait. $0.21428...$ vs $0.21428...$
    *   $s = 3/7$. $s/2 = 3/14 \approx 0.2142857$.
    *   Error for 1.5: $1.5 - 4*(3/7) = 1.5 - 12/7 = 10.5/7 - 12/7 = -1.5/7 = -3/14$.
    *   Absolute value $3/14$. Bound $3/14$.
    *   It matches exactly.
    *   **Conclusion**: Correct.
*   **B1**: Per-row quantization.
    *   Question: "改成逐行... 沿第 1 軸 $D_{out}$ reduction... 尺度形狀 $(D_{in}, 1)$".
    *   Solution: "逐行版本的 critical 行是 `max_abs = ... axis=1`... 形狀 $(D_{in}, 1)$".
    *   Terminology check:
        *   $W$ is $(D_{in}, D_{out})$.
        *   Axis 0 is $D_{in}$ (Rows). Axis 1 is $D_{out}$ (Columns).
        *   Reduction along Axis 1 collapses Columns. Result is per-row.
        *   Term "逐行" (Per-Row) matches the operation `axis=1` on shape $(D_{in}, D_{out})$.
        *   Term "逐列" (Per-Column) usually implies reduction along Axis 0 (collapsing Rows) to get per-column scales, which corresponds to `axis=0`.
        *   The text in Definition 26.1 says: "若把同一條公式套到矩陣... 的每一欄（輸出通道）上... 這就是「逐通道量化」... 反之，逐列量化才沿第 1 軸 reduction".
        *   Wait, let's re-read Definition 26.1 carefully.
        *   "若把同一條公式套到矩陣 $W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}}$ 的每一欄（輸出通道）上... 對 $W$ 沿第 0 軸（$D_{\mathrm{in}}$）取 $\max$ ... 這就是「逐通道量化」... 反之，逐列量化才沿第 1 軸 reduction..."
        *   There is a contradiction in terms here.
        *   Standard terminology in ML (e.g., PyTorch `QuantStub`):
            *   `QuantPerTensor`: Global scale.
            *   `QuantPerAxis`: Usually axis 0 of the weight tensor. If $W$ is $(In, Out)$, axis 0 is `In`. So it's per-input-channel?
            *   Let's stick to the matrix indices defined in the volume.
            *   $W (D_{in}, D_{out})$.
            *   "每一欄" (Every Column) corresponds to $D_{out}$. To get a scale for each column, we must reduce along the rows (Axis 0).
            *   So "逐通道" (Per-Channel, assuming Channel=Output) $\implies$ Reduction along Axis 0. Scale shape $(1, D_{out})$.
            *   "逐列" (Per-Column) in Chinese matrix terms usually means "For each column". This implies reduction along Axis 0.
            *   However, the text says: "反之，逐列量化才沿第 1 軸 reduction".
            *   Reduction along Axis 1 ($D_{out}$) collapses the columns. This gives a scale for each **Row** ($D_{in}$).
            *   So "逐列量化" (Quantizing by columns?) is confusing. If you reduce along the column axis (axis 1), you are aggregating the columns to get a row-wise statistic. This is "Per-Row" quantization.
            *   The term "逐列" (Zhu Lie) literally means "Per Column". If you want Per-Column quantization, you should reduce along Axis 0.
            *   The text says "逐列量化才沿第 1 軸 reduction". This is semantically inconsistent. Reduction along Axis 1 produces Per-Row scales.
            *   Let's check B1 again.
            *   B1 asks for "逐行" (Per-Row). Solution says "逐行版本". Consistent.
            *   The issue is in Definition 26.1's terminology for the *alternative*.
            *   "反之，逐列量化才沿第 1 軸 reduction".
            *   If "逐列" means "Per-Column" (one scale per column), it should be Axis 0.
            *   If it means "Reduction along the column axis", it is Axis 1, but the result is Per-Row.
            *   Given the context of "Per-Channel" (Output) being the standard (Axis 0), the alternative is usually "Per-Input" or "Per-Row" (Axis 1).
            *   The label "逐列量化" for Axis 1 reduction is a **terminology error**. It should be "逐行量化" (Per-Row Quantization) or "沿輸出維度約化" (Reduction along output dimension).
            *   Wait, let's look at the Chinese phrasing again.
            *   "若把...套到...每一欄（輸出通道）上... 這就是「逐通道量化」" -> Per-Channel (Output). Axis 0. Correct.
            *   "反之，逐列量化才沿第 1 軸 reduction" -> This implies that the *other* mode is called "逐列".
            *   But Axis 1 reduction is Per-Row.
            *   So calling Axis 1 reduction "逐列" is wrong. It should be "逐行".
            *   And calling Axis 0 reduction "逐通道" is okay if Channel=Output.
            *   So the error is in the label of the Axis 1 reduction mode.
            *   **Correction**: Change "逐列量化" in Definition 26.1 to "逐行量化" (or Per-Input-Channel if referring to standard ML terms, but "Per-Row" is safer given the matrix definition).
            *   Also, in B1, the question asks for "逐行" (Per-Row). This is consistent with the corrected terminology.
            *   So the fix is in Definition 26.1.

*   **B3**: Edge case `z_t=[500, -500]`.
    *   Explanation is precise about `logp` being finite and `0 * -inf` not occurring.
    *   It correctly states that this example does *not* demonstrate the `np.where` safety net for `0 * -inf`, because `logp` is finite.
    *   This is a high-quality pedagogical point.

*   **C1**: Counter-example for 4-bit vs 8-bit.
    *   Logic holds. Scale depends on max value. If max value is the same, 4-bit has coarser scale.
    *   The example $x=[0.51, 0.3]$ works.

### 4. 其他檢查

*   **Data Leakage**: Explicitly stated that calibration set comes from training/validation, not test.
*   **Efficiency vs Quality**: Clearly separated. No fake benchmarks.
*   **Reproducibility**: Seed used. "Expected" results stated.
*   **Format**: LaTeX correct. Code blocks correct.

### 5. 結論

章節內容嚴謹，數學推導正確，程式碼實作考慮了數值穩定性（如 `np.where` 的 eager evaluation 特性）。
**發現一個術語錯誤**：在定義 26.1 中，將「沿第 1 軸 reduction」的模式稱為「逐列量化」。
*   $W (D_{in}, D_{out})$。
*   Axis 1 是 $D_{out}$ (Columns)。
*   沿 Axis 1 Reduction 會得到每一**行**的統計量。
*   因此應稱之為「逐行量化」(Per-Row Quantization)。
*   將其稱為「逐列」會與「Per-Column」(通常指 Axis 0 reduction) 混淆。
*   **最小修法**：在定義 26.1 中，將「反之，逐列量化才沿第 1 軸 reduction」修改為「反之，**逐行**量化才沿第 1 軸 reduction」。

VERDICT: REVISE