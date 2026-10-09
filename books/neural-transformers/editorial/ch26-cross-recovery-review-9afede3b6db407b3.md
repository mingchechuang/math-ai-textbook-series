## 獨立審稿報告

### 1. 定義與證明核對
*   **量化定義 (26.1)**：
    *   $s = \max|x| / q_{max}$。
    *   $q_i = \text{clip}(\text{round}(x_i/s), -q_{max}, q_{max})$。
    *   **誤差界證明 (26.1)**：
        *   前提：$s = M/q_{max}$，其中 $M = \max|x|$。
        *   $x_i/s \in [-q_{max}, q_{max}]$。
        *   $\text{round}(x_i/s)$ 產生整數 $m_i$。
        *   若 $x_i/s$ 在 $[-q_{max}, q_{max}]$ 內，$\text{round}$ 後的值不會超出此範圍（因為端點是整數，round會保持在端點或內部）。
        *   Clip 是恆等映射。
        *   $|x_i - s \cdot m_i| = s |x_i/s - m_i| \le s \cdot 0.5$。
        *   **結論**：證明邏輯正確。
    *   **文字修正**：「量化必然引入誤差」改為「量化可能引入誤差，但落在量化格點的值也可能精確還原」。此修正正確，因為若 $x_i$ 恰好落在 $k \cdot s$ 上，誤差為 0。
*   **蒸餾梯度 (26.2)**：
    *   $\mathcal{L} = \text{KL}(p || q) = \sum p \log p - \sum p \log q$。
    *   $\frac{\partial \mathcal{L}}{\partial z_{s,j}} = \frac{q_j - p_j}{T}$。
    *   **證明檢查**：
        *   $\frac{\partial}{\partial z_j} (-\sum p_k \log q_k) = -\sum p_k \frac{1}{q_k} \frac{\partial q_k}{\partial z_j}$。
        *   $\frac{\partial q_k}{\partial z_j} = \frac{1}{T} q_k (\delta_{kj} - q_j)$。
        *   $-\sum p_k \frac{1}{T} (\delta_{kj} - q_j) = -\frac{1}{T} (p_j - q_j \sum p_k) = -\frac{1}{T} (p_j - q_j) = \frac{q_j - p_j}{T}$。
        *   **結論**：公式與證明正確。

### 2. 程式碼與實作核對
*   **Shape/Broadcast**：
    *   `quantize_per_column`: `W` is `(Din, Dout)`. `max_abs` along `axis=0` is `(1, Dout)`. Correct.
    *   `distill_kl_and_grad`: `z_t, z_s` are `(N, C)`. `logp, logq` are `(N, C)`. `term` is `(N, C)`. `kl_per_sample` is `(N,)`. `grad` is `(N, C)`. Correct.
*   **Gradient Check**：
    *   `manual_grads`: `dW = X.T @ dz`. `dz` includes `1/N`. So `dW` is correct for the averaged loss.
    *   `fd_grad`: Central difference. Correct.
*   **NaN Handling**：
    *   `np.where(p > 0, p * (logp - logq), 0.0)`.
    *   **關鍵檢查**：NumPy 的 `np.where` 會先計算兩個分支。如果 `logp` 或 `logq` 是 `-inf`（例如直接對 softmax 結果取 log 且結果為 0），會產生 NaN。但這裡 `logp` 是透過 log-sum-exp 公式計算的，通常保持有限，除非輸入極端。
    *   程式碼中明確檢查 `np.isfinite(logp)` 和 `logq`，並拋出錯誤。這比依賴 `np.where` 的副作用更安全。
    *   文檔說明：「`np.where` **不是**「先判斷再乘」... 若連中間運算都不得產生非有限值，須改以有效索引取值再相乘。」這段說明非常準確且專業。
    *   `grad = (q - p) / (T * N)`。符合平均損失的梯度定義。

### 3. 手算與測試
*   **手算 26.1**：
    *   $x=[-1.2, 0.4, 0.9, -0.3]$。
    *   $s=1.2/3=0.4$。
    *   $x/s = [-3, 1, 2.25, -0.75]$。
    *   Round: $[-3, 1, 2, -1]$。
    *   Dequant: $[-1.2, 0.4, 0.8, -0.4]$。
    *   Error: $[0, 0, 0.1, 0.1]$。Max error $0.1 \le 0.2$。
    *   **結論**：計算正確。
*   **手算 26.2**：
    *   $z_t=[2,1,0], z_s=[0,1,2], T=2$。
    *   $p \approx [0.506, 0.307, 0.186]$。
    *   $q \approx [0.186, 0.307, 0.506]$。
    *   Grad: $(q-p)/2 \approx [-0.16, 0, 0.16]$。
    *   **直觀檢查**：
        *   Index 0: $q < p$. Grad negative. Descent increases $z$. Correct.
        *   Index 2: $q > p$. Grad positive. Descent decreases $z$. Correct.
    *   **結論**：計算與直觀檢查正確。

### 4. 習題與解答核對
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
        *   Axis 1 is $D_{out}$ (Columns).
        *   Reduction along Axis 1 collapses Columns. Result is per-row.
        *   Term "逐行" (Per-Row) matches the operation `axis=1` on shape $(D_{in}, D_{out})$.
        *   The text in Definition 26.1 says: "反之，逐行量化才沿第 1 軸 reduction". This is consistent.
    *   **Conclusion**: Correct and consistent.
*   **B3**: Edge case `z_t=[500, -500]`.
    *   Explanation is precise about `logp` being finite and `0 * -inf` not occurring.
    *   It correctly states that this example does *not* demonstrate the `np.where` safety net for `0 * -inf`, because `logp` is finite.
    *   This is a high-quality pedagogical point.
*   **C1**: Counter-example for 4-bit vs 8-bit.
    *   Logic holds. Scale depends on max value. If max value is the same, 4-bit has coarser scale.
    *   The example $x=[0.51, 0.3]$ works.

### 5. 其他檢查
*   **Data Leakage**: Explicitly stated that calibration set comes from training/validation, not test.
*   **Efficiency vs Quality**: Clearly separated. No fake benchmarks.
*   **Reproducibility**: Seed used. "Expected" results stated.
*   **Format**: LaTeX correct. Code blocks correct.

### 6. 結論

章節內容嚴謹，數學推導正確，程式碼實作考慮了數值穩定性（如 `np.where` 的 eager evaluation 特性）。
**發現一個術語錯誤**：在定義 26.1 中，將「沿第 1 軸 reduction」的模式稱為「逐行量化」。
*   $W (D_{in}, D_{out})$。
*   Axis 1 是 $D_{out}$ (Columns)。
*   沿 Axis 1 Reduction 會得到每一**行**的統計量。
*   因此應稱之為「逐行量化」(Per-Row Quantization)。
*   將其稱為「逐列」會與「Per-Column」(通常指 Axis 0 reduction) 混淆。
*   **修正**：文中已寫「逐行量化才沿第 1 軸 reduction」，這是正確的。
*   **B1 題目與解答**：
    *   題目：「把 `quantize_symmetric` 改成逐行（對 $W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}}$ 沿第 1 軸 $D_{\mathrm{out}}$ reduction）版本...」
    *   解答：「逐行版本的 critical 行是 `max_abs = ... axis=1`... 誤差界：對第 $i$ 行...」
    *   **一致性檢查**：
        *   $W$ 形狀 $(D_{in}, D_{out})$。
        *   Axis 0 是 $D_{in}$ (Rows)。Axis 1 是 $D_{out}$ (Columns)。
        *   Reduction along Axis 1 collapses Columns. This gives a scale for each **Row** ($D_{in}$).
        *   So "Per-Row" quantization corresponds to reduction along Axis 1.
        *   The text says "逐行量化才沿第 1 軸 reduction". This is **Correct** terminology in the context of the matrix indices ($i$ is row index, reduced over $j$).
        *   Wait, let's re-verify standard terminology.
        *   Usually "Per-Channel" means per output channel. Output channels are columns in $W(D_{in}, D_{out})$.
        *   So Per-Channel = Per-Column. Reduction along Axis 0 (Input features).
        *   "Per-Row" means per input feature (row). Reduction along Axis 1 (Output features).
        *   The text defines: "逐通道量化" (Per-Channel/Output) uses Axis 0.
        *   "逐行量化" (Per-Row/Input) uses Axis 1.
        *   This is internally consistent and mathematically correct for the shape $(D_{in}, D_{out})$.

**Final Check:**
The previous concern about "Per-Column" vs "Per-Row" terminology was resolved in the draft.
- Axis 0 reduction on $(D_{in}, D_{out})$ -> Shape $(1, D_{out})$ -> Scale per Column (Output). Text calls this "逐通道" (Per-Channel).
- Axis 1 reduction on $(D_{in}, D_{out})$ -> Shape $(D_{in}, 1)$ -> Scale per Row (Input). Text calls this "逐行" (Per-Row).
This is consistent.

The mathematical proofs are sound. The code is robust. The pedagogical points are accurate.

VERDICT: APPROVE