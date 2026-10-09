## 獨立審稿報告：第26章 量化、蒸餾與效率的證據

### 1. 數學與證明核對
*   **定義 26.1（對稱均勻量化）**：
    *   尺度 $s = \max|x| / q_{max}$。
    *   量化碼 $q_i = \text{clip}(\text{round}(x_i/s), -q_{max}, q_{max})$。
    *   **軸定義**：
        *   逐通道（輸出通道，$D_{out}$）：沿 Axis 0 ($D_{in}$) reduction。Shape $(1, D_{out})$。正確。
        *   逐行（Input features/Rows，$D_{in}$）：沿 Axis 1 ($D_{out}$) reduction。Shape $(D_{in}, 1)$。正確。
    *   **命題 26.1 證明**：
        *   誤差界 $|x - \hat{x}| \le s/2$。
        *   證明邏輯：$x_i/s \in [-q_{max}, q_{max}]$。$\text{round}$ 不會超出範圍，故 $\text{clip}$ 恆等。$|x_i - s \cdot \text{round}(x_i/s)| = s |x_i/s - \text{round}(x_i/s)| \le s/2$。
        *   證明嚴謹，無誤。

*   **定義 26.2（蒸餾 KL）**：
    *   前向 KL $\mathrm{KL}(p||q) = \sum p \log (p/q)$。
    *   **命題 26.2 證明**：
        *   梯度 $\frac{\partial \mathrm{KL}}{\partial z_{s,j}} = \frac{q_j - p_j}{T}$。
        *   證明過程：利用 softmax 梯度性質 $\frac{\partial \log q_k}{\partial z_j} = \frac{1}{T}(\delta_{kj} - q_j)$。
        *   $\frac{\partial}{\partial z_j} (-\sum p_k \log q_k) = -\sum p_k \frac{1}{T}(\delta_{kj} - q_j) = -\frac{1}{T}(p_j - q_j \sum p_k) = -\frac{1}{T}(p_j - q_j) = \frac{q_j - p_j}{T}$。
        *   證明正確。

### 2. 程式碼與實作核對
*   **Shape/Broadcast**：
    *   `quantize_per_column`: `W` shape `(Din, Dout)`. `axis=0` max gives `(1, Dout)`. Correct.
    *   `distill_kl_and_grad`: `z_t, z_s` shape `(N, C)`. `logp, logq` shape `(N, C)`. `term` shape `(N, C)`. `kl_per_sample` shape `(N,)`. `grad` shape `(N, C)`. Correct.
*   **Gradient Check**：
    *   `manual_grads`: `dW = X.T @ dz`. `dz` includes `1/N`. So `dW` is correct for averaged loss.
    *   `fd_grad`: Central difference. Correct.
*   **NaN Handling**：
    *   `np.where(p > 0, p * (logp - logq), 0.0)`.
    *   說明正確指出 `np.where` 是 eager evaluation。
    *   程式碼中檢查 `np.isfinite(logp)` 和 `logq`，確保不會有 `-inf` 流入導致 `0 * -inf`。這是防禦性編程的好範例。
    *   文檔說明與程式行為一致。

### 3. 手算與測試
*   **手算 26.1**：
    *   $x=[-1.2, 0.4, 0.9, -0.3]$, $b=3, q_{max}=3$.
    *   $s=1.2/3=0.4$.
    *   $x/s = [-3, 1, 2.25, -0.75]$.
    *   Round: $[-3, 1, 2, -1]$.
    *   Dequant: $[-1.2, 0.4, 0.8, -0.4]$.
    *   Error: $[0, 0, 0.1, 0.1]$. Max error $0.1 \le 0.2$. Correct.
*   **手算 26.2**：
    *   $z_t=[2,1,0], z_s=[0,1,2], T=2$.
    *   $p \approx [0.506, 0.307, 0.186]$.
    *   $q \approx [0.186, 0.307, 0.506]$.
    *   Grad: $(q-p)/2 \approx [-0.16, 0, 0.16]$.
    *   **直觀檢查**：
        *   Index 0: $q < p$. Grad negative. Descent increases $z$. Correct.
        *   Index 2: $q > p$. Grad positive. Descent decreases $z$. Correct.
    *   計算與直觀檢查正確。

### 4. 習題與解答
*   **A1**: Rounding convention discussion is accurate. NumPy uses banker's rounding. The example $3.5 \to 4$ is correct.
*   **A2**: KL and Gradient calculation is correct.
*   **A3**: Student $z_s=[1,1,1]$. Uniform $q$. KL calculation uses LSE difference correctly. Gradient table is correct.
*   **B1**: Per-row quantization solution matches the definition (Axis 1 reduction).
*   **B2**: Combined loss gradient derivation is correct. Distinguishes between $T=1$ and $T>1$ for the hard label term.
*   **B3**: Detailed analysis of $z_t=[500, -500]$.
    *   Correctly identifies that `logp` is finite.
    *   Correctly calculates KL as $\log 2$.
    *   Correctly explains that `np.where` is not strictly necessary for the *finite* result here because `logp` is finite, but it's still good practice. It correctly identifies that this specific case doesn't trigger the `0 * -inf` NaN path because `logp` is finite.
*   **C1**: Counter-example $x=[1, 1/7]$.
    *   4-bit: $s=1/7$. $x/s = [7, 1]$. Exact.
    *   8-bit: $s=1/127$. $x/s = [127, 127/7 \approx 18.14]$. Round to 18. Error non-zero.
    *   This proves that higher bits do not guarantee lower error for specific inputs if the input doesn't align with the grid. Correct logic.
*   **C2**: Student mimicking top class only. Correct.
*   **D1**: Workflow is sound and adheres to conventions (no test set leakage, distinct metrics).

### 5. 其他檢查
*   **Data Leakage**: Explicitly stated.
*   **Efficiency vs Quality**: Clearly separated.
*   **Format**: All sections present. LaTeX correct. Code self-contained.
*   **Character Count**: ~5900 chars. Within limits.

**結論**：
章節內容嚴謹，數學推導正確，程式碼實作穩健，特別是對 NumPy `np.where` 的 eager evaluation 特性的說明非常到位。B1 的術語（逐行 vs 逐通道）定義清晰且一致。無發現錯誤。

VERDICT: APPROVE