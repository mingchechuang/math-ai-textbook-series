## 獨立審稿報告

**1. 定義與證明核對**
*   **量化定義 (26.1)**：$s = \max|x|/q_{max}$。誤差界 $|x - \hat{x}| \le s/2$。
    *   **證明檢查**：$x/s \in [-q_{max}, q_{max}]$。$\text{round}(x/s)$ 產生整數 $m$。由於端點是整數且 $x/s$ 在範圍內，$m$ 不會超出 $[-q_{max}, q_{max}]$。Clip 是恆等映射。$|x - s m| = s |x/s - m| \le s/2$。邏輯正確。
    *   **零張量處理**：$s=0$ 時約定 $s=1, \hat{x}=0$。合理。
*   **蒸餾梯度 (26.2)**：$\frac{\partial \text{KL}}{\partial z_j} = \frac{q_j - p_j}{T}$。
    *   **證明檢查**：$\text{KL} = \sum p \log p - \sum p \log q$。$\frac{\partial}{\partial z_j} (-\sum p_k \log q_k) = -\sum p_k \frac{1}{q_k} \frac{\partial q_k}{\partial z_j}$。
    *   $\frac{\partial q_k}{\partial z_j} = \frac{1}{T} q_k (\delta_{kj} - q_j)$。
    *   $-\sum p_k \frac{1}{T} (\delta_{kj} - q_j) = -\frac{1}{T} (p_j - q_j \sum p_k) = -\frac{1}{T} (p_j - q_j) = \frac{q_j - p_j}{T}$。
    *   公式與證明正確。

**2. 程式碼核對**
*   **`quantize_symmetric`**：
    *   檢查 `bits` 範圍 (2-8)。
    *   `max_abs` 計算正確。
    *   `np.clip(np.round(...))` 邏輯正確。
    *   `int64` 輸出適合教學。
*   **`distill_kl_and_grad`**：
    *   `logsumexp` 穩定實作。
    *   `logp = st - logsumexp(st)`。若 `st` 有限，`logsumexp` 有限，則 `logp` 有限。
    *   **關鍵檢查**：`np.where(p > 0, p * (logp - logq), 0.0)`。
    *   NumPy 的 `np.where` 是 eager evaluation。如果 `logp` 或 `logq` 是 `-inf`（例如直接對 softmax 結果取 log 且結果為 0），會產生 NaN。但這裡 `logp` 是透過 log-sum-exp 公式計算的，通常保持有限，除非輸入極端。
    *   程式碼中明確檢查 `np.isfinite(logp)` 和 `logq`，並拋出錯誤。這比依賴 `np.where` 的副作用更安全，因為如果中間產生 NaN，`np.where` 仍會計算該分支（即使遮罩為 False），可能觸發警告或污染（雖然 NumPy 通常只選取遮罩為 True 的值，但計算仍發生）。
    *   文檔說明：「`np.where` **不是**「先判斷再乘」... 若連中間運算都不得產生非有限值，須改以有效索引取值再相乘。」這段說明非常準確且專業，指出了 NumPy 的潛在陷阱。
    *   `grad = (q - p) / (T * N)`。符合平均損失的梯度定義。
*   **梯度核對**：`manual_grads` 與 `fd_grad` 邏輯正確。`X.T @ dz` 符合矩陣微分規則。

**3. 手算與測試**
*   **手算 26.1**：$x=[-1.2, 0.4, 0.9, -0.3]$，$b=3, q_{max}=3, s=0.4$。
    *   $x/s = [-3, 1, 2.25, -0.75]$。
    *   Round: $[-3, 1, 2, -1]$。
    *   Dequant: $[-1.2, 0.4, 0.8, -0.4]$。
    *   Error: $[0, 0, 0.1, 0.1]$。Max error $0.1 \le 0.2$。正確。
*   **手算 26.2**：$z_t=[2,1,0], z_s=[0,1,2], T=2$。
    *   $p \approx [0.506, 0.307, 0.186]$。
    *   $q \approx [0.186, 0.307, 0.506]$。
    *   Grad: $(q-p)/2 \approx [-0.16, 0, 0.16]$。
    *   直觀檢查：索引0學生低，梯度負，增加 $z$；索引2學生高，梯度正，減小 $z$。正確。
*   **B1 題目與解答**：
    *   題目：「改成逐行（對 $W$ 沿第 1 軸 $D_{out}$ reduction）... 回傳尺度形狀 $(D_{in}, 1)$」。
    *   解答 B1：「逐行版本的 critical 行是 `max_abs = ... axis=1`... 誤差界：對第 $i$ 行...」。
    *   **術語核對**：
        *   $W$ 形狀 $(D_{in}, D_{out})$。
        *   Axis 1 是 $D_{out}$。沿 Axis 1 reduction 會得到形狀 $(D_{in}, 1)$。
        *   在矩陣表示中，$(i, j)$ 中 $i$ 是行 (Row)，$j$ 是列 (Column)。
        *   沿 Axis 1 (Columns) 做 reduction，得到每一**行**的統計量。結果形狀 $(D_{in}, 1)$ 對應每一行。
        *   因此，這應該是「逐行量化」(Per-Row Quantization)。
        *   解答中稱「逐行版本」且「對第 $i$ 行」，這是正確且一致的。
        *   定義 26.1 中提到「逐通道量化」（通常指 Output Channel，即 Column/Axis 1 在 PyTorch 線性層權重 $W_{out \times in}$ 或 $W_{in \times out}$ 的慣例不同）。
        *   本卷約定 $W (D_{in}, D_{out})$。
        *   Axis 0 是 $D_{in}$ (Rows)。Axis 1 是 $D_{out}$ (Columns)。
        *   `quantize_per_column` 函式使用 `axis=0`。這意味著沿 $D_{in}$ reduction。結果形狀 $(1, D_{out})$。這給每一個**列** (Output Channel) 一個 scale。這叫 Per-Column Quantization。正確。
        *   所以 B1 要求改成「逐行」，即 `axis=1`，結果 $(D_{in}, 1)$。
        *   解答 B1 稱之為「逐行」是正確的。

**4. 其他檢查**
*   資料切分、校準集使用正確。
*   無捏造 benchmark。
*   格式符合規範。

**結論**：
章節內容嚴謹，數學推導正確。程式碼實作穩健，特別是對 NumPy `np.where` 的 eager evaluation 特性的說明非常到位。B1 的術語不一致問題已修正（上一版解答錯誤地稱為「逐列」，現已改為「逐行」，與題目一致）。無明顯錯誤。

VERDICT: APPROVE