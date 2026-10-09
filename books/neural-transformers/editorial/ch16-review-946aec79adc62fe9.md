# 審稿報告：第16章 多頭注意力與拆分合併

## 1. 核心數學與定義核對

### 1.1 拆分/合併互逆證明（命題 16.1）
- **重算**：`split` = `reshape(B,T,D) -> (B,T,H,dh)` 再 `transpose(0,2,1,3)`。`merge` = `transpose(0,2,1,3)` 再 `reshape(B,T,H*dh)`。
- **驗證**：`transpose` 是自逆操作（involution），`reshape` 與 `reshape'` 在 row-major 下互為逆（前提是元素線性索引不變）。
- **結論**：證明邏輯正確。$merge(split(X)) = X$ 成立。

### 1.2 雙頭注意力手算（手算 2）
- **設定**：$B=1, T=2, D=4, H=2, dh=2$。$X = \begin{bmatrix} 1 & 2 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}$。$W_Q=W_K=W_V=W_O=I$。
- **拆分**：
    - Head 0 ($h=0$): 取 $X$ 的前 $dh=2$ 維。
        - $Q_h^0 = \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}$
        - $K_h^0 = \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}$
        - $V_h^0 = \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}$
    - Head 1 ($h=1$): 取 $X$ 的後 $dh=2$ 維。
        - $Q_h^1 = \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix}$
        - $K_h^1 = \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix}$
        - $V_h^1 = \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix}$
- **Head 0 計算**：
    - $S_0 = \frac{1}{\sqrt{2}} Q_h^0 (K_h^0)^\top = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 2 & 0 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 5 & 0 \\ 0 & 0 \end{bmatrix}$。
    - $S_0 = \begin{bmatrix} 3.5355 & 0 \\ 0 & 0 \end{bmatrix}$。
    - Softmax Row 0: $e^{3.5355} \approx 34.31$, Sum $\approx 35.31$. $34.31/35.31 \approx 0.9717$.
    - Softmax Row 1: $[0,0] \to [0.5, 0.5]$.
    - $O_h^0 = A_0 V_h^0$.
        - Row 0: $0.9717 [1,2] + 0.0283 [0,0] = [0.9717, 1.9434]$.
        - Row 1: $0.5 [1,2] + 0.5 [0,0] = [0.5, 1.0]$.
    - 結果正確。
- **Head 1 計算**：
    - $S_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 0 & 1 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 0 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 1.4142 \end{bmatrix}$.
    - Softmax Row 0: $[0,0] \to [0.5, 0.5]$.
    - Softmax Row 1: $[0, 1.4142]$. Max=1.4142.
        - $exp(0-1.4142) \approx 0.2431$. $exp(0)=1$.
        - $A[1,0] \approx 0.1955, A[1,1] \approx 0.8044$. (原文用未減max公式算出 0.1956, 0.8044，數值一致)
    - $O_h^1 = A_1 V_h^1$. $V_h^1 = \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix}$.
        - Row 0: $0.5[0,0] + 0.5[1,1] = [0.5, 0.5]$.
        - Row 1: $0.1956[0,0] + 0.8044[1,1] = [0.8044, 0.8044]$.
    - 結果正確。

### 1.3 程式碼與形狀廣播
- `split_heads`: `X.reshape(B, T, H, dh).transpose(0, 2, 1, 3)`。正確。
- `merge_heads`: `Y.transpose(0, 2, 1, 3).reshape(B, T, H * dh)`。正確。
- `forward`:
    - `Qh @ Kh.transpose(0, 1, 3, 2)`。`(B,H,T,dh) @ (B,H,dh,T) -> (B,H,T,T)`。正確。
    - `A @ Vh`。`(B,H,T,T) @ (B,H,T,dh) -> (B,H,T,dh)`。正確。
    - `Y = O @ self.W_O + self.b_O`。`(B,T,D) @ (D, D_out)`。正確。
- **Mask 處理**：
    - `np.where(m, S, -np.inf)`。True=允許，保留 S；False=遮罩，置 -inf。符合約定。
    - 全遮罩檢查：`np.any(np.all(~m, axis=-1))`。如果某一列全為 False，則拋錯。正確。
    - 程式碼在 softmax **之前** 檢查並拋錯，避免 NaN。正確。

### 1.4 習題 16.3 分析
- 錯誤代碼：`X.transpose(0, 2, 1).reshape(1, 2, 2, 2)`。
- $X$ 形狀 `(1, 2, 4)`。
- `X.transpose(0, 2, 1)` -> `(1, 4, 2)`。
    - $Y[0, i, j] = X[0, j, i]$。
    - $Y[0, 0, :] = [1, 5]$。$Y[0, 1, :] = [2, 6]$。$Y[0, 2, :] = [3, 7]$。$Y[0, 3, :] = [4, 8]$。
- `reshape(1, 2, 2, 2)`。Row-major 展平：`1, 5, 2, 6, 3, 7, 4, 8`。
    - `Xh[0, 0, :, :]` -> `[[1, 5], [2, 6]]`。
    - `Xh[0, 1, :, :]` -> `[[3, 7], [4, 8]]`。
- 解答中：`Xh[0,0,:,:] = [[1,5],[2,6]]`。正確。
- 正確拆分 `split_heads`：
    - `Xh[0,0,:,:] = [[1,2],[5,6]]`。
    - `Xh[0,1,:,:] = [[3,4],[7,8]]`。
- 解答正確。

## 2. 一致性與規範檢查

- **字數**：3279 字。符合 3000-4500 區間。
- **執行聲稱**：明確標示「未在本寫作環境執行」、「預期結果」。符合規範。
- **來源**：引用 N1-N6，註明 N4-N6 未核對。符合規範。
- **公式**：LaTeX 格式正確。
- **遮罩約定**：True=允許，程式碼與文字一致。
- **全遮罩處理**：明確拒絕，避免 NaN。符合規範。

## 3. 潛在問題與細微糾正

### 3.1 手算 2 的細節描述
原文在 Head 0 的 $S_0$ 計算中：
$$
S_0 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1\cdot 1 + 2\cdot 2 & 1\cdot 0 + 2\cdot 0 \ 0\cdot 1 + 0\cdot 2 & 0\cdot 0 + 0\cdot 0 \end{pmatrix}
= \begin{pmatrix} 3.5355 & 0 \ 0 & 0 \end{pmatrix}.
$$
- $1\cdot 1 + 2\cdot 2 = 5$。$5/\sqrt{2} \approx 3.5355$。
- 矩陣正確。

### 3.2 程式碼中的 `mask` 擴充
- 如果 mask 是 `(T,)`，`m.ndim == 1`。
    - 程式碼檢查：`if m.ndim == 2: ... elif m.ndim == 3: ... elif m.ndim != 4: raise ValueError`。
    - `m.ndim == 1` 會進入 `m.ndim != 4` 並拋錯。
    - 測試節點說：`mask.shape=(T,)` -> 拋 `ValueError`。
    - 程式碼行為：拋錯。正確。

### 3.3 參數量計算
原文：`多頭用 4(D^2 + D) 個參數（含偏置）`。
- $W_Q: D \times D = D^2$。$b_Q: D$。
- 四個矩陣/偏置：$4(D^2 + D)$。
- 正確。

### 3.4 習題 16.4
- PyTorch `MultiheadAttention` 的 `in_proj_weight` 形狀是 `(3*D, D)`。
- 切分為 Q, K, V 三塊，每塊 `(D, D)`。
- 這與本章 `W_Q, W_K, W_V` 一致。
- `out_proj` 形狀 `(D, D_out)`。
- 提示讀者注意 `(B,H,T,dh)` vs `(B*H,T,dh)` 的轉換。
- PyTorch 內部在計算注意力時，確實會將 batch 和 head 合併為 `B*H`，或者保持 `(B, H, T, dh)` 並使用 einsum/matmul broadcasting。
- 現代 PyTorch 使用 `_scaled_dot_product_attention`，它接受 `(B, H, T, dh)` 格式。
- 輸出 `attn_output` 是 `(B, H, T, dh)`，然後 `transpose` 回 `(B, T, H*dh)`，再 `proj`。
- 所以形狀對應是匹配的。解答正確。

## 4. 總體評價

該章節數學嚴謹，程式碼自足，測試涵蓋正常、邊界與故障模式。手算驗證正確。對常見陷阱（如 reshape/transpose 順序、縮放因子、遮罩語義）有清晰說明。符合 Volume V 的規範。

VERDICT: APPROVE