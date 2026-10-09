# 第16章 多頭注意力與拆分合併 審稿報告

## 1. 核心數學與形狀核對

### 1.1 拆分/合併互逆證明
- **定義**：`split` = `reshape(B,T,D) -> (B,T,H,dh)` 再 `transpose(0,2,1,3)`。`merge` = `transpose(0,2,1,3)` 再 `reshape(B,T,H*dh)`。
- **重算**：`transpose` 是自逆操作（involution），`reshape` 與 `reshape'` 在 row-major 順序下互為逆（前提是元素線性索引對應不變）。
- **結論**：證明邏輯嚴謹，$merge(split(X)) = X$ 成立。

### 1.2 手算例題 2 驗證
- **設定**：$B=1, T=2, D=4, H=2, dh=2$。$X = \begin{bmatrix} 1 & 2 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}$。$W_Q=W_K=W_V=W_O=I$。
- **拆分**：
    - Head 0 ($h=0$): 取 $X$ 的前 $dh=2$ 維。$Q_h^0 = K_h^0 = V_h^0 = \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}$。
    - Head 1 ($h=1$): 取 $X$ 的後 $dh=2$ 維。$Q_h^1 = K_h^1 = V_h^1 = \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix}$。
- **Head 0 計算**：
    - $S_0 = \frac{1}{\sqrt{2}} Q_h^0 (K_h^0)^\top = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 2 & 0 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 5 & 0 \\ 0 & 0 \end{bmatrix} \approx \begin{bmatrix} 3.5355 & 0 \\ 0 & 0 \end{bmatrix}$。
    - Softmax Row 0: $e^{3.5355} \approx 34.31$, Sum $\approx 35.31$. $A_{0,0} \approx 0.9717, A_{0,1} \approx 0.0283$。
    - Softmax Row 1: $[0,0] \to [0.5, 0.5]$。
    - $O_h^0 = A_0 V_h^0$:
        - Row 0: $0.9717 [1,2] + 0.0283 [0,0] = [0.9717, 1.9434]$。
        - Row 1: $0.5 [1,2] + 0.5 [0,0] = [0.5, 1.0]$。
    - 結果與文稿一致。
- **Head 1 計算**：
    - $S_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 0 & 1 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 0 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 1.4142 \end{bmatrix}$。
    - Softmax Row 0: $[0,0] \to [0.5, 0.5]$。
    - Softmax Row 1: $[0, 1.4142]$。Max=1.4142。
        - $exp(0-1.4142) \approx 0.2431$。$exp(0)=1$。
        - $A_{1,0} \approx 0.2431/(1.2431) \approx 0.1956$。
        - $A_{1,1} \approx 1/(1.2431) \approx 0.8044$。
    - $O_h^1 = A_1 V_h^1$. $V_h^1 = \begin{bmatrix} 0 & 0 \\ 1 & 1 \end{bmatrix}$。
        - Row 0: $0.5[0,0] + 0.5[1,1] = [0.5, 0.5]$。
        - Row 1: $0.1956[0,0] + 0.8044[1,1] = [0.8044, 0.8044]$。
    - 結果與文稿一致。

### 1.3 參數數量
- 文稿指出：一般情況含偏置共 `3D^2 + 3D + D·D_out + D_out`。
- 重算：此公式隱含假設 $dv=dh$ (即 $H \cdot dv = D$)。
    - $W_Q, W_K$ 各 $D \times D$ (共 $2D^2$)。
    - $W_V$ 為 $D \times D$ (因 $H \cdot dh = D$)。
    - $W_O$ 為 $D \times D_{out}$。
    - 偏置 $D + D + D + D_{out}$。
    - 總計 $2D^2 + D^2 + D D_{out} + 3D + D_{out} = 3D^2 + 3D + D D_{out} + D_{out}$。
    - 當 $D_{out}=D$ 時，$4D^2+4D$。
- **潛在問題**：本章程式碼明確支援 $dv \neq dh$。在此情況下，參數數量應為 $2D^2 + D(H dv) + (H dv)D_{out} + 2D + H dv + D_{out}$。文稿中的「一般情況」若未限定 $dv=dh$，則與支援 $dv \neq dh$ 的程式碼不一致。
- **判定**：雖然在標準 Transformer 中 $dv=dh$ 是常態，但既然本章明確推廣並測試了 $dv \neq dh$，該註解應更精確或標註「在預設 $dv=dh$ 時」。然而，這不構成核心數學錯誤，僅是描述上的不夠周全。

## 2. 程式碼與測試核對

### 2.1 程式碼邏輯
- `split_heads`: `X.reshape(B, T, H, dh).transpose(0, 2, 1, 3)`。正確。
- `merge_heads`: `Y.transpose(0, 2, 1, 3).reshape(B, T, H * dh)`。正確。
- `forward`:
    - `Qh @ Kh.transpose(0, 1, 3, 2)`。`(B,H,T,dh) @ (B,H,dh,T) -> (B,H,T,T)`。正確。
    - `A @ Vh`。`(B,H,T,T) @ (B,H,T,dv) -> (B,H,T,dv)`。正確。
    - `Y = O @ self.W_O + self.b_O`。`(B,T,H*dv) @ (H*dv, D_out)`。正確。
- **Mask 處理**：
    - `np.where(m, S, -np.inf)`。True=允許，保留 S；False=遮罩，置 -inf。符合約定。
    - 全遮罩檢查：`np.any(np.all(~m, axis=-1))`。如果某一列全為 False，則拋錯。正確。
    - 程式碼在 softmax **之前** 檢查並拋錯，避免 NaN。正確。
    - 廣播檢查：`m.shape[0] not in (1, B)` 等。正確。

### 2.2 測試用例
- **正常路徑**：形狀檢查正確。
- **邊界條件**：
    - `H=1`: 單頭。
    - `H=D`: `dh=1`。
    - `B=1`: 保留軸。
    - `T=1`: Softmax 退化為 1.0。
- **故障測試**：
    - 非整除：`ValueError`。
    - 全遮罩：`ValueError`。
    - 遮罩形狀錯：`ValueError`。
    - 非布林遮罩：`TypeError`。文稿描述已修正為 `TypeError`，與程式碼一致。
- **梯度測試**：
    - 提供了 `numerical_gradient` 和 `run_mha_checks` 函數，用於驗證解析梯度與數值梯度的一致性。
    - 測試涵蓋了無遮罩、因果遮罩、$dv \neq dh$ 的情況。
    - 使用 `np.testing.assert_allclose` 進行驗證。

### 2.3 習題解答
- **16.1**: 解答正確。
- **16.2**: 解答正確，並提供了延伸程式碼驗證。
- **16.3**: 解答正確。
- **16.4**: 解答正確，並提供了 PyTorch 對照程式碼。

## 3. 發現的問題

### 問題 1：參數量公式的適用範圍
- **逐字原句**：「**參數數量**：一般情況含偏置共 `3D^2 + 3D + D·D_out + D_out` 個參數；當 `D_out = D` 時，化為 `4(D^2 + D)`。」
- **原因**：此公式僅在 $dv=dh$ 時成立。由於本章明確支援並測試 $dv \neq dh$，此處的「一般情況」可能誤導讀者以為該公式適用於所有配置。
- **最小修法**：將「一般情況」改為「在預設 `dv = dh` 的一般情況」或補充說明「若 `dv != dh`，參數數量需依 `H*dv` 調整」。

VERDICT: REVISE