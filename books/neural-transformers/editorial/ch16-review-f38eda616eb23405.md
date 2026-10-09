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
- 文稿指出：預設`dv=dh=D/H`，即V總寬為D時，含偏置共`3D^2 + 3D + D·D_out + D_out`個參數。
- 重算：
    - $W_Q, W_K$: $2 \times D \times D = 2D^2$。
    - $W_V$: $D \times (H \cdot dv) = D \times D = D^2$ (因為 $H \cdot dv = D$)。
    - $W_O$: $D \times D_{out}$ (因為合併後寬度為 $H \cdot dv = D$)。
    - 權重總和: $2D^2 + D^2 + D D_{out} = 3D^2 + D D_{out}$。
    - 偏置 $b_Q, b_K, b_V$: $3D$。
    - 偏置 $b_O$: $D_{out}$。
    - 總數: $3D^2 + D D_{out} + 3D + D_{out}$。
    - 當 $D_{out}=D$ 時: $3D^2 + D^2 + 3D + D = 4D^2 + 4D = 4(D^2+D)$。
- 文稿進一步給出 `dv != dh` 的一般公式：`2D^2+2D+(D+1)H·dv+(H·dv+1)D_out`。
    - $W_Q, W_K$: $2D^2$。
    - $b_Q, b_K$: $2D$。
    - $W_V$: $D \times (H \cdot dv) = D H dv$。
    - $b_V$: $H \cdot dv$。
    - $W_O$: $(H \cdot dv) \times D_{out} = H dv D_{out}$。
    - $b_O$: $D_{out}$。
    - 總和: $2D^2 + 2D + D H dv + H dv + H dv D_{out} + D_{out} = 2D^2 + 2D + (D+1)H dv + (H dv + 1)D_{out}$。
    - 公式正確。

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
- **Backward**:
    - 實現了完整的反向傳播，包括 $W_O, b_O, dO, dA, dS, dQ, dK, dV$ 以及 $W_Q, W_K, W_V, b_Q, b_K, b_V$ 和 $dX$。
    - 梯度計算公式符合矩陣微分鏈式法則。
    - 使用 cache 儲存中間變量，確保反向傳播使用正確的前向值。
    - 錯誤處理：檢查輸入和輸出的有限性。
    - **新增檢查**：`self._last_dS = dS.copy()`。這行代碼儲存了 $dS$ 用於測試。
    - **潛在問題**：在 `backward` 中，如果 `forward` 未成功執行（例如因錯誤拋出），`self._cache` 會被設為 `None`。但 `self._last_dS` 可能會保留前一次成功的值。然而，`backward` 首先檢查 `if self._cache is None: raise ValueError`，所以如果 `forward` 失敗，`backward` 會直接拋錯，不會執行到使用 `_last_dS` 的地方。這看起來是安全的，但 `self._last_dS` 作為類別屬性未在 `__init__` 中初始化，首次呼叫 `backward` 前若無 `forward` 會因 `_cache is None` 而失敗。若 `forward` 成功，`_last_dS` 會被賦值。測試程式碼中 `att._last_dS` 在 `backward` 後被訪問。這是合法的，因為 `backward` 成功執行意味著 `forward` 也成功過（因為 `backward` 檢查了 `_cache`）。

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
    - **新增斷言**：
        - `assert np.all(weights[~allowed4] == 0.0)`：檢查遮罩位置的注意力權重是否為 0。正確。
        - `assert np.all(att._last_dS[~allowed4] == 0.0)`：檢查遮罩位置的 $dS$ 是否為 0。
            - 重算：$dS = A \odot (dA - \sum_k A \odot dA)$。若 $A=0$（遮罩位置），則 $dS=0$。正確。
        - `np.testing.assert_allclose(att._last_dS.sum(axis=-1),0.0,atol=1e-14)`：檢查 $dS$ 沿 key 軸之和是否為 0。
            - 重算：$\sum_k dS_{qk} = \sum_k A_{qk} dA_{qk} - \sum_k A_{qk} (\sum_k' A_{qk'} dA_{qk'}) = (\sum_k A_{qk}) (\sum_k dA_{qk}) - (\sum_k A_{qk}) (\sum_k dA_{qk}) = 0$（因為 $\sum_k A_{qk} = 1$）。正確。

### 2.3 習題解答
- **16.1**: 解答正確。
- **16.2**: 解答正確，並提供了延伸程式碼驗證。
- **16.3**: 解答正確。
- **16.4**: 解答正確，並提供了 PyTorch 對照程式碼。

## 3. 發現的問題

### 問題 1：無重大問題
- 文稿在參數數量部分已明確區分預設 `dv=dh` 與一般 `dv!=dh` 的情況，公式均正確。
- 程式碼、測試、手算、證明均核對無誤。
- 符合 Volume V 規範（無虛構執行、明確預期結果、自足程式等）。

## 4. 最終檢查
- 字數 3931，符合 3000-4500。
- 無虛構執行紀錄。
- 數學證明與手算正確。
- 程式碼邏輯正確，包含完整的反向傳播。
- 描述與程式碼行為一致。

VERDICT: APPROVE