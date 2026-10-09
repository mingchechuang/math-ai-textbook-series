# 審稿結果

主要公式與反例已修正，但程式仍把每步 residual replacement 與標準 PCG recurrence 混合；另有篇幅及量綱契約未達標。

## 必須修正

### 1. 每步 residual replacement 後不再是所推導的標準 PCG recurrence

- **原句／程式**：
  ```python
  r -= alpha * Ap
  ...
  r_true = b - Ax_new
  ...
  r = r_true
  ...
  beta = rz_new / rz
  p = z_new + beta * p
  ```
  並稱「Residual Replacement……以保持穩定性」。
- **原因**：標準 PCG 的有限步共軛性推導使用遞推殘差。浮點下將 `r` 每一步強制換成重新計算的真殘差，卻仍沿用標準 $\beta_k$，會擾動殘差正交性與方向共軛性；不能直接宣稱這就是所推導的標準 PCG或必然更穩定。真殘差應用於停止診斷，但不必每步取代 recurrence 狀態。
- **最小修法**：保留遞推 `r` 計算 `z_new`、`rz_new` 與 `beta`；每輪另外計算 `r_true` 只作停止檢查。若真殘差已達容差才成功；若要定期 replacement，應明列週期並在 replacement 時重啟方向，例如令 `r=r_true`、`z=M^{-1}r`、`p=z`，而不是延續舊方向。

### 2. CG／PCG 正交與共軛性缺少「精確算術」限定

- **原句**：
  > 殘差序列彼此正交；方向序列彼此 $A$-共軛。
- **原因**：浮點算術中這些性質會逐漸喪失；目前程式又每步替換殘差，更不能直接保證所列性質。
- **最小修法**：改成「在精確算術與標準 recurrence 下」。另說浮點實作僅近似保持，不能把連續／精確算術性質當成程式保證。

### 3. Gauss–Seidel 名稱仍不正確

- **原句**：
  > 標準 Gauss–Seidel 預條件矩陣為 $M_{GS}=(D+L)D^{-1}(D+U)$
- **原因**：這是 symmetric Gauss–Seidel（SGS）形式；標準單向 Gauss–Seidel 是 $D+L$。後一句其實也如此區分，故標題與公式互相矛盾。
- **最小修法**：將前句改為「symmetric Gauss–Seidel 預條件」，記為 $M_{\rm SGS}$，並保留 $U=L^T$、$D$ 正定的條件。

### 4. 量綱與網格位置契約仍未補齊

- **定位**：1D Poisson 手算及「池塘溫度穩態求解」。
- **原因**：
  - 1D 例未說明 $x\in[0,1]$ 是無因次座標或公尺，故 $f=1$、$A$ 與 $b$ 的單位不明。
  - 池塘例仍未列池域長寬、$x,y$ 的 m、$T$ 使用 K 或攝氏溫差。
  - 2D 例未明示 node-centered 內部未知數及邊界節點消去，容易與全卷預設 cell-centered `q[j,i]` 混用。
- **最小修法**：將手算例明訂為無因次問題；池塘例明列 SI 單位及 node-centered 配置，說明內部 shape、展平 $k=jN_x+i$、Dirichlet 邊界不屬未知向量且其貢獻移至 $b_{\rm boundary}$。

### 5. `L_2` 誤差仍未定義

- **原句**：
  > 計算 $L_2$ 誤差。
- **原因**：習題未說是 PCG 與直接離散解之未加權向量範數，還是對解析 PDE 解的網格加權 $L_2$ 誤差。兩者目的與尺度不同。
- **最小修法**：分別定義，例如
  $$
  E_{\rm solve}=\|x_{\rm PCG}-x_{\rm direct}\|_2,
  $$
  以及均勻 2D 節點網格的
  $$
  E_{L_2,h}=\sqrt{h_xh_y\sum_{i,j}|u_{j,i}-u_{\rm exact}(x_i,y_j)|^2}.
  $$
  沒有解析解時不得把前者稱為 PDE 離散誤差。

### 6. 篇幅仍低於明示最低要求

- **定位**：`measured_characters = 2790`，最低要求為 3000 中文字。
- **原因**：這是交付規格，不是風格偏好。
- **最小修法**：以量綱、node-centered shape、標準 PCG 與真殘差診斷的區分，以及完整有／無 Jacobi 預期測試補足內容。

## 建議一併修正

### 7. Jacobi 對此特例的效果應寫成「不變」，不是「可能略減」

- **原句**：
  > $\kappa(\tilde A)\approx\kappa(A)$，改善有限。迭代次數可能略減或相同。
- **原因**：本例 $D=(2/h^2)I$，故 $D^{-1}A$ 只是 $A$ 的正純量倍數，條件數在 2-範數下恰好相同；精確算術中的 Krylov 收斂不會因其改善。
- **最小修法**：改為「條件數恰好不變，精確算術迭代次數相同；浮點下若有差異，只能視為捨入與停止判定差異」。

### 8. NumPy 整數 dtype 檢查過度嚴格

- **原句／程式**：
  ```python
  indptr.dtype != int
  indices.dtype != int
  ```
- **原因**：合法的 `np.int32` 等整數索引可能被拒絕。
- **最小修法**：
  ```python
  np.issubdtype(indptr.dtype, np.integer)
  ```
  `indices` 同理；並檢查 `b` 為一維、`rtol >= 0`、`atol >= 0`、`max_iter >= 0`。

### 9. Breakdown 閾值仍應說明是啟發式判定

- **原句**：固定 `eps_curv = 1e-12`。
- **原因**：即使採 $\|p\|\|Ap\|$ 縮放，固定 $10^{-12}$ 仍可能對極病態 SPD 系統造成近零曲率誤判；它不是 SPD 的完整檢驗。
- **最小修法**：將其設為函式參數或基於機器精度，並在文字中稱為「數值 breakdown 啟發式」，不得稱為完整 SPD 檢查。

VERDICT: REVISE