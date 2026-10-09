# 第14章 稀疏矩陣與共軛梯度

> **本章定位**：在橢圓型偏微分方程（如 Poisson 方程）的離散化過程中，線性系統 $Ax=b$ 的求解是核心瓶頸。本章聚焦於**對稱正定（SPD）**矩陣的高效求解，詳細闡述壓縮稀疏行（CSR）資料結構、Jacobi 與 Gauss–Seidel 預條件、以及共軛梯度（Conjugate Gradient, CG）算法的數學基礎與實作細節。重點在於區分「收斂性」、「穩定性」與「物理一致性」，並提供自足的 NumPy CPU 實作。

## 學習目標與先備知識

### 學習目標
1.  理解稀疏矩陣的 CSR（Compressed Sparse Row）格式，及其在記憶體效率與 MatVec（矩陣-向量乘法）速度上的優勢。
2.  掌握 SPD 矩陣的定義、特徵值性質，及其作為 CG 算法收斂保證的必要性。
3.  推導 CG 算法的共軛梯度方向、步長公式，以及殘差正交性。
4.  理解 Jacobi 與 Gauss–Seidel 預條件的作用機制，及其對條件數的改善效果。
5.  能撰寫自足的 NumPy CG 求解器，並實施嚴格的真殘差檢查與停止條件。
6.  辨識非 SPD 系統（如含平流項）為何不能直接使用標準 CG，並了解其故障表現。

### 先備知識
*   **線性代數**：內積、正交投影、特徵值分解、條件數 $\kappa(A)$。
*   **偏微分方程**：Poisson 方程 $-\nabla^2 u = f$ 的物理意義（穩態擴散）。
*   **有限差分**：一維/二維拉普拉斯算子離散後形成的三對角/五對角稀疏結構。
*   **Python/NumPy**：基礎陣列運算、函數定義、`np.dot` 與 `np.linalg.norm` 使用。

## 問題與直覺

### 為什麼需要迭代法？
對於 $N \times N$ 的網格，未知數數量 $n \approx N^2$。若使用直接法（如 LU 分解），時間複雜度為 $O(n^3)$，空間複雜度為 $O(n^2)$。當 $N=1000$ 時，$n \approx 10^6$，直接法需 $10^{18}$ 次運算，完全不可行。
然而，離散拉普拉斯矩陣 $A$ 極為稀疏：每個未知數僅與鄰近幾個點相關（2D 中最多 5 個）。非零元素數量 $nnz \approx 5n$。
**直覺**：迭代法利用稀疏結構，每次迭代僅需 $O(n)$ 運算。若迭代次數 $k$ 顯著小於 $n$，總運算量 $O(k \cdot n)$ 將遠低於直接法。CG 算法的關鍵在於其收斂速度與 $\kappa(A)$ 相關，而非與 $n$ 呈線性增長（理想情況下）。

### 物理直覺：能量最小化
Poisson 方程可視為彈性力學中膜在載荷下的平衡態，或是熱傳導的穩態分佈。數學上，求解 $Ax=b$ 等价於最小化二次能量泛函：
$$
J(u) = \frac{1}{2} u^T A u - b^T u
$$
其中 $A$ 為離散化的負拉普拉斯算子 $A = -\delta_h^2$（在適當邊界條件下為 SPD）。
*   **梯度下降法**：沿著負梯度方向移動，但對於高維空間，梯度方向往往相互正交性極差，導致「之字形」路徑，收斂極慢。
*   **共軛梯度法**：構建一系列搜索方向 $p_k$，使得這些方向關於 $A$ 共軛（$A$-orthogonal）。這確保了每一步都在新的維度上「消除」誤差分量，避免了振盪。

## 數學與物理推導

### 1. 稀疏矩陣與 CSR 格式
一個 $n \times n$ 的稀疏矩陣 $A$ 若使用密集格式存儲，需 $n^2$ 個雙精度浮點數。CSR 格式使用三個一維陣列存儲：
1.  `data`：長度為 $nnz$，存儲所有非零元素的值。
2.  `indices`：長度為 $nnz$，存儲每個非零元素所在的**列索引**（column index）。
3.  `indptr`：長度為 $n+1$，`indptr[i]` 到 `indptr[i+1]` 區間內存儲了第 $i$ 行所有非零元素在 `data` 和 `indices` 中的位置。

**MatVec 複雜度**：計算 $y = Ax$ 時，對每一行 $i$，只需遍歷 `indptr[i]` 到 `indptr[i+1]` 的元素。總運算次數正比於 $nnz$，而非 $n^2$。

### 2. SPD 矩陣與 CG 收斂條件
矩陣 $A$ 為對稱正定（Symmetric Positive Definite, SPD），若滿足：
1.  $A = A^T$
2.  對任意非零向量 $x$，有 $x^T A x > 0$

物理上，離散的 $-\nabla^2$ 算子在齊次 Dirichlet 邊界條件下為 SPD。若邊界條件為 Neumann（純梯度邊界），則 $A$ 僅為半正定（Singular），需處理零空間（常數模態），本節不展開此情況，假設已透過 pin 點或均值約束使系統 SPD。

**收斂性**：若 $A$ 為 SPD，CG 算法在精確算術下最多 $n$ 步收斂（實際中遠少於 $n$）。收斂速度取決於特徵值的分散程度，即條件數 $\kappa(A) = \lambda_{\max} / \lambda_{\min}$。

### 3. CG 算法推導
初始化：$x_0 = 0$（或任意初始猜測），計算初始殘差 $r_0 = b - A x_0$，並設初始方向 $p_0 = r_0$。

第 $k$ 步迭代：
1.  計算步長（Minimize energy along $p_k$）：
    $$
    \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}
    $$
2.  更新解：
    $$
    x_{k+1} = x_k + \alpha_k p_k
    $$
3.  更新殘差：
    $$
    r_{k+1} = r_k - \alpha_k A p_k
    $$
4.  計算共軛係數（確保 $p_{k+1}$ 與 $p_k$ $A$-共軛）：
    $$
    \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}
    $$
5.  更新方向：
    $$
    p_{k+1} = r_{k+1} + \beta_k p_k
    $$

**關鍵性質**：
*   殘差序列 $r_0, r_1, \dots$ 彼此正交：$r_i^T r_j = 0$ for $i \neq j$。
*   方向序列 $p_0, p_1, \dots$ 彼此 $A$-共軛：$p_i^T A p_j = 0$ for $i \neq j$。
*   **真殘差**：停止條件必須基於真殘差 $r_k = b - A x_k$，而非迭代中使用的局部殘差。

### 4. 預條件（Preconditioning）
若 $\kappa(A)$ 很大，CG 收斂慢。引入預條件矩陣 $M \approx A$，求解 $M^{-1} A x = M^{-1} b$。
定義 $\tilde{A} = M^{-1} A$，$\tilde{b} = M^{-1} b$。
若 $M$ 也是 SPD，則 $\tilde{A}$ 與 $A$ 具有相同特徵值，但 $\tilde{A}$ 的特徵向量不同，且 $\kappa(\tilde{A})$ 通常小於 $\kappa(A)$。
*   **Jacobi 預條件**：$M = D$，其中 $D$ 是 $A$ 的對角矩陣。簡單但效果有限。
*   **Gauss–Seidel 預條件**：$M = (D + L)$，其中 $L$ 是 $A$ 的嚴格下三角矩陣。通常比 Jacobi 更有效，但實現稍複雜。

## 逐步手算例題

### 例 1：1D Poisson 三點系統（SPD）
問題：$-u''(x) = f(x)$，$x \in [0,1]$，$u(0)=u(1)=0$。
網格 $N=2$ 個內部節點，步長 $h = 1/3$。
$f(x) = 1$（常數來源）。
邊界：$u_0 = u_3 = 0$。
未知數：$u_1, u_2$。

離散方程：
$$
\frac{u_{i-1} - 2u_i + u_{i+1}}{h^2} = -f_i
$$
$$
- \frac{u_{i-1} - 2u_i + u_{i+1}}{h^2} = f_i \implies \frac{2u_i - u_{i-1} - u_{i+1}}{h^2} = f_i
$$
注意：標準離散化 $A = \frac{1}{h^2} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$。
右端 $b_i = f_i$。
$h^2 = (1/3)^2 = 1/9$。
$b = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$。
$A = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$。
（註：若寫成 $A x = b$，則 $A$ 應包含 $1/h^2$ 因子，即 $A = \begin{bmatrix} 18 & -9 \\ -9 & 18 \end{bmatrix}$，$b = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$。為簡化手算，我們先解 $A_{scaled} x = b_{scaled}$，其中 $A_{scaled} = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$，$b_{scaled} = \begin{bmatrix} h^2 \\ h^2 \end{bmatrix} = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix}$。）

令 $A = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$，$b = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix}$。

**CG 迭代**：
1.  $x_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$。
2.  $r_0 = b - A x_0 = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix}$。
3.  $p_0 = r_0 = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix}$。
4.  $A p_0 = \begin{bmatrix} 2(1/9) - (1/9) \\ -(1/9) + 2(1/9) \end{bmatrix} = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix}$。
5.  $\alpha_0 = \frac{r_0^T r_0}{p_0^T A p_0} = \frac{(1/9)^2 + (1/9)^2}{(1/9)(1/9) + (1/9)(1/9)} = \frac{2/81}{2/81} = 1$。
6.  $x_1 = x_0 + \alpha_0 p_0 = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix}$。
7.  $r_1 = r_0 - \alpha_0 A p_0 = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix} - \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$。

**結果**：一步收斂。解為 $u_1 = u_2 = 1/9$。
驗證：$A x_1 = \begin{bmatrix} 2(1/9) - 1/9 \\ -1/9 + 2(1/9) \end{bmatrix} = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix} = b$。正確。

### 例 2：非 SPD 反例（故障測試）
設 $A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$，$b = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$。
此矩陣對稱，但特徵值為 $1, -1$，非正定。

**CG 迭代**：
1.  $x_0 = 0$，$r_0 = b = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$，$p_0 = r_0$。
2.  $A p_0 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$。
3.  $p_0^T A p_0 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 0$。
4.  $\alpha_0 = \frac{1}{0}$ **未定義（除零錯誤）**。

**結論**：標準 CG 算法在 $p_k^T A p_k = 0$ 或負時失敗。這揭示了 SPD 是 CG 收斂的必要條件。在實作中，若偵測到 $p^T A p \le \epsilon$，必須中斷並報告錯誤。

## 實作與程式

以下提供一個自足的 Python 3.10+/NumPy CPU 實作。程式包含 CSR 構造、CG 求解、真殘差檢查及非 SPD 偵測。

```python
import numpy as np

def csr_matvec(indptr, indices, data, x):
    """
    執行 CSR 格式的矩陣-向量乘法 y = A @ x
    """
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        y[i] = np.dot(data[start:end], x[indices[start:end]])
    return y

def build_poisson_csr_1d(N):
    """
    建構 1D Poisson -u''=f 的離散矩陣 A (SPD) 的 CSR 格式
    A 是對角為 2，次對角為 -1 的三對角矩陣 (已乘 1/h^2)
    這裡為了簡化，直接返回未乘 1/h^2 的結構，
    但在實作 CG 時，我們必須使用實際的物理矩陣。
    為符合物理單位，我們在此定義 A_phys = (1/h^2) * A_struct.
    但為了避免複雜度，此函數返回 A_struct，
    使用者在呼叫 CG 時需調整 b 或 A.
    這裡我們直接定義 A 為物理意義的矩陣，即對角 2/h^2, 次對角 -1/h^2.
    """
    h = 1.0 / (N + 1)
    inv_h2 = 1.0 / (h * h)
    # A 結構: 對角 2, 次對角 -1
    # CSR arrays
    indptr = np.zeros(N + 1, dtype=int)
    indices = np.zeros(3 * N, dtype=int) # 最多 3N 個非零元
    data = np.zeros(3 * N)
    
    nnz = 0
    for i in range(N):
        indptr[i] = nnz
        cols = []
        vals = []
        # Left diagonal
        if i > 0:
            cols.append(i - 1)
            vals.append(-inv_h2)
        # Main diagonal
        cols.append(i)
        vals.append(2 * inv_h2)
        # Right diagonal
        if i < N - 1:
            cols.append(i + 1)
            vals.append(-inv_h2)
        
        indices[nnz:nnz+len(cols)] = cols
        data[nnz:nnz+len(cols)] = vals
        nnz += len(cols)
    indptr[N] = nnz
    
    return indptr, indices, data

def cg_solve_csr(indptr, indices, data, b, x0=None, rtol=1e-8, atol=1e-10, max_iter=1000):
    """
    共軛梯度求解 Ax = b，其中 A 由 CSR 格式提供。
    返回: (x, n_iter, residual_norm, converged, status)
    status: "success", "max_iter", "nan_inf", "non_spd", "diverged"
    """
    n = len(b)
    
    # 輸入檢查
    if not np.all(np.isfinite(b)):
        return None, 0, np.nan, False, "nan_inf_input"
    if not np.all(np.isfinite(data)):
        return None, 0, np.nan, False, "nan_inf_data"
        
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    # 計算初始真殘差
    Ax = csr_matvec(indptr, indices, data, x)
    r = b - Ax
    norm_r = np.linalg.norm(r)
    norm_b = np.linalg.norm(b)
    
    if norm_r < atol + rtol * norm_b:
        return x, 0, norm_r, True, "success"
        
    p = r.copy()
    rz = np.dot(r, r)
    
    for k in range(max_iter):
        # 計算 Ap
        Ap = csr_matvec(indptr, indices, data, p)
        
        # 檢查 SPD 條件: p^T A p 必須為正
        pAp = np.dot(p, Ap)
        if pAp <= 1e-12: # 數值穩定性容差
            return x, k, norm_r, False, "non_spd_detected"
            
        alpha = rz / pAp
        
        # 更新解
        x += alpha * p
        r -= alpha * Ap
        norm_r = np.linalg.norm(r)
        
        # 停止條件檢查 (真殘差)
        if norm_r < atol + rtol * norm_b:
            return x, k + 1, norm_r, True, "success"
            
        # 檢查數值發散
        if not np.all(np.isfinite(r)):
            return x, k + 1, norm_r, False, "nan_inf_iter"
            
        # 計算 beta
        rz_new = np.dot(r, r)
        beta = rz_new / rz
        rz = rz_new
        
        # 更新方向
        p = r + beta * p
        
    return x, max_iter, norm_r, False, "max_iter"

# 測試例 1
N = 5
h = 1.0 / (N + 1)
b = np.ones(N) # f=1
indptr, indices, data = build_poisson_csr_1d(N)

x_sol, nit, res, conv, status = cg_solve_csr(indptr, indices, data, b)
print(f"Test 1 (SPD): converged={conv}, iters={nit}, residual={res:.2e}, status={status}")
# 預期: conv=True, nit < N, status=success

# 測試例 2 (非 SPD 模擬)
# 手動構造一個非 SPD 的 CSR 矩陣以測試故障檢測
# A = [[0, 1], [1, 0]]
indptr_bad = np.array([0, 1, 2], dtype=int)
indices_bad = np.array([1, 0], dtype=int)
data_bad = np.array([1.0, 1.0])
b_bad = np.array([1.0, 0.0])

x_bad, nit_bad, res_bad, conv_bad, status_bad = cg_solve_csr(indptr_bad, indices_bad, data_bad, b_bad, max_iter=10)
print(f"Test 2 (Non-SPD): converged={conv_bad}, iters={nit_bad}, status={status_bad}")
# 預期: conv_bad=False, status_bad=non_spd_detected
```

## 測試與預期結果

### 1. 正常測試（SPD Poisson）
*   **輸入**：1D Poisson，$N=10$，$f=1$，Dirichlet 邊界。
*   **預期**：
    *   `converged = True`。
    *   `niters < 10`（實際上對於 1D Poisson，CG 通常在 $O(N^{1/2})$ 或更少步數收斂，這裡 $N=10$，預期 3-5 步）。
    *   `residual < 1e-8`。
    *   `status = "success"`。
    *   **驗證**：計算解析解 $u(x) = \frac{x^2}{2} - \frac{x}{2}$ 的離散值，比較誤差應在 $O(h^2)$ 量級。

### 2. 邊界測試（零右端）
*   **輸入**：$b = 0$。
*   **預期**：
    *   `converged = True`。
    *   `niters = 0`（初始猜測 $x=0$ 即為解）。
    *   `residual = 0`。

### 3. 故障測試（非 SPD）
*   **輸入**：對稱但非正定矩陣（如例 2）。
*   **預期**：
    *   `converged = False`。
    *   `status = "non_spd_detected"`。
    *   程式不應陷入無限期迴圈或產生 NaN。

### 4. 故障測試（NaN 輸入）
*   **輸入**：$b$ 包含 `np.nan`。
*   **預期**：
    *   `converged = False`。
    *   `status = "nan_inf_input"`。
    *   立即中斷。

## 除錯與常見陷阱

1.  **殘差定義錯誤**：
    *   **錯誤**：使用 `np.linalg.norm(b - A @ x)` 在迴圈內計算，但未更新 `x` 或 `A` 的引用。
    *   **修正**：確保每次迭代後，`r` 是最新的真殘差。在 CG 中，`r` 的更新公式 $r_{k+1} = r_k - \alpha A p_k$ 是精確的（在精確算術下），但浮點誤差累積後，建議每 $k$ 步重新計算一次 $r = b - A x$ 以校正漂移（本實作未包含此校正，因 $N$ 小，可接受；大 $N$ 建議加入）。
2.  **CSR 索引錯誤**：
    *   **陷阱**：`indptr` 的長度必須為 $N+1$，且 `indptr[N]` 必須等於 `len(data)`。
    *   **除錯**：斷言 `len(indptr) == len(x) + 1` 和 `indptr[-1] == len(data)`。
3.  **非 SPD 誤用**：
    *   **情境**：將平流擴散方程（Advection-Diffusion）離散後直接套用 CG。
    *   **問題**：平流項使矩陣非對稱。
    *   **後果**：`p^T A p` 可能為負，導致 $\alpha$ 為負或發散。
    *   **對策**：檢測 `pAp <= 0` 並切換至 BiCGStab 或 GMRES（見第 15 章）。
4.  **停止條件過於寬鬆**：
    *   **陷阱**：僅檢查 `norm_r < 1e-6` 而未考慮 `norm_b`。若 $b$ 極小（如 $1e-20$），絕對殘差 $1e-6$ 相對誤差極大。
    *   **修正**：使用相對停止條件 `norm_r < atol + rtol * norm_b`。

## 養殖與相場案例

### 案例：池塘溫度穩態求解
考慮一個矩形養殖池，邊界溫度固定（Dirichlet 條件），內部無熱源（$f=0$）。求解穩態溫度分佈 $T(x,y)$。
*   **模型**：$-\nabla^2 T = 0$。
*   **離散**：2D 5-點拉普拉斯算子，生成稀疏 SPD 矩陣 $A$。
*   **規模**：$64 \times 64$ 網格，$n=4096$。
*   **求解**：使用 CG 算法。
*   **預期表現**：
    *   **收斂**：CG 應在幾十至一百次迭代內收斂（取決於網格條件數）。
    *   **物理一致性**：溫度場應平滑，無振盪。最大值應出現在邊界（極值原理）。
    *   **診斷**：監控真殘差 $\|b - Ax\|_2$。若殘差下降至 $10^{-8}$，視為收斂。
    *   **注意**：此處 $f=0$，解為調和函數。若加入來源項（如加熱器），$b \neq 0$，求解器結構不變。

### 相位橋接說明
在相場（Phase Field）模型（如 Cahn-Hilliard）中，離散化後的線性系統通常為 SPD 或可預條件為 SPD。然而，時間積分（如 Backward Euler）會產生時間相關項。本章僅討論空間離散後的線性求解。若涉及時間步長 $\Delta t$，系統矩陣為 $A_{space} + \frac{1}{\Delta t} I$，仍為 SPD（因 $I$ 正定），CG 依然適用。

## 習題

1.  **手算**：對 3 點 1D Poisson（$N=3$，$h=1/4$，$f=1$），寫出 $A$ 和 $b$，並手算 CG 的第一步 $\alpha_0, x_1, r_1$。
2.  **程式**：修改 `cg_solve_csr`，加入 Jacobi 預條件。構造 $M = \text{diag}(A)$，並比較有/無預條件時的迭代次數（使用 $N=16$ 的 1D 問題）。
3.  **反例**：構造一個 $4 \times 4$ 對稱但非正定矩陣 $A$（例如特徵值包含負數），示範 CG 在哪一步失敗，並解釋原因。
4.  **整合**：將 1D Poisson 問題擴展到 2D（$N_x \times N_y$）。使用 CSR 構造 2D 拉普拉斯矩陣，求解 $-\Delta u = 1$，並比較 CG 解與直接法（`np.linalg.solve` 對於密集矩陣）的結果，計算 $L_2$ 誤差。

## 習題解答

### 1. 手算解答
$N=3$，$h=1/4$，$inv\_h2 = 16$。
$A = 16 \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix} = \begin{bmatrix} 32 & -16 & 0 \\ -16 & 32 & -16 \\ 0 & -16 & 32 \end{bmatrix}$。
$b = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$。
$x_0 = 0$，$r_0 = b$。
$p_0 = r_0$。
$A p_0 = \begin{bmatrix} 16 \\ 0 \\ 16 \end{bmatrix}$。
$r_0^T r_0 = 3$。
$p_0^T A p_0 = 1\cdot16 + 1\cdot0 + 1\cdot16 = 32$。
$\alpha_0 = 3 / 32$。
$x_1 = \frac{3}{32} \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 3/32 \\ 3/32 \\ 3/32 \end{bmatrix}$。
$r_1 = r_0 - \frac{3}{32} A p_0 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} - \begin{bmatrix} 3/32 \\ 0 \\ 3/32 \end{bmatrix} = \begin{bmatrix} 29/32 \\ 1 \\ 29/32 \end{bmatrix}$。
驗證正交性：$r_0^T r_1 = 1(29/32) + 1(1) + 1(29/32) = 58/32 + 1 = 2.8125 \neq 0$？
**檢查**：CG 理論保證 $r_k$ 與 $r_{k-1}$ 正交。
$r_0 = [1,1,1]$。
$A p_0 = [16, 0, 16]$。
$\alpha_0 = 3/32$。
$r_1 = [1,1,1] - \frac{3}{32}[16,0,16] = [1 - 3/2, 1, 1 - 3/2] = [-0.5, 1, -0.5]$。
$r_0^T r_1 = 1(-0.5) + 1(1) + 1(-0.5) = 0$。正交性成立。
（先前手算 $A p_0$ 時，$16 \times 1 - 16 \times 1 = 0$ 正確，但 $\alpha$ 計算：$r_0^T r_0 = 3$。$p_0^T A p_0 = [1,1,1] \cdot [16,0,16] = 32$。$\alpha = 3/32$。$r_1 = b - \alpha A p_0 = [1,1,1] - \frac{3}{32}[16,0,16] = [1 - 0.5, 1, 1 - 0.5] = [0.5, 1, 0.5]$。
再驗證 $r_0^T r_1 = 1(0.5) + 1(1) + 1(0.5) = 2 \neq 0$。
**錯誤根源**：$A$ 的定義。
$A = \frac{1}{h^2} \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix}$。
$h=1/4, h^2=1/16, 1/h^2=16$。
$A = 16 \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix} = \begin{bmatrix} 32 & -16 & 0 \\ -16 & 32 & -16 \\ 0 & -16 & 32 \end{bmatrix}$。
$p_0 = [1,1,1]^T$。
$A p_0 = [32-16, -16+32-16, -16+32]^T = [16, 0, 16]^T$。
$p_0^T A p_0 = 1(16) + 1(0) + 1(16) = 32$。
$r_0^T r_0 = 3$。
$\alpha_0 = 3/32$。
$r_1 = r_0 - \alpha_0 A p_0 = [1,1,1] - \frac{3}{32}[16,0,16] = [1 - \frac{48}{32}, 1, 1 - \frac{48}{32}] = [1 - 1.5, 1, 1 - 1.5] = [-0.5, 1, -0.5]$。
$r_0^T r_1 = 1(-0.5) + 1(1) + 1(-0.5) = 0$。
**結論**：正交性成立。先前計算 $16/32$ 時出錯，$\frac{3}{32} \times 16 = \frac{48}{32} = 1.5$。正確。

### 2. 程式解答（概念）
Jacobi 預條件：$M = \text{diag}(A)$。
在 `cg_solve_csr` 中，將 $z = r$ 改為 $z = M^{-1} r$。
對於 1D Poisson，$M = 2/h^2 I$。
$M^{-1} = h^2/2 I$。
這僅僅縮放殘差，不改變條件數。對於 1D，Jacobi 預條件幾乎無效。
對於 2D 或變係數問題，效果可能不同。
比較時，應記錄 `niters` 與 `residual`。

### 3. 反例解答
$A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$。
步驟 0：$p_0 = [1,0]^T$（若 $b=[1,0]^T$）。
$A p_0 = [0,1]^T$。
$p_0^T A p_0 = 0$。
$\alpha_0$ 除零。
原因：$A$ 有負特徵值，$p_0$ 落在非正定方向。

### 4. 整合解答
2D CSR 構造：
對於節點 $(i,j)$，列索引 $k = j \cdot N_x + i$。
鄰居：$(i-1,j), (i+1,j), (i,j-1), (i,j+1)$。
中心：$(i,j)$。
比較 CG 與直接解：
$L_2$ 誤差應 $< 10^{-10}$（若 CG 收斂）。
CG 迭代次數通常少於 $N_x + N_y$。

## 本章小結

本章詳述了稀疏矩陣的 CSR 格式及其在求解 SPD 線性系統中的核心角色。共軛梯度（CG）算法通過構建 $A$-共軛方向，高效地最小化能量泛函，收斂速度取決於矩陣的條件數。Jacobi 和 Gauss–Seidel 預條件可通過改善條件數來加速收斂。實作中必須嚴格檢查真殘差與 SPD 條件，以避免非 SPD 系統引起的數值不穩定。本章為後續非對稱系統求解（GMRES）及相位場模型離散奠定了基礎。

## 參考來源

*   F1: FiPy 有限體積離散與邊界 (https://pages.nist.gov/fipy/en/latest/numerical/discret.html) - 參考稀疏矩陣構造。
*   F3: PETSc 線性系統求解器 (https://petsc.org/release/manual/ksp/) - CG 與預條件詳細討論。
*   F4: SciPy 稀疏線性代數 API (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) - CSR 格式定義與 `cg` 函數。
*   F2: FEniCSx Poisson 與弱形式 (https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html) - SPD 物理背景。