# 第14章 稀疏矩陣與共軛梯度

> **本章定位**：在橢圓型偏微分方程（如 Poisson 方程）的離散化過程中，線性系統 $Ax=b$ 的求解是核心瓶頸。本章聚焦於**對稱正定（SPD）**矩陣的高效求解，詳細闡述壓縮稀疏行（CSR）資料結構、Jacobi 預條件、以及共軛梯度（Conjugate Gradient, CG）與預條件共軛梯度（PCG）算法的數學基礎與實作細節。重點在於區分「收斂性」、「穩定性」與「物理一致性」，並提供自足的 NumPy CPU 實作與直接解比較。

## 學習目標與先備知識

### 學習目標
1.  理解稀疏矩陣的 CSR（Compressed Sparse Row）格式，及其在記憶體效率與 MatVec（矩陣-向量乘法）速度上的優勢。
2.  掌握 SPD 矩陣的定義、特徵值性質，及其作為 CG 算法收斂保證的必要性。
3.  推導 CG 與 PCG 算法的共軛梯度方向、步長公式，以及殘差正交性。
4.  理解 Jacobi 預條件的作用機制，及其對條件數的改善效果。
5.  能撰寫自足的 NumPy PCG 求解器，並實施嚴格的真殘差檢查、停止條件與直接解比較。
6.  辨識非 SPD 系統（如含平流項）為何不能直接使用標準 CG，並了解其故障表現（breakdown）。
7.  掌握 $A$-範數誤差界與真殘差之間的關係，理解病態系統中殘差小不等於誤差小。

### 先備知識
*   **線性代數**：內積、正交投影、特徵值分解、條件數 $\kappa(A)$、相似變換。
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

**CSR 契約**：
*   `indptr[0]` 必須為 0。
*   `indptr` 必須嚴格遞增（除非有全零行）。
*   `indptr[n]` 必須等於 `len(data)` 與 `len(indices)`。
*   對於每個 $i \in [0, n)$，`indices` 中的值必須在 $[0, n)$ 範圍內。

**MatVec 複雜度**：計算 $y = Ax$ 時，對每一行 $i$，只需遍歷 `indptr[i]` 到 `indptr[i+1]` 的元素。總運算次數正比於 $nnz$，而非 $n^2$。

### 2. SPD 矩陣與 CG 收斂條件
矩陣 $A$ 為對稱正定（Symmetric Positive Definite, SPD），若滿足：
1.  $A = A^T$
2.  對任意非零向量 $x$，有 $x^T A x > 0$

物理上，離散的 $-\nabla^2$ 算子在齊次 Dirichlet 邊界條件下為 SPD。若邊界條件為 Neumann（純梯度邊界），則 $A$ 僅為半正定（Singular），需處理零空間（常數模態）。
**注意**：若採用純 Neumann 邊界，必須先檢查右端 $b$ 的相容性（即 $b$ 必須與零空間正交，$\mathbf{1}^T b = 0$）。處理方式通常包括：
1.  **Pin 點法**：將某一點的方程替換為 $u_i=0$。這會改變矩陣結構，使其不再嚴格 SPD（或破壞對稱性，取決於實施方式），需謹慎處理。
2.  **零空間投影**：在與常數向量正交的子空間中求解。CG 僅適用於該子空間上的正定限制。
本節假設系統已通過適當的邊界條件（如 Dirichlet）或投影處理，使得 $A$ 為 SPD。

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
*   **真殘差**：停止條件必須基於真殘差 $r_k = b - A x_k$，而非迭代中使用的局部遞推殘差（因浮點誤差累積）。

### 4. 誤差界與殘差關係
設 $e_k = x_k - x_*$ 為解誤差，$\|\cdot\|_A$ 為 $A$-範數，即 $\|v\|_A = \sqrt{v^T A v}$。
CG 的收斂界為：
$$
\|e_k\|_A \le 2 \left( \frac{\sqrt{\kappa(A)} - 1}{\sqrt{\kappa(A)} + 1} \right)^k \|e_0\|_A
$$
此外，解誤差的 2-範數與真殘差 $\|r_k\|_2$ 的關係為：
$$
\|e_k\|_2 \le \|A^{-1}\|_2 \|r_k\|_2 = \frac{1}{\lambda_{\min}(A)} \|r_k\|_2
$$
這意味著，即使殘差 $\|r_k\|_2$ 很小，若 $\lambda_{\min}(A)$ 極小（病態系統），解誤差 $\|e_k\|_2$ 仍可能很大。因此，僅依賴殘差小來判斷求解成功是不夠的，需結合問題物理意義或參考解驗證。

### 5. 預條件（Preconditioning）
若 $\kappa(A)$ 很大，CG 收斂慢。引入預條件矩陣 $M \approx A$。標準 PCG 要求 $M$ 為 SPD。
定義 $z_k = M^{-1} r_k$。
PCG 迭代步長為：
$$
\alpha_k = \frac{r_k^T z_k}{p_k^T A p_k}
$$
$$
\beta_k = \frac{r_{k+1}^T z_{k+1}}{r_k^T z_k}
$$
$$
p_{k+1} = r_{k+1} + \beta_k ( \frac{M}{M_{prev}} ) p_k \quad (\text{簡化為 } p_{k+1} = r_{k+1} + \beta_k M^{-1} M p_k \text{ 的變體，標準公式為 } p_{k+1} = r_{k+1} + \beta_k p_k \text{ 若 } M \text{ 對角或 } M \text{ 不變})
$$
嚴格來說，PCG 的 $p$ 更新為：
$$
p_{k+1} = r_{k+1} + \beta_k \frac{M_k}{M_{k-1}} p_k
$$
若 $M$ 不變，且我們使用 $z_k = M^{-1} r_k$，則標準形式為：
$$
p_{k+1} = r_{k+1} + \beta_k M^{-1} M p_k \quad \text{(此處需注意推導細節，通常實現為 } p_{k+1} = z_{k+1} + \beta_k p_k \text{ 若 } M \text{ 對角)}
$$
對於對角預條件（如 Jacobi），$M$ 是對角矩陣，$M^{-1}$ 是元素級除法。

*   **Jacobi 預條件**：$M = D$，其中 $D$ 是 $A$ 的對角矩陣。簡單且為 SPD（若 $A$ SPD 且對角元素非零）。
*   **Gauss–Seidel 預條件**：$M = (D + L)(D + U)^T$（Symmetric Gauss-Seidel / SSOR），以確保 $M$ 為 SPD 且對稱。單向 Gauss-Seidel $M=D+L$ 不對稱，不能直接用於標準 PCG，僅可用於 stationary iteration 或作為非對稱預條件（需配合 BiCGStab 等）。本章聚焦於對稱 SPD 系統，故僅討論 Jacobi 作為示範。

## 逐步手算例題

### 例 1：1D Poisson 三點系統（SPD）
問題：$-u''(x) = f(x)$，$x \in [0,1]$，$u(0)=u(1)=0$。
網格 $N=2$ 個內部節點，步長 $h = 1/3$。
$f(x) = 1$（常數來源）。
邊界：$u_0 = u_3 = 0$。
未知數：$u_1, u_2$。

離散方程：
$$
\frac{u_{i-1} - 2u_i + u_{i+1}}{h^2} = -f_i \implies -\frac{u_{i-1} - 2u_i + u_{i+1}}{h^2} = f_i
$$
即 $\frac{2u_i - u_{i-1} - u_{i+1}}{h^2} = f_i$。
$A = \frac{1}{h^2} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} = 9 \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 18 & -9 \\ -9 & 18 \end{bmatrix}$。
$b = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$。

**CG 迭代**：
1.  $x_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$。
2.  $r_0 = b - A x_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$。
3.  $p_0 = r_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$。
4.  $A p_0 = \begin{bmatrix} 18(1) - 9(1) \\ -9(1) + 18(1) \end{bmatrix} = \begin{bmatrix} 9 \\ 9 \end{bmatrix}$。
5.  $r_0^T r_0 = 2$。
6.  $p_0^T A p_0 = 1(9) + 1(9) = 18$。
7.  $\alpha_0 = \frac{2}{18} = \frac{1}{9}$。
8.  $x_1 = x_0 + \alpha_0 p_0 = \begin{bmatrix} 1/9 \\ 1/9 \end{bmatrix}$。
9.  $r_1 = r_0 - \alpha_0 A p_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} - \frac{1}{9} \begin{bmatrix} 9 \\ 9 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$。

**結果**：一步收斂。解為 $u_1 = u_2 = 1/9$。
驗證解析解：$u(x) = \frac{x(1-x)}{2}$。
$u(1/3) = \frac{1/3 (2/3)}{2} = \frac{2/9}{2} = 1/9$。
$u(2/3) = \frac{2/3 (1/3)}{2} = \frac{2/9}{2} = 1/9$。
完全吻合。

### 例 2：非 SPD 反例（故障測試）
設 $A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$，$b = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$。
此矩陣對稱，但特徵值為 $1, -1$，非正定。

**CG 迭代**：
1.  $x_0 = 0$，$r_0 = b = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$，$p_0 = r_0$。
2.  $A p_0 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$。
3.  $p_0^T A p_0 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 0$。
4.  $\alpha_0 = \frac{1}{0}$ **未定義（Breakdown）**。

**結論**：標準 CG 算法在 $p_k^T A p_k \le 0$ 時失敗（Breakdown）。這揭示了 SPD 是 CG 收斂的必要條件。在實作中，若偵測到 $p^T A p$ 接近零或負，必須中斷並報告錯誤狀態 `breakdown` 或 `nonpositive_curvature`。

## 實作與程式

以下提供一個自足的 Python 3.10+/NumPy CPU 實作。程式包含 CSR 構造（含契約檢查）、PCG 求解（含 Jacobi 預條件）、真殘差檢查、Breakdown 偵測及與直接解比較。

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
    A 是對角為 2/h^2，次對角為 -1/h^2 的三對角矩陣。
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    h = 1.0 / (N + 1)
    inv_h2 = 1.0 / (h * h)
    
    # 計算非零元素總數
    # 第 0 行: 2 個
    # 第 N-1 行: 2 個
    # 中間 N-2 行: 3 個
    if N == 1:
        total_nnz = 1 # 對角
    elif N == 2:
        total_nnz = 4 # 2+2
    else:
        total_nnz = 2 + 3*(N-2) + 2
    
    indptr = np.zeros(N + 1, dtype=int)
    indices = np.zeros(total_nnz, dtype=int)
    data = np.zeros(total_nnz)
    
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
    
    # 契約檢查
    assert nnz == total_nnz, "NNZ mismatch"
    assert indptr[N] == len(data), "indptr last element mismatch"
    assert np.all(np.isfinite(data)), "Data contains NaN/Inf"
    
    return indptr, indices, data

def pcg_solve_csr(indptr, indices, data, b, x0=None, rtol=1e-8, atol=1e-10, max_iter=1000, use_jacobi=True):
    """
    預條件共軛梯度求解 Ax = b，其中 A 由 CSR 格式提供。
    使用 Jacobi 預條件 (M = diag(A))。
    返回: (x, n_iter, true_residual_norm, converged, status)
    status: "success", "max_iter", "nan_inf", "breakdown"
    """
    n = len(b)
    
    # 輸入檢查
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(data)):
        return None, 0, np.nan, False, "nan_inf_input"
    if len(indptr) != n + 1 or indptr[-1] != len(data):
        return None, 0, np.nan, False, "invalid_csr_structure"
        
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    # 計算初始真殘差
    Ax = csr_matvec(indptr, indices, data, x)
    r = b - Ax
    norm_b = np.linalg.norm(b)
    
    # 計算真殘差範數
    norm_r_true = np.linalg.norm(r)
    if norm_r_true < atol + rtol * norm_b:
        return x, 0, norm_r_true, True, "success"
        
    # Jacobi 預條件: M = D, M_inv(r) = r / D
    # 提取對角元素
    diag = np.zeros(n)
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        # 找到列索引為 i 的位置
        row_indices = indices[start:end]
        if i in row_indices:
            idx_in_row = np.where(row_indices == i)[0][0]
            diag[i] = data[start + idx_in_row]
        else:
            return None, 0, np.nan, False, "missing_diagonal"
            
    if np.any(diag <= 0):
        return None, 0, np.nan, False, "non_spd_diagonal"
        
    # 初始預條件向量
    if use_jacobi:
        z = r / diag
    else:
        z = r.copy()
        
    p = z.copy()
    rz = np.dot(r, z)
    
    for k in range(max_iter):
        # 計算 Ap
        Ap = csr_matvec(indptr, indices, data, p)
        
        # 檢查 SPD 條件 / Breakdown: p^T A p 必須為正
        pAp = np.dot(p, Ap)
        # 使用相對容差避免尺度問題
        if pAp <= 1e-12 * np.dot(p, p) * np.dot(Ap, Ap):
            return x, k, norm_r_true, False, "breakdown"
            
        alpha = rz / pAp
        
        # 更新解
        x += alpha * p
        # 更新遞推殘差
        r -= alpha * Ap
        
        # 計算真殘差以進行停止檢查
        Ax_new = csr_matvec(indptr, indices, data, x)
        r_true = b - Ax_new
        norm_r_true = np.linalg.norm(r_true)
        
        # 停止條件檢查 (真殘差)
        if norm_r_true < atol + rtol * norm_b:
            return x, k + 1, norm_r_true, True, "success"
            
        # 檢查數值發散
        if not np.all(np.isfinite(r_true)):
            return x, k + 1, norm_r_true, False, "nan_inf_iter"
            
        # 計算預條件向量 z_new
        if use_jacobi:
            z_new = r_true / diag
        else:
            z_new = r_true.copy()
            
        rz_new = np.dot(r_true, z_new)
        beta = rz_new / rz
        rz = rz_new
        
        # 更新方向
        # 標準 PCG 更新: p_new = z_new + beta * p
        p = z_new + beta * p
        
    return x, max_iter, norm_r_true, False, "max_iter"

def direct_solve_dense(N, b):
    """
    使用密集矩陣直接求解，作為參考。
    """
    h = 1.0 / (N + 1)
    inv_h2 = 1.0 / (h * h)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 2 * inv_h2
        if i > 0: A[i, i-1] = -inv_h2
        if i < N - 1: A[i, i+1] = -inv_h2
    return np.linalg.solve(A, b)

# 測試例 1
N = 10
b = np.ones(N) # f=1
indptr, indices, data = build_poisson_csr_1d(N)

x_pcg, nit, res, conv, status = pcg_solve_csr(indptr, indices, data, b)
x_direct = direct_solve_dense(N, b)
diff = np.linalg.norm(x_pcg - x_direct)

print(f"Test 1 (SPD, N={N}):")
print(f"  Converged: {conv}, Iters: {nit}, Residual: {res:.2e}, Status: {status}")
print(f"  Difference vs Direct: {diff:.2e}")
# 預期: conv=True, nit < N, status=success, diff < 1e-10

# 測試例 2 (非 SPD 模擬)
# 手動構造一個非 SPD 的 CSR 矩陣以測試故障檢測
# A = [[0, 1], [1, 0]]
indptr_bad = np.array([0, 1, 2], dtype=int)
indices_bad = np.array([1, 0], dtype=int)
data_bad = np.array([1.0, 1.0])
b_bad = np.array([1.0, 0.0])

x_bad, nit_bad, res_bad, conv_bad, status_bad = pcg_solve_csr(indptr_bad, indices_bad, data_bad, b_bad, max_iter=10)
print(f"\nTest 2 (Non-SPD):")
print(f"  Converged: {conv_bad}, Iters: {nit_bad}, Status: {status_bad}")
# 預期: conv_bad=False, status_bad=breakdown
```

## 測試與預期結果

### 1. 正常測試（SPD Poisson）
*   **輸入**：1D Poisson，$N=10$，$f=1$，Dirichlet 邊界。
*   **預期**：
    *   `converged = True`。
    *   `niters < N`（通常小於 $N$）。
    *   `residual < 1e-8`。
    *   `status = "success"`。
    *   **驗證**：計算解析解 $u(x) = \frac{x(1-x)}{2}$ 的離散值，比較誤差應在 $O(h^2)$ 量級。與直接解的差異應在機精度範圍內（$<10^{-10}$）。

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
    *   `status = "breakdown"`。
    *   程式不應陷入無限期迴圈或產生 NaN。

### 4. 故障測試（NaN 輸入）
*   **輸入**：$b$ 包含 `np.nan`。
*   **預期**：
    *   `converged = False`。
    *   `status = "nan_inf_input"`。
    *   立即中斷。

## 除錯與常見陷阱

1.  **遞推殘差 vs 真殘差**：
    *   **錯誤**：僅使用 $r_{k+1} = r_k - \alpha A p_k$ 來檢查收斂。
    *   **風險**：浮點誤差累積導致遞推殘差與真殘差 $b-Ax$ 偏離，可能導致提前假收斂或延遲收斂。
    *   **修正**：在判斷停止前，重新計算 $r_{true} = b - A x$ 並檢查其範數。本實作在每次迭代後都計算真殘差以確保正確性（對於小 $N$ 可行；大 $N$ 可每隔 $K$ 步檢查）。
2.  **Breakdown 偵測**：
    *   **陷阱**：$p^T A p \le 0$。
    *   **原因**：$A$ 非 SPD 或數值誤差。
    *   **修正**：使用相對容差檢查 $p^T A p > \epsilon \|p\|_2 \|Ap\|_2$。若違反，回報 `breakdown` 而非除零。
3.  **CSR 契約錯誤**：
    *   **陷阱**：`indptr` 長度錯誤或 `indptr[n] != len(data)`。
    *   **修正**：在 MatVec 前加入斷言檢查。
4.  **預條件矩陣不對稱**：
    *   **陷阱**：使用單向 Gauss-Seidel ($M=D+L$) 作為標準 PCG 預條件。
    *   **原因**：$M$ 需為 SPD 以保證收斂。
    *   **修正**：使用 Jacobi ($M=D$) 或 SSOR ($M=(D+L)(D+U)^T$)。本章僅示範 Jacobi。
5.  **病態系統誤判**：
    *   **陷阱**：殘差小就認為解準確。
    *   **原因**：$\|e\|_2 \le \frac{1}{\lambda_{\min}} \|r\|_2$。
    *   **修正**：報告殘差時，需說明條件數或提供誤差估計。

## 養殖與相場案例

### 案例：池塘溫度穩態求解
考慮一個矩形養殖池，邊界溫度固定（Dirichlet 條件），內部無熱源（$f=0$）。求解穩態溫度分佈 $T(x,y)$。
*   **模型**：$-\nabla^2 T = 0$。
*   **離散**：2D 5-點拉普拉斯算子，生成稀疏 SPD 矩陣 $A$。
*   **右端**：$b = b_{\text{source}} + b_{\text{boundary}}$。若 $f=0$，則 $b_{\text{source}}=0$，但 $b_{\text{boundary}}$ 來自邊界值消去，通常非零。
*   **規模**：$64 \times 64$ 網格，$n=4096$。
*   **求解**：使用 PCG（Jacobi 預條件）。
*   **預期表現**：
    *   **收斂**：PCG 應在幾十至幾百次迭代內收斂。
    *   **物理一致性**：溫度場應平滑，無振盪。最大值/最小值應出現在邊界（極值原理）。
    *   **診斷**：監控真殘差 $\|b - Ax\|_2$。若殘差下降至 $10^{-8} \|b\|_2$，視為收斂。

### 相位橋接說明
在相場（Phase Field）模型（如 Cahn-Hilliard）中，線性化後的系統可能為非對稱或不定。若使用 Backward Euler 時間積分，得到系統 $(M/\Delta t + L) \phi^{n+1} = M \phi^n / \Delta t + \dots$。若空間算子 $L$ 為半正定且 $M$ 正定，則 $(M/\Delta t + L)$ 為 SPD，可用 CG。但若涉及混合形式或四階導數的特定離散，可能需鞍點求解器。本章僅限於 SPD 線性系統。

## 習題

1.  **手算**：對 3 點 1D Poisson（$N=3$，$h=1/4$，$f=1$），寫出 $A$ 和 $b$，並手算 CG 的第一步 $\alpha_0, x_1, r_1$。
2.  **程式**：修改 `pcg_solve_csr`，比較有/無 Jacobi 預條件時的迭代次數（使用 $N=16$ 的 1D 問題），並記錄真殘差與直接解的差異。
3.  **反例**：構造一個 $2 \times 2$ 對稱但非正定矩陣 $A$，示範 PCG 在哪一步失敗，並解釋 `breakdown` 的狀態。
4.  **整合**：將 1D Poisson 問題擴展到 2D（$N_x \times N_y$）。使用 CSR 構造 2D 拉普拉斯矩陣，求解 $-\Delta u = 1$，並比較 PCG 解與直接法（`np.linalg.solve` 對於密集矩陣）的結果，計算 $L_2$ 誤差。

## 習題解答

### 1. 手算解答
$N=3$，$h=1/4$，$inv\_h2 = 16$。
$A = 16 \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix} = \begin{bmatrix} 32 & -16 & 0 \\ -16 & 32 & -16 \\ 0 & -16 & 32 \end{bmatrix}$。
$b = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$。
$x_0 = 0$，$r_0 = b$。
$p_0 = r_0 = [1,1,1]^T$。
$A p_0 = [32-16, -16+32-16, -16+32]^T = [16, 0, 16]^T$。
$r_0^T r_0 = 3$。
$p_0^T A p_0 = 1(16) + 1(0) + 1(16) = 32$。
$\alpha_0 = 3 / 32$。
$x_1 = \frac{3}{32} [1, 1, 1]^T = [3/32, 3/32, 3/32]^T$。
$r_1 = r_0 - \alpha_0 A p_0 = [1,1,1]^T - \frac{3}{32}[16,0,16]^T = [1 - 1.5, 1, 1 - 1.5]^T = [-0.5, 1, -0.5]^T$。
驗證正交性：$r_0^T r_1 = 1(-0.5) + 1(1) + 1(-0.5) = 0$。成立。

### 2. 程式解答
（概念性）
無預條件：迭代次數約 $O(N)$。
有 Jacobi 預條件：對於 1D Poisson，$M=D=2/h^2 I$，預條件後條件數變為 $\kappa(\tilde{A}) \approx \kappa(A)/2$，改善有限。迭代次數可能略減。
直接解差異：應為 $0$（在機精度內）。

### 3. 反例解答
$A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$。
$p_0 = [1,0]^T$。
$Ap_0 = [0,1]^T$。
$p_0^T A p_0 = 0$。
檢查 $p^T A p > \epsilon \|p\| \|Ap\|$。
$\|p\|=1, \|Ap\|=1$。
$0 > \epsilon$ 為假。
返回 `status = "breakdown"`。

### 4. 整合解答
2D CSR 構造：
節點 $(i,j)$ 索引 $k = j \cdot N_x + i$。
鄰居：$(i-1,j), (i+1,j), (i,j-1), (i,j+1)$。
中心：$(i,j)$。
權重：中心 $4/h^2$（若 $dx=dy=h$），鄰居 $-1/h^2$。
比較 PCG 與直接解：
$L_2$ 誤差應 $< 10^{-10}$（若 PCG 收斂至 $10^{-8}$ 相對殘差）。

## 本章小結

本章詳述了稀疏矩陣的 CSR 格式及其在求解 SPD 線性系統中的核心角色。共軛梯度（CG）與預條件共軛梯度（PCG）算法通過構建 $A$-共軛方向，高效地最小化能量泛函，收斂速度取決於矩陣的條件數。Jacobi 預條件可通過改善條件數來加速收斂。實作中必須嚴格檢查真殘差與 SPD 條件（Breakdown），以避免數值不穩定。誤差界 $\|e\|_2 \le \lambda_{\min}^{-1} \|r\|_2$ 強調了病態系統中殘差小不等於誤差小。本章為後續非對稱系統求解（GMRES）及相位場模型離散奠定了基礎。

## 參考來源

*   F1: FiPy 有限體積離散與邊界 (https://pages.nist.gov/fipy/en/latest/numerical/discret.html) - 參考稀疏矩陣構造。
*   F3: PETSc 線性系統求解器 (https://petsc.org/release/manual/ksp/) - CG 與預條件詳細討論。
*   F4: SciPy 稀疏線性代數 API (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) - CSR 格式定義與 `cg` 函數。
*   F2: FEniCSx Poisson 與弱形式 (https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html) - SPD 物理背景。