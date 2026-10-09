# 第09章 隱式時間積分與剛性

## 學習目標與先備知識

本章深入探討處理剛性（Stiffness）偏微分方程（PDE）的隱式時間積分策略。剛性問題的特徵在於存在極大差異的時間尺度：高頻模態衰減極快，迫使顯式方法必須使用極小的時間步長 $ \Delta t $ 以維持穩定，即使我們只關心低頻模態的長期演化。讀者在完成前章的空間離散化與顯式擴散分析後，已理解 CFL 條件對顯式格式的嚴苛限制。本章目標如下：

1.  **剛性本質與 $O(N^2)$ 耦合**：理解離散擴散算子特徵值譜寬與網格密度 $N$ 的關係，推導剛性比量級為 $O(N^2)$。
2.  **隱式格式推導**：詳細推導 Backward Euler (BE) 與 Crank–Nicolson (CN) 的離散更新公式，明確區分線性系統求解的符號慣例。
3.  **A-穩定與 L-穩定性分析**：透過放大因子 $R(q)$ 分析穩定區域，解釋 BE 的 L-穩定性（高頻阻尼強）與 CN 的非 L-穩定性（高頻振盪）。
4.  **數值診斷與驗證**：建立包含線性殘差檢查、能量耗散測試與高頻模態衰減追蹤的完整驗證框架，區分時間誤差、空間誤差與數值穩定性。

**先備知識要求**：
*   熟悉線性代數中的特徵值分解，特別是對稱實矩陣的正交对角化。
*   理解稀疏矩陣存儲（CSR/CSC）與迭代求解器的基礎概念（如共軛梯度法 CG 或預條件無預條件的 GMRES）。
*   具備 Python 與 NumPy 基礎，能進行陣列運算與基本的線性系統求解。

## 問題與直覺

### 剛性的物理與數值根源

考慮一維熱擴散方程 $ u_t = \nu \partial_{xx} u $，其中 $\nu > 0$。在空間上使用二階中心差分並施加週期邊界條件，得到的半離散常微分方程組（ODE）為：
$$ \frac{d\mathbf{u}}{dt} = \mathbf{L}_h \mathbf{u} $$
其中 $\mathbf{L}_h$ 是離散 Laplacian 矩陣，其元素為 $ (1/\Delta x^2) $ 乘上差分模板。

顯式 Euler 方法的穩定性要求對於所有特徵值 $\lambda_k(\mathbf{L}_h)$，滿足 $ |1 + \Delta t \lambda_k| \le 1 $。由於 $\mathbf{L}_h$ 是對稱負半定矩陣，$\lambda_k$ 為實數且 $\le 0$。最壞情況來自於最高頻模態（Nyquist 模式），其特徵值約為：
$$ \lambda_{max} \approx -\frac{4\nu}{\Delta x^2} $$
穩定條件變為 $ \Delta t \le \frac{\Delta x^2}{2\nu} $。

**剛性問題定義**：
當域長 $L$ 固定，網格數 $N = L/\Delta x$ 增加時，$\Delta x \propto 1/N$，故 $\Delta t_{stable} \propto 1/N^2$。
然而，低頻模態（如 $k=1$）的物理衰減時間尺度為 $\tau_{phys} \sim L^2/\nu$，與 $N$ 無關。
為了計算 $\tau_{phys}$ 量級的演化，顯式方法需要的步數 $n_{steps} \propto N^2$。這就是剛性：數值穩定性要求遠超物理演化所需的時間解析度。

**隱式方法的直覺**：
隱式方法（如 BE）將未知量 $ \mathbf{u}^{n+1} $ 放在方程右側，導致需要求解線性系統。其穩定性不再依賴於 $ \Delta t |\lambda_k| $ 的上限，而是對於負實軸上的特徵值，隱式方法往往能保持穩定。這允許我們使用比顯式方法大得多的 $ \Delta t $，只要該步長能足夠準確地捕捉低頻模態的演化。

## 數學與物理推導

### 1. 符號慣例與變數定義

為了避免混淆，本章統一使用以下定義：
*   **ODE 形式**：$ \mathbf{u}' = \mathbf{A} \mathbf{u} $，其中 $\mathbf{A}$ 為空間離散算子（對擴散問題，$\mathbf{A}$ 為負半定對稱矩陣）。
*   **特徵值**：設 $\mathbf{A}$ 的特徵值為 $\mu_k \le 0$。
*   **正定變數 $q$**：定義 $ q_k = -\mu_k \Delta t \ge 0 $。此變數代表離散衰減的強度。$q$ 越大，表示該模態在一個時間步長內衰減得越劇烈。

### 2. Backward Euler (BE)

BE 格式定義為：
$$ \frac{\mathbf{u}^{n+1} - \mathbf{u}^n}{\Delta t} = \mathbf{A} \mathbf{u}^{n+1} $$
整理得線性系統：
$$ (\mathbf{I} - \Delta t \mathbf{A}) \mathbf{u}^{n+1} = \mathbf{u}^n $$
對於單個模態 $ y' = \mu y $，更新公式為：
$$ y^{n+1} = \frac{1}{1 - \Delta t \mu} y^n = \frac{1}{1 + q} y^n $$
其中 $ q = -\mu \Delta t \ge 0 $。

*   **放大因子**：$ R_{BE}(q) = \frac{1}{1+q} $。
*   **A-穩定性**：對於所有 $ q \ge 0 $，$ |R_{BE}(q)| \le 1 $。BE 是 A-穩定的。
*   **L-穩定性**：當 $ q \to \infty $ 時，$ \lim_{q \to \infty} R_{BE}(q) = 0 $。這意味著對於高頻剛性模態，BE 會將其幅度迅速抑制至零。這稱為 L-穩定性。
*   **截斷誤差**：
    *   差分殘差（Residual）：$ O(\Delta t) $。
    *   單步局部誤差（Local Error）：$ O(\Delta t^2) $。
    *   全局誤差（Global Error）：$ O(\Delta t) $。

### 3. Crank–Nicolson (CN)

CN 格式定義為：
$$ \frac{\mathbf{u}^{n+1} - \mathbf{u}^n}{\Delta t} = \frac{1}{2} \mathbf{A} (\mathbf{u}^{n+1} + \mathbf{u}^n) $$
整理得線性系統：
$$ (\mathbf{I} - \frac{\Delta t}{2} \mathbf{A}) \mathbf{u}^{n+1} = (\mathbf{I} + \frac{\Delta t}{2} \mathbf{A}) \mathbf{u}^n $$
對於單個模態，更新公式為：
$$ y^{n+1} = \frac{1 - \frac{\Delta t \mu}{2}}{1 + \frac{\Delta t \mu}{2}} y^n = \frac{1 - \frac{q}{2}}{1 + \frac{q}{2}} y^n $$

*   **放大因子**：$ R_{CN}(q) = \frac{1 - q/2}{1 + q/2} $。
*   **A-穩定性**：對於所有 $ q \ge 0 $，$ |R_{CN}(q)| \le 1 $。CN 也是 A-穩定的。
*   **非 L-穩定性**：當 $ q \to \infty $ 時，$ \lim_{q \to \infty} R_{CN}(q) = -1 $。
    *   **振盪問題**：對於高頻模態（$q$ 很大），CN 不會衰減，而是每步反號。這會導致網格尺度振盪（Grid-scale oscillations），且在長期模擬中不會消失。這是 CN 在處理剛性問題時的主要缺點。
*   **截斷誤差**：
    *   差分殘差：$ O(\Delta t^2) $。
    *   單步局部誤差：$ O(\Delta t^3) $。
    *   全局誤差：$ O(\Delta t^2) $。

### 4. 能量與耗散分析

定義離散 $L_2$ 能量（每單位長度）：
$$ E^n = \frac{1}{2} \sum_{i=0}^{N-1} (u_i^n)^2 \Delta x $$
對於擴散問題，連續能量滿足 $ \frac{dE}{dt} = -\nu \int (\partial_x u)^2 dx \le 0 $。

*   **BE 能量**：
    由於 $ |R_{BE}(q)| < 1 $ 對於所有 $ q > 0 $，每個模態的能量都在衰減。對於對稱算子，離散能量 $ E^{n+1} \le E^n $ 嚴格成立（除零模態外）。
*   **CN 能量**：
    雖然 $ |R_{CN}(q)| \le 1 $ 保證範數不增（穩定），但對於高頻模態 $ R_{CN} \approx -1 $，其能量 $ |y|^2 $ 保持不變。這意味著 CN 在數值上「保存」了高頻雜訊的能量，而物理上這些能量應被耗散。

## 逐步手算例題

### 例 1：標量 ODE 的穩定性與誤差比較

考慮 ODE $ y' = -\lambda y $，$ y(0)=1 $，其中 $\lambda = 100$。這代表一個極快的衰減過程。
目標時間 $ t=1 $。解析解 $ y(t) = e^{-100 t} $。在 $ t=1 $，$ y(1) \approx 3.72 \times 10^{-44} \approx 0 $。

我們比較不同步長下的表現，關注數值解是否「過度衰減」或「振盪」。

#### 情況 A：大步長 $\Delta t = 0.1$（10 步）
$q = \lambda \Delta t = 10$。

1.  **顯式 Euler (FE)**：
    $ R_{FE} = 1 - q = 1 - 10 = -9 $。
    $ |R| > 1 $，**不穩定**。解將指數爆炸並交替符號。

2.  **Backward Euler (BE)**：
    $ R_{BE} = \frac{1}{1+10} = \frac{1}{11} \approx 0.0909 $。
    $ y^{10} = (1/11)^{10} \approx 8.2 \times 10^{-10} $。
    解析解接近 0。BE 結果合理，迅速衰減至零。

3.  **Crank–Nicolson (CN)**：
    $ R_{CN} = \frac{1 - 5}{1 + 5} = \frac{-4}{6} = -\frac{2}{3} \approx -0.667 $。
    $ y^{10} = (-2/3)^{10} \approx 0.017 $。
    解析解 $\approx 0$。CN 結果顯示 $1.7\%$ 的剩餘振幅，且每步反號。對於剛性問題，這被視為非物理的振盪殘留。

#### 情況 B：細步長 $\Delta t = 0.001$（1000 步）
$q = \lambda \Delta t = 0.1$。

1.  **BE**：
    $ R_{BE} = \frac{1}{1.1} \approx 0.9091 $。
    $ y^{1000} = (0.9091)^{1000} \approx 1.27 \times 10^{-40} $。
    接近解析解。

2.  **CN**：
    $ R_{CN} = \frac{1 - 0.05}{1 + 0.05} = \frac{0.95}{1.05} \approx 0.9048 $。
    $ y^{1000} \approx 2.6 \times 10^{-40} $。
    同樣接近解析解。在此小 $q$ 下，CN 精度更高（二階 vs 一階）。

**結論**：在 $ q \gg 1 $（剛性區）時，BE 表現優於 CN（無振盪）；在 $ q \ll 1 $（非剛性區）時，CN 精度優於 BE。

### 例 2：一維擴散方程的高頻模態衰減

問題：$ u_t = \nu u_{xx} $，域 $ [0, 2\pi] $，週期邊界。$ \nu = 1 $。
網格：$ N=20 $，$ \Delta x = 2\pi/20 = 0.1\pi $。
初始條件：$ u(x,0) = \cos(x) + 0.1 \cos(10x) $。
注意：$\cos(10x)$ 對應波數 $k=10$，即 $N/2$，為 Nyquist 模態。

離散特徵值公式（週期節點中心網格）：
$$ \lambda_k = -\frac{4\nu}{\Delta x^2} \sin^2\left(\frac{\pi k}{N}\right) $$
1.  **低頻模態 $k=1$** ($ \cos(x) $)：
    $ \lambda_1 = -\frac{4}{(0.1\pi)^2} \sin^2\left(\frac{\pi}{20}\right) \approx -\frac{40}{\pi^2} (0.156)^2 \approx -0.397 $。
    物理衰減率 $\approx e^{-t}$。

2.  **高頻模態 $k=10$** ($ \cos(10x) $)：
    $ \lambda_{10} = -\frac{4}{(0.1\pi)^2} \sin^2\left(\frac{\pi}{2}\right) = -\frac{40}{\pi^2} \approx -4.05 $。
    衰減率極快。

設定 $\Delta t = 0.1$。
*   **低頻 $q_1$**：$ q_1 = -\lambda_1 \Delta t \approx 0.0397 $。
    $ R_{BE} \approx 0.962 $，$ R_{CN} \approx 0.923 $。
*   **高頻 $q_{10}$**：$ q_{10} = -\lambda_{10} \Delta t \approx 0.405 $。
    $ R_{BE} = \frac{1}{1.405} \approx 0.712 $。
    $ R_{CN} = \frac{1 - 0.2025}{1 + 0.2025} \approx 0.666 $。

若增加 $\nu$ 至 $100$（強剛性）：
$ \lambda_{10} \approx -405 $。
$ q_{10} = 40.5 $。
*   **BE**：$ R_{BE} \approx 1/41.5 \approx 0.024 $。高頻迅速衰減。
*   **CN**：$ R_{CN} \approx \frac{1-20.25}{1+20.25} \approx -0.94 $。高頻振幅幾乎不減，且每步反號。

## 實作與程式

以下提供一個自足的 Python 實作，使用 NumPy 和 SciPy（僅用於稀疏矩陣操作，符合依賴契約）。程式包含輸入驗證、殘差檢查與能量追蹤。

```python
import numpy as np
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve

def validate_inputs(u0, nu, dx, dt, n_steps):
    if not np.all(np.isfinite(u0)):
        raise ValueError("u0 contains non-finite values")
    if nu <= 0 or dx <= 0 or dt <= 0:
        raise ValueError("nu, dx, dt must be positive")
    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")

def assemble_laplacian_periodic(N, dx):
    """
    Assemble the periodic second-order difference matrix L_h 
    such that L_h @ u approximates u_xx.
    Diagonal: -2/dx^2, Off-diagonal: 1/dx^2.
    """
    main_diag = np.full(N, -2.0 / dx**2)
    off_diag = np.full(N - 1, 1.0 / dx**2)
    
    # Sparse matrix construction
    L = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(N, N), format='csr')
    
    # Add periodic boundary conditions (corners)
    # In CSR, we need to modify the matrix or construct with lil first
    L_lil = L.tolil()
    L_lil[0, -1] = 1.0 / dx**2
    L_lil[-1, 0] = 1.0 / dx**2
    
    return L_lil.tocsr()

def solve_implicit(u0, nu, dx, dt, n_steps, method='BE', atol=1e-10):
    """
    Solve u_t = nu * u_xx using Backward Euler (BE) or Crank-Nicolson (CN).
    """
    validate_inputs(u0, nu, dx, dt, n_steps)
    N = len(u0)
    
    # Assemble Laplacian
    L_h = assemble_laplacian_periodic(N, dx)
    
    # Precompute matrices
    I = eye(N, format='csr')
    
    if method == 'BE':
        # (I - dt * nu * L_h) u_new = u_old
        A_mat = I - dt * nu * L_h
        B_mat = I
    elif method == 'CN':
        # (I - 0.5 * dt * nu * L_h) u_new = (I + 0.5 * dt * nu * L_h) u_old
        A_mat = I - 0.5 * dt * nu * L_h
        B_mat = I + 0.5 * dt * nu * L_h
    else:
        raise ValueError("Method must be 'BE' or 'CN'")
        
    u = u0.copy()
    residual_history = []
    energy_history = []
    
    for step in range(n_steps):
        rhs = B_mat @ u
        u_new = spsolve(A_mat, rhs)
        
        # Check linear residual r = rhs - A_mat @ u_new
        r = rhs - A_mat @ u_new
        rel_res = np.linalg.norm(r, ord=2) / max(np.linalg.norm(rhs, ord=2), 1e-12)
        residual_history.append(rel_res)
        
        # Check Energy E = 0.5 * sum(u^2) * dx
        energy = 0.5 * np.sum(u_new**2) * dx
        energy_history.append(energy)
        
        u = u_new
        
    return u, residual_history, energy_history

# --- Example Usage ---
if __name__ == "__main__":
    # Parameters
    L_domain = 2 * np.pi
    N = 20
    dx = L_domain / N
    x = np.linspace(0, L_domain, N, endpoint=False) # Node-centered, periodic
    
    nu = 10.0 # Stiff diffusivity
    dt = 0.05
    n_steps = 10
    t_end = dt * n_steps
    
    # Initial condition: Low freq + High freq (Nyquist)
    u0 = np.cos(x) + 0.1 * np.cos(10 * x)
    
    # Solve
    u_be, res_be, e_be = solve_implicit(u0, nu, dx, dt, n_steps, method='BE')
    u_cn, res_cn, e_cn = solve_implicit(u0, nu, dx, dt, n_steps, method='CN')
    
    # Analyze High Frequency Component (projection onto cos(10x))
    # Since grid is periodic, we can check the amplitude of the highest mode
    # by looking at alternating sign or FFT. For simplicity, check max deviation from smooth solution.
    
    print(f"Time step dt = {dt}")
    print(f"Max Linear Residual (BE): {max(res_be):.2e}")
    print(f"Max Linear Residual (CN): {max(res_cn):.2e}")
    print(f"Final Energy (BE): {e_be[-1]:.4f} (Initial: {0.5*np.sum(u0**2)*dx:.4f})")
    print(f"Final Energy (CN): {e_cn[-1]:.4f}")
    
    # Check for oscillations in CN high-frequency part
    # If CN oscillates, the solution will have significant alternating components.
    # We can estimate high freq energy by subtracting a smoothed version or using FFT.
    # Here we just print the last few values to inspect visually or compute variance.
    var_be = np.var(np.diff(u_be)) # Rough proxy for gradient energy
    var_cn = np.var(np.diff(u_cn))
    print(f"Gradient Variance Proxy (BE): {var_be:.6f}")
    print(f"Gradient Variance Proxy (CN): {var_cn:.6f}")
```

## 測試與預期結果

### 1. 正常測試
*   **常數場測試**：$ u(x,0) = 1 $。
    *   預期：$ u(x,t) = 1 $ 對所有 $t$。
    *   殘差：應接近機器精度（$10^{-15}$）。
    *   能量：保持不變。
*   **線性收斂測試**：
    *   固定 $ \Delta x $，變化 $ \Delta t $。
    *   比較 $ \|\mathbf{u}^{num} - \mathbf{u}^{ref}\| $，其中 $ \mathbf{u}^{ref} $ 由極小 $ \Delta t $（如 $ \Delta t/10 $）的 BE 解得到。
    *   BE 誤差應以 $ O(\Delta t) $ 下降。
    *   CN 誤差應以 $ O(\Delta t^2) $ 下降（僅限於低頻模態主導時）。

### 2. 邊界與剛性測試
*   **高頻振盪檢測**：
    *   設定 $ \nu $ 很大，使得 $ q_{high} \gg 1 $。
    *   **預期**：BE 解的高頻部分迅速衰減至零。CN 解的高頻部分振幅保持初始水平，且符號交替。
    *   **診斷**：計算解的 $ L_2 $ 能量。CN 的能量衰減緩慢於物理預期（因為高頻能量未被耗散）。

### 3. 故障測試
*   **輸入驗證**：
    *   $ \nu \le 0 $：應拋出 `ValueError`。
    *   非有限值 `u0`：應拋出 `ValueError`。
*   **線性系統奇異**：
    *   若邊界條件導致 $ \mathbf{A} $ 奇異（如純 Neumann 且右端不相容），`spsolve` 可能失敗或返回不正確結果。應檢查 $ \mathbf{b} $ 是否屬於行空間。

### 預期結果總結
1.  **BE**：在高剛性情況下，解平滑，無振盪，但時間誤差較大。
2.  **CN**：精度較高（對低頻），但高頻部分可能出現持久振盪，導致解不平滑。
3.  **殘差**：兩者線性殘差均應小於 $10^{-8}$（取決於 `atol`）。
4.  **能量**：兩者離散能量均不增（穩定），但 CN 的能量衰減曲線在高頻部分會「平坦化」。

## 除錯與常見陷阱

1.  **Laplacian 符號錯誤**：
    *   陷阱：在組裝 $ \mathbf{A} = \mathbf{I} - \Delta t \nu \mathbf{L}_h $ 時，若 $\mathbf{L}_h$ 定義為正定（即 $ -\Delta $），則符號會反。
    *   修正：明確定義 $\mathbf{L}_h$ 為近似 $ \partial_{xx} $ 的算子（負半定）。檢查特徵值是否 $\le 0$。
2.  **週期邊界漏角**：
    *   陷阱：使用 `diags` 時忘記添加 $(0, N-1)$ 和 $(N-1, 0)$ 的元素。
    *   影響：邊界節點不滿足週期性，導致低頻模態錯誤，質量不守恆。
3.  **CN 振盪誤解**：
    *   陷阱：認為 CN 的振盪是「不穩定」。
    *   修正：CN 在 $ L_2 $ 範數下是穩定的（$ |R| \le 1 $），但缺乏耗散性。這稱為「偽穩定」或「非耗散穩定」。
4.  **誤差評估基準**：
    *   陷阱：直接用 $ \Delta t $ 細化來評估收斂，但未固定 $ \Delta x $。
    *   修正：時間收斂測試必須固定空間網格，並使用半離散參考解或極細時間步長。

## 養殖與相場案例

### 1. 養殖池溫度與溶氧擴散
**情境**：
養殖池溫度分布受水體混合與熱擴散影響。局部加熱（如曝氣）產生高頻溫度梯度。
**應用**：
*   **顯式**：若網格細化至捕捉局部加熱區（$\Delta x$ 小），顯式步長需極小，計算量過大。
*   **隱式 (BE)**：可用較大步長（如每小時一步），快速計算全域溫度。BE 的強阻尼特性會平滑掉微小的溫度波動，這在物理上是合理的（熱擴散具有平滑效應）。
*   **隱式 (CN)**：若需精確追蹤溫躍層，CN 精度較高，但若初始條件或源項含有高頻雜訊（如感測器噪聲），CN 可能保留這些雜訊，導致溫度場出現非物理振盪。
**建議**：對於監控級別應用，BE 通常更穩健；對於研究級別應用，可先用 BE 進行幾步以消除高頻雜訊，再切換至 CN 提高精度（Rannacher 平滑策略）。

### 2. 相場（相分離）模型中的剛性
**情境**：
Cahn–Hilliard 方程是四階 PDE，具有極強的剛性。
**應用**：
*   純 BE/CN 在四階問題上效率低。
*   實務上常用「半隱式」方法：對線性項（擴散）用隱式，對非線性項（反應/雙井勢）用顯式。
*   本章的 BE/CN 分析可用於評估線性部分的穩定性，並為半隱式格式的步長選擇提供依據。

## 習題

1.  **手算**：
    考慮 ODE $ y' = -100 y $，$ y(0)=1 $。使用 $\Delta t = 0.1$ 進行兩步計算。
    (a) 計算 Backward Euler 的 $ y^1, y^2 $。
    (b) 計算 Crank–Nicolson 的 $ y^1, y^2 $。
    (c) 計算解析解 $ y(0.2) $。
    (d) 解釋為何 CN 結果在高頻剛性下可能不被物理接受。

2.  **程式**：
    修改 `solve_implicit` 函式，增加 `method='FE'`（顯式 Euler）。
    (a) 實現 FE 更新公式。
    (b) 設定 $\nu=1, \Delta x=0.1, \Delta t=0.06$。觀察並記錄 FE 的振幅變化（爆炸）。
    (c) 比較 FE 與 BE 在同一參數下的穩定性。

3.  **反例**：
    構造一個一週期內具有正負交替的初始條件（如 $ u_i = (-1)^i $）。
    (a) 使用 CN 求解 10 步。
    (b) 證明其振幅未衰減（或衰減極少），且符號保持交替。
    (c) 使用 BE 求解 10 步，證明振幅迅速衰減。

4.  **整合**：
    假設你要模擬一個剛性擴散問題，要求相對誤差 $< 1\%$。
    (a) 估計 BE 所需的時間步長 $ \Delta t_{BE} $ 與 CN 所需的 $ \Delta t_{CN} $ 的比例（假設誤差主要來自時間離散）。
    (b) 若高頻模態衰減率為 $10^{-6}$，哪種方法更適合？說明理由。

## 習題解答

1.  **手算解答**：
    *   (a) BE: $ q = 10 $。$ R = 1/11 $。
        $ y^1 = 1/11 \approx 0.0909 $。
        $ y^2 = (1/11)^2 \approx 0.00826 $。
    *   (b) CN: $ R = (1-5)/(1+5) = -2/3 $。
        $ y^1 = -2/3 \approx -0.6667 $。
        $ y^2 = (-2/3)^2 \approx 0.4444 $。
    *   (c) 解析: $ y(0.2) = e^{-20} \approx 2.06 \times 10^{-9} \approx 0 $。
    *   (d) CN 結果 $0.44$ 遠大於解析解 $0$，且第一步符號反轉。這顯示 CN 在 $ q \gg 1 $ 時缺乏耗散，保留了高頻振幅，屬於非物理行為。

2.  **程式解答**：
    *   (a) FE: $ u^{n+1} = u^n + \Delta t \nu L_h u^n $。
    *   (b) $ \Delta t = 0.06, \nu=1, \Delta x=0.1 $。
        $ \Delta t / \Delta x^2 = 0.06 / 0.01 = 6 $。
        穩定條件 $ \nu \Delta t / \Delta x^2 \le 0.5 $。$ 6 > 0.5 $，故 FE 不穩定。
        預期結果：$ u $ 的振幅將指數爆炸。
    *   (c) BE 無此限制，振幅將衰減。

3.  **反例解答**：
    *   初始 $ u_0 = [1, -1, 1, -1, \dots] $。這是 Nyquist 模態。
    *   CN: $ R \approx -1 $（若 $ q $ 大）。$ u_{n+1} \approx - u_n $。
        10 步後：$ u_{10} \approx u_0 $。振幅不變。
    *   BE: $ R \approx 0 $（若 $ q $ 大）。$ u_{10} \approx 0 $。振幅消失。

4.  **整合解答**：
    *   (a) BE 誤差 $ \propto \Delta t $，CN 誤差 $ \propto \Delta t^2 $。
        若 $ \epsilon $ 為目標誤差，$ \Delta t_{BE} \propto \epsilon $，$ \Delta t_{CN} \propto \sqrt{\epsilon} $。
        比例 $ \Delta t_{BE} / \Delta t_{CN} \propto \sqrt{\epsilon} $。
        若 $ \epsilon = 0.01 $，$ \Delta t_{BE} \approx 0.1 \Delta t_{CN} $。BE 需要更小的步長（更多步數）來達到相同精度。
    *   (b) 若高頻衰減極快（剛性強），BE 更適合。因為 BE 的 L-穩定性能立即抑制高頻模態，避免其污染低頻計算。CN 的高頻模態不衰減，可能導致數值雜訊累積或影響後續非線性耦合項的穩定性。

## 本章小結

本章詳細探討了隱式時間積分方法在剛性 PDE 中的應用。主要結論如下：
1.  **剛性問題**迫使顯式方法使用極小步長（$O(N^{-2})$），而隱式方法可突破此限制。
2.  **Backward Euler (BE)** 是一階、A-穩定且 L-穩定的方法。其 L-穩定性使其在抑制高頻數值雜訊方面表現優異，但代價是一階收斂速度和低頻過度衰減。
3.  **Crank–Nicolson (CN)** 是二階、A-穩定但非 L-穩定的方法。其精度高，但對高頻模態的衰減不足（放大因子趨近 -1），可能導致持續振盪。
4.  **選擇策略**：
    *   若關注高頻穩定性和雜訊抑制，選 BE。
    *   若關注精度且高頻模態不重要或已預處理，選 CN。
    *   實務上可採用混合策略（Rannacher smoothing）：先做幾步 BE 以平滑高頻，再轉 CN 以提高低頻精度。
5.  **驗證**：必須通過能量下降（耗散）、模態衰減分析和製造解收斂測試來驗證數值解的可信度。線性殘差檢查是偵測組裝錯誤的必要手段。

## 參考來源

1.  [F1] FiPy有限體積離散與邊界. NIST FiPy Documentation.
2.  [F3] PETSc線性系統求解器. PETSc User Manual.
3.  [F4] SciPy稀疏線性代數API. SciPy Documentation.
4.  LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
5.  Quarteroni, A., Sacco, R., & Saleri, F. (2007). *Numerical Mathematics*. Springer.