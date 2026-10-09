# 第29章 固液相變、潛熱與Stefan橋接

## 學習目標與先備知識

本章探討固體與液體相變（如冰融化、合金凝固）的數值模擬，核心在於處理潛熱（latent heat）釋放或吸收對熱傳導的影響。與第25章（熱力學橋接）及第28章（Cahn–Hilliard）不同，本章聚焦於經典的熱傳導-相變耦合問題，即 Stefan 問題的擴散形式。讀者應已掌握前章的有限體積法（FVM）基礎、一維熱擴散方程的離散化，以及非線性方程的線性化技巧。

**核心目標**：
1.  **理解焓法（Enthalpy Method）**：將非連續的潛熱項轉化為連續的焓-溫度關係，避免顯式追蹤移動邊界。
2.  **Stefan 界面條件的隱式處理**：理解「糊狀區」（mushy zone）或「Stefan 橋接」概念，通過寬化相變溫度區間來數值穩定化界面位置。
3.  **數值穩定性與熱收支**：分析有效熱容對時間步長的剛性影響，並建立基於焓變化的熱收支檢驗（heat balance check）。
4.  **區分類似現象**：明確區分固液相變（涉及內能跳變、界面移動）與溶氧閾值管理（純傳輸，無潛熱），以及 Cahn–Hilliard 相分離（自由能驅動的相態演化）。

**先備知識**：
*   熱傳導方程：$ \rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q $。
*   有限體積法（FVM）在 1D 域上的組裝，特別是共享面通量（face flux）的概念。
*   隱式時間積分方法（如後向 Euler），用於處理剛性系統。

## 問題與直覺

### 顯熱與潛熱的挑戰

在常規熱擴散問題中，溫度 $T$ 是連續變化的狀態變量，比熱容 $c_p$ 通常為常數。然而，在固液相變過程中（如冰融化為水），物質在相變溫度 $T_{melt}$ 附近吸收或釋放潛熱 $L$（單位：J/kg），而溫度保持不變（或變化極小）。這導致了兩個數值挑戰：

1.  **熱容跳變**：焓 $H$（單位體積內能，J/m³）對溫度 $T$ 的導數 $\frac{dH}{dT} = \rho c_p + \rho L \delta(T - T_{melt})$ 在相變點出現狄拉克 delta 函數（Dirac delta），意味著有效比熱容在該點趨近無窮大。
2.  **界面運動**：傳統 Stefan 問題要求明確追蹤固液界面位置 $s(t)$，並滿足界面連續條件（溫度相等）與 Stefan 條件（能量守恆）。顯式追蹤界面在界面發生合併或分裂時非常困難。

**直覺解法：糊狀區焓法（Mushy Zone Enthalpy Method）**

為避免顯式追蹤界面，我們引入序參量 $\phi$（液相體積分數）。在純固體區 $\phi=0$，純液體區 $\phi=1$。
**重要假設**：
*   **純物質 vs 合金**：對於純物質（如冰），理想相變發生於單一溫度。此時的「糊狀區」寬度 $\Delta T_m$ 主要是一個**數值正則化參數**（Numerical Regularization），用於平滑陡峭的 $\phi(T)$ 關係。對於合金或多組分材料，$\Delta T_m$ 可對應物理上的固相線（solidus）與液相線（liquidus）之間的溫區。
*   **質量守恆約束**：本模型假設固液密度相同且為常數 $\rho$，無對流，固定域。這意味著我們只追蹤能量與相態分佈，忽略因密度差引起的浮力對流或體積變化。若密度不同，需耦合動量方程，本章不予討論。

總單位體積焓定義為：
$$ H(T, \phi) = \rho \int_{T_{ref}}^{T} c_p(\xi) d\xi + \rho L \phi $$
其中 $\rho$ 為密度（kg/m³），$c_p$ 為單位質量比熱容（J/(kg·K)），$L$ 為單位質量潛熱（J/kg）。
簡化假設 $c_p$ 為常數，則：
$$ H(T, \phi) = \rho c_p (T - T_{ref}) + \rho L \phi $$

### 與溶氧閾值的區別

養殖池中的溶氧濃度跨過「管理閾值」（如 5 mg/L）並不涉及任何物質的相態改變或潛熱釋放。這是一個純傳輸-反應問題，其物性參數（擴散係數、反應速率）不隨閾值發生斷裂或跳變。因此，溶氧模型不需要焓法，也不需要相變項。混淆這兩者會導致模型物理意義錯誤。

## 數學與物理推導

### 1. 有效熱容與離散化方程

將焓的時間導數展開：
$$ \frac{\partial H}{\partial t} = \rho c_p \frac{\partial T}{\partial t} + \rho L \frac{\partial \phi}{\partial t} $$
若假設相態分數 $\phi$ 僅依賴於溫度 $T$（局部平衡假設），則 $\frac{\partial \phi}{\partial t} = \frac{d\phi}{dT} \frac{\partial T}{\partial t}$。
定義 **單位體積有效熱容** $C_{eff}(T)$：
$$ C_{eff}(T) = \rho c_p + \rho L \frac{d\phi}{dT} $$
能量守恆方程變為：
$$ C_{eff}(T) \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) $$

在 1D 有限體積法中，設域長 $L_x$，分為 $N$ 個 cell-centered 節點，$\Delta x = L_x/N$。
第 $i$ 個 cell 中心位於 $x_i = (i+0.5)\Delta x$，體積 $V_i = A \Delta x$（設截面積 $A=1$ m²）。
對時間採用後向 Euler 隱式離散：
$$ \frac{C_{eff,i}^n (T_i^{n+1} - T_i^n)}{\Delta t} V_i = F_{i-1/2}^{n+1} - F_{i+1/2}^{n+1} $$
其中 $F$ 為通過面的熱流率（W）。

### 2. 相態分數函數 $\phi(T)$

為了數值穩定，使用線性平滑函數定義 $\phi(T)$。設相變溫度區間為 $[T_{melt} - \Delta T_m/2, T_{melt} + \Delta T_m/2]$：
$$ \phi(T) = \begin{cases} 0 & T \le T_{melt} - \frac{\Delta T_m}{2} \\ \frac{T - (T_{melt} - \Delta T_m/2)}{\Delta T_m} & T_{melt} - \frac{\Delta T_m}{2} < T < T_{melt} + \frac{\Delta T_m}{2} \\ 1 & T \ge T_{melt} + \frac{\Delta T_m}{2} \end{cases} $$
在此區間內，$\frac{d\phi}{dT} = \frac{1}{\Delta T_m}$。
**重要區分**：$\Delta T_m$ 是溫度區間（單位 K），不是空間厚度（單位 m）。對應的空間糊狀區厚度 $l_{mushy} \approx \Delta T_m / |\nabla T|$。

### 3. Stefan 條件的隱式化

傳統 Stefan 條件在界面 $x=s(t)$ 處為：
$$ \rho L \frac{ds}{dt} = k_s \left. \frac{\partial T}{\partial x} \right|_{solid} - k_l \left. \frac{\partial T}{\partial x} \right|_{liquid} $$
**符號慣例**：假設液相位於 $x < s(t)$，固相位於 $x > s(t)$，界面法向沿 $+x$。左側為單位面積的質量流率乘以潛熱（W/m²），右側為兩側熱流差（W/m²）。
在糊狀區方法中，我們不顯式計算 $ds/dt$，而是通過求解上述耦合方程，讓 $\phi$ 場自然演化，從而隱式確定界面位置。

### 4. 有限體積矩陣組裝推導

將離散方程寫為標準線性系統 $ \mathbf{A} \mathbf{T}^{n+1} = \mathbf{b}^n $。
對於內部 cell $i$ ($0 < i < N-1$)：
西面通量 $F_{i-1/2} = -k A \frac{T_i - T_{i-1}}{\Delta x}$，東面通量 $F_{i+1/2} = -k A \frac{T_{i+1} - T_i}{\Delta x}$。
注意通量方向定義：$F$ 為流入 cell 的正值。
$$ C_{eff,i}^n \frac{T_i^{n+1} - T_i^n}{\Delta t} A \Delta x = \left( -k A \frac{T_i^{n+1} - T_{i-1}^{n+1}}{\Delta x} \right) - \left( -k A \frac{T_{i+1}^{n+1} - T_i^{n+1}}{\Delta x} \right) $$
兩邊除以 $A \Delta x$：
$$ \frac{C_{eff,i}^n}{\Delta t} (T_i^{n+1} - T_i^n) = \frac{k}{\Delta x^2} (T_{i-1}^{n+1} + T_{i+1}^{n+1} - 2T_i^{n+1}) $$
整理得：
$$ -\frac{k}{\Delta x^2} T_{i-1}^{n+1} + \left( \frac{C_{eff,i}^n}{\Delta t} + \frac{2k}{\Delta x^2} \right) T_i^{n+1} - \frac{k}{\Delta x^2} T_{i+1}^{n+1} = \frac{C_{eff,i}^n}{\Delta t} T_i^n $$

**邊界條件處理**：
1.  **左邊界 (Dirichlet $T_0^{bnd}$)**：
    Cell 0 中心位於 $x_0 = \Delta x/2$。
    西面距離為 $\Delta x/2$。通量 $F_{-1/2} = -k A \frac{T_0^{n+1} - T_0^{bnd}}{\Delta x/2} = \frac{2kA}{\Delta x}(T_0^{bnd} - T_0^{n+1})$。
    東面距離為 $\Delta x$。通量 $F_{1/2} = -k A \frac{T_1^{n+1} - T_0^{n+1}}{\Delta x}$。
    方程：
    $$ \frac{C_{eff,0}^n}{\Delta t} A \Delta x (T_0^{n+1} - T_0^n) = \frac{2kA}{\Delta x}(T_0^{bnd} - T_0^{n+1}) - \left( -k A \frac{T_1^{n+1} - T_0^{n+1}}{\Delta x} \right) $$
    除以 $A \Delta x$：
    $$ \frac{C_{eff,0}^n}{\Delta t} (T_0^{n+1} - T_0^n) = \frac{2k}{\Delta x^2}(T_0^{bnd} - T_0^{n+1}) + \frac{k}{\Delta x^2}(T_1^{n+1} - T_0^{n+1}) $$
    整理得：
    $$ \left( \frac{C_{eff,0}^n}{\Delta t} + \frac{3k}{\Delta x^2} \right) T_0^{n+1} - \frac{k}{\Delta x^2} T_1^{n+1} = \frac{C_{eff,0}^n}{\Delta t} T_0^n + \frac{2k}{\Delta x^2} T_0^{bnd} $$

2.  **右邊界 (Zero Flux / Neumann 0)**：
    Cell $N-1$ 中心位於 $x_{N-1} = (N-0.5)\Delta x$。
    東面為邊界，通量為 0。
    西面通量 $F_{N-3/2} = -k A \frac{T_{N-1}^{n+1} - T_{N-2}^{n+1}}{\Delta x}$。
    方程：
    $$ \frac{C_{eff,N-1}^n}{\Delta t} (T_{N-1}^{n+1} - T_{N-1}^n) = \frac{k}{\Delta x^2}(T_{N-2}^{n+1} - T_{N-1}^{n+1}) $$
    整理得：
    $$ -\frac{k}{\Delta x^2} T_{N-2}^{n+1} + \left( \frac{C_{eff,N-1}^n}{\Delta t} + \frac{k}{\Delta x^2} \right) T_{N-1}^{n+1} = \frac{C_{eff,N-1}^n}{\Delta t} T_{N-1}^n $$

## 逐步手算例題

### 例 1：有限一維域的單步融化近似

**設定**：
*   域：$x \in [0, 1]$ m，cell-centered 網格，$N=2$ 個 cell，$\Delta x = 0.5$ m。
*   初態：$T_0^0 = 5$ K, $T_1^0 = 5$ K。
*   邊界：$x=0$ 處 $T_{left}=10$ K，$x=1$ 處零通量。
*   參數：$\rho = 1$ kg/m³, $c_p = 1$ J/(kg·K), $k = 1$ W/(m·K), $L = 10$ J/kg。
*   $T_{melt} = 5$ K, $\Delta T_m = 2$ K（相變區 4 K 至 6 K）。
*   $\Delta t = 1$ s。

**步驟 1：計算初始有效熱容**
$T_0^0 = 5$ K 位於糊狀區中心。
$\phi_0^0 = (5 - 4) / 2 = 0.5$。
$\frac{d\phi}{dT} = 1/2 = 0.5$。
$C_{eff,0}^0 = \rho c_p + \rho L \frac{d\phi}{dT} = 1 + 10(0.5) = 6$ J/(m³·K)。
同理，$C_{eff,1}^0 = 6$ J/(m³·K)。

**步驟 2：組裝線性系統**
參數計算：
$\frac{C_{eff}}{\Delta t} = 6/1 = 6$。
$\frac{k}{\Delta x^2} = 1 / (0.5)^2 = 1 / 0.25 = 4$。
$\frac{2k}{\Delta x^2} = 8$。
$\frac{3k}{\Delta x^2} = 12$。

**Cell 0 方程**：
$$ (6 + 12) T_0^1 - 4 T_1^1 = 6(5) + 8(10) $$
$$ 18 T_0^1 - 4 T_1^1 = 30 + 80 = 110 $$

**Cell 1 方程**（右邊界零通量）：
$$ -4 T_0^1 + (6 + 4) T_1^1 = 6(5) $$
$$ -4 T_0^1 + 10 T_1^1 = 30 $$

**步驟 3：求解**
由第二式得：$4 T_0^1 = 10 T_1^1 - 30 \implies T_0^1 = 2.5 T_1^1 - 7.5$。
代入第一式：
$18(2.5 T_1^1 - 7.5) - 4 T_1^1 = 110$
$45 T_1^1 - 135 - 4 T_1^1 = 110$
$41 T_1^1 = 245$
$T_1^1 = 245 / 41 \approx 5.9756$ K。

$T_0^1 = 2.5(5.9756) - 7.5 = 14.939 - 7.5 = 7.439$ K。

**步驟 4：更新相態**
$T_0^1 = 7.439$ K $> 6$ K $\implies \phi_0^1 = 1.0$。
$T_1^1 = 5.9756$ K $< 6$ K $\implies \phi_1^1 = (5.9756 - 4) / 2 = 0.9878$。

**步驟 5：熱收支檢查**
左邊界通量（使用 $T_0^1$）：
$Q_{in} = \frac{2kA}{\Delta x} (T_{left} - T_0^1) \Delta t = \frac{2(1)(1)}{0.5} (10 - 7.439) (1) = 4 \times 2.561 = 10.244$ J/m²。

初始焓 $E^0$：
$H_0^0 = \rho c_p (5-4) + \rho L (0.5) = 1 + 5 = 6$ J/m³。
$H_1^0 = 6$ J/m³。
$E^0 = (6 + 6) \Delta x = 12 \times 0.5 = 6$ J/m²。

最終焓 $E^1$：
$H_0^1 = \rho c_p (7.439-4) + \rho L (1) = 3.439 + 10 = 13.439$ J/m³。
$H_1^1 = \rho c_p (5.9756-4) + \rho L (0.9878) = 1.9756 + 9.878 = 11.8536$ J/m³。
$E^1 = (13.439 + 11.8536) \times 0.5 = 25.2926 \times 0.5 = 12.6463$ J/m²。

收支誤差 $R = E^1 - E^0 - Q_{in} = 12.6463 - 6 - 10.244 = -3.5977$ J/m²。
*註：由於使用顯式 $C_{eff}^n$，當溫度跨越相變區端點時，$C_{eff}^n \Delta T \neq \Delta H$，導致能量不守恆。此誤差隨 $\Delta t$ 細化而減小。*

## 實作與程式

以下提供一個 1D cell-centered 有限體積實現。
*   **網格**：$N$ 個 cell，$\Delta x = L_x/N$。
*   **變量**：$T[i]$ 對應 cell $i$ 的中心溫度。
*   **邊界**：左側 Dirichlet $T_L$，右側零通量（Neumann 0）。
*   **方法**：後向 Euler 擴散，顯式有效熱容 $C_{eff}^n$。

```python
import numpy as np

def solve_stefan_1d(N, Lx, T_init, T_left, dt, n_steps, 
                    rho, cp, k, L_latent, T_melt, dT_m, 
                    dx=None):
    """
    Solve 1D Stefan problem using Enthalpy Method (Explicit Ceff).
    Cell-centered FVM.
    """
    if dx is None:
        dx = Lx / N
    else:
        if not np.isclose(N * dx, Lx, rtol=1e-5):
            raise ValueError("dx * N must equal Lx")
            
    # Input Validation
    params = [rho, cp, k, dt, dT_m, L_latent, Lx, T_init, T_left, T_melt]
    if not np.all(np.isfinite(params)):
        raise ValueError("All parameters must be finite")
    if any(val <= 0 for val in [rho, cp, k, dt, dT_m, Lx]):
        raise ValueError("Physical parameters and dt must be positive")
    if L_latent < 0:
        raise ValueError("L_latent must be non-negative")
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer >= 2")
    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")

    # Initial Conditions
    T = np.full(N, T_init)
    
    def get_phi(T_val):
        """Calculate phase fraction."""
        if T_val < T_melt - dT_m/2:
            return 0.0
        elif T_val > T_melt + dT_m/2:
            return 1.0
        else:
            return (T_val - (T_melt - dT_m/2)) / dT_m

    def get_ceff(T_val):
        """Calculate effective volumetric heat capacity."""
        base = rho * cp
        if T_val < T_melt - dT_m/2 or T_val > T_melt + dT_m/2:
            return base
        else:
            return base + rho * L_latent / dT_m

    def get_enthalpy(T_val, phi_val):
        """Calculate enthalpy relative to T_ref = T_melt - dT_m/2."""
        T_ref = T_melt - dT_m/2
        return rho * cp * (T_val - T_ref) + rho * L_latent * phi_val

    phi = np.array([get_phi(T[i]) for i in range(N)])
    E_init = np.sum([get_enthalpy(T[i], phi[i]) for i in range(N)]) * dx
    
    # History storage
    T_hist = [T.copy()]
    phi_hist = [phi.copy()]
    E_hist = [E_init]
    Q_in_cum = [0.0]
    R_balance_hist = [0.0]

    for step in range(n_steps):
        # 1. Compute Ceff for current time step (Explicit)
        Ceff = np.array([get_ceff(T[i]) for i in range(N)])
        
        # 2. Assemble Linear System A T_new = b
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        # Internal cells and boundary corrections
        for i in range(N):
            # Storage term
            storage = Ceff[i] / dt
            b[i] = storage * T[i]
            
            # Diffusion coefficients
            k_dx2 = k / (dx**2)
            
            # Diagonal starts with storage
            A[i, i] = storage
            
            if i > 0:
                # West face connection to i-1
                A[i, i-1] = -k_dx2
                A[i, i] += k_dx2
            else:
                # Left Boundary (Dirichlet)
                # West face is at distance dx/2, coefficient 2k/dx^2
                A[i, i] += 2 * k_dx2
                b[i] += 2 * k_dx2 * T_left
                
            if i < N - 1:
                # East face connection to i+1
                A[i, i+1] = -k_dx2
                A[i, i] += k_dx2
            else:
                # Right Boundary (Zero Flux)
                # No east neighbor, no extra diagonal term from diffusion
                # (Standard tridiagonal logic: center has 2 neighbors internally.
                # Boundary cells have 1 internal neighbor + 1 boundary.
                # Left: 1 internal + 1 boundary (weight 2) -> total diff weight 3
                # Right: 1 internal + 1 boundary (weight 0) -> total diff weight 1
                # Our assembly:
                # If i < N-1, we added k_dx2 for East.
                # If i == N-1, we didn't add East.
                # So for N-1, diagonal only got West's k_dx2.
                # Total diff contribution to diagonal is 1 * k_dx2. Correct.
                pass
        
        # Solve
        try:
            T_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"Matrix singular at step {step}")
        
        # 3. Update Phi
        phi_new = np.array([get_phi(T_new[i]) for i in range(N)])
        
        # 4. Calculate Heat Input for Balance Check
        # Flux at left boundary: q_left = (2k/dx) * (T_left - T_new[0]) * Area
        q_left = (2.0 * k / dx) * (T_left - T_new[0]) * 1.0 # Area = 1
        heat_in_step = q_left * dt
        
        # 5. Update State
        T = T_new
        phi = phi_new
        
        # 6. Store History and Check Balance
        E_curr = np.sum([get_enthalpy(T[i], phi[i]) for i in range(N)]) * dx
        Q_in_cum.append(Q_in_cum[-1] + heat_in_step)
        
        # Balance Residual
        R_balance = E_curr - E_init - Q_in_cum[-1]
        R_balance_hist.append(R_balance)
        
        T_hist.append(T.copy())
        phi_hist.append(phi.copy())
        E_hist.append(E_curr)

    return T_hist, phi_hist, E_hist, Q_in_cum, R_balance_hist

if __name__ == "__main__":
    N = 10
    Lx = 1.0
    T_init = 5.0
    T_left = 10.0
    dt = 1.0
    n_steps = 10
    
    rho = 1.0
    cp = 1.0
    k = 1.0
    L_latent = 10.0
    T_melt = 5.0
    dT_m = 2.0
    
    T_hist, phi_hist, E_hist, Q_in, R_bal = solve_stefan_1d(
        N, Lx, T_init, T_left, dt, n_steps,
        rho, cp, k, L_latent, T_melt, dT_m
    )
    
    print(f"Final Avg Phi: {np.mean(phi_hist[-1]):.4f}")
    print(f"Final Energy: {E_hist[-1]:.4f}")
    print(f"Total Heat Input: {Q_in[-1]:.4f}")
    print(f"Final Balance Error: {R_bal[-1]:.4e}")
```

## 測試與預期結果

### 1. 正常測試
*   **恆定溫度邊界**：若 $T_{left} > T_{melt}$，界面應向 $x>0$ 方向移動。$\phi$ 的積分（總液相體積）應隨時間增加。
*   **能量守恆**：
    定義總焓 $E = \sum_i V_i [\rho c_p (T_i - T_{ref}) + \rho L \phi_i]$。
    熱收支殘差 $R_E = E^{n+1} - E^0 - \sum_{k=0}^n Q_{in}^k$。
    由於使用顯式 $C_{eff}^n$，當溫度跨過相變區時，$C_{eff}^n \Delta T \neq H(T^{n+1}) - H(T^n)$，因此 $R_E$ 不為零，但應隨 $\Delta t$ 細化而收斂至 0。

### 2. 邊界與故障測試
*   **無潛熱（純熱擴散）**：設定 $L_{latent} = 0$。結果應與標準熱擴散解析解一致。
*   **過冷（Undercooling）**：若 $T_{left} < T_{melt} - \Delta T_m/2$，物體不應融化，$\phi$ 保持 0。
*   **輸入驗證**：
    *   $N < 2$：應拋出 `ValueError`。
    *   $dt \le 0$：應拋出 `ValueError`。
    *   $L_{latent} < 0$：應拋出 `ValueError`。
    *   非有限值輸入（如 `nan`）：應拋出 `ValueError`。
*   **數值抖動**：檢查 $T$ 在 $T_{melt}$ 附近是否有非物理振盪。若 $\Delta T_m$ 太小，可能需要更細的網格或隱式熱容。

### 3. 網格與時間收斂測試
*   **空間細化**：固定極小 $\Delta t$，增加 $N$（減小 $\Delta x$），觀察界面位置 $s_h(t)$ 的收斂。
*   **時間細化**：固定細網格，減少 $\Delta t$，觀察能量收支誤差 $R_E$ 的收斂階數。
*   **正則化極限**：固定 $\Delta x, \Delta t$，減少 $\Delta T_m$，解應趨近於尖銳 Stefan 問題（界面更薄），但計算剛性增加。

## 除錯與常見陷阱

1.  **單位量綱不一致**：
    *   **陷阱**：混淆比質量熱容 $c_p$ (J/kg·K) 與體積熱容 $\rho c_p$ (J/m³·K)。
    *   **修正**：確保焓 $H$ 的單位為 J/m³。$H = \rho c_p (T-T_{ref}) + \rho L \phi$。
2.  **邊界處理錯誤**：
    *   **陷阱**：在 cell-centered 網格中，誤認為 cell 中心距離邊界為 $\Delta x$。
    *   **修正**：第一個 cell 中心距離左邊界為 $\Delta x/2$。導熱通量係數為 $2k/\Delta x$，矩陣係數為 $2k/\Delta x^2$。
3.  **顯式熱容的能量誤差**：
    *   **陷阱**：預期能量守恆到機器精度。
    *   **修正**：顯式 $C_{eff}$ 僅在 $\Delta T$ 極小時近似焓差。對於跨過相變區的步長，能量誤差為 $O(\Delta t)$。若要高精度守恆，需使用基於焓的隱式求解或修正器。
4.  **與溶氧模型混淆**：
    *   **陷阱**：將溶氧閾值當作相變溫度。
    *   **修正**：溶氧沒有潛熱 $L$，沒有序參量 $\phi$ 的演化方程（除非引入化學反應模型）。其擴散係數 $D_{O2}$ 是連續函數。

## 養殖與相場案例

### 1. 合成冰融化模型
在養殖池模擬中，若考慮池底積冰在春季融化的過程，可使用本章方法。
*   **模型**：1D 垂直熱傳導，底部邊界為冰面，頂部為水體。
*   **參數**：冰的潛熱遠大於比熱。
*   **應用**：預測冰層完全融化的時間。
*   **注意**：此處是物理相變，與溶氧濃度無關。不提供現場操作建議，僅作為合成示例。

### 2. 相場與熱相變的區別
*   **Cahn–Hilliard（第28章）**：描述合金或聚合物中的相分離，由自由能最小化驅動，界面寬度由梯度能量係數決定，不直接涉及溫度場（除非耦合）。
*   **Stefan/焓法（本章）**：由熱傳導驅動，界面寬度由人工參數 $\Delta T_m$ 決定，直接耦合溫度場。
*   **關鍵區別**：相分離中，總質量（序參量積分）守恆；熱相變中，總質量守恆，但相態分佈隨熱流變化。溶氧閾值既非相變也非相分離，僅為管理指標。

## 習題

1.  **手算**：
    1D 域，$N=2$ cells，$\Delta x=1$ m。$T_0^0 = 5$ K, $T_1^0 = 5$ K。
    $T_{left}=10$ K。$T_{melt}=5$ K, $\Delta T_m=2$ K。
    $c_p=1, \rho=1, L=10, k=1, \Delta t=1$ s。
    計算第一步後 $T_0^{1}$ 的值（顯式潛熱，假設右邊界零通量）。

2.  **程式**：
    修改程式，加入能量守恆檢查。計算總輸入熱能與內部能量增加的差異。
    驗證當 $\Delta t \to 0$ 時，能量誤差是否趨近於 0。

3.  **反例**：
    設定 $L=0$，驗證程式退化為標準熱擴散。
    比較解析解 $T(x,t) = T_{init} + (T_{left}-T_{init}) \operatorname{erfc}\left(\frac{x}{2\sqrt{\alpha t}}\right)$（半無限域近似）。

4.  **整合**：
    比較 $\Delta T_m=0.5$ K 和 $\Delta T_m=5$ K 對界面平滑度和計算穩定性的影響。
    解釋為何較小的 $\Delta T_m$ 可能導致時間步長限制更嚴。

## 習題解答

1.  **手算解答**：
    $\Delta x=1$ m。$\frac{k}{\Delta x^2} = 1$。$\frac{2k}{\Delta x^2} = 2$。
    $C_{eff}^0 = 1 + 10(0.5) = 6$。
    $\frac{C_{eff}}{\Delta t} = 6$。
    
    Cell 0 方程：
    $(6 + 1 + 2) T_0^1 - 1 T_1^1 = 6(5) + 2(10)$
    $9 T_0^1 - T_1^1 = 50$
    
    Cell 1 方程（右邊界零通量）：
    $-1 T_0^1 + (6 + 1) T_1^1 = 6(5)$
    $-T_0^1 + 7 T_1^1 = 30$
    
    解聯立方程：
    由第二式 $T_0^1 = 7 T_1^1 - 30$。
    代入第一式：$9(7 T_1^1 - 30) - T_1^1 = 50$
    $63 T_1^1 - 270 - T_1^1 = 50$
    $62 T_1^1 = 320 \implies T_1^1 = 320/62 \approx 5.161$ K。
    $T_0^1 = 7(5.161) - 30 = 36.127 - 30 = 6.127$ K。

2.  **程式解答**：
    在迴圈外累積 `heat_in`，在迴圈內計算 `E_curr`。
    `error = abs(heat_in - (E_curr - E_init))`。
    預期 `error` 應隨 $\Delta t$ 細化而下降，但不一定接近機器精度（由於顯式熱容近似）。

3.  **反例解答**：
    當 $L=0$，$\phi$ 恆為 0，$C_{eff}=\rho c_p$。
    數值解應收斂於解析解。注意邊界條件需匹配（半無限域 vs 有限域）。

4.  **整合解答**：
    $\Delta T_m$ 小：$C_{eff}$ 大，剛性強，需更小 $\Delta t$ 以滿足穩定性/精度。界面更銳利。
    $\Delta T_m$ 大：$C_{eff}$ 小，剛性弱，計算快，但物理上界面較模糊。
    選擇依據於所需的界面解析度與計算成本平衡。

## 本章小結

本章介紹了固液相變的焓法數值實現。重點在於通過引入有效體積熱容 $C_{eff}$ 來處理潛熱項，從而將非連續的 Stefan 問題轉化為連續的熱擴散問題。通過 1D cell-centered 有限體積實作，我們展示了如何進行熱收支檢查，並強調了與溶氧傳輸及 Cahn–Hilliard 相分離的區別。讀者應理解 $\Delta T_m$ 參數對數值穩定性的影響，並能根據物理需求選擇合適的相變溫度區間。

## 參考來源

1.  [F1] FiPy有限體積離散與邊界. NIST FiPy Documentation.
2.  [F7] FiPy簡單相場與固液相變示範. NIST FiPy Documentation.
3.  Crank, J. (1984). *The Mathematics of Diffusion*. Oxford University Press.
4.  本卷第25章（熱力學橋接）與第28章（Cahn–Hilliard）。