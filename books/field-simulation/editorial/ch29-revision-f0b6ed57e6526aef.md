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

為避免顯式追蹤界面，我們引入序參量 $\phi$（液相分數）。在純固體區 $\phi=0$，純液體區 $\phi=1$。在實際物理中，相變發生在一個狹窄的溫度區間 $[T_{solidus}, T_{liquidus}]$ 內（糊狀區）。在數值模型中，我們將這一陡峭的過渡平滑化到一個人為定義的溫度區間 $\Delta T_m$（即「Stefan 橋接」的寬度）。

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

在 1D 有限體積法中，對中心位於 $x_i$ 的控制體（cell），體積 $V_i = A \Delta x$（$A$ 為截面積，設 $A=1$ m² 以便於計算）。
對時間採用後向 Euler 隱式離散：
$$ \frac{C_{eff,i}^n (T_i^{n+1} - T_i^n)}{\Delta t} \Delta x = \left[ k \frac{T_{i-1/2}^{n+1} - T_i^{n+1}}{\Delta x} - k \frac{T_{i+1/2}^{n+1} - T_i^{n+1}}{\Delta x} \right] $$
其中 $C_{eff,i}^n$ 取自時間步 $n$ 的值（顯式熱容，隱式擴散）。注意：此處 $C_{eff}$ 已包含密度 $\rho$，因此左側不需再乘 $\rho$。

### 2. 相態分數函數 $\phi(T)$

為了數值穩定，使用線性平滑函數定義 $\phi(T)$。設相變溫度區間為 $[T_{melt} - \Delta T_m/2, T_{melt} + \Delta T_m/2]$：
$$ \phi(T) = \begin{cases} 0 & T \le T_{melt} - \frac{\Delta T_m}{2} \\ \frac{T - (T_{melt} - \Delta T_m/2)}{\Delta T_m} & T_{melt} - \frac{\Delta T_m}{2} < T < T_{melt} + \frac{\Delta T_m}{2} \\ 1 & T \ge T_{melt} + \frac{\Delta T_m}{2} \end{cases} $$
在此區間內，$\frac{d\phi}{dT} = \frac{1}{\Delta T_m}$。
**重要區分**：$\Delta T_m$ 是溫度區間（單位 K），不是空間厚度（單位 m）。對應的空間糊狀區厚度 $l_{mushy} \approx \Delta T_m / |\nabla T|$。

### 3. Stefan 條件的隱式化

傳統 Stefan 條件在界面 $x=s(t)$ 處為：
$$ \rho L \frac{ds}{dt} = k_s \left. \frac{\partial T}{\partial x} \right|_{solid} - k_l \left. \frac{\partial T}{\partial x} \right|_{liquid} $$
左側為單位面積的質量流率乘以潛熱（W/m²），右側為兩側熱流差（W/m²）。
在糊狀區方法中，我們不顯式計算 $ds/dt$，而是通過求解上述耦合方程，讓 $\phi$ 場自然演化，從而隱式確定界面位置。

## 逐步手算例題

### 例 1：1D 半無限固體的融化（單步計算）

**設定**：
*   域：$x \in [0, 1]$ m，cell-centered 網格，$N=10$ 個 cell，$\Delta x = 0.1$ m。
*   初態：所有 cell $T = 273.0$ K（低於 $T_{melt}=273.15$ K），$\phi=0$。
*   邊界：$x=0$ 處 $T=283.15$ K ($10^\circ$C)，$x=1$ 處零通量（$\partial T/\partial x = 0$）。
*   參數：$\rho = 1000$ kg/m³, $c_p = 4200$ J/(kg·K), $k = 0.6$ W/(m·K), $L = 334000$ J/kg。
*   $\Delta T_m = 1.0$ K（相變區間 272.65 K 至 273.65 K）。
*   $\Delta t = 10$ s。

**計算第一個 cell ($i=0$) 的更新**：
1.  **初始狀態**：$T_0^0 = 273.0$ K。
    由於 $T_0^0 < 273.15 - 0.5 = 272.65$ K？不，$273.0 > 272.65$。
    仔細檢查：$T_{melt}=273.15$，$\Delta T_m/2 = 0.5$。
    下界 $272.65$，上界 $273.65$。
    $T_0^0 = 273.0$ K 位於糊狀區內。
    $\phi_0^0 = \frac{273.0 - 272.65}{1.0} = 0.35$。
    $C_{eff,0}^0 = \rho c_p + \rho L \frac{1}{\Delta T_m} = 1000 \times 4200 + 1000 \times 334000 \times 1 = 4.2 \times 10^6 + 3.34 \times 10^8 = 3.382 \times 10^8$ J/(m³·K)。

2.  **熱流計算**（隱式，使用 $n+1$ 時刻的 $T$）：
    西面（$i=-1/2$，即 $x=0$ 邊界）：$T_{-1/2}^{n+1} = 283.15$ K。
    東面（$i=1/2$，即 $x=0.1$ m，相鄰 cell 1 的中心）：
    假設相鄰 cell 1 初始溫度也為 $273.0$ K，則其 $C_{eff,1}^0$ 相同。
    由於對稱性假設（或假設 cell 1 遠未受影響，$T_1^{n+1} \approx T_1^0 = 273.0$ K 以簡化手算），
    $q_{east} = -k \frac{T_1^{n+1} - T_0^{n+1}}{\Delta x} \approx -0.6 \frac{273.0 - T_0^{n+1}}{0.1}$。
    $q_{west} = k \frac{T_{-1/2}^{n+1} - T_0^{n+1}}{\Delta x} = 0.6 \frac{283.15 - T_0^{n+1}}{0.1} = 6(283.15 - T_0^{n+1})$。

3.  **離散方程**：
    $$ C_{eff,0}^0 \frac{T_0^{n+1} - T_0^0}{\Delta t} \Delta x = (q_{west} - q_{east}) \times 1 \text{ (A=1)} $$
    注意：FVM 平衡式為 $\frac{\Delta H}{\Delta t} V = \text{Net Flux}$。
    $\text{Net Flux} = q_{west} + (-q_{east\_out}) = q_{west} + k \frac{T_1 - T_0}{\Delta x}$。
    代入數值：
    左側：$3.382 \times 10^8 \times \frac{T_0^{n+1} - 273.0}{10} \times 0.1 = 3.382 \times 10^6 (T_0^{n+1} - 273.0)$。
    右側：$6(283.15 - T_0^{n+1}) + 6(273.0 - T_0^{n+1}) = 6(556.15 - 2T_0^{n+1})$。
    
    方程：
    $3.382 \times 10^6 (T_0^{n+1} - 273.0) = 6(556.15 - 2T_0^{n+1})$
    $3.382 \times 10^6 T_0^{n+1} - 9.23286 \times 10^8 = 3336.9 - 12 T_0^{n+1}$
    $(3.382 \times 10^6 + 12) T_0^{n+1} = 9.23286 \times 10^8 + 3336.9$
    $T_0^{n+1} \approx \frac{9.23289 \times 10^8}{3.382012 \times 10^6} \approx 272.97$ K。
    
    **觀察**：溫度從 273.0 K 下降至 272.97 K？這看起來違反物理直覺（邊界加熱應使溫度上升）。
    **錯誤檢查**：
    $T_0^0 = 273.0$。邊界 $283.15$。熱流應流入。
    $q_{west} = 6(283.15 - T)$. 若 $T \approx 273$，$q_{west} \approx 60$ W/m²。
    $q_{east} = 6(273 - T) \approx 0$。
    淨熱流 $\approx 60$ W/m²。
    能量增加 $\approx 60 \times 10 \times 0.1 = 60$ J/m²。
    體積熱容 $\approx 3.38 \times 10^8 \times 0.1 = 3.38 \times 10^7$ J/(m²·K)。
    $\Delta T \approx 60 / 3.38 \times 10^7 \approx 1.7 \times 10^{-6}$ K。
    
    重算方程右側：
    $q_{west} = \frac{k}{\Delta x}(T_{bnd} - T_0) = 6(283.15 - T_0)$
    $-q_{east} = \frac{k}{\Delta x}(T_1 - T_0) = 6(273.0 - T_0)$ （假設 $T_1$ 不變）
    右側總和 $= 6(283.15 - T_0 + 273.0 - T_0) = 6(556.15 - 2T_0)$。
    若 $T_0 = 273$，右側 $= 6(556.15 - 546) = 6(10.15) = 60.9$。
    左側係數 $3.382 \times 10^6$。
    $3.382 \times 10^6 \Delta T = 60.9 \implies \Delta T \approx 1.8 \times 10^{-5}$ K。
    $T_0^{n+1} \approx 273.000018$ K。
    
    **結論**：由於潛熱極大，溫度幾乎不變，能量主要用於增加 $\phi$（熔化）。
    $\Delta \phi \approx \frac{\Delta T}{\Delta T_m} \approx 1.8 \times 10^{-5}$。

### 例 2：顯式 vs 隱式潛熱處理

*   **顯式潛熱（$C_{eff}^n$）**：若 $T^n$ 在固體區，$C_{eff} = \rho c_p$。熱流快速加熱節點，$T^{n+1}$ 可能跳過 $T_{melt}$ 進入液體區。這可能導致「過熱」（Overheating），即溫度瞬間高於相變點，物理上不準確，且可能違反能量守恆（因為跳過潛熱吸收）。
*   **隱式潛熱（$C_{eff}^{n+1}$ 或迭代）**：需要求解非線性系統。在 $T^{n+1}$ 接近 $T_{melt}$ 時，$C_{eff}$ 變大，自動限制溫度跳變，模擬界面滯留。這更符合物理真實。本章實作採用顯式熱容以簡化線性求解，但會導致時間步長限制較嚴，或需要在跨過相變區時使用更小的 $\Delta t$。

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
                    dx=None, atol=1e-10, rtol=1e-10):
    """
    Solve 1D Stefan problem using Enthalpy Method (Explicit Ceff).
    Cell-centered FVM.
    """
    if dx is None:
        dx = Lx / N
    else:
        if not np.isclose(N * dx, Lx):
            raise ValueError("dx * N must equal Lx")
            
    # Input Validation
    if any(val <= 0 for val in [rho, cp, k, dt, dT_m, L_latent]):
        raise ValueError("Physical parameters and dt must be positive")
    if N < 2:
        raise ValueError("N must be at least 2")
        
    # Initial Conditions
    T = np.full(N, T_init)
    # Calculate initial phi
    phi = np.zeros(N)
    for i in range(N):
        if T[i] < T_melt - dT_m/2:
            phi[i] = 0.0
        elif T[i] > T_melt + dT_m/2:
            phi[i] = 1.0
        else:
            phi[i] = (T[i] - (T_melt - dT_m/2)) / dT_m

    def get_ceff(T_val):
        """Calculate effective volumetric heat capacity."""
        base = rho * cp
        if T_val < T_melt - dT_m/2 or T_val > T_melt + dT_m/2:
            return base
        else:
            return base + rho * L_latent / dT_m

    def get_phi(T_val):
        if T_val < T_melt - dT_m/2:
            return 0.0
        elif T_val > T_melt + dT_m/2:
            return 1.0
        else:
            return (T_val - (T_melt - dT_m/2)) / dT_m

    # Pre-allocate storage for history
    T_hist = [T.copy()]
    phi_hist = [phi.copy()]
    energy_hist = [np.sum((rho*cp*(T - (T_melt-dT_m/2)) + rho*L_latent*phi)) * dx]
    
    # Accumulated heat input from left boundary
    heat_in_cumulative = 0.0
    
    for step in range(n_steps):
        # 1. Compute Ceff for current time step (Explicit)
        Ceff = np.array([get_ceff(T[i]) for i in range(N)])
        
        # 2. Assemble Linear System A T_new = b
        # Equation for cell i:
        # Ceff_i * (T_i_new - T_i_old) / dt * dx = k*(T_{i-1}_new - T_i_new)/dx + k*(T_{i+1}_new - T_i_new)/dx
        # Rearranged:
        # -k/dx^2 T_{i-1}_new + (Ceff_i/(dt*dx) + 2k/dx^2) T_i_new - k/dx^2 T_{i+1}_new = Ceff_i/(dt*dx) T_i_old
        
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        for i in range(N):
            diag_val = Ceff[i] / (dt * dx) + 2.0 * k / (dx**2)
            A[i, i] = diag_val
            b[i] = (Ceff[i] / (dt * dx)) * T[i]
            
            # West face (i-1)
            if i > 0:
                A[i, i-1] = -k / (dx**2)
            else:
                # Left Boundary (Dirichlet T_left)
                # Flux = k * (T_left - T_i_new) / dx
                # Contribution to LHS: Move -k/dx^2 * T_left to RHS? 
                # Original term: k/dx^2 * T_left. 
                # In matrix form: The term involving T_left is constant.
                b[i] += (k / dx**2) * T_left
                
            # East face (i+1)
            if i < N - 1:
                A[i, i+1] = -k / (dx**2)
            else:
                # Right Boundary (Zero Flux / Neumann 0)
                # No unknown T_{N} to add. The term for east face is 0.
                # The diagonal term 2*k/dx^2 assumed 2 neighbors. 
                # For last cell, only West neighbor exists.
                # Correction: The diagonal should be Ceff/(dt*dx) + k/dx^2 (only one face).
                # We added 2*k/dx^2 above. Subtract one.
                A[i, i] -= k / (dx**2)
                # Alternatively, construct A more carefully. 
                # Let's stick to the general assembly and correct the boundary row.
                # For i=N-1, the term k*(T_N - T_{N-1}) is 0. 
                # So we don't add A[i, N]. 
                # The diagonal 2*k/dx^2 is too high by k/dx^2.
                # So A[i, i] -= k/dx^2 is correct.
        
        # Solve
        try:
            T_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"Matrix singular at step {step}")
        
        # 3. Update Phi
        phi_new = np.array([get_phi(T_new[i]) for i in range(N)])
        
        # 4. Calculate Heat Input for Balance Check
        # Flux at left boundary: q_left = k * (T_left - T_new[0]) / dx
        # Note: We use T_new for the current step's flux.
        q_left = k * (T_left - T_new[0]) / dx
        heat_in_step = q_left * dt * 1.0 # Area = 1
        heat_in_cumulative += heat_in_step
        
        # 5. Update State
        T = T_new
        phi = phi_new
        
        # 6. Store History
        T_hist.append(T.copy())
        phi_hist.append(phi.copy())
        
        # Calculate Total Enthalpy (Energy)
        # H = rho * cp * (T - T_ref) + rho * L * phi
        # Let T_ref = T_melt - dT_m/2
        T_ref = T_melt - dT_m/2
        H_vec = rho * cp * (T - T_ref) + rho * L_latent * phi
        E_curr = np.sum(H_vec) * dx
        energy_hist.append(E_curr)
        
        # Energy Balance Residual
        # R = E_curr - E_init - Heat_In_Cumulative
        E_init = energy_hist[0]
        R_balance = E_curr - E_init - heat_in_cumulative
        
        if step % 10 == 0:
            print(f"Step {step}: T_min={np.min(T):.4f}, T_max={np.max(T):.4f}, "
                  f"Phi_avg={np.mean(phi):.4f}, Energy_Resid={R_balance:.2e}")
            
        # Check for non-physical oscillations or divergence
        if not np.all(np.isfinite(T_new)):
            raise ValueError(f"Non-finite values encountered at step {step}")

    return T_hist, phi_hist, energy_hist, heat_in_cumulative

# Example Usage
if __name__ == "__main__":
    N = 100
    Lx = 1.0
    T_init = 273.0
    T_left = 283.15
    dt = 1.0
    n_steps = 100
    
    rho = 1000.0
    cp = 4200.0
    k = 0.6
    L_latent = 334000.0
    T_melt = 273.15
    dT_m = 1.0 # K
    
    T_hist, phi_hist, E_hist, Q_in = solve_stefan_1d(
        N, Lx, T_init, T_left, dt, n_steps,
        rho, cp, k, L_latent, T_melt, dT_m
    )
    
    print(f"Final Avg Phi: {np.mean(phi_hist[-1]):.4f}")
    print(f"Final Energy: {E_hist[-1]:.2e}")
    print(f"Total Heat Input: {Q_in:.2e}")
    print(f"Balance Error: {abs(E_hist[-1] - E_hist[0] - Q_in):.2e}")
```

## 測試與預期結果

### 1. 正常測試
*   **恆定溫度邊界**：若 $T_{left} > T_{melt}$，界面應向 $x>0$ 方向移動。$\phi$ 的積分（總液相體積）應隨時間增加。
*   **能量守恆**：
    定義總焓 $E = \sum_i V_i [\rho c_p (T_i - T_{ref}) + \rho L \phi_i]$。
    熱收支殘差 $R_E = E^{n+1} - E^0 - \sum_{k=0}^n Q_{in}^k$。
    由於使用顯式 $C_{eff}^n$，當溫度跨過相變區時，$C_{eff}^n \Delta T \neq H(T^{n+1}) - H(T^n)$，因此 $R_E$ 不為零，但應隨 $\Delta t$ 細化而收斂至 0。
*   **網格細化**：減小 $\Delta x$ 和 $\Delta t$，界面位置 $s(t)$（定義為 $\phi=0.5$ 的位置）應收斂。

### 2. 邊界與故障測試
*   **無潛熱（純熱擴散）**：設定 $L_{latent} = 0$。結果應與標準熱擴散解析解一致。
*   **過冷（Undercooling）**：若 $T_{left} < T_{melt} - \Delta T_m/2$，物體不應融化，$\phi$ 保持 0。
*   **輸入驗證**：
    *   $N < 2$：應拋出 `ValueError`。
    *   $dt \le 0$：應拋出 `ValueError`。
    *   非有限值輸入：應拋出 `ValueError`。
*   **數值抖動**：檢查 $T$ 在 $T_{melt}$ 附近是否有非物理振盪。若 $\Delta T_m$ 太小，可能需要更細的網格或隱式熱容。

## 除錯與常見陷阱

1.  **單位量綱不一致**：
    *   **陷阱**：混淆比質量熱容 $c_p$ (J/kg·K) 與體積熱容 $\rho c_p$ (J/m³·K)。
    *   **修正**：確保焓 $H$ 的單位為 J/m³。$H = \rho c_p (T-T_{ref}) + \rho L \phi$。
2.  **邊界處理錯誤**：
    *   **陷阱**：在最後一個 cell 使用與內部 cell 相同的對角線係數（假設兩個鄰居），但右邊界是零通量（無東側鄰居）。
    *   **修正**：對最後一行，對角線係數應減去 $k/\Delta x^2$，因為只有西側貢獻擴散項。
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
    $c_p=1, \rho=1, L=10, k=1$。
    計算第一步後 $T_0^{1}$ 的值（顯式潛熱，假設右邊界零通量）。
    *提示：$T_0^0=5$ 在糊狀區中心，$C_{eff} = 1 + 10(0.5) = 6$。*

2.  **程式**：
    修改程式，加入能量守恆檢查。計算總輸入熱能與內部能量增加的差異。
    驗證當 $\Delta t \to 0$ 時，能量誤差是否趨近於 0。

3.  **反例**：
    設定 $L=0$，驗證程式退化為標準熱擴散。
    比較解析解 $T(x,t) = T_{init} + (T_{left}-T_{init}) \text{erf}\left(\frac{x}{2\sqrt{\alpha t}}\right)$（半無限域近似）。

4.  **整合**：
    比較 $\Delta T_m=0.5$ K 和 $\Delta T_m=5$ K 對界面平滑度和計算穩定性的影響。
    解釋為何較小的 $\Delta T_m$ 可能導致時間步長限制更嚴。

## 習題解答

1.  **手算解答**：
    $T_0^0 = 5$ K。$T_{melt}=5$。$\Delta T_m=2$。
    $C_{eff,0} = \rho c_p + \rho L / \Delta T_m = 1 + 10/2 = 6$ J/(m³·K)。
    $C_{eff,1} = 1 + 10/2 = 6$ J/(m³·K)。
    方程 0:
    $6 \frac{T_0^1 - 5}{1} \cdot 1 = 1 \frac{10 - T_0^1}{1} + 1 \frac{T_1^1 - T_0^1}{1}$
    $6(T_0^1 - 5) = 10 - T_0^1 + T_1^1 - T_0^1 = 10 + T_1^1 - 2T_0^1$
    $6T_0^1 - 30 = 10 + T_1^1 - 2T_0^1$
    $8T_0^1 - T_1^1 = 40$  (Eq 1)
    
    方程 1 (Right Boundary Zero Flux):
    $6 \frac{T_1^1 - 5}{1} \cdot 1 = 1 \frac{T_0^1 - T_1^1}{1} + 0$
    $6(T_1^1 - 5) = T_0^1 - T_1^1$
    $6T_1^1 - 30 = T_0^1 - T_1^1$
    $7T_1^1 - T_0^1 = 30$  (Eq 2)
    
    From Eq 2: $T_0^1 = 7T_1^1 - 30$.
    Substitute into Eq 1:
    $8(7T_1^1 - 30) - T_1^1 = 40$
    $56T_1^1 - 240 - T_1^1 = 40$
    $55T_1^1 = 280 \implies T_1^1 = 280/55 \approx 5.091$ K.
    $T_0^1 = 7(5.091) - 30 \approx 35.637 - 30 = 5.637$ K.

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