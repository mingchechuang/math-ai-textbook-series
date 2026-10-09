# 第29章 固液相變、潛熱與Stefan橋接

## 學習目標與先備知識

本章探討固體與液體相變的數值模擬方法，重點在於處理潛熱（latent heat）釋放或吸收對熱傳導的影響。與第四章（熱力學橋接）及第五章（Cahn–Hilliard）不同，本章聚焦於經典的熱傳導-相變耦合問題，即 Stefan 問題的擴散形式。讀者應已掌握前章的有限體積法（FVM）基礎、一維熱擴散方程的離散化，以及非線性方程的線性化技巧。

**核心目標**：
1.  **理解焓法（Enthalpy Method）**：如何將非連續的潛熱項轉化為連續的焓溫度關係，從而允許使用標準的熱擴散求解器。
2.  **Stefan 界面條件的隱式處理**：理解傳統 Stefan 問題（顯式界面追蹤）與相場（Phase-field）方法的區別，特別是「Stefan 橋接」中如何通過寬化界面來避免顯式移動邊界。
3.  **數值穩定性與熱收支**：分析潛熱項對時間步長的限制，並建立熱收支檢驗（heat balance check）以確保能量守恆。
4.  **區分類似現象**：明確區分固液相變（質量守恆、能量守恆、界面移動）與溶氧閾值管理（純傳輸、無相變）以及 Cahn–Hilliard 相分離（自由能驅動的扩散）。

**先備知識**：
*   熱傳導方程：$ c_p \rho \frac{\partial T}{\partial t} = k \nabla^2 T + Q $。
*   有限體積法（FVM）在 1D 域上的組裝。
*   非線性單調函數的線性化（Picard 或 Newton 方法）。

## 問題與直覺

### 顯熱與潛熱的挑戰

在常規熱擴散問題中，溫度 $T$ 是連續變化的狀態變量，比熱容 $c_p$ 通常為常數。然而，在固液相變過程中（如冰融化為水），物質在相變溫度 $T_{melt}$ 附近吸收或釋放潛熱 $L$（單位：J/kg），而溫度保持不變。這導致了兩個數值挑戰：

1.  **非連續導數**：焓 $H$（單位體積內能）對溫度 $T$ 的導數 $\frac{dH}{dT} = c_p \rho + \rho L \delta(T - T_{melt})$ 在相變點出現狄拉克 delta 函數（Dirac delta），意味著有效比熱容在該點趨近無窮大。
2.  **界面運動**：傳統 Stefan 問題要求明確追蹤固液界面位置 $s(t)$，並滿足界面連續條件（溫度相等）與 Stefan 條件（能量守恆）。顯式追蹤界面在界面發生合併或分裂時非常困難。

**直覺解法：焓法與相場橋接**

為避免顯式追蹤界面，我們引入序參量 $\phi$（相態變量）。在純固體區 $\phi=0$，純液體區 $\phi=1$。在寬化的界面區（Stefan 橋接），$\phi$ 平滑過渡。總內能由顯熱部分和潛熱部分組成：
$$ H(T, \phi) = \int_{T_{ref}}^{T} c_p(\phi) dT' + \rho L \phi $$
其中 $\phi$ 代表已熔化的體積分數。通過聯立求解溫度場 $T$ 和相態場 $\phi$，我們可以隱式地確定界面位置，而不需要顯式的幾何追蹤。

**與溶氧閾值的區別**：
養殖池中的溶氧濃度跨過「管理閾值」（如 5 mg/L）並不涉及任何物質的相態改變或潛熱釋放。這是一個純傳輸-反應問題，其物性參數（擴散係數、反應速率）不隨閾值發生斷裂或跳變。因此，溶氧模型不需要焓法，也不需要相變項。

## 數學與物理推導

### 1. 連通性方程與焓定義

考慮 1D 穩態密度 $\rho$ 的固液相變。能量守恆方程寫為：
$$ \frac{\partial H}{\partial t} = \nabla \cdot (k \nabla T) $$
其中 $H$ 是單位體積焓。為了將此方程轉化為可解的形式，我們定義 $H$ 為 $T$ 和 $\phi$ 的函數：
$$ H(T, \phi) = \int_{T_0}^{T} c_p(\xi, \phi) d\xi + \rho L \phi $$
在簡化模型中，假設固體和液體的比熱容相同（$c_p$ 為常數），且潛熱集中在 $T_{melt}$ 附近。則：
$$ H = c_p (T - T_0) + \rho L \phi $$

### 2. Stefan 條件的隱式化

傳統 Stefan 條件在界面 $x=s(t)$ 處為：
$$ \rho L \frac{ds}{dt} = k \left( \frac{\partial T}{\partial x}\Big|_{solid} - \frac{\partial T}{\partial x}\Big|_{liquid} \right) $$
在相場方法中，界面被寬化為 $\epsilon$ 厚度的區域。序參量 $\phi$ 滿足 Cahn–Hilliard 或 Allen–Cahn 型方程，但在熱相變中，我們通常使用更簡單的「局部平衡」假設：
$$ \phi(T) = \begin{cases} 0 & T < T_{melt} - \epsilon/2 \\ \frac{T - (T_{melt} - \epsilon/2)}{\epsilon} & T_{melt} - \epsilon/2 \le T \le T_{melt} + \epsilon/2 \\ 1 & T > T_{melt} + \epsilon/2 \end{cases} $$
這稱為 **Stefan 橋接（Stefan Bridge）**，它將陡峭的界面平滑化。在此區域內，$\frac{\partial \phi}{\partial T} = \frac{1}{\epsilon}$。

### 3. 離散化方程

將時間導數項展開：
$$ \frac{\partial H}{\partial t} = c_p \frac{\partial T}{\partial t} + \rho L \frac{\partial \phi}{\partial t} = c_p \frac{\partial T}{\partial t} + \rho L \frac{\partial \phi}{\partial T} \frac{\partial T}{\partial t} $$
因此，能量方程變為：
$$ \left( c_p + \rho L \frac{\partial \phi}{\partial T} \right) \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) $$
定義 **有效比熱容** $C_{eff}(T) = c_p + \rho L \frac{\partial \phi}{\partial T}$。
在 1D 有限體積法中，對節點 $i$：
$$ \frac{\rho V_i C_{eff}^n (T_i^{n+1} - T_i^n)}{\Delta t} = A_{i-1/2} k_{i-1/2} \frac{T_{i-1}^{n+1} - T_i^{n+1}}{\Delta x} - A_{i+1/2} k_{i+1/2} \frac{T_{i+1}^{n+1} - T_i^{n+1}}{\Delta x} $$
這裡採用後向 Euler（Backward Euler）時間離散以處理剛性。注意 $C_{eff}^n$ 取自時間步 $n$ 的值（顯式潛熱處理）或需要在時間步內線性化（隱式潛熱處理）。為簡化，我們先使用顯式潛熱項（Explicit Latent Heat），即 $C_{eff}$ 使用舊值 $C_{eff}^n$。

## 逐步手算例題

### 例 1：1D 半無限固體的融化

**設定**：
*   域：$x \in [0, 1]$ m。
*   初態：$T(x,0) = 273.15$ K ($0^\circ$C)，$\phi=0$。
*   邊界：$x=0$ 處 $T=283.15$ K ($10^\circ$C)，$x=1$ 處 $\frac{\partial T}{\partial x}=0$。
*   參數：$\rho = 1000$ kg/m³, $c_p = 4200$ J/(kg·K), $k = 0.6$ W/(m·K), $L = 334000$ J/kg (冰的熔化潛熱)。
*   $T_{melt} = 273.15$ K, $\epsilon = 0.5$ K。

**手算步驟（單時間步）**：
考慮靠近邊界的單個網格節點 $i=1$（$\Delta x = 0.1$ m）。
假設當前 $T_1^0 = 273.15$ K，$T_0 = 283.15$ K（邊界固定）。
$T_2^0 = 273.15$ K。

1.  **計算有效比熱 $C_{eff}$**：
    由於 $T_1 = T_{melt}$，$\phi$ 在過渡區中心。
    $\frac{\partial \phi}{\partial T} = \frac{1}{\epsilon} = 2$ K$^{-1}$。
    $C_{eff} = c_p + \rho L \frac{1}{\epsilon} = 4200 + 1000 \times 334000 \times 2 = 4200 + 668,000,000 \approx 6.68 \times 10^8$ J/(m³·K)。
    *注意：此值極大，表明時間步長必須極小或使用隱式求解。*

2.  **熱流計算**：
    西面熱流 $q_{W} = k \frac{T_0 - T_1}{\Delta x} = 0.6 \frac{283.15 - 273.15}{0.1} = 60$ W/m²。
    東面熱流 $q_{E} = k \frac{T_1 - T_2}{\Delta x} = 0$ W/m²。
    淨熱流 $q_{net} = q_W - q_E = 60$ W/m²。

3.  **溫度更新**：
    體積 $V = \Delta x \times 1 \times 1 = 0.1$ m³。
    能量增加 $\Delta E = q_{net} \times A_{face} \times \Delta t = 60 \times 1 \times \Delta t$。
    溫度變化 $\Delta T = \frac{\Delta E}{\rho V C_{eff}} = \frac{60 \Delta t}{1000 \times 0.1 \times 6.68 \times 10^8} = \frac{60 \Delta t}{6.68 \times 10^{10}} \approx 8.98 \times 10^{-10} \Delta t$。
    
    若 $\Delta t = 10$ s：
    $\Delta T \approx 8.98 \times 10^{-9}$ K。
    
    **觀察**：由於潛熱巨大，溫度幾乎不升，能量主要用於增加 $\phi$（熔化）。
    新的 $\phi_1 \approx \phi_1^0 + \frac{1}{\epsilon} \Delta T$。若 $\phi_1^0 = 0.5$，則 $\phi_1 \approx 0.5$。界面緩慢推進。

### 例 2：顯式 vs 隱式潛熱處理

比較 $C_{eff}$ 取 $T^n$ 或 $T^{n+1}$ 的差異。
*   **顯式**：$C_{eff}^n$。若 $T^n < T_{melt}$，則 $C_{eff} = c_p$。熱流快速加熱節點，$T^{n+1}$ 可能跳過 $T_{melt}$ 進入液體區。這可能導致過熱（Overheating），即溫度瞬間高於相變點，物理上不準確。
*   **隱式**：$C_{eff}^{n+1}$。需要求解非線性系統。在 $T^{n+1}$ 接近 $T_{melt}$ 時，$C_{eff}$ 變大，自動限制溫度跳變，模擬界面滯留。這更符合物理真實。

## 實作與程式

以下提供一個 1D 隱式焓法實現，使用後向 Euler 時間積分和線性化潛熱項。

```python
import numpy as np

def solve_stefan_1d(N, Lx, T0, T_left, dx, dt, n_steps, cp, rho, L_latent, k, T_melt, epsilon):
    """
    Solve 1D Stefan problem using Enthalpy Method with Explicit Latent Heat.
    """
    x = np.linspace(0, Lx, N)
    T = np.zeros(N)
    phi = np.zeros(N)
    
    # Initial Condition
    T[:] = T0
    phi[:] = 0.0
    
    # Helper function to get latent heat fraction derivative
    def dphi_dT(T_val):
        # Smooth step function derivative
        if T_val < T_melt - epsilon/2:
            return 0.0
        elif T_val > T_melt + epsilon/2:
            return 0.0
        else:
            return 1.0 / epsilon
    
    # Effective Heat Capacity
    def effective_heat_capacity(T_val):
        cp_base = cp * rho
        cp_latent = rho * L_latent * dphi_dT(T_val)
        return cp_base + cp_latent
    
    # Pre-compute matrices for Laplacian (Tridiagonal)
    # (1 - D*dt*Ceff/T) T_new = T_old + ...
    # For simplicity, we use explicit latent heat (Ceff from old time)
    # to avoid non-linear solves in this basic example.
    # Matrix A: -a on sub/super diag, 1 + 2a on main diag
    # a = k * dt / (rho * V * Ceff * dx)  (for uniform dx)
    
    histories = {'T': [T.copy()], 'phi': [phi.copy()]}
    
    for step in range(n_steps):
        # Compute Ceff for all nodes based on T (old)
        Ceff = np.array([effective_heat_capacity(T[i]) for i in range(N)])
        
        # Build diagonal elements of the system
        # Equation: Ceff_i * (T_i_new - T_i_old) / dt = k * (T_{i-1}_new - 2 T_i_new + T_{i+1}_new) / dx^2
        # Rearranged for T_new:
        # -k/dx^2 T_{i-1}_new + (Ceff_i/dt + 2k/dx^2) T_i_new - k/dx^2 T_{i+1}_new = Ceff_i/dt T_i_old
        
        main_diag = np.zeros(N)
        off_diag = np.zeros(N-1)
        
        for i in range(N):
            # Note: dx is assumed uniform
            coeff_implicit = k / (dx**2)
            # Heat capacity term
            cp_term = Ceff[i] / dt
            main_diag[i] = cp_term + 2 * coeff_implicit
            
            if i > 0:
                off_diag[i-1] = -coeff_implicit
            if i < N-1:
                off_diag[i] = -coeff_implicit  # Same value for symmetric Laplacian
        
        # Apply Boundary Conditions
        # Left Dirichlet: T_0 = T_left
        # Remove row 0, adjust RHS for row 1
        # New system size N-1 (nodes 1 to N-1)
        
        if N > 1:
            A_sub = np.zeros((N-1, N-1))
            b_vec = np.zeros(N-1)
            
            # Fill submatrix (nodes 1..N-1)
            for i in range(1, N):
                idx = i - 1 # index in submatrix
                # Diagonal
                A_sub[idx, idx] = main_diag[i]
                # Off-diagonals
                if i > 1:
                    A_sub[idx, idx-1] = off_diag[i-1]
                if i < N-1:
                    A_sub[idx, idx+1] = off_diag[i]
                
                # RHS = Ceff[i]/dt * T[i]
                b_vec[idx] = (Ceff[i] / dt) * T[i]
                
                # Boundary contribution from node 0
                if i == 1:
                    # The term -k/dx^2 * T_0_new moves to RHS
                    b_vec[idx] -= off_diag[0] * T_left # off_diag[0] is -k/dx^2, so -(-k/dx^2)*T_left = k/dx^2*T_left
            
            # Solve linear system
            # Using simple Gaussian Elimination or NumPy linalg for small N
            try:
                T_new_sub = np.linalg.solve(A_sub, b_vec)
            except np.linalg.LinAlgError:
                print("Matrix singular at step", step)
                break
            
            # Update T
            T_new = np.zeros(N)
            T_new[0] = T_left
            T_new[1:] = T_new_sub
            
            # Update phi based on new T
            phi_new = np.zeros(N)
            for i in range(N):
                if T_new[i] < T_melt - epsilon/2:
                    phi_new[i] = 0.0
                elif T_new[i] > T_melt + epsilon/2:
                    phi_new[i] = 1.0
                else:
                    phi_new[i] = (T_new[i] - (T_melt - epsilon/2)) / epsilon
            
            T = T_new
            phi = phi_new
        else:
            T[0] = T_left
            phi[0] = 1.0 if T_left > T_melt else 0.0

        histories['T'].append(T.copy())
        histories['phi'].append(phi.copy())
        
    return T, phi, histories

# Example Usage
if __name__ == "__main__":
    N = 50
    Lx = 1.0
    dx = Lx / (N-1)
    T0 = 273.0 # slightly below melt
    T_left = 280.0
    dt = 1.0
    n_steps = 1000
    
    # Parameters
    rho = 1000.0
    cp = 4200.0
    k = 0.6
    L_latent = 334000.0
    T_melt = 273.15
    epsilon = 0.5
    
    T_final, phi_final, hist = solve_stefan_1d(
        N, Lx, T0, T_left, dx, dt, n_steps, 
        cp, rho, L_latent, k, T_melt, epsilon
    )
    
    # Check Energy Balance (Simplified)
    # Total Energy = Integral of H dV
    # H = cp*(T-T0_ref) + rho*L*phi
    T_ref = 273.15
    H_final = cp * (T_final - T_ref) + rho * L_latent * phi_final
    E_final = np.sum(H_final) * dx
    
    print(f"Final Avg Temp: {np.mean(T_final):.2f} K")
    print(f"Final Avg Phi: {np.mean(phi_final):.4f}")
    print(f"Final Total Energy: {E_final:.2e} J")
```

## 測試與預期結果

### 正常測試
1.  **恆定溫度邊界**：若 $T_{left} > T_{melt}$，界面應向 $x>0$ 方向移動。$\phi$ 的積分（總熔質量）應隨時間增加。
2.  **能量守恆**：輸入邊界的熱能應等於內部焓的增加加上累積的潛熱。
    $$ \int_0^t k \frac{\partial T}{\partial x}\Big|_{x=0} A dt = \int_\Omega (c_p(T-T_{ref}) + \rho L \phi) dV - E_{initial} $$
3.  **網格細化**：減小 $\Delta x$ 和 $\Delta t$，界面位置 $s(t)$ 應收斂。

### 邊界與故障測試
1.  **無潛熱（純熱擴散）**：設定 $L_{latent} = 0$。結果應與標準熱擴散解一致。
2.  **過冷（Undercooling）**：若 $T_{melt}$ 設定極高，物體不應融化，$\phi$ 保持 0。
3.  **數值抖動**：檢查 $T$ 在 $T_{melt}$ 附近是否有非物理振盪。若 $\epsilon$ 太小，可能需要更細的網格。
4.  **穩定性**：顯式潛熱處理在 $\Delta t$ 過大時可能導致溫度跳變過大。檢查 $T_{new}$ 是否超出物理範圍。

## 除錯與常見陷阱

1.  **比熱容跳變**：
    *   **陷阱**：在計算熱容時，直接使用 $c_p$ 而不加潛熱項。
    *   **後果**：溫度會瞬間跨越相變點，潛熱未被正確吸收，能量守恆失敗。
    *   **修正**：確保 $C_{eff}$ 包含 $\rho L \frac{\partial \phi}{\partial T}$ 項。

2.  **界面厚度 $\epsilon$ 選擇**：
    *   **陷阱**：$\epsilon$ 選擇過小（如 $10^{-6}$ K），導致 $\frac{\partial \phi}{\partial T}$ 極大，系統剛性增強，顯式方法失穩。
    *   **修正**：$\epsilon$ 應與網格尺度 $\Delta x$ 和期望的界面解析度相關。通常 $\epsilon \approx \Delta x$ 或稍大。

3.  **顯式 vs 隱式潛熱**：
    *   **陷阱**：使用顯式潛熱（$C_{eff}^n$）處理強烈相變。
    *   **後果**：可能出現「過熱」，即溫度在一步內跳過相變區。
    *   **修正**：對於剛性問題，應使用隱式潛熱或混合方法，或使用更小的時間步長。

4.  **與溶氧模型混淆**：
    *   **陷阱**：將溶氧閾值當作相變溫度。
    *   **修正**：溶氧沒有潛熱 $L$，沒有序參量 $\phi$ 的演化方程（除非引入化學反應模型）。其擴散係數 $D_O2$ 是連續函數。

## 養殖與相場案例

### 1. 合成冰融化模型
在養殖池模擬中，若考慮池底積冰在春季融化的過程，可使用本章方法。
*   **模型**：1D 垂直熱傳導，底部邊界為冰面，頂部為水體。
*   **參數**：冰的潛熱遠大於比熱。
*   **應用**：預測冰層完全融化的時間，以決定是否可以開始投餵。
*   **注意**：此處是物理相變，與溶氧濃度無關。

### 2. 相場與熱相變的區別
*   **Cahn–Hilliard（第28章）**：描述合金或聚合物中的相分離，由自由能最小化驅動，界面寬度由梯度能量係數決定，不直接涉及溫度場（除非耦合）。
*   **Stefan/焓法（本章）**：由熱傳導驅動，界面寬度由人工參數 $\epsilon$ 或熱滯後決定，直接耦合溫度場。
*   **關鍵區別**：相分離中，總質量（序參量積分）守恆；熱相變中，總質量守恆，但相態分佈隨熱流變化。

## 習題

1.  **手算**：
    1D 域，$N=2$ 節點，$\Delta x=1$ m。$T_0=0$ K, $T_1=0$ K。$T_{left}=10$ K。
    $c_p=1, \rho=1, L=10, k=1, T_{melt}=5, \epsilon=2$。
    計算第一步後 $T_1$ 的值（顯式潛熱）。

2.  **程式**：
    修改程式，加入能量守恆檢查。計算總輸入熱能與內部能量增加的差異。

3.  **反例**：
    設定 $L=0$，驗證程式退化為標準熱擴散。比較解析解 $T(x,t) = 10 \text{erfc}(\frac{x}{2\sqrt{kt}})$（半無限域）。

4.  **整合**：
    比較 $\epsilon=0.5$ 和 $\epsilon=5$ 對界面平滑度和計算穩定性的影響。

## 習題解答

1.  **手算解答**：
    $C_{eff}^0 = c_p + \rho L \frac{1}{\epsilon} = 1 + 1 \cdot 10 \cdot 0.5 = 6$。
    $a = k / \Delta x^2 = 1$。
    主對角 $D = C_{eff}/\Delta t + 2a$。假設 $\Delta t=1$。
    $D = 6 + 2 = 8$。
    副對角 $O = -a = -1$。
    右邊項 $RHS = C_{eff} T_{old} - O \cdot T_{left} = 6 \cdot 0 - (-1) \cdot 10 = 10$。
    方程：$8 T_1^{new} = 10 \Rightarrow T_1^{new} = 1.25$ K。

2.  **程式解答**：
    在迴圈外累積 `heat_in`，在迴圈內計算 `E_curr`。
    `error = abs(heat_in - (E_curr - E_init))`。
    預期 `error` 應接近 0（在浮點誤差內）。

3.  **反例解答**：
    當 $L=0$，$\phi$ 恆為 0，$C_{eff}=c_p \rho$。
    數值解應收斂於解析解。注意邊界條件需匹配（半無限域 vs 有限域）。

4.  **整合解答**：
    $\epsilon$ 小：界面陡，剛性大，需更小 $\Delta t$。
    $\epsilon$ 大：界面平滑，計算快，但物理上界面較模糊。
    選擇依據於所需的界面解析度與計算成本平衡。

## 本章小結

本章介紹了固液相變的焓法數值實現。重點在於通過引入有效比熱容 $C_{eff}$ 來處理潛熱項，從而將非連續的 Stefan 問題轉化為連續的熱擴散問題。通過 1D 實作與手算例子，我們展示了如何進行熱收支檢查，並強調了與溶氧傳輸及 Cahn–Hilliard 相分離的區別。讀者應理解 $\epsilon$ 參數對數值穩定性的影響，並能根據物理需求選擇合適的界面寬度。

## 參考來源

1.  [F1] FiPy有限體積離散與邊界. (用於理解 FVM 組裝)
2.  [F7] FiPy簡單相場與固液相變示範. (參考相變模型的具體實現細節)
3.  Crank, J. (1984). *The Mathematics of Diffusion*. (Stefan 問題經典推導)
4.  本卷第4章（熱力學橋接）與第28章（Cahn–Hilliard）.