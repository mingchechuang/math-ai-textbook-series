# 第29章 變分法、第一變分與能量泛函

## 學習目標與先備知識

本章從有限維微積分的極值問題延伸至無限維函數空間，建立變分法的嚴格的數學框架。核心目標是理解能量泛函 $J: \mathcal{X} \to \mathbb{R}$ 的臨界點（駐點）性質，並區分 Gateaux 導數、Fréchet 導數與 Hessian。我們將推導 Euler–Lagrange 方程的充分條件，解釋自然邊界條件的幾何意義，並透過離散化模型驗證連續與離散梯度的對應關係。

讀者需具備以下先備知識：
1. **Fréchet 導數**：映射 $F: U \subset X \to Y$ 在 $x$ 點可微，若存在有界線性算子 $DF(x)$ 使得 $F(x+h) = F(x) + DF(x)h + r(h)$，且 $\|r(h)\|/\|h\| \to 0$ 當 $h \to 0$。
2. **線性泛函**：標量函數 $f: U \subset \mathbb{R}^n \to \mathbb{R}$ 的導數 $Df(x)$ 是定義在 $\mathbb{R}^n$ 上的線性泛函，其代表元為梯度 $\nabla f(x) \in \mathbb{R}^n$，滿足 $Df(x)[h] = \nabla f(x)^T h$。
3. **積分與Green公式**：熟悉分部積分在邊界項處理中的角色。
4. **有限維線性代數**：對稱矩陣的正定性、特征值與譜半徑。

本章將明確區分以下概念：
- **第一變分**：通常是 Gateaux 方向導數，需附加條件方可等同於 Fréchet 導數。
- **Euler–Lagrange 方程**：是能量泛函臨界點的必要條件，而非充分條件。
- **駐點與最小值**：駐點可能為極大、鞍點或極小，需透過第二變分或凸性判斷。

## 問題與直覺

在有限維空間 $\mathbb{R}^n$ 中，尋找 $f(x)$ 的最小值需解決 $\nabla f(x) = 0$。在變分法中，變量是函數 $u \in \mathcal{X}$（例如 $H^1_0(\Omega)$），目標是最小化能量泛函：
$$ J[u] = \int_\Omega L(x, u, \nabla u) \, dx $$
直覺上，若 $u^*$ 是最小值，則對任意微小擾動 $v$，$J[u^* + t v]$ 關於 $t$ 的導數在 $t=0$ 處應為零。這導出了第一變分方程。然而，無限維空間的拓撲結構使得「駐點」與「最小值」的關係更為複雜：
1. **導數的定義**：方向導數存在不保證 Fréchet 可微。
2. **邊界效應**：擾動 $v$ 在邊界上的行為決定了邊界條件是預設的（Essential/Dirichlet）還是自然產生的（Natural/Neumann）。
3. **正定性**：第二變分正定是局部極小的充分條件，但需滿足一致強制性（coercivity），單純逐點正定在無限維中不足夠。

## 定義、定理與推導

### 1. 能量泛函與導數的嚴格定義

設 $\Omega \subset \mathbb{R}^n$ 為有界光滑開區域。定義賦範函數空間 $\mathcal{X} = H^1_0(\Omega)$（Dirichlet 邊界）或 $H^1(\Omega)$（一般邊界）。能量泛函 $J: \mathcal{X} \to \mathbb{R}$ 定義為：
$$ J[u] = \int_\Omega L(x, u, \nabla u) \, dx $$
其中 $L: \Omega \times \mathbb{R} \times \mathbb{R}^n \to \mathbb{R}$ 為 $C^2$ 光滑函數，且關於 $p = \nabla u$ 滿足凸性條件 $L_{pp} \geq \lambda I$（$\lambda > 0$）以保證凸性。

**定義 1（Gateaux 第一變分）**：
若對所有 $u, v \in \mathcal{X}$，極限存在：
$$ \delta J[u; v] = \left. \frac{d}{dt} J[u + t v] \right|_{t=0} $$
則稱 $\delta J[u; \cdot]$ 為 $J$ 在 $u$ 沿方向 $v$ 的 Gateaux 導數。

**定義 2（Fréchet 導數）**：
若存在有界線性泛函 $DJ(u) \in \mathcal{X}^*$，使得：
$$ J[u+h] = J[u] + DJ(u)[h] + o(\|h\|_{\mathcal{X}}) \quad (\|h\|_{\mathcal{X}} \to 0) $$
則 $J$ 在 $u$ 處 Fréchet 可微，且 $DJ(u)$ 唯一。
**注意**：若 $J$ 在 $u$ 處 Fréchet 可微，則其 Gateaux 導數必為線性泛函且等於 Fréchet 導數，即 $\delta J[u; v] = DJ(u)[v]$。反之不成立。

### 2. Euler–Lagrange 方程的推導

**定理 1（弱 Euler–Lagrange 方程）**：
設 $L$ 關於 $(u, \nabla u)$ 具有連續的一階偏導數。若 $u \in C^1(\bar{\Omega})$ 滿足 $J[u+v] \ge J[u]$ 對所有 $v \in C_c^\infty(\Omega)$ 成立（即內部極小），則 $u$ 滿足弱形式：
$$ \int_\Omega \left( \frac{\partial L}{\partial u} v - \nabla \left( \frac{\partial L}{\partial (\nabla u)} \right) \cdot v \right) dx = 0 \quad \forall v \in C_c^\infty(\Omega) $$
若 $u$ 及相關導數足夠光滑（$C^2$），則由變分法基本引理（Fundamental Lemma of Calculus of Variations），可得經典形式：
$$ \nabla \cdot \frac{\partial L}{\partial (\nabla u)} - \frac{\partial L}{\partial u} = 0 \quad \text{in } \Omega $$

**證明**：
計算 $t \mapsto J[u+tv]$ 的導數：
$$ \frac{d}{dt} J[u+tv] = \frac{d}{dt} \int_\Omega L(x, u+tv, \nabla u + t \nabla v) \, dx $$
假設交換導數與積分是合法的（需 $L$ 的光滑性及有界性），應用鏈式法則：
$$ = \int_\Omega \left( \frac{\partial L}{\partial u} v + \frac{\partial L}{\partial (\nabla u)} \cdot \nabla v \right) dx $$
對第二項應用分部積分（Green 第一公式）：
$$ \int_\Omega \frac{\partial L}{\partial (\nabla u)} \cdot \nabla v \, dx = \int_{\partial \Omega} \left( \frac{\partial L}{\partial (\nabla u)} \cdot \mathbf{n} \right) v \, ds - \int_\Omega \nabla \cdot \left( \frac{\partial L}{\partial (\nabla u)} \right) v \, dx $$
由於 $v \in C_c^\infty(\Omega)$，在邊界 $\partial \Omega$ 上 $v=0$，故邊界項消失。
因此：
$$ \delta J[u; v] = \int_\Omega \left( \frac{\partial L}{\partial u} - \nabla \cdot \frac{\partial L}{\partial (\nabla u)} \right) v \, dx $$
若 $\delta J[u; v] = 0$ 對所有 $v \in C_c^\infty(\Omega)$ 成立，根據變分法基本引理，被積分函數必處處為零（或幾乎處處為零，依空間而定）。

**推論（自然邊界條件）**：
若邊界值未指定（即 $v$ 在 $\partial \Omega$ 上不恆為零），則分部積分後的邊界項必須獨立消失：
$$ \int_{\partial \Omega} \left( \frac{\partial L}{\partial (\nabla u)} \cdot \mathbf{n} \right) v \, ds = 0 \quad \forall v|_{\partial \Omega} $$
這要求：
$$ \frac{\partial L}{\partial (\nabla u)} \cdot \mathbf{n} = 0 \quad \text{on } \partial \Omega $$
對於標準能量 $L = \frac{1}{2} |\nabla u|^2$，此即 $\nabla u \cdot \mathbf{n} = 0$（Neumann 邊界條件）。

### 3. 第二變分與最小值條件

**定義 3（第二變分）**：
$$ \delta^2 J[u; v] = \left. \frac{d^2}{dt^2} J[u + t v] \right|_{t=0} $$

**定理 2（局部極小的充分條件）**：
設 $J$ 在 $u$ 處二階 Fréchet 可微。若存在常數 $c > 0$ 使得對所有 $v \in \mathcal{X}$：
$$ \delta^2 J[u; v] \geq c \|v\|_{\mathcal{X}}^2 $$
則 $u$ 是 $J$ 的嚴格局部最小值。
**注意**：僅有 $\delta^2 J[u; v] > 0$ 對所有非零 $v$ 成立（逐點正定）在無限維空間中通常不足以保证局部極小，因為缺乏一致强制性（uniform coercivity）。

**手算核對**：
對於 $L = \frac{1}{2} |\nabla u|^2$，有 $\frac{\partial^2 L}{\partial p \partial q} = I$。
$$ \delta^2 J[u; v] = \int_\Omega \nabla v \cdot \nabla v \, dx = \|\nabla v\|_{L^2}^2 $$
由 Poincaré 不等式，若 $v \in H^1_0(\Omega)$，則 $\|v\|_{H^1} \leq C \|\nabla v\|_{L^2}$。因此 $\delta^2 J[u; v] = \|\nabla v\|_{L^2}^2$ 控制了 $H^1$ 範數的平方（差一個常數倍），滿足強制性條件。

## 逐步手算例題

### 例 1：線性彈性膜的能量最小化

**問題**：考慮單位區間 $\Omega=(0,1)$ 上的膜，能量泛函為：
$$ J[u] = \frac{1}{2} \int_0^1 (u')^2 \, dx - \int_0^1 f(x) u(x) \, dx $$
邊界條件 $u(0)=0, u(1)=0$。求使 $J$ 最小化的 $u$。

**解答**：
1. **第一變分**：
   取容許擾動 $v \in C_c^\infty(0,1)$。
   $$ \delta J[u; v] = \int_0^1 (u' v') \, dx - \int_0^1 f v \, dx $$
   對第一項分部積分：
   $$ \int_0^1 u' v' \, dx = [u' v]_0^1 - \int_0^1 u'' v \, dx = - \int_0^1 u'' v \, dx $$
   因此：
   $$ \delta J[u; v] = \int_0^1 (-u'' - f) v \, dx $$
   令其為零，得 Euler–Lagrange 方程：
   $$ -u'' = f \quad \text{in } (0,1) $$

2. **求解**：
   設 $f(x) = \pi^2 \sin(\pi x)$。則 $-u'' = \pi^2 \sin(\pi x)$。
   通解為 $u(x) = -\sin(\pi x) + C_1 x + C_2$。
   邊界 $u(0)=0 \implies C_2=0$。
   邊界 $u(1)=0 \implies -0 + C_1 = 0 \implies C_1=0$。
   故 $u^*(x) = -\sin(\pi x)$。
   驗證能量：
   $$ J[-\sin(\pi x)] = \frac{1}{2} \int_0^1 \pi^2 \cos^2(\pi x) dx - \int_0^1 \pi^2 \sin(\pi x) (-\sin(\pi x)) dx $$
   $$ = \frac{\pi^2}{4} + \frac{\pi^2}{2} = \frac{3\pi^2}{4} $$

3. **第二變分**：
   $$ \delta^2 J[u^*; v] = \int_0^1 (v')^2 dx \geq 0 $$
   且由 Wirtinger 不等式，此形式正定，故 $u^*$ 為全局最小。

### 例 2：含非線性項的自然邊界

**問題**：考慮 $\Omega=(0,1)$，$L(u, u') = \frac{1}{2}(u')^2 + \frac{1}{4}u^4 - g u$。左端 $u(0)=0$，右端 $x=1$ 未指定邊界值。

**解答**：
1. **Euler–Lagrange 方程**：
   $$ \frac{\partial L}{\partial u} = u^3 - g, \quad \frac{\partial L}{\partial u'} = u' $$
   $$ \nabla \cdot (u') - (u^3 - g) = 0 \implies u'' - u^3 + g = 0 \implies u'' = u^3 - g $$
   **注意**：前稿錯誤寫成 $-u'' + u^3 = g$，正確推導為 $u'' = u^3 - g$。

2. **自然邊界條件**：
   在 $x=0$，$v(0)=0$。在 $x=1$，$v(1)$ 自由。
   邊界項為 $[u' v]_0^1 = u'(1)v(1) - u'(0)v(0)$。
   要使 $\delta J = 0$ 對所有 $v(1)$ 成立，需 $u'(1) = 0$。
   故邊界條件為：$u(0)=0, u'(1)=0$。

## 實作與程式

我們構建一個自足的 CPU 實作，驗證離散能量泛函的梯度計算與 Euler–Lagrange 方程的一致性。

**模型**：一維 Dirichlet 問題 $-u''=f$ on $(0,1)$, $u(0)=u(1)=0$。
離散網格：$x_i = i h, i=0,\dots,N$，其中 $h=1/N$。
未知量：$U = [u_1, \dots, u_{N-1}]^T \in \mathbb{R}^{N-1}$。
離散能量（梯形法近似或中點法，此處用節點加權）：
$$ J_d(U) = \frac{1}{2h} \sum_{i=0}^{N-1} (u_{i+1} - u_i)^2 - \sum_{j=1}^{N-1} f_j u_j h $$
其中 $u_0=u_N=0$。

**離散梯度推導**：
對 $u_j$ ($1 \le j \le N-1$) 求偏導：
$$ \frac{\partial J_d}{\partial u_j} = \frac{1}{2h} \left[ 2(u_j - u_{j-1}) + 2(u_{j+1} - u_j)(-1) \right] - f_j h $$
$$ = \frac{1}{h} (u_j - u_{j-1} - u_{j+1} + u_j) - f_j h $$
$$ = \frac{2u_j - u_{j-1} - u_{j+1}}{h} - f_j h $$
令梯度為零：
$$ \frac{-u_{j-1} + 2u_j - u_{j+1}}{h^2} = f_j $$
這正是標準的中心差分 Laplacian 離散形式。

**程式實作**：

```python
import numpy as np

def discrete_energy(U, f, h):
    """
    計算離散 Dirichlet 能量。
    U: (N-1,) 內部節點值
    f: (N-1,) 右側項
    h: 步長
    """
    N = len(U) + 2
    U_full = np.zeros(N)
    U_full[1:-1] = U
    
    dU = np.diff(U_full)
    gradient_term = 0.5 / h * np.sum(dU**2)
    potential_term = np.dot(f, U) * h
    return gradient_term - potential_term

def discrete_gradient(U, f, h):
    """
    計算離散能量關於 U 的梯度。
    公式: (2U_j - U_{j-1} - U_{j+1})/h - f_j * h
    """
    N = len(U)
    grad = np.zeros(N)
    
    if N == 1:
        # 只有一個內點 U_1
        # d/dU1 [ 0.5/h * (U1-0)^2 + 0.5/h * (0-U1)^2 ] - f1*h
        # = (1/h * U1) - f1*h
        grad[0] = (2*U[0] - 0 - 0) / h - f[0] * h
    else:
        # 內部點
        grad[1:-1] = (2*U[1:-1] - U[:-2] - U[2:]) / h - f[1:-1] * h
        # 邊界點 j=1 (index 0)
        grad[0] = (2*U[0] - 0 - U[1]) / h - f[0] * h
        # 邊界點 j=N-1 (index -1)
        grad[-1] = (2*U[-1] - U[-2] - 0) / h - f[-1] * h
        
    return grad

def solve_laplacian(f, h, method='direct'):
    """
    求解離散 Laplacian 系統 A u = b
    A 是三對角矩陣，元素為 [1, -2, 1] / h^2
    b = f * h^2
    """
    N = len(f)
    # 構造對稱正定矩陣 A
    A = np.zeros((N, N))
    b = f * h**2
    
    for i in range(N):
        A[i, i] = -2.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < N - 1:
            A[i, i+1] = 1.0
            
    A /= h**2
    
    # 直接求解 (小規模問題)
    u = np.linalg.solve(A, b)
    return u

# 測試參數
N = 100
h = 1.0 / N
# 內點 x_j = j*h, j=1..N-1
x_internal = np.arange(1, N) * h
f_exact = np.pi**2 * np.sin(np.pi * x_internal)  # -u'' = pi^2 sin(pi x) => u = sin(pi x)

# 解析解
u_exact = np.sin(np.pi * x_internal)

# 求解
u_num = solve_laplacian(f_exact, h)

# 計算誤差
error = np.max(np.abs(u_num - u_exact))
print(f"Max Error (Direct Solve): {error:.2e}")

# 驗證梯度
# 在解析解處，離散梯度應接近 0
grad_at_exact = discrete_gradient(u_exact, f_exact, h)
print(f"Gradient Norm at Exact Solution: {np.linalg.norm(grad_at_exact):.2e}")

# 方向導數核對
epsilon = 1e-5
j_test = N // 2
u_pert = u_exact.copy()
u_pert[j_test] += epsilon
fd_grad = (discrete_energy(u_pert, f_exact, h) - discrete_energy(u_exact, f_exact, h)) / epsilon
analytic_grad = discrete_gradient(u_exact, f_exact, h)[j_test]
print(f"FD vs Analytic Gradient Diff: {abs(fd_grad - analytic_grad):.2e}")
```

**預期結果**：
1. **Max Error**：應為 $O(h^2)$ 量級，約 $10^{-4}$。
2. **Gradient Norm**：離散解滿足離散 Euler–Lagrange 方程，故梯度應極小（$< 10^{-8}$）。
3. **FD vs Analytic**：有限差分導數應與解析梯度高度一致（差異 $< 10^{-4}$）。

## 測試與預期結果

我們設計三類測試以驗證程式的健壯性。

1. **正常測試（Normal）**：
   - 輸入：$f(x) = \pi^2 \sin(\pi x)$，$N=50$。
   - 預期：數值解與解析解最大誤差小於 $10^{-3}$。梯度範數小於 $10^{-6}$。

2. **邊界測試（Boundary）**：
   - 輸入：$N=2$（僅 $u_1$ 一個內點）。
   - 預期：
     $$ \frac{\partial J_d}{\partial u_1} = \frac{2u_1}{h} - f_1 h $$
     程式應正確處理長度為 1 的陣列，不發生索引越界。
     解析解 $u_1 = f_1 h^2 / 2$。

3. **故障測試（Fault）**：
   - 輸入：$h \le 0$ 或 $f$ 含 NaN。
   - 預期：程式應拋出 `ValueError` 或 `AssertionError`。
     需在入口處添加：
     ```python
     if h <= 0:
         raise ValueError("Step size h must be positive.")
     if not np.all(np.isfinite(f)):
         raise ValueError("f must be finite.")
     ```

## 反例與常見陷阱

1. **駐點非最小值**：
   考慮 $J[u] = -\frac{1}{2} \int (u')^2 dx$。Euler–Lagrange 方程 $u''=0$ 的線性解是駐點，但由於能量負定，這些點是局部最大值。必須檢查第二變分的正負。

2. **忽略自然邊界**：
   在求解 $J[u] = \frac{1}{2}\int (u')^2 dx - \int f u dx$ 且僅指定 $u(0)=0$ 時，若錯誤地強加 $u(1)=0$，則解會不符合自然邊界條件 $u'(1)=0$，導致物理意義錯誤（例如，梁的自由端力矩應為零，而非位移為零）。

3. **離散與連續梯度混淆**：
   離散梯度 $\nabla J_d(U)$ 是 $\mathbb{R}^{N-1}$ 中的向量。連續梯度 $DJ[u]$ 是函數空間中的對偶元素。在數值優化中，我們使用離散梯度，但收斂性分析需考慮連續空間的性質。若 $h \to 0$，離散梯度在特定範數下收斂至連續梯度，但這依賴於離散方案的相容性。

4. **步長選擇發散**：
   若使用梯度下降法 $U_{k+1} = U_k - \alpha \nabla J_d(U_k)$，步長 $\alpha$ 必須小於 $2/\lambda_{\max}$。對於離散 Laplacian，$\lambda_{\max} \approx 4/h^2$（依定義不同可能為 $4/h$ 或 $4/h^2$，需嚴格對應能量定義）。若 $\alpha$ 過大，解將發散。

## AI、幾何與養殖案例

**幾何直覺**：
變分法可視為在函數空間的黎曼幾何。能量泛函的等高線曲面，Euler–Lagrange 方程是測地線方程的推廣。在有限維，梯度指向最陡上升方向；在無限維，Fréchet 梯度定義了切空間中的方向。

**養殖案例（合成）**：
假設魚塘水溫 $T(x)$ 滿足熱平衡方程，我們希望通過控制加熱器分布 $q(x)$ 使水溫均勻（最小化變異能量）：
$$ J[q] = \int_\Omega (T(x) - \bar{T})^2 dx + \lambda \int_\Omega q^2 dx $$
其中 $T$ 是 $q$ 的線性函數 $T = G q$（$G$ 為Green算子）。
這是一個二次凸優化問題。
- **第一變分**：$\delta J = 2 \int (T-\bar{T}) \delta T \, dx + 2\lambda \int q \delta q \, dx = 0$。
- **解**：導出 $q^* = -\frac{1}{\lambda} G^T (T-\bar{T})$。
- **陷阱**：若 $G$ 不可逆（例如低頻模式不可控），則解不唯一，需加邊界條件或正則化。

**AI 應用**：
深度學習中的物理資訊神經網絡（PINN）通過最小化物理殘差能量來訓練。本章的變分框架提供了理解損失函數收斂性與解穩定性的數學基礎。若損失函數非凸（如高階非線性 PDE），則可能陷入鞍點而非最小值。

## 習題

1. **手算**：
   對於 $J[u] = \int_0^1 \left( \frac{1}{2} (u')^2 + \frac{1}{2} u^2 - f u \right) dx$，邊界 $u(0)=1, u(1)=0$。
   (a) 推導 Euler–Lagrange 方程。
   (b) 若 $f=0$，求解析解 $u(x)$。

2. **程式**：
   修改 `discrete_gradient` 函數以支持 Neumann 邊界條件 $u'(0)=0, u'(1)=0$。
   提示：在 $x=0$ 處，使用對稱差分 $u_0 = u_1$ 或引入虛點（ghost node）。驗證梯度計算。

3. **反例**：
   構造一個泛函 $J[u]$ 及其容許空間，使得其駐點不是局部最小值。請明確給出第二變分並展示其不正定。

4. **整合**：
   比較連續空間中的 $L^2$ 梯度與離散空間中的 $\ell^2$ 梯度。若離散方案是一階收斂的，分析兩者差異的量級。

## 習題解答

1. **解答**：
   (a) $L_u = u - f$, $L_{u'} = u'$。
       EL 方程：$u'' - u + f = 0$ 即 $-u'' + u = f$。
   (b) $f=0 \implies u'' - u = 0$。
       通解 $u(x) = A e^x + B e^{-x}$ 或 $A \cosh x + B \sinh x$。
       $u(0)=1 \implies A=1$ (若用 cosh/sinh)。
       $u(1)=0 \implies \cosh 1 + B \sinh 1 = 0 \implies B = -\coth 1$。
       $u(x) = \cosh x - \coth 1 \sinh x$。
       驗證：$u(0)=1, u(1)=\cosh 1 - \coth 1 \sinh 1 = 0$。

2. **解答**：
   對於 Neumann 邊界，能量定義需調整。
   若 $u'(0)=0$，可視 $u_0 = u_1$。
   在 $j=1$ (index 0) 處，梯度項 $\frac{2u_1 - u_0 - u_2}{h}$ 變為 $\frac{2u_1 - u_1 - u_2}{h} = \frac{u_1 - u_2}{h}$。
   程式修改：
   ```python
   # Neumann case at left boundary
   grad[0] = (U[0] - U[1]) / h - f[0] * h  # Approx
   # Or more accurately depending on discretization
   ```
   需仔細推導離散算子以保持一致性。

3. **解答**：
   取 $J[u] = \int_0^1 (u')^4 dx$。
   $\delta J = \int 4(u')^3 v' dx = 0 \implies (u')^3 = C \implies u' = C^{1/3} = \text{const}$。
   駐點為線性函數 $u(x) = ax+b$。
   $\delta^2 J = \int 12 (u')^2 (v')^2 dx$。
   若 $u'=0$ (即 $u=\text{const}$)，$\delta^2 J = 0$。
   若 $u' \neq 0$，$\delta^2 J > 0$。
   此例中線性解為極小。
   要構造非極小，取 $J[u] = -\int (u')^4 dx$。
   $\delta^2 J = -\int 12 (u')^2 (v')^2 dx \le 0$。
   故任何駐點（線性函數）都是局部最大值，而非最小值。

4. **解答**：
   連續梯度 $g \in L^2(\Omega)$，離散梯度 $G \in \ell^2$。
   $G_j \approx g(x_j)$。
   若離散方案為一階收斂 $O(h)$，則 $G - g|_{grid} = O(h)$ 點態或 $L^2$ 意義下。
   在優化中，使用 $G$ 進行更新 $U \leftarrow U - \alpha G$。
   當 $h \to 0$，此過程收斂至連續梯度下降流。

## 本章小結

本章嚴格建立了變分法的基礎，區分了 Gateaux 與 Fréchet 導數，推導了 Euler–Lagrange 方程及其邊界條件。我們強調駐點與最小值的區別，並通過手算與程式實作驗證了離散與連續方法的對應。常見陷阱包括忽略自然邊界、錯誤判斷正定性及步長選擇不當。變分法是連接微積分、PDE 與數值優化的橋樑。

## 參考來源

1.  Lebl, J. (2017). *Basic Analysis: An Introduction to Real Analysis*.
2.  MIT OpenCourseWare. (2020). *18.100A Real Analysis*.
3.  Evans, L. C. (2010). *Partial Differential Equations*. 2nd ed. (Chapter 6: Calculus of Variations).
4.  JAX Documentation. *Autodiff Cookbook*.
5.  SciPy Documentation. *scipy.optimize.minimize*.