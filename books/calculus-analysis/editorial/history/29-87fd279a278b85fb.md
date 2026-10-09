# 第29章 變分法、第一變分與能量泛函

## 學習目標與先備知識

本章建立變分法的基礎語言，將函數的局部變化推廣至函數空間。我們定義能量泛函、第一變分（Fréchet 導數）與第二變分（Hessian），推導 Euler–Lagrange 方程，並區分站點與最小值。讀者需熟悉以下先備知識：
1. **Fréchet 導數**：$f(x+h) = f(x) + Df(x)[h] + r(h)$，其中 $\|r(h)\|/\|h\| \to 0$。
2. **梯度和 Hessian**：標量函數的梯度 $\nabla f$ 與對稱 Hessian $H_f$。
3. **積分與換變數**：Riemann 積分與 Jacobian 行列式在變換中的角色。
4. **矩陣微分**：$df = \text{tr}(G^T dX)$ 及鏈式法則。

本章核心目標是理解：變分問題的「駐點」僅滿足必要條件，必須通過第二變分或凸性分析才能判定是否為最小值。我們將展示連續空間中的梯度概念與離數化模型（如第三卷的相場模擬）之間的對應關係。

## 問題與直覺

在常規微積分中，我們尋找函數 $f: \mathbb{R}^n \to \mathbb{R}$ 的極值，變量是向量 $x$。在變分法中，變量是函數 $u: \Omega \to \mathbb{R}$。我們定義一個「能量泛函」$J: C^1(\bar{\Omega}) \to \mathbb{R}$，例如二維拉普拉斯問題的狄利克雷能量：
$$ J[u] = \frac{1}{2} \int_\Omega |\nabla u(x)|^2 \, dx $$
我們的目標是找到使 $J[u]$ 最小化的函數 $u$。

**直覺**：如果 $u$ 是最小值，那麼對 $u$ 進行任意小幅度的「容許擾動」 $v$，能量 $J[u+tv]$ 關於 $t$ 的導數在 $t=0$ 處應為零。這對應於經典微積分中 $\nabla f(x^*)=0$ 的條件。然而，與有限維不同，無限維空間中的「駐點」可能對應於極小、極大或鞍點。此外，邊界條件會影響擾動的定義，產生「自然邊界條件」。

## 定義、定理與推導

### 1. 能量泛函與容許擾動

設 $\Omega \subset \mathbb{R}^n$ 為有界開區域。定義能量泛函：
$$ J[u] = \int_\Omega L(x, u, \nabla u) \, dx $$
其中 $L$ 為拉格朗日量。考慮擾動 $v \in C^1(\bar{\Omega})$，定義路径 $u_t = u + t v$。
**第一變分**（First Variation）定義為：
$$ \delta J[u; v] = \left. \frac{d}{dt} J[u + t v] \right|_{t=0} $$
若 $\delta J[u; v] = 0$ 對所有容許擾動 $v$ 成立，則 $u$ 為駐點。

**定理（Euler–Lagrange 方程）**：
設 $L$ 關於 $\nabla u$ 和 $u$ 具有連續的一階偏導數。若 $u$ 使 $J$ 取駐點，且邊界條件為 Dirichlet（即 $v|_{\partial \Omega} = 0$），則 $u$ 滿足：
$$ \nabla \cdot \frac{\partial L}{\partial (\nabla u)} - \frac{\partial L}{\partial u} = 0 \quad \text{in } \Omega $$
**證明概要**：
計算 $\delta J[u; v] = \int_\Omega \left( \frac{\partial L}{\partial u} v + \frac{\partial L}{\partial (\nabla u)} \cdot \nabla v \right) dx$。
對第二項使用分部積分（Green 公式）：
$$ \int_\Omega \frac{\partial L}{\partial (\nabla u)} \cdot \nabla v \, dx = \int_{\partial \Omega} \left( \frac{\partial L}{\partial (\nabla u)} \cdot \mathbf{n} \right) v \, ds - \int_\Omega \nabla \cdot \left( \frac{\partial L}{\partial (\nabla u)} \right) v \, dx $$
由於 $v|_{\partial \Omega} = 0$，邊界項消失。因此：
$$ \delta J[u; v] = \int_\Omega \left( \frac{\partial L}{\partial u} - \nabla \cdot \frac{\partial L}{\partial (\nabla u)} \right) v \, dx $$
若此式對所有光滑且邊界為零的 $v$ 成立，由基里爾-斯頓姆引理（Fundamental Lemma of Calculus of Variations），被積分函數必處處為零。

**自然邊界條件**：
若邊界值未指定（Neumann 條件），則 $v|_{\partial \Omega}$ 不恆為零。此時邊界項必須獨立消失：
$$ \left( \frac{\partial L}{\partial (\nabla u)} \cdot \mathbf{n} \right) v = 0 \quad \forall v \implies \frac{\partial L}{\partial (\nabla u)} \cdot \mathbf{n} = 0 \quad \text{on } \partial \Omega $$

### 2. 第二變分與最小值條件

第一變分為零僅保證 $u$ 是駐點。為了判斷是否為局部最小，需考察第二變分。
**第二變分**定義為：
$$ \delta^2 J[u; v] = \left. \frac{d^2}{dt^2} J[u + t v] \right|_{t=0} $$
對於二維 Dirichlet 能量 $J[u] = \frac{1}{2} \int_\Omega |\nabla u|^2 dx$，拉格朗日量 $L = \frac{1}{2} |\nabla u|^2$。
計算：
$$ \frac{\partial L}{\partial (\nabla u)} = \nabla u, \quad \frac{\partial^2 L}{\partial (\nabla u)^2} = I $$
$$ \delta^2 J[u; v] = \int_\Omega \nabla v \cdot \nabla v \, dx = \int_\Omega |\nabla v|^2 \, dx \ge 0 $$
若 $\delta^2 J[u; v] > 0$ 對所有非零容許 $v$ 成立，則 $u$ 是嚴格局部最小。若存在 $v$ 使 $\delta^2 J[u; v] < 0$，則 $u$ 不是局部最小。

### 3. 駐點非必為最小

在有限維，若 Hessian 正定則為極小。在無限維，必須確認第二變分正定。
**反例**：考慮 $L = -\frac{1}{2} |\nabla u|^2$（負能量）。
$$ \delta^2 J[u; v] = -\int_\Omega |\nabla v|^2 dx \le 0 $$
任何滿足 Euler–Lagrange 方程（$\Delta u = 0$）的函數都是駐點，但能量泛函在這些點處是「凹」的，因此它們是局部最大值（或無界，取決於邊界），而非最小值。

## 逐步手算例題

### 例 1：一維二維拉普拉斯能量的 Euler–Lagrange 方程

**問題**：設 $\Omega = (0, \pi)$，$L(u, u') = \frac{1}{2} (u')^2 - f u$，邊界 $u(0)=0, u(\pi)=0$。求使 $J[u]$ 最小的 $u$。
**解答**：
1. **第一變分**：
   $$ J[u] = \int_0^\pi \left( \frac{1}{2} (u')^2 - f u \right) dx $$
   $$ \delta J = \int_0^\pi (u' v' - f v) dx $$
   分部積分 $\int_0^\pi u' v' dx = [u' v]_0^\pi - \int_0^\pi u'' v dx$。
   由於 $v(0)=v(\pi)=0$，邊界項為零。
   $$ \delta J = \int_0^\pi (-u'' - f) v dx $$
   令係數為零：$-u'' - f = 0 \implies u'' = -f$。
2. **求解**：
   設 $f(x) = \sin(x)$。則 $u'' = -\sin(x)$。
   積分得 $u' = \cos(x) + C_1$，再積分得 $u = \sin(x) + C_1 x + C_2$。
   邊界 $u(0)=0 \implies C_2=0$。
   邊界 $u(\pi)=0 \implies \sin(\pi) + C_1 \pi = 0 \implies C_1=0$。
   解為 $u^*(x) = \sin(x)$。
3. **第二變分**：
   $$ \delta^2 J = \int_0^\pi (v')^2 dx \ge 0 $$
   正定，故 $u^*$ 為全局最小。

### 例 2：含非線性項的能量泛函與自然邊界

**問題**：設 $\Omega = (0, 1)$，$L(u, u') = \frac{1}{2} (u')^2 + \frac{1}{4} u^4 - g u$。右端點 $x=1$ 未指定 $u(1)$，左端 $u(0)=0$。
**解答**：
1. **Euler–Lagrange**：
   $$ \frac{\partial L}{\partial u} = u^3 - g, \quad \frac{\partial L}{\partial u'} = u' $$
   $$ -\frac{d}{dx}(u') - (u^3 - g) = 0 \implies u'' + u^3 = g \quad \text{in } (0,1) $$
2. **自然邊界條件**：
   在 $x=0$，$v(0)=0$。在 $x=1$，$v(1)$ 自由。
   邊界項為 $[u' v]_0^1 = u'(1) v(1) - u'(0) v(0)$。
   要使 $\delta J=0$ 對所有 $v(1)$ 成立，需 $u'(1) = 0$。
3. **結論**：
   解需滿足 $u'' + u^3 = g$，$u(0)=0$，$u'(1)=0$。
   注意此處 $u'(1)=0$ 是自然產生的，而非預設的 Dirichlet 條件。

## 實作與程式

我們使用 Python 和 NumPy 構建一個自足的 CPU 實作，驗證離散能量泛函的一階導數（梯度）與 Euler–Lagrange 方程的離散形式的一致性。

**模型**：一維 Dirichlet 問題，域 $[0, 1]$ 離散為 $N+1$ 個節點，步長 $h = 1/N$。
未知量 $u_1, \dots, u_{N-1}$（邊界 $u_0=0, u_N=0$）。
離散能量：
$$ J_d(u) = \frac{1}{2} \sum_{i=0}^{N-1} \left( \frac{u_{i+1} - u_i}{h} \right)^2 h - \sum_{i=1}^{N-1} f_i u_i h $$
**梯度計算**：
對 $u_j$ ($1 \le j \le N-1$) 求導：
$$ \frac{\partial J_d}{\partial u_j} = \frac{1}{h} \left( \frac{u_{j+1} - u_j}{h} - \frac{u_j - u_{j-1}}{h} \right) h - f_j h $$
$$ = \frac{u_{j-1} - 2u_j + u_{j+1}}{h^2} h - f_j h $$
$$ = \frac{u_{j-1} - 2u_j + u_{j+1}}{h} - f_j h $$
令梯度為零，即離散 Laplacian 方程：
$$ \frac{u_{j-1} - 2u_j + u_{j+1}}{h^2} = f_j $$

```python
import numpy as np

def discrete_energy(u, f, h):
    """
    計算離散 Dirichlet 能量。
    u: 內部節點值 (N-1,)
    f: 右側項 (N-1,)
    h: 步長
    """
    N = len(u) + 2  # 總節點數
    u_full = np.zeros(N)
    u_full[1:-1] = u
    # u_full[0] = 0, u_full[-1] = 0 by initialization
    
    # 導數
    du = np.diff(u_full) / h
    gradient_energy = 0.5 * np.sum(du**2) * h
    potential_energy = np.dot(f, u) * h
    return gradient_energy - potential_energy

def discrete_gradient(u, f, h):
    """
    計算能量泛函關於內部節點 u 的梯度。
    """
    N = len(u)
    grad = np.zeros(N)
    # 邊界項處理
    # du_{j-1/2} = (u_j - u_{j-1})/h
    # d/d(u_j) [ 0.5 * ( (u_j - u_{j-1})/h )^2 * h ] = (u_j - u_{j-1})/h
    # d/d(u_j) [ 0.5 * ( (u_{j+1} - u_j)/h )^2 * h ] = -(u_{j+1} - u_j)/h
    
    left_diffs = np.empty(N)
    right_diffs = np.empty(N)
    
    # u_0 is 0
    left_diffs[0] = (u[0] - 0) / h
    if N > 1:
        left_diffs[1:] = (u[1:] - u[:-1]) / h
        
    # u_N is 0
    right_diffs[-1] = (0 - u[-1]) / h
    if N > 1:
        right_diffs[:-1] = (u[:-1] - u[1:]) / h
        
    grad = left_diffs - right_diffs - f * h
    
    return grad

def solve_laplacian(f, h, tol=1e-10, max_iter=1000):
    """
    使用梯度下降求解離散 Laplacian。
    """
    N = len(f)
    u = np.zeros(N)
    alpha = 0.01 # Step size
    
    for _ in range(max_iter):
        g = discrete_gradient(u, f, h)
        if np.linalg.norm(g) < tol:
            break
        u = u - alpha * g
        
    return u, np.linalg.norm(discrete_gradient(u, f, h))

# 測試
N = 100
h = 1.0 / N
x = np.linspace(h, 1-h, N)
f_exact = np.sin(np.pi * x)  # u'' = -sin(pi x) => -u'' = sin(pi x)
# 注意離散方程是 u_{j-1}-2u_j+u_{j+1}/h^2 = f_j
# 所以 f 對應的是 -u''

u_num, final_grad_norm = solve_laplacian(f_exact, h)
u_exact = np.sin(np.pi * x)

error = np.max(np.abs(u_num - u_exact))
print(f"Max Error: {error:.2e}")
print(f"Final Gradient Norm: {final_grad_norm:.2e}")
```

**預期結果**：
對於 $N=100$，二階精度的離散 Laplacian 應產生約 $O(h^2)$ 的誤差，即 $10^{-4}$ 量級。最終梯度範數應接近零（在 $10^{-10}$ 容差內）。

## 測試與預期結果

我們設計三類測試：正常、邊界、故障。

1. **正常測試（Normal）**：
   - 輸入：$f(x) = \sin(\pi x)$，$N=50$。
   - 預期：數值解 $u$ 與解析解 $\sin(\pi x)$ 的最大誤差小於 $10^{-3}$。梯度範數收斂至零。

2. **邊界測試（Boundary）**：
   - 輸入：$N=1$（僅一個內部節點）。
   - 預期：梯度公式退化。$u_1$ 的梯度應為 $(u_0 - 2u_1 + u_2)/h - f_1 h = (-2u_1)/h - f_1 h$。程式應正確處理陣列邊界，不產生索引錯誤。

3. **故障測試（Fault）**：
   - 輸入：負步長 $h < 0$ 或非平滑 $f$（含 NaN）。
   - 預期：程式不應崩潰，但應檢測輸入有效性。若 $f$ 含 NaN，梯度計算應傳播 NaN，以便調試。在生產環境中，應在入口處驗證 $h > 0$ 且 $f$ 為有限數。

**驗證連續與離散梯度的對應**：
比較 $\nabla J_d$ 與有限差分近似 $\frac{J(u+\epsilon e_j) - J(u)}{\epsilon}$。
```python
def verify_gradient(u, f, h, epsilon=1e-6):
    grad = discrete_gradient(u, f, h)
    fd_grad = np.zeros(len(u))
    for j in range(len(u)):
        u_pert = u.copy()
        u_pert[j] += epsilon
        fd_grad[j] = (discrete_energy(u_pert, f, h) - discrete_energy(u, f, h)) / epsilon
    return np.max(np.abs(grad - fd_grad))
```
預期輸出應小於 $10^{-5}$，驗證解析梯度計算正確。

## 反例與常見陷阱

1. **駐點即最小值的誤解**：
   如前所述，若 $L = -\frac{1}{2}|\nabla u|^2$，Euler–Lagrange 方程 $\Delta u = 0$ 的解是能量最大值（在固定邊界下），而非最小值。必須檢查第二變分或泛函的凸性。

2. **忽略自然邊界條件**：
   在求解 Neumann 問題時，若錯誤地施加 Dirichlet 邊界條件，將導致解錯誤。例如，在 $x=1$ 處若 $u'(1) \neq 0$，則第一變分不會為零，該點不是駐點。

3. **離散梯度的邊界處理**：
   在程式中，若錯誤地假設 $u_{-1}$ 或 $u_N$ 存在而未特別處理邊界節點，將導致離散梯度計算錯誤，從而收斂到錯誤的解。

4. **步長選擇**：
   梯度下降的收斂取決於步長 $\alpha$。若 $\alpha$ 過大，可能發散；過小則收斂緩慢。對於離散 Laplacian，譜半徑約為 $4/h^2$，建議 $\alpha < 2 h^2 / 4$。

## AI、幾何與養殖案例

**幾何直覺**：
變分法可視為在函數空間的「地形圖」上尋找山谷。能量曲面 $J[u]$ 的梯度指向能量增加最快的方向。Euler–Lagrange 方程描述了地形平坦的方向（梯度為零）。在幾何上，最小曲率曲面（極小曲面）的解滿足平均曲率為零，這正是變分法在曲面面積泛函上的應用。

**養殖案例（合成）**：
假設我們有一個水產養殖池的溫度分佈模型。溫度 $T(x, t)$ 滿足熱方程。我們希望設計一個控制輸入（加熱器功率 $q(x)$），使池內溫度均勻（即最小化溫度變異能量）：
$$ J[q] = \int_\Omega (\text{Var}(T))^2 dx + \lambda \int_\Omega q^2 dx $$
其中第二項是能量懲罰項。
- **第一變分**：對 $q$ 擾動，利用熱方程的對偶系統計算靈敏度。
- **Euler–Lagrange**：導出控制方程，確定最佳加熱功率分布。
- **陷阱**：若忽略邊界的自然條件（例如池壁絕熱），模型將高估邊界溫度，導致控制策略失效。
- **AI 應用**：神經網絡可學習此變分問題的解映射，但必須確保訓練損失函數與物理能量泛函一致，並通過梯度核對驗證物理一致性。

## 習題

1. **手算**：對於 $L(u, u') = \frac{1}{2} (u')^2 + \frac{1}{2} u^2 - f u$，推導 Euler–Lagrange 方程。若 $f=0$，邊界 $u(0)=1, u(L)=0$，求最小能量函數。
2. **程式**：修改上述程式，支持 Neumann 邊界條件 $u'(0)=0, u'(L)=0$。驗證能量函數的梯度。
3. **反例**：構造一個泛函 $J[u] = \int (u')^2 dx$，使得存在兩個不同的駐點，其中一個是極小，另一個是鞍點。
4. **整合**：結合第三卷的相場模擬，將離散能量梯度與市場訂單簿的深度變異聯繫起來。解釋為何離散梯度不同於連續梯度的導數。

## 習題解答

1. **解答**：
   $\frac{\partial L}{\partial u} = u - f$, $\frac{\partial L}{\partial u'} = u'$。
   EL 方程：$-u'' - u + f = 0 \implies u'' + u = f$。
   若 $f=0$，通解 $u = A \cos x + B \sin x$。
   $u(0)=1 \implies A=1$。
   $u(L)=0 \implies \cos L + B \sin L = 0 \implies B = -\cot L$。
   解：$u(x) = \cos x - \cot L \sin x$。

2. **解答**：
   Neumann 條件意味著在邊界節點，擾動 $v$ 不為零，但物理上要求通量為零。
   離散能量需調整邊界項。對於 $u'(0)=0$，使用對稱差分或一階外推。
   程式修改點：在計算 `left_diffs[0]` 時，若 $u'(0)=0$，則 $u_0 = u_1$。
   驗證：計算梯度並確保對邊界節點的擾動導數正確反映自然邊界條件 $u_1 = u_2$ (for uniform grid with ghost nodes or adjusted weights)。

3. **解答**：
   考慮 $J[u] = \int_0^1 ((u')^2 - 1) dx$，邊界 $u(0)=0, u(1)=1$。
   EL 方程：$u'' = 0$。解 $u(x) = x$。
   第二變分：$\delta^2 J = \int_0^1 2 (v')^2 dx > 0$。這是極小。
   要構造鞍點，需非凸泛函，例如 $J[u] = \int ((u')^4 - (u')^2) dx$。
   $L = (u')^4 - (u')^2$。
   $\frac{\partial L}{\partial u'} = 4(u')^3 - 2(u')$。
   EL: $(4(u')^3 - 2u')' = 0$。
   解 $u(x) = 0$ (if boundaries allow) or linear.
   若 $u=0$，$\frac{\partial^2 L}{\partial (u')^2} = 12(u')^2 - 2 = -2$。
   $\delta^2 J = \int -2 (v')^2 dx < 0$。故 $u=0$ 是局部極大。
   若存在其他解，可能為鞍點。

4. **解答**：
   連續梯度是函數空間中的 Fréchet 導數。離散梯度是向量空間 $\mathbb{R}^N$ 中的導數。
   在相場模擬中，價格 $p_i$ 是離散變量。能量泛函可能是訂單簿不平衡度的積分。
   離散梯度 $\nabla J_d$ 給出價格變動對能量的影響。
   連續梯度 $\nabla J$ 是微分算子。
   差異在於：離散梯度包含離散誤差項 $O(h^k)$。
   若 $h \to 0$，離散梯度收斂至連續梯度（在弱收斂意義下）。
   實務上，使用離散梯度進行優化，但需用連續理論分析收斂性與穩定性。

## 本章小結

本章介紹了變分法的基礎，包括能量泛函、第一變分、Euler–Lagrange 方程、自然邊界條件及第二變分。我們強調駐點不等於最小值，需通過第二變分或凸性判定。透過手算與程式實作，我們驗證了離散與連續梯度的對應關係，並討論了常見陷阱。變分法為後續 PDE 求解、結構力學及優化問題提供了嚴格的數學基礎。

## 參考來源

1.  J. Lebl, Basic Analysis: An Introduction to Real Analysis, 2nd Ed. (2017).
2.  MIT OpenCourseWare, 18.100A Real Analysis, Fall 2020.
3.  MIT OpenCourseWare, 18.02SC Multivariable Calculus, Fall 2010.
4.  JAX Autodiff Cookbook: JVP/VJP, JAX Documentation.
5.  SciPy Documentation: scipy.linalg.expm.
6.  SciPy Documentation: scipy.optimize.minimize.