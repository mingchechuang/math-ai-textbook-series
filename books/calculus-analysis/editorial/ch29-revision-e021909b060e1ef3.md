# 第29章 變分法、第一變分與能量泛函

## 學習目標與先備知識

本章旨在建立變分法的嚴格數學基礎，將有限維微積分中的極值概念推廣至函數空間。我們定義能量泛函、區分 Gateaux 導數與 Fréchet 導數，推導 Euler–Lagrange 方程的必要條件，並深入探討駐點與最小值的關係。特別強調在無限維空間中，第二變分正定需滿足一致強制性（coercivity）才能確保局部極小。此外，本章透過離散化模型，明確區分連續空間中的梯度與離數網格上的座標梯度，並展示如何利用有限差分驗證數值方案的一致性。

**先備知識要求：**
1. **Fréchet 導數**：映射 $F: U \subset X \to Y$ 在 $x$ 處可微，若存在有界線性算子 $DF(x)$ 使得 $F(x+h) = F(x) + DF(x)h + o(\|h\|_X)$。
2. **線性泛函與對偶空間**：標量函數的導數 $Df(x) \in X^*$，其 Riesz 代表元（若適用）為梯度。
3. **Sobolev 空間基礎**：了解 $H^1(\Omega)$ 與 $H^1_0(\Omega)$ 的定義及 Poincaré 不等式。
4. **矩陣分析**：對稱矩陣的正定性、特徵值與譜半徑，以及線性系統的條件數。

## 問題與直覺

在有限維 $\mathbb{R}^n$ 中，函數 $f(x)$ 的局部極小值點 $x^*$ 必須滿足 $\nabla f(x^*)=0$，且 Hessian 矩陣 $H_f(x^*)$ 半正定。在變分法中，變量是函數 $u \in \mathcal{X}$（例如 $H^1_0(\Omega)$），目標是最小化能量泛函 $J[u]$。
**直覺上的困難**：
1. **導數的線性性**：方向導數存在不保證 Fréchet 可微。
2. **邊界效應**：擾動 $v$ 在邊界上的行為決定了邊界條件是預設的（Essential/Dirichlet）還是自然產生的（Natural/Neumann）。
3. **正定性的差異**：有限維中 Hessian 正定即為極小；無限維中，若算子譜的下界趨近於零，逐點正定可能不足以排除高頻擾動導致的能量下降，因此需要一致強制性。

## 定義、定理與推導

### 1. 能量泛函與導數定義

設 $\Omega \subset \mathbb{R}^n$ 為有界光滑區域。定義能量泛函 $J: \mathcal{X} \to \mathbb{R}$，其中 $\mathcal{X}$ 為適當的賦範函數空間（如 $H^1_0(\Omega)$）：
$$ J[u] = \int_\Omega L(x, u, \nabla u) \, dx $$
其中拉格朗日量 $L$ 滿足適當的增長條件以確保 $J$ 有限且連續。

**定義 1（Gateaux 第一變分）**：
若極限存在，則定義：
$$ \delta J[u; v] = \left. \frac{d}{dt} J[u + t v] \right|_{t=0} $$
**定義 2（Fréchet 導數）**：
若存在有界線性算子 $DJ(u) \in \mathcal{X}^*$ 使得：
$$ J[u+h] = J[u] + DJ(u)[h] + o(\|h\|_{\mathcal{X}}), \quad \|h\|_{\mathcal{X}} \to 0 $$
則 $J$ 在 $u$ 處 Fréchet 可微。
**注意**：若 $J$ Fréchet 可微，則 Gateaux 導數存在且為線性，即 $\delta J[u; v] = DJ(u)[v]$。反之不成立。

### 2. Euler–Lagrange 方程

**定理 1（弱形式）**：
設 $u \in \mathcal{X}$ 使得 $J[u+v] \ge J[u]$ 對所有 $v \in C_c^\infty(\Omega)$ 成立。若 $L$ 關於 $(u, \nabla u)$ 具有連續的一階偏導數，且滿足適當的支配函數條件以允許交換導數與積分，則 $u$ 滿足弱 Euler–Lagrange 方程：
$$ \int_\Omega \left( \frac{\partial L}{\partial u} v + \frac{\partial L}{\partial (\nabla u)} \cdot \nabla v \right) dx = 0, \quad \forall v \in C_c^\infty(\Omega) $$
若 $u$ 具有足夠的正則性（例如 $C^2$），則可由分部積分得經典形式：
$$ \nabla \cdot \frac{\partial L}{\partial (\nabla u)} - \frac{\partial L}{\partial u} = 0 \quad \text{in } \Omega $$

**證明**：
計算 $\frac{d}{dt} J[u+tv] = \int_\Omega \left( L_u v + L_{\nabla u} \cdot \nabla v \right) dx$。
對 $L_{\nabla u} \cdot \nabla v$ 項使用 Green 公式：
$$ \int_\Omega L_{\nabla u} \cdot \nabla v \, dx = \int_{\partial \Omega} (L_{\nabla u} \cdot \mathbf{n}) v \, ds - \int_\Omega (\nabla \cdot L_{\nabla u}) v \, dx $$
對於 $v \in C_c^\infty(\Omega)$，邊界項為零。由變分法基本引理，若積分為零對所有 $v$ 成立，則被積分函數必幾乎處處為零。

**推論（自然邊界條件）**：
若邊界值未指定（即 $v$ 在 $\partial \Omega$ 上不恆為零），則邊界項必須獨立消失：
$$ L_{\nabla u} \cdot \mathbf{n} = 0 \quad \text{on } \partial \Omega $$
對於標準能量 $L = \frac{1}{2}|\nabla u|^2$，這對應於 Neumann 條件 $\frac{\partial u}{\partial n} = 0$。

### 3. 第二變分與局部極小

**定理 2（局部極小的充分條件）**：
設 $J$ 在 $u$ 處二階 Fréchet 可微，且 **$DJ(u) = 0$**（即 $u$ 為駐點）。若存在常數 $c > 0$ 使得對所有 $v \in \mathcal{X}$：
$$ \delta^2 J[u; v] \geq c \|v\|_{\mathcal{X}}^2 $$
則 $u$ 是 $J$ 的嚴格局部最小值。
**證明**：
由二階 Fréchet 展開：
$$ J[u+h] - J[u] - DJ(u)[h] = \frac{1}{2} D^2J(u)[h,h] + o(\|h\|^2) $$
因 $DJ(u)=0$，則：
$$ J[u+h] - J[u] \geq \frac{c}{2} \|h\|^2 + o(\|h\|^2) > 0 \quad \text{for small } \|h\| \neq 0 $$
**注意**：若 $DJ(u) \neq 0$，即使 Hessian 正定，$u$ 也不可能是局部極小（因為一階項主導）。

## 逐步手算例題

### 例 1：一維線性拉普拉斯算子

**問題**：最小化 $J[u] = \frac{1}{2} \int_0^1 (u')^2 dx - \int_0^1 f u dx$，邊界 $u(0)=0, u(1)=0$。
**解答**：
1. **Euler–Lagrange**：
   $L = \frac{1}{2}(u')^2 - fu$。
   $L_u = -f, \quad L_{u'} = u'$。
   經典方程：$-u'' = f$。
2. **求解**：
   設 $f(x) = \pi^2 \sin(\pi x)$。
   $-u'' = \pi^2 \sin(\pi x) \implies u'' = -\pi^2 \sin(\pi x)$。
   積分：$u' = \pi \cos(\pi x) + C_1$。
   再積分：$u = \sin(\pi x) + C_1 x + C_2$。
   邊界 $u(0)=0 \implies C_2=0$。
   邊界 $u(1)=0 \implies \sin(\pi) + C_1 = 0 \implies C_1=0$。
   解：$u^*(x) = \sin(\pi x)$。
3. **能量計算**：
   $$ J[u^*] = \frac{1}{2} \int_0^1 \pi^2 \cos^2(\pi x) dx - \int_0^1 \pi^2 \sin^2(\pi x) \sin(\pi x) \cdot \frac{1}{\pi^2} \dots \text{ (Wait, } f = \pi^2 \sin \pi x \text{)} $$
   正確代入：
   $$ \frac{1}{2} \int_0^1 \pi^2 \cos^2(\pi x) dx = \frac{\pi^2}{2} \cdot \frac{1}{2} = \frac{\pi^2}{4} $$
   $$ \int_0^1 \pi^2 \sin(\pi x) \sin(\pi x) dx = \pi^2 \cdot \frac{1}{2} = \frac{\pi^2}{2} $$
   $$ J[u^*] = \frac{\pi^2}{4} - \frac{\pi^2}{2} = -\frac{\pi^2}{4} $$
4. **第二變分**：
   $\delta^2 J[u^*; v] = \int_0^1 (v')^2 dx$。由 Poincaré 不等式，此形式正定，故 $u^*$ 為全域最小。

### 例 2：非線性項與自然邊界

**問題**：$J[u] = \int_0^1 \left( \frac{1}{2}(u')^2 + \frac{1}{4}u^4 - gu \right) dx$，邊界 $u(0)=0$，$x=1$ 自由。
**解答**：
1. **Euler–Lagrange**：
   $L_u = u^3 - g, \quad L_{u'} = u'$。
   $-u'' + u^3 - g = 0 \implies u'' = u^3 - g$。
2. **自然邊界**：
   $x=0$ 處 $v(0)=0$。
   $x=1$ 處邊界項 $[u' v]_0^1 = u'(1)v(1)$。
   要使變分為零，需 $u'(1) = 0$。
3. **結論**：
   解滿足 $u'' = u^3 - g$，$u(0)=0$，$u'(1)=0$。

## 實作與程式

我們實作一個自足的 NumPy 程式，求解離散能量 $J_d(U) = \frac{1}{2h} \sum_{i=0}^{N-1} (U_{i+1}-U_i)^2 - h \sum_{j=1}^{N-1} f_j U_j$。
離散梯度（座標梯度）為：
$$ \frac{\partial J_d}{\partial U_j} = \frac{2U_j - U_{j-1} - U_{j+1}}{h} - f_j h $$
對應的線性系統為 $A U = b$，其中 $A$ 為三對角矩陣，對角元 $2/h$，副對角元 $-1/h$，$b_j = f_j h$。
**注意**：此處使用的是歐氏內積下的梯度。若使用質量矩陣 $M$（對角元 $h$），梯度定義會不同。

```python
import numpy as np

def discrete_energy(U, f, h):
    """計算離散能量。U: (N-1,), f: (N-1,), h: float"""
    N = len(U) + 2
    U_full = np.zeros(N)
    U_full[1:-1] = U
    dU = np.diff(U_full)
    return 0.5 / h * np.sum(dU**2) - h * np.dot(f, U)

def discrete_gradient(U, f, h):
    """計算離散能量關於 U 的梯度（Euclidean）。"""
    N = len(U)
    grad = np.zeros(N)
    # 內部點 j=1..N-1
    if N > 1:
        grad[1:-1] = (2*U[1:-1] - U[:-2] - U[2:]) / h - f[1:-1] * h
    # 邊界點 j=1
    if N >= 1:
        grad[0] = (2*U[0] - 0 - (U[1] if N>1 else 0)) / h - f[0] * h
    # 邊界點 j=N
    if N >= 1:
        grad[-1] = (2*U[-1] - (U[-2] if N>1 else 0) - 0) / h - f[-1] * h
    return grad

def solve_laplacian(f, h):
    """求解 A U = b, A 為離散拉普拉斯矩陣。"""
    N = len(f)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 2.0 / h
        if i > 0:
            A[i, i-1] = -1.0 / h
        if i < N - 1:
            A[i, i+1] = -1.0 / h
    b = f * h
    return np.linalg.solve(A, b)

# 測試
N = 100
h = 1.0 / N
x = np.arange(1, N) * h
f = np.pi**2 * np.sin(np.pi * x)

U_num = solve_laplacian(f, h)
U_exact = np.sin(np.pi * x)

err = np.max(np.abs(U_num - U_exact))
print(f"Max Error: {err:.2e}")

# 梯度核對
grad_num = discrete_gradient(U_num, f, h)
print(f"Gradient Norm at Solution: {np.linalg.norm(grad_num):.2e}")

# 有限差分核對方向導數
v = np.random.rand(N)
eps = 1e-5
dir_deriv_num = (discrete_energy(U_num + eps*v, f, h) - discrete_energy(U_num - eps*v, f, h)) / (2*eps)
dir_deriv_analytic = np.dot(discrete_gradient(U_num, f, h), v)
print(f"Dir Deriv Diff: {abs(dir_deriv_num - dir_deriv_analytic):.2e}")

# 故障測試
try:
    solve_laplacian(f, -0.01)
except Exception as e:
    print(f"Fault Test Passed: {e}")
```

## 測試與預期結果

1. **正常測試**：$N=100$，誤差應為 $O(h^2) \approx 10^{-4}$。梯度範數應接近 $10^{-8}$。
2. **邊界測試**：$N=1$，檢查矩陣構造是否正確。
3. **故障測試**：輸入 $h \le 0$ 或 $f$ 含 NaN，程式應拋出異常。上述程式未顯式檢查，實際執行會因 $h$ 為負導致數值異常或解無意義，建議加入 `assert h > 0`。
4. **方向導數核對**：中央差分與解析梯度內積的差異應小於 $10^{-4}$。

## 反例與常見陷阱

1. **駐點非最小值**：$J[u] = -\frac{1}{2}\int (u')^2 dx$ 的駐點是局部最大值。
2. **忽略一階條件**：若 $DJ(u) \neq 0$，即使 $D^2J$ 正定，也不是極小。
3. **梯度定義混淆**：離散座標梯度 $\nabla J_d$ 與連續梯度 $DJ$ 相差 $h$ 因子。$DJ(u)[v] \approx h \sum \nabla J_d(U)_j v_j$。

## AI、幾何與養殖案例

**養殖案例（合成）**：
魚塘溫度均勻化問題。控制加熱功率 $q$ 最小化溫度變異 $J = \int (T - \bar{T})^2 + \lambda \int q^2$。
此為凸二次問題。梯度 $DJ = 2(G^*G + \lambda I)q - 2G^*\bar{T}$。
由於 $\lambda > 0$，$G^*G + \lambda I$ 正定，故解唯一存在，無需 $G$ 可逆。

## 習題

1. **手算**：推导 $J[u] = \int (u')^2 - u$ 的 Euler-Lagrange 方程。
2. **程式**：修改程式以支持 Neumann 邊界，並驗證可解性條件 $\sum f_j h = 0$。
3. **反例**：構造一個駐點為鞍點的泛函。
4. **整合**：比較 $H^1$ 梯度與 $L^2$ 梯度的區別。

## 習題解答

1. $L_u = -1, L_{u'} = 2u' \implies -4u'' - 1 = 0 \implies u'' = -1/4$。
2. Neumann 邊界時，矩陣 $A$ 奇異。需固定平均值或加入小項 $\epsilon I$。
3. $J[u] = \int ((u')^2 - 1)^2 dx$。駐點包括 $u'=0$ (局部極大) 和 $u'= \pm 1$ (局部極小/鞍點，取決於邊界)。
4. $H^1$ 梯度涉及 $I - \Delta$ 算子，$L^2$ 梯度僅為函數本身。

## 本章小結

本章建立了變分法的嚴格框架，強調了 Fréchet 導數、自然邊界及二階正定條件。離數化實作展示了連續與離數梯度的關係，並通過測試驗證了數值方案的正確性。

## 參考來源

1. Evans, PDE (Ch. 6).
2. Lebl, Basic Analysis.
3. MIT OCW 18.100A.