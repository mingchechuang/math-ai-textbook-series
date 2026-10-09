<<<PATCH 01>>>
<<<OLD>>>
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
<<<NEW>>>
若合成場 $x \mapsto L_{\nabla u}(x,u(x),\nabla u(x))$ 屬於 $C^1$（例如 $u \in C^2$ 且 $L$ 具有足夠光滑性），則可由分部積分得經典形式：
$$ \nabla \cdot \frac{\partial L}{\partial (\nabla u)} - \frac{\partial L}{\partial u} = 0 \quad \text{in } \Omega $$

**證明**：
計算 $\frac{d}{dt} J[u+tv] = \int_\Omega \left( L_u v + L_{\nabla u} \cdot \nabla v \right) dx$。
對 $L_{\nabla u} \cdot \nabla v$ 項使用 Green 公式：
$$ \int_\Omega L_{\nabla u} \cdot \nabla v \, dx = \int_{\partial \Omega} (L_{\nabla u} \cdot \mathbf{n}) v \, ds - \int_\Omega (\nabla \cdot L_{\nabla u}) v \, dx $$
對於 $v \in C_c^\infty(\Omega)$，邊界項為零。由變分法基本引理，若積分為零對所有 $v$ 成立，則被積分函數必幾乎處處為零。

**推論（自然邊界條件）**：
將邊界分為 $\partial\Omega=\Gamma_D\cup\Gamma_N$，其中在 $\Gamma_D$ 上 $v=0$，在 $\Gamma_N$ 上 $v$ 任意。若 $\Gamma_N$ 上邊界值未指定，則邊界項必須獨立消失：
$$ L_{\nabla u} \cdot \mathbf{n} = 0 \quad \text{on } \Gamma_N. $$
對於標準能量 $L = \frac{1}{2}|\nabla u|^2$，在 $\Gamma_N$ 上對應 Neumann 條件 $\frac{\partial u}{\partial n} = 0$。若能量含非齊次邊界項，自然條件須由總變分一併推出。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
3. **能量計算**：
   $$ J[u^*] = \frac{1}{2} \int_0^1 \pi^2 \cos^2(\pi x) dx - \int_0^1 \pi^2 \sin^2(\pi x) \sin(\pi x) \cdot \frac{1}{\pi^2} \dots \text{ (Wait, } f = \pi^2 \sin \pi x \text{)} $$
   正確代入：
   $$ \frac{1}{2} \int_0^1 \pi^2 \cos^2(\pi x) dx = \frac{\pi^2}{2} \cdot \frac{1}{2} = \frac{\pi^2}{4} $$
   $$ \int_0^1 \pi^2 \sin(\pi x) \sin(\pi x) dx = \pi^2 \cdot \frac{1}{2} = \frac{\pi^2}{2} $$
   $$ J[u^*] = \frac{\pi^2}{4} - \frac{\pi^2}{2} = -\frac{\pi^2}{4} $$
<<<NEW>>>
3. **能量計算**：
   因 $u^*(x)=\sin(\pi x)$，$u^{*\prime}(x)=\pi\cos(\pi x)$，故
   $$ \frac{1}{2} \int_0^1 (u^{*\prime})^2 dx = \frac{1}{2} \int_0^1 \pi^2 \cos^2(\pi x) dx = \frac{\pi^2}{4}. $$
   又 $f(x)=\pi^2\sin(\pi x)$，所以
   $$ \int_0^1 f u^* dx = \int_0^1 \pi^2 \sin^2(\pi x) dx = \frac{\pi^2}{2}. $$
   因此
   $$ J[u^*] = \frac{\pi^2}{4} - \frac{\pi^2}{2} = -\frac{\pi^2}{4}. $$
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
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
<<<NEW>>>
def _validate_common(U, f, h):
    U = np.asarray(U, dtype=float)
    f = np.asarray(f, dtype=float)
    if U.ndim != 1 or f.ndim != 1:
        raise ValueError("U 與 f 必須是一維陣列")
    if U.shape != f.shape:
        raise ValueError("U 與 f 的形狀必須相同")
    if not np.isscalar(h) or h <= 0:
        raise ValueError("h 必須為正數")
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(f))):
        raise ValueError("U 與 f 必須是有限值")
    return U, f

def discrete_energy(U, f, h):
    """計算離散能量。U: (N-1,), f: (N-1,), h: float"""
    U, f = _validate_common(U, f, h)
    N = len(U) + 2
    U_full = np.zeros(N)
    U_full[1:-1] = U
    dU = np.diff(U_full)
    return 0.5 / h * np.sum(dU**2) - h * np.dot(f, U)

def discrete_gradient(U, f, h):
    """計算離散能量關於 U 的梯度（Euclidean）。"""
    U, f = _validate_common(U, f, h)
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
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("f 必須是一維陣列")
    if not np.isscalar(h) or h <= 0:
        raise ValueError("h 必須為正數")
    if not np.all(np.isfinite(f)):
        raise ValueError("f 必須是有限值")
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
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
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
<<<NEW>>>
# 測試
num_intervals = 100
h = 1.0 / num_intervals
x = np.arange(1, num_intervals) * h
f = np.pi**2 * np.sin(np.pi * x)
num_unknowns = len(f)

U_num = solve_laplacian(f, h)
U_exact = np.sin(np.pi * x)

err = np.max(np.abs(U_num - U_exact))
print(f"Max Error: {err:.2e}")

# 梯度核對
grad_num = discrete_gradient(U_num, f, h)
print(f"Gradient Norm at Solution: {np.linalg.norm(grad_num):.2e}")

# 有限差分核對方向導數
rng = np.random.default_rng(0)
v = rng.standard_normal(U_num.shape)
eps = 1e-5
dir_deriv_num = (discrete_energy(U_num + eps*v, f, h) - discrete_energy(U_num - eps*v, f, h)) / (2*eps)
dir_deriv_analytic = np.dot(discrete_gradient(U_num, f, h), v)
print(f"Dir Deriv Diff: {abs(dir_deriv_num - dir_deriv_analytic):.2e}")

# 故障測試：h<=0 應拋出 ValueError
try:
    solve_laplacian(f, -0.01)
except ValueError as e:
    print(f"Fault Test Passed: {e}")
else:
    raise AssertionError("h<=0 時 solve_laplacian 應拋出 ValueError")

# 邊界測試：一個內點（num_intervals=2, num_unknowns=1）
h_bd = 1.0 / 2
f_bd = np.array([1.0])
U_bd = solve_laplacian(f_bd, h_bd)
assert np.allclose(U_bd, f_bd * h_bd**2 / 2), U_bd
print("Boundary Test Passed")

# 相場雙井勢能量與離散方向導數核對
def double_well_energy(U, lam=1.0):
    """離散相場雙井勢能量：sum (U_i^2 - 1)^2。"""
    U = np.asarray(U, dtype=float)
    return lam * np.sum((U**2 - 1.0)**2)

def double_well_gradient(U, lam=1.0):
    U = np.asarray(U, dtype=float)
    return 4.0 * lam * U * (U**2 - 1.0)

U_dw = rng.standard_normal(10)
v_dw = rng.standard_normal(U_dw.shape)
eps_dw = 1e-6
dir_dw_num = (double_well_energy(U_dw + eps_dw*v_dw) - double_well_energy(U_dw - eps_dw*v_dw)) / (2*eps_dw)
dir_dw_analytic = np.dot(double_well_gradient(U_dw), v_dw)
print(f"Double-Well Dir Deriv Diff: {abs(dir_dw_num - dir_dw_analytic):.2e}")
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
3. **梯度定義混淆**：離散座標梯度 $\nabla J_d$ 與連續梯度 $DJ$ 相差 $h$ 因子。$DJ(u)[v] \approx h \sum \nabla J_d(U)_j v_j$。
<<<NEW>>>
3. **梯度定義混淆**：離散座標梯度已滿足 $DJ_d(U)[V]=\nabla_{\mathrm E}J_d(U)^T V$；若以質量矩陣 $M$ 定義離散梯度，則 $\nabla_M J_d = M^{-1}\nabla_{\mathrm E}J_d$。均勻網格 $M\approx hI$，故 $\nabla_{\mathrm E}J_d \approx h g$，不能再額外乘 $h$。
<<<END>>>
<<<PATCH 06>>>
<<<OLD>>>
1. $L_u = -1, L_{u'} = 2u' \implies -4u'' - 1 = 0 \implies u'' = -1/4$。
2. Neumann 邊界時，矩陣 $A$ 奇異。需固定平均值或加入小項 $\epsilon I$。
3. $J[u] = \int ((u')^2 - 1)^2 dx$。駐點包括 $u'=0$ (局部極大) 和 $u'= \pm 1$ (局部極小/鞍點，取決於邊界)。
4. $H^1$ 梯度涉及 $I - \Delta$ 算子，$L^2$ 梯度僅為函數本身。

## 本章小結

本章建立了變分法的嚴格框架，強調了 Fréchet 導數、自然邊界及二階正定條件。離數化實作展示了連續與離數梯度的關係，並通過測試驗證了數值方案的正確性。
<<<NEW>>>
1. $L_u=-1$，$L_{u'}=2u'$。Euler–Lagrange 方程為 $L_u-\frac{d}{dx}L_{u'}=0$，即 $-1-2u''=0$，故 $u''=-\frac12$。未指定邊界條件時，只能得到微分方程，不能唯一決定 $u$。
2. Neumann 邊界離散：在 $U_0=U_1$、$U_N=U_{N-1}$ 下，內部方程為 $(2U_j-U_{j-1}-U_{j+1})/h=f_jh$。對 $j=1$ 得 $(U_1-U_2)/h=f_1h$，對 $j=N-1$ 得 $(U_{N-1}-U_{N-2})/h=f_{N-1}h$。矩陣 $A_N$ 對角 $1/h$、副對角 $-1/h$。可解性條件 $\sum_{j=1}^{N-1} f_j h=0$。求解時固定 $U_1=0$ 或加入零平均約束 $\sum U_j=0$，再核對殘差與常數零空間；直接加入 $\epsilon I$ 會改變原問題，不等同於單純固定零空間。
3. 鞍點例：$J[u]=\int_\Omega((u_x)^2-(u_y)^2)dxdy$，在 $u=0$ 且 $u|_{\partial\Omega}=0$。第二變分 $\delta^2J[0;v]=\int(v_x^2-v_y^2)$。取 $v=\sin(\pi x)$ 得正值；取 $v=\sin(\pi y)$ 得負值。故 $0$ 是鞍點。
4. 指定 $J[u]=\frac12\int_\Omega|\nabla u|^2-\int fu$，$u\in H_0^1$。$DJ(u)[v]=\int\nabla u\cdot\nabla v-\int fv$。$L^2$ 梯度 $g_{L^2}$ 為 $-\Delta u-f$（弱意義）。$H^1$ 內積 $(v,w)_{H^1}=\int(\nabla v\cdot\nabla w+vw)$，$g_{H^1}$ 滿足 $(g_{H^1},v)_{H^1}=DJ(u)[v]$，即 $(I-\Delta)g_{H^1}=g_{L^2}$，故 $g_{H^1}=(I-\Delta)^{-1}g_{L^2}$。不能籠統說 $H^1$ 梯度就是 $I-\Delta$。

## 本章小結

本章建立了變分法的嚴格框架，強調了 Fréchet 導數、自然邊界及二階正定條件。離散化實作設計了連續與離散梯度的數值核對；所有數值陳述為預期結果，須在執行程式後才能作為驗證證據。
<<<END>>>