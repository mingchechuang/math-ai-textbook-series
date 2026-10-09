<<<PATCH 29>>>
<<<OLD>>>
若 $u$ 具有足夠的正則性（例如 $C^2$），則可由分部積分得經典形式：
<<<NEW>>>
若合成向量場 $x\mapsto L_{\nabla u}(x,u(x),\nabla u(x))$ 屬於 $C^1(\Omega)$，則可由分部積分得經典形式：
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
若邊界值未指定（即 $v$ 在 $\\partial \\Omega$ 上不恆為零），則邊界項必須獨立消失：
<<<NEW>>>
將邊界分為固定部分 $\Gamma_D$ 與自由部分 $\Gamma_N$。若容許擾動在 $\Gamma_D$ 上的跡為零、在 $\Gamma_N$ 上可任意取值，則邊界項只須在自由部分消失：
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
$$ L_{\\nabla u} \\cdot \\mathbf{n} = 0 \\quad \\text{on } \\partial \\Omega $$
<<<NEW>>>
$$ L_{\\nabla u} \\cdot \\mathbf{n} = 0 \\quad \\text{on } \\Gamma_N $$
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
   $$ J[u^*] = \\frac{1}{2} \\int_0^1 \\pi^2 \\cos^2(\\pi x) dx - \\int_0^1 \\pi^2 \\sin^2(\\pi x) \\sin(\\pi x) \\cdot \\frac{1}{\\pi^2} \\dots \\text{ (Wait, } f = \\pi^2 \\sin \\pi x \\text{)} $$
<<<NEW>>>
<<<END>>>
<<<PATCH 29>>>
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
def _validate(U, f, h):
    U = np.asarray(U, dtype=float)
    f = np.asarray(f, dtype=float)
    if U.ndim != 1 or f.ndim != 1 or U.shape != f.shape:
        raise ValueError("U 與 f 必須是形狀相同的一維陣列")
    if not np.isfinite(U).all() or not np.isfinite(f).all():
        raise ValueError("U 與 f 必須全為有限值")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h 必須是有限正數")
    return U, f

def discrete_energy(U, f, h):
    """計算零 Dirichlet 邊界下的離散能量。"""
    U, f = _validate(U, f, h)
    U_full = np.zeros(len(U) + 2)
    U_full[1:-1] = U
    dU = np.diff(U_full)
    return 0.5 / h * np.sum(dU**2) - h * np.dot(f, U)

def discrete_gradient(U, f, h):
    """離散能量對內點座標的 Euclidean 梯度。"""
    U, f = _validate(U, f, h)
    grad = np.empty_like(U)
    if len(U) == 0:
        return grad
    U_full = np.zeros(len(U) + 2)
    U_full[1:-1] = U
    grad[:] = (2 * U - U_full[:-2] - U_full[2:]) / h - h * f
    return grad

def solve_laplacian(f, h):
    """零 Dirichlet 邊界；f 含每個內點的值，h 為網格間距。"""
    f = np.asarray(f, dtype=float)
    if f.ndim != 1 or not np.isfinite(f).all():
        raise ValueError("f 必須是一維有限值陣列")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h 必須是有限正數")
    n = len(f)
    if n == 0:
        return np.empty(0, dtype=float)
    A = np.diag(np.full(n, 2.0 / h))
    if n > 1:
        A += np.diag(np.full(n - 1, -1.0 / h), 1)
        A += np.diag(np.full(n - 1, -1.0 / h), -1)
    return np.linalg.solve(A, h * f)
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
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

1. **正常測試**：$N=100$，誤差應為 $O(h^2) \\approx 10^{-4}$。梯度範數應接近 $10^{-8}$。
2. **邊界測試**：$N=1$，檢查矩陣構造是否正確。
3. **故障測試**：輸入 $h \\le 0$ 或 $f$ 含 NaN，程式應拋出異常。上述程式未顯式檢查，實際執行會因 $h$ 為負導致數值異常或解無意義，建議加入 `assert h > 0`。
4. **方向導數核對**：中央差分與解析梯度內積的差異應小於 $10^{-4}$。
<<<NEW>>>
# 有限差分核對方向導數；固定亂數種子使測試可重現
rng = np.random.default_rng(0)
v = rng.standard_normal(U_num.shape)
eps = 1e-5
dir_deriv_num = (discrete_energy(U_num + eps*v, f, h) - discrete_energy(U_num - eps*v, f, h)) / (2*eps)
dir_deriv_analytic = np.dot(discrete_gradient(U_num, f, h), v)
print(f"Dir Deriv Diff: {abs(dir_deriv_num - dir_deriv_analytic):.2e}")

# 故障測試
for bad_call in (lambda: solve_laplacian(f, -0.01),
                 lambda: solve_laplacian(np.full_like(f, np.nan), h)):
    try:
        bad_call()
    except ValueError:
        pass
    else:
        raise AssertionError("故障輸入未觸發 ValueError")

# 一個內點、兩個子區間的邊界測試
U_one = solve_laplacian(np.array([3.0]), 0.5)
assert np.allclose(U_one, [3.0 * 0.5**2 / 2])
assert discrete_gradient(U_one, np.array([3.0]), 0.5).shape == (1,)
```

## 測試與預期結果

此處 $N$ 表示子區間數，內點數為 $N-1$；求解器收到的內點向量長度另記為 $n$。對 $N=100$ 的網格，最大誤差預期為 $O(h^2)$、量級約 $10^{-4}$；這是中心差分截斷誤差的推論，不是已執行結果。線性系統求解後梯度範數預期接近浮點捨入誤差量級，實際值取決於矩陣條件數與求解器。

邊界測試取一個內點、兩個子區間，即 $n=1$、$h=1/2$。矩陣為 $[2/h]=[4]$，右端為 $hf=3/2$，故 $U_1=3/8=fh^2/2$。故障測試要求負步長與含 NaN 輸入都引發 `ValueError`；若未拒絕輸入，`AssertionError` 會使測試失敗。方向導數測試比較中心差分與解析座標梯度內積；有限差分核對只能檢查實作，不能代替解析推導。

Euclidean 座標梯度滿足
$$ DJ_d(U)[V]=\nabla_{\mathrm E}J_d(U)^TV. $$
若以質量矩陣 $M=hI$ 定義梯度，則
$$ \nabla_MJ_d=M^{-1}\nabla_{\mathrm E}J_d. $$
在與連續 $L^2$ 內積一致的離散化下，方向導數近似 $h\sum_jg_jV_j$，Euclidean 座標梯度約為 $hg_j$；不可對座標梯度額外再乘一次 $h$。
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
## 反例與常見陷阱

1. **駐點非最小值**：$J[u] = -\\frac{1}{2}\\int (u')^2 dx$ 的駐點是局部最大值。
2. **忽略一階條件**：若 $DJ(u) \\neq 0$，即使 $D^2J$ 正定，也不是極小。
3. **梯度定義混淆**：離散座標梯度 $\\nabla J_d$ 與連續梯度 $DJ$ 相差 $h$ 因子。$DJ(u)[v] \\approx h \\sum \\nabla J_d(U)_j v_j$。
<<<NEW>>>
## 反例與常見陷阱

1. **駐點非最小值**：在固定零邊界的容許空間上，$J[u] = -\\frac{1}{2}\\int (u')^2 dx$ 的 $u=0$ 是嚴格局部最大值。
2. **忽略一階條件**：若 $DJ(u) \\neq 0$，即使 $D^2J$ 正定，也不是極小。
3. **梯度定義混淆**：離散座標梯度與連續 $L^2$ 梯度不是同一種表示。Euclidean 座標梯度滿足 $DJ_d(U)[V]=\\nabla_{\\mathrm E}J_d(U)^TV$；若質量矩陣 $M=hI$，則 $\\nabla_MJ_d=M^{-1}\\nabla_{\\mathrm E}J_d$。
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
本章建立了變分法的嚴格框架，強調了 Fréchet 導數、自然邊界及二階正定條件。離數化實作展示了連續與離數梯度的關係，並通過測試驗證了數值方案的正確性。
<<<NEW>>>
本章建立變分法的基本框架，說明第一變分、Euler–Lagrange 方程、自然邊界與二階強制性判準。離散例展示了座標梯度、質量矩陣梯度及方向導數核對的關係；所列測試是程式的預期檢查，並非已執行或已驗證的結果。
<<<END>>>