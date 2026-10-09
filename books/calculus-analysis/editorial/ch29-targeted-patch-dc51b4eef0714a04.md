<<<PATCH 29>>>
<<<OLD>>>
若 $u$ 具有足夠的正則性（例如 $C^2$），則可由分部積分得經典形式：
<<<NEW>>>
若合成向量場 $x\mapsto L_{\nabla u}(x,u(x),\nabla u(x))$ 屬於 $C^1(\Omega)$，則可由分部積分得經典形式：
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
   $$ J[u^*] = \frac{1}{2} \int_0^1 \pi^2 \cos^2(\pi x) dx - \int_0^1 \pi^2 \sin^2(\pi x) \sin(\pi x) \cdot \frac{1}{\pi^2} \dots \text{ (Wait, } f = \pi^2 \sin \pi x \text{)} $$
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

1. **正常測試**：$N=100$，誤差應為 $O(h^2) \approx 10^{-4}$。梯度範數應接近 $10^{-8}$。
2. **邊界測試**：$N=1$，檢查矩陣構造是否正確。
3. **故障測試**：輸入 $h \le 0$ 或 $f$ 含 NaN，程式應拋出異常。上述程式未顯式檢查，實際執行會因 $h$ 為負導致數值異常或解無意義，建議加入 `assert h > 0`。
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

此處 $N$ 表示子區間數，內點數為 $N-1$；求解器輸入向量長度另記為 $n$。對 $N=100$，最大誤差預期為 $O(h^2)$、量級約 $10^{-4}$，這是中心差分截斷誤差的推論，不是已執行結果。線性系統求解後梯度範數預期接近浮點捨入誤差量級，實際值取決於矩陣條件數與求解器。

邊界測試取一個內點、兩個子區間，即 $n=1$、$h=1/2$。矩陣為 $[2/h]=[4]$，右端為 $hf=3/2$，故 $U_1=3/8=fh^2/2$。故障測試要求負步長與含 NaN 輸入都引發 `ValueError`；若未拒絕輸入，`AssertionError` 會使測試失敗。方向導數測試比較中心差分與解析座標梯度內積；有限差分核對只能檢查實作，不能代替解析推導。

Euclidean 座標梯度滿足
$$ DJ_d(U)[V]=\nabla_{\mathrm E}J_d(U)^TV. $$
若以質量矩陣 $M=hI$ 定義梯度，則
$$ \nabla_MJ_d=M^{-1}\nabla_{\mathrm E}J_d. $$
在與連續 $L^2$ 內積一致的離散化下，方向導數近似 $h\sum_jg_jV_j$，Euclidean 座標梯度約為 $hg_j$；不可對座標梯度額外再乘一次 $h$。
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
1. **手算**：推导 $J[u] = \int (u')^2 - u$ 的 Euler-Lagrange 方程。
2. **程式**：修改程式以支持 Neumann 邊界，並驗證可解性條件 $\sum f_j h = 0$。
3. **反例**：構造一個駐點為鞍點的泛函。
4. **整合**：比較 $H^1$ 梯度與 $L^2$ 梯度的區別。

## 習題解答

1. $L_u = -1, L_{u'} = 2u' \implies -4u'' - 1 = 0 \implies u'' = -1/4$。
2. Neumann 邊界時，矩陣 $A$ 奇異。需固定平均值或加入小項 $\epsilon I$。
3. $J[u] = \int ((u')^2 - 1)^2 dx$。駐點包括 $u'=0$ (局部極大) 和 $u'= \\pm 1$ (局部極小/鞍點，取決於邊界)。
4. $H^1$ 梯度涉及 $I - \\Delta$ 算子，$L^2$ 梯度僅為函數本身。
<<<NEW>>>
1. **手算**：推导 $J[u] = \int_0^1 ((u')^2-u)\,dx$ 的 Euler–Lagrange 方程。若未指定邊界條件，說明為何不能唯一決定 $u$。
2. **程式**：修改程式以支持 Neumann 邊界，檢查離散相容條件，並以固定一個節點消除常數零空間。
3. **反例**：構造一個駐點為鞍點的泛函，給出第二變分分別為正與負的容許擾動。
4. **整合**：對同一線性泛函指定 $L^2$ 與 $H^1$ 內積，推導兩種梯度的關係。

## 習題解答

1. 令 $L=(u')^2-u$，則 $L_u=-1$、$L_{u'}=2u'$，故 Euler–Lagrange 方程為 $-1-2u''=0$，即 $u''=-1/2$。一般解為 $u(x)=-x^2/4+C_1x+C_2$；沒有邊界條件時，$C_1,C_2$ 未定。
2. 在 $[0,1]$ 取均勻網格，$h=1/N$、節點值 $U_0,\ldots,U_N$，定義梯形權重 $w_0=w_N=1/2$、其餘 $w_i=1$。離散能量及其駐點方程為
   $$ J_N(U)=\frac{1}{2h}\sum_{i=0}^{N-1}(U_{i+1}-U_i)^2-h\sum_{i=0}^Nw_if_iU_i,\qquad \frac1hD^TD\,U=hWf. $$
   因 $D\mathbf1=0$，方程可解的必要條件為 $h\sum_iw_if_i=0$。先檢查此相容條件，再固定 $U_0=0$ 解去除常數自由度後的系統；這是選定代表元，不是加入 $\epsilon I$ 修改原問題。最後檢查方程殘差，並確認未固定系統的常數方向相容。
3. 在 $\Omega=(0,1)^2$、容許空間 $H^1_0(\Omega)$ 上取 $J[u]=\frac12\int_\Omega((u_x)^2-(u_y)^2)\,dx\,dy$。$u=0$ 是駐點，第二變分為 $\int_\Omega((v_x)^2-(v_y)^2)\,dx\,dy$。取 $v_+=\sin(\pi x)\sin(2\pi y)$，有 $\|(v_+)_y\|_2^2=4\|(v_+)_x\|_2^2$，故第二變分為負；取 $v_-=\sin(2\pi x)\sin(\pi y)$，則 $\|(v_-)_x\|_2^2=4\|(v_-)_y\|_2^2$，故第二變分為正。任意小倍數的這兩種擾動分別使能量下降與上升，證明 $u=0$ 為鞍點。
4. 對 $\ell(v)=\int_0^1gv\,dx$，$L^2$ 內積下的梯度為 $g$。採用 $H^1(0,1)$ 內積 $\langle z,v\rangle_{H^1}=\int_0^1(zv+z'v')\,dx$，其 Riesz 代表元 $z$ 滿足 $\int_0^1(zv+z'v')\,dx=\int_0^1gv\,dx$。分部積分得到弱方程 $z-z''=g$ 及自然邊界條件 $z'(0)=z'(1)=0$；故在此內積與邊界條件下 $(I-\Delta)z=g$。這不是對任意泛函都成立的梯度公式，而是由指定內積與線性泛函導出的表示。
<<<END>>>