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
對於標準能量 $L = \frac{1}{2}| \nabla u|^2$，這對應於 Neumann 條件 $\frac{\partial u}{\partial n} = 0$。
<<<NEW>>>
若合成向量場 $x\mapsto L_{\nabla u}(x,u(x),\nabla u(x))$ 屬於 $C^1(\Omega)$，則可將弱方程以分部積分寫成經典形式：
$$ \nabla \cdot \frac{\partial L}{\partial (\nabla u)} - \frac{\partial L}{\partial u} = 0 \quad \text{in } \Omega $$

**證明**：
在可交換微分與積分的支配條件下，計算 $\frac{d}{dt} J[u+tv] = \int_\Omega \left( L_u v + L_{\nabla u} \cdot \nabla v \right) dx$。
對 $L_{\nabla u} \cdot \nabla v$ 項使用分部積分：
$$ \int_\Omega L_{\nabla u} \cdot \nabla v \, dx = \int_{\partial \Omega} (L_{\nabla u} \cdot \mathbf{n}) v \, ds - \int_\Omega (\nabla \cdot L_{\nabla u}) v \, dx $$
對於 $v \in C_c^\infty(\Omega)$，邊界項為零。由變分法基本引理，若積分為零對所有 $v$ 成立，則被積分函數幾乎處處為零；若該函數連續，則在 $\Omega$ 中處處為零。

**推論（自然邊界條件）**：
將邊界分為固定部分 $\Gamma_D$ 與自由部分 $\Gamma_N$，並假設容許擾動在 $\Gamma_D$ 上的跡為零、在 $\Gamma_N$ 上可任意取值。若第一變分對所有此類擾動為零，則邊界項只在自由邊界上要求消失：
$$ L_{\nabla u} \cdot \mathbf{n} = 0 \quad \text{on } \Gamma_N $$
對於標準能量 $L = \frac{1}{2}| \nabla u|^2$，此條件對應於 $\frac{\partial u}{\partial n} = 0$。若能量另含邊界項或外加邊界功，所得自然通量條件則須連同相應邊界項推導，不能直接套用零通量結論。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
   $$ J[u^*] = \frac{1}{2} \int_0^1 \pi^2 \cos^2(\pi x) dx - \int_0^1 \pi^2 \sin^2(\pi x) \sin(\pi x) \cdot \frac{1}{\pi^2} \dots \text{ (Wait, } f = \pi^2 \sin \pi x \text{)} $$
   正確代入：
<<<NEW>>>
   正確代入：
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
<<<PATCH 04>>>
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

# 故障測試：驗證非法步長及非有限輸入確實被拒絕
for bad_call in (lambda: solve_laplacian(f, -0.01),
                 lambda: solve_laplacian(np.full_like(f, np.nan), h)):
    try:
        bad_call()
    except ValueError:
        pass
    else:
        raise AssertionError("故障輸入未觸發 ValueError")

# 邊界測試：一個內點對應兩個子區間
U_one = solve_laplacian(np.array([3.0]), 0.5)
assert np.allclose(U_one, [3.0 * 0.5**2 / 2])
assert discrete_gradient(U_one, np.array([3.0]), 0.5).shape == (1,)
```

## 測試與預期結果

此處 $N$ 表示子區間數，內點數為 $N-1$；求解器收到的是內點向量，長度記為 $n$。對 $N=100$ 的網格，離散解近似 $\sin(\pi x)$，最大誤差預期為 $O(h^2)$，量級約 $10^{-4}$；這是由中心差分截斷誤差預期，不是已執行的測試結果。線性系統求解後，梯度範數應接近浮點捨入誤差量級，實際值取決於矩陣條件數與求解器。

邊界測試取一個內點、兩個子區間，即 $n=1$、$h=1/2$。矩陣為 $[2/h]=[4]$，右端為 $hf=3/2$，故 $U_1=3/8=f h^2/2$，程式斷言核對此解析值。故障測試要求負步長與含 NaN 的輸入都引發 `ValueError`；程式以 `AssertionError` 確認若未拒絕輸入便使測試失敗。方向導數測試比較中心差分與解析座標梯度內積，誤差門檻應依步長、浮點精度與向量尺度設定；有限差分核對只能檢查實作，不能代替對解析公式的推導。

離散梯度須分清所用內積。對此離散能量，Euclidean 座標梯度滿足
$$ DJ_d(U)[V]=\nabla_{\mathrm E}J_d(U)^T V. $$
若以均勻網格質量矩陣 $M=hI$ 定義梯度，則
$$ \nabla_M J_d=M^{-1}\nabla_{\mathrm E}J_d. $$
在與連續 $L^2$ 內積一致的離散化下，方向導數近似為 $h\sum_j g_jV_j$，而 Euclidean 座標梯度約為 $h g_j$；不可把已是座標梯度的量再額外乘一次 $h$。
<<<END>>>
<<<PATCH 05>>>
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
1. **手算**：推导 $J[u] = \int_0^1 ((u')^2-u)\,dx$ 的 Euler–Lagrange 方程。說明若未指定邊界條件，方程只能決定解的二階導數，不能唯一決定 $u$。
2. **程式**：修改程式以支持 Neumann 邊界，檢查離散相容條件，並以固定平均值消除常數零空間。
3. **反例**：構造一個駐點為鞍點的泛函，明確給出兩個容許擾動，使第二變分一正一負。
4. **整合**：對同一個具體泛函，指定 $L^2$ 與 $H^1$ 內積，推導兩種梯度的關係。

## 習題解答

1. 取 $L=(u')^2-u$，則 $L_u=-1$、$L_{u'}=2u'$。Euler–Lagrange 方程為 $L_u-\frac{d}{dx}L_{u'}=-1-2u''=0$，因此 $u''=-1/2$。其一般解為 $u(x)=-x^2/4+C_1x+C_2$；沒有邊界條件時，常數 $C_1,C_2$ 未定，不能宣稱解唯一。
2. 考慮區間 $[0,1]$ 的均勻網格，令 $h=1/N$，節點值為 $U_0,\ldots,U_N$。Neumann 離散能量取
   $$ J_N(U)=\frac{1}{2h}\sum_{i=0}^{N-1}(U_{i+1}-U_i)^2-h\sum_{i=0}^{N}w_i f_iU_i, $$
   其中梯形權重 $w_0=w_N=1/2$，其餘 $w_i=1$。令 $D$ 為差分矩陣，則平穩方程為
   $$ \frac1hD^TD\,U=hWf, $$
   其中 $W=\operatorname{diag}(w_i)$。常數向量 $\mathbf1$ 滿足 $D\mathbf1=0$，故矩陣有常數零空間；左乘 $\mathbf1^T$ 得相容條件 $h\sum_iw_if_i=0$。若相容，取 $U_0=0$ 固定規範後解 $U_1,\ldots,U_N$，再將解減去其加權平均以得到零平均代表。這是移除規範自由度，不是加入 $\epsilon I$ 改變原方程。求得解後檢查自由節點殘差 $\|(D^TD/h)U-hWf\|$；原系統未固定的常數方向殘差亦應為零。以相容性未滿足的資料測試時，求解器應先回報不相容，而非默默產生解。
3. 在 $\Omega=(0,1)^2$、固定零邊界的容許空間 $H^1_0(\Omega)$ 上，取
   $$ J[u]=\frac12\int_\Omega\big((u_x)^2-(u_y)^2\big)\,dx\,dy. $$
   $u=0$ 是駐點，且第二變分為 $\delta^2J[0;v]=\int_\Omega((v_x)^2-(v_y)^2)\,dx\,dy$。令 $v_+(x,y)=\sin(\pi x)\sin(2\pi y)$，則 $\| (v_+)_y\|_2^2=4\|(v_+)_x\|_2^2$，第二變分為負；令 $v_-(x,y)=\sin(2\pi x)\sin(\pi y)$，則 $\|(v_-)_x\|_2^2=4\|(v_-)_y\|_2^2$，第二變分為正。因此任意小的正、負倍數擾動都能使能量分別上升、下降，$u=0$ 是鞍點。
4. 取 $J[u]=\frac12\int_0^1(u^2+(u')^2)\,dx$，在 $H^1(0,1)$ 上使用內積 $\langle a,b\rangle_{H^1}=\int_0^1(ab+a'b')\,dx$，而 $L^2$ 內積為 $\langle a,b\rangle_{L^2}=\int_0^1ab\,dx$。對任意擾動 $v$，$DJ(u)[v]=\int_0^1(uv+u'v')\,dx$，故此泛函的 $H^1$ 梯度正是 $u$。若將同一個線性泛函 $\ell(v)=\int_0^1gv\,dx$ 分別用兩種內積表示，則其 $L^2$ 梯度為 $g$；其 $H^1$ 梯度 $z$ 滿足 $\int_0^1(zv+z'v')dx=\int_0^1gv\,dx$。對任意 $v\in H^1(0,1)$ 分部積分可得弱方程 $z-z''=g$，自然邊界條件為 $z'(0)=z'(1)=0$。因此在此內積與邊界條件下，$(I-\Delta)z=g$；這個關係取決於泛函、內積及其定義域，並非所有 $H^1$ 梯度都可不加條件地稱為 $I-\Delta$。
<<<END>>>