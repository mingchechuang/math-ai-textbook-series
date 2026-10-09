<<<PATCH 29>>>
<<<OLD>>>
若 $u$ 具有足夠的正則性（例如 $C^2$），則可由分部積分得經典形式：
<<<NEW>>>
若合成向量場 $x\mapsto L_{\nabla u}(x,u(x),\nabla u(x))$ 屬於 $C^1(\Omega)$，則可由分部積分得經典形式：
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
若邊界值未指定（即 $v$ 在 $\\partial \\Omega$ 上不恆為零），則邊界項必須獨立消失：
$$ L_{\\nabla u} \\cdot \\mathbf{n} = 0 \\quad \\text{on } \\partial \\Omega $$
<<<NEW>>>
將邊界分為固定部分 $\Gamma_D$ 與自由部分 $\Gamma_N$。若容許擾動在 $\Gamma_D$ 上的跡為零、在 $\Gamma_N$ 上可任意取值，則邊界項只須在自由部分消失：
$$ L_{\\nabla u} \\cdot \\mathbf{n} = 0 \\quad \\text{on } \\Gamma_N $$
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
   $$ J[u^*] = \\frac{1}{2} \\int_0^1 \\pi^2 \\cos^2(\\pi x) dx - \\int_0^1 \\pi^2 \\sin^2(\\pi x) \\sin(\\pi x) \\cdot \\frac{1}{\\pi^2} \\dots \\text{ (Wait, } f = \\pi^2 \\sin \\pi x \\text{)} $$
<<<NEW>>>
<<<END>>>
<<<PATCH 29>>>
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

# 邊界測試：一個內點、兩個子區間
U_one = solve_laplacian(np.array([3.0]), 0.5)
assert np.allclose(U_one, [3.0 * 0.5**2 / 2])
assert discrete_gradient(U_one, np.array([3.0]), 0.5).shape == (1,)
```

## 測試與預期結果

此處 $N$ 表示子區間數，內點數為 $N-1$；求解器輸入向量長度另記為 $n$。對 $N=100$ 的網格，最大誤差預期為 $O(h^2)$、量級約 $10^{-4}$；這是中心差分截斷誤差的推論，不是已執行結果。梯度範數預期接近浮點捨入誤差量級，實際值取決於矩陣條件數與求解器。

邊界測試取一個內點、兩個子區間，即 $n=1$、$h=1/2$。矩陣為 $[2/h]=[4]$，右端為 $hf=3/2$，故 $U_1=3/8=fh^2/2$。故障測試要求負步長與含 NaN 輸入都引發 `ValueError`；若未拒絕輸入，`AssertionError` 會使測試失敗。方向導數測試比較中心差分與解析座標梯度內積；有限差分核對只能檢查實作，不能代替解析推導。

Euclidean 座標梯度滿足
$$ DJ_d(U)[V]=\nabla_{\mathrm E}J_d(U)^TV. $$
若以質量矩陣 $M=hI$ 定義梯度，則
$$ \nabla_MJ_d=M^{-1}\nabla_{\mathrm E}J_d. $$
在與連續 $L^2$ 內積一致的離散化下，方向導數近似 $h\sum_jg_jV_j$，Euclidean 座標梯度約為 $hg_j$；不可對座標梯度額外再乘一次 $h$。
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
## 習題解答

1. $L_u = -1, L_{u'} = 2u' \\implies -4u'' - 1 = 0 \\implies u'' = -1/4$。
2. Neumann 邊界時，矩陣 $A$ 奇異。需固定平均值或加入小項 $\\epsilon I$。
3. $J[u] = \\int ((u')^2 - 1)^2 dx$。駐點包括 $u'=0$ (局部極大) 和 $u'= \\pm 1$ (局部極小/鞍點，取決於邊界)。
4. $H^1$ 梯度涉及 $I - \\Delta$ 算子，$L^2$ 梯度僅為函數本身。
<<<NEW>>>
## 習題解答

1. 取容許擾動在端點為零，$L(u,u')=(u')^2-u$，故 $L_u=-1$、$L_{u'}=2u'$。Euler–Lagrange 方程為 $L_u-\frac{d}{dx}L_{u'}=0$，因此 $-1-2u''=0$，即 $u''=-1/2$。若未指定端點值，還須另由邊界項 $[2u'v]_0^1$ 推出相應自然邊界條件；微分方程本身不唯一決定解，仍需邊界資料。

2. 以均勻網格節點 $j=0,\ldots,n$ 表示 Neumann 問題 $-u''=f$，並令 $U_j$ 為節點值。採用端點半權重的梯形離散內積時，剛度矩陣為
$$K=\frac1h\begin{bmatrix}1&-1\\-1&2&-1\\&\ddots&\ddots&\ddots\\&&-1&2&-1\\&&&-1&1\end{bmatrix},$$
而離散右端為 $b_j=h f_j$（內點）、$b_0=h f_0/2$、$b_n=h f_n/2$。每列和為零，所以 $K\mathbf1=0$；左乘 $\mathbf1^T$ 得可解必要條件 $\sum_j b_j=0$，也就是梯形權重下的離散積分相容條件。由於 $K$ 對稱且其核恰為常數向量，這也是充分條件。解只在加常數的意義下唯一；固定一個節點，或要求加權平均為零，便可選定唯一代表，不應加入 $\epsilon I$ 來改變方程。

   以下是簡短的相容性檢查示例：先組出端點半權重矩陣，再以增廣系統同時施加零平均規範。若相容性不成立，程式明確拒絕輸入；若成立，增廣解的乘子應接近零。
```python
import numpy as np

def solve_neumann(f, h, tol=1e-12):
    f = np.asarray(f, dtype=float)
    if f.ndim != 1 or len(f) < 2 or not np.isfinite(f).all():
        raise ValueError("f 必須是至少含兩個節點的有限值向量")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h 必須是有限正數")
    n = len(f) - 1
    if not np.isclose(n * h, 1.0):
        raise ValueError("此範例網格區間固定為 [0,1]")
    w = np.full(n + 1, h)
    w[[0, -1]] *= 0.5
    b = w * f
    if abs(np.sum(b)) > tol * max(1.0, np.sum(np.abs(b))):
        raise ValueError("Neumann 相容條件不成立：加權離散積分必須為零")
    K = np.diag(np.r_[1.0, np.full(n - 1, 2.0), 1.0]) / h
    if n > 1:
        K += np.diag(np.full(n, -1.0 / h), 1)
        K += np.diag(np.full(n, -1.0 / h), -1)
    A = np.zeros((n + 2, n + 2))
    A[:n+1, :n+1] = K
    A[:n+1, n+1] = w
    A[n+1, :n+1] = w
    rhs = np.r_[b, 0.0]
    return np.linalg.solve(A, rhs)[:n+1]

# 邊界／相容性核對：常數零源可解；非零常數源必須拒絕
assert np.allclose(solve_neumann(np.zeros(5), 0.25), np.zeros(5))
try:
    solve_neumann(np.ones(5), 0.25)
except ValueError:
    pass
else:
    raise AssertionError("不相容的 Neumann 右端未被拒絕")
```

3. 令容許函數空間為 $H^1_0(0,1)$，考慮
$$J[u]=\int_0^1\left(\frac12(u')^2-\frac{\pi^2}{2}u^2\right)\,dx.$$
$u=0$ 是駐點。取 $v_+(x)=\sin(\pi x)$，則
$$D^2J(0)[v_+,v_+]=\int_0^1((v_+')^2-\pi^2v_+^2)\,dx=0.$$
為得到正負方向，改取
$$J[u]=\int_0^1\left(\frac12(u')^2-\pi^2u^2\right)\,dx.$$
此時對 $v_+$，第二變分為 $-\frac{\pi^2}{2}<0$；對 $v_-(x)=\sin(2\pi x)$，第二變分為
$$\int_0^1(4\pi^2\cos^2(2\pi x)-\pi^2\sin^2(2\pi x))\,dx=\frac{3\pi^2}{2}>0.$$
兩方向一正一負，故 $u=0$ 是鞍點。

4. 梯度由泛函的一階導數與所選內積共同決定。若在同一函數空間內有 $DJ(u)[v]=\langle g,v\rangle_{L^2}$，則 $g$ 是該泛函的 $L^2$ 梯度；若改用 $H^1$ 內積 $\langle a,b\rangle_{H^1}=\int_0^1(ab+a'b')\,dx$，其梯度 $g_{H^1}$ 滿足
$$\int_0^1(g_{H^1}v+g_{H^1}'v')\,dx=DJ(u)[v].$$
在適當邊界條件下，分部積分把左側寫為 $\int_0^1(g_{H^1}-g_{H^1}'')v\,dx$，因此須解 Riesz 方程 $(I-\partial_{xx})g_{H^1}=g_{L^2}$，而不是把 $L^2$ 梯度一概說成函數本身。只有給定具體泛函、函數空間與邊界條件後，兩種梯度才有明確可比較的表示。
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
本章建立了變分法的嚴格框架，強調了 Fréchet 導數、自然邊界及二階正定條件。離數化實作展示了連續與離數梯度的關係，並通過測試驗證了數值方案的正確性。
<<<NEW>>>
本章建立了變分法的基本框架，說明第一變分、Euler–Lagrange 方程、自然邊界與二階強制性判準。離散例說明座標梯度與質量矩陣梯度的關係；列出的檢查屬預期測試，並非已執行或已驗證的結果。
<<<END>>>