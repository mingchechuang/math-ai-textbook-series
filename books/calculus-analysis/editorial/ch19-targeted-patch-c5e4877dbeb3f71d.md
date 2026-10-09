<<<PATCH 01>>>
<<<OLD>>>
**注意**：Fubini定理要求 $D$ 是矩形。若 $D$ 非矩形，需將區域拆分或補零。此外，定理保證迭代積分等於重積分，但不保證迭代積分本身收斂（若未假設 $f$ 可積，僅假設 $f$ 連續，通常也成立，但若有奇異點，需檢查內層積分是否對幾乎所有 $x$ 存在）。
<<<NEW>>>
**注意**：定理19.3處理緊矩形上的連續函數；此時函數及每條截面均Riemann可積，兩個迭代積分都存在。非矩形區域須確認拆分後的積分或補零函數的可積性；有奇異點的瑕積分則須另行分析。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
def compute_integral_analytic(func_x, func_y, limits_x, limits_y):
    """
    計算迭代積分 ∫_{x_min}^{x_max} ( ∫_{y_min}^{y_max} f(x,y) dy ) dx
    假設 func_y(x, y) 可對 y 積分，返回內層積分結果 g(x)，再對 x 積分。
    此處為演示，使用簡單的多項式閉式解或scipy.integrate.quad（若可用）。
    為了自足，我們手動實現簡單的多項式積分或僅演示數值。
    """
    # 此函數僅作為佔位，實際計算在 numeric_riemann 中
    pass

def numeric_riemann_2d(f, x_range, y_range, nx, ny, method='midpoint'):
    """
    使用Riemann和近似二重積分。
    f: 函數 f(x, y)
    x_range: (x_min, x_max)
    y_range: (y_min, y_max)
    nx, ny: 分割數
    method: 'midpoint' 或 'node'
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # 計算步長
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    
    # 生成網格點
    if method == 'midpoint':
        # Cell中心：x_i = x_min + (i - 0.5) * dx
        x_points = x_min + np.arange(nx) * dx + 0.5 * dx
        y_points = y_min + np.arange(ny) * dy + 0.5 * dy
    elif method == 'node':
        # 節點：x_i = x_min + i * dx (排除最後一點以符合Riemann和定義，或使用Trapezoidal)
        # 標準Riemann和通常取左、右或中點。這裡我們比較中點與「節點加權」（近似Trapezoidal規則）
        # 為了對齊，我們這裡展示：
        # 1. 中點規則（高階精確）
        # 2. 簡單節點規則（左端點），顯示較低精度
        # 為了對齊“cell/node權重”，我們使用Trapezoidal規則作為“節點加權”的代表
        # 但題目要求區分cell/node。我們實現：
        # - Midpoint: f(x_i + dx/2, y_j + dy/2) * dx * dy
        # - Node (Trapezoidal approximation): 使用邊界半權重
        pass # 此處為結構，具體實現在下
    
    # 為了簡潔，我們只實現Midpoint作為高階，並展示一個錯誤的“節點”（如左端點）來對比誤差
    # 或者實現Trapezoidal
    # 這裡我們實現兩個函數以清晰區分
    
    # 重新設計：
    if method == 'midpoint':
        x_coords = np.array([x_min + (i + 0.5) * dx for i in range(nx)])
        y_coords = np.array([y_min + (j + 0.5) * dy for j in range(ny)])
    elif method == 'trapezoidal':
        x_coords = np.linspace(x_min, x_max, nx + 1)
        y_coords = np.linspace(y_min, y_max, ny + 1)
        dx_trap = (x_max - x_min) / nx
        dy_trap = (y_max - y_min) / ny
        # 建立權重
        wx = np.ones(nx + 1)
        wx[0] = 0.5
        wx[-1] = 0.5
        wy = np.ones(ny + 1)
        wy[0] = 0.5
        wy[-1] = 0.5
        weights = np.outer(wx * dx_trap, wy * dy_trap)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        values = f(X, Y)
        return np.sum(values * weights)
    else:
        raise ValueError("Invalid method")
    
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    values = f(X, Y)
    return np.sum(values) * dx * dy

def test_integration():
    # 定義函數 f(x, y) = 3x^2 + 2y
    def f(x, y):
        return 3 * x**2 + 2 * y
    
    x_range = (0, 1)
    y_range = (0, 2)
    exact_val = 6.0
    
    print(f"Exact Value: {exact_val}")
    
    # 測試不同分割數
    for n in [10, 50, 100, 200]:
        # 使用相同數量的x和y分割
        approx_mid = numeric_riemann_2d(f, x_range, y_range, n, n, method='midpoint')
        approx_trap = numeric_riemann_2d(f, x_range, y_range, n, n, method='trapezoidal')
        
        err_mid = abs(approx_mid - exact_val)
        err_trap = abs(approx_trap - exact_val)
        
        print(f"n={n:3d} | Midpoint: {approx_mid:.6f} (Err: {err_mid:.2e}) | Trapezoidal: {approx_trap:.6f} (Err: {err_trap:.2e})")
<<<NEW>>>
def numeric_riemann_2d(f, x_range, y_range, nx, ny, method='midpoint'):
    """矩形上的cell中點或node梯形求積；f 須支援陣列輸入。"""
    if (isinstance(nx, (bool, np.bool_)) or
        isinstance(ny, (bool, np.bool_)) or
        not isinstance(nx, (int, np.integer)) or
        not isinstance(ny, (int, np.integer)) or nx <= 0 or ny <= 0):
        raise ValueError("nx and ny must be positive integers")
    if method not in ('midpoint', 'trapezoidal'):
        raise ValueError("method must be 'midpoint' or 'trapezoidal'")
    x_min, x_max = x_range
    y_min, y_max = y_range
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("ranges must have increasing endpoints")
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    if method == 'midpoint':
        xs = x_min + (np.arange(nx) + 0.5) * dx
        ys = y_min + (np.arange(ny) + 0.5) * dy
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        return float(np.sum(f(X, Y) + np.zeros_like(X)) * dx * dy)

    xs = np.linspace(x_min, x_max, nx + 1)
    ys = np.linspace(y_min, y_max, ny + 1)
    wx = np.ones(nx + 1)
    wy = np.ones(ny + 1)
    wx[[0, -1]] = 0.5
    wy[[0, -1]] = 0.5
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    values = f(X, Y) + np.zeros_like(X)
    return float(np.sum(values * np.outer(wx, wy)) * dx * dy)

def test_integration():
    f = lambda x, y: 3 * x**2 + 2 * y
    for n in [10, 50, 100, 200]:
        mid = numeric_riemann_2d(f, (0, 1), (0, 2), n, n)
        trap = numeric_riemann_2d(
            f, (0, 1), (0, 2), n, n, method='trapezoidal')
        assert np.isclose(mid, 6 - 0.5 / n**2, atol=1e-12)
        assert np.isclose(trap, 6 + 1 / n**2, atol=1e-12)
        print(f"n={n}: midpoint={mid:.8f}, trapezoidal={trap:.8f}")

    # 正常及邊界測試
    linear = lambda x, y: x + y
    for method in ('midpoint', 'trapezoidal'):
        assert np.isclose(numeric_riemann_2d(
            linear, (0, 1), (0, 1), 7, 9, method), 1.0, atol=1e-12)
        assert np.isclose(numeric_riemann_2d(
            lambda x, y: 1.0, (0, 1e-6), (0, 1e-6),
            4, 4, method), 1e-12, rtol=1e-12, atol=0)

    # 故障測試：拒絕未實作的方法與零分割
    for changes in ({'method': 'node'}, {'nx': 0}):
        args = dict(f=linear, x_range=(0, 1), y_range=(0, 1),
                    nx=4, ny=4)
        args.update(changes)
        try:
            numeric_riemann_2d(**args)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {changes}")
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
**程式說明**：
1. `numeric_riemann_2d` 函數實現了兩種求積方法：
   - `midpoint`：在每個子矩形中心取樣。對於多項式，此方法通常具有 $O(h^4)$ 的誤差收斂率（對於一維梯形是 $O(h^2)$，中點是 $O(h^2)$ 但常數更小，實際上對於線性函數中點規則精確；對於二次函數，中點規則誤差為 $O(h^2)$？不，中點規則對於線性函數是精確的，誤差項涉及二階導數。對於二重積分，收斂率取決於函數光滑性）。
   - `trapezoidal`：使用節點加權（邊界半權重）。
2. 我們使用 $f(x,y) = 3x^2 + 2y$。由於該函數在 $y$ 方向是線性的，在 $x$ 方向是二次的。
   - 對於線性函數，梯形規則和中點規則都是精確的（誤差為0）？
   - 檢查：$f_y = 2$ 是常數，對 $y$ 積分精確。$f_x = 3x^2$ 是二次。
   - 一維梯形規則對 $x^2$ 的誤差是 $O(h^2)$。一維中點規則對 $x^2$ 的誤差也是 $O(h^2)$，但係數不同。
   - 實際上，對於 $3x^2$，中點規則在 $[0,1]$ 上：$\sum 3(x_i - 0.5/h)^2 \dots$。
   - 讓我們預期結果：誤差應隨 $n$ 增加而減小。

**預期結果**：
- $n=10$: 誤差約 $10^{-3}$ 量級。
- $n=200$: 誤差應小於 $10^{-6}$。
- 中點規則通常比簡單左端點規則收斂更快且更穩定。
<<<NEW>>>
**程式說明與預期結果（未執行）**：`midpoint` 在各cell中心取樣，每格使用面積權重；`trapezoidal` 在node取樣，各座標方向的邊界節點給半權重，角點因此給四分之一權重。對本例的 $n\times n$ 網格，$y$ 項在兩法下均精確，誤差只來自 $3x^2$。利用 $\sum_{i=0}^{n-1}(i+\tfrac12)^2=n(4n^2-1)/12$ 可得中點值 $6-1/(2n^2)$；梯形值為 $6+1/n^2$。兩法的絕對誤差分別是 $1/(2n^2)$、$1/n^2$，皆為二階。預期 $n=10$ 時誤差為 $0.005$、$0.01$；$n=200$ 時為 $1.25\times10^{-5}$、$2.5\times10^{-5}$。這些是本多項式的解析結果，不是任意函數的誤差常數。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
**反例**：
考慮 $f(x,y) = \frac{x^2 - y^2}{(x^2 + y^2)^2}$ 在 $D = [0, \infty) \times [0, \infty)$。
這函數在原點發散。
$\int_0^{\infty} \int_0^{\infty} f \, dx \, dy$ 可能收斂到 $-\pi/4$ 或類似值，而反向順序收斂到 $\pi/4$ 或發散。
**教訓**：對於瑕積分，若未證明 $\iint |f| < \infty$，不可隨意交換順序。必須先檢查絕對收斂性。
<<<NEW>>>
**反例**：
考慮19.4節的函數 $f(x,y)=(x^2-y^2)/(x^2+y^2)^2$，區域為 $[0,1]^2\setminus\{(0,0)\}$。按該節的逐次瑕積分，先對 $x$ 積分得 $-\pi/4$，先對 $y$ 積分得 $\pi/4$；原點附近的絕對積分發散。不能將區域改為無界正象限而沿用這兩個數值。
**教訓**：絕對收斂是保障瑕積分換序的常用充分條件，並非兩種次序恰好同值的必要條件。沒有適用定理或其他證明時，不可僅憑形式運算換序。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
3. **反例**：
   $f(x,y) = \frac{x-y}{(x+y)^2}$。
   考慮 $\int_0^1 \int_0^1 f \, dy \, dx$。
   固定 $x$，對 $y$ 積分：$\int_0^1 \frac{x-y}{(x+y)^2} dy$。
   令 $u = x+y, du = dy$。$y = u-x$。
   $\int_x^{x+1} \frac{x-(u-x)}{u^2} du = \int_x^{x+1} \frac{2x-u}{u^2} du = \int_x^{x+1} (2x u^{-2} - u^{-1}) du$。
   $= \left[ -2x u^{-1} - \ln u \right]_x^{x+1} = \left( \frac{-2x}{x+1} - \ln(x+1) \right) - \left( \frac{-2x}{x} - \ln x \right) = \frac{-2x}{x+1} - \ln\frac{x+1}{x} + 2$。
   當 $x \to 0$ 時，$\ln(x+1) \to 0$, $\ln x \to -\infty$。此處有奇異性。
   實際上，$\int_0^1 \frac{x-y}{(x+y)^2} dy$ 在 $x=0$ 處發散。因此迭代積分不作為有限值存在，或者取決於積分順序。
   關鍵點：$f$ 在 $(0,0)$ 不絕對可積。$\iint |f| dA = \infty$。因此Fubini定理不適用，交換順序結果不同或發散。
<<<NEW>>>
3. **反例辨析**：
   題目所聲稱的不等式實際上不成立。在第一象限，$|x-y|\le x+y$，故 $|f(x,y)|\le1/(x+y)$。固定 $x>0$ 積分此上界得 $\ln((x+1)/x)$，再對 $x\in(0,1]$ 積分有限，因為 $\int_0^1|\ln x|\,dx=1$。所以此處的瑕積分絕對收斂。先對 $y$ 積分的結果為
   $$
   g(x)=2-\frac{2x}{x+1}-\ln\frac{x+1}{x}\qquad(x>0).
   $$
   雖然 $g(x)$ 在零附近無界，該對數奇異性仍可積。交換 $x,y$ 會使 $f$ 變號，而正方形保持不變；由絕對收斂可換序，兩個瑕迭代積分均為 $0$。單一截面 $x=0$ 的內層積分發散，不足以推出外層瑕積分發散。真正得到不同值的反例見19.4節。
<<<END>>>