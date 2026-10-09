# 第19章 多重積分、Riemann和與Fubini

## 學習目標與先備知識

本章將單變量的 Riemann 積分推廣到二維：從矩形分割與任意標記的 Riemann 和出發，利用 Darboux 上下和判定可積性，證明連續函數在緊矩形上可積，並說明何時能把二重積分寫成兩種迭代積分。讀完後，應能分辨「積分值存在」、「某一種格網求積有近似值」與「交換積分順序合法」是三件不同的事。

先備知識包括單變量 Riemann 積分及其上下和、連續函數在緊區間上的一致連續性，以及一元積分的基本計算。本章使用通常的歐氏距離衡量平面點的接近程度；不預設 Lebesgue 積分或測度論。核心定理以**緊矩形上的連續函數**為假設。對不連續函數或含奇異點的瑕積分，須另外檢查條件，不能把連續函數定理直接套用。

本章也會比較元胞中心取樣與網格節點加權。程式給出的有限精度數值只能核對手算並揭示可能的實作錯誤；即使多組網格的結果相近，也不能據此證明任意分割、任意取樣點的無限命題。

## 問題與直覺

單變量 Riemann 和把區間切成小段，以取樣值乘區間長度。二維情形則把矩形切成小格，以函數取樣值乘格子面積，再將各項相加。若網格愈來愈細時，**不論如何選擇分割與格內取樣點**，和都逼近同一數值，才得到矩形上的 Riemann 積分。當 $f\geq0$ 時，可以把積分想成曲面下的體積；當 $f$ 有正有負時，所得是正負相抵的代數體積。圖像有助於理解，卻不能取代極限條件。

對 $R=[a,b]\times[c,d]$ 上的連續函數，一種計算方式是先固定 $x$，沿 $y$ 方向積分，再對 $x$ 積分；另一種方式交換兩個方向。Fubini 定理在這些條件下保證兩種計算等於同一個二重積分，但不表示任何帶有奇異點的式子都能換序。

非矩形區域又多了一項邊界問題。若把區域放進外接矩形並在區域外補零，標記 Riemann 和仍是「每格完整面積乘該格的補零函數取樣值」。另一種數值策略是求出小格與區域的交集面積，再按交集面積加權；它並非補零定義的必要步驟。有限格網如何處理邊界，會影響近似誤差，但不能只憑一張格網判定函數是否可積。

## 定義、定理與推導

### 19.1 矩形分割與 Riemann 和

先令 $R=[a,b]\times[c,d]$ 為非退化閉矩形，即 $a<b$ 且 $c<d$。選取分割
$$
a=x_0<x_1<\cdots<x_m=b,\qquad
c=y_0<y_1<\cdots<y_n=d.
$$
子矩形 $R_{ij}=[x_{i-1},x_i]\times[y_{j-1},y_j]$ 的面積為 $\Delta A_{ij}=\Delta x_i\Delta y_j$，其中 $\Delta x_i=x_i-x_{i-1}$、$\Delta y_j=y_j-y_{j-1}$。網格細度定為 $\|P\|=\max\{\Delta x_i,\Delta y_j:1\leq i\leq m,\ 1\leq j\leq n\}$。每格任取標記點 $(x_{ij}^*,y_{ij}^*)\in R_{ij}$，得到
$$
S(P,f)=\sum_{i=1}^{m}\sum_{j=1}^{n}
f(x_{ij}^*,y_{ij}^*)\Delta A_{ij}.
$$
若存在 $I\in\mathbb R$，使對每個 $\epsilon>0$ 都有 $\delta>0$，而每個滿足 $\|P\|<\delta$ 的分割及其**所有**標記點選法均使 $|S(P,f)-I|<\epsilon$，便稱 $f$ 在 $R$ 上 Riemann 可積，記 $I=\iint_R f\,dA$。

若 $D\subset R$，可定義補零函數 $F=f1_D$；**僅當** $F$ 在 $R$ 上可積時，才以 $\iint_D f\,dA:=\iint_R F\,dA$ 定義區域上的積分。一個常用充分條件是 $D$ 有界、其邊界可由總面積任意小的有限個矩形覆蓋，且 $f$ 是包含 $D$ 的閉矩形上的連續函數；此時邊界附近格子的影響可受控。這類區域通常稱為 Jordan 可測區域。任意有界區域並不自動符合條件。

### 19.2 可積條件

**定理 19.1（Darboux 判準）。** 設 $f:R\to\mathbb R$ 有界。令 $M_{ij}$、$m_{ij}$ 分別為 $f$ 在 $R_{ij}$ 上的上、下確界，並令
$$
U(P,f)=\sum_{i,j}M_{ij}\Delta A_{ij},\qquad
L(P,f)=\sum_{i,j}m_{ij}\Delta A_{ij}.
$$
則 $f$ Riemann 可積，當且僅當對每個 $\epsilon>0$，存在分割 $P$ 使 $U(P,f)-L(P,f)<\epsilon$。

**證明。** 任意兩個分割都有共同細分；細分不增加上和，也不減少下和。因此下積分 $\underline I=\sup_P L(P,f)$ 不大於上積分 $\overline I=\inf_P U(P,f)$。若能使 $U(P,f)-L(P,f)<\epsilon$，則
$$
0\leq\overline I-\underline I\leq U(P,f)-L(P,f)<\epsilon.
$$
由 $\epsilon$ 任意可知兩者相等。固定一個上下和差很小的分割後，對充分細的其他分割取共同細分；除靠近固定分割線的小格外，標記和可由固定分割的上下和夾住。因 $f$ 有界，靠近那些分割線的小格之總面積隨細度趨零，故其貢獻亦趨零。於是所有充分細的標記和都趨於共同值。反之，若所有充分細的標記和都趨於 $I$，選一個細分割，使任意兩個標記和的差小於 $\epsilon/2$；再逐格選點，使兩組標記和分別任意接近上和與下和，即得 $U(P,f)-L(P,f)<\epsilon$。$\square$

**定理 19.2（連續函數可積性）。** 若 $f$ 在非退化緊矩形 $R$ 上連續，則 $f$ 在 $R$ 上 Riemann 可積。

**證明。** 有限維空間中的閉有界矩形緊緻，故 $f$ 一致連續。令 $A(R)=(b-a)(d-c)>0$。給定 $\epsilon>0$，可選 $\delta>0$，使距離小於 $\delta$ 的兩點函數值相差小於 $\epsilon/A(R)$。取 $\|P\|<\delta/\sqrt2$，則每格直徑小於 $\delta$，格內振盪 $\omega_{ij}=M_{ij}-m_{ij}<\epsilon/A(R)$。因此
$$
U(P,f)-L(P,f)
=\sum_{i,j}\omega_{ij}\Delta A_{ij}
<\frac{\epsilon}{A(R)}\sum_{i,j}\Delta A_{ij}
=\epsilon.
$$
由定理 19.1，$f$ 可積。$\square$

一致連續性在此使**所有**小格的振盪同時受控。不能把定理直接套到任意區域的補零函數：即使原函數連續，補零後也可能在邊界不連續。例如 $1_{\mathbb Q^2}$ 在矩形中每個非退化小格的上確界為 $1$、下確界為 $0$，故不可積；反過來說，存在不連續點也不必然使函數不可積。

### 19.3 連續函數的迭代積分

**定理 19.3（本章使用的 Fubini 定理）。** 若 $f$ 在 $R=[a,b]\times[c,d]$ 上連續，則每條水平及垂直截面均可作一元 Riemann 積分，兩個迭代積分存在，且
$$
\iint_R f(x,y)\,dA
=\int_a^b\left(\int_c^d f(x,y)\,dy\right)dx
=\int_c^d\left(\int_a^b f(x,y)\,dx\right)dy.
$$

**證明。** 令 $g(x)=\int_c^d f(x,y)\,dy$。對任意 $x,x'$，
$$
|g(x)-g(x')|
\leq(d-c)\sup_{y\in[c,d]}|f(x,y)-f(x',y)|.
$$
由 $f$ 在 $R$ 上一致連續，$g$ 連續，故外層積分存在。對矩形分割，在每個固定的 $x$ 標記值處，以 $y$ 方向的標記和近似內層積分；一致連續性使這項近似誤差對所有所選 $x$ 同時趨零。乘上各 $x$ 小段長度並相加，便與矩形標記和相差趨零；前者趨於 $\int_a^b g(x)\,dx$，後者由定理 19.2 趨於 $\iint_R f\,dA$。交換 $x,y$ 同理。$\square$

這個定理的連續性假設已同時保證矩形可積及各截面可積，不必再把它們當成未檢查的附加假設。對符合前述邊界條件的非矩形區域，可以研究補零函數，或依區域上下界拆成迭代積分；但其截面與邊界仍須逐項核對。若 $\\alpha,\\beta$ 在 $[a,b]$ 上連續、$\\alpha(x)\\leq\\beta(x)$，且 $f$ 在包含
$$
D=\\{(x,y):a\\leq x\\leq b,\\ \\alpha(x)\\leq y\\leq\\beta(x)\\}
$$
的矩形 $R=[a,b]\\times[c,d]$ 上連續，則補零函數 $F=f1_D$ 在 $R$ 上 Riemann 可積，且
$$
\\iint_D f\\,dA
=\\int_a^b\\left(\\int_{\\alpha(x)}^{\\beta(x)}f(x,y)\\,dy\\right)dx.
$$
證明如下。由連續性，$\\alpha,\\beta$ 在緊區間上一致連續且有界。其圖形可用總面積任意小的有限個矩形覆蓋，故分割足夠細時，與 $D$ 邊界相交的小格總面積可任意小。令 $M=\\sup_R|f|$；邊界格對上下和差的貢獻至多為 $2M$ 乘這些格子的總面積。完全位於 $D$ 內的格子上，$f$ 一致連續，故細化時振盪和趨零；完全在 $D$ 外的格子上，$F=0$。由 Darboux 判準，$F$ Riemann 可積。對每個固定 $x$，$\\int_c^d F(x,y)\\,dy=\\int_{\\alpha(x)}^{\\beta(x)}f(x,y)\\,dy$；右側隨 $x$ 連續。對矩形分割求和時，內部格的和由一致連續性收斂，邊界格的貢獻則由上述面積估計趨零，因此補零函數的矩形積分等於這些截面積分的外層積分。這證明了簡單區域公式，並非直接把矩形版定理 19.3 套用於非矩形區域。若函數有奇異點，本定理及此推論均不適用，不能只憑形式上能寫出兩個積分符號便換序。

### 19.4 瑕積分與交換順序

瑕積分涉及額外的極限，與緊矩形上的普通 Riemann 積分不同。絕對收斂是常用的**充分**換序條件，不是必要條件；若未建立適用的換序定理，也未控制截斷後的尾項，不能只交換積分符號。本章不以未證明的測度論定理代替這些檢查。

具體反例限於 $[0,1]^2$：在原點以外令
$$
f(x,y)=\frac{x^2-y^2}{(x^2+y^2)^2}.
$$
固定 $y>0$，對 $x$ 的原函數是 $-x/(x^2+y^2)$，所以先對 $x$、再對 $y$ 作瑕迭代積分，得到
$$
\int_0^1\left(\int_0^1 f(x,y)\,dx\right)dy
=\int_0^1\frac{-1}{1+y^2}\,dy=-\frac{\pi}{4}.
$$
固定 $x>0$，對 $y$ 的原函數是 $y/(x^2+y^2)$；反向迭代則得到 $\pi/4$。原點附近，$|f|$ 沿避開零值方向的角度區間與 $1/r^2$ 同階，而面積元含因子 $r$，其絕對積分含發散的 $\int_0^\varepsilon dr/r$。兩種瑕迭代積分各有定義卻不相等，正說明連續緊矩形定理不能跨過原點奇異性使用。

## 逐步手算例題

### 例 19.1：解析多項式二重積分

計算 $I = \iint_D (3x^2 + 2y) \, dA$，其中 $D$ 為矩形 $[0,1] \times [0,2]$。

**解**：
由於被積函數 $f(x,y) = 3x^2 + 2y$ 在緊矩形上連續，故可積且可應用Fubini。
選擇先對 $x$ 積分：
$$
I = \int_0^2 \left( \int_0^1 (3x^2 + 2y) \, dx \right) dy
$$
內層積分（視 $y$ 為常數）：
$$
\int_0^1 (3x^2 + 2y) \, dx = \left[ x^3 + 2yx \right]_0^1 = 1 + 2y
$$
代入外層積分：
$$
I = \int_0^2 (1 + 2y) \, dy = \left[ y + y^2 \right]_0^2 = (2 + 4) - 0 = 6
$$
驗證反向順序：
$$
I = \int_0^1 \left( \int_0^2 (3x^2 + 2y) \, dy \right) dx
$$
內層：
$$
\int_0^2 (3x^2 + 2y) \, dy = \left[ 3x^2 y + y^2 \right]_0^2 = 6x^2 + 4
$$
外層：
$$
\int_0^1 (6x^2 + 4) \, dx = \left[ 2x^3 + 4x \right]_0^1 = 2 + 4 = 6
$$
結果一致，$I = 6$。

**直覺檢查**：
$3x^2$ 在 $[0,1]$ 上的平均值为 $1$，面積 $1 \times 2 = 2$，貢獻 $2$。
$2y$ 在 $[0,2]$ 上的平均值为 $2$，面積 $1 \times 2 = 2$，貢獻 $4$。
總和 $2+4=6$。符合。

### 例 19.2：非矩形區域的積分

計算 $I = \iint_D (x + y) \, dA$，其中 $D$ 為三角形區域：$0 \le x \le 1$, $0 \le y \le x$。

**解**：
區域描述為 $0 \le x \le 1$ 且 $0 \le y \le x$。
應用Fubini：
$$
I = \int_0^1 \left( \int_0^x (x + y) \, dy \right) dx
$$
內層積分：
$$
\int_0^x (x + y) \, dy = \left[ xy + \frac{1}{2}y^2 \right]_{y=0}^{y=x} = x^2 + \frac{1}{2}x^2 = \frac{3}{2}x^2
$$
外層積分：
$$
I = \int_0^1 \frac{3}{2}x^2 \, dx = \frac{3}{2} \left[ \frac{x^3}{3} \right]_0^1 = \frac{1}{2}
$$
結果 $I = 0.5$。

**圖形直覺**：
在點 $(0,0)$，$f=0$；在 $(1,1)$，$f=2$。三角形面積為 $0.5$。平均高度約為 $1$（線性增長）。體積約 $0.5 \times 1 = 0.5$。符合。

## 實作與程式

以下程式只處理非退化矩形上的有限格網求積，不宣稱判定任意函數是否 Riemann 可積。`midpoint` 在每個元胞中心取樣，每格權重為 $\Delta x\Delta y$；`trapezoidal` 在包含邊界的節點取樣，沿每個座標方向的兩端各用半權重，因此角點權重為四分之一。兩者的取樣位置與資料筆數不同，不能只更換方法名稱而沿用同一組資料。

陣列遵循物理網格約定：`q[j, i]` 的 $j$ 沿 $+Y$、$i$ 沿 $+X$；`X`、`Y` 及函數輸出均須具有形狀 `(ny, nx)` 或節點法的 `(ny+1, nx+1)`。本程式要求函數回傳同形狀、有限的實數陣列；常數函數也須明確產生同形狀陣列，而不依賴隱含廣播。座標端點須為有限實數且依遞增順序排列，分割數須為正整數並拒絕布林值。

```python
import numpy as np


def numeric_riemann_2d(f, x_range, y_range, nx, ny,
                       method="midpoint"):
    """以元胞中點或張量積梯形規則近似矩形上的二重積分。"""
    for name, count in (("nx", nx), ("ny", ny)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def checked_range(bounds, name):
        try:
            if len(bounds) != 2:
                raise ValueError
            lo, hi = (float(value) for value in bounds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must have two real endpoints") from exc
        if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
            raise ValueError(f"{name} must have finite increasing endpoints")
        return lo, hi

    a, b = checked_range(x_range, "x_range")
    c, d = checked_range(y_range, "y_range")
    if method not in ("midpoint", "trapezoidal"):
        raise ValueError("method must be 'midpoint' or 'trapezoidal'")

    dx, dy = (b - a) / nx, (d - c) / ny
    if not np.isfinite(dx * dy) or dx <= 0 or dy <= 0:
        raise ValueError("cell area is not finite and positive")

    if method == "midpoint":
        xs = a + (np.arange(nx) + 0.5) * dx
        ys = c + (np.arange(ny) + 0.5) * dy
        wx, wy = np.ones(nx), np.ones(ny)
    else:
        xs = np.linspace(a, b, nx + 1)
        ys = np.linspace(c, d, ny + 1)
        wx, wy = np.ones(nx + 1), np.ones(ny + 1)
        wx[0] = wx[-1] = 0.5
        wy[0] = wy[-1] = 0.5

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    values = np.asarray(f(X, Y))
    if values.shape != X.shape:
        raise ValueError("f must return an array matching the grid shape")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("f must return real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("f returned non-finite values")

    result = np.sum(values * wy[:, None] * wx[None, :]) * dx * dy
    if not np.isfinite(result):
        raise ValueError("quadrature result is not finite")
    return float(result)


def polynomial(x, y):
    return 3.0 * x**2 + 2.0 * y


if __name__ == "__main__":
    for n in (10, 50, 100, 200):
        mid = numeric_riemann_2d(
            polynomial, (0, 1), (0, 2), n, n, "midpoint"
        )
        trap = numeric_riemann_2d(
            polynomial, (0, 1), (0, 2), n, n, "trapezoidal"
        )
        print(n, mid, trap, abs(mid - 6.0), abs(trap - 6.0))
```

此處的解析基準 $6$ 來自前節手算，並非程式內一個未定義的「解析積分器」。對 $n$ 等分的 $x$ 方向，中點規則對 $x^2$ 的一維求和比其積分少 $1/(12n^2)$，梯形規則則多 $1/(6n^2)$。函數的 $y$ 項為一次式，兩種規則沿 $y$ 方向均精確；把 $3x^2$ 項乘上寬度 $2$，便得到本例的**精確格網誤差式**
$$
I_{\mathrm{mid}}=6-\frac{1}{2n^2},
\qquad
I_{\mathrm{trap}}=6+\frac{1}{n^2}.
$$
因此本例的誤差為二階，而不是四階或指數下降。對其他函數，不能僅憑這個多項式的公式宣稱相同誤差階；須另列光滑性假設與求積誤差界。

## 測試與預期結果

下列斷言是**供讀者執行的預期測試**，不是已執行紀錄。正常測試用兩個方向均為一次式的 $x+y$；兩種規則理應給出 $1$。邊界測試用極小但非退化的正方形及常數函數，檢查面積權重；$10^{-12}$ 在一般雙精度表示範圍內，不應把它描述為必然下溢。故障測試則確認錯誤輸入明確失敗，而非悄悄給出一個貌似合理的數值。

```python
def expect_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def run_expected_tests():
    linear = lambda x, y: x + y
    constant = lambda x, y: np.ones_like(x, dtype=float)

    for method in ("midpoint", "trapezoidal"):
        actual = numeric_riemann_2d(
            linear, (0, 1), (0, 1), 4, 7, method
        )
        assert np.isclose(actual, 1.0, rtol=0, atol=1e-14)

        tiny = numeric_riemann_2d(
            constant, (0, 1e-6), (0, 1e-6), 3, 5, method
        )
        assert np.isclose(tiny, 1e-12, rtol=1e-14, atol=0)

    for n in (10, 200):
        mid = numeric_riemann_2d(
            polynomial, (0, 1), (0, 2), n, n, "midpoint"
        )
        trap = numeric_riemann_2d(
            polynomial, (0, 1), (0, 2), n, n, "trapezoidal"
        )
        assert np.isclose(mid, 6 - 1 / (2 * n**2), atol=1e-12)
        assert np.isclose(trap, 6 + 1 / n**2, atol=1e-12)

    base = lambda **kwargs: numeric_riemann_2d(
        constant, (0, 1), (0, 1), 2, 2, **kwargs
    )
    expect_value_error(lambda: base(method="node"))
    expect_value_error(lambda: numeric_riemann_2d(
        constant, (0, 1), (0, 1), 0, 2
    ))
    expect_value_error(lambda: numeric_riemann_2d(
        constant, (0, 1), (0, 1), True, 2
    ))
    expect_value_error(lambda: numeric_riemann_2d(
        constant, (1, 0), (0, 1), 2, 2
    ))
    expect_value_error(lambda: numeric_riemann_2d(
        constant, (0, np.inf), (0, 1), 2, 2
    ))
    expect_value_error(lambda: numeric_riemann_2d(
        lambda x, y: 1.0, (0, 1), (0, 1), 2, 2
    ))
    expect_value_error(lambda: numeric_riemann_2d(
        lambda x, y: np.full_like(x, np.nan),
        (0, 1), (0, 1), 2, 2
    ))
```

預期 $n=10$ 時中點值為 $5.995$、梯形值為 $6.01$；$n=200$ 時中點誤差為 $1.25\times10^{-5}$、梯形誤差為 $2.5\times10^{-5}$。斷言的浮點容差只用於核對有限精度運算，不是積分定義中的 $\epsilon$。

另可用 $1_{\{x<1/2,\ y<1/2\}}$ 作數值觀察：其積分為 $1/4$，但網格線、節點與嚴格不等號的相對位置會改變有限網格輸出。這是**可積而有限取樣仍有誤差**的例子，不是函數不可積或演算法必然故障的證據。程式拒絕非法輸入的測試，則與這種合法函數的取樣誤差屬於不同類型。

## 反例與常見陷阱

**瑕積分不可只交換符號。** 第 19.4 節在單位正方形上給出的函數，先對 $x$ 再對 $y$ 為 $-\pi/4$，反序為 $\pi/4$。兩個結果來自不同的迭代極限；不能將它們當成同一個普通二重 Riemann 積分的兩種算法。絕對收斂可提供常用的換序充分條件，但「沒有證明絕對收斂」本身也不等於「必定不能換序」；須檢查所用定理及截斷極限。

**可積性與截面性質不能混為一談。** 緊矩形上的連續性足以保證本章 Fubini 定理的全部積分存在。若函數在一點奇異，即使某些固定座標的內層積分可以算出，也不能援引該連續函數定理。反過來，少數點或邊界上的不連續不必然破壞 Riemann 可積性；應檢查上下和的差能否任意縮小，而非只看「是否處處連續」。

**元胞與節點不能共用權重。** 中點資料每個值代表一個完整小格；節點梯形法則沿邊界使用半權重，角點因兩方向相乘而使用四分之一權重。若將含兩側邊界的 $(n+1)\times(n+1)$ 個節點全乘相同完整格子面積，就連常數函數的總面積都會算錯。節點亦不「天生必須」採梯形法：權重取決於所選求積規則，本章只是明確採用其中一種。

**曲邊格子須交代策略。** 補零取樣法以完整格面積乘 $f1_D$ 在格內的取樣值；交集面積法以格子和 $D$ 的交集面積加權。二者不能混寫成同一個定義，也不宜對任意複雜邊界宣稱固定誤差階。若只保留完全落在區域內的小格，會遺漏邊界帶；若把所有碰到邊界的小格都當成完整內部格，則會多算邊界帶。邊界幾何及函數界限決定如何估計這些誤差。

## AI、幾何與養殖案例

### 面積平均與單位

對有界區域 $D$ 上的濃度場 $C(x,y)$，面積平均濃度定義為
$$
\overline C_A=\frac{1}{A(D)}\iint_D C(x,y)\,dA,
\qquad A(D)>0.
$$
若 $x,y$ 的單位為公尺、$C$ 的單位為 $\mathrm{mg/L}$，則積分的單位是 $\mathrm{mg\,m^2/L}$，除以面積後才回到 $\mathrm{mg/L}$。這是**面積平均濃度**，不是水體總氧量；要推算總量還需要深度、體積分布及相應的單位換算資料。

考慮純合成模型，令 $D=[0,10]\times[0,5]$，座標以公尺計，
$$
C(x,y)=5-0.1x^2-0.05y^2\quad\mathrm{mg/L}.
$$
常數項的單位為 $\mathrm{mg/L}$，兩個二次項係數的單位均為 $\mathrm{mg\,L^{-1}m^{-2}}$。由矩形上的連續性及 Fubini 定理，解析面積平均為
$$
\overline C_A
=5-0.1\frac{10^2}{3}-0.05\frac{5^2}{3}
=1.25\quad\mathrm{mg/L}.
$$
用 $10\times10$ 個元胞作中點求積時，對 $x^2$ 的格點平均為 $33.25\ \mathrm{m^2}$，對 $y^2$ 為 $8.3125\ \mathrm{m^2}$，故**預期**數值平均為
$$
5-0.1(33.25)-0.05(8.3125)
=1.259375\quad\mathrm{mg/L}.
$$
差異是此格網的取樣誤差，而非現場測量誤差。

更重要的是模型有效性：角點給出 $C(10,5)=-6.25\ \mathrm{mg/L}$，不可能解釋為物理濃度。解析積分 $1.25$ 雖然計算正確，卻不能使整個矩形上的合成公式成為可信的現場模型。也不能擅自將負值裁成零，再稱它仍是原模型的積分。可行的稽核紀錄應分列「依公式得到的數學結果」及「模型在部分區域失效」；沒有獨立量測與校準，就不給出曝氣或其他設備操作建議。

### 幾何網格與 AI 的界線

在幾何計算中，若非負高度函數定義於合適的平面底域，二重積分可表示其下方體積。對曲邊底域，首先要說明是採補零取樣，還是估計元胞與底域的交集面積。更密的格網可能有助於觀察數值差異，卻不會自動修正錯誤的區域、單位或函數模型。

唯讀的 AI 助理可以整理解析基準、格網規格、各次近似值與失效位置，並建議哪些區域值得進一步量測；它不能把合成濃度場當成已驗證感測資料，更不能根據本章的面積平均值直接控制設備。此處「數學求積正確」與「物理模型有效」是兩項必須分開呈報的證據。

## 習題

1. **手算。** 計算 $\iint_D(2x+3y)\,dA$，其中 $D=\{(x,y):x\geq0,\ y\geq0,\ x+y\leq1\}$。列出迭代積分的上下限與計算過程。

2. **程式。** 使用本章兩種求積方法近似 $f(x,y)=\sin x\cos y$ 在 $[0,\pi]\times[0,\pi/2]$ 上的積分。先求解析值，再寫出呼叫程式的方式；說明有限網格值與解析值的關係，不捏造已執行輸出。

3. **條件辨析。** 在 $[0,1]^2\setminus\{(0,0)\}$ 令 $h(x,y)=(x-y)/(x+y)^2$。證明它在原點附近絕對可積，並求兩個瑕迭代積分。再與第 19.4 節的反例比較，指出「原點有奇異性」何以不足以判定換序結果。

4. **整合。** 半徑 $R=10\ \mathrm m$ 的圓形區域上有合成模型 $C(x,y)=C_0-kr^2$，其中 $r=\sqrt{x^2+y^2}$，$C_0$ 以 $\mathrm{mg/L}$、$k$ 以 $\mathrm{mg\,L^{-1}m^{-2}}$ 計。利用極座標求**面積平均濃度**，並分別說明矩形格網的補零取樣法與邊界格交集面積法。極座標的面積因子可先使用，正式換變數條件留待下一章。

## 習題解答

1. 區域可寫成 $0\leq x\leq1$、$0\leq y\leq1-x$。三角形有界且邊界可受控，連續函數在其上的補零積分可用此上下限計算：
   $$
   \begin{aligned}
   I
   &=\int_0^1\int_0^{1-x}(2x+3y)\,dy\,dx\\
   &=\int_0^1\left(2x(1-x)+\frac32(1-x)^2\right)dx\\
   &=\int_0^1\left(\frac32-x-\frac12x^2\right)dx
   =\frac56.
   \end{aligned}
   $$
   檢查量級時，三角形面積為 $1/2$，被積函數非負，結果也應非負；這項檢查不能代替上下限計算。

2. 函數在緊矩形上連續，解析值可拆為兩個一元積分：
   $$
   \int_0^\pi\sin x\,dx
   \int_0^{\pi/2}\cos y\,dy
   =2\cdot1=2.
   $$
   沿用已定義函數的呼叫方式，例如：
   ```python
   wave = lambda x, y: np.sin(x) * np.cos(y)
   mid = numeric_riemann_2d(
       wave, (0, np.pi), (0, np.pi / 2), 40, 40, "midpoint"
   )
   trap = numeric_riemann_2d(
       wave, (0, np.pi), (0, np.pi / 2), 40, 40, "trapezoidal"
   )
   print(mid, trap, abs(mid - 2.0), abs(trap - 2.0))
   ```
   預期兩個有限網格值接近 $2$，但一般不恰好等於 $2$；若要主張某個誤差階，還須另行引用或推導適用於此函數的求積誤差界。此段是可執行範例，並非執行紀錄。

3. 因 $|x-y|\leq x+y$，在原點以外有
   $$
   |h(x,y)|\leq\frac1{x+y}.
   $$
   後者在單位正方形上的瑕積分可直接計算為
   $$
   \int_0^1\int_0^1\frac1{x+y}\,dy\,dx
   =\int_0^1\bigl(\log(x+1)-\log x\bigr)\,dx
   =2\log2<\infty.
   $$
   因此 $h$ 絕對可積。為核對迭代值，固定 $x>0$ 時先對 $y$ 積分得到
   $$
   \int_0^1h(x,y)\,dy
   =2-\frac{2x}{x+1}-\log\frac{x+1}{x}.
   $$
   其外層瑕積分為 $0$：前兩項的積分為 $2-2(1-\log2)=2\log2$，最後一項的積分也為 $2\log2$。函數滿足 $h(y,x)=-h(x,y)$，故反向迭代積分同樣存在且為 $0$。在 $x=0$ 的單獨截面，內層積分發散；這不改變上述以 $x>0$ 的內層結果形成的外層瑕積分。這個例子與第 19.4 節的非絕對可積反例不同：**有奇異點並不足以推出換序不合法，更不足以推出兩個答案不相等。**

4. 使用極座標面積元 $dA=r\,dr\,d\theta$，圓域面積為 $\pi R^2$，所以
   $$
   \begin{aligned}
   \overline C_A
   &=\frac{1}{\pi R^2}
     \int_0^{2\pi}\int_0^R(C_0-kr^2)r\,dr\,d\theta\\
   &=C_0-\frac{kR^2}{2}
    =C_0-50k.
   \end{aligned}
   $$
   最後一式須連同題設單位理解：$50$ 來自 $R^2/2=50\ \mathrm{m^2}$，故結果仍以 $\mathrm{mg/L}$ 計。這是面積平均濃度，不是圓池的總氧量；若模型在圓域某處給出負濃度，仍須另行檢查其物理有效範圍。

   使用外接矩形網格時，補零取樣法在每格取一點，以完整格面積乘 $C1_D$ 的取樣值；交集面積法則對邊界格使用該格與圓盤交集的面積作權重。兩者均須處理邊界，卻不是同一個權重規則，也不能說補零定義「必須」先算交集面積。

## 本章小結

二重 Riemann 積分要求所有充分細的矩形分割及標記選法趨於同一值；Darboux 上下和提供可積性判準。有限維緊矩形上的連續函數一致連續，因而可積，且其兩種迭代積分都存在並等於二重積分。對非矩形區域，應檢查邊界並區分補零定義與交集面積求積策略；對含奇異點的瑕積分，則須另行證明換序條件。絕對收斂是常用充分條件，並非必要條件。

格網實作還須分清元胞中心及節點權重，核對陣列方向、單位、解析值與輸入失敗行為。本章多項式的二階誤差有精確公式，但有限次數值核對不能證明一般函數的可積性或收斂階。下一章將進一步說明積分換變數及其 Jacobian 面積因子所需的條件。

## 參考來源

- [A1] Jiří Lebl，*Basic Analysis* 作者目錄與教材入口：<https://www.jirka.org/ra/>。作為實分析背景入口；此處不宣稱已逐條核對教材內的定理。
- [A3] MIT OpenCourseWare，*18.02SC Multivariable Calculus* 課程概要：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>。作為多變量積分主題的延伸學習入口。

本章數值程式僅用 NumPy；上述來源不構成程式已執行或章內每項推導已由外部逐條驗證的紀錄。
