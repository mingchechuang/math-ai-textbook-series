# 第07章 有限差分算子與邊界閉合

## 學習目標與先備知識

本章把連續微分算子轉成可組裝、可檢查的有限差分算子。重點不是背誦公式，而是理解矩陣每一個橫列代表哪一個內點方程或邊界條件。完成本章後，讀者應能：

1. 由 Taylor 展開推導中心差分與單邊差分，判定截斷誤差階數。
2. 分辨節點中心與單元中心網格，避免混用索引及邊界位置。
3. 正確定義離散 Laplacian 的符號，理解 $L\approx\Delta$ 通常為負半定，而 $A=-L$ 才是正半定或正定。
4. 使用 ghost cell 表示 Dirichlet、Neumann 與週期邊界。
5. 組裝小型矩陣，檢查常數場、線性場、二次場與邊界橫列。
6. 分開判斷一致性、穩定性、守恆、能量下降、非負性與物理可信度。

先備知識包括偏導數、Taylor 展開、矩陣乘法，以及 Dirichlet 與 Neumann 邊界條件。以下程式只需要 Python 3.10 以上與 NumPy，並以 CPU 小格網為範圍。本文只列出依推導所得的預期結果，不宣稱已實際執行。

---

## 問題與直覺

考慮一維擴散方程

$$
\frac{\partial c}{\partial t}
=
D\frac{\partial^2c}{\partial x^2}+s,
$$

其中 $c$ 可為濃度，單位是 $\mathrm{kg/m^3}$；$D$ 是擴散係數，單位是 $\mathrm{m^2/s}$；來源 $s$ 的單位是 $\mathrm{kg/(m^3\,s)}$。二階導數不能直接交給電腦，必須用相鄰格點的數值近似。

內部格點左右都有資料，通常可以使用對稱中心差分。邊界卻缺少域外資料，因此必須「閉合」：

- 直接將邊界值寫成代數方程；
- 使用單邊差分；
- 引入 ghost cell；
- 以週期條件連接另一側資料。

邊界閉合不是單純的程式細節。它會改變矩陣的第一個與最後一個橫列，也可能改變對稱性、精度、守恆性與零空間。內點格式即使完全正確，錯誤的邊界橫列仍可能使整個解不收斂。

本章主要採用一維節點中心網格：

$$
x_i=ih,\qquad i=0,\ldots,N,\qquad h=\frac{L_x}{N}.
$$

此時 $x_0$ 與 $x_N$ 就在物理邊界上。後文另有單元中心例子，其中心為

$$
x_i=\left(i+\frac12\right)h.
$$

兩種配置的 ghost cell 公式不同，不可直接搬用。

---

## 數學與物理推導

### 1. Taylor 展開與一階導數

若 $u$ 在 $x_i$ 附近足夠光滑，則

$$
u_{i+1}
=
u_i+h u_i'
+\frac{h^2}{2}u_i''
+\frac{h^3}{6}u_i'''
+O(h^4),
$$

$$
u_{i-1}
=
u_i-h u_i'
+\frac{h^2}{2}u_i''
-\frac{h^3}{6}u_i'''
+O(h^4).
$$

兩式相減後除以 $2h$：

$$
\frac{u_{i+1}-u_{i-1}}{2h}
=
u_i'+\frac{h^2}{6}u_i'''+O(h^4).
$$

因此中心一階差分

$$
(D_0u)_i=\frac{u_{i+1}-u_{i-1}}{2h}
$$

具有二階截斷誤差：

$$
(D_0u)_i-u_i'=O(h^2).
$$

若只有右側資料，可用前向差分：

$$
(D_+u)_i
=
\frac{u_{i+1}-u_i}{h}
=
u_i'+\frac{h}{2}u_i''+O(h^2),
$$

故它通常只有一階精度。

要取得二階單邊公式，令

$$
u_i'\approx\frac{a u_i+b u_{i+1}+c u_{i+2}}{h}.
$$

代入 Taylor 展開，要求常數項與二階導數項消失、一階導數係數為一：

$$
a+b+c=0,\qquad b+2c=1,\qquad \frac{b}{2}+2c=0.
$$

解得 $a=-3/2$、$b=2$、$c=-1/2$，所以

$$
u_i'
=
\frac{-3u_i+4u_{i+1}-u_{i+2}}{2h}
+O(h^2).
$$

右邊界的鏡射公式為

$$
u_N'
=
\frac{3u_N-4u_{N-1}+u_{N-2}}{2h}
+O(h^2).
$$

這些階數結論都依賴相應階數導數存在且有界。若函數在目標點不可微，Taylor 推導便不適用。

### 2. 二階導數與 Laplacian 符號

Taylor 展開兩式相加：

$$
u_{i+1}-2u_i+u_{i-1}
=
h^2u_i''+\frac{h^4}{12}u_i^{(4)}+O(h^6).
$$

因此

$$
(Lu)_i
=
\frac{u_{i-1}-2u_i+u_{i+1}}{h^2}
=
u_i''+O(h^2).
$$

本卷固定令 $L$ 近似 Laplacian $\Delta$。對週期邊界，離散分部積分給出

$$
\boldsymbol{u}^{T}L\boldsymbol{u}
=
-\frac{1}{h^2}\sum_i (u_{i+1}-u_i)^2
\leq0,
$$

其中索引按週期連接。因此 $L$ 是負半定矩陣。適當加權的齊次 Neumann 離散也具有相應的非正能量性質。

Poisson 方程通常寫成

$$
A\boldsymbol{u}=\boldsymbol{b},
\qquad A=-L.
$$

在 Dirichlet 邊界消去後，$A$ 通常為對稱正定。若誤把 $L$ 當成正定矩陣，後續擴散穩定性、能量估計與線性求解器選擇都會發生符號錯誤。

### 3. Dirichlet 邊界閉合

考慮

$$
u''=f,\qquad 0<x<L_x,
$$

$$
u(0)=a,\qquad u(L_x)=b.
$$

若只把內點 $u_1,\ldots,u_{N-1}$ 當未知量，第一個內點方程為

$$
\frac{a-2u_1+u_2}{h^2}=f_1.
$$

令直向量

$$
\boldsymbol{v}=(u_1,\ldots,u_{N-1})^T,
$$

則

$$
L_D
=
\frac{1}{h^2}
\begin{bmatrix}
-2&1&&\\
1&-2&1&\\
&\ddots&\ddots&\ddots\\
&&1&-2
\end{bmatrix},
$$

而右端必須扣除邊界貢獻：

$$
L_D\boldsymbol{v}
=
\boldsymbol{f}
-
\frac{1}{h^2}
\begin{bmatrix}
a\\0\\ \vdots\\0\\b
\end{bmatrix}.
$$

另一種做法是保留全部節點，將第一個與最後一個橫列直接換成

$$
u_0=a,\qquad u_N=b.
$$

這種完整矩陣易於閱讀，但替換後一般不再保持原 Laplacian 的對稱結構。若要使用要求對稱正定的演算法，通常宜先消去 Dirichlet 自由度。

### 4. Neumann 邊界、外法向與 ghost cell

假設左端給定沿 $+x$ 方向的導數

$$
u_x(0)=g_L.
$$

引入域外節點 $x_{-1}=-h$，中心差分給出

$$
\frac{u_1-u_{-1}}{2h}=g_L,
$$

所以

$$
u_{-1}=u_1-2hg_L.
$$

代入邊界節點的 Laplacian：

$$
(Lu)_0
=
\frac{u_{-1}-2u_0+u_1}{h^2}
=
\frac{2(u_1-u_0)}{h^2}-\frac{2g_L}{h}.
$$

必須注意：雖然表示一階導數的中心公式為二階，但上述邊界 Laplacian 對一般函數只有一階點態精度：

$$
\frac{2(u_1-u_0)}{h^2}-\frac{2u_x(0)}{h}
=
u_{xx}(0)+\frac{h}{3}u_{xxx}(0)+O(h^2).
$$

不能因為使用了中心 ghost 公式，就直接宣稱整個邊界算子二階。

法向符號也必須區分。區間左端的外法向量為 $n=-1$，右端為 $n=+1$，所以

$$
\partial_nu=
\begin{cases}
-u_x,&x=0,\\
u_x,&x=L_x.
\end{cases}
$$

若物理條件寫成 $\partial_nu=q_n$，左端應使用 $u_x=-q_n$，而不是 $u_x=q_n$。

對 Fick 擴散通量

$$
J=-D u_x,
$$

外向通量為 $J_n=Jn$。西側與東側即使具有相同的 $u_x$ 數值，外向通量符號仍相反。

### 5. 週期閉合與零空間

對 $N$ 個不重複儲存端點的週期節點，索引為 $i=0,\ldots,N-1$，有

$$
u_{-1}=u_{N-1},\qquad u_N=u_0.
$$

因此

$$
L_P
=
\frac{1}{h^2}
\begin{bmatrix}
-2&1&0&\cdots&1\\
1&-2&1&&0\\
0&1&-2&\ddots&0\\
\vdots&&\ddots&\ddots&1\\
1&0&0&1&-2
\end{bmatrix}.
$$

每個橫列的元素和均為零，所以

$$
L_P\boldsymbol{1}=\boldsymbol{0}.
$$

常數場是零模態。矩陣不能直接反解任意 Poisson 右端；至少必須滿足

$$
\sum_i f_i=0
$$

並另加均值條件。週期 Laplacian 的奇異性不是程式故障，而是連續問題只決定到任意加法常數的離散反映。

### 6. 單元中心 ghost cell

若第一個未知量位於 $x_0=h/2$，左側 ghost cell 位於 $x_{-1}=-h/2$，而邊界值為 $u(0)=a$，常用線性插值：

$$
\frac{u_{-1}+u_0}{2}=a,
\qquad
u_{-1}=2a-u_0.
$$

因此第一個中心的三點 Laplacian 為

$$
(Lu)_0
=
\frac{u_1-3u_0+2a}{h^2}.
$$

這與節點中心的 $u_0=a$ 完全不同。若把兩種配置混用，等效邊界會被錯放半格。

### 7. 二維算子與索引

在右手座標系中，$X$ 向右、$Y$ 向上。二維物理陣列 `q[j,i]` 的形狀為 `(Ny,Nx)`，其中 $i$ 沿 $+X$、$j$ 沿 $+Y$。五點 Laplacian 為

$$
(Lq)_{j,i}
=
\frac{q_{j,i-1}-2q_{j,i}+q_{j,i+1}}{\Delta x^2}
+
\frac{q_{j-1,i}-2q_{j,i}+q_{j+1,i}}{\Delta y^2}.
$$

展平索引固定為

$$
k=jN_x+i.
$$

內點的東西鄰居對應 $k\pm1$，南北鄰居對應 $k\pm N_x$。但在每一列網格的左右端，不能無條件把 $k+1$ 視為東鄰居，否則會把一列尾端錯接到下一列開頭。若繪圖，應使用 `origin="lower"` 或明示翻轉，不能把陣列橫列向下的影像慣例誤當成物理 $+Y$。

---

## 逐步手算例題

### 例題一：比較中心與單邊導數

令

$$
u(x)=x^3,\qquad x=1,\qquad h=0.1.
$$

解析導數為 $u'(1)=3$。

第一步，中心差分：

$$
\frac{u(1.1)-u(0.9)}{0.2}
=
\frac{1.331-0.729}{0.2}
=3.01.
$$

誤差為 $0.01$。

第二步，一階前向差分：

$$
\frac{u(1.1)-u(1)}{0.1}
=
\frac{1.331-1}{0.1}
=3.31.
$$

誤差為 $0.31$。

第三步，二階前向差分。因為 $u(1.2)=1.728$，

$$
\frac{-3u(1)+4u(1.1)-u(1.2)}{0.2}
=
\frac{-3+5.324-1.728}{0.2}
=2.98.
$$

誤差為 $-0.02$。此例只比較單一步長，不能單憑三個數字證明觀測收斂階；正式檢查還需要多個 $h$ 並確認已進入漸近區域。

### 例題二：組裝 Dirichlet Laplacian

考慮

$$
u''=-2,\qquad 0<x<1,
$$

$$
u(0)=u(1)=0.
$$

解析解為

$$
u(x)=x(1-x).
$$

取 $h=1/4$，內點解析值為

$$
\boldsymbol{u}
=
\begin{bmatrix}
3/16\\1/4\\3/16
\end{bmatrix}.
$$

離散 Laplacian 為

$$
L_D
=
16
\begin{bmatrix}
-2&1&0\\
1&-2&1\\
0&1&-2
\end{bmatrix}.
$$

第一個橫列給出

$$
16\left(-2\frac{3}{16}+\frac14\right)=-2.
$$

第二個橫列給出

$$
16\left(\frac{3}{16}-2\frac14+\frac{3}{16}\right)=-2.
$$

第三個橫列同樣得到 $-2$。二次函數的四階導數為零，因此三點二階差分在此例恰好精確。

### 例題三：週期矩陣與高頻模態

取四個週期點且 $h=1$：

$$
L=
\begin{bmatrix}
-2&1&0&1\\
1&-2&1&0\\
0&1&-2&1\\
1&0&1&-2
\end{bmatrix}.
$$

對常數直向量 $\boldsymbol{c}=(5,5,5,5)^T$，

$$
L\boldsymbol{c}=\boldsymbol{0}.
$$

對交錯模態 $\boldsymbol{v}=(1,-1,1,-1)^T$，

$$
L\boldsymbol{v}=(-4,4,-4,4)^T=-4\boldsymbol{v}.
$$

因此高頻模態對應負特徵值。擴散半離散方程

$$
\dot{\boldsymbol{u}}=DL\boldsymbol{u}
$$

會衰減此模態，而不是放大它。

---

## 實作與程式

以下自足程式建立一維週期 Laplacian、Dirichlet 內點 Laplacian，以及非齊次 Neumann ghost 閉合。它會拒絕布林網格數、非有限輸入與非法網格。

```python
import numpy as np


def require_integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} 不可為布林值")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} 必須是整數")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} 必須至少為 {minimum}")
    return value


def require_positive_finite(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必須是正且有限的數")
    return value


def periodic_laplacian_1d(n, h):
    """N 個不重複端點的週期節點，回傳 L ≈ d²/dx²。"""
    n = require_integer("n", n, 3)
    h = require_positive_finite("h", h)

    L = np.zeros((n, n), dtype=float)
    inv_h2 = 1.0 / h**2
    for i in range(n):
        L[i, i] = -2.0 * inv_h2
        L[i, (i - 1) % n] += inv_h2
        L[i, (i + 1) % n] += inv_h2
    return L


def dirichlet_interior_laplacian(n_intervals, length):
    """節點 x_i=i*h；只回傳 N-1 個內點未知量的 L。"""
    n_intervals = require_integer("n_intervals", n_intervals, 2)
    length = require_positive_finite("length", length)

    h = length / n_intervals
    m = n_intervals - 1
    L = np.zeros((m, m), dtype=float)
    inv_h2 = 1.0 / h**2

    for i in range(m):
        L[i, i] = -2.0 * inv_h2
        if i > 0:
            L[i, i - 1] = inv_h2
        if i + 1 < m:
            L[i, i + 1] = inv_h2
    return L, h


def dirichlet_rhs(f_inner, h, left_value, right_value):
    """建立 L u = f 的內點右端。"""
    f_inner = np.asarray(f_inner, dtype=float)
    if f_inner.ndim != 1 or f_inner.size < 1:
        raise ValueError("f_inner 必須是一維非空陣列")
    if not np.all(np.isfinite(f_inner)):
        raise ValueError("f_inner 含 NaN 或無窮大")

    h = require_positive_finite("h", h)
    left_value = float(left_value)
    right_value = float(right_value)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        raise ValueError("Dirichlet 邊界值必須有限")

    rhs = f_inner.copy()
    rhs[0] -= left_value / h**2
    rhs[-1] -= right_value / h**2
    return rhs


def neumann_ghost_laplacian_1d(
    n_intervals, length, gx_left, gx_right
):
    """
    節點中心且包含兩端。
    gx_left=u_x(0)，gx_right=u_x(L)，不是外法向導數。
    回傳 L、c、h，使離散 Laplacian 為 L@u+c。
    """
    n_intervals = require_integer("n_intervals", n_intervals, 2)
    length = require_positive_finite("length", length)
    gx_left = float(gx_left)
    gx_right = float(gx_right)

    if not np.isfinite(gx_left) or not np.isfinite(gx_right):
        raise ValueError("Neumann 導數必須有限")

    h = length / n_intervals
    n = n_intervals + 1
    L = np.zeros((n, n), dtype=float)
    c = np.zeros(n, dtype=float)
    inv_h2 = 1.0 / h**2

    for i in range(1, n - 1):
        L[i, i - 1] = inv_h2
        L[i, i] = -2.0 * inv_h2
        L[i, i + 1] = inv_h2

    # 左端：u_-1 = u_1 - 2 h gx_left
    L[0, 0] = -2.0 * inv_h2
    L[0, 1] = 2.0 * inv_h2
    c[0] = -2.0 * gx_left / h

    # 右端：u_{N+1} = u_{N-1} + 2 h gx_right
    L[-1, -1] = -2.0 * inv_h2
    L[-1, -2] = 2.0 * inv_h2
    c[-1] = 2.0 * gx_right / h

    return L, c, h


def periodic_laplacian_2d(nx, ny, dx, dy):
    """q[j,i] 展平為 k=j*nx+i 的二維週期 Laplacian。"""
    nx = require_integer("nx", nx, 3)
    ny = require_integer("ny", ny, 2)
    dx = require_positive_finite("dx", dx)
    dy = require_positive_finite("dy", dy)

    n = nx * ny
    L = np.zeros((n, n), dtype=float)

    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            west = j * nx + ((i - 1) % nx)
            east = j * nx + ((i + 1) % nx)
            south = ((j - 1) % ny) * nx + i
            north = ((j + 1) % ny) * nx + i

            L[k, k] -= 2.0 / dx**2 + 2.0 / dy**2
            L[k, west] += 1.0 / dx**2
            L[k, east] += 1.0 / dx**2
            L[k, south] += 1.0 / dy**2
            L[k, north] += 1.0 / dy**2
    return L


def run_diagnostics():
    Lp = periodic_laplacian_1d(4, 1.0)
    assert np.allclose(Lp @ np.ones(4), 0.0)
    assert np.allclose(Lp, Lp.T)
    assert np.max(np.linalg.eigvalsh(Lp)) <= 1.0e-12

    Ld, h = dirichlet_interior_laplacian(4, 1.0)
    x = np.arange(1, 4) * h
    u = x * (1.0 - x)
    rhs = dirichlet_rhs(-2.0 * np.ones(3), h, 0.0, 0.0)
    assert np.allclose(Ld @ u, rhs, atol=1.0e-12)

    Ln, c, h = neumann_ghost_laplacian_1d(
        n_intervals=4,
        length=1.0,
        gx_left=2.0,
        gx_right=2.0,
    )
    x_all = np.arange(5) * h
    u_linear = 2.0 * x_all + 3.0
    assert np.allclose(Ln @ u_linear + c, 0.0, atol=1.0e-12)

    L2 = periodic_laplacian_2d(
        np.int64(3), np.int64(2), 1.0, 1.0
    )
    assert np.allclose(L2 @ np.ones(6), 0.0)
    assert np.allclose(L2, L2.T)
    assert np.allclose(L2.sum(axis=1), 0.0)

    bad_calls = [
        lambda: periodic_laplacian_1d(4, np.nan),
        lambda: periodic_laplacian_1d(True, 1.0),
        lambda: periodic_laplacian_2d(3, False, 1.0, 1.0),
        lambda: dirichlet_rhs(
            np.array([1.0, np.inf]), 0.25, 0.0, 0.0
        ),
    ]
    for call in bad_calls:
        try:
            call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("非法輸入未被拒絕")

    return Lp, Ld, Ln, L2


if __name__ == "__main__":
    run_diagnostics()
    print("若未觸發例外，則通過指定的代數診斷。")
```

對 $N_y=2$，週期北鄰居與南鄰居會落在同一個格點，因此同一矩陣元素應累加兩次；程式使用 `+=` 保留此重數。ghost 值不屬於物理未知量，也不應計入真實質量。

---

## 測試與預期結果

下列都是依代數推導得到的預期結果。

### 正常測試

1. **常數場測試**：週期矩陣每個橫列和為零，因此 `Lp @ np.ones(4)` 應為零。
2. **矩陣符號測試**：`Lp` 應對稱，理論上所有特徵值不大於零。
3. **二次函數測試**：對 $u=x(1-x)$，Dirichlet 內點差分應精確給出 $u''=-2$。
4. **Neumann 線性場測試**：對 $u=2x+3$，兩端傳入 $u_x=2$，`Ln @ u + c` 應為零。
5. **二維索引測試**：二維週期矩陣應對稱、橫列和為零，並消去常數場。

### 邊界測試

- `n_intervals=2` 應可建立一個 Dirichlet 內點未知量。
- 一維週期網格拒絕少於三點。
- 齊次 Neumann 情況下 `Ln @ ones` 應為零，但這不表示矩陣可唯一反解。
- `np.int64` 網格數應被接受；`True` 與 `False` 必須被拒絕，因為布林值雖是 Python `int` 的子類，卻不是有意義的網格數。

### 故障測試

以下輸入必須拒絕，而不是繼續產生看似平滑的答案：

- $h\leq0$ 或區間長度非正；
- `NaN` 或無窮大的係數、來源、間距或邊界值；
- 非整數或布林網格數；
- 與矩陣尺寸不一致的直向量；
- 把外法向導數誤當成 $u_x$ 而未轉換符號。

### 各種數值性質不可混為一談

- **一致性**：把光滑解析函數代入差分，檢查局部截斷誤差是否隨 $h$ 消失。
- **穩定性**：時間步進或迭代中的擾動是否受控；只檢查空間矩陣不足以證明完整演算法穩定。
- **守恆**：內部通量是否逐面抵消，總量變化是否等於邊界通量與來源。
- **能量下降**：必須先定義離散能量及時間格式；$L$ 負半定不保證任意顯式時間步長都使能量下降。
- **非負性**：即使總量守恆且某個能量下降，濃度仍可能出現負值。
- **物理可信度**：還需要正確單位、合理係數、可信邊界資料及適切模型；代數測試通過不等於完成物理驗證。

---

## 除錯與常見陷阱

### 1. Laplacian 符號反轉

若擴散寫成

$$
u_t=D\Delta u,
$$

離散後應為

$$
\dot{\boldsymbol{u}}=DL\boldsymbol{u},
$$

其中 $L$ 負半定。若使用 $A=-L$，方程就要寫成

$$
\dot{\boldsymbol{u}}=-DA\boldsymbol{u}.
$$

### 2. 邊界值未移到右端

若 $u(0)=a\neq0$，第一個內點方程右端必須包含 $-a/h^2$。遺漏後仍可能得到平滑曲線，但解的是另一個問題。

### 3. 法向導數與座標導數混淆

左端 $\partial_nu=-u_x$，右端 $\partial_nu=u_x$。兩端指定相同外法向導數，不代表兩端的 $u_x$ 相同。

### 4. 把 ghost cell 當成真實體積

ghost cell 只是邊界閉合工具。計算總質量時若將 ghost 值也乘體積加入，就會製造虛假質量。

### 5. 只看平滑圖形

符號相反、邊界偏移半格或週期接錯的解仍可能看起來平滑。至少應測試常數、線性、二次函數及高頻模態。

### 6. 對不可微函數宣稱二階

例如 $u(x)=|x|$ 在 $x=0$ 不可微。中心差分可能因對稱而得到零，卻不能視為對不存在導數的二階近似。

### 7. 用裁零掩蓋負濃度

若後續顯式擴散產生負濃度，直接執行 `np.maximum(c, 0)` 會改變總量。應先檢查時間步長、邊界通量與離散係數；若確實採用裁切，必須記錄其質量改變。

---

## 養殖與相場案例

### 合成池域濃度

考慮長度 $L_x=10\,\mathrm{m}$ 的一維合成水道，濃度 $c$ 的單位為 $\mathrm{kg/m^3}$：

$$
c_t=Dc_{xx}+s,
\qquad
D=10^{-4}\,\mathrm{m^2/s}.
$$

初始條件可指定為合成場

$$
c(x,0)=c_0(x).
$$

若左端為封閉牆，邊界條件是零外向擴散通量：

$$
J_n=(-Dc_x)n=0.
$$

因 $D>0$，等價於 $c_x(0)=0$。若右端連接固定合成濃度槽，可設 $c(L_x,t)=c_R$。這是一端 Neumann、一端 Dirichlet 的混合邊界，矩陣兩端不能使用同一種閉合。

濃度換算為

$$
1\,\mathrm{mg/L}
=
\frac{10^{-6}\,\mathrm{kg}}{10^{-3}\,\mathrm{m^3}}
=
10^{-3}\,\mathrm{kg/m^3}.
$$

若 $s$ 表示體積來源，其單位必須是 $\mathrm{kg/(m^3\,s)}$。網格平滑、殘差小或濃度非負，都不能單獨證明這些合成係數代表真實池域。

### 相場中的 Laplacian

本卷相場採無因次序參量 $\phi$ 與

$$
W(\phi)=\frac{(\phi^2-1)^2}{4},
$$

$$
\mu=\phi^3-\phi-\kappa\Delta\phi.
$$

離散後為

$$
\boldsymbol{\mu}
=
\boldsymbol{\phi}^{\circ3}
-\boldsymbol{\phi}
-\kappa L\boldsymbol{\phi}.
$$

若週期邊界下 $L$ 的符號錯置，梯度項會放大高頻振盪，而不是抑制它。常數場測試只能檢查零模態，還應使用交錯模態檢查特徵值符號。

相場序參量通常不是養殖濃度；跨越某個溶氧管理門檻也不是物理相變。邊界閉合的代數正確性與模型詮釋必須分開。

---

## 習題

### 習題一：手算推導

利用 Taylor 展開推導

$$
u''(x_0)
\approx
\frac{2u_0-5u_1+4u_2-u_3}{h^2},
$$

其中 $u_k=u(x_0+kh)$，並判定精度。

### 習題二：程式與矩陣

建立 $N_x=3$、$N_y=2$、$\Delta x=\Delta y=1$ 的二維週期五點 Laplacian，展平規則為 $k=jN_x+i$。檢查常數零模態、矩陣對稱性與每個橫列的元素和。

### 習題三：反例

某人主張：「中心差分總是二階準確。」提出一個反例，說明 Taylor 推導為何失效，以及數值結果剛好為零為何不代表導數存在。

### 習題四：整合邊界閉合

考慮

$$
u''=0,\qquad 0<x<1,
$$

$$
\partial_nu(0)=-2,\qquad u(1)=5.
$$

1. 將左端外法向導數換成 $u_x(0)$。
2. 求解析解。
3. 取 $h=1/2$，以 $(u_0,u_1,u_2)^T$ 為未知直向量；左端使用二階前向導數，內點使用中心 Laplacian，右端使用 Dirichlet 條件，寫出線性系統。
4. 驗證解析節點值。

### 習題五：性質辨析

某擴散程式維持總質量至機器精度，但出現少量負濃度。能否推論它穩定、能量下降或物理可信？可否直接裁零而仍宣稱守恆？

---

## 習題解答

### 解答一

展開

$$
u_k
=
u_0+khu_0'
+\frac{k^2h^2}{2}u_0''
+\frac{k^3h^3}{6}u_0'''
+\frac{k^4h^4}{24}u_0^{(4)}
+O(h^5).
$$

令

$$
S=2u_0-5u_1+4u_2-u_3.
$$

常數、一階及三階項係數均為零；二階項係數為一，而四階項係數為 $-11/12$。因此

$$
S=h^2u_0''-\frac{11}{12}h^4u_0^{(4)}+O(h^5),
$$

所以

$$
\frac{S}{h^2}
=
u_0''-\frac{11}{12}h^2u_0^{(4)}+O(h^3).
$$

此單邊二階導數公式為二階準確。

### 解答二

本章 `periodic_laplacian_2d` 可直接使用：

```python
L = periodic_laplacian_2d(
    np.int64(3), np.int64(2), 1.0, 1.0
)
assert np.allclose(L @ np.ones(6), 0.0)
assert np.allclose(L, L.T)
assert np.allclose(L.sum(axis=1), 0.0)
```

對 $N_y=2$，北鄰居與南鄰居是同一個週期格點，但代表兩個方向的差分貢獻，因此係數必須累加兩次。這些檢查只證明指定的代數性質，不等同於時間積分穩定性。

### 解答三

取

$$
u(x)=|x|
$$

並在 $x=0$ 使用中心差分：

$$
\frac{u(h)-u(-h)}{2h}
=
\frac{h-h}{2h}=0.
$$

然而左導數為 $-1$，右導數為 $1$，所以 $u'(0)$ 不存在。Taylor 推導要求目標點附近有足夠光滑性，此條件在尖點失效。差分值為零只是函數對稱造成的結果，不是對真實導數的近似。

### 解答四

左端外法向量為 $n=-1$，故

$$
\partial_nu=-u_x=-2
$$

給出

$$
u_x(0)=2.
$$

由 $u''=0$ 得 $u=ax+b$。使用 $u_x=2$ 與 $u(1)=5$，得到

$$
u(x)=2x+3.
$$

取 $h=1/2$。左端二階前向公式為

$$
\frac{-3u_0+4u_1-u_2}{2h}=2.
$$

因 $2h=1$，完整系統為

$$
\begin{bmatrix}
-3&4&-1\\
1&-2&1\\
0&0&1
\end{bmatrix}
\begin{bmatrix}
u_0\\u_1\\u_2
\end{bmatrix}
=
\begin{bmatrix}
2\\0\\5
\end{bmatrix}.
$$

解析節點值為

$$
(u_0,u_1,u_2)=(3,4,5).
$$

代入三個橫列分別得到 $2$、$0$、$5$，故解析節點值滿足離散系統。

### 解答五

1. **不能推論穩定。** 守恆只表示總和維持，局部振盪仍可能增長。
2. **不能推論能量下降。** 必須先定義離散能量，再逐步檢查相應不等式。
3. **不能推論物理可信。** 負濃度可能違反模型允許範圍，係數與邊界資料也可能不可信。
4. **不可無記錄地裁零。** 裁零會增加總質量，破壞原本的守恆。若採用修正，必須報告修正前後的質量差。

---

## 本章小結

有限差分以 Taylor 展開建立局部近似，再利用邊界條件閉合缺失的域外資料。中心公式通常對稱且精度較高，但邊界必須使用單邊公式、ghost cell、週期連接或直接代數約束。

本卷固定使用 $L\approx\Delta$，因此週期及適當齊次 Neumann Laplacian 為負半定；Poisson 矩陣則常取 $A=-L$。常數場檢查橫列和，線性場檢查 Neumann 符號，二次場檢查二階差分，高頻模態則檢查算子符號。

一致性、穩定性、守恆、能量下降、非負性與物理可信度是不同問題。正確的邊界矩陣是後續擴散、Poisson、平流擴散與相場計算的必要基礎，但不是完整可信模擬的充分條件。

---

## 參考來源

1. **F1：FiPy 有限體積離散與邊界**  
   <https://pages.nist.gov/fipy/en/latest/numerical/discret.html>  
   可對照單元中心、面通量與邊界條件的有限體積觀點；其配置不能不加轉換地套用到本章節點中心公式。

2. **F2：FEniCSx Poisson 與弱形式**  
   <https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html>  
   提供 Poisson 符號、Dirichlet 邊界與弱形式的比較背景。

3. **F3：PETSc 線性系統求解器**  
   <https://petsc.org/release/manual/ksp/>  
   後續求解離散系統時，應依對稱性、正定性與零空間選擇方法，並檢查真殘差。

4. **F4：SciPy 稀疏線性代數 API**  
   <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>  
   大型差分矩陣宜使用稀疏儲存；本章為保持自足，只使用 NumPy 小型稠密矩陣。

5. **F6：FiPy Cahn–Hilliard 相分離示範**  
   <https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html>  
   可觀察 Laplacian 與邊界閉合在相場模型中的角色；其序參量與參數慣例不必然等同本卷的 $[-1,1]$ 雙井形式。