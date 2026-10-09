# 第07章 Fréchet導數與餘項條件

## 學習目標與先備知識

本章把單變量的「導數是斜率」推廣為多變量的「導數是最佳一階線性映射」。完成本章後，讀者應能：

1. 對 $f:U\subset\mathbb R^n\to\mathbb R^m$ 寫出 Fréchet 可微的精確定義。
2. 區分函數增量、線性主部與非線性餘項：
   $$
   f(x+h)=f(x)+Df(x)[h]+r(h).
   $$
3. 使用指定範數驗證
   $$
   \frac{\|r(h)\|}{\|h\|}\to0.
   $$
4. 證明 Fréchet 導數若存在則唯一，並證明可微必連續。
5. 理解可微性要求對所有充分小的允許擾動成立，而不只是沿有限條路徑成立。
6. 以解析式求導數矩陣，並以矩陣擬合與方向殘差比作有限精度核對。
7. 辨識偏導存在或所有方向導數存在但仍不可微的情況。
8. 在有物理單位的問題中標示導數元素的單位，並以無因次尺度比較擾動。

先備知識包括有限維向量範數、矩陣乘法、多變量極限、連續性與單變量導數。除非另行說明，本章輸入與輸出空間皆使用 Euclidean 範數 $\|\cdot\|_2$，矩陣使用其誘導算子範數。

在有限維空間中，不同範數彼此等價，因此給出相同的 Fréchet 可微概念；但具體誤差界的常數會隨範數及維度改變。輸入擾動 $h$ 是 $n\times1$ 列向量。若輸出維度為 $m$，則 $J_f(x)\in\mathbb R^{m\times n}$，且 $J_f(x)h$ 是 $m\times1$ 列向量。

---

## 問題與直覺

在工作點 $x$ 附近，希望用固定線性映射 $A$ 預測輸出變化：

$$
f(x+h)-f(x)\approx Ah.
$$

定義誤差

$$
r(h)=f(x+h)-f(x)-Ah.
$$

僅知道 $r(h)\to0$ 並不足夠。若 $f$ 在 $x$ 連續，取 $A=0$ 時，增量也可能趨近零，但這不表示零映射是一階導數。Fréchet 可微要求更強的相對誤差條件：

$$
\frac{\|r(h)\|_2}{\|h\|_2}\to0.
$$

也就是說，扣除線性主部後，餘項必須比擾動的一階尺度更快消失。若 $\|r(h)\|$ 是 $\|h\|^2$ 等級，則殘差比是 $\|h\|$ 等級，會趨近零；若餘項仍與 $\|h\|$ 同階，殘差比通常不會趨近零。

「最佳線性近似」不是指針對有限資料做最小平方後得到的最佳矩陣，而是指存在一個固定線性映射，使誤差為 $o(\|h\|)$。這是涵蓋所有充分小、非零且留在定義域內之擾動的極限命題。有限資料可以估計候選矩陣，也可以找到反例，但不能證明這個無限量詞命題。

---

## 定義、定理與推導

### 1. Fréchet 可微的定義

**定義 7.1（Fréchet 可微）**  
設 $U\subset\mathbb R^n$ 為開集，$x\in U$，且 $f:U\to\mathbb R^m$。若存在線性映射

$$
A:\mathbb R^n\to\mathbb R^m
$$

使得

$$
\lim_{\substack{h\to0,\ h\ne0\\x+h\in U}}
\frac{\|f(x+h)-f(x)-Ah\|_2}{\|h\|_2}=0,
$$

則稱 $f$ 在 $x$ Fréchet 可微，並記 $Df(x)=A$。

因 $U$ 開且 $x\in U$，存在 $\eta>0$ 使 $B(x,\eta)\subset U$，所以充分小的 $h$ 自動滿足 $x+h\in U$。極限中仍明寫域限制，以保留完整量詞。

等價地，

$$
f(x+h)=f(x)+Df(x)[h]+r(h),
$$

其中

$$
\|r(h)\|_2=o(\|h\|_2).
$$

小 $o$ 條件的量詞形式為：對每個 $\varepsilon>0$，存在 $\delta>0$，使得只要

$$
0<\|h\|_2<\delta,\qquad x+h\in U,
$$

就有

$$
\|r(h)\|_2\le\varepsilon\|h\|_2.
$$

在標準座標下，有限維線性映射由唯一矩陣 $J_f(x)\in\mathbb R^{m\times n}$ 表示：

$$
Df(x)[h]=J_f(x)h.
$$

導數本身是線性映射；矩陣是選定基底後的表示。

### 2. 導數的唯一性

**命題 7.2（Fréchet 導數唯一）**  
設 $U\subset\mathbb R^n$ 開，$x\in U$，且 $f:U\to\mathbb R^m$。若線性映射 $A,B:\mathbb R^n\to\mathbb R^m$ 都滿足 Fréchet 餘項條件，則 $A=B$。

**證明。**  
設

$$
f(x+h)-f(x)-Ah=r_A(h),
$$

以及

$$
f(x+h)-f(x)-Bh=r_B(h),
$$

其中

$$
\frac{\|r_A(h)\|_2}{\|h\|_2}\to0,
\qquad
\frac{\|r_B(h)\|_2}{\|h\|_2}\to0.
$$

兩式相減得

$$
(A-B)h=r_B(h)-r_A(h).
$$

任取非零 $v\in\mathbb R^n$，令 $h=tv$，其中 $t\ne0$ 且 $t\to0$。由 $U$ 開，充分小的 $t$ 可保證 $x+tv\in U$。利用線性性及三角不等式，

$$
\|(A-B)v\|_2
\le
\frac{\|r_A(tv)\|_2+\|r_B(tv)\|_2}{|t|}.
$$

又因 $\|tv\|_2=|t|\|v\|_2$，

$$
\frac{\|r_A(tv)\|_2}{|t|}
=
\|v\|_2\frac{\|r_A(tv)\|_2}{\|tv\|_2}\to0,
$$

而 $r_B$ 的項亦趨近零。因此 $\|(A-B)v\|_2=0$。零向量的情況顯然成立，故 $(A-B)v=0$ 對所有 $v$ 成立，所以 $A=B$。證畢。

這項唯一性是由全方向餘項條件推出，不是由某個有限資料擬合結果推出。

### 3. 可微必連續

**命題 7.3（必要條件）**  
若 $f$ 在 $x$ Fréchet 可微，則 $f$ 在 $x$ 連續。

**證明。**  
令 $A=Df(x)$。由定義，

$$
f(x+h)-f(x)=Ah+r(h).
$$

使用 Euclidean 誘導算子範數，

$$
\|f(x+h)-f(x)\|_2
\le
\|A\|_{\mathrm{op}}\|h\|_2+\|r(h)\|_2.
$$

第一項隨 $h\to0$ 而趨零。又因 $\|r(h)\|_2/\|h\|_2\to0$，所以 $\|r(h)\|_2\to0$。因此 $f(x+h)\to f(x)$。證畢。

連續是可微的必要條件，但不是充分條件；例如 $t\mapsto|t|$ 在零點連續而不可微。

### 4. 與方向導數的關係

若 $f$ 在 $x$ Fréchet 可微，對任意固定 $v\in\mathbb R^n$，

$$
f(x+tv)-f(x)=tDf(x)[v]+r(tv).
$$

除以 $t\ne0$，得到

$$
\frac{f(x+tv)-f(x)}{t}
=
Df(x)[v]+\frac{r(tv)}{t}.
$$

因為

$$
\left\|\frac{r(tv)}{t}\right\|_2
=
\|v\|_2\frac{\|r(tv)\|_2}{\|tv\|_2}\to0,
$$

所以雙側方向導數存在，且

$$
D_vf(x)=Df(x)[v].
$$

因此 Fréchet 可微是所有方向導數存在的充分條件，並且方向導數對方向 $v$ 的依賴必須是線性的。反向敘述一般不成立。

### 5. 連續偏導的充分條件

**定理 7.4（常用充分條件）**  
設 $U\subset\mathbb R^n$ 開，$f:U\to\mathbb R^m$。若所有一階偏導數在 $x$ 的某個開鄰域存在，且在 $x$ 連續，則 $f$ 在 $x$ Fréchet 可微，導數的矩陣表示為 $J_f(x)$。

這是充分條件，而不是可微定義本身。證明可按輸入座標逐一改變，對各輸出分量套用單變量中值定理，再以偏導在 $x$ 的連續性控制誤差。最終得到

$$
\|f(x+h)-f(x)-J_f(x)h\|_2
\le \omega(h)\|h\|_2,
$$

其中 $\omega(h)\to0$。關鍵是此界對所有充分小的 $h$ 有效，而不是只處理固定方向。

---

## 逐步手算例題

### 例題一：直接估計餘項

令

$$
f(x,y)=
\begin{bmatrix}
x^2+xy\\
\sin y
\end{bmatrix}.
$$

求 $f$ 在 $(1,0)$ 的導數，並直接驗證餘項條件。

Jacobian 候選為

$$
J_f(x,y)=
\begin{bmatrix}
2x+y&x\\
0&\cos y
\end{bmatrix},
$$

所以

$$
J_f(1,0)=
\begin{bmatrix}
2&1\\
0&1
\end{bmatrix}.
$$

令 $h=(a,b)^T$。直接展開得

$$
f(1+a,b)-f(1,0)
=
\begin{bmatrix}
2a+b+a^2+ab\\
\sin b
\end{bmatrix}.
$$

線性主部為

$$
J_f(1,0)h=
\begin{bmatrix}
2a+b\\
b
\end{bmatrix},
$$

故餘項是

$$
r(h)=
\begin{bmatrix}
a^2+ab\\
\sin b-b
\end{bmatrix}.
$$

設 $\rho=\sqrt{a^2+b^2}$。則

$$
|a^2+ab|\le |a|^2+|a||b|\le2\rho^2.
$$

由單變量 Taylor 餘項，對所有實數 $b$，

$$
|\sin b-b|\le\frac{|b|^3}{6}\le\frac{\rho^3}{6}.
$$

所以

$$
\|r(h)\|_2
\le
2\rho^2+\frac{\rho^3}{6},
$$

進而

$$
\frac{\|r(h)\|_2}{\|h\|_2}
\le
2\rho+\frac{\rho^2}{6}\to0.
$$

此估計同時涵蓋所有方向，故證明 $f$ 在 $(1,0)$ Fréchet 可微。

### 例題二：範數函數在原點不可微

令

$$
g(x,y)=\sqrt{x^2+y^2}.
$$

假設 $g$ 在原點可微，導數為線性泛函 $L$。取 $e_1=(1,0)^T$。沿 $h=te_1$ 且 $t>0$，餘項條件要求

$$
|1-L(e_1)|=0,
$$

所以 $L(e_1)=1$。

改沿 $h=-te_1$ 且 $t>0$。因 $L(-te_1)=-tL(e_1)$，餘項比要求

$$
|1+L(e_1)|=0,
$$

所以 $L(e_1)=-1$，與前式矛盾。因此 $g$ 在原點不可 Fréchet 微分。

若只測試零映射，殘差比更直接等於

$$
\frac{g(h)}{\|h\|_2}=1,
$$

不會趨近零。

### 例題三：所有方向導數存在仍不夠

定義

$$
q(x,y)=
\begin{cases}
\dfrac{x^3}{x^2+y^2},&(x,y)\ne(0,0),\\
0,&(x,y)=(0,0).
\end{cases}
$$

對非零方向 $v=(a,b)^T$，

$$
\frac{q(ta,tb)-q(0,0)}{t}
=
\frac{a^3}{a^2+b^2}.
$$

所以每個方向導數都存在。然而此方向導數映射在 $e_1$ 上為 $1$、在 $e_2$ 上為 $0$，在 $e_1+e_2$ 上卻為 $1/2$，不符合線性映射應滿足的可加性。若 Fréchet 導數存在，所有方向導數必由同一線性映射給出，故 $q$ 在原點不可 Fréchet 微分。

---

## 實作與程式

以下程式只使用 NumPy 與 CPU，以局部資料擬合矩陣並計算方向殘差比。程式未在本章撰寫過程中執行；後文只列解析推導所得的預期現象。

擬合關係為

$$
\Delta F\approx HA^T,
$$

其中 `H` 的每一方程橫列是 $h^T$。函數介面明確使用 `(n,1)` 與 `(m,1)`，不以 NumPy 一維陣列的 `.T` 冒充列、橫方向改變。

```python
import numpy as np

def as_column(values, expected_size=None):
    col = np.asarray(values, dtype=float).reshape(-1, 1).copy()
    if expected_size is not None and col.shape != (expected_size, 1):
        raise ValueError("列向量維度不符")
    if not np.all(np.isfinite(col)):
        raise ValueError("輸入含 NaN 或無窮值")
    return col

def f_smooth(x_col):
    x_col = as_column(x_col, 2)
    x = float(x_col[0, 0])
    y = float(x_col[1, 0])
    return np.array([
        [x * x + x * y],
        [np.sin(y)]
    ], dtype=float)

def jacobian_smooth(x_col):
    x_col = as_column(x_col, 2)
    x = float(x_col[0, 0])
    y = float(x_col[1, 0])
    return np.array([
        [2.0 * x + y, x],
        [0.0, np.cos(y)]
    ], dtype=float)

def f_nondiff(x_col):
    x_col = as_column(x_col, 2)
    x = float(x_col[0, 0])
    y = float(x_col[1, 0])
    den = x * x + y * y
    if den == 0.0:
        return np.array([[0.0]], dtype=float)
    return np.array([[x ** 3 / den]], dtype=float)

def residual_ratio(fun, x0, A, h_col):
    x0 = as_column(x0)
    h_col = as_column(h_col, x0.shape[0])
    A = np.asarray(A, dtype=float)

    h_norm = float(np.linalg.norm(h_col, ord=2))
    if h_norm == 0.0:
        raise ValueError("h 必須非零；h=0 時殘差比不定義")
    if not np.all(np.isfinite(A)):
        raise ValueError("矩陣含 NaN 或無窮值")

    f0 = fun(x0)
    f1 = fun(x0 + h_col)
    if A.shape != (f0.shape[0], x0.shape[0]):
        raise ValueError("線性映射矩陣形狀不符")

    remainder = f1 - f0 - A @ h_col
    return float(np.linalg.norm(remainder, ord=2) / h_norm)

def fit_local_matrix(fun, x0, radius, directions):
    x0 = as_column(x0)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius 必須為有限正數")
    if len(directions) == 0:
        raise ValueError("至少需要一個擾動方向")

    f0 = fun(x0)
    h_rows = []
    df_rows = []

    for direction in directions:
        v_col = as_column(direction, x0.shape[0])
        v_norm = float(np.linalg.norm(v_col, ord=2))
        if v_norm == 0.0:
            raise ValueError("方向向量不可為零")

        h_col = radius * v_col / v_norm
        delta_col = fun(x0 + h_col) - f0
        h_rows.append(h_col[:, 0].copy())
        df_rows.append(delta_col[:, 0].copy())

    H = np.vstack(h_rows)
    dF = np.vstack(df_rows)

    # 解 H @ A.T ≈ dF
    A_transpose, _, rank, singular_values = np.linalg.lstsq(
        H, dF, rcond=None
    )
    return A_transpose.T, int(rank), singular_values

def maximum_sampled_ratio(fun, x0, A, radius, directions):
    x0_col = as_column(x0)
    ratios = []
    for direction in directions:
        v_col = as_column(direction, x0_col.shape[0])
        v_norm = float(np.linalg.norm(v_col, ord=2))
        if v_norm == 0.0:
            raise ValueError("方向向量不可為零")
        h_col = radius * v_col / v_norm
        ratios.append(residual_ratio(fun, x0_col, A, h_col))
    return max(ratios)

def expected_error(callable_object):
    try:
        callable_object()
    except ValueError as exc:
        print("預期捕捉:", type(exc).__name__, str(exc))
    else:
        raise AssertionError("本測試預期 ValueError")

def run_all_tests():
    x0 = np.array([[1.0], [0.0]], dtype=float)
    A_exact = jacobian_smooth(x0)
    directions = [
        [1.0, 0.0], [-1.0, 0.0],
        [0.0, 1.0], [0.0, -1.0],
        [1.0, 1.0], [1.0, -2.0]
    ]

    print("正常測試")
    print("解析 Jacobian:\n", A_exact)
    for radius in [1e-1, 1e-2, 1e-3, 1e-4]:
        A_fit, rank, singular_values = fit_local_matrix(
            f_smooth, x0, radius, directions
        )
        op_error = np.linalg.norm(A_fit - A_exact, ord=2)
        max_ratio = maximum_sampled_ratio(
            f_smooth, x0, A_exact, radius, directions
        )
        print(radius, A_fit, op_error, max_ratio,
              rank, singular_values)

    print("邊界測試：極小但非零步長")
    tiny = np.finfo(float).eps
    print(residual_ratio(
        f_smooth, x0, A_exact,
        np.array([[tiny], [0.0]], dtype=float)
    ))

    print("故障測試：零擾動")
    expected_error(lambda: residual_ratio(
        f_smooth, x0, A_exact, np.zeros((2, 1))
    ))

    print("故障測試：輸入維度錯誤")
    expected_error(lambda: f_smooth(
        np.array([[1.0], [0.0], [2.0]])
    ))

    print("故障測試：NaN 擾動")
    expected_error(lambda: residual_ratio(
        f_smooth, x0, A_exact,
        np.array([[np.nan], [0.0]])
    ))

    print("故障測試：方向數值秩不足")
    collinear = [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]]
    _, rank, singular_values = fit_local_matrix(
        f_smooth, x0, 1e-3, collinear
    )
    print("rank =", rank, "singular values =", singular_values)

    print("故障測試：非可微映射")
    origin = np.zeros((2, 1))
    zero_map = np.zeros((1, 2))
    direction = np.array([[1.0], [2.0]], dtype=float)
    direction = direction / np.linalg.norm(direction, ord=2)
    for radius in [1e-1, 1e-2, 1e-3, 1e-4]:
        print(radius, residual_ratio(
            f_nondiff, origin, zero_map, radius * direction
        ))

if __name__ == "__main__":
    run_all_tests()
```

`np.linalg.lstsq` 回傳的 `rank` 是依浮點容差判定的數值秩，不是抽象線性代數中的精確秩證明。奇異值可協助判斷方向是否接近線性相關；即使回報滿秩，若最小奇異值非常小，擬合仍可能病態。

---

## 測試與預期結果

以下均為未執行的預期結果。

### 正常測試

在 $x_0=(1,0)^T$，解析矩陣是

$$
A=
\begin{bmatrix}
2&1\\
0&1
\end{bmatrix}.
$$

六個方向在精確算術下張成 $\mathbb R^2$，預期數值程序回報 `rank == 2`。在捨入誤差尚未主導的尺度內，擬合矩陣應接近 $A$，解析矩陣的採樣殘差比應隨半徑縮小而下降。

由解析餘項可知此例的殘差比為 $O(\|h\|)$。程式結果只能與此結論相容，不能單獨證明漸近階。

### 邊界測試

以機器精度附近的非零步長計算時，餘項是由數個接近數值相減後得到，可能受到捨入誤差影響；是否出現明顯相消，依函數、工作點及浮點運算細節而定。本例不預先宣稱一定觀察到嚴重相消。若殘差比停止下降，也不能據此推翻解析可微性。

### 故障測試：零、維度與非有限輸入

傳入 $h=0$ 時，預期拋出 `ValueError`，因為殘差比分母為零。傳入三維輸入給二維函數時，預期拋出維度錯誤。傳入含 `NaN` 的擾動時，也預期在進行函數運算前被拒絕。

### 故障測試：方向秩不足

若方向全部平行於第一座標軸，預期 `lstsq` 回報 `rank == 1`。這表示程式判定方向資料數值秩不足，第二輸入方向無法由該批資料辨識。

反過來，即使回報 `rank == 2`，仍須檢查奇異值。近乎平行的方向可能被判定為滿秩，卻使估計對捨入及資料誤差高度敏感。

### 故障測試：非可微映射

對 $q(x,y)=x^3/(x^2+y^2)$ 使用零映射作候選，若單位方向為 $v=(a,b)^T$，則

$$
\frac{|q(tv)|}{\|tv\|_2}
=
\frac{|a|^3}{a^2+b^2},
$$

與半徑無關。因此預期各尺度的輸出大致不變，而不是趨近零。這只能否定零映射；排除所有線性候選仍依賴例題三的解析論證。

---

## 反例與常見陷阱

### 1. 只要求餘項趨零

正確條件是 $r(h)=o(\|h\|)$。對 $f(t)=|t|$ 與候選導數零映射，餘項雖趨零，但

$$
\frac{|r(t)|}{|t|}=1.
$$

### 2. 把 Jacobian 候選當成可微性證明

偏導存在時可排列出候選矩陣，但偏導存在不保證可微。若滿足定理 7.4 的鄰域條件，可引用充分條件；否則仍須直接控制餘項。

### 3. 只檢查有限方向

有限方向掃描可以否定候選，不能證明全方向極限。即使所有固定方向導數存在，也可能不形成線性映射。

### 4. 讓線性映射隨方向改變

Fréchet 導數是一個固定線性映射。不能沿不同方向分別挑選不同斜率，再把它們稱為同一個導數。

### 5. 把有限尺度現象當作極限定理

在幾個步長上看到殘差比下降，只能視為數值證據。有限計算未涵蓋任意小尺度，也未涵蓋所有方向。

### 6. 把數值秩視為精確秩

數值秩依賴浮點容差。檢查回報秩之外，還應觀察奇異值尺度與方向資料的條件性。

### 7. 混合單位後直接計算 Euclidean 長度

若 $h=(\Delta T,\Delta S)^T$ 的分量分別使用攝氏度與 PSU，則原始 Euclidean 長度依賴單位選擇。應先指定參考尺度 $s_T,s_S$，定義

$$
\widehat h=
\begin{bmatrix}
\Delta T/s_T\\
\Delta S/s_S
\end{bmatrix},
$$

再比較無因次擾動。

---

## AI、幾何與養殖案例

### 1. AI 模型的局部敏感度

對模型映射 $f:\mathbb R^n\to\mathbb R^m$，$Df(x)$ 描述輸入小擾動的一階傳播。不同方法提供的證據不同：

- 解析餘項界可以建立可微性。
- 自動微分依賴實作運算與分支語義。
- 局部矩陣擬合只反映有限尺度、有限方向及資料條件性。

若模型包含絕對值、最大值或離散分支，在切換面上可能不可微。軟體傳回某個約定值、單側值或次梯度，不表示 Fréchet 導數存在。

### 2. 幾何形變

考慮平面形變

$$
F(x,y)=
\begin{bmatrix}
x+0.1xy\\
y+0.05x^2
\end{bmatrix}.
$$

其局部線性映射為

$$
DF(x,y)=
\begin{bmatrix}
1+0.1y&0.1x\\
0.1x&1
\end{bmatrix}.
$$

短切向量 $h$ 的一階變化是 $DF(x,y)h$。這描述的是基點附近的局部形變，不表示有限大小的整個圖形由同一矩陣精確變換；有限位移仍含餘項。

### 3. 合成養殖感測校準

考慮純合成模型，輸入為水溫 $T$（攝氏度）與鹽度 $S$（PSU），輸出為電壓 $V$（伏特）與相位 $\phi$（弧度）：

$$
f(T,S)=
\begin{bmatrix}
0.40+0.012T+0.003S+10^{-4}TS\\
0.10+0.002T^2-0.004S
\end{bmatrix}.
$$

其導數矩陣為

$$
J_f(T,S)=
\begin{bmatrix}
0.012+10^{-4}S&0.003+10^{-4}T\\
0.004T&-0.004
\end{bmatrix}.
$$

第一橫列元素的單位分別為伏特／攝氏度及伏特／PSU；第二橫列元素的單位分別為弧度／攝氏度及弧度／PSU。不同位置的數值不能脫離單位直接比較。

取輸入、輸出參考尺度矩陣

$$
S_x=\operatorname{diag}(5\,^\circ\mathrm C,10\,\mathrm{PSU}),
$$

$$
S_y=\operatorname{diag}(0.1\,\mathrm V,0.1\,\mathrm{rad}),
$$

則無因次導數為

$$
\widehat J=S_y^{-1}J_fS_x.
$$

在無因次座標中計算 Euclidean 殘差比，才有清楚的跨分量比較意義。返回物理量時使用

$$
\Delta x=S_x\Delta\widehat x,
\qquad
\Delta y=S_y\Delta\widehat y.
$$

局部校準矩陣只描述指定工作點附近的小擾動。合成模型校準不等於現場驗證，也不保證工作範圍外有效。唯讀分析代理只能整理單位、尺度、殘差及失效警告，不直接控制投餌、加藥或其他設備。

---

## 習題

### 習題一：手算與證明

令

$$
f(x,y)=
\begin{bmatrix}
e^x\cos y\\
xy+y^2
\end{bmatrix}.
$$

求 $Df(0,0)$，並用餘項界證明 $f$ 在原點 Fréchet 可微。

### 習題二：程式與數值秩

使用本章程式回報每個半徑下的擬合矩陣、算子範數誤差、最大採樣殘差比、數值秩及奇異值。說明為何回報滿秩仍不保證擬合問題具有良好條件性。

### 習題三：路徑反例

定義

$$
p(x,y)=
\begin{cases}
\dfrac{x^2y}{x^4+y^2},&(x,y)\ne(0,0),\\
0,&(x,y)=(0,0).
\end{cases}
$$

證明沿每條直線 $y=kx$ 都有 $p(x,kx)\to0$，但 $p$ 在原點不連續，因而不可 Fréchet 微分。

### 習題四：整合與單位

某合成校準模型為

$$
z(T,C)=
\begin{bmatrix}
2T+0.5C+0.01TC\\
0.1T^2-C
\end{bmatrix}.
$$

其中 $T$ 以攝氏度計，$C$ 以毫克／公升計；第一輸出以毫伏計，第二輸出為無因次指標。

1. 求 $J_z(20,4)$。
2. 對 $\Delta T=0.1$、$\Delta C=-0.2$ 求一階輸出預測。
3. 求精確輸出增量及餘項。
4. 解釋為何原始輸入的無權重 Euclidean 長度不具單位不變性。

---

## 習題解答

### 習題一解答

Jacobian 為

$$
J_f(x,y)=
\begin{bmatrix}
e^x\cos y&-e^x\sin y\\
y&x+2y
\end{bmatrix},
$$

所以

$$
Df(0,0)=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix}.
$$

令 $h=(a,b)^T$，餘項為

$$
r(a,b)=
\begin{bmatrix}
e^a\cos b-1-a\\
ab+b^2
\end{bmatrix}.
$$

為建立第一分量的統一界，在原點取一個有限半徑閉球。這是在有限維 Euclidean 空間中；依 Heine–Borel 定理，閉且有界的球為緊緻集。函數

$$
G(a,b)=e^a\cos b
$$

的所有二階偏導在該閉球上連續，因此有界。沿原點到 $(a,b)$ 的線段套用二階 Taylor 定理，存在常數 $C>0$，使所有充分小的 $(a,b)$ 滿足

$$
|e^a\cos b-1-a|\le C(a^2+b^2).
$$

第二分量滿足

$$
|ab+b^2|
\le\frac12(a^2+b^2)+b^2
\le\frac32(a^2+b^2).
$$

令 $\rho=\sqrt{a^2+b^2}$，則存在 $C'>0$ 使

$$
\|r(a,b)\|_2\le C'\rho^2.
$$

故

$$
\frac{\|r(a,b)\|_2}{\rho}\le C'\rho\to0.
$$

因此 $f$ 在原點 Fréchet 可微。

### 習題二解答

程式中的 `rank` 是由 `np.linalg.lstsq` 按浮點容差估計的數值秩。若在二維問題中回報 `rank < 2`，程式判定方向資料數值秩不足，不能由該資料辨識完整導數矩陣。

若回報 `rank == 2`，也不能立刻斷言問題條件良好。設奇異值為

$$
\sigma_1\ge\sigma_2>0.
$$

當 $\sigma_2$ 遠小於 $\sigma_1$ 時，方向雖被判為數值滿秩，卻接近線性相關，小量捨入或資料誤差可能造成擬合矩陣的大變化。因此須同時檢查奇異值比例，而不能只看秩。

殘差比下降只涵蓋有限方向與有限半徑，仍不能取代 Fréchet 定義中的全方向極限證明。

### 習題三解答

沿 $y=kx$，當 $x\ne0$ 時，

$$
p(x,kx)
=
\frac{kx^3}{x^4+k^2x^2}
=
\frac{kx}{x^2+k^2}.
$$

若 $k\ne0$，此式在 $x\to0$ 時趨零；若 $k=0$，函數恆為零。因此每條固定直線都給出零極限。

但沿拋物線 $y=x^2$，

$$
p(x,x^2)
=
\frac{x^4}{x^4+x^4}
=
\frac12
$$

對所有 $x\ne0$ 成立。它不趨近 $p(0,0)=0$，所以 $p$ 在原點不連續。由命題 7.3，可微必連續，因此 $p$ 在原點不可 Fréchet 微分。

### 習題四解答

Jacobian 為

$$
J_z(T,C)=
\begin{bmatrix}
2+0.01C&0.5+0.01T\\
0.2T&-1
\end{bmatrix},
$$

所以

$$
J_z(20,4)=
\begin{bmatrix}
2.04&0.70\\
4&-1
\end{bmatrix}.
$$

令

$$
h=
\begin{bmatrix}
0.1\\
-0.2
\end{bmatrix}.
$$

一階輸出預測為

$$
J_z(20,4)h
=
\begin{bmatrix}
0.064\\
0.6
\end{bmatrix}.
$$

第一分量的餘項為

$$
0.01\Delta T\Delta C
=
0.01(0.1)(-0.2)
=
-0.0002,
$$

第二分量的餘項為

$$
0.1(\Delta T)^2=0.001.
$$

所以

$$
r(h)=
\begin{bmatrix}
-0.0002\\
0.001
\end{bmatrix},
$$

精確輸出增量是

$$
\begin{bmatrix}
0.0638\\
0.601
\end{bmatrix}.
$$

攝氏度與毫克／公升是不同物理量。若把濃度改以微克／公升表示，其數值會改變千倍，因此原始 Euclidean 長度不是單位不變量。應指定參考尺度 $s_T,s_C$，再計算

$$
\sqrt{
\left(\frac{\Delta T}{s_T}\right)^2+
\left(\frac{\Delta C}{s_C}\right)^2
}.
$$

---

## 本章小結

Fréchet 導數把多變量導數定義為固定線性映射：

$$
f(x+h)=f(x)+Df(x)[h]+r(h),
\qquad
\frac{\|r(h)\|}{\|h\|}\to0.
$$

核心不是單純把偏導排成矩陣，而是證明扣除線性主部後，餘項相對於輸入的一階尺度消失。若導數存在，則它唯一；可微必連續，且所有方向導數由同一線性映射給出。反之，偏導存在、有限方向測試通過，甚至所有方向導數存在，都不保證 Fréchet 可微。

矩陣擬合、方向掃描與殘差比可用來核對公式、探索有效尺度與發現反例，但受到採樣、數值秩、病態性及浮點誤差限制，不能證明全方向極限。在感測問題中，還必須記錄工作點、單位、無因次尺度及局部有效範圍。

---

## 參考來源

1. Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>  
   可供極限、連續性與嚴格證明方法的延伸閱讀。

2. MIT OpenCourseWare，*18.100A Real Analysis*：  
   <https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>  
   可參考極限、量詞與實分析證明架構。

3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：  
   <https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>  
   可參考多變量線性近似與導數矩陣的計算背景。

4. JAX，*Autodiff Cookbook*：  
   <https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html>  
   僅作自動微分概念的延伸參考；本章核心程式不依賴 JAX。

以上來源為延伸入口。本章未宣稱已通讀所有教材內容，也未執行所列程式；數值結果只作預期說明，不構成 Fréchet 可微性的證明。