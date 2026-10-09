# 第07章 Fréchet導數與餘項條件

## 學習目標與先備知識

本章把單變量的「導數是切線斜率」推廣為多變量的「導數是最佳一階線性映射」。完成本章後，讀者應能：

1. 對 $f:U\subset\mathbb R^n\to\mathbb R^m$ 寫出 Fréchet 可微的精確定義。
2. 分辨函數增量、線性主部與非線性餘項：
   $$
   f(x+h)=f(x)+Df(x)[h]+r(h).
   $$
3. 使用指定範數驗證
   $$
   \frac{\|r(h)\|}{\|h\|}\to0.
   $$
4. 證明 Fréchet 導數若存在則唯一，並證明可微必連續。
5. 理解 Fréchet 可微是對所有充分小擾動成立的敘述，而不是只沿有限條路徑成立。
6. 由解析式求導數矩陣，並以矩陣擬合及殘差比作有限精度核對。
7. 辨識「偏導存在」或「所有方向導數存在」仍不足以保證 Fréchet 可微。
8. 在具有物理單位的感測模型中標示導數元素的單位，並以無因次尺度比較誤差。

先備知識包括有限維向量範數、矩陣乘法、多變量極限、連續性與單變量導數。除非另行說明，本章輸入與輸出空間皆使用 Euclidean 範數 $\|\cdot\|_2$，矩陣使用其誘導算子範數。有限維空間中的不同範數彼此等價，因此給出相同的可微概念；然而，誤差界常數可能隨範數與維度改變。

輸入擾動 $h$ 視為 $n\times1$ 列向量。若 $f$ 有 $m$ 個輸出，導數的矩陣表示為 $J_f(x)\in\mathbb R^{m\times n}$，因此 $J_f(x)h$ 是 $m\times1$ 列向量。

---

## 問題與直覺

若工作點為 $x$，而輸入受到小擾動 $h$，我們希望以固定矩陣 $A$ 預測輸出改變：

$$
f(x+h)-f(x)\approx Ah.
$$

令預測誤差為

$$
r(h)=f(x+h)-f(x)-Ah.
$$

僅要求 $r(h)\to0$ 並不足夠。當 $f$ 在 $x$ 連續時，取 $A=0$ 就可能使增量趨零，但這不代表零映射是真正的一階導數。Fréchet 可微要求更強的相對誤差條件：

$$
\frac{\|r(h)\|_2}{\|h\|_2}\to0.
$$

也就是說，扣除線性主部後，剩餘誤差必須比擾動本身更快消失。若 $\|r(h)\|$ 約為 $\|h\|^2$，則殘差比約為 $\|h\|$，符合條件；若餘項仍與 $\|h\|$ 同階，殘差比通常不會趨近零。

這裡的「最佳線性近似」不是最小平方意義下針對一批有限資料得到的最佳矩陣。分析上的意思是：存在一個固定線性映射 $A$，使餘項為 $o(\|h\|)$，而且條件涵蓋所有充分小、非零並留在定義域內的擾動。有限方向的擬合可以估計候選矩陣，卻不能證明這個無限量詞命題。

---

## 定義、定理與推導

### 1. Fréchet 可微

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

因為 $U$ 為開集且 $x\in U$，存在 $\eta>0$ 使 $B(x,\eta)\subset U$；因此對所有 $\|h\|_2<\eta$ 的擾動，$x+h$ 自動仍在 $U$。在極限式中明寫 $x+h\in U$，則也清楚保留了定義域限制。

等價地，可寫成

$$
f(x+h)=f(x)+Df(x)[h]+r(h),
$$

其中

$$
\|r(h)\|_2=o(\|h\|_2).
$$

小 $o$ 條件的量詞形式是：對每個 $\varepsilon>0$，存在 $\delta>0$，使得只要

$$
0<\|h\|_2<\delta,\qquad x+h\in U,
$$

便有

$$
\|r(h)\|_2\le\varepsilon\|h\|_2.
$$

在標準座標下，有限維線性映射 $Df(x)$ 由唯一矩陣 $J_f(x)\in\mathbb R^{m\times n}$ 表示：

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

任取 $v\in\mathbb R^n$。若 $v=0$，結論顯然成立。若 $v\ne0$，令 $h=tv$，其中 $t\ne0$ 且 $t\to0$。由 $U$ 開，充分小的 $t$ 可保證 $x+tv\in U$。利用線性性與三角不等式，

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

而 $r_B$ 的項亦趨近零。因此 $\|(A-B)v\|_2=0$，故 $(A-B)v=0$。由於 $v$ 任意，$A=B$。證畢。

唯一性不是由「某矩陣擬合得最好」推出，而是由全方向餘項條件推出。

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

第一項在 $h\to0$ 時趨零。又因 $\|r(h)\|_2/\|h\|_2\to0$，可知 $\|r(h)\|_2\to0$。因此 $f(x+h)\to f(x)$。證畢。

所以連續是可微的必要條件，但不是充分條件。例如 $t\mapsto|t|$ 在零點連續而不可微。

### 4. 與方向導數的關係

若 $f$ 在 $x$ Fréchet 可微，則對任意固定 $v\in\mathbb R^n$，

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

所以雙側方向導數存在且

$$
D_vf(x)=Df(x)[v].
$$

因此 Fréchet 可微是「所有方向導數存在」的充分條件，而且方向導數對 $v$ 的依賴必須是線性的。反向敘述一般不成立。

### 5. 連續偏導的充分條件

**定理 7.4（常用充分條件）**  
設 $U\subset\mathbb R^n$ 開，$f:U\to\mathbb R^m$。若所有一階偏導數在 $x$ 的某個開鄰域存在，且在 $x$ 連續，則 $f$ 在 $x$ Fréchet 可微，導數的矩陣表示為 $J_f(x)$。

這是充分條件，不是必要條件。函數可能在一點可微，而其導數映射在該點不連續。

證明思路是逐一改變輸入座標，對每個輸出分量套用單變量中值定理。每一段的偏導值與其在 $x$ 的值之差，可由偏導在 $x$ 的連續性統一控制。最後得到

$$
\|f(x+h)-f(x)-J_f(x)h\|_2
\le \omega(h)\|h\|_2,
$$

其中 $\omega(h)\to0$。重點是這個界對所有小 $h$ 成立，而不是只對固定方向成立。

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

令 $h=(a,b)^T$。展開可得

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

故

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

由單變量 Taylor 餘項可得，對所有實數 $b$，

$$
|\sin b-b|\le\frac{|b|^3}{6}\le\frac{\rho^3}{6}.
$$

因此

$$
\|r(h)\|_2
\le
|a^2+ab|+|\sin b-b|
\le
2\rho^2+\frac{\rho^3}{6}.
$$

所以

$$
\frac{\|r(h)\|_2}{\|h\|_2}
\le
2\rho+\frac{\rho^2}{6}\to0.
$$

這個不等式同時控制所有方向，故確實證明 $f$ 在 $(1,0)$ Fréchet 可微。

### 例題二：範數函數在原點不可微

令

$$
g(x,y)=\sqrt{x^2+y^2}.
$$

假設 $g$ 在原點可微，導數為線性泛函 $L$。取 $h=te_1$，其中 $e_1=(1,0)^T$。若餘項比趨零，則當 $t>0$ 時，

$$
\frac{g(te_1)-g(0)-L(te_1)}{|t|}
=
|1-L(e_1)|\to0,
$$

所以必須有 $L(e_1)=1$。

改取 $h=-te_1$ 且 $t>0$。由線性性 $L(-te_1)=-tL(e_1)$，故

$$
\frac{g(-te_1)-g(0)-L(-te_1)}{t}
=
1+L(e_1).
$$

餘項條件要求其絕對值趨零，因此必須有 $L(e_1)=-1$。這與 $L(e_1)=1$ 矛盾，故 $g$ 在原點不可 Fréchet 微分。

若只測試候選 $L=0$，殘差比更直接等於

$$
\frac{g(h)}{\|h\|_2}=1,
$$

完全不會下降。

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

所以每個方向導數都存在。然而方向導數映射

$$
v\longmapsto\frac{a^3}{a^2+b^2}
$$

不是線性的：它在 $e_1$ 上為 $1$、在 $e_2$ 上為 $0$，但在 $e_1+e_2$ 上為 $1/2$，不等於 $1+0$。若 Fréchet 導數存在，方向導數必由同一線性映射給出，故 $q$ 在原點不可 Fréchet 微分。

---

## 實作與程式

以下自足程式只使用 NumPy 與 CPU。它以局部資料擬合矩陣、計算解析殘差比，並提供正常、邊界與故障測試。程式在本章撰寫過程中未執行；後文結果皆標為預期，不虛構實際輸出。

擬合資料使用

$$
\Delta F\approx HA^T,
$$

其中 `H` 的每一方程橫列是 $h^T$。函數輸入與輸出明確使用 `(n,1)` 與 `(m,1)`；NumPy 一維陣列只用於組裝資料，不以 `.T` 冒充形狀改變。

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

    # H @ A.T ≈ dF
    A_transpose, _, rank, singular_values = np.linalg.lstsq(
        H, dF, rcond=None
    )
    A = A_transpose.T
    return A, int(rank), singular_values

def maximum_sampled_ratio(fun, x0, A, radius, directions):
    ratios = []
    for direction in directions:
        v_col = as_column(direction, as_column(x0).shape[0])
        norm_v = float(np.linalg.norm(v_col, ord=2))
        if norm_v == 0.0:
            raise ValueError("方向向量不可為零")
        h_col = radius * v_col / norm_v
        ratios.append(residual_ratio(fun, x0, A, h_col))
    return max(ratios)

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
        print(radius, A_fit, op_error, max_ratio, rank, singular_values)

    print("邊界測試：極小但非零步長")
    tiny = np.finfo(float).eps
    h_tiny = np.array([[tiny], [0.0]])
    print(residual_ratio(f_smooth, x0, A_exact, h_tiny))

    print("故障測試：h=0")
    try:
        residual_ratio(f_smooth, x0, A_exact, np.zeros((2, 1)))
    except ValueError as exc:
        print(type(exc).__name__, str(exc))

    print("故障測試：方向秩不足")
    collinear = [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]]
    _, rank, _ = fit_local_matrix(f_smooth, x0, 1e-3, collinear)
    print("rank =", rank)

    print("故障測試：非可微函數")
    origin = np.zeros((2, 1))
    zero_map = np.zeros((1, 2))
    direction = np.array([[1.0], [2.0]], dtype=float)
    direction /= np.linalg.norm(direction, ord=2)
    for radius in [1e-1, 1e-2, 1e-3, 1e-4]:
        h_col = radius * direction
        print(radius, residual_ratio(
            f_nondiff, origin, zero_map, h_col
        ))

if __name__ == "__main__":
    run_all_tests()
```

局部矩陣擬合要求擾動方向張成輸入空間。若方向資料秩不足，最小平方仍可能傳回一個最小範數答案，但未被激發的輸入方向無法辨識；此時不應把結果稱為完整 Jacobian。

---

## 測試與預期結果

以下均為未執行的預期結果。

### 正常測試

在 $x_0=(1,0)^T$，解析矩陣為

$$
A=
\begin{bmatrix}
2&1\\
0&1
\end{bmatrix}.
$$

六個方向張成 $\mathbb R^2$，所以預期 `rank == 2`。在浮點誤差尚未主導的尺度內，擬合矩陣應接近 $A$，解析矩陣的最大採樣殘差比也應隨半徑縮小而下降。

這個函數的餘項主要是二階項，所以殘差比在適當尺度內通常表現出近似一階下降。然而有限資料不足以證明真正的漸近階，故不把列印結果稱為收斂階證明。

### 邊界測試

當步長取機器精度附近時，`f(x0+h)-f(x0)` 可能發生嚴重相消。殘差比可能停止下降或上升。這表示有限精度計算失去解析能力，不表示數學上的導數不存在。

若直接傳入 $h=0$，程式預期拋出 `ValueError`。極限中的 $h$ 必須非零；不能把零除當作一個可計算的極限值。

### 故障測試：秩不足

只使用平行於第一座標軸的方向時，預期 `rank == 1`。第二縱行對應的輸入敏感度不能由這批資料識別。訓練殘差即使很小，也只表示矩陣符合已採樣方向。

### 故障測試：非可微映射

對 $q(x,y)=x^3/(x^2+y^2)$ 使用零映射作候選，若單位方向為 $v=(a,b)^T$，則

$$
\frac{|q(tv)|}{\|tv\|_2}
=
\frac{|a|^3}{a^2+b^2}.
$$

右側與半徑 $|t|$ 無關，所以預期四個尺度輸出大致相同，而不趨近零。這個測試支持「零映射不是導數」，而解析的非線性方向導數論證才排除了所有可能的線性映射。

---

## 反例與常見陷阱

### 1. 只要求餘項趨零

正確條件是 $r(h)=o(\|h\|)$，不是只有 $r(h)\to0$。對 $f(t)=|t|$ 與候選導數零映射，餘項雖趨零，但

$$
\frac{|r(t)|}{|t|}=1.
$$

### 2. 把 Jacobian 候選當成證明

偏導若存在，可以排列成一個候選矩陣；但偏導存在本身不保證可微。若符合定理 7.4 的鄰域連續條件，可以引用該充分條件；否則必須直接控制餘項或使用其他可微性定理。

### 3. 只檢查座標軸或有限方向

有限方向掃描可以推翻一個候選，卻不能證明全方向極限。甚至所有固定方向導數都存在，也可能不形成線性映射；例題三正是如此。

### 4. 忽略候選映射必須固定

不能沿不同方向選不同矩陣。Fréchet 導數是一個固定線性映射，必須同時描述所有足夠小的輸入擾動。

### 5. 把有限尺度近似當作漸近定理

在 $10^{-2}$ 到 $10^{-5}$ 之間觀察到殘差下降，只能說該尺度上的數值與理論相容。它沒有涵蓋任意小的 $h$，也沒有涵蓋未採樣方向。

### 6. 混合物理單位後直接計算長度

若 $h=(\Delta T,\Delta S)^T$ 的兩個分量分別使用攝氏度與 PSU，則未加權 Euclidean 長度取決於單位選擇。應先指定參考尺度 $s_T,s_S>0$，再定義

$$
\widehat h=
\begin{bmatrix}
\Delta T/s_T\\
\Delta S/s_S
\end{bmatrix}.
$$

輸出若也混合伏特與弧度，應同樣無因次化後才比較殘差。

---

## AI、幾何與養殖案例

### 1. AI 模型的局部敏感度

對模型映射 $f:\mathbb R^n\to\mathbb R^m$，$Df(x)$ 描述輸入擾動的一階傳播。解析推導、自動微分與數值擬合屬於不同證據：

- 解析餘項界可以建立可微性。
- 自動微分依賴實作中的運算規則與分支語義。
- 局部擬合只反映有限尺度、有限方向及資料秩。

若模型包含絕對值、最大值或離散分支，在切換面上可能不可微。軟體傳回某個次梯度或單側規則，不表示 Fréchet 導數存在。

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

短切向量 $h$ 的一階變化是 $DF(x,y)h$。這只描述基點附近的局部形變，不表示有限大小的整個圖形都由同一矩陣精確變換；有限位移仍含非線性餘項。

### 3. 合成養殖感測校準

考慮純合成模型，輸入為水溫 $T$（攝氏度）及鹽度 $S$（PSU），輸出為電壓 $V$（伏特）及相位 $\phi$（弧度）：

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

第一橫列元素的單位分別是伏特／攝氏度與伏特／PSU；第二橫列元素的單位分別是弧度／攝氏度與弧度／PSU。不同位置的數值不能脫離單位直接比較。

取輸入參考尺度

$$
S_x=\operatorname{diag}(5\,^\circ\mathrm C,10\,\mathrm{PSU}),
$$

輸出參考尺度

$$
S_y=\operatorname{diag}(0.1\,\mathrm V,0.1\,\mathrm{rad}),
$$

則無因次導數為

$$
\widehat J=S_y^{-1}J_fS_x.
$$

在無因次座標中計算 Euclidean 殘差比，才有清楚的跨分量比較意義。若要返回物理量，使用 $\Delta x=S_x\Delta\widehat x$ 與 $\Delta y=S_y\Delta\widehat y$。

此矩陣只描述指定工作點附近的局部變化。合成模型校準不等於現場驗證，也不保證工作範圍外仍有效。唯讀分析代理可以整理單位、尺度、殘差與失效警告，但不應直接控制投餌、加藥或其他設備。

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

求 $Df(0,0)$，並以餘項界證明 $f$ 在原點 Fréchet 可微。

### 習題二：程式與證據界線

使用本章程式，在每個半徑回報擬合矩陣、與解析矩陣的算子範數誤差、最大採樣殘差比及資料矩陣秩。說明哪個輸出能偵測方向集合沒有張成輸入空間，以及為何殘差比下降仍不是可微性證明。

### 習題三：路徑反例

定義

$$
p(x,y)=
\begin{cases}
\dfrac{x^2y}{x^4+y^2},&(x,y)\ne(0,0),\\
0,&(x,y)=(0,0).
\end{cases}
$$

證明沿每條直線 $y=kx$ 都有 $p(x,kx)\to0$，但 $p$ 在原點不連續。

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
3. 求精確輸出增量與餘項。
4. 解釋為何原始輸入的無權重 Euclidean 長度沒有單位不變性。

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

令 $h=(a,b)^T$。餘項是

$$
r(a,b)=
\begin{bmatrix}
e^a\cos b-1-a\\
ab+b^2
\end{bmatrix}.
$$

為嚴格建立第一分量的界，在原點取一個閉球 $\overline B(0,\eta)$。函數

$$
G(a,b)=e^a\cos b
$$

的所有二階偏導在此緊緻閉球上連續，故有界。對原點到 $(a,b)$ 的線段套用二階 Taylor 定理，可得某個常數 $C>0$，使所有充分小的 $(a,b)$ 滿足

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

`run_all_tests` 已完整回報四項資料。`rank` 用來判斷方向資料是否張成輸入空間；在二維問題中，若 `rank < 2`，完整導數矩陣便不可由該資料辨識。

算子範數誤差衡量擬合矩陣與已知解析矩陣的差距；最大殘差比只涵蓋程式列出的有限方向。即使它在所有列出尺度都下降，也沒有涵蓋任意小的擾動與所有方向，因此不能取代定義中的極限證明。

若另行撰寫正規化片段，應使用新陣列，避免原地修改呼叫端資料，例如：

```python
v_col = np.asarray(v, dtype=float).reshape(-1, 1).copy()
v_col = v_col / np.linalg.norm(v_col, ord=2)
```

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

對所有 $x\ne0$ 成立。它不趨近 $p(0,0)=0$，所以 $p$ 在原點不連續。由可微必連續，$p$ 在原點不可能 Fréchet 可微。所有直線測試一致仍不足以證明多變量極限存在。

### 習題四解答

Jacobian 為

$$
J_z(T,C)=
\begin{bmatrix}
2+0.01C&0.5+0.01T\\
0.2T&-1
\end{bmatrix}.
$$

故

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

第一分量的二階餘項為

$$
0.01\Delta T\Delta C
=
0.01(0.1)(-0.2)
=
-0.0002.
$$

第二分量的二階餘項為

$$
0.1(\Delta T)^2=0.001.
$$

因此

$$
r(h)=
\begin{bmatrix}
-0.0002\\
0.001
\end{bmatrix},
$$

而精確輸出增量是

$$
\begin{bmatrix}
0.0638\\
0.601
\end{bmatrix}.
$$

攝氏度與毫克／公升屬於不同物理量。若把濃度由毫克／公升改寫成微克／公升，其數值會改變千倍，因此原始 Euclidean 長度不是單位不變量。應先選參考尺度 $s_T,s_C$，再以

$$
\sqrt{\left(\frac{\Delta T}{s_T}\right)^2+
\left(\frac{\Delta C}{s_C}\right)^2}
$$

計算無因次擾動大小。

---

## 本章小結

Fréchet 導數把導數定義為一個固定線性映射：

$$
f(x+h)=f(x)+Df(x)[h]+r(h),
\qquad
\frac{\|r(h)\|}{\|h\|}\to0.
$$

其核心不是把偏導排列成矩陣，而是證明扣除線性主部後，餘項相對於輸入的一階尺度消失。若導數存在，則它唯一；可微必連續，且所有方向導數都由同一線性映射給出。反之，偏導存在、有限方向測試通過，甚至所有方向導數存在，都不必然保證 Fréchet 可微。

矩陣擬合、方向掃描與殘差比可核對公式、探索有效尺度及發現反例，但受採樣、秩與浮點精度限制，不能證明全方向極限。在感測問題中，還須記錄工作點、單位、無因次尺度與局部有效範圍，避免把數值線性化誤解為全域模型或現場安全保證。

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