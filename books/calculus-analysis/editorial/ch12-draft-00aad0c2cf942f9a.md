# 第12章 隱函數定理與靈敏度

## 學習目標與先備知識

許多模型不是直接給出輸出公式 $y=g(x)$，而是以平衡條件

$$
F(x,y)=0
$$

間接決定 $y$。本章研究：在什麼條件下，這個方程能在某一解附近把 $y$ 唯一表示成 $x$ 的可微函數？當參數 $x$ 改變時，解 $y$ 有多敏感？條件失效時，可能出現哪些分支、折疊或不可辨識現象？

完成本章後，讀者應能：

1. 正確陳述有限維隱函數定理，明確交代開集、正則性與可逆條件。
2. 區分「某點存在解」、「局部唯一函數分支」與「全域唯一解」。
3. 推導並使用靈敏度公式
   $$
   Dg(x)=-\bigl(D_yF(x,g(x))\bigr)^{-1}D_xF(x,g(x)).
   $$
4. 不顯式形成逆矩陣，而以線性方程求取靈敏度。
5. 解釋 $D_yF$ 接近奇異時的靈敏度放大與數值困難。
6. 以圓、耦合平衡方程及合成養殖感測模型辨認局部分支。
7. 使用殘差、條件數與有限差分作數值稽核，同時不把有限測試誤稱為數學證明。

先備知識包括：Fréchet 導數、Jacobian、鏈式法則、矩陣可逆性、算子範數，以及前章反函數定理。全文採列向量（column vector）表示輸入與輸出。若

$$
F:U\subset\mathbb R^n\times\mathbb R^m\to\mathbb R^m,
$$

則

$$
D_xF(x,y)\in\mathbb R^{m\times n},
\qquad
D_yF(x,y)\in\mathbb R^{m\times m}.
$$

以下有限維範數預設為 Euclidean 範數，矩陣範數為其誘導二範數；涉及物理量時將另行處理尺度與單位。

---

## 問題與直覺

考慮單一方程

$$
F(x,y)=x^2+y^2-1=0.
$$

它描述單位圓。若只看點 $(0,1)$ 附近，圓的上半部可以寫成

$$
y=g(x)=\sqrt{1-x^2}.
$$

但整個圓不能由單一函數 $y=g(x)$ 表示，因為同一個 $x\in(-1,1)$ 通常對應兩個 $y$。因此「局部可表示」不等於「全域可表示」。

在 $(0,1)$，對 $y$ 的偏導數為

$$
D_yF(0,1)=2,
$$

它可逆。直覺上，在該點附近稍微改變 $y$，方程值會以非零的一階速率改變，因此可藉由調整 $y$ 抵銷 $x$ 的擾動。

若在右端點 $(1,0)$，則

$$
D_yF(1,0)=0.
$$

圓在此具有垂直切線，無法在該點附近把圓寫成可微的 $y=g(x)$。但這不代表圓不存在，也不代表不能改用 $x=h(y)$；因為此時 $D_xF(1,0)=2$，所以可局部表示成 $x=\sqrt{1-y^2}$。

將方程在解 $(x_0,y_0)$ 附近線性化：

$$
F(x_0+\Delta x,y_0+\Delta y)
\approx
F(x_0,y_0)
+D_xF(x_0,y_0)\Delta x
+D_yF(x_0,y_0)\Delta y.
$$

因為 $F(x_0,y_0)=0$，為了繼續滿足平衡，線性項應近似滿足

$$
D_yF\,\Delta y=-D_xF\,\Delta x.
$$

若 $D_yF$ 可逆，便得到

$$
\Delta y\approx
-\bigl(D_yF\bigr)^{-1}D_xF\,\Delta x.
$$

這就是隱函數靈敏度公式的線性代數核心。定理的工作不只是在形式上「移項」；它還要證明附近確實存在唯一函數分支，而且該分支可微。

---

## 定義、定理與推導

### 隱式解與局部分支

**定義（隱式解集合）。**  
對映射 $F:U\subset\mathbb R^n\times\mathbb R^m\to\mathbb R^m$，集合

$$
\mathcal Z=\{(x,y)\in U:F(x,y)=0\}
$$

稱為 $F$ 的零集合或隱式解集合。

**定義（局部函數分支）。**  
若 $(a,b)\in\mathcal Z$，且存在 $a$ 的鄰域 $V\subset\mathbb R^n$、$b$ 的鄰域 $W\subset\mathbb R^m$ 及函數 $g:V\to W$，使得

$$
\mathcal Z\cap(V\times W)
=
\{(x,g(x)):x\in V\},
$$

則稱 $g$ 是通過 $(a,b)$ 的局部隱函數分支。

此定義含有局部唯一性：在指定的 $V\times W$ 內，每個 $x$ 恰有一個相應的 $y$。它沒有聲稱 $W$ 外沒有其他解。

### 隱函數定理

**定理（有限維 $C^1$ 隱函數定理）。**  
令 $U\subset\mathbb R^n\times\mathbb R^m$ 為開集，且

$$
F:U\to\mathbb R^m
$$

為 $C^1$ 映射。設 $(a,b)\in U$ 滿足

$$
F(a,b)=0,
$$

並假設方陣

$$
D_yF(a,b)\in\mathbb R^{m\times m}
$$

可逆。則存在開鄰域 $V\subset\mathbb R^n$、$W\subset\mathbb R^m$，以及唯一的 $C^1$ 函數 $g:V\to W$，使得

$$
g(a)=b
$$

且

$$
F(x,g(x))=0,\qquad x\in V.
$$

此外，在縮小鄰域後，$V\times W$ 內所有零點恰為 $(x,g(x))$，且

$$
Dg(x)
=
-\bigl(D_yF(x,g(x))\bigr)^{-1}D_xF(x,g(x)).
$$

這裡 $Dg(x)\in\mathbb R^{m\times n}$。公式右側形狀為

$$
(m\times m)(m\times n)=m\times n.
$$

定理條件是**充分條件**。$D_yF(a,b)$ 不可逆時，不能套用此版本的定理；但不能反過來斷言局部函數必不存在。例如

$$
F(x,y)=(y-x)^3
$$

在 $(0,0)$ 滿足 $D_yF=0$，然而零集合仍是唯一的光滑函數 $y=x$。失效的是定理的保證，而非結論必然為假。

### 從反函數定理得到存在性

定義

$$
H(x,y)=(x,F(x,y)).
$$

其 Jacobian 具有分塊形式

$$
DH(a,b)=
\begin{pmatrix}
I_n & 0\\
D_xF(a,b) & D_yF(a,b)
\end{pmatrix}.
$$

這是分塊下三角矩陣。因 $I_n$ 與 $D_yF(a,b)$ 都可逆，所以 $DH(a,b)$ 可逆。反函數定理因此保證 $H$ 在 $(a,b)$ 附近有 $C^1$ 局部反函數。

對附近的 $(x,z)$，可將反函數寫成

$$
H^{-1}(x,z)=(x,\psi(x,z)).
$$

令 $z=0$ 並定義 $g(x)=\psi(x,0)$，便有

$$
H(x,g(x))=(x,0),
$$

也就是 $F(x,g(x))=0$。局部反函數的唯一性同時給出局部分支唯一性。這是隱函數定理的標準證明架構；完整反函數定理已是前章結果。

### 靈敏度公式

由恆等式

$$
F(x,g(x))=0
$$

對 $x$ 微分。鏈式法則給出

$$
D_xF(x,g(x))
+
D_yF(x,g(x))Dg(x)=0.
$$

因此

$$
D_yF(x,g(x))Dg(x)=-D_xF(x,g(x)).
$$

左乘逆矩陣即可得公式。不過數值實作不應顯式計算逆矩陣；應解矩陣方程

$$
D_yF\,S=-D_xF,
\qquad S=Dg(x).
$$

### 小命題：局部導數的唯一性

**命題。**  
設 $F$ 在 $(a,b)$ 可微，$F(a,b)=0$。假設 $g_1,g_2$ 都在 $a$ 可微，且對 $x$ 接近 $a$，

$$
F(x,g_i(x))=0,\qquad g_i(a)=b.
$$

若 $D_yF(a,b)$ 可逆，則

$$
Dg_1(a)=Dg_2(a).
$$

**證明。**  
對每個 $i\in\{1,2\}$，由鏈式法則，

$$
D_xF(a,b)+D_yF(a,b)Dg_i(a)=0.
$$

兩式相減得

$$
D_yF(a,b)\bigl(Dg_1(a)-Dg_2(a)\bigr)=0.
$$

左乘 $\bigl(D_yF(a,b)\bigr)^{-1}$，得到

$$
Dg_1(a)-Dg_2(a)=0.
$$

故 $Dg_1(a)=Dg_2(a)$。證畢。

此命題只證明導數必須相同，並未單獨證明分支存在。存在性來自隱函數定理。

### 一階誤差與條件放大

設量測參數發生小擾動 $\delta x$，則

$$
\delta y
=
Dg(x)\delta x+o(\|\delta x\|).
$$

因而

$$
\|\delta y\|
\le
\|Dg(x)\|\,\|\delta x\|+o(\|\delta x\|).
$$

再由靈敏度公式，

$$
\|Dg(x)\|
\le
\|D_yF(x,g(x))^{-1}\|\,\|D_xF(x,g(x))\|.
$$

若 $D_yF$ 的最小奇異值很小，則

$$
\|D_yF^{-1}\|_2
=
\frac{1}{\sigma_{\min}(D_yF)}
$$

很大，參數誤差便可能被放大。這是上界與局部一階分析，不代表每個擾動方向都一定達到最大放大量。

---

## 逐步手算例題

### 例一：圓的分支與垂直切線

令

$$
F(x,y)=x^2+y^2-1.
$$

在上半圓且 $y>0$ 時，

$$
D_xF=2x,\qquad D_yF=2y.
$$

因此

$$
g'(x)=-\frac{D_xF}{D_yF}=-\frac{x}{y}.
$$

在點

$$
(x,y)=\left(\frac35,\frac45\right)
$$

有

$$
g'\left(\frac35\right)
=
-\frac{3/5}{4/5}
=
-\frac34.
$$

若 $\Delta x=0.01$，一階預測為

$$
\Delta y\approx-\frac34(0.01)=-0.0075.
$$

原點附近的上分支為 $g(x)=\sqrt{1-x^2}$，其直接微分結果

$$
g'(x)=-\frac{x}{\sqrt{1-x^2}}
$$

與隱函數公式一致。

當 $x\to1^-$ 時，$y=\sqrt{1-x^2}\to0^+$，故

$$
|g'(x)|=\frac{|x|}{|y|}\to\infty.
$$

這說明接近垂直切線時，以 $x$ 為輸入、$y$ 為輸出的靈敏度急遽放大。在 $(1,0)$，$D_yF=0$，定理不能建立 $y=g(x)$；但 $D_xF=2$，可交換角色建立 $x=h(y)$，而

$$
h'(y)=-\frac{D_yF}{D_xF}=-y
$$

在 $y=0$ 等於零。

### 例二：二元耦合平衡方程

令參數 $x=(p,q)^T\in\mathbb R^2$，未知量 $y=(u,v)^T\in\mathbb R^2$，並定義

$$
F(x,y)=
\begin{pmatrix}
u+v+p-3\\
u^2+2v-q-3
\end{pmatrix}.
$$

取

$$
x_0=
\begin{pmatrix}0\\0\end{pmatrix},
\qquad
y_0=
\begin{pmatrix}1\\2\end{pmatrix}.
$$

代入可得 $F(x_0,y_0)=0$。兩個偏 Jacobian 為

$$
D_xF=
\begin{pmatrix}
1&0\\
0&-1
\end{pmatrix},
\qquad
D_yF=
\begin{pmatrix}
1&1\\
2u&2
\end{pmatrix}.
$$

在 $u=1$ 時，

$$
D_yF(x_0,y_0)=
\begin{pmatrix}
1&1\\
2&2
\end{pmatrix},
$$

其行列式為零。這個選點其實是秩失效點，無法套用定理。這提醒我們：找到平衡解後仍須檢查 Jacobian。

改取 $y_0=(2,1)^T$，並將第二式常數改為 $u^2+2v-q-6$，即考慮

$$
\widetilde F(x,y)=
\begin{pmatrix}
u+v+p-3\\
u^2+2v-q-6
\end{pmatrix}.
$$

此時 $\widetilde F(0,0,2,1)=0$，而

$$
D_y\widetilde F=
\begin{pmatrix}
1&1\\
4&2
\end{pmatrix},
\qquad
\det(D_y\widetilde F)=-2\ne0.
$$

令 $S=Dg(x_0)$，解

$$
\begin{pmatrix}
1&1\\
4&2
\end{pmatrix}S
=
-
\begin{pmatrix}
1&0\\
0&-1
\end{pmatrix}
=
\begin{pmatrix}
-1&0\\
0&1
\end{pmatrix}.
$$

矩陣逆為

$$
\begin{pmatrix}
1&1\\
4&2
\end{pmatrix}^{-1}
=
\begin{pmatrix}
-1&1/2\\
2&-1/2
\end{pmatrix},
$$

所以

$$
S=
\begin{pmatrix}
1&1/2\\
-2&-1/2
\end{pmatrix}.
$$

因此，若 $\Delta p=0.01$、$\Delta q=-0.02$，則

$$
\Delta y
\approx
S
\begin{pmatrix}
0.01\\-0.02
\end{pmatrix}
=
\begin{pmatrix}
0\\-0.01
\end{pmatrix}.
$$

這只是局部一階預測；有限擾動下仍有非線性餘項。

---

## 實作與程式

以下程式只使用 Python 標準庫與 NumPy，在 CPU 上求解合成平衡方程。Newton 步驟與靈敏度都使用 `numpy.linalg.solve`，不顯式計算矩陣逆。程式亦檢查殘差、步長及最小奇異值。

```python
import numpy as np

def F(x, y):
    """x=[p,q], y=[u,v]，回傳兩個平衡殘差。"""
    p, q = x
    u, v = y
    return np.array([
        u + v + p - 3.0,
        u*u + 2.0*v - q - 6.0
    ], dtype=float)

def DxF(x, y):
    return np.array([
        [1.0,  0.0],
        [0.0, -1.0]
    ], dtype=float)

def DyF(x, y):
    u, v = y
    return np.array([
        [1.0, 1.0],
        [2.0*u, 2.0]
    ], dtype=float)

def solve_equilibrium(x, y0, tol=1e-12, max_iter=30,
                      singular_tol=1e-10):
    x = np.asarray(x, dtype=float).reshape(2)
    y = np.asarray(y0, dtype=float).reshape(2).copy()

    history = []
    for k in range(max_iter):
        r = F(x, y)
        A = DyF(x, y)
        svals = np.linalg.svd(A, compute_uv=False)
        sigma_min = float(svals[-1])
        residual = float(np.linalg.norm(r, 2))

        if residual <= tol:
            return y, history, {
                "status": "converged",
                "residual": residual,
                "sigma_min": sigma_min
            }

        if sigma_min <= singular_tol:
            raise np.linalg.LinAlgError(
                "DyF is singular or too close to singular"
            )

        step = np.linalg.solve(A, -r)
        step_norm = float(np.linalg.norm(step, 2))
        history.append((k, residual, step_norm, sigma_min))
        y = y + step

        if step_norm <= tol * (1.0 + np.linalg.norm(y, 2)):
            final_residual = float(np.linalg.norm(F(x, y), 2))
            return y, history, {
                "status": "small_step",
                "residual": final_residual,
                "sigma_min": float(
                    np.linalg.svd(DyF(x, y),
                                  compute_uv=False)[-1]
                )
            }

    return y, history, {
        "status": "max_iter",
        "residual": float(np.linalg.norm(F(x, y), 2)),
        "sigma_min": float(
            np.linalg.svd(DyF(x, y), compute_uv=False)[-1]
        )
    }

def sensitivity(x, y, singular_tol=1e-10):
    A = DyF(x, y)
    sigma_min = float(np.linalg.svd(A, compute_uv=False)[-1])
    if sigma_min <= singular_tol:
        raise np.linalg.LinAlgError(
            "Sensitivity is not certified: DyF is near singular"
        )
    return np.linalg.solve(A, -DxF(x, y))

def finite_difference_check(x, y, direction, eps=1e-6):
    direction = np.asarray(direction, dtype=float).reshape(2)
    S = sensitivity(x, y)
    predicted = S @ direction

    xp = x + eps * direction
    yp, _, info = solve_equilibrium(xp, y, tol=1e-13)
    observed = (yp - y) / eps
    error = np.linalg.norm(observed - predicted, 2)

    return {
        "predicted": predicted,
        "observed": observed,
        "error": float(error),
        "solver_status": info["status"]
    }

if __name__ == "__main__":
    x0 = np.array([0.0, 0.0])
    y0 = np.array([2.0, 1.0])

    y, history, info = solve_equilibrium(x0, [2.1, 0.9])
    print("solution =", y)
    print("info =", info)
    print("sensitivity =\n", sensitivity(x0, y))

    check = finite_difference_check(
        x0, y, direction=np.array([1.0, -2.0])
    )
    print("finite difference check =", check)
```

此處 NumPy 的一維陣列只作儲存；數學上 `direction`、`y` 都視為列向量。`S` 是 $2\times2$ 矩陣，`S @ direction` 是二維列向量的儲存表示。

Newton 法只是求解器，不是隱函數定理的證明。即使數值迭代成功，也只能提供特定資料下的數值證據；定理結論來自 $C^1$ 與可逆 Jacobian 等分析條件。

---

## 測試與預期結果

以下結果均為解析推導或程式的**預期結果**；本章未執行程式。

### 正常測試

輸入

```python
x0 = np.array([0.0, 0.0])
y_initial = np.array([2.1, 0.9])
```

預期 Newton 法收斂到接近

```text
solution = [2. 1.]
```

殘差應接近設定容許值。靈敏度預期為

```text
[[ 1.   0.5]
 [-2.  -0.5]]
```

方向 $(1,-2)^T$ 的解析方向靈敏度為

$$
\begin{pmatrix}
1&1/2\\
-2&-1/2
\end{pmatrix}
\begin{pmatrix}
1\\-2
\end{pmatrix}
=
\begin{pmatrix}
0\\-1
\end{pmatrix}.
$$

有限差分觀察值應接近 $(0,-1)^T$，但不應期待浮點下完全相等。

### 邊界測試：接近秩失效

由

$$
D_yF=
\begin{pmatrix}
1&1\\
2u&2
\end{pmatrix},
\qquad
\det(D_yF)=2-2u,
$$

可知 $u\to1$ 時矩陣接近奇異。選擇解附近使 $u$ 接近 $1$，預期最小奇異值下降，靈敏度範數上升。這是「接近折疊點」的警示，而不是自動證明存在真正分岔；分岔判定需要更多結構與高階分析。

### 故障測試：奇異 Jacobian

若直接呼叫

```python
sensitivity(
    np.array([0.0, 0.0]),
    np.array([1.0, 2.0])
)
```

則

$$
D_yF=
\begin{pmatrix}
1&1\\
2&2
\end{pmatrix}
$$

奇異。預期函數拋出 `LinAlgError`，而非回傳巨大的不可靠數值。

### 輸入形狀故障

若輸入長度不是二，例如 `x=[0,0,0]`，`reshape(2)` 預期拋出 `ValueError`。這是介面故障，不是數學上的秩失效。

---

## 反例與常見陷阱

### 陷阱一：把局部唯一誤作全域唯一

圓在 $(0,1)$ 附近有唯一上分支，但對 $x=0$，全域上存在 $y=1$ 與 $y=-1$ 兩個解。隱函數定理只控制選定點附近的解。

### 陷阱二：把可逆條件說成必要條件

對

$$
F(x,y)=(y-x)^3,
$$

有唯一解 $y=x$，但在零集合上

$$
D_yF=3(y-x)^2=0.
$$

因此 $D_yF$ 可逆不是局部函數存在的必要條件。它是能保證 $C^1$ 分支與靈敏度公式的標準充分條件。

### 陷阱三：在奇異點直接除法

單一方程常寫成

$$
\frac{dy}{dx}=-\frac{F_x}{F_y}.
$$

此式只在 $F_y\ne0$ 且存在相應可微分支時成立。於 $F_y=0$ 處直接除法沒有意義。

### 陷阱四：顯式形成逆矩陣

數學公式中的 $(D_yF)^{-1}$ 不要求程式先算逆矩陣。實作應解

$$
D_yF\,S=-D_xF.
$$

這通常更穩定，也更容易利用矩陣結構。

### 陷阱五：殘差小便宣稱靈敏度可靠

$F(x,y)$ 很小只說明候選點近似滿足方程。若 $D_yF$ 接近奇異，解仍可能對微小誤差極敏感。報告至少應同時包含殘差與最小奇異值或適當條件估計。

### 陷阱六：混合單位直接取 Euclidean 長度

若 $y$ 的一個分量以攝氏度表示，另一個以毫克每公升表示，則無權重的

$$
\|\delta y\|_2
$$

沒有直接物理解釋。應先選尺度矩陣，例如

$$
\widehat y=S_y^{-1}(y-y_{\rm ref}),
\qquad
\widehat x=S_x^{-1}(x-x_{\rm ref}),
$$

再分析無因次靈敏度

$$
D\widehat g
=
S_y^{-1}Dg\,S_x.
$$

尺度選擇必須記錄，回到物理量時再乘回 $S_y$。

---

## AI、幾何與養殖案例

### AI 模型中的隱式層

某些模型以平衡方程定義隱藏狀態：

$$
F(\theta,z)=z-\tanh(W(\theta)z+b(\theta))=0.
$$

若 $D_zF$ 在某平衡點可逆，則局部上 $z=z(\theta)$，而

$$
Dz(\theta)
=
-\bigl(D_zF\bigr)^{-1}D_\theta F.
$$

實際反向計算常解轉置線性系統，而不是形成完整 Jacobian 或逆矩陣。這是隱式微分的應用，但局部可逆性仍須檢查；求解器找到一個固定點，不代表固定點全域唯一。

### 幾何約束

曲線 $F(x,y)=0$ 在 $F_y\ne0$ 處可用 $x$ 作局部座標；在 $F_x\ne0$ 處可用 $y$ 作局部座標。圓的垂直切線並非幾何物件壞掉，而是所選座標圖失效。適時交換自變量與應變量，可避免無窮斜率。

### 合成養殖平衡模型

考慮只作教學用途的合成模型。令參數

$$
x=
\begin{pmatrix}
T\\R
\end{pmatrix},
$$

其中 $T$ 是水溫，單位 $\mathrm{^\circ C}$；$R$ 是合成曝氣設定，單位 $\mathrm{L\,min^{-1}}$。未知平衡狀態

$$
y=
\begin{pmatrix}
O\\C
\end{pmatrix},
$$

其中 $O$ 是溶氧，單位 $\mathrm{mg\,L^{-1}}$；$C$ 是一個合成無因次代謝指標。

平衡方程可寫成

$$
F(T,R,O,C)=
\begin{pmatrix}
k_O(O-O_*)+\alpha C-\beta(R-R_*)\\
C^2+\gamma(O-O_*)-\eta(T-T_*)-C_*
\end{pmatrix}
=0.
$$

則

$$
D_yF=
\begin{pmatrix}
k_O&\alpha\\
\gamma&2C
\end{pmatrix},
\qquad
D_xF=
\begin{pmatrix}
0&-\beta\\
-\eta&0
\end{pmatrix}.
$$

每個 Jacobian 元素的單位都是「該方程輸出單位除以輸入單位」。若兩條方程採不同物理尺度，直接比較矩陣元素大小可能誤導，應先無因次化。

當

$$
2k_OC-\alpha\gamma
$$

接近零時，$D_yF$ 接近奇異，溫度或曝氣設定的微小變化可能造成較大的預測平衡變化。這是模型局部靈敏度警示，不是現場安全結論，也不能據此自動控制曝氣、投餌或加藥。感測校準不等於現場驗證；AI agent 在此只整理方程、單位、殘差與條件證據。

---

## 習題

### 習題一：手算

設

$$
F(x,y)=e^y+xy-2.
$$

已知 $F(0,\ln2)=0$。

1. 判斷隱函數定理是否適用。
2. 求通過該點之分支的 $g'(0)$。
3. 以一階近似估計 $x=0.01$ 時的 $y$。

### 習題二：程式

修改本章程式，對 $\varepsilon=10^{-2},10^{-4},10^{-6}$ 計算方向有限差分誤差

$$
\left\|
\frac{g(x+\varepsilon d)-g(x)}{\varepsilon}
-Dg(x)d
\right\|_2.
$$

說明何種結果支持實作正確，以及為何它仍不是定理證明。

### 習題三：反例

找出一個 $C^1$ 函數 $F:\mathbb R^2\to\mathbb R$，使 $F(0,0)=0$、$F_y(0,0)=0$，但零集合在原點附近仍唯一表示為可微函數 $y=g(x)$。說明此例推翻哪一個錯誤主張。

### 習題四：整合

令

$$
F(p,u,v)=
\begin{pmatrix}
u+v-p\\
u^2+v-2
\end{pmatrix}.
$$

在 $(p,u,v)=(2,1,1)$：

1. 驗證平衡條件。
2. 檢查 $D_{(u,v)}F$ 是否可逆。
3. 求 $du/dp$ 與 $dv/dp$。
4. 說明接近哪個 $u$ 值時靈敏度會放大。
5. 判斷上述結果是局部還是全域結論。

---

## 習題解答

### 解答一

有

$$
F_x(x,y)=y,\qquad F_y(x,y)=e^y+x.
$$

在 $(0,\ln2)$，

$$
F_y(0,\ln2)=2\ne0.
$$

因 $F$ 為 $C^\infty$，隱函數定理適用，存在唯一局部 $C^1$ 分支。其導數為

$$
g'(0)
=
-\frac{F_x(0,\ln2)}{F_y(0,\ln2)}
=
-\frac{\ln2}{2}.
$$

故當 $\Delta x=0.01$，

$$
g(0.01)
\approx
\ln2-\frac{\ln2}{2}(0.01)
=
0.995\ln2.
$$

數值約為 $0.68968$，僅是一階估計。

### 解答二

可加入：

```python
x = np.array([0.0, 0.0])
y = np.array([2.0, 1.0])
d = np.array([1.0, -2.0])
S = sensitivity(x, y)
target = S @ d

for eps in [1e-2, 1e-4, 1e-6]:
    yp, _, info = solve_equilibrium(x + eps*d, y, tol=1e-13)
    observed = (yp - y) / eps
    err = np.linalg.norm(observed - target, 2)
    print(eps, observed, err, info["status"])
```

在截斷誤差主導且求解器精度足夠的區間，預期 $\varepsilon$ 變小時誤差先下降。若 $\varepsilon$ 過小，浮點消去與求解誤差可能使下降停止甚至反彈。

這支持程式中的 Jacobian、符號與線性求解可能正確，但只檢查有限個步長與方向，不能證明所有鄰近點上的極限，也不能證明定理本身。

### 解答三

取

$$
F(x,y)=(y-x)^3.
$$

它是多項式，故為 $C^1$，且

$$
F(0,0)=0,\qquad F_y(x,y)=3(y-x)^2,
$$

所以 $F_y(0,0)=0$。然而

$$
F(x,y)=0
\iff
(y-x)^3=0
\iff
y=x.
$$

因此零集合全域上就是唯一可微函數 $g(x)=x$。此例推翻「若 $F_y=0$，則局部隱函數必不存在」的錯誤主張。它不推翻隱函數定理，因為定理只給充分條件。

### 解答四

先代入：

$$
F(2,1,1)
=
\begin{pmatrix}
1+1-2\\
1+1-2
\end{pmatrix}
=
\begin{pmatrix}
0\\0
\end{pmatrix}.
$$

令 $y=(u,v)^T$，則

$$
D_yF=
\begin{pmatrix}
1&1\\
2u&1
\end{pmatrix}.
$$

在 $u=1$，

$$
D_yF=
\begin{pmatrix}
1&1\\
2&1
\end{pmatrix},
\qquad
\det(D_yF)=-1\ne0.
$$

另外

$$
D_pF=
\begin{pmatrix}
-1\\0
\end{pmatrix}.
$$

令

$$
s=
\begin{pmatrix}
du/dp\\dv/dp
\end{pmatrix}.
$$

靈敏度方程為

$$
\begin{pmatrix}
1&1\\
2&1
\end{pmatrix}s
=
-\begin{pmatrix}
-1\\0
\end{pmatrix}
=
\begin{pmatrix}
1\\0
\end{pmatrix}.
$$

即

$$
s_1+s_2=1,\qquad 2s_1+s_2=0.
$$

相減得 $s_1=-1$，再得 $s_2=2$。所以

$$
\frac{du}{dp}=-1,\qquad
\frac{dv}{dp}=2.
$$

因

$$
\det(D_yF)=1-2u,
$$

故 $u\to1/2$ 時矩陣接近奇異，靈敏度可能放大。這些結論是通過指定平衡點之分支的局部結論，不能直接推出所有 $p$ 上解的全域唯一性。

---

## 本章小結

隱函數定理把隱式平衡方程轉化為局部函數分支。對

$$
F(x,y)=0,
$$

若 $F$ 在解附近為 $C^1$，且 $D_yF$ 在基準解可逆，則附近存在唯一的 $C^1$ 分支 $y=g(x)$。其靈敏度滿足

$$
D_yF\,Dg=-D_xF,
$$

亦即

$$
Dg=-(D_yF)^{-1}D_xF.
$$

數值上應解線性方程，而非顯式形成逆矩陣。$D_yF$ 接近奇異時，靈敏度可能放大，Newton 步驟也可能不可靠。可逆性是本定理的充分條件而非局部函數存在的必要條件；條件失效不能單憑邏輯判定分支不存在。

最重要的界線是：局部不等於全域，殘差小不等於條件良好，有限差分符合不等於完成證明，而模型靈敏度更不等於實際系統的安全操作結論。

---

## 參考來源

1. Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>
2. MIT OpenCourseWare，*18.100A Real Analysis*：<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>
4. JAX，*Autodiff Cookbook: JVP/VJP*：<https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html>

以上來源用作分析與多變量微積分的延伸閱讀。JAX 文件僅補充自動微分觀點；本章核心實作不依賴 JAX、GPU 或自動微分框架。線上 API 與教材版本可能更新，使用時應核對實際版本與定理假設。