# 第22章 線積分、勢函數與拓撲界線

## 學習目標與先備知識

本章研究向量場沿曲線的線積分，以及三個密切相關但不可任意混同的性質：

1. 向量場是否為某個標量函數的梯度；
2. 線積分是否只依賴端點；
3. 每一條閉曲線上的線積分是否皆為零。

完成本章後，讀者應能：

- 定義分段 $C^1$ 曲線與向量場線積分；
- 分辨標量弧長積分 $\int_\gamma f\,ds$ 與有向線積分 $\int_\gamma F\cdot dr$；
- 證明梯度場的線積分基本定理；
- 說明勢函數、路徑獨立及閉路零積分之間的等價關係；
- 區分「旋度零是必要條件」與「配合域條件後成為充分條件」；
- 以穿孔平面說明局部微分條件與全域拓撲之間的界線；
- 解釋單連通是常用充分條件，但不是某個特定向量場保守的必要條件；
- 使用 NumPy 在 CPU 上近似閉路積分，並測試取向、繞數與奇異點；
- 區分定理證明、解析計算、數值證據及有限採樣。

先備知識包括多變量微分、梯度、鏈式法則、單變量積分、曲線參數化，以及開集、連通、路徑連通與單連通的基本概念。本章主要在 $\mathbb R^2$ 與 $\mathbb R^3$ 中工作。

---

## 問題與直覺

設 $F$ 是平面上的力場，粒子沿曲線 $\gamma$ 從 $A$ 移到 $B$。力場所做的功為

$$
\int_\gamma F\cdot dr.
$$

一般而言，積分可能依賴實際路徑。例如旋轉型向量場可能記錄曲線繞某點幾圈。若存在標量函數 $\phi$ 使

$$
F=\nabla\phi,
$$

則由鏈式法則，

$$
F(\gamma(t))^T\gamma'(t)
=
\nabla\phi(\gamma(t))^T\gamma'(t)
=
\frac{d}{dt}\phi(\gamma(t)).
$$

因此積分只剩下端點的勢差：

$$
\int_\gamma F\cdot dr
=
\phi(B)-\phi(A).
$$

梯度場可直觀地想成一份全域高度函數的斜率資料。若沿閉路回到起點，高度淨變化必為零。

在二維中，若 $F=(P,Q)$，標量旋度為

$$
\operatorname{curl}F
=
\frac{\partial Q}{\partial x}
-
\frac{\partial P}{\partial y}.
$$

旋度零描述局部交叉偏導的相容性。但局部相容不必然產生全域勢函數：域中若有孔洞，某些閉路可能無法在域內縮成一點。穿孔平面上的經典渦旋場正是這種現象。

數值計算也有明確界線。找到一條可靠的非零閉路積分，可以推翻保守性；但檢查有限多條閉路皆接近零，不能證明所有閉路積分都等於零。無限命題需要定理或解析構造，不能由有限採樣取代。

---

## 定義、定理與推導

### 1. 分段光滑曲線與線積分

**定義 22.1（分段 $C^1$ 曲線）**  
映射 $\gamma:[a,b]\to\mathbb R^n$ 若連續，且存在有限分割

$$
a=t_0<t_1<\cdots<t_k=b
$$

使 $\gamma$ 在每個子區間上為 $C^1$，則稱 $\gamma$ 為分段 $C^1$ 曲線。

設 $U\subset\mathbb R^n$，$F:U\to\mathbb R^n$ 連續，且 $\gamma([a,b])\subset U$。向量場沿 $\gamma$ 的有向線積分定義為

$$
\int_\gamma F\cdot dr
=
\int_a^b F(\gamma(t))^T\gamma'(t)\,dt.
$$

在 Euclidean 內積下，$F(\gamma(t))$ 與 $\gamma'(t)$ 都視為 $n\times1$ 的 column vector，其內積是純量。

若反向參數化為

$$
\bar\gamma(t)=\gamma(a+b-t),
$$

則

$$
\bar\gamma'(t)=-\gamma'(a+b-t),
$$

因而

$$
\int_{\bar\gamma}F\cdot dr
=
-\int_\gamma F\cdot dr.
$$

相對地，標量弧長積分為

$$
\int_\gamma f\,ds
=
\int_a^b f(\gamma(t))\|\gamma'(t)\|_2\,dt.
$$

因速度使用範數，反向參數化不改變弧長積分的符號。兩類積分不可混用。

### 2. 勢函數與保守場

**定義 22.2（勢函數）**  
設 $U\subset\mathbb R^n$ 為開集。若存在 $\phi\in C^1(U)$ 使

$$
F=\nabla\phi,
$$

則稱 $\phi$ 為 $F$ 的勢函數，並稱 $F$ 為保守場或梯度場。

若 $U$ 連通，兩個勢函數 $\phi$ 與 $\psi$ 滿足

$$
\nabla(\phi-\psi)=0,
$$

所以 $\phi-\psi$ 為常數。若 $U$ 不連通，則每個連通分支上的常數可以不同。

### 3. 線積分基本定理

**定理 22.1（梯度場的線積分基本定理）**  
設 $U\subset\mathbb R^n$ 為開集，$\phi\in C^1(U)$，且 $F=\nabla\phi$。若 $\gamma:[a,b]\to U$ 是分段 $C^1$ 曲線，則

$$
\int_\gamma F\cdot dr
=
\phi(\gamma(b))-\phi(\gamma(a)).
$$

**證明。**  
先設 $\gamma$ 為 $C^1$。由鏈式法則，

$$
\frac{d}{dt}(\phi\circ\gamma)(t)
=
D\phi(\gamma(t))[\gamma'(t)]
=
\nabla\phi(\gamma(t))^T\gamma'(t).
$$

因此由單變量微積分基本定理，

$$
\begin{aligned}
\int_\gamma F\cdot dr
&=
\int_a^b\nabla\phi(\gamma(t))^T\gamma'(t)\,dt\\
&=
\int_a^b\frac{d}{dt}\phi(\gamma(t))\,dt\\
&=
\phi(\gamma(b))-\phi(\gamma(a)).
\end{aligned}
$$

若 $\gamma$ 只分段 $C^1$，便在每個子區間套用上述公式。各段相加時，中間端點的勢函數值兩兩消去，仍得到相同結論。證畢。

這是對所有符合條件曲線的證明，不是從有限條曲線的計算結果歸納而來。

### 4. 路徑獨立與閉路積分

**定義 22.3（路徑獨立）**  
設 $U$ 路徑連通。若對任意 $A,B\in U$，以及任意兩條由 $A$ 到 $B$ 的域內分段 $C^1$ 曲線 $\gamma_1,\gamma_2$，皆有

$$
\int_{\gamma_1}F\cdot dr
=
\int_{\gamma_2}F\cdot dr,
$$

則稱 $F$ 的線積分在 $U$ 上路徑獨立。

**命題 22.2（三種全域性質等價）**  
設 $U\subset\mathbb R^n$ 為開且路徑連通的集合，$F:U\to\mathbb R^n$ 連續。下列敘述等價：

1. 存在 $\phi\in C^1(U)$ 使 $F=\nabla\phi$；
2. $F$ 的線積分在 $U$ 上路徑獨立；
3. 對每條位於 $U$ 中的分段 $C^1$ 閉曲線 $\Gamma$，皆有
   $$
   \oint_\Gamma F\cdot dr=0.
   $$

**證明。**

由定理 22.1，若 $F=\nabla\phi$，則任一條由 $A$ 到 $B$ 的曲線積分都是 $\phi(B)-\phi(A)$，故 $(1)\Rightarrow(2)$。

若積分路徑獨立，一條閉曲線的起點與終點相同，其積分等於常值路徑的積分，所以為零。故 $(2)\Rightarrow(3)$。

假設所有閉路積分均為零。令 $\gamma_1,\gamma_2$ 都由 $A$ 走到 $B$。先沿 $\gamma_1$ 前進，再沿 $\gamma_2$ 反向返回，便形成閉曲線，因此

$$
0
=
\int_{\gamma_1}F\cdot dr
-
\int_{\gamma_2}F\cdot dr.
$$

所以 $(3)\Rightarrow(2)$。

最後，假設路徑獨立。固定 $x_0\in U$，定義

$$
\phi(x)
=
\int_{x_0}^{x}F\cdot dr.
$$

因路徑獨立，這個值不依賴所選路徑。由於 $U$ 開，對每個 $x\in U$ 及座標方向 $e_i$，足夠短的線段 $x+se_i$ 位於 $U$。因此

$$
\phi(x+he_i)-\phi(x)
=
\int_0^h F_i(x+se_i)\,ds.
$$

除以 $h$ 得

$$
\frac{\phi(x+he_i)-\phi(x)}{h}
=
\frac1h\int_0^hF_i(x+se_i)\,ds.
$$

由 $F_i$ 在 $x$ 的連續性，令 $h\to0$ 可得

$$
\frac{\partial\phi}{\partial x_i}(x)=F_i(x).
$$

所有分量 $F_i$ 都連續，故 $\phi\in C^1(U)$ 且 $\nabla\phi=F$。所以 $(2)\Rightarrow(1)$。證畢。

### 5. 旋度零的必要性

在三維中，若 $F=(P,Q,R)\in C^1(U;\mathbb R^3)$，定義

$$
\nabla\times F
=
\begin{pmatrix}
R_y-Q_z\\
P_z-R_x\\
Q_x-P_y
\end{pmatrix}.
$$

在二維中，對 $F=(P,Q)$ 定義標量旋度

$$
\operatorname{curl}F=Q_x-P_y.
$$

若 $F=\nabla\phi$ 且 $\phi\in C^2(U)$，則由連續混合偏導相等，

$$
\frac{\partial F_i}{\partial x_j}
=
\frac{\partial^2\phi}{\partial x_j\partial x_i}
=
\frac{\partial^2\phi}{\partial x_i\partial x_j}
=
\frac{\partial F_j}{\partial x_i}.
$$

因此在二維與三維中皆可說梯度場無旋。這給出：

> 在上述正則性下，旋度零是保守性的必要條件。

它尚未給出充分性，因為交叉偏導相等是局部條件。

### 6. 何時旋度零足以推出保守性

**定理 22.3（星形域上的充分條件）**  
設 $U\subset\mathbb R^n$ 為開星形集，$F\in C^1(U;\mathbb R^n)$，並且對所有 $i,j$ 都有

$$
\frac{\partial F_i}{\partial x_j}
=
\frac{\partial F_j}{\partial x_i}.
$$

則 $F$ 在 $U$ 上存在勢函數。

星形是指存在 $a\in U$，使每個 $x\in U$ 與 $a$ 之間的整條線段都包含於 $U$。可構造

$$
\phi(x)
=
\int_0^1F(a+t(x-a))^T(x-a)\,dt.
$$

再利用 $F\in C^1$、積分下微分與交叉偏導對稱，驗證 $\nabla\phi=F$。此處的正則性不是裝飾：它保證相關偏導連續，並允許對參數積分微分。

另一個常用版本是：

**定理 22.4（單連通域上的充分條件）**  
設 $U\subset\mathbb R^n$ 為開、連通且單連通的域，$F\in C^1(U;\mathbb R^n)$。若

$$
\frac{\partial F_i}{\partial x_j}
=
\frac{\partial F_j}{\partial x_i}
\quad\text{對所有 }i,j,
$$

則 $F$ 為保守場。

在二維，上述條件就是 $Q_x-P_y=0$；在三維，就是 $\nabla\times F=0$。此定理可由閉路同倫與局部勢函數論證，或由微分形式的 Poincaré 型結果證明。本章引用此全域結果，不在此重建完整同倫理論。

必須注意：

> 單連通是使每個適當正則無旋場都保守的一項充分域條件，但不是某個特定向量場保守的必要條件。

非單連通域上仍可能存在大量梯度場。

---

## 逐步手算例題

### 例 1：由勢函數計算端點差

令

$$
F(x,y)=
\begin{pmatrix}
2xy+3\\
x^2+2y
\end{pmatrix}.
$$

由 $\phi_x=2xy+3$ 對 $x$ 積分：

$$
\phi(x,y)=x^2y+3x+g(y).
$$

對 $y$ 微分：

$$
\phi_y=x^2+g'(y).
$$

與第二分量比較得 $g'(y)=2y$，故可取

$$
\phi(x,y)=x^2y+3x+y^2.
$$

由 $A=(0,0)$ 到 $B=(2,1)$ 的任意域內分段 $C^1$ 路徑皆有

$$
\int_\gamma F\cdot dr
=
\phi(2,1)-\phi(0,0)
=
4+6+1=11.
$$

再以折線核對。第一段 $\gamma_1(t)=(t,0)$，$0\le t\le2$：

$$
\int_{\gamma_1}F\cdot dr
=
\int_0^2 3\,dt=6.
$$

第二段 $\gamma_2(s)=(2,s)$，$0\le s\le1$：

$$
\int_{\gamma_2}F\cdot dr
=
\int_0^1(4+2s)\,ds=5.
$$

總和確為 $11$。這次直接積分是解析核對；路徑獨立本身來自勢函數定理，而不是只由這條折線證明。

### 例 2：穿孔平面的無旋非保守場

令

$$
U=\mathbb R^2\setminus\{(0,0)\},
\qquad
F(x,y)=
\begin{pmatrix}
-\dfrac{y}{x^2+y^2}\\[4pt]
\dfrac{x}{x^2+y^2}
\end{pmatrix}.
$$

寫成 $F=(P,Q)$。在 $U$ 上，

$$
Q_x
=
\frac{y^2-x^2}{(x^2+y^2)^2},
\qquad
P_y
=
\frac{y^2-x^2}{(x^2+y^2)^2}.
$$

所以

$$
Q_x-P_y=0.
$$

取逆時針單位圓

$$
\gamma(t)=(\cos t,\sin t),
\qquad 0\le t\le2\pi.
$$

其切向量為

$$
\gamma'(t)=(-\sin t,\cos t),
$$

而

$$
F(\gamma(t))=(-\sin t,\cos t).
$$

故 integrand 恰為

$$
F(\gamma(t))^T\gamma'(t)=1.
$$

所以

$$
\oint_\gamma F\cdot dr
=
\int_0^{2\pi}1\,dt
=
2\pi\ne0.
$$

由命題 22.2，$F$ 不可能在整個穿孔平面上具有單值勢函數。局部可將它視為角度函數的梯度，但角度繞原點一圈後增加 $2\pi$，無法成為整個 $U$ 上的單值連續勢函數。

### 例 3：單連通不是必要條件

仍取

$$
U=\mathbb R^2\setminus\{(0,0)\},
$$

但令

$$
G(x,y)=(2x,2y).
$$

雖然 $U$ 不是單連通，仍有全域勢函數

$$
\phi(x,y)=x^2+y^2.
$$

因此對每條位於 $U$ 中的閉曲線 $\Gamma$，

$$
\oint_\Gamma G\cdot dr=0.
$$

域中有孔只表示某些無旋場可能不保守，不能推出所有向量場都不保守。

---

## 實作與程式

下列自足 NumPy 程式使用複合梯形公式近似

$$
\int_a^bF(\gamma(t))^T\gamma'(t)\,dt.
$$

程式只使用 CPU。此處未執行程式；所有輸出均標示為預期結果。

```python
import numpy as np


def trapezoid(y, x):
    """避免依賴特定 NumPy 版本的 trapezoid/trapz 名稱。"""
    return np.sum(0.5 * (y[:-1] + y[1:]) * (x[1:] - x[:-1]))


def line_integral(field, curve, dcurve, a, b, n=4096):
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n 必須是至少 2 的整數")
    if not (np.isfinite(a) and np.isfinite(b)) or a == b:
        raise ValueError("積分端點必須有限且彼此不同")

    t = np.linspace(a, b, n + 1)
    points = np.asarray(curve(t), dtype=float)
    tangents = np.asarray(dcurve(t), dtype=float)
    expected = (n + 1, 2)

    if points.shape != expected:
        raise ValueError(f"curve 必須回傳 shape {expected}")
    if tangents.shape != expected:
        raise ValueError(f"dcurve 必須回傳 shape {expected}")
    if not np.all(np.isfinite(points)):
        raise FloatingPointError("曲線包含非有限座標")
    if not np.all(np.isfinite(tangents)):
        raise FloatingPointError("切向量包含非有限數值")

    values = np.asarray(field(points), dtype=float)
    if values.shape != expected:
        raise ValueError(f"field 必須回傳 shape {expected}")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("向量場產生非有限數值")

    integrand = np.sum(values * tangents, axis=1)
    return float(trapezoid(integrand, t))


def circle(radius=1.0, turns=1, clockwise=False):
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius 必須是有限正數")
    if not isinstance(turns, int) or isinstance(turns, bool) or turns == 0:
        raise ValueError("turns 必須是非零整數")

    sign = -1.0 if clockwise else 1.0

    def curve(t):
        u = sign * turns * t
        return np.column_stack((
            radius * np.cos(u),
            radius * np.sin(u)
        ))

    def dcurve(t):
        u = sign * turns * t
        factor = sign * turns * radius
        return np.column_stack((
            -factor * np.sin(u),
            factor * np.cos(u)
        ))

    return curve, dcurve


def potential_field(points):
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack((
        2.0 * x * y + 3.0,
        x * x + 2.0 * y
    ))


def vortex_field(points):
    x = points[:, 0]
    y = points[:, 1]
    r2 = x * x + y * y
    if np.any(r2 <= 1.0e-24):
        raise FloatingPointError("vortex_field 在原點未定義")
    return np.column_stack((-y / r2, x / r2))


def assert_close(actual, expected, tol, label):
    error = abs(actual - expected)
    if error > tol:
        raise AssertionError(
            f"{label}: actual={actual}, expected={expected}, error={error}"
        )


def assert_raises(exception_type, function, label):
    try:
        function()
    except exception_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"{label}: 得到錯誤型別 {type(exc).__name__}"
        ) from exc
    raise AssertionError(f"{label}: 未丟出預期例外")


def run_tests():
    # 正常測試：保守場沿閉路積分為零。
    c1, dc1 = circle(radius=1.0)
    value = line_integral(
        potential_field, c1, dc1, 0.0, 2.0 * np.pi
    )
    assert_close(value, 0.0, 1.0e-10, "conservative closed loop")

    # 正常測試：逆時針繞原點一次。
    value = line_integral(
        vortex_field, c1, dc1, 0.0, 2.0 * np.pi
    )
    assert_close(value, 2.0 * np.pi, 1.0e-10, "vortex ccw")

    # 邊界測試：改變半徑不改變繞數。
    csmall, dcsmall = circle(radius=1.0e-4)
    value = line_integral(
        vortex_field, csmall, dcsmall, 0.0, 2.0 * np.pi
    )
    assert_close(value, 2.0 * np.pi, 1.0e-9, "small circle")

    # 邊界測試：繞兩圈，積分加倍。
    c2, dc2 = circle(radius=2.5, turns=2)
    value = line_integral(
        vortex_field, c2, dc2, 0.0, 2.0 * np.pi
    )
    assert_close(value, 4.0 * np.pi, 1.0e-10, "two turns")

    # 取向測試：順時針變號。
    cw, dcw = circle(radius=1.0, clockwise=True)
    value = line_integral(
        vortex_field, cw, dcw, 0.0, 2.0 * np.pi
    )
    assert_close(value, -2.0 * np.pi, 1.0e-10, "clockwise")

    # 故障測試：非法半徑與節點數。
    assert_raises(
        ValueError,
        lambda: circle(radius=0.0),
        "zero radius"
    )
    assert_raises(
        ValueError,
        lambda: line_integral(
            vortex_field, c1, dc1, 0.0, 2.0 * np.pi, n=1
        ),
        "too few panels"
    )

    # 故障測試：此曲線在 t=pi 通過原點；n=1000 會取到該點。
    def through_origin(t):
        return np.column_stack((1.0 + np.cos(t), np.sin(t)))

    def dthrough_origin(t):
        return np.column_stack((-np.sin(t), np.cos(t)))

    assert_raises(
        FloatingPointError,
        lambda: line_integral(
            vortex_field,
            through_origin,
            dthrough_origin,
            0.0,
            2.0 * np.pi,
            n=1000
        ),
        "curve through singularity"
    )

    print("所有測試通過")


if __name__ == "__main__":
    run_tests()
```

程式用 shape $(N,2)$ 儲存 $N$ 個平面點，每個矩陣橫列是一個點。這是批次資料的儲存方式；公式中的單一向量仍是 $2\times1$ 的 column vector。NumPy 一維陣列的 `.T` 不會改變 shape，因此程式不以一維 `.T` 冒充列向量與橫向量的轉換。

---

## 測試與預期結果

上述測試程式尚未執行，以下均是解析推導所得的預期。

### 正常測試

保守場 `potential_field` 的勢函數是

$$
\phi(x,y)=x^2y+3x+y^2.
$$

沿任何閉路的精確積分皆為零。因此第一個 assertion 預期通過，數值只可能留下浮點捨入誤差。

渦旋場沿逆時針圓的 integrand 恰為常數 $1$，所以預期值為

$$
2\pi\approx6.283185307179586.
$$

順時針圓的取向相反，預期值為 $-2\pi$。

### 邊界測試

半徑改成 $10^{-4}$ 時，解析積分仍為 $2\pi$。不過場值大小約為 $10^4$，切向量大小約為 $10^{-4}$；乘積雖仍為正常尺度，中間量卻較大。半徑繼續縮小可能增加浮點風險，因此程式設置奇異閾值不能被解讀為數學定義的一部分。

若曲線繞原點兩圈，預期積分為

$$
4\pi.
$$

更一般地，對不通過原點的分段 $C^1$ 閉曲線 $\Gamma$，渦旋場積分滿足

$$
\oint_\Gamma F\cdot dr
=
2\pi\,\operatorname{wind}(\Gamma,0),
$$

其中 $\operatorname{wind}(\Gamma,0)$ 是曲線對原點的繞數。因此正確敘述是：

> 若閉曲線不經過原點且對原點的繞數為零，則積分為零。

不能只用含糊的「不包圍原點」處理自交、多圈或方向相反的曲線。

### 故障測試

- `radius=0` 應丟出 `ValueError`；
- `n=1` 應丟出 `ValueError`；
- 明確通過原點的曲線應丟出 `FloatingPointError`；
- 曲線、切向量或向量場回傳錯誤 shape 時應丟出 `ValueError`；
- 非有限座標或場值應被拒絕。

離散節點未碰到奇異點，不表示連續曲線沒有碰到奇異點。本程式的故障案例特別選擇偶數分割，使 $t=\pi$ 成為節點。對一般輸入，還需要解析幾何檢查、區間界或更可靠的碰撞檢測；有限節點本身不能證明曲線完全位於定義域。

---

## 反例與常見陷阱

### 1. 把旋度零當作無條件充分條件

穿孔平面的渦旋場在每個定義點都滿足旋度零，卻有非零閉路積分。旋度零是局部微分條件；要推出全域勢函數，還需要星形、單連通或其他足以排除拓撲障礙的條件。

### 2. 把單連通當作必要條件

非單連通域上也能定義 $\phi(x,y)=x^2+y^2$，其梯度當然是保守場。單連通保證一整類無旋場都保守，不是某個特定場保守所必須具備的域性質。

### 3. 忽略向量場的定義域

渦旋場在原點沒有定義。不能在原點計算旋度，也不能把通過原點的曲線積分當成普通線積分。形式微分前必須先寫明域。

### 4. 遺漏依賴其他變數的積分常數

由 $\phi_x=P$ 積分時，應寫成

$$
\phi(x,y)=\int P(x,y)\,dx+g(y),
$$

而不是只加普通常數。接著必須用 $\phi_y=Q$ 決定 $g$。若所得 $g'(y)$ 仍依賴 $x$，便代表候選場可能不相容。

### 5. 混淆必要與充分

在 $C^2$ 勢函數條件下，保守推出旋度零；這是必要性。旋度零推出保守則需要 $F\in C^1$ 與適當域條件；這是有條件的充分性。兩個方向的假設並不相同。

### 6. 把有限數值採樣當成證明

有限條閉路的積分接近零，不能證明所有閉路積分都為零。反過來，若能嚴格證明某一條合法閉路的積分非零，便足以否定保守性。數值非零值仍需誤差界，不能只憑單次浮點結果下定論。

### 7. 忽略取向與繞數

反向曲線使有向線積分變號；重複繞行使積分累加。對自交曲線，「內部」未必只有一個，因此應使用繞數，而不是只說曲線是否包圍某點。

---

## AI、幾何與養殖案例

### 1. 學習模型中的梯度一致性

許多最佳化方法從損失函數 $L(\theta)$ 取得更新方向 $\nabla L(\theta)$。若資料驅動模型聲稱輸出的是某個全域損失的精確梯度，則理論上

$$
\oint_\Gamma\nabla L(\theta)\cdot d\theta=0
$$

對每條合法閉路都成立。

可在參數空間選取小矩形或圓形閉路作數值診斷。若解析度加密後仍得到明顯非零的環流，可能原因包括：

- 所學向量場並非梯度場；
- 不同路段使用了不同正規化；
- 隨機梯度噪聲未被控制；
- 路徑穿越模型不可微或未定義區域；
- 參數域刪除了奇異集合；
- 程式中的切向量、取向或 broadcasting 有誤。

但閉路測試接近零仍只是數值證據。若要證明存在全域損失函數，需要明確勢函數，或驗證正則性、無旋性及域的拓撲條件。

### 2. 幾何觀點

梯度 $\nabla\phi$ 在 Euclidean 內積下代表微分 $D\phi$。對切向量 $v$，

$$
D\phi(x)[v]
=
\nabla\phi(x)^Tv.
$$

線積分就是沿曲線逐點配對微分與切向量。若改用非 Euclidean 度量，將微分轉成梯度向量的方式會改變；不能無條件把協向量與向量視為相同物件。本章的 $F\cdot dr$ 使用標準 Euclidean 內積。

### 3. 合成養殖感測校準

考慮完全合成的溫度與鹽度校準座標

$$
u=\frac{T-T_0}{s_T},
\qquad
v=\frac{S-S_0}{s_S},
$$

其中 $T$ 及 $s_T$ 的單位為攝氏度，$S$ 及 $s_S$ 的單位為 PSU；$u,v$ 無因次。定義合成代價

$$
\Phi(u,v)=\frac12u^2+uv+2v^2.
$$

則

$$
\nabla_{(u,v)}\Phi
=
\begin{pmatrix}
u+v\\
u+4v
\end{pmatrix}.
$$

由鏈式法則返回物理變數：

$$
\frac{\partial\Phi}{\partial T}
=
\frac{u+v}{s_T},
\qquad
\frac{\partial\Phi}{\partial S}
=
\frac{u+4v}{s_S}.
$$

兩個分量的單位分別是代價每攝氏度與代價每 PSU，不能把未加權的物理分量直接當成同單位 Euclidean 向量。無因次化後可以使用明確尺度，而返回物理量時必須乘回相應換算。

若校準程序沿不同參數路徑從同一初值到同一終值，精確梯度積分應給出相同的 $\Phi$ 變化。閉路殘差可作資料管線稽核訊號，但不能單獨證明感測器故障，更不能直接觸發投餌、加藥或設備控制。此案例完全合成，agent 只整理證據；模型一致性不等於現場安全驗證。

---

## 習題

### 習題 1：手算

令

$$
F(x,y)=(e^x\cos y,-e^x\sin y).
$$

1. 求一個勢函數。
2. 計算由 $(0,0)$ 到 $(1,\pi)$ 的任意分段 $C^1$ 路徑積分。
3. 說明答案為何與路徑無關。

### 習題 2：程式

使用本章程式計算渦旋場沿橢圓

$$
\gamma(t)=(2\cos t,\sin t),
\qquad 0\le t\le2\pi
$$

的積分。再定義完整的反向曲線及其導數，比較兩個結果。列出預期值及一個故障測試。

### 習題 3：反例

判斷下列敘述真假，並證明或給出反例：

> 若開集 $U\subset\mathbb R^2$ 不是單連通，則 $U$ 上不存在保守向量場。

### 習題 4：整合

令

$$
U=\{(x,y):x>0\},
\qquad
F(x,y)=
\left(
\frac{x}{x^2+y^2},
\frac{y}{x^2+y^2}
\right).
$$

1. 驗證 $F$ 在 $U$ 上旋度為零。
2. 求勢函數。
3. 計算由 $(1,0)$ 到 $(1,1)$ 的線積分。
4. 說明域條件在此扮演的角色。

### 習題 5：數值證據判讀

某系統在三條閉路上得到積分近似值 $10^{-8}$、$-3\times10^{-7}$ 與 $0.42$。有人宣稱前兩條已證明向量場保守，而第三條只是雜訊。分析還需要哪些資訊，並說明目前能得到什麼結論。

### 習題 6：概念辨析

逐一判斷下列敘述，並說明必要條件、充分條件及域條件：

1. 每個 $C^2$ 勢函數的梯度場都無旋。
2. 每個無旋的 $C^1$ 向量場都保守。
3. 在開、連通且單連通域上，每個 $C^1$ 無旋場都保守。
4. 若向量場保守，則其定義域必須單連通。
5. 若一條閉路的數值積分接近零，則向量場保守。

---

## 習題解答

### 解答 1

由第一分量對 $x$ 積分：

$$
\phi(x,y)=e^x\cos y+g(y).
$$

因此

$$
\phi_y=-e^x\sin y+g'(y).
$$

與第二分量比較得 $g'(y)=0$，可取

$$
\phi(x,y)=e^x\cos y.
$$

所以

$$
\int_\gamma F\cdot dr
=
\phi(1,\pi)-\phi(0,0)
=
-e-1.
$$

因 $F=\nabla\phi$ 且 $\phi\in C^\infty(\mathbb R^2)$，線積分基本定理保證答案與路徑無關。

### 解答 2

可增補下列程式：

```python
def ellipse(t):
    return np.column_stack((
        2.0 * np.cos(t),
        np.sin(t)
    ))


def dellipse(t):
    return np.column_stack((
        -2.0 * np.sin(t),
        np.cos(t)
    ))


def reverse_ellipse(t):
    return ellipse(2.0 * np.pi - t)


def dreverse_ellipse(t):
    return -dellipse(2.0 * np.pi - t)


forward = line_integral(
    vortex_field,
    ellipse,
    dellipse,
    0.0,
    2.0 * np.pi,
    n=4096
)

backward = line_integral(
    vortex_field,
    reverse_ellipse,
    dreverse_ellipse,
    0.0,
    2.0 * np.pi,
    n=4096
)

assert_close(forward, 2.0 * np.pi, 1.0e-7, "ellipse forward")
assert_close(backward, -2.0 * np.pi, 1.0e-7, "ellipse backward")
```

橢圓逆時針繞原點一次，故正向預期為 $2\pi$；反向導數必須同時加入負號，所以反向預期為 $-2\pi$。

故障測試可用

$$
\eta(t)=(1+\cos t,\sin t),
$$

它在 $t=\pi$ 通過原點。若離散節點包含 $t=\pi$，程式應丟出 `FloatingPointError`。若節點沒有剛好命中奇異點，有限取樣可能漏掉問題，因此正式使用時仍需額外的連續曲線域檢查。

### 解答 3

敘述為假。取

$$
U=\mathbb R^2\setminus\{(0,0)\},
\qquad
\phi(x,y)=x^2+y^2.
$$

雖然 $U$ 不是單連通，但

$$
F=\nabla\phi=(2x,2y)
$$

在 $U$ 上是保守場。因此單連通不是特定場保守的必要條件。

### 解答 4

令

$$
P=\frac{x}{x^2+y^2},
\qquad
Q=\frac{y}{x^2+y^2}.
$$

計算得

$$
Q_x=-\frac{2xy}{(x^2+y^2)^2},
\qquad
P_y=-\frac{2xy}{(x^2+y^2)^2}.
$$

所以 $Q_x-P_y=0$。可取勢函數

$$
\phi(x,y)=\frac12\log(x^2+y^2),
$$

因為

$$
\nabla\phi
=
\left(
\frac{x}{x^2+y^2},
\frac{y}{x^2+y^2}
\right).
$$

因此

$$
\int_\gamma F\cdot dr
=
\phi(1,1)-\phi(1,0)
=
\frac12\log2.
$$

右半平面是凸集，因而也是星形及單連通的開連通域，所以無旋充分條件可以套用。不過本題已直接構造勢函數，不必把單連通誤當成必要條件。事實上，同一勢函數也定義在整個穿孔平面上。

### 解答 5

前兩個小值不能證明保守性，因為只測試有限條閉路。判讀第三個值至少需要：

- 求積方法與離散解析度；
- 解析度加密後的變化；
- 場值誤差與隨機噪聲尺度；
- 曲線是否完整位於定義域；
- 曲線的取向與繞數；
- 是否接近奇異點；
- 是否存在獨立可驗證的勢函數；
- 是否有嚴格誤差上界。

若能證明第三條合法閉路的真實積分非零，就可推出場不保守。僅憑三個近似值，尚不能嚴格證明保守或非保守；但 $0.42$ 是重要的反證候選，不能在沒有誤差模型時任意歸為雜訊。

### 解答 6

1. **真。** 若 $\phi\in C^2$，連續混合偏導相等，因此 $\nabla\phi$ 無旋。這是保守推出無旋的必要性方向。
2. **假。** 穿孔平面渦旋場是 $C^1$ 無旋場，卻有閉路積分 $2\pi$。
3. **真。** 這是定理 22.4；需要開、連通、單連通與 $C^1$ 正則性。
4. **假。** $\nabla(x^2+y^2)$ 在穿孔平面上仍是保守場。
5. **假。** 單一路徑的近零數值只能提供有限證據，既有離散誤差，也沒有涵蓋所有閉路。

---

## 本章小結

向量場線積分把場值與曲線切向量配對，因此依賴向量場、曲線及取向。若 $F=\nabla\phi$，則

$$
\int_\gamma F\cdot dr
=
\phi(\gamma(b))-\phi(\gamma(a)).
$$

在開且路徑連通的域上，對連續向量場而言，存在勢函數、路徑獨立與所有閉路積分為零彼此等價。

若勢函數具有足夠正則性，保守場必定無旋。反方向不是無條件成立：在開星形域，或開、連通且單連通的域上，$C^1$ 無旋場才可由相應定理推出保守性。穿孔平面渦旋場顯示，局部旋度零可能與全域非零環流並存。

單連通是常用充分條件，不是特定向量場保守的必要條件。判斷時必須分清：

- 場的局部微分條件；
- 定義域的全域拓撲；
- 曲線的取向與繞數；
- 解析證明與有限數值證據。

數值閉路積分適合尋找反例、核對取向及診斷程式，但不能用有限採樣證明無限多條閉路的命題。

---

## 參考來源

1. Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>
2. MIT OpenCourseWare，*18.100A Real Analysis*：<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>

以上來源用於一般實分析、多變量微積分與向量場背景。線上內容及版本可能更新；本章的定理應以本文明列的域、正則性與曲線條件為準。