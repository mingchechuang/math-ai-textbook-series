# 第17章 等式約束、切空間與Lagrange乘數

## 學習目標與先備知識

本章研究有限維實數空間中的等式約束最佳化：

$$
\min_{x\in\mathbb R^n} f(x)
\quad\text{使得}\quad
c(x)=0,
$$

其中 $U\subset\mathbb R^n$ 為開集，$f:U\to\mathbb R$，$c:U\to\mathbb R^m$。完成本章後，讀者應能：

1. 正確寫出 $J_c(x)\in\mathbb R^{m\times n}$，並辨認每條約束對應一個方程橫列。
2. 說明 LICQ 與正則等位集合的關係。
3. 在正則條件下求切空間
   $$
   T_xM=\ker J_c(x).
   $$
4. 推導 Lagrange 一階必要條件
   $$
   \nabla f(x)+J_c(x)^T\lambda=0.
   $$
5. 區分一階必要、二階必要、二階充分與全域最優條件。
6. 使用零空間基底 $Z$ 建立受限 Hessian
   $$
   Z^T\nabla_{xx}^2L(x,\lambda)Z.
   $$
7. 正確解讀正定、負定、不定、半定及零維切空間。
8. 用 NumPy 在 CPU 上解線性等式約束二次問題，並檢查秩、殘差與二階曲率。

先備知識包括 Jacobian、梯度、Hessian、隱函數定理、零空間、正交補與二次型。全文採 Euclidean 內積；梯度是 $n\times1$ 列向量，而標量函數的導數是 $1\times n$ 線性泛函。

---

## 問題與直覺

無約束最佳化允許從內點朝任意小方向移動，因此可微函數在局部極值點必須滿足 $\nabla f(x)=0$。有等式約束後，多數方向不再可行。例如在單位圓

$$
x_1^2+x_2^2=1
$$

上，瞬時移動必須沿圓周切線，不能任意朝徑向移動。

設可行集合為

$$
M=\{x\in U:c(x)=0\}.
$$

若 $\gamma:(-\epsilon,\epsilon)\to M$ 是 $C^1$ 可行曲線，且 $\gamma(0)=x$，則

$$
c(\gamma(t))=0.
$$

由鏈式法則，

$$
J_c(x)\gamma'(0)=0.
$$

因此可行曲線的速度必在 $\ker J_c(x)$ 中。在適當的滿秩條件下，每個零空間方向都可由某條可行曲線產生，因而切空間就是 Jacobian 的零空間。

若 $x^\star$ 是受限局部極值，目標函數沿每個切向 $d$ 的一階變化應為零：

$$
\nabla f(x^\star)^Td=0.
$$

這表示 $\nabla f(x^\star)$ 垂直於切空間。有限維線性代數給出

$$
(\ker J_c(x^\star))^\perp
=\operatorname{range}(J_c(x^\star)^T).
$$

所以存在 $\lambda\in\mathbb R^m$ 使

$$
\nabla f(x^\star)+J_c(x^\star)^T\lambda=0.
$$

Lagrange 乘數的幾何意義，就是用約束梯度的線性組合平衡目標梯度。

---

## 定義、定理與推導

### 1. 約束 Jacobian 與 LICQ

將約束寫成列向量：

$$
c(x)=
\begin{bmatrix}
c_1(x)\\
\vdots\\
c_m(x)
\end{bmatrix}.
$$

其 Jacobian 為

$$
J_c(x)=
\begin{bmatrix}
(\nabla c_1(x))^T\\
\vdots\\
(\nabla c_m(x))^T
\end{bmatrix}
\in\mathbb R^{m\times n}.
$$

每個約束對應一個方程橫列。$J_c(x)$ 的列空間（row space）是 $\mathbb R^n$ 的子空間，其維度為 $\operatorname{rank}J_c(x)$；$J_c(x)$ 的行空間（column space）是 $\mathbb R^m$ 的子空間。Lagrange 條件使用的是 $J_c(x)^T$ 的行空間，亦即約束梯度所張成的法空間。

**定義（LICQ）。** 在可行點 $x$，若

$$
\nabla c_1(x),\ldots,\nabla c_m(x)
$$

線性獨立，等價地，

$$
\operatorname{rank}J_c(x)=m,
$$

則稱線性獨立約束資格 LICQ 在 $x$ 成立。LICQ 必然要求 $m\le n$。

LICQ 是保證標準 Lagrange 必要條件的充分正則性條件，但不是乘數存在的必要條件。秩退化時，乘數可能不存在，也可能存在但不唯一。

### 2. 曲線切空間與線性化切空間

**定義（曲線切空間）。** 對 $x\in M$，定義

$$
T_xM=
\{\gamma'(0):
\gamma:(-\epsilon,\epsilon)\to M
\text{ 為 }C^1,\ \gamma(0)=x\}.
$$

對任一可行曲線使用鏈式法則，可得

$$
T_xM\subseteq\ker J_c(x).
$$

這個包含關係不需要 LICQ。反方向則需要正則性。

**定理（正則等位集合的切空間）。**  
設 $U\subset\mathbb R^n$ 為開集，$c:U\to\mathbb R^m$ 為 $C^1$，$x\in U$、$c(x)=0$，且

$$
\operatorname{rank}J_c(x)=m.
$$

則 $M=c^{-1}(0)$ 在 $x$ 附近是維度 $n-m$ 的 $C^1$ 子流形，而且

$$
T_xM=\ker J_c(x).
$$

這是隱函數定理的局部推論。若 LICQ 失效，$\ker J_c(x)$ 只是線性化可行方向集合，可能比真實曲線切空間大很多。

### 3. Lagrangian 與一階必要條件

定義 Lagrangian：

$$
L(x,\lambda)=f(x)+\lambda^Tc(x).
$$

其中 $\lambda\in\mathbb R^m$ 是列向量。其對 $x$ 的梯度是

$$
\nabla_xL(x,\lambda)
=\nabla f(x)+J_c(x)^T\lambda.
$$

因此站立條件為

$$
\nabla f(x)+J_c(x)^T\lambda=0.
$$

若改採 $L=f-\lambda^Tc$，乘數符號會相反；兩種符號約定不能在同一推導中混用。

**命題（一階必要條件與乘數唯一性）。**  
設 $f:U\to\mathbb R$ 與 $c:U\to\mathbb R^m$ 均為 $C^1$。若 $x^\star$ 是 $f$ 在 $M=\{x:c(x)=0\}$ 上的局部極小點或局部極大點，且 LICQ 在 $x^\star$ 成立，則存在唯一的 $\lambda^\star\in\mathbb R^m$ 使

$$
c(x^\star)=0,
$$

$$
\nabla f(x^\star)+J_c(x^\star)^T\lambda^\star=0.
$$

**證明。**

由正則等位集合定理，

$$
T_{x^\star}M=\ker J_c(x^\star).
$$

任取 $d\in T_{x^\star}M$。存在局部 $C^1$ 可行曲線 $\gamma$，使

$$
\gamma(0)=x^\star,\qquad \gamma'(0)=d.
$$

因 $x^\star$ 是受限局部極值，單變量函數 $f(\gamma(t))$ 在 $t=0$ 有局部極值，所以

$$
0=\frac{d}{dt}f(\gamma(t))\bigg|_{t=0}
=\nabla f(x^\star)^Td.
$$

此式對所有 $d\in\ker J_c(x^\star)$ 成立，因此

$$
\nabla f(x^\star)\in
(\ker J_c(x^\star))^\perp.
$$

有限維基本子空間關係給出

$$
(\ker J_c(x^\star))^\perp
=\operatorname{range}(J_c(x^\star)^T).
$$

故存在 $\mu\in\mathbb R^m$ 使

$$
\nabla f(x^\star)=J_c(x^\star)^T\mu.
$$

令 $\lambda^\star=-\mu$，即得站立條件。

再證唯一性。若 $\lambda_1,\lambda_2$ 都滿足站立條件，則

$$
J_c(x^\star)^T(\lambda_1-\lambda_2)=0.
$$

LICQ 表示 $J_c(x^\star)$ 滿橫列秩，因此 $J_c(x^\star)^T$ 的零空間只有零向量，故 $\lambda_1=\lambda_2$。證畢。

這是必要條件，不是充分條件。Lagrange 駐點可能是最小、最大或鞍點。

### 4. 二階條件與受限 Hessian

假設 $f,c_1,\ldots,c_m$ 在 $x^\star$ 的鄰域為 $C^2$，且 $(x^\star,\lambda^\star)$ 滿足一階條件。定義 Lagrangian 對 $x$ 的 Hessian：

$$
H_L=
\nabla_{xx}^2L(x^\star,\lambda^\star)
=
\nabla^2f(x^\star)
+\sum_{i=1}^m
\lambda_i^\star\nabla^2c_i(x^\star).
$$

非線性約束的 Hessian 項不可省略。只檢查 $\nabla^2f$ 通常會漏掉可行曲面的曲率。

在 LICQ 下，選取

$$
Z\in\mathbb R^{n\times(n-m)}
$$

使其縱行構成 $\ker J_c(x^\star)$ 的基底：

$$
J_c(x^\star)Z=0.
$$

任一切向可寫為 $d=Zp$，所以

$$
d^TH_Ld=p^T(Z^TH_LZ)p.
$$

矩陣

$$
H_R=Z^TH_LZ
$$

稱為受限 Hessian 或約化 Hessian。若 $Z^TZ=I$，則 $H_R$ 的特徵值是在正交切向座標中的二階曲率。換用另一組基底會改變矩陣元素，但不會改變二次型的正定、負定或不定性。

**二階必要條件。**  
若 $x^\star$ 是受限局部極小點、LICQ 成立且資料為 $C^2$，則

$$
d^TH_Ld\ge 0
\quad\text{對所有 }d\in\ker J_c(x^\star).
$$

若 $x^\star$ 是受限局部極大點，則必有

$$
d^TH_Ld\le 0
\quad\text{對所有 }d\in\ker J_c(x^\star).
$$

**二階充分條件。**  
若一階條件與 LICQ 成立，且

$$
d^TH_Ld>0
\quad\text{對所有非零 }d\in\ker J_c(x^\star),
$$

則 $x^\star$ 是嚴格受限局部極小點。若對所有非零切向皆小於零，則是嚴格受限局部極大點。

二階分類如下：

- $H_R$ 正定：嚴格受限局部最小的充分條件。
- $H_R$ 負定：嚴格受限局部最大的充分條件。
- $H_R$ 不定：同時違反局部最小的半正定必要條件與局部最大的半負定必要條件，所以在上述 $C^2$、LICQ 與站立條件下，不可能是局部極值。
- $H_R$ 正半定但不正定：與局部最小的必要條件相容，且若含正特徵值則排除局部最大；但不能保證嚴格局部最小。
- $H_R$ 負半定但不負定：與局部最大的必要條件相容，且若含負特徵值則排除局部最小；但不能保證嚴格局部最大。
- $H_R=0$：二階測試無法分類，必須研究高階項或直接分析可行曲線。
- 切空間為零維：在 LICQ 下，可行點局部孤立；不應把空矩陣解讀為觀察到正曲率。

值得特別核對：不定二次型的正、負二階項是二次階主導量，高階項不能把兩種符號都消除。因此「不定排除局部極值」是二階必要條件的直接逆否推論；它不是只有線性約束二次問題才成立。

---

## 逐步手算例題

### 例一：圓上的線性目標

求

$$
\min f(x,y)=x+2y
\quad\text{使得}\quad
c(x,y)=x^2+y^2-1=0.
$$

約束 Jacobian 為

$$
J_c(x,y)=
\begin{bmatrix}
2x&2y
\end{bmatrix}.
$$

圓上 $(x,y)\ne(0,0)$，所以 Jacobian 秩為 $1$，LICQ 成立。Lagrangian 為

$$
L=x+2y+\lambda(x^2+y^2-1).
$$

一階條件是

$$
1+2\lambda x=0,
\qquad
2+2\lambda y=0,
\qquad
x^2+y^2=1.
$$

由前兩式，

$$
x=-\frac1{2\lambda},
\qquad
y=-\frac1{\lambda}=2x.
$$

代入約束：

$$
5x^2=1.
$$

候選點為

$$
\left(\frac1{\sqrt5},\frac2{\sqrt5}\right),
\qquad
\left(-\frac1{\sqrt5},-\frac2{\sqrt5}\right).
$$

其目標值分別為 $\sqrt5$ 與 $-\sqrt5$，所以後者是全域最小，前者是全域最大。全域結論可由直接比較或 Cauchy–Schwarz 不等式得到，不能只由 Lagrange 方程推出。

在最小點，

$$
\lambda^\star=\frac{\sqrt5}{2},
\qquad
H_L=2\lambda^\star I=\sqrt5I.
$$

可取單位切向基底

$$
Z=\frac1{\sqrt5}
\begin{bmatrix}
-2\\1
\end{bmatrix}.
$$

故

$$
Z^TH_LZ=\sqrt5>0.
$$

二階充分條件確認它是嚴格受限局部最小。

在最大點，

$$
\lambda^\star=-\frac{\sqrt5}{2},
\qquad
Z^TH_LZ=-\sqrt5<0,
$$

因此是嚴格受限局部最大。

### 例二：線性約束二次問題

求

$$
\min_{x,y}
\frac12(x^2+2y^2)-2x-6y
\quad\text{使得}\quad x+y=2.
$$

令

$$
Q=
\begin{bmatrix}
1&0\\
0&2
\end{bmatrix},
\quad
q=
\begin{bmatrix}
-2\\-6
\end{bmatrix},
\quad
A=
\begin{bmatrix}
1&1
\end{bmatrix},
\quad b=2.
$$

問題為

$$
\min_z\frac12z^TQz+q^Tz
\quad\text{使得}\quad Az=b.
$$

一階條件為

$$
Qz+q+A^T\lambda=0,
\qquad
Az=b.
$$

展開：

$$
x-2+\lambda=0,
$$

$$
2y-6+\lambda=0,
$$

$$
x+y=2.
$$

由前兩式，

$$
x=2-\lambda,
\qquad
y=3-\frac{\lambda}{2}.
$$

代入約束得 $\lambda=2$，因此

$$
(x^\star,y^\star)=(0,2).
$$

切空間為 $\ker A$，可取

$$
Z=\frac1{\sqrt2}
\begin{bmatrix}
1\\-1
\end{bmatrix}.
$$

約束是線性的，所以 $H_L=Q$。受限 Hessian 為

$$
Z^TQZ=\frac32>0.
$$

因此該點是嚴格受限局部最小。又因 $Q$ 正定、目標嚴格凸且可行集合是仿射集合，所以它也是唯一全域最小。這個全域結論額外使用了凸性。

---

## 實作與程式

以下程式只依賴 NumPy，在 CPU 上解線性等式約束二次問題：

$$
\min_x\frac12x^TQx+q^Tx
\quad\text{使得}\quad Ax=b.
$$

KKT 系統是

$$
\begin{bmatrix}
Q&A^T\\
A&0
\end{bmatrix}
\begin{bmatrix}
x\\\lambda
\end{bmatrix}
=
\begin{bmatrix}
-q\\b
\end{bmatrix}.
$$

程式明確區分正定、負定、不定、非零正半定、非零負半定、數值零及零維切空間。半定分類同時說明「與哪一類必要條件相容」及「不能推出哪一類充分結論」。

```python
import numpy as np

def nullspace_basis(A, rtol=1e-12):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")

    _, s, vt = np.linalg.svd(A, full_matrices=True)
    scale = s[0] if s.size else 1.0
    threshold = rtol * max(scale, 1.0)
    rank = int(np.sum(s > threshold))
    Z = vt[rank:, :].T
    return Z, rank

def classify_reduced_hessian(eigvals, tol=1e-10):
    eigvals = np.asarray(eigvals, dtype=float)

    if eigvals.size == 0:
        return {
            "curvature_class": "zero-dimensional tangent space",
            "second_order_conclusion":
                "locally isolated feasible point under LICQ"
        }

    all_pos = bool(np.all(eigvals > tol))
    all_neg = bool(np.all(eigvals < -tol))
    has_pos = bool(np.any(eigvals > tol))
    has_neg = bool(np.any(eigvals < -tol))
    all_near_zero = bool(np.all(np.abs(eigvals) <= tol))

    if all_pos:
        return {
            "curvature_class": "positive definite",
            "second_order_conclusion":
                "sufficient for a strict local minimum"
        }

    if all_neg:
        return {
            "curvature_class": "negative definite",
            "second_order_conclusion":
                "sufficient for a strict local maximum"
        }

    if has_pos and has_neg:
        return {
            "curvature_class": "indefinite",
            "second_order_conclusion":
                "rules out both a local minimum and a local maximum"
        }

    if all_near_zero:
        return {
            "curvature_class": "numerically zero",
            "second_order_conclusion":
                "inconclusive; higher-order analysis is required"
        }

    if has_pos and not has_neg:
        return {
            "curvature_class": "positive semidefinite, not definite",
            "second_order_conclusion":
                "compatible with a minimum necessary condition, "
                "rules out a maximum, but is not sufficient for "
                "a strict minimum"
        }

    return {
        "curvature_class": "negative semidefinite, not definite",
        "second_order_conclusion":
            "compatible with a maximum necessary condition, "
            "rules out a minimum, but is not sufficient for "
            "a strict maximum"
    }

def solve_equality_qp(Q, q, A, b, tol=1e-10):
    Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1, 1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1, 1)

    n = q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must have shape (n, n)")
    if not np.allclose(Q, Q.T, atol=tol, rtol=0.0):
        raise ValueError("Q must be symmetric")
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("A must have shape (m, n)")

    m = A.shape[0]
    if b.shape != (m, 1):
        raise ValueError("b must have shape (m, 1)")

    Z, rank = nullspace_basis(A)
    if rank < m:
        raise ValueError("LICQ fails: A lacks full row rank")

    K = np.block([
        [Q, A.T],
        [A, np.zeros((m, m))]
    ])
    rhs = np.vstack([-q, b])

    try:
        solution = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("KKT matrix is singular") from exc

    x = solution[:n]
    lam = solution[n:]

    feasibility = A @ x - b
    stationarity = Q @ x + q + A.T @ lam
    reduced = Z.T @ Q @ Z
    eigvals = (
        np.linalg.eigvalsh(reduced)
        if reduced.size
        else np.array([], dtype=float)
    )
    classification = classify_reduced_hessian(eigvals, tol)

    result = {
        "x": x,
        "lambda": lam,
        "rank_A": rank,
        "Z": Z,
        "reduced_hessian": reduced,
        "reduced_eigenvalues": eigvals,
        "feasibility_norm":
            float(np.linalg.norm(feasibility, 2)),
        "stationarity_norm":
            float(np.linalg.norm(stationarity, 2)),
    }
    result.update(classification)
    return result

if __name__ == "__main__":
    Q = np.array([[1.0, 0.0],
                  [0.0, 2.0]])
    q = np.array([[-2.0],
                  [-6.0]])
    A = np.array([[1.0, 1.0]])
    b = np.array([[2.0]])

    result = solve_equality_qp(Q, q, A, b)
    for key, value in result.items():
        print(key, value)
```

程式中的 `q`、`b`、`x` 與 `lambda` 都使用明確的二維列向量形狀，沒有以 NumPy 一維陣列的 `.T` 冒充轉置。

本章未執行程式。下一節結果均為解析推導所得的預期結果，不是實際執行紀錄。此程式直接處理線性約束二次問題；對非線性約束，必須另行計算完整的 $H_L$。

---

## 測試與預期結果

### 正常測試：正定受限 Hessian

使用例二資料，預期

$$
x=
\begin{bmatrix}
0\\2
\end{bmatrix},
\qquad
\lambda=
\begin{bmatrix}
2
\end{bmatrix}.
$$

可行殘差與站立殘差預期接近浮點捨入尺度。受限 Hessian 的唯一特徵值為 $1.5$，預期分類為正定，並標示「嚴格局部最小的充分條件」。

SVD 所產生的 $Z$ 可能與手算基底相差一個負號，但 $Z^TQZ$ 不變。

### 正常測試：負定

取

$$
Q=-I_2,\qquad
q=0,\qquad
A=\begin{bmatrix}1&0\end{bmatrix},
\qquad b=0.
$$

切空間由第二座標方向張成，受限 Hessian 為 $[-1]$。預期分類為負定，並標示嚴格局部最大的充分條件。

### 正常測試：不定

取

$$
Q=\operatorname{diag}(1,-1,2),
\qquad
A=\begin{bmatrix}0&0&1\end{bmatrix},
\qquad b=0.
$$

切空間是前兩座標平面，受限 Hessian 特徵值為 $1,-1$。預期分類為不定，因而排除局部最小與局部最大。對此二次問題，沿可行方向 $(t,0,0)^T$ 目標增加，而沿 $(0,t,0)^T$ 目標減少，可直接看出鞍型行為。

### 邊界測試：正半定但不正定

取零條約束，並令

$$
Q=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix}.
$$

受限 Hessian 特徵值為 $0,1$。預期分類為「正半定但不正定」。它與局部最小的二階必要條件相容，也因存在正方向而排除局部最大，但不能保證嚴格局部最小。

對這個精確二次目標，若 $q=0$，原點確實是非嚴格全域最小；這是利用完整二次模型得到的額外結論，不是一般半正定測試自動提供的嚴格性結論。

### 邊界測試：零維切空間

取 $A=I_n$。可行點由 $x=b$ 唯一決定，$Z$ 的形狀為 $n\times0$，受限 Hessian 是 $0\times0$ 矩陣。預期分類為「零維切空間」與「LICQ 下局部孤立可行點」，而非宣稱觀察到正曲率。

### 故障測試：冗餘約束

令

$$
A=
\begin{bmatrix}
1&1\\
2&2
\end{bmatrix},
\qquad
b=
\begin{bmatrix}
2\\4
\end{bmatrix}.
$$

因 $\operatorname{rank}A=1<2$，預期程式拋出 `ValueError`，指出 LICQ 失敗。可行集合本身仍可能良好，但乘數不唯一，KKT 矩陣也可能奇異。

### 故障測試：不相容約束

若沿用上述 $A$，但令

$$
b=
\begin{bmatrix}
2\\5
\end{bmatrix},
$$

則可行集合為空。現有程式會先因秩不足拒絕輸入，並不自行區分「冗餘且相容」與「冗餘且不相容」。實務上應另以最小平方殘差檢查 $Ax=b$ 的相容性。

---

## 反例與常見陷阱

### 1. 秩退化使零空間高估真實切空間

考慮

$$
c(x,y)=x^2+y^2=0.
$$

可行集合只有原點，但

$$
J_c(0,0)=
\begin{bmatrix}
0&0
\end{bmatrix},
$$

因此

$$
\ker J_c(0,0)=\mathbb R^2.
$$

任何可行曲線都只能是常曲線，故真實曲線切空間只有 $\{0\}$。沒有 LICQ 時，

$$
T_xM=\ker J_c(x)
$$

可能完全失效。

再令 $f(x,y)=x$。原點是唯一可行點，因此同時是受限最小與最大；可是 Lagrange 方程要求

$$
\begin{bmatrix}
1\\0
\end{bmatrix}
+
\lambda
\begin{bmatrix}
0\\0
\end{bmatrix}
=0,
$$

不可能成立。這說明約束資格是必要條件定理中的實質假設。

### 2. 一階駐點不等於最小點

在單位圓上令 $f(x,y)=x$。點 $(1,0)$ 與 $(-1,0)$ 都滿足 Lagrange 方程，但前者是最大點，後者是最小點。站立條件只能產生候選點。

### 3. 不能只檢查目標 Hessian

對非線性約束，正確的二階矩陣是

$$
H_L=
\nabla^2f+
\sum_i\lambda_i\nabla^2c_i.
$$

圓上的線性目標滿足 $\nabla^2f=0$，但沿圓周仍有非零二階曲率；此曲率來自約束 Hessian。

### 4. 半定或零 Hessian 無法完整分類

在約束 $y=0$ 上比較

$$
f_1(x,y)=x^4,
\qquad
f_2(x,y)=-x^4,
\qquad
f_3(x,y)=x^3.
$$

三者在原點的受限 Hessian 都是零，但 $f_1$ 有嚴格局部最小，$f_2$ 有嚴格局部最大，$f_3$ 不是局部極值。因此零 Hessian 必須標示為未判定。

### 5. 不定與高階項

若受限 Hessian 不定，存在切向 $d_+$、$d_-$ 使

$$
d_+^TH_Ld_+>0,
\qquad
d_-^TH_Ld_-<0.
$$

沿對應的局部可行曲線，二階 Taylor 主項分別具有正負符號，而餘項是 $o(t^2)$，不能消除固定非零二次係數的符號。因此在 $C^2$、LICQ 與站立條件下，不定確實排除局部極值。不能把「半定未判定」錯誤延伸成「不定也未判定」。

### 6. 數值分類不是定理證明

浮點特徵值依賴尺度與容差。接近零的特徵值可能因捨入略正或略負。程式中的 `tol` 只是數值證據門檻，不是數學正定性的證明。殘差小也不代表問題條件良好。

---

## AI、幾何與養殖案例

考慮純合成的養殖感測校準。設兩個無因次化增益修正為

$$
\theta=
\begin{bmatrix}
\theta_1\\
\theta_2
\end{bmatrix}.
$$

為固定總體增益，施加

$$
c(\theta)=\theta_1+\theta_2=0.
$$

其 Jacobian 為

$$
J_c=
\begin{bmatrix}
1&1
\end{bmatrix},
$$

切空間為

$$
\ker J_c=
\operatorname{span}
\left\{
\begin{bmatrix}
1\\-1
\end{bmatrix}
\right\}.
$$

可取單位基底

$$
Z=\frac1{\sqrt2}
\begin{bmatrix}
1\\-1
\end{bmatrix}.
$$

設合成損失為

$$
f(\theta)=
\frac12\|W(M\theta-r)\|_2^2.
$$

若原始殘差同時包含攝氏溫度與溶氧濃度 $\mathrm{mg/L}$，不同單位不能直接放進無權重 Euclidean 長度。矩陣 $W$ 應包含各輸出的參考尺度倒數，使殘差先無因次化；返回物理值時再乘回相應尺度。

令

$$
H=M^TW^TWM,
\qquad
g=-M^TW^TWr.
$$

問題成為

$$
\min_\theta
\frac12\theta^TH\theta+g^T\theta
\quad\text{使得}\quad
J_c\theta=0.
$$

切向可辨識性由

$$
Z^THZ
$$

衡量。若此量明顯為正，二次模型在唯一可行參數方向上有正曲率；若接近零，參數可能對資料噪音高度敏感。即使 KKT 殘差很小，也不能把弱曲率誤解為可靠辨識。

AI 系統可以整理 Jacobian、LICQ、乘數、殘差與受限 Hessian，但有限樣本不能證明整個參數域上的正則性。此處是合成案例；模型校準不等於現場驗證，數學穩定也不等於操作安全。唯讀分析不控制設備、不投餌、不加藥。

---

## 習題

1. **手算題。** 求 $f(x,y)=x^2+y^2$ 在約束 $x+2y=3$ 下的最小點、乘數及受限 Hessian。

2. **程式題。** 使用本章程式解
   $$
   Q=
   \begin{bmatrix}
   4&1\\
   1&2
   \end{bmatrix},
   \quad
   q=
   \begin{bmatrix}
   -1\\-1
   \end{bmatrix},
   \quad
   A=\begin{bmatrix}1&-1\end{bmatrix},
   \quad b=0.
   $$
   先手算預期值，再列出應檢查的殘差與分類。

3. **反例題。** 對約束
   $$
   c(x,y)=y^2-x^3=0,
   $$
   求原點的 $J_c$ 與零空間，並給出通過原點的可行曲線。解釋線性化為何失效。

4. **整合題。** 設
   $$
   f(x,y)=x,\qquad
   c(x,y)=x^2+y^2-1.
   $$
   求所有 Lagrange 駐點，以受限 Hessian 分類，並區分一階必要、二階充分與全域結論。

---

## 習題解答

### 第1題

Lagrangian 為

$$
L=x^2+y^2+\lambda(x+2y-3).
$$

一階條件為

$$
2x+\lambda=0,
\qquad
2y+2\lambda=0,
\qquad
x+2y=3.
$$

因此

$$
x=-\frac{\lambda}{2},
\qquad
y=-\lambda.
$$

代入約束得

$$
\lambda=-\frac65,
$$

所以

$$
(x^\star,y^\star)=
\left(\frac35,\frac65\right).
$$

可取切向基底

$$
Z=\frac1{\sqrt5}
\begin{bmatrix}
-2\\1
\end{bmatrix}.
$$

約束為線性，故 $H_L=2I$，因此

$$
Z^TH_LZ=2>0.
$$

此點是嚴格受限局部最小。由目標嚴格凸與約束仿射，亦知它是唯一全域最小。

### 第2題

約束 $x-y=0$ 給出 $x=y=t$。代入目標：

$$
\frac12
\begin{bmatrix}
t&t
\end{bmatrix}
Q
\begin{bmatrix}
t\\t
\end{bmatrix}
-
\begin{bmatrix}
1&1
\end{bmatrix}
\begin{bmatrix}
t\\t
\end{bmatrix}
=4t^2-2t.
$$

微分得 $8t-2=0$，故

$$
x^\star=y^\star=\frac14.
$$

又

$$
Qx^\star+q=
\begin{bmatrix}
1/4\\-1/4
\end{bmatrix},
$$

所以

$$
\lambda=-\frac14.
$$

切向基底可取

$$
Z=\frac1{\sqrt2}
\begin{bmatrix}
1\\1
\end{bmatrix},
$$

並有

$$
Z^TQZ=4>0.
$$

預期程式分類為正定。應檢查

$$
\|Ax^\star-b\|_2
$$

與

$$
\|Qx^\star+q+A^T\lambda\|_2
$$

皆接近浮點捨入尺度。因 $Q$ 正定，此解也是唯一全域最小。

### 第3題

有

$$
J_c(x,y)=
\begin{bmatrix}
-3x^2&2y
\end{bmatrix}.
$$

因此

$$
J_c(0,0)=
\begin{bmatrix}
0&0
\end{bmatrix},
\qquad
\ker J_c(0,0)=\mathbb R^2.
$$

可行曲線可取

$$
\gamma_+(t)=(t^2,t^3),
\qquad
\gamma_-(t)=(t^2,-t^3).
$$

兩者皆滿足 $y^2=x^3$，但其在原點的速度都是零。可行集合在原點形成尖點，不是正則的一維 $C^1$ 子流形；Jacobian 零空間卻是整個平面。失效原因正是 LICQ 不成立。

### 第4題

Lagrangian 為

$$
L=x+\lambda(x^2+y^2-1).
$$

一階條件為

$$
1+2\lambda x=0,
\qquad
2\lambda y=0,
\qquad
x^2+y^2=1.
$$

$\lambda$ 不可能為零，所以 $y=0$、$x=\pm1$。在 $(1,0)$，

$$
\lambda=-\frac12;
$$

在 $(-1,0)$，

$$
\lambda=\frac12.
$$

兩點皆可取切向基底

$$
Z=
\begin{bmatrix}
0\\1
\end{bmatrix}.
$$

由 $H_L=2\lambda I$ 可得：

$$
Z^TH_LZ=-1
$$

於 $(1,0)$，所以它是嚴格受限局部最大；

$$
Z^TH_LZ=1
$$

於 $(-1,0)$，所以它是嚴格受限局部最小。

Lagrange 方程提供一階必要候選點；受限 Hessian 的定號提供二階充分判別。最後，由單位圓上 $-1\le x\le1$，可知兩點分別是全域最大與全域最小。圓的緊緻性與 $f$ 的連續性則保證全域極值存在。

---

## 本章小結

等式約束最佳化的核心鏈條是

$$
\text{LICQ}
\Longrightarrow
T_xM=\ker J_c(x)
\Longrightarrow
\nabla f(x)\perp T_xM
\Longrightarrow
\nabla f(x)+J_c(x)^T\lambda=0.
$$

其中 $J_c(x)\in\mathbb R^{m\times n}$；正則情況下，切空間維度為 $n-m$。一階 Lagrange 條件只產生候選點。

二階判別必須使用完整的 Lagrangian Hessian：

$$
H_L=
\nabla^2f+
\sum_i\lambda_i\nabla^2c_i.
$$

在切空間基底 $Z$ 下，分析

$$
Z^TH_LZ.
$$

正定與負定分別給出嚴格局部最小與最大的充分條件；不定違反極值的二階必要條件，因而排除局部極值；半定只提供必要條件相容性，通常不能完成分類。秩退化時，線性化切空間可能失真，乘數也可能不存在或不唯一。

數值實作應報告秩、可行殘差、站立殘差、特徵值與分類容差。有限精度計算只能提供數值證據，不能取代 LICQ、可微性與二階定理的證明。

---

## 參考來源

1. Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>
2. MIT OpenCourseWare，18.100A Real Analysis：<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>
3. MIT OpenCourseWare，18.02SC Multivariable Calculus：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>
4. SciPy，`minimize` 方法與參數文件，作為延伸最佳化 API 參考：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>

本章核心程式不依賴 SciPy，也未執行外部程式。以上來源僅作教材與延伸入口；各定理仍以正文所列的有限維、正則性與秩條件為準。