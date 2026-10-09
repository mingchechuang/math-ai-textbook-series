# 第10章 矩陣微分與Frobenius內積

## 學習目標與先備知識

本章把矩陣視為有限維向量空間中的元素，並以 Frobenius 內積描述矩陣變數的微分。完成本章後，你應能：

1. 說明矩陣變數的微分如何成為線性映射，並辨認微分、梯度與矩陣形狀。
2. 使用 $df=\operatorname{tr}(G^T dX)$，從微分式找出與 $X$ 同形狀的梯度 $G$。
3. 推導 $\frac12\|AX-B\|_F^2$ 對 $X$ 的梯度與方向導數。
4. 對矩陣函數組合使用鏈式法則，並用方向擾動核對梯度。
5. 辨認轉置、維度及程式廣播造成的錯誤，並說明數值檢查能支持什麼、不能證明什麼。

先備知識包括矩陣乘法、轉置、跡、向量微分與 Fréchet 可微的定義。除非另有說明，矩陣空間採 Frobenius 範數，向量空間採 Euclidean 範數。本章的矩陣均為實矩陣。

## 問題與直覺

令 $X\in\mathbb R^{p\times q}$。它有 $pq$ 個實數分量，可以列成一個長度 $pq$ 的向量；但推導和實作時保留矩陣形狀，通常更能看清運算結構。矩陣擾動記為 $H\in\mathbb R^{p\times q}$。若純量函數 $f$ 在 $X$ 可微，則在 $X$ 附近有線性近似

$$
f(X+H)=f(X)+Df(X)[H]+r(H),
\qquad
\frac{|r(H)|}{\|H\|_F}\longrightarrow0
\quad\text{當 }\|H\|_F\to0.
$$

這裡 $Df(X)$ 是作用於矩陣擾動的線性泛函：輸入是 $p\times q$ 矩陣，輸出是純量。它描述一階變化，不是另一個與 $X$ 同形狀的矩陣。

Frobenius 內積定義為

$$
\langle U,V\rangle_F=\operatorname{tr}(U^T V)
=\sum_{i,j}U_{ij}V_{ij},
$$

對應範數為

$$
\|U\|_F=\sqrt{\langle U,U\rangle_F}.
$$

若存在唯一矩陣 $G\in\mathbb R^{p\times q}$，使得對所有 $H$ 都有

$$
Df(X)[H]=\langle G,H\rangle_F
=\operatorname{tr}(G^T H),
$$

便稱 $G$ 為 $f$ 在 $X$ 的 Frobenius 梯度，記為 $\nabla_X f(X)$。梯度與變數同形狀，是因為此處選用 Frobenius 內積來代表線性泛函；微分和梯度仍是不同的數學物件。

方向導數提供一個直覺：沿矩陣方向 $H$ 改變 $X$，函數的一階變化率為 $\langle\nabla_X f,H\rangle_F$。若此值為正，沿 $H$ 的小幅變化使函數增加；若為負，則使函數減少；若為零，只表示這個方向的一階變化為零，不表示梯度本身為零。方向導數是一個數，梯度則是一個矩陣，兩者不可互換。

## 定義、定理與推導

### 矩陣微分的定義

對 $f:\mathbb R^{p\times q}\to\mathbb R$，若存在線性映射 $L:\mathbb R^{p\times q}\to\mathbb R$，使得

$$
f(X+H)=f(X)+L[H]+r(H),
\qquad
\frac{|r(H)|}{\|H\|_F}\to0,
$$

則稱 $f$ 在 $X$ Fréchet 可微，並記 $L=Df(X)$。這個極限的意思是：當擾動縮小時，線性近似的誤差相對於擾動大小也趨近於零。只知道某個固定方向上的差商有極限，並不足以得到這個結論；Fréchet 可微要求所有小擾動一致地滿足所述餘項條件。

微分記號 $df$ 是在固定基點 $X$ 下對線性映射 $Df(X)$ 的簡記。若推導得到

$$
df=\operatorname{tr}(G^T dX),
$$

其精確含義是：對任意方向 $H$，都有 $Df(X)[H]=\operatorname{tr}(G^T H)$。式中的 $dX$ 不應被當成可獨立任意運算的數字。

### Frobenius 內積表示命題

**命題。** 對任意線性泛函 $L:\mathbb R^{p\times q}\to\mathbb R$，存在唯一的 $G\in\mathbb R^{p\times q}$，使得對所有 $H\in\mathbb R^{p\times q}$，

$$
L[H]=\operatorname{tr}(G^T H).
$$

**證明。** 設 $E_{ij}$ 為只有第 $(i,j)$ 個元素等於 $1$、其餘元素皆為 $0$ 的矩陣。每個 $H$ 都能寫成有限和

$$
H=\sum_{i=1}^{p}\sum_{j=1}^{q}H_{ij}E_{ij}.
$$

由 $L$ 的線性，

$$
L[H]=\sum_{i=1}^{p}\sum_{j=1}^{q}H_{ij}L[E_{ij}].
$$

定義 $G_{ij}=L[E_{ij}]$，便有

$$
L[H]=\sum_{i,j}G_{ij}H_{ij}
=\operatorname{tr}(G^T H),
$$

故表示矩陣存在。若 $G$ 與 $\widetilde G$ 都能表示同一個 $L$，對 $H=E_{ij}$ 代入，得到 $G_{ij}=L[E_{ij}]=\widetilde G_{ij}$。這對每一個 $i,j$ 都成立，因此 $G=\widetilde G$，唯一性得證。證畢。

此命題依賴有限維矩陣空間與指定的 Frobenius 內積。它保證一個線性泛函有唯一的同形狀矩陣表示，卻不代表任何未整理的微分式都能直接把某個因子當梯度。必須先把微分寫成 $\operatorname{tr}(G^T dX)$，並確認維度相容，才能讀出梯度。

### 跡運算與微分規則

矩陣尺寸相容時，常用恆等式包括

$$
\operatorname{tr}(U^T V)=\operatorname{tr}(V^T U),
\qquad
\operatorname{tr}(PQ)=\operatorname{tr}(QP),
$$

以及循環移位

$$
\operatorname{tr}(P_1P_2\cdots P_k)
=\operatorname{tr}(P_2\cdots P_kP_1).
$$

循環移位要求乘積有定義，移位後的乘積也形成方陣。它不是任意交換：通常不能把矩陣乘積中的因子任意調換，更不能由跡相等推出矩陣本身相等。推導時應同時記錄乘法順序與矩陣形狀。

對固定矩陣 $A,C$，令 $Y=AXC$，則

$$
dY=A(dX)C.
$$

這是因為 $Y(X+H)-Y(X)=AHC$，其對 $H$ 線性，且沒有額外餘項。

若 $R=R(X)$，則

$$
\begin{aligned}
d\left(\frac12\|R\|_F^2\right)
&=d\left(\frac12\operatorname{tr}(R^T R)\right)\\
&=\frac12\operatorname{tr}\big((dR)^T R+R^T dR\big)\\
&=\operatorname{tr}(R^T dR).
\end{aligned}
$$

最後一步使用前兩項的跡相等。係數 $\frac12$ 正好抵消兩項相同的貢獻。

### 矩形最小平方梯度

令 $A\in\mathbb R^{m\times n}$、$X\in\mathbb R^{n\times p}$、$B\in\mathbb R^{m\times p}$，並定義

$$
R=AX-B,\qquad F(X)=\frac12\|AX-B\|_F^2.
$$

因為 $A,B$ 固定，$dR=A\,dX$。因此

$$
\begin{aligned}
dF
&=\operatorname{tr}(R^T dR)\\
&=\operatorname{tr}(R^T A\,dX)\\
&=\operatorname{tr}\big((A^T R)^T dX\big).
\end{aligned}
$$

所以

$$
\nabla_X F=A^T(AX-B)\in\mathbb R^{n\times p}.
$$

形狀也可獨立核對：$A^T$ 為 $n\times m$，殘差為 $m\times p$，乘積是 $n\times p$，與 $X$ 同形狀。若目標函數改為 $\|AX-B\|_F^2$，則梯度多一個因子 $2$。

此處的結論對所有實矩陣 $A,X,B$ 都成立，不要求 $A$ 可逆，也不要求最小平方問題有唯一解。唯一解與否牽涉最佳化問題的秩與幾何性質，不影響這個梯度公式的推導。

### 矩陣鏈式法則

設 $Y=\Phi(X)\in\mathbb R^{a\times b}$，且 $f(Y)$ 為純量函數。對任意方向 $H$，鏈式法則是

$$
D(f\circ\Phi)(X)[H]
=Df(Y)[D\Phi(X)[H]].
$$

令 $G_Y=\nabla_Y f$，則

$$
D(f\circ\Phi)(X)[H]
=\langle G_Y,D\Phi(X)[H]\rangle_F.
$$

因此，$X$ 的梯度由 $D\Phi(X)$ 在 Frobenius 內積下的伴隨映射作用於 $G_Y$ 得到。若 $D\Phi(X)[H]=AHC$，則

$$
\begin{aligned}
\langle G_Y,AHC\rangle_F
&=\operatorname{tr}(G_Y^T AHC)\\
&=\operatorname{tr}(C G_Y^T A H)\\
&=\operatorname{tr}\big((A^T G_Y C^T)^T H\big).
\end{aligned}
$$

所以

$$
\nabla_X(f\circ\Phi)=A^T G_Y C^T.
$$

此式中轉置的位置來自 Frobenius 內積與跡的整理，並非可任意套用的記憶口訣。對不熟悉的矩陣映射，最可靠的方法是先算 $D\Phi(X)[H]$，再將純量方向導數整理成 $\operatorname{tr}(G_X^T H)$。

### 正則性與證據界線

本章用到的線性與多項式型矩陣映射，例如 $X\mapsto AX-B$、$X\mapsto X^T X$，在有限維矩陣空間處處可微。以 $X^T X$ 為例，

$$
(X+H)^T(X+H)
=X^T X+X^T H+H^T X+H^T H.
$$

前兩個新增的一階項構成線性微分；餘項 $H^T H$ 的範數可由矩陣範數不等式界定為二階量。因此餘項除以 $\|H\|_F$ 後趨於零。這說明相應微分不是只靠形式規則猜出來的。

數值檢查則回答較窄的問題。若某次方向差分與解析方向導數明顯不符，可以指出該實作在這個輸入、方向或步長附近有問題；若若干方向吻合，只能說這些測試沒有發現錯誤。有限測試不涵蓋所有方向，也不證明餘項比值在所有趨近方式下都趨於零。

## 逐步手算例題

### 例題一：內積型純量函數

令 $A,X\in\mathbb R^{2\times2}$，其中

$$
A=
\begin{bmatrix}
1&2\\
-1&3
\end{bmatrix},
\qquad
X=
\begin{bmatrix}
2&0\\
1&-1
\end{bmatrix},
\qquad
f(X)=\operatorname{tr}(A^T X).
$$

先展開函數：

$$
f(X)=\sum_{i,j}A_{ij}X_{ij}
=1\cdot2+2\cdot0+(-1)\cdot1+3\cdot(-1)=-2.
$$

對任意擾動 $H$，

$$
f(X+H)-f(X)=\operatorname{tr}(A^T H).
$$

此差值恰為線性函數，餘項為零，因此

$$
Df(X)[H]=\operatorname{tr}(A^T H),
\qquad
\nabla_X f=A.
$$

再取

$$
H=\begin{bmatrix}1&-2\\0&1\end{bmatrix}.
$$

方向導數為

$$
Df(X)[H]=1\cdot1+2\cdot(-2)+(-1)\cdot0+3\cdot1=0.
$$

這個方向與 $A$ 在 Frobenius 內積下正交。方向導數為零不表示梯度為零；梯度仍是 $A$，而且與 $X$ 同為 $2\times2$。

### 例題二：矩形最小平方目標

令

$$
A=\begin{bmatrix}1&2\\0&1\\1&0\end{bmatrix},
\quad
X=\begin{bmatrix}1\\-1\end{bmatrix},
\quad
B=\begin{bmatrix}0\\1\\2\end{bmatrix},
\qquad
F(X)=\frac12\|AX-B\|_F^2.
$$

$A$ 為 $3\times2$，$X$ 為 $2\times1$，$B$ 為 $3\times1$，故殘差 $R$ 是 $3\times1$。直接相乘得到

$$
AX=
\begin{bmatrix}-1\\-1\\1\end{bmatrix},
\qquad
R=AX-B=
\begin{bmatrix}-1\\-2\\-1\end{bmatrix}.
$$

因此

$$
F(X)=\frac12(1+4+1)=3.
$$

由一般公式，

$$
\nabla_XF=A^TR
=
\begin{bmatrix}1&0&1\\2&1&0\end{bmatrix}
\begin{bmatrix}-1\\-2\\-1\end{bmatrix}
=
\begin{bmatrix}-2\\-4\end{bmatrix}.
$$

梯度是 $2\times1$，與 $X$ 同形狀。取 $H=(1,1)^T$，則

$$
DF(X)[H]=\langle\nabla_XF,H\rangle_F=-2-4=-6.
$$

也可將目標沿此方向展開：

$$
F(X+tH)=\frac12\|R+tAH\|_2^2.
$$

其中 $R^TAH=(A^TR)^TH=-6$，所以 $t$ 的線性係數為 $-6$，與方向導數相符。這是代數核對，不依賴選定有限差分步長。

## 實作與程式

以下實作使用 Python 標準函式庫，不依賴 NumPy 或 SciPy。矩陣以巢狀列表表示，程式會檢查矩形形狀與矩陣乘法的內維度，避免形狀錯誤被廣播或不明確的運算掩蓋。程式包含轉置、矩陣乘法、加減、Frobenius 內積、最小平方目標與解析梯度，也用中心有限差分核對指定方向。

方向差分是數值診斷，不是定理證明。它有有限步長，並受到浮點捨入影響。本文未執行程式，因此下方測試均以預期結果敘述，不宣稱已通過。

```python
def shape(A):
    if not isinstance(A, list) or not A:
        raise ValueError("矩陣必須是非空列表")
    if not all(isinstance(row, list) for row in A):
        raise ValueError("矩陣每一橫列必須是列表")
    width = len(A[0])
    if width == 0 or any(len(row) != width for row in A):
        raise ValueError("矩陣必須是非空矩形")
    return len(A), width


def transpose(A):
    rows, cols = shape(A)
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def matmul(A, B):
    m, n = shape(A)
    n2, p = shape(B)
    if n != n2:
        raise ValueError("矩陣乘法內維度不相容")
    return [
        [sum(A[i][k] * B[k][j] for k in range(n))
         for j in range(p)]
        for i in range(m)
    ]


def subtract(A, B):
    if shape(A) != shape(B):
        raise ValueError("相減矩陣形狀必須相同")
    return [
        [A[i][j] - B[i][j] for j in range(len(A[0]))]
        for i in range(len(A))
    ]


def add(A, B):
    if shape(A) != shape(B):
        raise ValueError("相加矩陣形狀必須相同")
    return [
        [A[i][j] + B[i][j] for j in range(len(A[0]))]
        for i in range(len(A))
    ]


def scale(c, A):
    shape(A)
    return [[c * value for value in row] for row in A]


def frobenius_inner(A, B):
    if shape(A) != shape(B):
        raise ValueError("Frobenius 內積要求矩陣形狀相同")
    return sum(
        A[i][j] * B[i][j]
        for i in range(len(A))
        for j in range(len(A[0]))
    )


def least_squares(A, X, B):
    if shape(A)[0] != shape(B)[0]:
        raise ValueError("A 和 B 必須有相同列數")
    residual = subtract(matmul(A, X), B)
    value = 0.5 * frobenius_inner(residual, residual)
    gradient = matmul(transpose(A), residual)
    return value, gradient, residual


def directional_finite_difference(A, X, B, H, step):
    plus = add(X, scale(step, H))
    minus = add(X, scale(-step, H))
    f_plus = least_squares(A, plus, B)[0]
    f_minus = least_squares(A, minus, B)[0]
    return (f_plus - f_minus) / (2.0 * step)


def run_checks():
    A = [[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]]
    X = [[1.0], [-1.0]]
    B = [[0.0], [1.0], [2.0]]
    H = [[1.0], [1.0]]

    value, gradient, residual = least_squares(A, X, B)
    assert shape(residual) == (3, 1)
    assert shape(gradient) == (2, 1)
    assert abs(value - 3.0) < 1e-12
    assert gradient == [[-2.0], [-4.0]]

    exact_direction = frobenius_inner(gradient, H)
    fd = directional_finite_difference(A, X, B, H, 1e-5)
    assert abs(exact_direction + 6.0) < 1e-12
    assert abs(fd - exact_direction) < 1e-8

    # 邊界案例：殘差為零時，目標與梯度都應為零。
    B_exact = matmul(A, X)
    value0, gradient0, _ = least_squares(A, X, B_exact)
    assert value0 == 0.0
    assert all(abs(v) < 1e-12 for row in gradient0 for v in row)

    # 故障案例：不相容的轉置形狀應被明確拒絕。
    transpose_rejected = False
    try:
        least_squares(A, transpose(X), B)
    except ValueError:
        transpose_rejected = True
    assert transpose_rejected

    # 故障案例：不同形狀不能直接作 Frobenius 內積。
    inner_shape_rejected = False
    try:
        frobenius_inner([[1.0, 2.0]], [[1.0], [2.0]])
    except ValueError:
        inner_shape_rejected = True
    assert inner_shape_rejected

    return value, gradient, exact_direction, fd


if __name__ == "__main__":
    print(run_checks())
```

若照預期執行，正常案例的目標值為 $3$、梯度為 $[[-2],[-4]]$，解析方向導數為 $-6$，中心有限差分應在浮點誤差範圍內接近 $-6$。邊界案例令 $B=AX$，因此殘差與目標值為零，梯度也為零。故障案例應因形狀不相容而觸發 $ValueError$，且檢查旗標應為真。這些是預期結果，並非實際執行紀錄。

## 測試與預期結果

測試分為正常、邊界與故障三類，避免只檢查一個典型輸入。

| 類別 | 測試 | 預期 |
|---|---|---|
| 正常 | 用矩形 $A$、$X$、$B$ 計算解析梯度，並與指定方向的中心差分比較 | 殘差為 $3\times1$，梯度為 $2\times1$；方向導數核對接近 $-6$ |
| 邊界 | 設 $B=AX$，使殘差為零 | 目標值與梯度皆為零 |
| 故障 | 把 $X$ 轉置後輸入，或以不同形狀矩陣計算 Frobenius 內積 | 明確拒絕不相容形狀，不進行隱式廣播 |

若改變差分步長，誤差可能先因截斷誤差下降而減小，之後又受捨入誤差影響而增加。這是一般數值分析上的可能現象，不是本程式已測得的收斂曲線。單一步長的吻合只是一項有限精度證據；解析展開才是本章梯度公式的數學依據。

## 反例與常見陷阱

**把矩陣微分當成沒有形狀的符號。** 在 $A\in\mathbb R^{m\times n}$、$X\in\mathbb R^{n\times p}$ 的最小平方目標中，梯度必須是 $n\times p$。把 $A^TR$ 寫成 $RA^T$，即使某些尺寸剛好可以相乘，也不表示該式是 $X$ 的梯度。每個中間結果都應標註形狀，而不能只靠「看起來有那些因子」判斷。

**混淆梯度和微分。** $Df(X)$ 是將擾動 $H$ 映到純量的一個線性泛函；$\nabla_Xf$ 則是在指定 Frobenius 內積下代表這個泛函的矩陣。換用加權內積時，線性泛函不變，代表矩陣卻可能改變。因此梯度不只取決於函數，也取決於採用的內積慣例。

**任意移動轉置或循環移位。** 跡有循環性，但矩陣乘法沒有交換律。每次移動因子都應確認跡的乘積仍有定義，並檢查最後是否確實得到 $\operatorname{tr}(G^T dX)$。若略去這些檢查，錯誤轉置有時會立刻造成維度不合，有時則會在方陣案例中悄悄留下錯誤數值。

**把程式廣播當成數學上的對齊。** 程式庫可能自動擴展形狀，例如把 $(n,1)$ 和 $(n,)$ 的陣列組合成二維結果。這種行為不一定對應所要的矩陣乘法或 Frobenius 內積。尤其 NumPy 的一維陣列形狀是 $(n,)$，對它使用 `.T` 不會變成 $(1,n)$。若要明確表示直向量與橫列，應使用 $(n,1)$ 與 $(1,n)$。

**忽略單位和權重。** 未加權 Frobenius 範數把每個分量的數值誤差等權計入。若矩陣包含不同物理單位或不同可靠度的資料，這代表一項需要說明的建模選擇，而不是自然成立的物理比較。可先將輸入與輸出按記錄的尺度無因次化，或在目標函數中明列權重，再重新推導梯度。不能直接把混合單位分量的 Euclidean 長度解釋成單一物理量。

## AI、幾何與養殖案例

**合成感測校準。** 設一個線性模型以 $X\in\mathbb R^{2\times2}$ 描述兩組感測輸出對兩個無因次校準特徵的響應；$A\in\mathbb R^{m\times2}$ 收集 $m$ 筆合成設計資料，$B\in\mathbb R^{m\times2}$ 是相應的合成量測摘要。殘差 $AX-B$ 的第 $(i,j)$ 個元素，是第 $i$ 筆合成資料中第 $j$ 個輸出的無因次誤差。目標

$$
F(X)=\frac12\|AX-B\|_F^2
$$

把各個分量的平方誤差相加，而其梯度為 $A^T(AX-B)$。梯度元素表示目標對相應參數元素的一階敏感度；它不是感測器輸出本身，也不自動代表可直接操作的調整量。

若原始輸入是以 SI 單位量測的溫度差、溶氧差或流速差，必須記錄各分量的單位與尺度。把不同單位的數值直接當作同尺度座標，會使距離與最小平方目標隱含不透明的權重。將輸入、輸出按明確尺度無因次化後，應保留尺度，以便把校準結果轉回物理量；Jacobian 元素的單位則是相應輸出單位除以輸入單位。無因次化能澄清尺度選擇，卻不代表模型已由現場資料驗證。

**AI 輔助推導。** 文字生成工具可以協助整理跡式或提出形狀檢查清單，但輸出的轉置和乘法次序仍須逐步核對。可稽核的檢查包括輸入、殘差、梯度的尺寸，從 $dF$ 整理到 $\operatorname{tr}(G^T dX)$ 的每一步，以及至少一個方向導數核對。數值吻合可以協助找錯，不能證明對所有矩陣擾動都成立。

養殖情境的資料與參數皆為合成案例。數學上的擬合不等於現場驗證，也不等於操作安全。分析 agent 可唯讀整理假設、單位與推導證據，不應據此改變設備、投餌或加藥。

## 習題

### 習題一：手算梯度與方向導數

令

$$
F(X)=\frac12\|AX-B\|_F^2,\quad
A=\begin{bmatrix}1&-1\\2&0\\0&1\end{bmatrix},\quad
X=\begin{bmatrix}1\\2\end{bmatrix},\quad
B=\begin{bmatrix}0\\1\\1\end{bmatrix}.
$$

求殘差、函數值與梯度。再取 $H=(2,-1)^T$，求方向導數，並用 $F(X+tH)$ 的線性項核對。

### 習題二：矩陣鏈式法則與轉置

設 $X\in\mathbb R^{2\times2}$、$Y=2XC$，其中

$$
C=\begin{bmatrix}1&0\\1&2\end{bmatrix},
\qquad
f(Y)=\frac12\|Y-D\|_F^2,
\qquad
D=\begin{bmatrix}0&1\\1&0\end{bmatrix}.
$$

求 $\nabla_X f$，說明各矩陣形狀，並以微分與跡式推導。

### 習題三：有限測試能否證明梯度

有人主張：「對若干個測試方向 $H_1,\ldots,H_k$，方向差分都接近 $\langle G,H_i\rangle_F$，所以 $G$ 已被證明是梯度。」此推論是否成立？說明有限方向測試能支持的結論，並描述一種可使有限測試吻合、但仍不足以證明可微的反例構造思路。

### 習題四：程式測試與證據標示

提出兩項最小平方實作測試：一項檢查解析梯度與方向差分，一項檢查形狀錯誤是否明確失敗。若沒有實際執行程式，應如何描述預期結果，才不會把預期當成測試證據？

## 習題解答

### 解答一

殘差為

$$
R=AX-B
=
\begin{bmatrix}
1&-1\\2&0\\0&1
\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}
-
\begin{bmatrix}0\\1\\1\end{bmatrix}
=
\begin{bmatrix}-1\\1\\1\end{bmatrix}.
$$

因此 $F(X)=\frac12(1+1+1)=\frac32$，而

$$
\nabla_XF=A^TR
=
\begin{bmatrix}1&2&0\\-1&0&1\end{bmatrix}
\begin{bmatrix}-1\\1\\1\end{bmatrix}
=
\begin{bmatrix}1\\2\end{bmatrix}.
$$

方向導數為

$$
DF(X)[H]
=
\begin{bmatrix}1&2\end{bmatrix}
\begin{bmatrix}2\\-1\end{bmatrix}
=0.
$$

又 $AH=(3,4,-1)^T$，所以

$$
F(X+tH)
=\frac12\|R+tAH\|_2^2
=\frac12\left(\|R\|_2^2+2tR^TAH+t^2\|AH\|_2^2\right).
$$

其中 $R^TAH=-3+4-1=0$，故線性項為零，與方向導數相符。二次項為 $13t^2$，不影響 $t=0$ 的一階導數。

### 解答二

令 $R=Y-D=2XC-D$，則

$$
dR=2(dX)C,
\qquad
df=\operatorname{tr}(R^T dR)
=2\operatorname{tr}(R^T(dX)C).
$$

循環移位並整理成與 $dX$ 的 Frobenius 內積：

$$
\begin{aligned}
df
&=2\operatorname{tr}(C R^T dX)\\
&=\operatorname{tr}\big((2RC^T)^T dX\big).
\end{aligned}
$$

所以

$$
\nabla_X f=2RC^T=2(2XC-D)C^T.
$$

$X,C,D,R$ 都是 $2\times2$，而 $dX$ 與梯度亦為 $2\times2$。$C^T$ 不能省略；它是跡式整理與梯度表示共同決定的。

### 解答三

推論不成立。有限方向只測試有限多個矩陣擾動，沒有檢查所有趨近零的方向，也沒有證明餘項相對於 $\|H\|_F$ 趨零。可以構造一個函數，使它在有限個已測方向上與候選線性近似一致，卻在未測方向或某條曲線路徑上有不符合該線性近似的變化。這說明有限測試吻合不等於 Fréchet 可微。若某個測試方向明顯不吻合，則能否定該輸入與方向下的實作主張；若測試吻合，只能說目前測試沒有發現錯誤。

### 解答四

正常測試可選尺寸相容的 $A,X,B,H$，用 $G=A^T(AX-B)$ 求解析方向導數 $\langle G,H\rangle_F$，再和中心差分比較。若未執行，應寫明「預期中心差分在浮點誤差範圍內接近解析值」，不能寫成「測試已通過」，也不能編造誤差數字。

故障測試可將 $X$ 改成與 $A$ 內維度不相容的形狀，或對形狀不同的矩陣呼叫 Frobenius 內積函式。預期程式明確拒絕輸入。沒有執行時，只能交代測試設計、預期錯誤路徑與判斷理由，不能聲稱實際觀察到特定例外。這些測試檢查程式的防護，不取代解析推導。

## 本章小結

矩陣微分將矩陣擾動映射到函數的一階變化。Fréchet 可微要求線性近似餘項相對於 $\|H\|_F$ 趨於零。在 Frobenius 內積下，任意有限維矩陣空間上的線性泛函可唯一表示為 $\operatorname{tr}(G^TH)$；因此，純量函數的 Frobenius 梯度 $G$ 與變數 $X$ 同形狀。

對半平方殘差目標 $\frac12\|AX-B\|_F^2$，微分為 $\operatorname{tr}((AX-B)^TA\,dX)$，梯度則是 $A^T(AX-B)$。矩陣鏈式法則可由方向微分與伴隨映射推導；轉置的位置應由跡運算和尺寸檢查決定。

解析推導、形狀檢查與方向差分的作用不同：推導提供數學依據，形狀檢查排除不相容運算，差分則協助發現實作錯誤。有限次數值測試不是微分極限的證明。遇到實際資料時，還要記錄單位、尺度與目標函數中的權重，避免把未加權的數值長度誤解為物理上可直接比較的距離。

## 參考來源

1. Jiří Lebl，*Basic Analysis*，作者教材入口。https://www.jirka.org/ra/
2. MIT OpenCourseWare，*18.02SC Multivariable Calculus*。https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/
3. JAX 文件，*Autodiff Cookbook: JVP and VJP*，作為方向推送與伴隨拉回的延伸參考。https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html

本章推導以有限維 Frobenius 內積、矩陣代數及 Fréchet 微分定義為基礎。外部資源提供延伸閱讀，不代表已逐條核查教材全文。程式未執行，測試結果均以預期方式標示。