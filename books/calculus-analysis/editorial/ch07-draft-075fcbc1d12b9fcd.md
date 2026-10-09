# 第07章 Fréchet導數與餘項條件

## 學習目標與先備知識

本章把「導數是斜率」推廣為「導數是最佳的一階線性映射」。完成本章後，讀者應能：

1. 對映射 $f:U\subset\mathbb R^n\to\mathbb R^m$ 寫出 Fréchet 可微的精確定義。
2. 區分增量、線性主部與非線性餘項：
   $$
   f(x+h)=f(x)+Df(x)[h]+r(h).
   $$
3. 使用指定範數驗證
   $$
   \frac{\|r(h)\|}{\|h\|}\to0.
   $$
4. 證明 Fréchet 導數若存在則唯一。
5. 理解 Fréchet 可微是「對所有足夠小擾動一致成立」的敘述，而不只是若干方向上的斜率存在。
6. 由解析式求出導數矩陣，並以數值殘差比作有限精度核對。
7. 說明矩陣擬合、方向掃描與有限差分只能提供數值證據，不能取代極限定義的證明。
8. 辨識方向導數存在但不 Fréchet 可微的反例。
9. 在具有物理單位的感測問題中，標示 Jacobian 各元素的單位，並先無因次化再比較不同分量的誤差。

先備知識包括有限維向量範數、矩陣乘法、極限定義、連續性，以及單變量導數。除非另行說明，本章在輸入空間與輸出空間都使用 Euclidean 範數 $\|\cdot\|_2$。在有限維空間中，其他範數給出相同的可微概念，但餘項界中的常數會改變，而且可能依賴維度。

---

## 問題與直覺

考慮一個雙輸入、雙輸出的映射。輸入可能是溫度與鹽度，輸出可能是兩個感測通道。若工作點是 $x$，而輸入受到小擾動 $h$，我們希望以矩陣 $A$ 預測輸出變化：

$$
f(x+h)-f(x)\approx Ah.
$$

這裡 $h$ 是 $n\times1$ 的列向量，$A$ 是 $m\times n$ 矩陣，$Ah$ 是 $m\times1$ 的列向量。近似是否真正代表可微，關鍵不只在「誤差很小」，而在誤差相對於擾動的一階尺度是否消失。令

$$
r(h)=f(x+h)-f(x)-Ah.
$$

要求是

$$
\frac{\|r(h)\|_2}{\|h\|_2}\to0
\qquad(h\to0,\ h\ne0).
$$

因此：

- 若 $\|r(h)\|$ 大約與 $\|h\|^2$ 同階，比例大約與 $\|h\|$ 同階，會趨近零。
- 若 $\|r(h)\|$ 大約與 $\|h\|$ 同階，比例通常不趨近零。
- 只知道 $r(h)\to0$ 不夠，因為任何連續函數都可能有增量趨零，但其誤差未必比 $\|h\|$ 更小。
- 檢查有限個方向不夠，因為「所有方向」本身仍可能比 Fréchet 可微要求弱；更不能以有限採樣取代無限方向與任意小尺度的量詞。

「最佳線性近似」不是指先指定一批資料點，再做最小平方所得到的最佳矩陣。分析中的意思是：存在一個固定線性映射，使其誤差為 $o(\|h\|)$。這個條件一旦成立，線性映射必然唯一。

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
\lim_{\substack{h\to0\\h\ne0}}
\frac{\|f(x+h)-f(x)-Ah\|_2}{\|h\|_2}=0,
$$

則稱 $f$ 在 $x$ Fréchet 可微，並記

$$
Df(x)=A.
$$

因為 $U$ 是開集，對足夠小的 $h$，$x+h$ 自動仍在 $U$。等價地，可寫成

$$
f(x+h)=f(x)+Df(x)[h]+r(h),
$$

其中

$$
\|r(h)\|_2=o(\|h\|_2).
$$

符號 $o(\|h\|)$ 的精確意思是：對每個 $\varepsilon>0$，存在 $\delta>0$，使得只要 $0<\|h\|_2<\delta$，就有

$$
\|r(h)\|_2\le\varepsilon\|h\|_2.
$$

這是極限定義，不是統計意義下的「大多數方向誤差很小」。

在標準座標下，有限維線性映射 $Df(x)$ 可由唯一矩陣表示。該矩陣記為 $J_f(x)\in\mathbb R^{m\times n}$，並滿足

$$
Df(x)[h]=J_f(x)h.
$$

本章重點是導數作為線性映射。矩陣只是選定座標後的表示。

### 2. 導數的唯一性

**命題 7.2（Fréchet 導數唯一）**  
設 $U\subset\mathbb R^n$ 為開集，$x\in U$，$f:U\to\mathbb R^m$。若線性映射 $A,B:\mathbb R^n\to\mathbb R^m$ 都滿足 Fréchet 可微的餘項條件，則 $A=B$。

**證明。**  
假設

$$
f(x+h)-f(x)-Ah=r_A(h),
$$

及

$$
f(x+h)-f(x)-Bh=r_B(h),
$$

且

$$
\frac{\|r_A(h)\|_2}{\|h\|_2}\to0,
\qquad
\frac{\|r_B(h)\|_2}{\|h\|_2}\to0.
$$

兩式相減得到

$$
(A-B)h=r_B(h)-r_A(h).
$$

任取 $v\in\mathbb R^n$。若 $v=0$，顯然 $(A-B)v=0$。若 $v\ne0$，令 $h=tv$，其中 $t\ne0$ 且 $t\to0$。由線性性，

$$
(A-B)(tv)=t(A-B)v.
$$

因此

$$
\|(A-B)v\|_2
=
\frac{\|(A-B)(tv)\|_2}{|t|}
\le
\frac{\|r_A(tv)\|_2+\|r_B(tv)\|_2}{|t|}.
$$

又因 $\|tv\|_2=|t|\|v\|_2$，

$$
\frac{\|r_A(tv)\|_2}{|t|}
=
\|v\|_2\frac{\|r_A(tv)\|_2}{\|tv\|_2}\to0,
$$

而 $r_B$ 亦同。令 $t\to0$，可得 $\|(A-B)v\|_2=0$，故 $(A-B)v=0$。由於 $v$ 任意，$A-B$ 為零映射，所以 $A=B$。證畢。

這個證明使用了「所有足夠小 $h$」的餘項條件。它不是由有限個數值擾動推得。

### 3. 可微必連續

**命題 7.3（必要條件）**  
若 $f$ 在 $x$ Fréchet 可微，則 $f$ 在 $x$ 連續。

**證明。**  
設 $A=Df(x)$。由定義，

$$
f(x+h)-f(x)=Ah+r(h).
$$

使用算子範數 $\|A\|_{\mathrm{op}}=\sup_{\|v\|_2=1}\|Av\|_2$，

$$
\|f(x+h)-f(x)\|_2
\le
\|A\|_{\mathrm{op}}\|h\|_2+\|r(h)\|_2.
$$

因 $\|r(h)\|_2/\|h\|_2\to0$，所以 $\|r(h)\|_2\to0$；第一項也隨 $h\to0$ 而趨零。因此 $f(x+h)\to f(x)$。證畢。

連續是可微的必要條件，但不是充分條件。例如 $f(t)=|t|$ 在零點連續，卻不可微。

### 4. 與方向導數的關係

若 $f$ 在 $x$ Fréchet 可微，對任意固定方向 $v$，

$$
f(x+tv)-f(x)=tDf(x)[v]+r(tv).
$$

除以 $t\ne0$：

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

故方向導數存在，且

$$
D_vf(x)=Df(x)[v].
$$

所以 Fréchet 可微是所有方向導數存在的一個**充分條件**，而且方向 $v$ 到導數值的映射必須是線性的。反向敘述一般不成立：所有方向導數存在，仍未必 Fréchet 可微。

### 5. 一個常用充分條件

**定理 7.4（連續偏導給出可微性）**  
設 $U\subset\mathbb R^n$ 開，$f:U\to\mathbb R^m$。若 $f$ 的所有一階偏導數在 $x$ 的某個開鄰域存在且於 $x$ 連續，則 $f$ 在 $x$ Fréchet 可微，且其矩陣表示為 $J_f(x)$。

此定理是充分條件，不是必要條件；函數可能在某點可微，但導數在該點附近不連續。本章不以有限差分觀察到「看似連續」來代替定理條件。

其證明可按輸入座標逐一改變，對每個輸出分量套用單變量中值定理，再以偏導在 $x$ 的連續性控制各段誤差。關鍵是得到對所有小 $h$ 有效的統一界，而非只處理固定方向。

---

## 逐步手算例題

### 例題一：多項式映射的餘項

令

$$
f(x,y)=
\begin{bmatrix}
x^2+xy\\
\sin y
\end{bmatrix}.
$$

求 $f$ 在 $(1,0)$ 的 Fréchet 導數，並直接驗證餘項條件。

令 $h=(a,b)^T$。先算

$$
f(1,0)=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$

Jacobian 為

$$
J_f(x,y)=
\begin{bmatrix}
2x+y & x\\
0 & \cos y
\end{bmatrix},
$$

故

$$
J_f(1,0)=
\begin{bmatrix}
2&1\\
0&1
\end{bmatrix}.
$$

線性主部是

$$
J_f(1,0)h=
\begin{bmatrix}
2a+b\\
b
\end{bmatrix}.
$$

另一方面，

$$
f(1+a,b)=
\begin{bmatrix}
(1+a)^2+(1+a)b\\
\sin b
\end{bmatrix}
=
\begin{bmatrix}
1+2a+a^2+b+ab\\
\sin b
\end{bmatrix}.
$$

所以餘項為

$$
r(h)=
\begin{bmatrix}
a^2+ab\\
\sin b-b
\end{bmatrix}.
$$

設 $\rho=\sqrt{a^2+b^2}$。因 $|a|\le\rho$、$|b|\le\rho$，

$$
|a^2+ab|\le |a|^2+|a||b|\le2\rho^2.
$$

在 $b$ 接近零時，由單變量 Taylor 餘項或積分界可得

$$
|\sin b-b|\le\frac{|b|^3}{6}\le\frac{\rho^3}{6}.
$$

因此

$$
\|r(h)\|_2
\le
2\rho^2+\frac{\rho^3}{6}.
$$

除以 $\|h\|_2=\rho$：

$$
\frac{\|r(h)\|_2}{\|h\|_2}
\le
2\rho+\frac{\rho^2}{6}\to0.
$$

故 $f$ 在 $(1,0)$ Fréchet 可微，導數即上述矩陣。這是全方向的解析界，不只是沿座標軸檢查。

### 例題二：不可微函數與方向殘差

定義

$$
g(x,y)=\sqrt{x^2+y^2}.
$$

考察原點。若 $g$ 在原點可微，設導數為線性泛函 $L$。沿 $e_1=(1,0)^T$ 的正方向，

$$
\lim_{t\downarrow0}\frac{g(te_1)-g(0)}{t}=1,
$$

故若使用雙側方向導數所需的線性候選，將遇到立即矛盾：沿 $t<0$，

$$
\frac{g(te_1)}{t}=\frac{|t|}{t}=-1.
$$

因此連方向 $e_1$ 的雙側方向導數都不存在，故不可能 Fréchet 可微。

也可從任何候選線性映射直接看出。若取 $L=0$，則

$$
\frac{|g(h)-g(0)-Lh|}{\|h\|_2}
=
\frac{\|h\|_2}{\|h\|_2}=1,
$$

不趨近零。若取其他 $L$，也無法同時符合 $h$ 與 $-h$，因為 $g(h)=g(-h)$，而線性映射滿足 $L(-h)=-L(h)$。

### 例題三：所有方向導數存在仍不夠

定義

$$
q(x,y)=
\begin{cases}
\dfrac{x^3}{x^2+y^2},&(x,y)\ne(0,0),\\
0,&(x,y)=(0,0).
\end{cases}
$$

對固定方向 $v=(a,b)^T$，

$$
\frac{q(ta,tb)-q(0,0)}{t}
=
\frac{t^3a^3}{t^2(a^2+b^2)}\frac1t
=
\frac{a^3}{a^2+b^2}
$$

（$v\ne0$）。所以每個方向導數都存在，但方向導數映射

$$
v\longmapsto\frac{a^3}{a^2+b^2}
$$

不是線性的。例如它在 $e_1$ 上為 $1$、在 $e_2$ 上為 $0$，若是線性泛函，則在 $e_1+e_2$ 上應為 $1$，實際卻是 $1/2$。因此 $q$ 在原點不可能 Fréchet 可微。

---

## 實作與程式

以下程式只需 Python 與 NumPy，在 CPU 上建立局部擾動資料，以最小平方擬合矩陣，並比較解析 Jacobian 與方向殘差比。程式未在此處執行；後文列出的是依解析結果推得的預期現象。

擬合資料滿足

$$
\Delta F\approx H A^T,
$$

其中 `H` 每一橫列是一個輸入擾動 $h^T$，`dF` 每一橫列是一個輸出增量。由於 NumPy 的一維陣列 `(n,)` 不顯示列向量或橫向量語義，程式在矩陣運算處明確使用 `(n,1)`。

```python
import numpy as np

def f_smooth(x_col):
    x = float(x_col[0, 0])
    y = float(x_col[1, 0])
    return np.array([
        [x * x + x * y],
        [np.sin(y)]
    ], dtype=float)

def jacobian_smooth(x_col):
    x = float(x_col[0, 0])
    y = float(x_col[1, 0])
    return np.array([
        [2.0 * x + y, x],
        [0.0, np.cos(y)]
    ], dtype=float)

def f_nondiff(x_col):
    x = float(x_col[0, 0])
    y = float(x_col[1, 0])
    den = x * x + y * y
    if den == 0.0:
        return np.array([[0.0]], dtype=float)
    return np.array([[x ** 3 / den]], dtype=float)

def residual_ratio(fun, x0, A, h_col):
    h_norm = np.linalg.norm(h_col, ord=2)
    if h_norm == 0.0:
        raise ValueError("h 必須非零；餘項比在 h=0 不定義")
    r = fun(x0 + h_col) - fun(x0) - A @ h_col
    return float(np.linalg.norm(r, ord=2) / h_norm)

def fit_local_matrix(fun, x0, radius, directions):
    if radius <= 0.0:
        raise ValueError("radius 必須為正")
    H_rows = []
    dF_rows = []

    f0 = fun(x0)
    for v in directions:
        v_col = np.asarray(v, dtype=float).reshape(-1, 1)
        v_norm = np.linalg.norm(v_col)
        if v_norm == 0.0:
            raise ValueError("方向向量不可為零")
        h_col = radius * v_col / v_norm
        delta = fun(x0 + h_col) - f0
        H_rows.append(h_col[:, 0])
        dF_rows.append(delta[:, 0])

    H = np.vstack(H_rows)
    dF = np.vstack(dF_rows)

    # 解 H @ A.T ≈ dF
    A_transpose, residuals, rank, singular_values = np.linalg.lstsq(
        H, dF, rcond=None
    )
    A = A_transpose.T
    return A, rank, singular_values

def run_demo():
    x0 = np.array([[1.0], [0.0]])
    A_exact = jacobian_smooth(x0)

    directions = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [1.0, -2.0]
    ]

    print("解析 Jacobian:")
    print(A_exact)

    for radius in [1e-1, 1e-2, 1e-3, 1e-4]:
        A_fit, rank, singular_values = fit_local_matrix(
            f_smooth, x0, radius, directions
        )
        error = np.linalg.norm(A_fit - A_exact, ord=2)
        print(radius, rank, error, singular_values)

    test_direction = np.array([[1.0], [2.0]])
    test_direction /= np.linalg.norm(test_direction)

    print("光滑函數的殘差比:")
    for radius in [1e-1, 1e-2, 1e-3, 1e-4]:
        h = radius * test_direction
        print(radius, residual_ratio(f_smooth, x0, A_exact, h))

    origin = np.zeros((2, 1))
    zero_map = np.zeros((1, 2))
    print("非可微函數沿固定方向、以零映射為候選:")
    for radius in [1e-1, 1e-2, 1e-3, 1e-4]:
        h = radius * test_direction
        print(radius, residual_ratio(f_nondiff, origin, zero_map, h))

if __name__ == "__main__":
    run_demo()
```

對光滑函數而言，若餘項為二階，殘差比通常呈現約一階下降；但當步長太小時，浮點消去誤差可能破壞此趨勢。這種下降是與理論相容的數值證據，不是對極限的證明。

矩陣擬合還有一個重要前提：擾動方向必須張成輸入空間。若所有方向都共線，資料矩陣秩不足，某些輸入方向完全沒有被識別。即使訓練殘差很小，也不能宣稱已得到完整導數。

---

## 測試與預期結果

以下測試均為設計與預期，未宣稱已實際執行。

### 正常測試：光滑映射

在 $x_0=(1,0)^T$ 使用六個非共線方向，擬合矩陣應逐步接近

$$
\begin{bmatrix}
2&1\\
0&1
\end{bmatrix}.
$$

資料矩陣的秩預期為 $2$。對固定方向計算殘差比，當半徑由 $10^{-1}$ 降至 $10^{-4}$ 時，理想精度範圍內應大致下降。不能要求每一步都嚴格按十倍下降，因為函數不同分量的高階項與浮點誤差皆會影響結果。

### 邊界測試：極小步長

若半徑下降到接近機器精度，`f(x0 + h) - f(x0)` 會發生相消。此時擬合誤差或殘差比可能停止下降，甚至上升。這不推翻解析可微性，只表示該數值程序已超出可靠尺度。

另一方面，`residual_ratio` 明確拒絕 $h=0$，因為分母為零。極限討論的是非零 $h$ 趨近零，不是在 $h=0$ 直接代值。

### 故障測試：秩不足方向

若把 `directions` 改為

```python
directions = [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]]
```

則預期回報 `rank == 1`。所得矩陣第二個輸入方向的係數不具可辨識性。`np.linalg.lstsq` 仍可能傳回一個數值答案，但不應把它解讀成完整 Jacobian。

### 故障測試：非可微映射

對 `f_nondiff` 在原點使用零映射候選，若方向為 $v=(a,b)^T$ 且 $\|v\|_2=1$，則

$$
q(tv)=t\frac{a^3}{a^2+b^2}=ta^3,
$$

所以殘差比為 $|a|^3$，與半徑無關。預期輸出不會趨近零。即使改用某個擬合矩陣，也只能迎合有限方向，不能修復方向導數映射的非線性。

---

## 反例與常見陷阱

### 1. 把 $r(h)\to0$ 誤當成可微

正確條件是

$$
r(h)=o(\|h\|),
$$

不是只有 $r(h)\to0$。例如 $f(t)=|t|$ 且候選導數取零，餘項 $|t|\to0$，但

$$
\frac{|t|}{|t|}=1.
$$

### 2. 只檢查座標軸

沿 $x$ 軸與 $y$ 軸的行為不能控制所有曲線或方向。即使所有固定方向導數存在，也可能如例題三那樣不形成線性映射。

### 3. 把方向相依的線性近似拼在一起

Fréchet 導數必須是一個固定線性映射 $A$。不能沿方向 $v_1$ 選 $A_1$、沿方向 $v_2$ 選 $A_2$，再宣稱函數可微。

### 4. 把 Jacobian 候選當成可微性證明

偏導存在時可以排成矩陣，但這只提供候選者。若偏導在鄰域連續，可引用定理得到可微性；若沒有該條件，就必須另外證明餘項比趨零。

### 5. 有限採樣冒充全方向證明

一萬個方向仍是有限個方向；一百個步長仍沒有涵蓋任意小的擾動。數值實驗可：

- 發現某條路徑上的反例；
- 檢查解析公式是否可能寫錯；
- 顯示特定尺度上的近似品質。

但它不能單獨證明極限對全部擾動成立。

### 6. 混合單位直接取 Euclidean 範數

若 $h=(\Delta T,\Delta S)^T$，其中一項以攝氏度計、另一項以 PSU 計，則 $\|h\|_2$ 的物理解釋依賴尺度。分析上可以選定範數，但應說明尺度。較穩健的方法是先取參考尺度 $s_T,s_S>0$，定義無因次擾動

$$
\hat h=
\begin{bmatrix}
\Delta T/s_T\\
\Delta S/s_S
\end{bmatrix}.
$$

輸出也可按量測尺度無因次化，再評估殘差比。

---

## AI、幾何與養殖案例

### 1. AI 局部敏感度

設模型在某層附近的映射為 $f:\mathbb R^n\to\mathbb R^m$。矩陣 $Df(x)$ 描述輸入小擾動如何一階傳到輸出。局部矩陣可由自動微分或解析式取得，也可由有限差分近似，但三者證據層級不同：

- 解析餘項界：可構成可微性證明。
- 自動微分：依賴程式圖中的運算規則與分支語義，通常計算的是所實作函數的導數。
- 局部擬合：只是在有限尺度與採樣方向下的估計。

若模型含 `abs`、最大值或離散分支，在切換面上可能不可微。某套軟體傳回一個數值，不表示數學上的 Fréchet 導數存在。

### 2. 幾何形變

平面形變

$$
F(x,y)=
\begin{bmatrix}
x+0.1xy\\
y+0.05x^2
\end{bmatrix}
$$

在點 $(x,y)$ 的局部變化由

$$
DF(x,y)=
\begin{bmatrix}
1+0.1y&0.1x\\
0.1x&1
\end{bmatrix}
$$

描述。短線段擾動 $h$ 經一階近似變成 $DF(x,y)h$。這是局部切向量的變換，不表示有限大小圖形完全由同一矩陣變換；有限形變仍含餘項。

### 3. 合成養殖感測校準

考慮純合成感測模型，輸入為水溫 $T$（攝氏度）與鹽度 $S$（PSU），輸出為電壓 $V$（伏特）與相位 $\phi$（弧度）：

$$
f(T,S)=
\begin{bmatrix}
0.40+0.012T+0.003S+10^{-4}TS\\
0.10+0.002T^2-0.004S
\end{bmatrix}.
$$

其 Jacobian 為

$$
J_f(T,S)=
\begin{bmatrix}
0.012+10^{-4}S & 0.003+10^{-4}T\\
0.004T & -0.004
\end{bmatrix}.
$$

各元素單位分別是輸出單位除以輸入單位。例如第一橫列第一縱行是伏特／攝氏度，第二橫列第二縱行是弧度／PSU。不同單位的矩陣元素不能只靠數值大小直接比較物理重要性。

取參考尺度 $s_T=5\,^\circ\mathrm C$、$s_S=10\,\mathrm{PSU}$，並取輸出尺度 $s_V=0.1\,\mathrm V$、$s_\phi=0.1\,\mathrm{rad}$。若

$$
S_x=\operatorname{diag}(s_T,s_S),\qquad
S_y=\operatorname{diag}(s_V,s_\phi),
$$

則無因次 Jacobian 為

$$
\widehat J=S_y^{-1}J_fS_x.
$$

這時才適合用無權重 Euclidean 範數比較不同通道的相對敏感度。尺度選擇仍是建模決策，必須記錄。

局部校準矩陣只能描述工作點附近的小擾動。它不等於現場驗證，也不保證模型在極端溫鹽條件下有效。唯讀分析代理可以整理輸入範圍、單位、殘差比及失效警告，但不應據此直接控制投餌、加藥或其他設備。

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

求 $Df(0,0)$，並證明 $f$ 在原點 Fréchet 可微。

### 習題二：程式修改

修改本章程式，使其同時回報每個半徑下：

1. 擬合矩陣；
2. 擬合矩陣與解析 Jacobian 的算子範數誤差；
3. 最大方向殘差比；
4. 資料矩陣的秩。

說明哪個輸出可偵測方向集合秩不足，以及為何數值下降不能單獨證明可微。

### 習題三：反例

定義

$$
f(x,y)=
\begin{cases}
\dfrac{x^2y}{x^4+y^2},&(x,y)\ne(0,0),\\
0,&(x,y)=(0,0).
\end{cases}
$$

證明沿每條直線 $y=kx$ 的函數值都趨近零，但 $f$ 在原點不連續，因而不可 Fréchet 微分。

### 習題四：整合與單位

某合成校準模型為

$$
z(T,C)=
\begin{bmatrix}
2T+0.5C+0.01TC\\
0.1T^2-C
\end{bmatrix},
$$

其中 $T$ 的單位為攝氏度，$C$ 的單位為毫克／公升；第一輸出單位為毫伏，第二輸出單位為無因次指標。

1. 求 $J_z(20,4)$。
2. 對擾動 $\Delta T=0.1$、$\Delta C=-0.2$ 求一階輸出預測。
3. 求精確輸出增量與餘項。
4. 說明為何不能直接以原始輸入 Euclidean 長度宣稱「溫度與濃度同等加權」。

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

因此

$$
Df(0,0)=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix}.
$$

令 $h=(a,b)^T$。線性主部為 $(a,0)^T$，餘項為

$$
r(a,b)=
\begin{bmatrix}
e^a\cos b-1-a\\
ab+b^2
\end{bmatrix}.
$$

在原點附近，$e^a$、$\sin b$ 與 $\cos b$ 的單變量 Taylor 界給出某個常數 $C>0$，使

$$
|e^a\cos b-1-a|\le C(a^2+b^2).
$$

另外，

$$
|ab+b^2|\le |a||b|+b^2
\le\frac{a^2+b^2}{2}+b^2
\le\frac32(a^2+b^2).
$$

令 $\rho=\sqrt{a^2+b^2}$，可得

$$
\|r(a,b)\|_2\le C'\rho^2
$$

其中 $C'$ 為某常數。因此

$$
\frac{\|r(a,b)\|_2}{\rho}\le C'\rho\to0.
$$

故 $f$ 在原點 Fréchet 可微。

### 習題二解答

可在每個半徑內加入：

```python
ratios = []
for v in directions:
    v_col = np.asarray(v, dtype=float).reshape(-1, 1)
    v_col /= np.linalg.norm(v_col)
    h_col = radius * v_col
    ratios.append(residual_ratio(f_smooth, x0, A_exact, h_col))

print("A_fit =", A_fit)
print("operator error =", np.linalg.norm(A_fit - A_exact, ord=2))
print("max ratio =", max(ratios))
print("rank =", rank)
```

`rank` 可偵測採樣方向是否張成整個輸入空間。若輸入維度為二而 `rank < 2`，便無法由資料辨識完整矩陣。

殘差比下降只涵蓋程式列出的有限方向與有限半徑。Fréchet 可微要求對所有充分小且非零的 $h$ 成立，所以程式只能提供相容證據或找出反例，不能單獨完成證明。

### 習題三解答

沿直線 $y=kx$，當 $x\ne0$ 時，

$$
f(x,kx)
=
\frac{x^2(kx)}{x^4+k^2x^2}
=
\frac{kx}{x^2+k^2}.
$$

若 $k\ne0$，當 $x\to0$ 時此式趨近零；若 $k=0$，函數恆為零。因此所有固定直線路徑都給出零極限。

但沿拋物線 $y=x^2$，

$$
f(x,x^2)
=
\frac{x^2x^2}{x^4+x^4}
=
\frac12
$$

對所有 $x\ne0$ 成立。它不趨近 $f(0,0)=0$，所以函數在原點不連續。由命題 7.3，可微必連續，因此此函數在原點不可 Fréchet 微分。

這也說明所有直線測試一致仍不足以證明多變量極限存在。

### 習題四解答

Jacobian 為

$$
J_z(T,C)=
\begin{bmatrix}
2+0.01C&0.5+0.01T\\
0.2T&-1
\end{bmatrix}.
$$

在 $(20,4)$，

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

一階預測為

$$
J_z(20,4)h
=
\begin{bmatrix}
2.04(0.1)+0.70(-0.2)\\
4(0.1)-(-0.2)
\end{bmatrix}
=
\begin{bmatrix}
0.064\\
0.6
\end{bmatrix}.
$$

精確計算第一分量的非線性部分只有 $0.01TC$。增量展開為

$$
0.01(T+\Delta T)(C+\Delta C)-0.01TC
=
0.01(C\Delta T+T\Delta C+\Delta T\Delta C).
$$

前兩項已包含在線性主部，故第一分量餘項是

$$
0.01(0.1)(-0.2)=-0.0002.
$$

第二分量的非線性部分是 $0.1T^2$，餘項為

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

精確輸出增量為

$$
\begin{bmatrix}
0.0638\\
0.601
\end{bmatrix}.
$$

由於 $0.1$ 攝氏度與 $0.2$ 毫克／公升屬不同物理量，原始 Euclidean 長度

$$
\sqrt{(0.1)^2+(-0.2)^2}
$$

依賴所選單位；若濃度改用微克／公升，數值會改變千倍。應先指定有物理意義的參考尺度，例如溫度尺度 $s_T$ 與濃度尺度 $s_C$，再使用 $(\Delta T/s_T,\Delta C/s_C)^T$ 計算無因次長度。

---

## 本章小結

Fréchet 導數把多變量導數定義為一個固定線性映射：

$$
f(x+h)=f(x)+Df(x)[h]+r(h),
\qquad
\frac{\|r(h)\|}{\|h\|}\to0.
$$

核心不只是寫出偏導矩陣，而是證明線性化後的剩餘誤差比輸入擾動更高階。導數若存在便唯一，且可微必連續；Fréchet 可微也保證所有方向導數存在並由同一線性映射給出。然而，偏導存在、有限方向測試通過，甚至所有方向導數存在，都不保證 Fréchet 可微。

數值矩陣擬合與殘差比掃描適合核對公式、辨識尺度範圍與發現失敗方向。其結論受採樣方向、資料秩、步長及浮點精度限制，不能代替對無限量詞的分析證明。在感測與校準問題中，還必須記錄輸入輸出單位、有效工作點、無因次尺度及餘項適用範圍。

---

## 參考來源

1. Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>  
   可供極限、連續性、導數與嚴格證明方法的延伸閱讀。

2. MIT OpenCourseWare，*18.100A Real Analysis*：  
   <https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>  
   可參考實分析中的量詞、極限與證明架構。

3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：  
   <https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>  
   可參考多變量線性近似與 Jacobian 的計算背景。

以上來源作延伸參考；本章的有限數值實驗不構成定理證明，且本文未宣稱已執行所列程式。