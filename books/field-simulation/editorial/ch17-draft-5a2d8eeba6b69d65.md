# 第17章 二維有限元素與能量觀點

## 學習目標與先備知識

本章把一維有限元素推廣到二維三角形網格，並以 Poisson 型問題為主線，建立「弱形式—局部元素—全域組裝—邊界處理—能量最小化」的完整鏈條。完成本章後，讀者應能：

1. 從二維橢圓方程推導弱形式，分辨本質邊界與自然邊界。
2. 在任意非退化三角形上建立 P1 線性基底。
3. 使用仿射 Jacobian 計算元素面積、基底梯度、剛度矩陣與負載列向量。
4. 將局部矩陣組裝成全域系統，正確消去非齊次 Dirichlet 自由度。
5. 解釋有限元素解何時等價於離散能量的唯一極小值。
6. 檢查翻轉繞序、退化元素、純 Neumann 零空間、能量梯度與線性殘差。
7. 分開判斷數值穩定、守恆、能量下降、非負性與物理可信度。

先備知識包括偏導數、梯度與散度、分部積分、對稱正定矩陣、線性方程組，以及一維 P1 有限元素。以下節點式網格不同於前述 cell-centered 網格：每個未知量位於網格節點，節點編號由元素連接表指定，不使用 $k=jN_x+i$ 的 cell 展平規則。

---

## 問題與直覺

考慮有界多邊形區域 $\Omega\subset\mathbb{R}^2$：

$$
-\nabla\cdot\left(k\nabla u\right)+cu=f
\quad\text{於 }\Omega,
$$

邊界分成互不重疊的 Dirichlet 部分 $\Gamma_D$ 與 Neumann 部分 $\Gamma_N$：

$$
u=g_D\quad\text{於 }\Gamma_D,
$$

$$
(k\nabla u)\cdot n=q_N\quad\text{於 }\Gamma_N.
$$

這裡 $n$ 是外法向量。若 $u$ 是溫度，則 $k$ 可代表導熱係數；但物理熱通量通常定義為

$$
j=-k\nabla u.
$$

因此 $(k\nabla u)\cdot n=q_N$ 等價於外向物理通量 $j\cdot n=-q_N$。若不先說明符號，Neumann 資料很容易輸入相反方向。

有限元素的核心直覺不是直接逼迫微分方程在每個節點成立，而是要求殘差對所有允許的測試函數平均為零。三角 P1 元素在每個三角形內以平面近似 $u$；函數本身跨元素連續，但梯度通常跨元素跳躍。這符合 $H^1$ 弱解只要求平方可積的一階弱導數，而不要求經典二階導數處處存在。

### 單位與模型範圍

若 $x,y$ 以 m 計，$u$ 為 K，而方程描述每單位厚度的穩態導熱，則：

- $\nabla u$：K/m；
- $k$：W/(m·K)；
- $k\nabla u$：W/m²；
- $f$：W/m³；
- 二維積分可解釋為厚度 $1$ m 的三維體積積分。

若二維域代表深度平均池域，係數與來源必須重新由深度積分推導，不能把純幾何二維模型直接宣稱為真實三維池體。

模型誤差與離散誤差也須分開：即使有限元素線性系統解得極精確，錯誤的係數、邊界資料或二維化假設仍會產生物理上錯誤的答案。

---

## 數學與物理推導

### 1. 弱形式

令測試函數 $v$ 在 $\Gamma_D$ 上為零。將方程乘以 $v$ 並積分：

$$
\int_\Omega
\left[-\nabla\cdot(k\nabla u)+cu\right]v\,d\Omega
=
\int_\Omega fv\,d\Omega.
$$

使用散度定理：

$$
-\int_\Omega \nabla\cdot(k\nabla u)v\,d\Omega
=
\int_\Omega k\nabla u\cdot\nabla v\,d\Omega
-\int_{\partial\Omega}(k\nabla u\cdot n)v\,d\Gamma.
$$

由於 $v=0$ 於 $\Gamma_D$，得到：

$$
a(u,v)=\ell(v),
$$

其中

$$
a(u,v)
=
\int_\Omega k\nabla u\cdot\nabla v\,d\Omega
+
\int_\Omega cuv\,d\Omega,
$$

$$
\ell(v)
=
\int_\Omega fv\,d\Omega
+
\int_{\Gamma_N}q_Nv\,d\Gamma.
$$

試探空間與測試空間分別為

$$
V_g=\{w\in H^1(\Omega):w=g_D\text{ 於 }\Gamma_D\},
$$

$$
V_0=\{v\in H^1(\Omega):v=0\text{ 於 }\Gamma_D\}.
$$

問題是：求 $u\in V_g$，使得對所有 $v\in V_0$ 都有 $a(u,v)=\ell(v)$。

Dirichlet 條件限制了解所在的空間，故稱本質邊界條件；Neumann 條件在分部積分後自然出現在右端，故稱自然邊界條件。

### 2. 三角 P1 基底與 Jacobian

參考三角形取

$$
\hat T=\{(\xi,\eta):\xi\ge 0,\ \eta\ge0,\ \xi+\eta\le1\},
$$

其頂點為 $(0,0),(1,0),(0,1)$，P1 基底為

$$
\hat N_1=1-\xi-\eta,\qquad
\hat N_2=\xi,\qquad
\hat N_3=\eta.
$$

實體三角形頂點為 $x_1,x_2,x_3$，仿射映射是

$$
x(\xi,\eta)=x_1+J
\begin{bmatrix}
\xi\\
\eta
\end{bmatrix},
$$

$$
J=
\begin{bmatrix}
x_2-x_1 & x_3-x_1\\
y_2-y_1 & y_3-y_1
\end{bmatrix}.
$$

有向面積與實際面積分別為

$$
A_s=\frac12\det J,\qquad
A=\frac12|\det J|.
$$

若 $\det J>0$，頂點依右手座標系中的逆時針方向排列；若 $\det J<0$，則為順時針。物理積分必須使用 $|\det J|$，不能因順時針繞序而得到負面積或負剛度。

鏈式法則給出

$$
\nabla_x N_a=J^{-T}\nabla_{\xi}\hat N_a.
$$

由於映射是仿射的，P1 基底梯度在每個元素中為常數。也可直接寫成

$$
\nabla N_1
=
\frac{1}{2A_s}
\begin{bmatrix}
y_2-y_3\\
x_3-x_2
\end{bmatrix},
$$

$$
\nabla N_2
=
\frac{1}{2A_s}
\begin{bmatrix}
y_3-y_1\\
x_1-x_3
\end{bmatrix},
\qquad
\nabla N_3
=
\frac{1}{2A_s}
\begin{bmatrix}
y_1-y_2\\
x_2-x_1
\end{bmatrix}.
$$

分母保留有向面積。翻轉繞序時，局部節點排列和梯度會一起改變；重新映射到相同全域自由度後，物理結果不變。

### 3. 元素矩陣

在元素 $T_e$ 內令

$$
u_h=\sum_{a=1}^3U_aN_a.
$$

若 $k$ 在元素內為常數，擴散剛度矩陣為

$$
K^{(e)}_{ab}
=
\int_{T_e}k\nabla N_a\cdot\nabla N_b\,d\Omega
=
kA\,\nabla N_a\cdot\nabla N_b.
$$

定義

$$
B=
\begin{bmatrix}
\partial_xN_1&\partial_xN_2&\partial_xN_3\\
\partial_yN_1&\partial_yN_2&\partial_yN_3
\end{bmatrix},
$$

則

$$
K^{(e)}_{\mathrm{diff}}=kA B^TB.
$$

若 $c$ 為元素常數，反應質量矩陣為

$$
K^{(e)}_{\mathrm{react}}
=
\frac{cA}{12}
\begin{bmatrix}
2&1&1\\
1&2&1\\
1&1&2
\end{bmatrix}.
$$

若 $f$ 為元素常數，

$$
F^{(e)}
=
\frac{fA}{3}
\begin{bmatrix}
1\\1\\1
\end{bmatrix}.
$$

對變係數或非多項式來源，需採用適當三角形積分公式。單一重心點可精確積分一次多項式，但不能精確處理一般的 $k(x,y)\nabla N_a\cdot\nabla N_b$ 或高階來源。

### 4. 全域組裝

設元素 $e$ 的局部節點對應全域編號 $(I_1,I_2,I_3)$，則

$$
K_{I_aI_b}\mathrel{+}=K^{(e)}_{ab},
\qquad
F_{I_a}\mathrel{+}=F^{(e)}_a.
$$

相鄰三角形共享節點，因此各元素對共享自由度的貢獻相加。這不是有限體積逐面通量的成對抵消，但對常數測試函數仍可導出相應的全域收支關係。

若自由度分成自由集合 $F$ 與 Dirichlet 集合 $D$：

$$
\begin{bmatrix}
K_{FF}&K_{FD}\\
K_{DF}&K_{DD}
\end{bmatrix}
\begin{bmatrix}
U_F\\U_D
\end{bmatrix}
=
\begin{bmatrix}
b_F\\b_D
\end{bmatrix},
$$

且 $U_D=g_D$，則應解

$$
K_{FF}U_F=b_F-K_{FD}g_D.
$$

只把 Dirichlet 對角線改成一而不先修正自由列右端，會使非齊次邊界答案錯誤。

### 5. 離散能量

對稱問題對應能量泛函

$$
\Pi(u)
=
\frac12a(u,u)-\ell(u).
$$

在有限元素空間中，

$$
\Pi_h(U)
=
\frac12U^TKU-b^TU.
$$

對自由變數微分：

$$
\nabla_{U_F}\Pi_h
=
K_{FF}U_F+K_{FD}g_D-b_F.
$$

因此能量駐點正是有限元素線性系統。若 $K_{FF}$ 對稱正定，則 Hessian 為

$$
\nabla^2_{U_F}\Pi_h=K_{FF},
$$

能量嚴格凸，駐點是唯一全域極小值。

常見充分條件是：

- $k(x,y)\ge k_{\min}>0$；
- $c(x,y)\ge0$；
- $\Gamma_D$ 具有正邊界測度。

若是純 Neumann 且 $c=0$，常數函數的梯度為零，所以 $K\mathbf1=0$。此時須滿足相容性：

$$
\int_\Omega f\,d\Omega+\int_{\partial\Omega}q_N\,d\Gamma=0.
$$

離散形式是 $\mathbf1^Tb=0$。解只確定到任意常數，必須加平均值約束或在零均值子空間求解；任意固定一點雖可移除代數奇異性，卻改變了規範，且不能修復不相容右端。

---

## 逐步手算例題

### 例一：單一直角三角形

考慮頂點

$$
x_1=(0,0),\quad x_2=(2,0),\quad x_3=(0,1).
$$

取 $k=3$、$c=0$、$f=6$。

**步驟一：Jacobian 與面積**

$$
J=
\begin{bmatrix}
2&0\\
0&1
\end{bmatrix},
\qquad
\det J=2,
\qquad
A=1.
$$

**步驟二：基底梯度**

$$
\nabla N_1=
\begin{bmatrix}-1/2\\-1\end{bmatrix},
\quad
\nabla N_2=
\begin{bmatrix}1/2\\0\end{bmatrix},
\quad
\nabla N_3=
\begin{bmatrix}0\\1\end{bmatrix}.
$$

三者和為零，故常數場梯度為零。

**步驟三：剛度矩陣**

先計算內積矩陣：

$$
B^TB=
\begin{bmatrix}
5/4&-1/4&-1\\
-1/4&1/4&0\\
-1&0&1
\end{bmatrix}.
$$

因此

$$
K^{(e)}
=
3
\begin{bmatrix}
5/4&-1/4&-1\\
-1/4&1/4&0\\
-1&0&1
\end{bmatrix}.
$$

每一橫列的和為零，故 $K^{(e)}\mathbf1=0$。

**步驟四：負載**

$$
F^{(e)}
=
\frac{6(1)}{3}
\begin{bmatrix}1\\1\\1\end{bmatrix}
=
\begin{bmatrix}2\\2\\2\end{bmatrix}.
$$

若三個節點全是 Dirichlet，自由度為零，無需求解；有限元素內部解只是指定頂點值的線性插值。這提醒我們：元素方程不能脫離全域邊界與鄰接關係獨立解讀。

### 例二：單位正方形的中心節點網格

取五個節點：

$$
(0,0),(1,0),(1,1),(0,1),(1/2,1/2),
$$

將中心依序連到四條邊，形成四個面積皆為 $1/4$ 的三角形。求解

$$
-\Delta u=1,\qquad u=0\text{ 於正方形邊界}.
$$

四個角點皆固定為零，唯一自由度是中心值 $U_c$。

對任一三角形，中心基底從中心值 $1$ 線性降至對邊值 $0$。中心到對邊的距離是 $1/2$，故

$$
|\nabla N_c|=2.
$$

單一元素對中心對角項的貢獻為

$$
K^{(e)}_{cc}=A|\nabla N_c|^2
=\frac14(4)=1.
$$

四個元素合計：

$$
K_{cc}=4.
$$

常數來源 $f=1$ 對單一元素中心負載是

$$
F^{(e)}_c=\frac{A}{3}=\frac1{12},
$$

故

$$
F_c=4\left(\frac1{12}\right)=\frac13.
$$

因此

$$
4U_c=\frac13,
\qquad
U_c=\frac1{12}.
$$

離散能量只剩一個變數：

$$
\Pi_h(U_c)
=
\frac12(4)U_c^2-\frac13U_c.
$$

其導數為

$$
\Pi_h'(U_c)=4U_c-\frac13,
$$

在 $U_c=1/12$ 為零，二階導數為 $4>0$，所以確為唯一極小值。

注意此解是該粗網格的離散解，不是解析解的宣稱，也不能由單一網格推定收斂階。

---

## 實作與程式

以下程式僅依賴 Python 3.10+ 與 NumPy，使用稠密矩陣以保持自足，適合教學小網格，不適合大型模型。程式採節點未知量，座標形狀為 `(nnode, 2)`，三角形連接表形狀為 `(nelem, 3)`。

```python
import numpy as np

def validate_inputs(points, triangles, kappa, reaction, source):
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points 必須具有形狀 (nnode, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles 必須具有形狀 (nelem, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("座標含 NaN 或無窮值")
    if not all(np.isfinite(v) for v in (kappa, reaction, source)):
        raise ValueError("係數或來源不是有限值")
    if kappa <= 0.0:
        raise ValueError("本例要求 kappa > 0")
    if reaction < 0.0:
        raise ValueError("本例要求 reaction >= 0")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("元素含越界節點編號")
    for tri in triangles:
        if len(set(map(int, tri))) != 3:
            raise ValueError("元素內有重複節點")
    return points, triangles

def triangle_matrices(xy, kappa, reaction, source,
                      area_tol=1.0e-14):
    # J 的兩個縱行分別是 x2-x1 與 x3-x1
    J = np.column_stack((xy[1] - xy[0], xy[2] - xy[0]))
    detJ = np.linalg.det(J)

    scale2 = max(1.0, np.max(np.abs(xy)) ** 2)
    if not np.isfinite(detJ) or abs(detJ) <= area_tol * scale2:
        raise ValueError("退化或近退化三角形")

    area = 0.5 * abs(detJ)
    grad_ref = np.array([[-1.0, 1.0, 0.0],
                         [-1.0, 0.0, 1.0]])
    B = np.linalg.solve(J.T, grad_ref)

    Ke_diff = kappa * area * (B.T @ B)
    Me = (area / 12.0) * np.array(
        [[2.0, 1.0, 1.0],
         [1.0, 2.0, 1.0],
         [1.0, 1.0, 2.0]]
    )
    Ke = Ke_diff + reaction * Me
    fe = source * area / 3.0 * np.ones(3)
    return Ke, fe, area, detJ, B

def assemble(points, triangles, kappa=1.0,
             reaction=0.0, source=1.0):
    points, triangles = validate_inputs(
        points, triangles, kappa, reaction, source
    )
    n = len(points)
    K = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    signed_dets = []

    for tri in triangles:
        xy = points[tri]
        Ke, fe, area, detJ, B = triangle_matrices(
            xy, kappa, reaction, source
        )
        signed_dets.append(detJ)
        for a, I in enumerate(tri):
            b[I] += fe[a]
            for c, Jidx in enumerate(tri):
                K[I, Jidx] += Ke[a, c]

    return K, b, np.asarray(signed_dets)

def solve_dirichlet(K, b, dirichlet):
    n = len(b)
    fixed = np.array(sorted(dirichlet.keys()), dtype=int)
    if np.any(fixed < 0) or np.any(fixed >= n):
        raise ValueError("Dirichlet 節點編號越界")

    values = np.array([dirichlet[int(i)] for i in fixed], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Dirichlet 值含 NaN 或無窮值")

    is_free = np.ones(n, dtype=bool)
    is_free[fixed] = False
    free = np.nonzero(is_free)[0]

    u = np.zeros(n, dtype=float)
    u[fixed] = values

    if len(free) > 0:
        rhs = b[free] - K[np.ix_(free, fixed)] @ values
        Kff = K[np.ix_(free, free)]
        u[free] = np.linalg.solve(Kff, rhs)

    residual_free = (
        K[np.ix_(free, np.arange(n))] @ u - b[free]
        if len(free) else np.zeros(0)
    )
    return u, free, residual_free

def energy(K, b, u):
    return 0.5 * u @ K @ u - b @ u

def energy_gradient_free(K, b, u, free):
    return K[np.ix_(free, np.arange(len(u)))] @ u - b[free]

def main():
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])

    triangles = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ])

    K, b, dets = assemble(
        points, triangles,
        kappa=1.0, reaction=0.0, source=1.0
    )

    boundary = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    u, free, rfree = solve_dirichlet(K, b, boundary)

    print("signed detJ =", dets)
    print("u =", u)
    print("free residual norm =", np.linalg.norm(rfree))
    print("energy =", energy(K, b, u))
    print("energy gradient =", energy_gradient_free(K, b, u, free))

    # 有意翻轉第一個元素的繞序
    triangles_flip = triangles.copy()
    triangles_flip[0] = triangles_flip[0, [0, 2, 1]]
    K2, b2, dets2 = assemble(points, triangles_flip)
    print("orientation K difference =", np.max(np.abs(K2 - K)))
    print("orientation b difference =", np.max(np.abs(b2 - b)))

if __name__ == "__main__":
    main()
```

程式沒有自動安裝套件、沒有使用 GPU，也沒有控制任何設備。大型問題應使用 CSR 稀疏矩陣與適當迭代法，但仍須保留同樣的幾何、邊界與殘差檢查。

---

## 測試與預期結果

以下是由推導得到的預期結果，並非宣稱已實際執行。

### 正常測試

1. **中心節點解**

   對上述單位正方形網格，預期：

   $$
   u=[0,0,0,0,1/12]^T.
   $$

   浮點輸出中的中心值應接近 `0.0833333333333`。

2. **真殘差**

   自由自由度的真殘差

   $$
   r_F=b_F-(KU)_F
   $$

   應接近機器精度。程式回傳的是 $(KU-b)_F$，只差一個負號，範數相同。

3. **能量梯度**

   `energy_gradient_free` 應接近零，因為它就是自由變數上的一階能量梯度。

4. **翻轉繞序**

   第一個三角形交換第二、第三節點後，其 $\det J$ 應改變符號，但組裝後的 $K$ 與 $b$ 應只差浮點捨入量。若差異顯著，通常是面積錯用 $\det J/2$ 而非 $|\det J|/2$，或梯度與局部編號沒有同步排列。

5. **常數場測試**

   當 `reaction=0` 時，擴散矩陣應滿足

   $$
   K\mathbf1\approx0.
   $$

   這是算子一致性檢查，不代表含 Dirichlet 邊界的常數場就是允許解。

### 邊界測試

若來源設為零，並把所有邊界節點設成同一常數 $g$，中心解也應為 $g$。此測試可揭露非齊次 Dirichlet 右端沒有扣除 $K_{FD}g_D$ 的錯誤。

### 故障測試

1. 三個共線點，例如 $(0,0),(1,0),(2,0)$，應拒絕並報告退化三角形。
2. 元素 `[0, 1, 1]` 應因重複節點而拒絕。
3. 座標含 `NaN`、係數為無窮值或 $k\le0$，應拒絕。
4. 純 Neumann、$c=0$ 且常數正來源的系統不相容；不得靠固定任意節點後假稱原問題已有解。
5. 極瘦三角形即使面積非零，也可能造成很差的條件數。僅通過面積門檻不等於網格品質良好。

### 五種性質不能混同

- **線性求解穩定或殘差小**：只描述代數求解。
- **守恆**：需由弱形式、邊界通量與積分量定義檢查。
- **能量下降**：本章是穩態最小化；若用迭代法，並非每種迭代都保證每一步能量下降。
- **非負性**：連續 Poisson 問題可能有最大值原理，但一般三角網格的 P1 解不自動非負。通常還需非鈍角網格、適當係數與矩陣 M-matrix 條件。
- **物理可信度**：還需確認模型、單位、係數、來源與邊界是否代表目標系統。

---

## 除錯與常見陷阱

### 1. 把有向面積直接當積分面積

順時針元素有 $\det J<0$。若以 $\det J/2$ 乘 $B^TB$，剛度可能變成負半定，能量觀點完全破壞。正確做法是梯度由 $J^{-T}$ 計算，積分測度使用 $|\det J|$。

### 2. 靜默修正元素繞序

自動翻轉所有順時針元素不是必要的；數學上兩種繞序皆可正確計算。若選擇統一繞序，必須記錄修改並同步重排局部資料。不能只交換座標、不交換材料或邊界標籤。

### 3. 漏掉非齊次 Dirichlet 貢獻

直接刪去固定自由度卻仍使用原始 $b_F$，等於忽略 $K_{FD}g_D$。齊次邊界測試無法揭露此錯誤，因此一定要增加常數非零邊界測試。

### 4. 退化與近退化元素

零面積元素使 $J^{-1}$ 不存在；極小角元素會放大基底梯度，造成條件數惡化。門檻應相對於幾何尺度，而非只用固定絕對面積。實務上還應記錄最小角、長寬比與 Jacobian 條件數。

### 5. 把線性殘差當作離散誤差

即使 $\|b-Ku\|$ 很小，也只能說線性系統解得準；不能說有限元素解接近連續解。後者需要網格細化、製造解或可證誤差估計。

### 6. 誤解共享邊上的梯度

P1 解跨共享邊連續，但法向梯度一般不連續。這不是自動代表錯誤；弱形式只要求整體積分關係。若物理問題要求局部守恆通量，可進一步做通量重建，或採有限體積、混合有限元素。

### 7. 由正定性誤推非負性

$K_{FF}$ 對稱正定只保證唯一極小值，不保證 $K_{FF}^{-1}$ 的元素非負。鈍角三角形可能產生正的非對角剛度項，破壞離散最大值原理。

---

## 養殖與相場案例

### 合成池域中的穩態擴散—反應場

設合成池域的溶質濃度為 $C$，單位 kg/m³，考慮深度平均穩態模型：

$$
-\nabla\cdot(D_{\mathrm{eff}}\nabla C)+\lambda C=S.
$$

其中 $D_{\mathrm{eff}}$ 的單位為 m²/s，$\lambda$ 為 1/s，$S$ 為 kg/(m³·s)。若二維積分以每單位厚度解釋，各項量綱一致。入口可設 Dirichlet 濃度，封閉池壁可設零 Neumann 通量。

這只是合成傳輸模型。真實池域中的垂向混合、平流、生化反應與設備作用未必可縮約成常係數反應項。有限元素圖形即使平滑，也不構成現場驗證；溶氧濃度跨過管理門檻更不是熱力學相變。

### 相場能量的空間離散橋接

對無因次序參量 $\phi$，自由能為

$$
F[\phi]
=
\int_\Omega
\left[
\frac{(\phi^2-1)^2}{4}
+\frac{\kappa}{2}|\nabla\phi|^2
\right]d\Omega.
$$

P1 有限元素可直接離散梯度能：

$$
F_{\mathrm{grad},h}
=
\frac{\kappa}{2}\Phi^TK\Phi.
$$

雙井項則需元素積分，不能簡單把節點上的 $W(\Phi_i)$ 無權重相加。其能量梯度包含非線性項與剛度項。這與本章線性 Poisson 能量的結構相通，但 Allen–Cahn 或 Cahn–Hilliard 還涉及時間離散、質量矩陣、非線性求解與邊界條件。

連續相場方程的能量耗散不保證任意時間步長、任意非線性容差下的離散能量下降。Cahn–Hilliard 的質量守恆、Allen–Cahn 通常不守恆，也不能由本章的穩態能量極小化直接推出。

---

## 習題

### 習題一：手算局部矩陣

對三角形 $(0,0),(1,0),(0,1)$，取 $k=2$、$c=0$、$f=3$。求面積、三個基底梯度、元素剛度矩陣與負載列向量，並驗證常數零模態。

### 習題二：程式與能量梯度

修改本章程式，把正方形四個邊界值全設為 $2$，來源設為零。預測中心值、自由殘差與能量梯度。再說明為何只檢查完整向量 $KU-b$ 會誤判 Dirichlet 節點。

### 習題三：反例與離散最大值原理

證明對一個三角形，兩個不同頂點 $i,j$ 的擴散剛度項可寫成

$$
K^{(e)}_{ij}
=
-\frac{k}{2}\cot\theta_k,
$$

其中 $\theta_k$ 是第三個頂點的內角。據此說明鈍角三角形為何可能破壞 M-matrix 所需的非正非對角結構。

### 習題四：整合純 Neumann 問題

考慮單位正方形上的

$$
-\Delta u=1,
\qquad
\nabla u\cdot n=0.
$$

判斷是否有解。若改成

$$
-\Delta u=x-\frac12,
\qquad
\nabla u\cdot n=0,
$$

說明相容性、零空間與一種合法的離散處理方式。比較「零均值約束」與「任意固定一點」。

### 習題五：翻轉與退化測試

設三角形連接表為 `[0,1,2]`。分別交換為 `[0,2,1]`，以及把第三點移到前兩點連線上。說明 $\det J$、面積、剛度矩陣及程式行為的預期變化。

---

## 習題解答

### 解答一

參考直角三角形面積為

$$
A=\frac12.
$$

基底為

$$
N_1=1-x-y,\qquad N_2=x,\qquad N_3=y,
$$

故

$$
\nabla N_1=
\begin{bmatrix}-1\\-1\end{bmatrix},
\quad
\nabla N_2=
\begin{bmatrix}1\\0\end{bmatrix},
\quad
\nabla N_3=
\begin{bmatrix}0\\1\end{bmatrix}.
$$

因此

$$
K^{(e)}
=
kA
\begin{bmatrix}
2&-1&-1\\
-1&1&0\\
-1&0&1
\end{bmatrix}.
$$

代入 $k=2$、$A=1/2$：

$$
K^{(e)}
=
\begin{bmatrix}
2&-1&-1\\
-1&1&0\\
-1&0&1
\end{bmatrix}.
$$

負載為

$$
F^{(e)}
=
\frac{fA}{3}\mathbf1
=
\frac12
\begin{bmatrix}1\\1\\1\end{bmatrix}.
$$

各橫列元素和為零，所以

$$
K^{(e)}\mathbf1=0.
$$

這反映常數函數的梯度為零。

### 解答二

當來源為零且所有邊界值皆為 $2$，常數函數 $u=2$ 的梯度為零，符合 Laplace 方程及邊界條件，因此中心值預期也是 $2$。

自由節點方程滿足

$$
(Ku-b)_F=0,
$$

所以自由殘差與自由能量梯度都應接近浮點捨入誤差。

在 Dirichlet 節點上，有限元素離散方程已被邊界值取代；原方程的該列殘差通常代表維持指定邊界值所需的反力或通量，不必為零。因此不能以完整 $\|Ku-b\|$ 作為消去式 Dirichlet 系統的停止判準；應檢查自由列真殘差。

### 解答三

P1 基底梯度垂直於其對邊。利用三角形幾何關係，可以導出對 $i\ne j$：

$$
A\nabla N_i\cdot\nabla N_j
=
-\frac12\cot\theta_k,
$$

其中 $k$ 是與 $i,j$ 不同的第三個頂點。因此

$$
K^{(e)}_{ij}
=
-\frac{k}{2}\cot\theta_k.
$$

若 $\theta_k<90^\circ$，則 $\cot\theta_k>0$，所以 $K^{(e)}_{ij}<0$。若 $\theta_k=90^\circ$，該項為零。若 $\theta_k>90^\circ$，則 $\cot\theta_k<0$，因而

$$
K^{(e)}_{ij}>0.
$$

M-matrix 型離散通常要求非對角項非正。鈍角元素可造成正非對角項，故即使矩陣仍對稱正定，也未必保證離散最大值原理或節點解非負。這是「正定性不等於單調性」的具體反例。

### 解答四

第一個問題要求

$$
\int_\Omega 1\,d\Omega
+
\int_{\partial\Omega}0\,d\Gamma=1\ne0.
$$

所以不滿足純 Neumann 相容性，沒有弱解。固定任意節點只會產生某個修改後代數問題的答案，不能使原偏微分方程變得相容。

第二個來源滿足

$$
\int_0^1\int_0^1
\left(x-\frac12\right)\,dy\,dx
=
\int_0^1\left(x-\frac12\right)dx=0.
$$

因此相容，但解只確定到常數。離散矩陣滿足 $K\mathbf1=0$，右端應滿足 $\mathbf1^Tb\approx0$。

合法處理之一是加入零均值約束。若 $m_i=\int_\Omega N_i\,d\Omega$，可解增廣系統

$$
\begin{bmatrix}
K&m\\
m^T&0
\end{bmatrix}
\begin{bmatrix}
U\\\lambda
\end{bmatrix}
=
\begin{bmatrix}
b\\0
\end{bmatrix}.
$$

約束 $m^TU=0$ 對應有限元素函數的積分均值為零。固定任意一點也可選定一個代表解，但它施加的是點值規範而非零均值規範；兩者所得解只應相差常數。若右端不相容，兩種規範都不能修復原問題。

### 解答五

把 `[0,1,2]` 改成 `[0,2,1]` 後，Jacobian 的兩個縱行交換，因此

$$
\det J_{\mathrm{new}}=-\det J_{\mathrm{old}}.
$$

實際面積 $|\det J|/2$ 不變。局部基底順序同步交換後，組裝到相同全域節點的剛度矩陣與負載應不變。

若第三點落在前兩點連線上，則 $\det J=0$，面積為零，$J^{-T}$ 不存在。此元素沒有二維面積，也無法定義正常的二維 P1 梯度；程式應拒絕，而不是加入零矩陣或使用偽逆悄悄繼續。

---

## 本章小結

二維三角 P1 有限元素以仿射映射把參考三角形帶到實體元素。Jacobian 同時控制基底梯度與面積尺度；物理面積使用 $|\det J|/2$，而翻轉繞序不應改變組裝後的物理解。

Poisson 型弱形式產生對稱剛度矩陣。當擴散係數正、反應係數非負，且有足夠 Dirichlet 邊界時，自由自由度矩陣通常對稱正定，有限元素解等價於離散能量的唯一極小值。純 Neumann 問題則具有常數零空間，必須先檢查右端相容性，再指定均值規範。

可靠實作至少要檢查非有限輸入、重複節點、退化元素、繞序不變性、常數零模態、非齊次 Dirichlet 消去、自由列真殘差與能量梯度。殘差小、能量極小、守恆、非負、收斂與物理可信度是不同命題，不能互相替代。

---

## 參考來源

1. FEniCSx Poisson 與弱形式，介紹 Poisson 問題、變分形式及有限元素實作概念：  
   https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html
2. PETSc KSP 手冊，供大型稀疏有限元素系統的 Krylov 求解與殘差診斷參考：  
   https://petsc.org/release/manual/ksp/
3. SciPy 稀疏線性代數 API，供後續以 CSR 與稀疏求解器取代教學用稠密矩陣：  
   https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
4. FiPy 離散與邊界說明，可用於比較有限體積通量觀點與本章有限元素弱形式：  
   https://pages.nist.gov/fipy/en/latest/numerical/discret.html

以上來源提供相應主題的延伸閱讀；本章 NumPy 程式為自足教學實作，未宣稱已執行來源程式或完成效能驗證。