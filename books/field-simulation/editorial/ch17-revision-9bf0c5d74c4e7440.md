# 第17章 二維有限元素與能量觀點

## 學習目標與先備知識

本章將一維有限元素推廣到二維三角形網格，並以 Poisson 型問題為主線，建立「弱形式—局部元素—全域組裝—邊界處理—能量最小化」的完整鏈條。完成本章後，讀者應能：

1. 從二維橢圓方程推導弱形式，分辨本質邊界與自然邊界。
2. 在任意非退化三角形上建立 P1 線性基底。
3. 使用仿射 Jacobian 計算元素面積、基底梯度、剛度矩陣與負載列向量。
4. 將局部矩陣組裝成全域系統，正確消去非齊次 Dirichlet 自由度。
5. 說明有限元素解何時等價於離散能量的唯一極小值。
6. 檢查翻轉繞序、退化元素、純 Neumann 零空間、能量梯度與線性殘差。
7. 分開判斷數值穩定、守恆、能量性質、非負性與物理可信度。

先備知識包括偏導數、梯度與散度、分部積分、對稱正定矩陣、線性方程組，以及一維 P1 有限元素。

本章使用**節點式網格**：未知量配置於節點，元素連接表指定每個三角形的三個節點。這不同於形狀為 `(Ny, Nx)` 的 cell-centered 陣列，也不使用 cell 索引 $k=jN_x+i$。座標採右手系，$X$ 向右、$Y$ 向上。

---

## 問題與直覺

考慮有界多邊形區域 $\Omega\subset\mathbb{R}^2$ 上的擴散—反應問題：

$$
-\nabla\cdot(k\nabla u)+cu=f
\quad\text{於 }\Omega.
$$

邊界分成 Dirichlet 部分 $\Gamma_D$ 與 Neumann 部分 $\Gamma_N$：

$$
u=g_D
\quad\text{於 }\Gamma_D,
$$

$$
(k\nabla u)\cdot n=q_N
\quad\text{於 }\Gamma_N,
$$

其中 $n$ 是外法向量。

若 $u$ 是溫度，物理熱通量通常定義為

$$
j=-k\nabla u.
$$

因此本章的 Neumann 資料 $q_N=(k\nabla u)\cdot n$ 與外向物理熱通量滿足

$$
j\cdot n=-q_N.
$$

兩種慣例皆可使用，但必須明列符號，否則入口與出口通量很容易顛倒。

有限元素並不要求微分方程在每個節點逐點成立，而是要求殘差對所有允許的測試函數加權後為零。三角 P1 元素在每個三角形內以平面近似 $u$；有限元素函數跨共享邊連續，但梯度通常在共享邊上跳躍。這符合 $H^1$ 弱解只要求函數及其一階弱導數平方可積，而不要求二階經典導數處處存在。

### 單位、資料與模型範圍

若 $x,y$ 以 m 計，$u$ 為 K，且方程代表每單位厚度的穩態導熱，則：

- $\nabla u$：K/m；
- $k$：W/(m·K)；
- $k\nabla u$：W/m²；
- $c$：W/(m³·K)；
- $f$：W/m³；
- $q_N$：W/m²。

二維面積積分可解釋為厚度 $1$ m 的三維體積積分。若實際厚度為 $H$，且場在厚度方向均勻，則總功率還須乘以 $H$。

本章問題是穩態橢圓問題，所以不需要初始條件。數值模型仍必須明列：

- 區域與節點座標；
- 元素連接表；
- 係數 $k,c$；
- 體來源 $f$；
- Dirichlet 與 Neumann 邊界；
- 單位與二維厚度解釋。

模型誤差與離散誤差必須分開。即使線性殘差接近零，錯誤的係數、錯誤的二維化假設或錯誤的邊界資料仍會產生物理上不可信的結果。

---

## 數學與物理推導

### 1. 弱形式

令測試函數 $v$ 在 $\Gamma_D$ 上為零。將偏微分方程乘以 $v$ 並積分：

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
-
\int_{\partial\Omega}(k\nabla u\cdot n)v\,d\Gamma.
$$

因為 $v=0$ 於 $\Gamma_D$，得到弱形式：

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
V_g=
\{w\in H^1(\Omega):w=g_D\text{ 於 }\Gamma_D\},
$$

$$
V_0=
\{v\in H^1(\Omega):v=0\text{ 於 }\Gamma_D\}.
$$

問題是求 $u\in V_g$，使得對所有 $v\in V_0$ 都有

$$
a(u,v)=\ell(v).
$$

Dirichlet 條件限制解所在的空間，故稱本質邊界條件；Neumann 條件在分部積分後自然進入右端，故稱自然邊界條件。

若某邊上的 $q_N=0$，弱形式中不需額外加入項，但這不代表邊界不存在，而是其邊界積分恰為零。

### 2. 三角 P1 基底與 Jacobian

取參考三角形

$$
\hat T=
\{(\xi,\eta):\xi\ge0,\ \eta\ge0,\ \xi+\eta\le1\},
$$

其頂點為 $(0,0),(1,0),(0,1)$。P1 基底為

$$
\hat N_1=1-\xi-\eta,\qquad
\hat N_2=\xi,\qquad
\hat N_3=\eta.
$$

設實體三角形頂點為

$$
x_a=
\begin{bmatrix}
x_a\\y_a
\end{bmatrix},
\qquad a=1,2,3.
$$

仿射映射為

$$
x(\xi,\eta)
=
x_1+
J
\begin{bmatrix}
\xi\\\eta
\end{bmatrix},
$$

其中

$$
J=
\begin{bmatrix}
x_2-x_1&x_3-x_1\\
y_2-y_1&y_3-y_1
\end{bmatrix}.
$$

有向面積與實際面積為

$$
A_s=\frac12\det J,
\qquad
A=\frac12|\det J|.
$$

若 $\det J>0$，節點為逆時針繞序；若 $\det J<0$，則為順時針繞序。實際積分測度必須使用 $|\det J|$，不能讓順時針元素產生負面積。

鏈式法則給出

$$
\nabla_xN_a=J^{-T}\nabla_{\xi}\hat N_a.
$$

參考基底梯度為

$$
\nabla_\xi\hat N_1=
\begin{bmatrix}-1\\-1\end{bmatrix},
\quad
\nabla_\xi\hat N_2=
\begin{bmatrix}1\\0\end{bmatrix},
\quad
\nabla_\xi\hat N_3=
\begin{bmatrix}0\\1\end{bmatrix}.
$$

由於映射是仿射的，實體 P1 基底梯度在每個元素內皆為常數。也可直接寫成

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

此處分母保留有向面積。若翻轉繞序，局部基底編號與梯度也一起重排；映射回相同全域節點後，物理矩陣不應改變。

### 3. 元素矩陣與負載

元素內的有限元素近似為

$$
u_h=\sum_{a=1}^3U_aN_a.
$$

定義梯度矩陣

$$
B=
\begin{bmatrix}
\partial_xN_1&\partial_xN_2&\partial_xN_3\\
\partial_yN_1&\partial_yN_2&\partial_yN_3
\end{bmatrix}.
$$

若 $k$ 在元素內為常數，擴散剛度矩陣為

$$
K^{(e)}_{ab}
=
\int_{T_e}k\nabla N_a\cdot\nabla N_b\,d\Omega
=
kA\nabla N_a\cdot\nabla N_b,
$$

亦即

$$
K^{(e)}_{\mathrm{diff}}=kAB^TB.
$$

若 $c$ 為元素常數，反應項為一致質量矩陣：

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

若 $f$ 為元素常數，元素負載列向量為

$$
F^{(e)}
=
\frac{fA}{3}
\begin{bmatrix}
1\\1\\1
\end{bmatrix}.
$$

對變係數或一般來源，數值積分是否精確取決於**完整被積函數的多項式次數**與積分公式的精度。仿射 P1 元素中的 $\nabla N_a\cdot\nabla N_b$ 為常數，因此若 $k(x,y)$ 是仿射函數，重心單點公式仍可精確積分剛度項；但對一般非線性或非多項式係數則未必精確。高次來源同樣需要較高階積分公式。

若 Neumann 邊界邊由節點 $i,j$ 組成，且 $q_N$ 在該邊為常數，邊長為 $L_e$，則其局部邊界負載為

$$
F^{(\Gamma)}=
\frac{q_NL_e}{2}
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

### 4. 全域組裝

設元素 $e$ 的局部節點對應全域節點 $(I_1,I_2,I_3)$，則

$$
K_{I_aI_b}\mathrel{+}=K^{(e)}_{ab},
\qquad
b_{I_a}\mathrel{+}=F^{(e)}_a.
$$

相鄰元素對共享節點的貢獻相加。P1 Galerkin 法不是有限體積逐面通量更新；若需要嚴格的逐控制體局部守恆，通常還要進行通量重建，或改用有限體積、混合有限元素等方法。

### 5. Dirichlet 消去

將自由度分成自由集合 $F$ 與固定集合 $D$：

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
\end{bmatrix}.
$$

已知 $U_D=g_D$ 後，應解

$$
K_{FF}U_F=b_F-K_{FD}g_D.
$$

若直接刪去固定自由度，卻未從右端扣除 $K_{FD}g_D$，非齊次 Dirichlet 問題將得到錯誤答案。齊次邊界因 $g_D=0$ 而無法揭露此錯誤，所以測試中必須包含非零邊界值。

### 6. 離散能量與唯一性

對稱擴散—反應問題的能量泛函為

$$
\Pi(u)=\frac12a(u,u)-\ell(u).
$$

離散後為

$$
\Pi_h(U)=\frac12U^TKU-b^TU.
$$

固定 $U_D=g_D$ 後，自由變數的約化能量可寫成

$$
\widetilde\Pi_h(U_F)
=
\frac12U_F^TK_{FF}U_F
+
U_F^TK_{FD}g_D
-
b_F^TU_F
+
C,
$$

其中 $C$ 與 $U_F$ 無關。其梯度為

$$
\nabla_{U_F}\widetilde\Pi_h
=
K_{FF}U_F+K_{FD}g_D-b_F.
$$

因此能量駐點恰好滿足自由度方程。其 Hessian 是

$$
\nabla^2_{U_F}\widetilde\Pi_h=K_{FF}.
$$

若 $K_{FF}$ 對稱正定，約化能量嚴格凸，駐點就是唯一全域極小值。常見充分條件為：

- $k(x,y)\ge k_{\min}>0$；
- $c(x,y)\ge0$；
- $\Gamma_D$ 具有正邊界測度。

若沒有 Dirichlet 邊界，但 $c$ 在足夠區域上嚴格為正，也可能消除常數零模態並提供唯一性。

### 7. 純 Neumann 零空間

只有在純 Neumann 且 $c=0$ 時，常數函數才是擴散算子的零模態：

$$
K\mathbf1=0.
$$

此時連續相容性為

$$
\int_\Omega f\,d\Omega
+
\int_{\partial\Omega}q_N\,d\Gamma
=0,
$$

離散相容性為

$$
\mathbf1^Tb=0.
$$

若資料相容，解只確定到任意常數。能量在常數方向上不是嚴格凸；只有在固定規範的子空間，例如零均值空間，才可談唯一極小值。若資料不相容，能量沿常數方向無下界或無駐點，不能用任意固定一點來修復原問題。

若 $c>0$，常數函數通常不再是零模態，也不再要求上述只含 $f$ 與 $q_N$ 的相容條件。此時將方程對全域積分得到

$$
\int_\Omega cu\,d\Omega
=
\int_\Omega f\,d\Omega
+
\int_{\partial\Omega}q_N\,d\Gamma,
$$

其中包含未知解 $u$，不能誤寫成對資料的零和條件。

---

## 逐步手算例題

### 例一：單一直角三角形

考慮頂點

$$
x_1=(0,0),\quad x_2=(2,0),\quad x_3=(0,1),
$$

並取 $k=3$、$c=0$、$f=6$。

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
\begin{bmatrix}
-1/2\\-1
\end{bmatrix},
\quad
\nabla N_2=
\begin{bmatrix}
1/2\\0
\end{bmatrix},
\quad
\nabla N_3=
\begin{bmatrix}
0\\1
\end{bmatrix}.
$$

三個梯度之和為零，符合 $N_1+N_2+N_3=1$。

**步驟三：剛度矩陣**

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

每一橫列的元素和為零，所以

$$
K^{(e)}\mathbf1=0.
$$

**步驟四：體來源負載**

$$
F^{(e)}
=
\frac{6(1)}{3}
\begin{bmatrix}
1\\1\\1
\end{bmatrix}
=
\begin{bmatrix}
2\\2\\2
\end{bmatrix}.
$$

元素矩陣只是全域組裝的材料之一；若三個節點全是 Dirichlet 節點，便沒有自由未知量可由此元素單獨求解。

### 例二：兩個三角形組成單位正方形

取四個節點

$$
x_0=(0,0),\quad
x_1=(1,0),\quad
x_2=(1,1),\quad
x_3=(0,1),
$$

並以對角線 $(0,2)$ 切成

$$
T_1=(0,1,2),\qquad T_2=(0,2,3).
$$

令 $k=1$、$c=0$。兩個元素面積皆為 $1/2$。第一個局部矩陣為

$$
K^{(1)}
=
\frac12
\begin{bmatrix}
1&-1&0\\
-1&2&-1\\
0&-1&1
\end{bmatrix},
$$

其局部次序是 $(0,1,2)$。第二個為

$$
K^{(2)}
=
\frac12
\begin{bmatrix}
1&0&-1\\
0&1&-1\\
-1&-1&2
\end{bmatrix},
$$

其局部次序是 $(0,2,3)$。

組裝後得到

$$
K=
\begin{bmatrix}
1&-1/2&0&-1/2\\
-1/2&1&-1/2&0\\
0&-1/2&1&-1/2\\
-1/2&0&-1/2&1
\end{bmatrix}.
$$

可驗證 $K\mathbf1=0$。若將 $T_1$ 改寫成 $(0,2,1)$，其 Jacobian 行列式會改變符號，但正確使用絕對面積並同步重排局部自由度後，全域矩陣仍應相同。

這個只有四個角點的網格若在整個外邊界施加 Dirichlet 條件，所有節點都已固定，沒有內部自由度。因此要展示實際求解，還需加入內部節點。

### 例三：帶中心節點的正方形

取節點

$$
(0,0),(1,0),(1,1),(0,1),(1/2,1/2),
$$

將中心連到四條外邊，形成四個面積皆為 $1/4$ 的三角形。求解

$$
-\Delta u=1,
\qquad
u=0\text{ 於 }\partial\Omega.
$$

四個角點固定為零，唯一自由度是中心值 $U_c$。中心到每一外邊的距離為 $1/2$，因此中心基底在各元素內滿足

$$
|\nabla N_c|=2.
$$

每個元素對中心對角項的貢獻為

$$
K^{(e)}_{cc}
=
A|\nabla N_c|^2
=
\frac14(4)=1.
$$

四個元素合計為

$$
K_{cc}=4.
$$

常數來源對每個元素的中心負載為

$$
F^{(e)}_c=\frac{A}{3}=\frac1{12},
$$

所以

$$
F_c=4\left(\frac1{12}\right)=\frac13.
$$

離散方程是

$$
4U_c=\frac13,
$$

故

$$
U_c=\frac1{12}.
$$

消去固定自由度後的約化能量為

$$
\widetilde\Pi_h(U_c)
=
\frac12(4)U_c^2-\frac13U_c+C.
$$

本例的 Dirichlet 值為零，故可取 $C=0$。其導數與二階導數為

$$
\widetilde\Pi_h'(U_c)=4U_c-\frac13,
\qquad
\widetilde\Pi_h''(U_c)=4>0.
$$

因此 $U_c=1/12$ 是約化能量的唯一極小值。這只是該粗網格的離散解，不能由單一網格推定連續解誤差或收斂階。

---

## 實作與程式

以下程式只依賴 Python 3.10+ 與 NumPy，使用稠密矩陣，適合 CPU 教學小網格。它處理常數 $k,c,f$、齊次或非齊次 Dirichlet 邊界，以及預設零 Neumann 邊界；一般非零 Neumann 邊負載可依前述公式另行組裝。

```python
import numpy as np


def validate_mesh(points, triangles, kappa, reaction, source):
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points 必須具有形狀 (nnode, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points 含 NaN 或無窮值")

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles 必須具有形狀 (nelem, 3)")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise TypeError("triangles 的節點編號必須是整數")
    triangles = triangles.astype(int, copy=False)

    scalars = np.array([kappa, reaction, source], dtype=float)
    if not np.all(np.isfinite(scalars)):
        raise ValueError("係數或來源不是有限值")
    if kappa <= 0.0:
        raise ValueError("本例要求 kappa > 0")
    if reaction < 0.0:
        raise ValueError("本例要求 reaction >= 0")

    nnode = len(points)
    if np.any(triangles < 0) or np.any(triangles >= nnode):
        raise ValueError("元素含越界節點編號")

    for tri in triangles:
        if len(set(int(i) for i in tri)) != 3:
            raise ValueError("元素內有重複節點")

    return points, triangles


def triangle_matrices(xy, kappa, reaction, source,
                      relative_tol=1.0e-14):
    edge01 = xy[1] - xy[0]
    edge02 = xy[2] - xy[0]
    edge12 = xy[2] - xy[1]

    J = np.column_stack((edge01, edge02))
    detJ = np.linalg.det(J)

    # 使用局部邊長尺度，避免判定受整體座標平移影響。
    local_scale2 = max(
        np.dot(edge01, edge01),
        np.dot(edge02, edge02),
        np.dot(edge12, edge12),
    )
    if local_scale2 == 0.0:
        raise ValueError("零尺寸三角形")
    if not np.isfinite(detJ):
        raise ValueError("Jacobian 行列式不是有限值")
    if abs(detJ) <= relative_tol * local_scale2:
        raise ValueError("退化或近退化三角形")

    area = 0.5 * abs(detJ)

    grad_ref = np.array([
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
    ])
    B = np.linalg.solve(J.T, grad_ref)

    Ke_diff = kappa * area * (B.T @ B)
    Me = (area / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    Ke = Ke_diff + reaction * Me
    fe = source * area / 3.0 * np.ones(3)

    return Ke, fe, area, detJ, B


def assemble(points, triangles, kappa=1.0,
             reaction=0.0, source=1.0):
    points, triangles = validate_mesh(
        points, triangles, kappa, reaction, source
    )

    nnode = len(points)
    K = np.zeros((nnode, nnode), dtype=float)
    b = np.zeros(nnode, dtype=float)
    signed_dets = []

    for tri in triangles:
        Ke, fe, area, detJ, B = triangle_matrices(
            points[tri], kappa, reaction, source
        )
        signed_dets.append(detJ)

        for a, I in enumerate(tri):
            b[I] += fe[a]
            for q, Jidx in enumerate(tri):
                K[I, Jidx] += Ke[a, q]

    return K, b, np.asarray(signed_dets)


def validate_linear_system(K, b):
    K = np.asarray(K, dtype=float)
    b = np.asarray(b, dtype=float)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K 必須是方陣")
    if b.ndim != 1 or len(b) != K.shape[0]:
        raise ValueError("b 的形狀與 K 不一致")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(b)):
        raise ValueError("K 或 b 含 NaN 或無窮值")

    return K, b


def parse_dirichlet(dirichlet, nnode):
    if not isinstance(dirichlet, dict):
        raise TypeError("dirichlet 必須是 {節點: 值} 字典")

    fixed_list = []
    values_list = []

    for key, value in dirichlet.items():
        # bool 是 int 的子類別，但不適合作節點編號。
        if isinstance(key, (bool, np.bool_)):
            raise TypeError("Dirichlet 節點編號不可為布林值")
        if not isinstance(key, (int, np.integer)):
            raise TypeError("Dirichlet 節點編號必須是整數")
        idx = int(key)
        if idx < 0 or idx >= nnode:
            raise ValueError("Dirichlet 節點編號越界")
        if not np.isfinite(value):
            raise ValueError("Dirichlet 值含 NaN 或無窮值")
        fixed_list.append(idx)
        values_list.append(float(value))

    order = np.argsort(fixed_list)
    fixed = np.asarray(fixed_list, dtype=int)[order]
    values = np.asarray(values_list, dtype=float)[order]
    return fixed, values


def solve_dirichlet(K, b, dirichlet):
    K, b = validate_linear_system(K, b)
    nnode = len(b)
    fixed, values = parse_dirichlet(dirichlet, nnode)

    is_free = np.ones(nnode, dtype=bool)
    is_free[fixed] = False
    free = np.nonzero(is_free)[0]

    u = np.zeros(nnode, dtype=float)
    u[fixed] = values

    if len(free) > 0:
        Kff = K[np.ix_(free, free)]
        rhs = b[free] - K[np.ix_(free, fixed)] @ values
        u[free] = np.linalg.solve(Kff, rhs)

    # 真殘差採 r=b-Ku；只檢查仍受方程約束的自由列。
    residual_free = (
        b[free] - K[np.ix_(free, np.arange(nnode))] @ u
        if len(free) > 0 else np.zeros(0)
    )
    return u, free, residual_free


def energy(K, b, u):
    K, b = validate_linear_system(K, b)
    u = np.asarray(u, dtype=float)
    if u.shape != b.shape or not np.all(np.isfinite(u)):
        raise ValueError("u 的形狀或數值無效")
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
    ], dtype=int)

    K, b, dets = assemble(
        points, triangles,
        kappa=1.0,
        reaction=0.0,
        source=1.0,
    )

    boundary = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }
    u, free, rfree = solve_dirichlet(K, b, boundary)

    print("signed detJ =", dets)
    print("u =", u)
    print("free true residual norm =", np.linalg.norm(rfree))
    print("energy =", energy(K, b, u))
    print(
        "free energy gradient =",
        energy_gradient_free(K, b, u, free),
    )

    # 翻轉第一個元素的繞序。
    triangles_flip = triangles.copy()
    triangles_flip[0] = triangles_flip[0, [0, 2, 1]]
    K2, b2, dets2 = assemble(
        points, triangles_flip,
        kappa=1.0,
        reaction=0.0,
        source=1.0,
    )

    print("flipped signed detJ =", dets2)
    print(
        "orientation K difference =",
        np.max(np.abs(K2 - K)),
    )
    print(
        "orientation b difference =",
        np.max(np.abs(b2 - b)),
    )


if __name__ == "__main__":
    main()
```

程式沒有自動安裝套件、沒有使用 GPU，也不連接遠端服務或設備。大型網格應改用 CSR 稀疏矩陣與合適的稀疏求解器，但幾何、邊界、相容性與真殘差檢查仍不可省略。

---

## 測試與預期結果

以下皆是依解析推導得到的**預期結果**，不是執行紀錄。

### 1. 正常測試

對中心節點正方形問題，預期

$$
u=
\begin{bmatrix}
0&0&0&0&1/12
\end{bmatrix}^T.
$$

中心值應接近 `0.0833333333333`。自由列真殘差

$$
r_F=b_F-(KU)_F
$$

應接近浮點捨入尺度。能量梯度

$$
(KU-b)_F
$$

也應接近零，且與真殘差只差負號。

由於

$$
\widetilde\Pi_h(1/12)
=
2\left(\frac1{12}\right)^2
-
\frac13\left(\frac1{12}\right)
=
-\frac1{72},
$$

程式能量預期接近 $-1/72$。

### 2. 兩三角形組裝測試

對兩三角形單位正方形，預期組裝矩陣為

$$
\begin{bmatrix}
1&-1/2&0&-1/2\\
-1/2&1&-1/2&0\\
0&-1/2&1&-1/2\\
-1/2&0&-1/2&1
\end{bmatrix}.
$$

應滿足：

$$
K=K^T,
\qquad
K\mathbf1=0.
$$

這是對稱性與常數場測試，不代表具有 Dirichlet 邊界後仍存在常數零模態。

### 3. 翻轉繞序測試

翻轉第一個三角形後，其 $\det J$ 應改變符號，但：

$$
K_{\mathrm{flip}}\approx K,
\qquad
b_{\mathrm{flip}}\approx b.
$$

若矩陣出現明顯差異，通常是：

- 積分面積誤用 $\det J/2$；
- 局部節點重新排序但局部資料未同步；
- 基底梯度轉換方向寫錯。

### 4. 非齊次邊界測試

將來源設為零，並令四個邊界節點皆為 $2$。常數函數 $u=2$ 滿足 Laplace 方程，因此中心值也應為 $2$。此測試專門檢查

$$
b_F-K_{FD}g_D
$$

是否正確形成。

### 5. 故障測試

下列輸入應被拒絕：

1. 共線三點，如 $(0,0),(1,0),(2,0)$；
2. 重複節點元素 `[0, 1, 1]`；
3. 越界節點編號；
4. 浮點元素編號，如 `1.5`；
5. Dirichlet 鍵為 `2.7`、字串或布林值；
6. 座標、矩陣、右端或邊界值含 `NaN` 或無窮值；
7. $k\le0$ 或本例中的 $c<0$；
8. 純 Neumann、$c=0$，但 $\mathbf1^Tb\ne0$。

極瘦三角形即使尚未被退化門檻拒絕，也可能造成很大的矩陣條件數。通過輸入檢查不等於具有良好網格品質。

### 6. 不同性質分開判斷

- **數值穩定或代數可解性**：討論矩陣條件數、演算法與浮點誤差。
- **守恆**：需明確定義積分量與邊界通量；標準連續 P1 法不自動提供逐控制體局部守恆。
- **能量性質**：本章穩態解是固定邊界下的能量駐點；只有在正定條件下才是唯一極小值。
- **能量下降**：這是時間演化或特定迭代序列的性質，不能由最終解是極小值推得每一步都下降。
- **非負性**：SPD 不保證節點值非負；還需要網格角度與 M-matrix 類條件。
- **物理可信度**：還需驗證方程、參數、單位、邊界與二維化假設。
- **收斂**：必須經網格細化、製造解或誤差估計檢查，不能只看平滑圖形。

---

## 除錯與常見陷阱

### 1. 把有向面積當成實際面積

順時針元素滿足 $\det J<0$。若直接以 $\det J/2$ 乘上 $B^TB$，剛度可能變成負半定，破壞能量凸性。正確做法是：

- 梯度由 $J^{-T}$ 計算；
- 面積積分使用 $|\det J|/2$。

### 2. 只改節點座標，不改局部資料

若程式為統一繞序而交換節點，材料標籤、局部來源、邊界邊編號等資料也必須同步重排。靜默交換部分資料會產生難以追蹤的錯誤。

### 3. 遺漏非齊次 Dirichlet 貢獻

只刪除固定自由度而未扣除 $K_{FD}g_D$，會使非零邊界值無法正確傳入內部。齊次邊界測試看不出此錯誤。

### 4. 面積門檻依賴絕對座標

若以 `max(abs(xy))**2` 作尺度，同一三角形平移到很大的座標後可能被錯判為退化。門檻應依局部邊長或局部 Jacobian 尺度決定。本章程式使用三條邊長平方的最大值。

### 5. 把完整殘差用於消去式 Dirichlet 系統

消去 Dirichlet 自由度後，只有自由列仍要求滿足原方程。固定列上的 $KU-b$ 可解釋為維持指定邊界值所需的離散反力或通量，不必為零。停止或驗收應檢查自由列真殘差。

### 6. 把殘差小當成解誤差小

小殘差只能表示線性方程求解得較準。若矩陣條件數很大，解誤差仍可能顯著；即使代數解精確，也仍存在有限元素離散誤差與模型誤差。

### 7. 誤解跨元素梯度跳躍

P1 解跨共享邊連續，但梯度一般不連續。這並非自動錯誤，而是 $H^1$ 相容有限元素的正常性質。若需要局部守恆通量，需另做通量重建或改用其他離散。

### 8. 把正定性誤認為非負性

$K_{FF}$ 對稱正定只保證能量嚴格凸與解唯一，不保證 $K_{FF}^{-1}$ 的元素皆非負。鈍角三角形可產生正的非對角剛度項，破壞離散最大值原理。

---

## 養殖與相場案例

### 合成池域的穩態擴散—反應場

設合成池域中的溶質濃度為 $C$，單位 kg/m³，考慮

$$
-\nabla\cdot(D_{\mathrm{eff}}\nabla C)+\lambda C=S.
$$

其中：

- $D_{\mathrm{eff}}$：m²/s；
- $\lambda$：1/s；
- $S$：kg/(m³·s)；
- $D_{\mathrm{eff}}\nabla C$：kg/(m²·s)。

入口可指定 Dirichlet 濃度，封閉池壁可設零 Neumann 條件。若二維模型代表每單位厚度，面積積分後的量是每單位厚度收支；若代表深度平均，必須另行推導深度因子，不能直接沿用三維係數而不說明。

若用 mg/L 表示濃度，由

$$
1\text{ mg}=10^{-6}\text{ kg},
\qquad
1\text{ L}=10^{-3}\text{ m}^3,
$$

可得

$$
1\text{ mg/L}=10^{-3}\text{ kg/m}^3.
$$

本案例只使用合成參數，不提供現場管理閾值，也不宣稱可控制實際設備。溶氧跨過管理門檻不是熱力學相變。

### 相場能量的空間離散橋接

對無因次序參量 $\phi$，採本卷慣例

$$
W(\phi)=\frac{(\phi^2-1)^2}{4},
$$

$$
F[\phi]
=
\int_\Omega
\left[
W(\phi)+\frac{\kappa}{2}|\nabla\phi|^2
\right]d\Omega.
$$

P1 有限元素可離散梯度能。若 $K$ 表示單位係數的擴散剛度矩陣，則

$$
F_{\mathrm{grad},h}
=
\frac{\kappa}{2}\Phi^TK\Phi.
$$

雙井項不能把節點值 $W(\Phi_i)$ 無權重相加，而應做元素積分：

$$
F_{\mathrm{bulk},h}
=
\sum_e\int_{T_e}
W\left(\sum_a\Phi_aN_a\right)d\Omega.
$$

其離散梯度包含非線性體能項與 $\kappa K\Phi$。這與 Poisson 能量有相似結構，但 Allen–Cahn 與 Cahn–Hilliard 還需要質量矩陣、時間格式、非線性求解與相應邊界條件。

連續能量耗散不保證任意時間步長下的離散能量都下降；非線性求解容差也會影響能量檢查。Cahn–Hilliard 在週期或適當無通量邊界下守恆，而 Allen–Cahn 一般不守恆，兩者不能只靠同一剛度矩陣混為一談。

---

## 習題

### 習題一：手算局部矩陣

對三角形 $(0,0),(1,0),(0,1)$，取 $k=2$、$c=0$、$f=3$。求面積、三個基底梯度、元素剛度矩陣與負載列向量，並驗證常數零模態。

### 習題二：程式與非齊次邊界

將程式中的來源設為零，並把四個外部節點都設為 $2$。預測中心值、自由真殘差與自由能量梯度。說明為何完整向量 $KU-b$ 不必為零。

### 習題三：反例與離散最大值原理

證明一個三角形上兩個不同頂點 $i,j$ 的擴散剛度項可寫成

$$
K^{(e)}_{ij}
=
-\frac{k}{2}\cot\theta_k,
$$

其中 $\theta_k$ 是第三個頂點的內角。據此說明鈍角元素如何破壞非正非對角結構。

### 習題四：純 Neumann 整合題

考慮單位正方形上的

$$
-\Delta u=1,
\qquad
\nabla u\cdot n=0.
$$

判斷是否有解。再考慮

$$
-\Delta u=x-\frac12,
\qquad
\nabla u\cdot n=0.
$$

檢查相容性，說明零空間，並寫出以拉格朗日乘子施加零均值條件的離散增廣系統。比較它與任意固定一點的差異。

### 習題五：翻轉與退化測試

將元素 `[0,1,2]` 改成 `[0,2,1]`，再把第三點移到前兩點所在直線上。說明 $\det J$、面積、剛度矩陣與程式行為的預期變化。

---

## 習題解答

### 解答一

面積為

$$
A=\frac12.
$$

基底與梯度為

$$
N_1=1-x-y,\qquad N_2=x,\qquad N_3=y,
$$

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

代入 $k=2$ 與 $A=1/2$：

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
\begin{bmatrix}
1\\1\\1
\end{bmatrix}.
$$

各橫列元素和為零，所以

$$
K^{(e)}\mathbf1=0.
$$

### 解答二

來源為零且所有邊界值皆為 $2$ 時，常數函數 $u=2$ 的梯度為零，滿足 Laplace 方程。因此中心值預期為

$$
U_c=2.
$$

自由列真殘差為

$$
r_F=b_F-(KU)_F=0,
$$

自由能量梯度為

$$
(KU-b)_F=0.
$$

在浮點計算中，兩者預期只剩捨入誤差。

固定節點上的原始方程已被 Dirichlet 條件取代，所以完整 $KU-b$ 在固定列上不必為零。這些分量可視為維持指定邊界值所需的離散反力或通量。

### 解答三

P1 基底 $N_i$ 的梯度垂直於與頂點 $i$ 相對的邊。由三角形面積、邊長與夾角關係可得，對 $i\ne j$：

$$
A\nabla N_i\cdot\nabla N_j
=
-\frac12\cot\theta_k,
$$

其中 $k$ 是第三個頂點。因此

$$
K^{(e)}_{ij}
=
kA\nabla N_i\cdot\nabla N_j
=
-\frac{k}{2}\cot\theta_k.
$$

若 $\theta_k<90^\circ$，則 $\cot\theta_k>0$，故 $K^{(e)}_{ij}<0$。若 $\theta_k>90^\circ$，則 $\cot\theta_k<0$，故

$$
K^{(e)}_{ij}>0.
$$

因此鈍角三角形可能產生正非對角項，破壞 M-matrix 型結構。矩陣仍可能對稱正定，但節點解不再自動滿足離散最大值原理。這證明正定性與非負性是不同性質。

### 解答四

第一個問題的相容性左側為

$$
\int_\Omega1\,d\Omega
+
\int_{\partial\Omega}0\,d\Gamma
=1\ne0.
$$

因此原純 Neumann 問題無解。固定一個節點只能使修改後的代數系統可解，不能修復原偏微分方程的不相容資料。

第二個來源滿足

$$
\int_0^1\int_0^1
\left(x-\frac12\right)\,dy\,dx
=
\int_0^1\left(x-\frac12\right)dx
=0.
$$

因此資料相容。離散矩陣滿足

$$
K\mathbf1=0,
$$

而一致組裝的右端應滿足

$$
\mathbf1^Tb=0.
$$

設

$$
m_i=\int_\Omega N_i\,d\Omega.
$$

以拉格朗日乘子施加有限元素函數的零均值條件，可解

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

第一組方程是

$$
KU+m\lambda=b,
$$

第二組是

$$
m^TU=0.
$$

對相容右端，理想精確算術下應得到 $\lambda=0$，因為拉格朗日乘子只用來固定常數規範，不應補償不相容來源。實作前仍須先檢查 $\mathbf1^Tb=0$；不得以非零 $\lambda$ 掩蓋不相容資料。

固定任意一點也可選出一個代表解，但其規範是點值規範，不是零均值規範。對相容問題，兩種解應只相差常數；對不相容問題，兩者都不能使原問題成立。

### 解答五

將 `[0,1,2]` 改成 `[0,2,1]` 等於交換 Jacobian 的兩個縱行，因此

$$
\det J_{\mathrm{new}}
=
-\det J_{\mathrm{old}}.
$$

實際面積

$$
A=\frac12|\det J|
$$

不變。局部基底與節點次序同步重排後，組裝到相同全域自由度的剛度矩陣與負載應不變。

若第三點落在前兩點所在直線上，則

$$
\det J=0,
\qquad
A=0.
$$

此時 $J^{-T}$ 不存在，三角形不再具有二維面積，P1 基底的二維梯度無法正常定義。程式應明確拒絕，不可用偽逆、裁切面積或加入零矩陣後繼續。

---

## 本章小結

二維三角 P1 有限元素利用仿射 Jacobian 將參考三角形映射到實體元素。Jacobian 同時控制基底梯度與面積尺度；梯度使用 $J^{-T}$，實際積分面積使用 $|\det J|/2$。因此正確實作對順時針與逆時針繞序應具有不變性。

擴散—反應弱形式產生對稱矩陣。當擴散係數嚴格為正、反應係數非負，且有足夠 Dirichlet 約束時，自由自由度矩陣通常為對稱正定，有限元素解是約化離散能量的唯一極小值。純 Neumann 且 $c=0$ 時則存在常數零空間；只有在右端相容並固定均值規範後，才可談唯一代表解與唯一受限極小值。

可靠實作至少要檢查非有限輸入、整數節點編號、重複節點、越界索引、退化元素、繞序不變性、常數零模態、非齊次 Dirichlet 消去、自由列真殘差與能量梯度。代數殘差、守恆、能量下降、非負性、收斂與物理可信度是不同命題，不能互相替代。

---

## 參考來源

1. **FEniCSx Poisson 與弱形式**：Poisson 問題、變分形式與有限元素基本實作。  
   https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html
2. **PETSc KSP 手冊**：大型稀疏線性系統、Krylov 方法與殘差診斷。  
   https://petsc.org/release/manual/ksp/
3. **SciPy 稀疏線性代數 API**：以 CSR 與稀疏求解器取代教學用稠密矩陣時的介面參考。  
   https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
4. **FiPy 有限體積離散與邊界**：比較有限體積通量觀點與有限元素弱形式。  
   https://pages.nist.gov/fipy/en/latest/numerical/discret.html
5. **FiPy Cahn–Hilliard 範例**：相場守恆語意與數值模型延伸參考。其序參量慣例不必然與本卷的 $[-1,1]$ 雙井相同。  
   https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html

以上來源僅供相應主題延伸閱讀。本章 NumPy 程式為自足教學實作；本文未宣稱已執行程式、完成效能測試或驗證任何現場物理模型。