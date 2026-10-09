# 第20章 Jacobian行列式與積分換變數

## 學習目標與先備知識

在平面或空間中改用另一組座標描述同一區域時，積分中的面積或體積元素如何改變？答案由座標映射的 Jacobian 行列式控制。行列式的絕對值描述局部體積縮放；正負則記錄方向是否反轉。要把這個局部幾何結論用於積分，還須確認映射的定義域、正則性、覆蓋次數，以及邊界或奇異點如何處理。

學完本章後，讀者應能：

1. 由可微映射的線性近似，解釋 Jacobian 行列式為何表示局部面積或體積縮放。
2. 正確使用普通多重積分的換變數公式，並區分面積／體積元素與有向量。
3. 手算仿射變換、方向反轉、極座標與簡單重複覆蓋例子。
4. 判斷換變數定理的假設是否成立，辨認奇異 Jacobian、重複覆蓋與邊界例外。
5. 撰寫有限精度的 CPU 計算檢查，並清楚標示數值檢查不是定理證明。

先備知識包括向量與矩陣乘法、行列式、連續與可微函數，以及 Riemann 多重積分。本文採列向量慣例：若 $T:\mathbb R^n\to\mathbb R^n$，則 Jacobian 矩陣 $J_T(u)$ 的形狀為 $n\times n$，輸入擾動 $h$ 為 $n\times1$，線性化為 $J_T(u)h$。

## 問題與直覺

考慮一個小正方形，經線性映射後成為平行四邊形。平行四邊形的面積由兩個方向共同決定：若映射矩陣為

$$
A=
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix},
$$

則單位正方形映成的平行四邊形面積為 $|\det A|=|ad-bc|$。若 $\det A<0$，平面方向反轉，但普通面積仍非負，因此普通面積積分使用 $|\det A|$。

對非線性映射 $T$，在點 $u$ 附近，可微性給出

$$
T(u+h)=T(u)+J_T(u)h+r_u(h),
\qquad
\frac{\|r_u(h)\|}{\|h\|}\longrightarrow 0
\quad(h\to0,\ h\ne0).
$$

這裡採用歐氏範數。第一項是基點，$J_T(u)h$ 是線性變形，而餘項相對於尺度趨近於零。因而愈小的區塊愈接近由 $J_T(u)$ 描述的平行多面體，其局部體積縮放因子為 $|\det J_T(u)|$。

這是局部幾何直覺，不是完整的積分定理證明。要將小區塊體積近似加總，必須控制映射在區域上的行為，並排除或妥善處理重複覆蓋與奇異點。

## 定義、定理與推導

### Jacobian 與局部體積縮放

令 $U\subset\mathbb R^n$ 為開集，$T:U\to\mathbb R^n$ 為 $C^1$ 映射。以 $u=(u_1,\ldots,u_n)$ 表示參數，以 $x=T(u)$ 表示目標座標，則

$$
J_T(u)=
\begin{pmatrix}
\frac{\partial T_1}{\partial u_1}&\cdots&\frac{\partial T_1}{\partial u_n}\\
\vdots&&\vdots\\
\frac{\partial T_n}{\partial u_1}&\cdots&\frac{\partial T_n}{\partial u_n}
\end{pmatrix}.
$$

若 $\det J_T(u)\ne0$，逆函數定理保證 $T$ 在 $u$ 的某個鄰域內局部可逆。這是局部結論，不保證整個 $U$ 上一對一。局部有向體積依 $\det J_T(u)$ 縮放；不帶方向的普通體積依 $|\det J_T(u)|$ 縮放。

在二維，兩個切向量 $v_1,v_2$ 張成的平行四邊形面積為 $|\det[v_1\ v_2]|$。線性映射 $A$ 把單位正方形的兩條邊送到 $Ae_1,Ae_2$，所以面積為

$$
|\det[Ae_1\ Ae_2]|=|\det A|.
$$

三維情形中，三個映射後的邊向量張成平行六面體，其體積亦為行列式絕對值。高維情形由相同的行列式體積公式描述。

### 換變數定理

**定理（多重積分換變數，經典形式）。** 設 $U,V\subset\mathbb R^n$ 為開集，$T:U\to V$ 是 $C^1$ 微分同胚，也就是 $T$ 為雙射且反函數 $T^{-1}:V\to U$ 亦為 $C^1$。若 $f:V\to\mathbb R$ 連續且在 $V$ 上具緊支集，則

$$
\int_V f(x)\,dx
=
\int_U f(T(u))\,|\det J_T(u)|\,du.
$$

這是本章採用的定理形式。對適當的可測函數，另有更一般的換變數定理，但其可測性與可積性條件須另行說明。

定理要求 $T$ 對整個使用區域一對一。若映射在區域內以固定有限次數 $k$ 覆蓋目標區域，且除去適當的零測集後各分支均正則，則將各分支分開套用公式時會計入覆蓋次數；不能把這種情況直接稱為一對一換變數。邊界在一般面積積分中常可忽略，前提是邊界為零測集；若邊界有正測度或積分具奇異性，便須另行檢查。對有向積分或微分形式則保留 $\det J_T$ 的符號，不能任意以絕對值取代。

### 小命題與證明：非奇異線性變換的體積縮放

**命題。** 若 $A$ 是 $n\times n$ 可逆矩陣，$E$ 是有限體積的可測集合，則

$$
\operatorname{vol}(AE)=|\det A|\,\operatorname{vol}(E).
$$

**證明。** 將 $A$ 作奇異值分解 $A=Q_1\Sigma Q_2^T$，其中 $Q_1,Q_2$ 為正交矩陣，$\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_n)$，且每個 $\sigma_i>0$。正交變換保持歐氏距離與體積，因此 $Q_2^T$ 和 $Q_1$ 都不改變集合體積。對角變換 $\Sigma$ 將第 $i$ 個座標方向的長度乘以 $\sigma_i$，故將長方體體積乘以 $\prod_i\sigma_i$；藉由長方體對可測集合體積的近似，這個縮放關係延伸至 $E$。另一方面，

$$
|\det A|
=
|\det Q_1|\,|\det\Sigma|\,|\det Q_2^T|
=
\prod_i\sigma_i,
$$

因為正交矩陣行列式的絕對值為 $1$。合併兩個結果即得命題。證畢。

此命題證明線性變換的體積縮放。非線性換變數定理還要利用局部線性化、區域細分與積分極限，不能只憑單點的 Jacobian 就當作整體公式。

### 仿射情形

若 $T(u)=Au+b$，其中 $A$ 可逆，則 $J_T(u)=A$，Jacobian 為常數。因此

$$
\int_{T(E)} f(x)\,dx
=
\int_E f(Au+b)\,|\det A|\,du.
$$

平移 $b$ 不改變體積；矩陣 $A$ 可包含旋轉、反射、剪切與伸縮。旋轉的行列式絕對值為 $1$；純伸縮則以各方向伸縮倍率的乘積改變體積。

### 極座標

極座標映射為

$$
T(r,\theta)=(r\cos\theta,r\sin\theta),
\qquad
J_T(r,\theta)=
\begin{pmatrix}
\cos\theta&-r\sin\theta\\
\sin\theta&r\cos\theta
\end{pmatrix}.
$$

因此

$$
\det J_T(r,\theta)=r.
$$

在 $r>0$ 且角度限制在不重複的區間時，映射局部正則，並可在相應區域上一對一。對適當的區域與可積函數，

$$
\iint_D f(x,y)\,dx\,dy
=
\iint_{T^{-1}(D)}
f(r\cos\theta,r\sin\theta)\,r\,dr\,d\theta.
$$

在 $r=0$，Jacobian 為零，而且所有角度都映到原點；角度區間若包含長度 $2\pi$ 的兩端，兩端表示同一條射線。這些點或邊界在一般面積積分中通常是零面積集合，但在陳述一對一條件時仍須處理。因子 $r$ 的幾何意義是：半徑較大處，每一小段角度所涵蓋的弧長較長，因此局部面積元素也較大。

## 逐步手算例題

### 例一：剪切與伸縮的仿射換元

令

$$
T(u,v)=(2u+v,v),
\qquad
S=\{(u,v):0\le u\le1,\ 0\le v\le1\}.
$$

計算 $\iint_{T(S)}1\,dx\,dy$。

**第一步：求 Jacobian。**

$$
J_T=
\begin{pmatrix}
2&1\\
0&1
\end{pmatrix},
\qquad
\det J_T=2.
$$

**第二步：使用仿射換變數公式。** 映射可逆，且將正方形一對一映成平行四邊形，因此

$$
\iint_{T(S)}1\,dx\,dy
=
\int_0^1\int_0^1 2\,dv\,du
=2.
$$

也可直接看兩條邊向量 $(2,0)$ 與 $(1,1)$；其平行四邊形面積為 $|2\cdot1-0\cdot1|=2$。

### 例二：圓盤上的二次函數積分

計算半徑 $R$ 的圓盤 $D=\{(x,y):x^2+y^2\le R^2\}$ 上的積分

$$
\iint_D (x^2+y^2)\,dx\,dy.
$$

使用 $x=r\cos\theta,\ y=r\sin\theta$，其中 $0\le r\le R$、$0\le\theta\le2\pi$。被積函數變成 $r^2$，面積元素變成 $r\,dr\,d\theta$，故

$$
\begin{aligned}
\iint_D (x^2+y^2)\,dx\,dy
&=\int_0^{2\pi}\int_0^R r^3\,dr\,d\theta\\
&=2\pi\left[\frac{r^4}{4}\right]_0^R
=\frac{\pi R^4}{2}.
\end{aligned}
$$

若 $x,y,R$ 的單位是長度，積分單位為長度的四次方，與被積函數乘上面積元素的單位一致。

### 例三：方向反轉

令 $T(u,v)=(v,u)$。則

$$
J_T=
\begin{pmatrix}
0&1\\
1&0
\end{pmatrix},
\qquad
\det J_T=-1.
$$

此映射交換兩個座標，面積不變；普通面積積分使用 $|\det J_T|=1$。若計算帶方向的積分，負號則反映方向反轉，不能忽略。

## 實作與程式

以下使用 Python 標準庫，對仿射例子做解析計算與中點格點求積。程式不安裝套件、不使用 GPU；有限格點只能核對特定案例，不能證明一般換變數定理。

```python
from math import isfinite

def determinant_2x2(a, b, c, d):
    """回傳 [[a,b],[c,d]] 的行列式。"""
    values = (a, b, c, d)
    if not all(isfinite(x) for x in values):
        raise ValueError("矩陣元素必須是有限實數")
    return a * d - b * c

def affine_rectangle_area(a, b, c, d, width, height):
    """可逆線性映射下，矩形像的普通面積。"""
    if width < 0 or height < 0:
        raise ValueError("邊長不可為負")
    det = determinant_2x2(a, b, c, d)
    if det == 0:
        raise ValueError("奇異映射不能用此一對一面積公式")
    return abs(det) * width * height

def midpoint_integral_affine_square(a, b, c, d, n):
    """在 [0,1]^2 中點格點積分常數 |det A|。"""
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n 必須是正整數")
    det = determinant_2x2(a, b, c, d)
    if det == 0:
        raise ValueError("奇異映射不能用此一對一換元")

    total = 0.0
    h = 1.0 / n
    for i in range(n):
        for j in range(n):
            # 中點座標用來確認格點落在單位正方形內。
            u = (i + 0.5) * h
            v = (j + 0.5) * h
            if not (0.0 < u < 1.0 and 0.0 < v < 1.0):
                raise ArithmeticError("中點落在預期區域之外")
            total += abs(det) * h * h
    return total

if __name__ == "__main__":
    a, b, c, d = 2.0, 1.0, 0.0, 1.0
    exact = affine_rectangle_area(a, b, c, d, 1.0, 1.0)
    estimate = midpoint_integral_affine_square(a, b, c, d, 10)
    print("det =", determinant_2x2(a, b, c, d))
    print("解析面積 =", exact)
    print("中點格點估計 =", estimate)

    reversed_det = determinant_2x2(0.0, 1.0, 1.0, 0.0)
    print("交換座標的 det =", reversed_det)
    print("交換座標的面積因子 =", abs(reversed_det))
```

預期輸出為行列式 $2$、解析面積 $2$、格點估計 $2$。交換座標的行列式預期為 $-1$，面積因子則為 $1$。程式未實際執行；上述是依公式與程式邏輯推得的預期結果。

若改成非線性映射或非定值被積函數，必須把 $f(T(u,v))|\det J_T(u,v)|$ 一起納入格點加權，並另行分析求積誤差。有限格點的穩定結果不能推出積分公式在連續域成立。

## 測試與預期結果

下列測試未實際執行；預期結果依照程式邏輯判斷。

| 類別 | 測試輸入 | 預期結果 | 意義 |
|---|---|---|---|
| 正常 | $A=\begin{pmatrix}2&1\\0&1\end{pmatrix}$，單位正方形 | 面積 $2$，格點估計 $2$ | 與手算一致 |
| 邊界 | $n=1$，邊長為零 | 格點合法；面積為 $0$ | 零面積區域不應產生負面積 |
| 方向反轉 | $A=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | 行列式 $-1$，面積因子 $1$ | 檢查絕對值的使用 |
| 故障 | $A=\begin{pmatrix}1&2\\2&4\end{pmatrix}$ | 拋出奇異映射錯誤 | 不可套用可逆換元公式 |
| 故障 | $n=0$ 或負數 | 拋出 `ValueError` | 格點數必須為正整數 |
| 故障 | $n=1.5$ | 拋出 `ValueError` | 非整數格點數須在進入迴圈前拒絕 |
| 故障 | 邊長為負 | 拋出 `ValueError` | 拒絕不合法的幾何輸入 |

奇異矩陣把二維區域壓到直線或點，其像的二維面積為零；但這不表示一般的可逆換變數定理可以套用到它上面。若被積函數、積分維度或像空間改變，必須重新定義所求的量。

## 反例與常見陷阱

1. **把 $\det J_T$ 當成普通面積因子。** 映射方向反轉時，行列式可為負；普通面積元素非負，應使用 $|\det J_T|$。有向積分則須依積分對象與取向慣例處理符號。

2. **局部可逆就假設全域一對一。** 極座標在非零半徑處局部正則，但讓角度跑滿兩圈會把大部分圓環覆蓋兩次。直接積分兩圈會把普通面積算兩遍；應限制角度區間，或明確計入覆蓋次數。

3. **忽略奇異點與邊界。** 在 $r=0$，極座標 Jacobian 為零，而且多個參數點代表同一原點。對一般圓盤面積積分，該單點不影響結果；若奇異性或集合測度使相關集合不能忽略，就須另行驗證。

4. **把 Jacobian 當作整個區域的縮放常數。** 只有仿射映射的 Jacobian 才是常數。非線性映射須在每個參數點使用 $|\det J_T(u)|$，不可只取中心點值乘上總面積。

5. **以數值抽樣取代定理條件。** 隨機點或有限網格看不到所有可能的折疊、自交與零行列式點。數值核對能發現特定實作錯誤，不能證明映射一對一或餘項具有所需性質。

6. **在座標變換中漏掉單位。** 若輸入參數的分量有不同單位，Jacobian 元素的單位是輸出分量單位除以輸入分量單位。行列式帶有相應的合成單位，不能把它誤當成無因次常數。

## AI、幾何與養殖案例

設一個合成感測器座標模型，把參數 $u$（單位 m）和 $\theta$（單位 rad）映到平面位置：

$$
T(u,\theta)=\bigl(u\cos\theta,\ u\sin\theta\bigr).
$$

兩個輸出分量單位都是 m。由於弧度在分析中視作無因次量，Jacobian 為

$$
J_T(u,\theta)=
\begin{pmatrix}
\cos\theta&-u\sin\theta\\
\sin\theta&u\cos\theta
\end{pmatrix},
\qquad
\det J_T(u,\theta)=u.
$$

第一欄元素的單位為 m/m，第二欄元素的單位為 m/rad。將弧度視為無因次量時，行列式的單位是 m；再乘上單位為 m 的 $du$ 與無因次的 $d\theta$，面積元素 $u\,du\,d\theta$ 的單位才是平方公尺。當 $u>0$ 且角度範圍不重複時，參數小區塊的面積約為 $u\,du\,d\theta$。當 $u=0$，角度無法識別位置，這不只是代數上的零行列式，也反映參數化在中心的退化。

若 AI 系統協助整理這類合成感測座標的校準資料，可以報告參數域、映射公式、單位、Jacobian、覆蓋範圍、奇異點和數值檢查方式。它不能只用幾個量測點就宣稱連續域上沒有重複覆蓋；模型校準也不等於現場驗證。本文案例是數學示範，不代表任何養殖設備的校準規格。

在右手 $X/Y/Z$ 慣例中，若平面幾何以 $X$ 為水平軸、$Y$ 為縱深軸，影像顯示時可能把原點放在左上角並反轉某一軸。這會改變顯示座標的方向慣例，卻不能悄悄改寫數學映射的行列式符號。分析資料時應明列座標軸方向、原點位置、參數範圍與面積單位。AI 的角色是提供可稽核的證據摘要，不控制設備、不改變投餌或加藥設定。

## 習題

### 習題一：手算仿射換元

令 $T(u,v)=(3u-v,2u+v)$，而 $S=[0,1]\times[0,2]$。

1. 求 $J_T$ 與行列式。
2. 求 $T(S)$ 的面積。
3. 說明為什麼這裡可以直接用仿射換變數公式。

### 習題二：極座標與重複覆蓋

考慮 $T(r,\theta)=(r\cos\theta,r\sin\theta)$。

1. 計算 Jacobian 行列式。
2. 計算單位圓盤面積。
3. 若參數範圍取 $0\le r\le1,\ 0\le\theta\le4\pi$，積分 $\iint 1\,dx\,dy$ 的參數積分會得到多少？解釋差異。

### 習題三：程式與反例

考慮 $A=\begin{pmatrix}1&2\\2&4\end{pmatrix}$。

1. 計算 $\det A$。
2. 解釋它為何不能用本章的可逆仿射換元公式計算像區域的二維面積縮放。
3. 若用程式檢查 $\det A=0$，此計算能證明一般非線性映射具有一對一性嗎？說明理由。

### 習題四：整合判斷

令

$$
T(r,\theta)=(2r\cos\theta,\ r\sin\theta),
\qquad 0\le r\le1,\quad 0\le\theta\le2\pi.
$$

1. 求 Jacobian 行列式與面積因子。
2. 描述參數域的像。
3. 計算像區域面積。
4. 指出一對一條件在哪些點或邊界上失效，並說明其對面積積分的影響。

## 習題解答

### 解答一

$$
J_T=
\begin{pmatrix}
3&-1\\
2&1
\end{pmatrix},
\qquad
\det J_T=3\cdot1-(-1)\cdot2=5.
$$

映射可逆，因為行列式非零；又是仿射映射，因此在 $S$ 上一對一。原矩形面積為 $1\cdot2=2$，像面積為

$$
|5|\cdot2=10.
$$

### 解答二

$$
J_T=
\begin{pmatrix}
\cos\theta&-r\sin\theta\\
\sin\theta&r\cos\theta
\end{pmatrix},
\qquad
\det J_T=r.
$$

單位圓盤面積為

$$
\int_0^{2\pi}\int_0^1r\,dr\,d\theta
=2\pi\cdot\frac12
=\pi.
$$

若讓角度跑到 $4\pi$，除中心外每個圓盤點被覆蓋兩次，參數積分得到

$$
\int_0^{4\pi}\int_0^1r\,dr\,d\theta=2\pi.
$$

目標圓盤的幾何面積仍為 $\pi$；多出的因子 $2$ 是重複覆蓋所致。此參數化不符合整段參數域上的一對一假設。

### 解答三

$$
\det A=1\cdot4-2\cdot2=0.
$$

矩陣奇異，兩個欄向量線性相關，平面被壓到一條直線上；它沒有可逆線性反變換，所以不能套用可逆仿射換元定理。計算這個特定矩陣的行列式，只能判斷這個線性映射退化，不能證明一般非線性映射一對一。一對一是整體域上的性質，必須由映射的定義與適用條件證明；有限計算或有限抽樣不足以涵蓋所有點。

### 解答四

$$
J_T=
\begin{pmatrix}
2\cos\theta&-2r\sin\theta\\
\sin\theta&r\cos\theta
\end{pmatrix},
\qquad
\det J_T=2r(\cos^2\theta+\sin^2\theta)=2r.
$$

面積因子為 $2r$。參數域描述半長軸為 $2$、半短軸為 $1$ 的橢圓形區域。其面積為

$$
\begin{aligned}
\operatorname{area}(T(S))
&=\int_0^{2\pi}\int_0^1 2r\,dr\,d\theta\\
&=2\pi.
\end{aligned}
$$

在 $r=0$ 時所有角度映到原點；在 $\theta=0$ 和 $\theta=2\pi$ 的邊界上，兩端角度表示同一條射線。因此參數化不是整個閉參數域上的一對一映射。然而重複點位於邊界或中心，對普通面積積分而言是零面積集合；公式可在去除這些邊界集合的區域上套用，所得面積為 $2\pi$。

## 本章小結

- 可微映射在小尺度上由 Jacobian 線性化；行列式絕對值是局部普通面積或體積縮放因子。
- 仿射映射的 Jacobian 為常數；非線性映射則須在參數點逐點使用 $|\det J_T|$。
- 普通積分用 $|\det J_T|$；有向積分是否保留行列式符號，取決於所積分的對象與取向慣例。
- 經典換變數公式需要足夠正則性及適當的一對一條件；局部可逆不等於全域一對一。
- 極座標的 Jacobian 為 $r$；中心退化與角度端點重複須在定理假設中交代。
- 數值程式可核對特定案例、發現實作錯誤；有限網格不能證明一般定理、一對一性或極限性質。

## 參考來源

- [A1] Jiří Lebl，*Basic Analysis*（作者目錄與教材入口）：https://www.jirka.org/ra/
- [A2] MIT OpenCourseWare，*18.100A Real Analysis*：https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/
- [A3] MIT OpenCourseWare，*18.02SC Multivariable Calculus*：https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/

本章採用的換變數定理是多變量微積分的標準結果，上述來源可供進一步閱讀。程式是本章自足示例，未執行。