# 第22章 線積分、勢函數與拓撲界線

## 學習目標與先備知識

本章研究向量場沿曲線的線積分，以及三個彼此密切但不可任意混同的概念：

1. 向量場是否為某個標量函數的梯度；
2. 線積分是否只依賴端點；
3. 每一條閉曲線上的環流是否皆為零。

完成本章後，讀者應能：

- 正確定義分段 $C^1$ 曲線及向量場線積分；
- 分辨標量弧長積分 $\int_\gamma f\,ds$ 與有向線積分 $\int_\gamma F\cdot dr$；
- 證明梯度場基本定理，並由勢函數計算線積分；
- 在合適的路徑連通域上，說明「存在勢函數」、「路徑獨立」及「所有閉路積分為零」的等價性；
- 理解旋度為零通常只是局部資訊，拓撲條件負責把局部資訊提升為全域結論；
- 以穿孔平面上的向量場說明「旋度為零但不保守」；
- 說明單連通是常用的充分條件，卻不是保守場存在的必要條件；
- 使用參數化與複合梯形公式近似閉路積分，並辨認離散誤差、奇異點及錯誤取向；
- 在感測與幾何案例中檢查單位、尺度、域及模型有效範圍。

先備知識包括：多變量微分、梯度與 Jacobian、鏈式法則、分段連續函數的單變量積分、曲線參數化，以及開集、連通與單連通的基本概念。本章主要在 $\mathbb R^2$ 與 $\mathbb R^3$ 中工作。

---

## 問題與直覺

設 $F$ 是平面上的力場，粒子沿曲線 $\gamma$ 從 $A$ 移到 $B$。做功為

$$
\int_\gamma F\cdot dr.
$$

一般而言，做功可能依賴實際路徑。例如阻力或旋轉流場會記錄粒子繞行的方式。若存在標量函數 $\phi$ 使

$$
F=\nabla\phi,
$$

則鏈式法則給出

$$
F(\gamma(t))\cdot\gamma'(t)
=\nabla\phi(\gamma(t))^T\gamma'(t)
=\frac{d}{dt}\phi(\gamma(t)).
$$

積分後只剩端點差：

$$
\int_\gamma F\cdot dr=\phi(B)-\phi(A).
$$

因此，梯度場像是「已有全域高度函數的斜率資料」。沿任何閉路回到原點後，高度淨變化必為零。

旋度零則是局部的相容性條件。在二維中，若

$$
F=(P,Q),\qquad
\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}=0,
$$

可直觀理解為無窮小環路的淨環流消失。然而，即使每個足夠小的環路都沒有環流，繞過域中無法縮掉的孔洞時仍可能累積非零環流。局部微分條件與全域拓撲條件因此必須分開。

數值取樣也有相同界線。檢查有限條閉路只能提供數值證據；它能推翻「所有閉路積分皆為零」，卻不能證明無限多條閉路皆成立。真正的全域結論必須來自勢函數、定理條件或其他嚴格論證。

---

## 定義、定理與推導

### 1. 曲線與線積分

**定義 22.1（分段 $C^1$ 曲線）**  
映射 $\gamma:[a,b]\to\mathbb R^n$ 若連續，且存在有限分割

$$
a=t_0<t_1<\cdots<t_k=b
$$

使 $\gamma$ 在每個子區間上為 $C^1$，稱為分段 $C^1$ 曲線。

設 $U\subset\mathbb R^n$，$F:U\to\mathbb R^n$ 連續，且 $\gamma([a,b])\subset U$。向量場的有向線積分定義為

$$
\int_\gamma F\cdot dr
=
\int_a^b F(\gamma(t))^T\gamma'(t)\,dt.
$$

若反向參數化為 $\bar\gamma(t)=\gamma(a+b-t)$，則

$$
\int_{\bar\gamma}F\cdot dr=-\int_\gamma F\cdot dr.
$$

相對地，標量弧長積分為

$$
\int_\gamma f\,ds
=
\int_a^b f(\gamma(t))\|\gamma'(t)\|_2\,dt,
$$

反向後不改變符號。兩者不可混用：前者含有方向，後者以非負弧長元素加權。

### 2. 勢函數與保守場

**定義 22.2（勢函數與保守場）**  
設 $U\subset\mathbb R^n$ 為開集。若存在 $\phi\in C^1(U)$ 使

$$
F=\nabla\phi,
$$

則稱 $\phi$ 為 $F$ 的勢函數，$F$ 為保守場或梯度場。

若 $U$ 連通，同一向量場的任意兩個勢函數只相差常數；若 $U$ 不連通，則每個連通分支可有不同常數。

### 3. 梯度場基本定理

**定理 22.1（線積分基本定理）**  
設 $U\subset\mathbb R^n$ 為開集，$\phi\in C^1(U)$，$F=\nabla\phi$。若 $\gamma:[a,b]\to U$ 是分段 $C^1$ 曲線，則

$$
\int_\gamma F\cdot dr
=
\phi(\gamma(b))-\phi(\gamma(a)).
$$

**證明。**  
先假設 $\gamma$ 為 $C^1$。由多變量鏈式法則，

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
&=\int_a^b\nabla\phi(\gamma(t))^T\gamma'(t)\,dt\\
&=\int_a^b\frac{d}{dt}\phi(\gamma(t))\,dt\\
&=\phi(\gamma(b))-\phi(\gamma(a)).
\end{aligned}
$$

若 $\gamma$ 只分段 $C^1$，則在每個子區間套用上述結果，再將端點差相加。中間端點的勢函數值兩兩消去，仍得到相同公式。證畢。

這個證明使用鏈式法則與微積分基本定理，不是由有限路徑實驗歸納而來。

### 4. 路徑獨立與閉路積分

**定義 22.3（路徑獨立）**  
若對任意 $A,B\in U$，以及任意兩條由 $A$ 到 $B$、位於 $U$ 中的分段 $C^1$ 曲線 $\gamma_1,\gamma_2$，皆有

$$
\int_{\gamma_1}F\cdot dr
=
\int_{\gamma_2}F\cdot dr,
$$

則稱線積分在 $U$ 上路徑獨立。

若域不具路徑連通性，上述性質應逐一在每個路徑連通分支中討論。

**命題 22.2（保守性、路徑獨立與閉路零環流）**  
設 $U\subset\mathbb R^n$ 為開且路徑連通的集合，$F:U\to\mathbb R^n$ 連續。則下列敘述等價：

1. 存在 $\phi\in C^1(U)$ 使 $F=\nabla\phi$；
2. $\int_\gamma F\cdot dr$ 路徑獨立；
3. 對每條位於 $U$ 中的分段 $C^1$ 閉曲線 $\Gamma$，有 $\oint_\Gamma F\cdot dr=0$。

**證明。**

- $(1)\Rightarrow(2)$：由定理 22.1，任一路徑的積分都是 $\phi(B)-\phi(A)$。
- $(2)\Rightarrow(3)$：閉曲線起點與終點相同，所以其積分等於常值路徑的積分，即零。
- $(3)\Rightarrow(2)$：令 $\gamma_1,\gamma_2$ 都從 $A$ 到 $B$。先走 $\gamma_1$，再反向走 $\gamma_2$，形成閉曲線。故
  $$
  0=\int_{\gamma_1}F\cdot dr-\int_{\gamma_2}F\cdot dr.
  $$
- $(2)\Rightarrow(1)$：固定 $x_0\in U$，定義
  $$
  \phi(x)=\int_{x_0}^{x}F\cdot dr,
  $$
  其中積分取任一條域內路徑；路徑獨立使定義良好。因 $U$ 開，對每個 $x\in U$ 及座標方向 $e_i$，足夠小的線段 $x+se_i$ 留在 $U$。於是
  $$
  \frac{\phi(x+he_i)-\phi(x)}{h}
  =
  \frac1h\int_0^h F_i(x+se_i)\,ds.
  $$
  由 $F_i$ 的連續性，令 $h\to0$ 得
  $$
  \frac{\partial\phi}{\partial x_i}(x)=F_i(x).
  $$
  因所有 $F_i$ 連續，$\phi\in C^1(U)$ 且 $\nabla\phi=F$。證畢。

### 5. 旋度零：必要條件與充分條件

在三維中，對 $F=(P,Q,R)\in C^1(U)$ 定義

$$
\nabla\times F
=
\begin{pmatrix}
R_y-Q_z\\
P_z-R_x\\
Q_x-P_y
\end{pmatrix}.
$$

在二維中常把 $F=(P,Q)$ 的標量旋度寫為

$$
\operatorname{curl}F=Q_x-P_y.
$$

若 $F=\nabla\phi$ 且 $\phi\in C^2(U)$，由混合偏導相等，

$$
\nabla\times F=0.
$$

因此，在此正則性下，旋度零是保守場的**必要條件**。

反方向需要域條件。一個常用版本如下。

**定理 22.3（星形域上的 Poincaré 型結論）**  
設 $U\subset\mathbb R^n$ 為開星形集，$F\in C^1(U;\mathbb R^n)$，且

$$
\frac{\partial F_i}{\partial x_j}
=
\frac{\partial F_j}{\partial x_i}
\quad\text{對所有 }i,j.
$$

則 $F$ 在 $U$ 上存在 $C^2$ 勢函數。

星形表示存在 $a\in U$，使每個 $x\in U$ 與 $a$ 之間的整條線段都位於 $U$。可定義

$$
\phi(x)=\int_0^1 F(a+t(x-a))^T(x-a)\,dt,
$$

再利用微分與積分交換及偏導對稱證明 $\nabla\phi=F$。交換步驟依賴 integrand 對 $(t,x)$ 的連續可微性。

更一般地，在合適的開且單連通域上，$C^1$ 無旋場也是保守場。此處的「單連通」排除了無法在域內連續縮成一點的閉路。然而：

> 單連通是常用充分條件，不是保守場存在的必要條件。

例如穿孔平面不是單連通，但函數 $\phi(x,y)=x^2+y^2$ 在其上仍有梯度場 $(2x,2y)$。域的孔洞只表示某些無旋場可能不保守，不表示每個場都不保守。

---

## 逐步手算例題

### 例 1：由勢函數直接算線積分

令

$$
F(x,y)=
\begin{pmatrix}
2xy+3\\
x^2+2y
\end{pmatrix}.
$$

先尋找勢函數。由

$$
\phi_x=2xy+3
$$

對 $x$ 積分：

$$
\phi(x,y)=x^2y+3x+g(y).
$$

再對 $y$ 微分：

$$
\phi_y=x^2+g'(y).
$$

與 $F$ 的第二分量比較可得 $g'(y)=2y$，故可取

$$
\phi(x,y)=x^2y+3x+y^2.
$$

計算由 $A=(0,0)$ 到 $B=(2,1)$ 的任一路徑積分：

$$
\int_\gamma F\cdot dr
=
\phi(2,1)-\phi(0,0)
=
4+6+1=11.
$$

核對折線路徑：先由 $(0,0)$ 到 $(2,0)$，再到 $(2,1)$。

第一段 $\gamma_1(t)=(t,0)$，$0\le t\le2$：

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

總和為 $11$，與勢函數法一致。

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

在 $U$ 上計算：

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

但取逆時針單位圓

$$
\gamma(t)=(\cos t,\sin t),\qquad 0\le t\le2\pi.
$$

有

$$
\gamma'(t)=(-\sin t,\cos t),
$$

且

$$
F(\gamma(t))=(-\sin t,\cos t).
$$

因此

$$
F(\gamma(t))^T\gamma'(t)=1,
$$

故

$$
\oint_\gamma F\cdot dr
=
\int_0^{2\pi}1\,dt
=
2\pi\ne0.
$$

由命題 22.2，$F$ 不可能在整個 $U$ 上具有單值 $C^1$ 勢函數。局部可寫成角度函數，但角度繞原點一圈增加 $2\pi$，無法成為整個穿孔平面上的單值連續函數。

若將圓改為順時針，積分變成 $-2\pi$。若圓繞原點兩圈，積分變成 $4\pi$。這些值反映閉路的繞行方向與繞數。

### 例 3：單連通不是必要條件

仍令 $U=\mathbb R^2\setminus\{0\}$，但改取

$$
G(x,y)=(2x,2y).
$$

雖然 $U$ 不是單連通，仍有全域勢函數

$$
\phi(x,y)=x^2+y^2.
$$

所以對任何域內閉路 $\Gamma$，

$$
\oint_\Gamma G\cdot dr=0.
$$

不能從「域有孔」推出「場不保守」；必須檢查特定向量場。

---

## 實作與程式

下列 NumPy 程式以複合梯形公式近似參數化曲線的線積分。程式只需 CPU；此處未執行，後文數值均為預期結果。

```python
import numpy as np

def line_integral(field, curve, dcurve, a, b, n=4096):
    """
    近似 integral F(gamma(t)) dot gamma'(t) dt。
    field: 接受 shape (..., 2)，回傳相同前置 shape 加最後維 2
    curve, dcurve: 接受一維 t，回傳 shape (len(t), 2)
    """
    if not isinstance(n, int) or n < 2:
        raise ValueError("n 必須是至少 2 的整數")
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        raise ValueError("端點必須有限且 a != b")

    t = np.linspace(a, b, n + 1)
    points = np.asarray(curve(t), dtype=float)
    tangents = np.asarray(dcurve(t), dtype=float)

    expected = (n + 1, 2)
    if points.shape != expected or tangents.shape != expected:
        raise ValueError(f"curve 與 dcurve 必須回傳 shape {expected}")

    values = np.asarray(field(points), dtype=float)
    if values.shape != expected:
        raise ValueError(f"field 必須回傳 shape {expected}")
    if not (np.all(np.isfinite(points))
            and np.all(np.isfinite(tangents))
            and np.all(np.isfinite(values))):
        raise FloatingPointError("曲線接觸奇異點或產生非有限數值")

    integrand = np.sum(values * tangents, axis=1)
    return np.trapezoid(integrand, t)

def circle(radius=1.0, turns=1, clockwise=False):
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius 必須是有限正數")
    sign = -1.0 if clockwise else 1.0

    def curve(t):
        u = sign * turns * t
        return np.column_stack((radius * np.cos(u),
                                radius * np.sin(u)))

    def dcurve(t):
        u = sign * turns * t
        factor = sign * turns * radius
        return np.column_stack((-factor * np.sin(u),
                                factor * np.cos(u)))
    return curve, dcurve

def potential_field(points):
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack((2.0 * x * y + 3.0,
                            x * x + 2.0 * y))

def vortex_field(points):
    x = points[:, 0]
    y = points[:, 1]
    r2 = x * x + y * y
    if np.any(r2 <= 1.0e-24):
        raise FloatingPointError("vortex_field 在原點未定義")
    return np.column_stack((-y / r2, x / r2))

if __name__ == "__main__":
    c1, dc1 = circle(radius=1.0)
    c2, dc2 = circle(radius=2.5)
    cw, dcw = circle(radius=1.0, clockwise=True)

    conservative = line_integral(
        potential_field, c1, dc1, 0.0, 2.0 * np.pi
    )
    vortex_1 = line_integral(
        vortex_field, c1, dc1, 0.0, 2.0 * np.pi
    )
    vortex_25 = line_integral(
        vortex_field, c2, dc2, 0.0, 2.0 * np.pi
    )
    vortex_cw = line_integral(
        vortex_field, cw, dcw, 0.0, 2.0 * np.pi
    )

    print("conservative closed loop =", conservative)
    print("vortex radius 1          =", vortex_1)
    print("vortex radius 2.5        =", vortex_25)
    print("vortex clockwise         =", vortex_cw)
```

程式中的 NumPy 陣列採 shape $(N,2)$ 儲存 $N$ 個平面點；每個橫列是一個點。這只是批次資料配置，不應誤稱為數學上的列向量表示。單一幾何向量在公式中仍視為 $2\times1$ 的 column vector。

若舊版 NumPy 沒有 `np.trapezoid`，可改用 `np.trapz`；這是 API 相容問題，不影響數學定義。

---

## 測試與預期結果

以下結果均為解析推導支持下的**預期**，不是宣稱已執行程式。

### 正常測試

1. `potential_field` 沿單位圓：
   $$
   \oint_\gamma\nabla\phi\cdot dr=0.
   $$
   預期輸出接近浮點零。

2. `vortex_field` 沿逆時針單位圓：
   $$
   \oint_\gamma F\cdot dr=2\pi.
   $$
   預期約為 `6.283185307179586`。

3. 渦旋場沿半徑 $2.5$ 的逆時針圓：
   積分仍為 $2\pi$。半徑改變不改變繞數。

4. 渦旋場沿順時針單位圓：
   預期為 $-2\pi$。

對這些圓形參數化，渦旋場的 integrand 理論上恰為常數，因此梯形公式除浮點捨入外應非常準確；這種特殊精確性不可泛化到任意曲線或任意場。

### 邊界測試

- 半徑很小但仍遠大於浮點奇異閾值時，解析積分仍為 $2\pi$；然而場值大小約為 $1/r$，中間量放大，數值條件可能惡化。
- `turns=2` 時預期為 $4\pi$。
- 若閉曲線不包圍原點，渦旋場積分預期為零；但需要確保整條曲線不穿過原點。

### 故障測試

- `radius=0`：`circle` 應丟出 `ValueError`。
- `n=1`：`line_integral` 應丟出 `ValueError`。
- 曲線穿過原點：`vortex_field` 應丟出 `FloatingPointError`，因向量場在該點沒有定義。
- `curve` 回傳錯誤 shape：應丟出 `ValueError`，避免 NumPy broadcasting 悄悄產生無意義結果。
- 只提供曲線點卻提供錯誤切向量，程式未必能自動識別；這屬於模型輸入錯誤。可另用有限差分核對 `dcurve`，但有限差分仍只是診斷工具。

---

## 反例與常見陷阱

### 1. 旋度零不自動推出保守

穿孔平面渦旋場是核心反例。其旋度在每個域內點皆為零，但繞原點閉路的積分是 $2\pi$。忽略域會把局部微分條件錯當成全域結論。

### 2. 單連通不是必要條件

單連通域配合適當正則性，可保證無旋場保守；但一個已知梯度場可定義在非單連通域上。必要條件是該場確實有勢函數，而不是域必須單連通。

### 3. 只查混合偏導可能漏掉奇異點

對渦旋場形式化計算 $Q_x-P_y=0$ 時，必須先寫明域排除原點。不能在未定義點談導數，也不能把曲線積分穿過奇異點當作普通線積分。

### 4. 勢函數的「不定積分常數」可能依賴其他變數

由 $\phi_x=P$ 積分時應寫

$$
\phi(x,y)=\int P(x,y)\,dx+g(y),
$$

而不是立即加一個普通常數。遺漏 $g(y)$ 常導致錯誤判定。

### 5. 閉路數值接近零不是證明

離散化只測試有限曲線與有限節點。即使許多測試都接近零，也可能漏掉包圍孔洞的曲線。反之，一條經可靠解析與數值核對後得到非零積分的閉路，足以反駁保守性。

### 6. 取向與參數速度

反向曲線會使 $\int F\cdot dr$ 變號。只要重參數化保持方向且足夠正則，積分值不變；若參數化重複繞行，積分會按繞行次數累加。

### 7. 旋度零的正則性不可省略

由梯度場推出混合偏導相等，通常要求勢函數至少具有連續二階偏導，或使用更精細的弱條件。不能只因一階偏導符號看似可交換就直接宣告相等。

---

## AI、幾何與養殖案例

### 1. 幾何與學習模型中的勢函數

許多最佳化演算法從損失函數 $L(\theta)$ 取得梯度 $\nabla L(\theta)$。若某個向量更新場聲稱是精確梯度，理論上沿參數空間中的任意閉路應有

$$
\oint\nabla L(\theta)\cdot d\theta=0.
$$

在自動微分、代理模型或資料驅動向量場中，可用小閉路積分作一致性診斷。若穩定得到顯著非零值，可能原因包括：

- 學得的場本來就不是梯度場；
- 不同路段使用了不一致的模型或正規化；
- 數值誤差、隨機估計或有限差分誤差；
- 路徑穿過不可微區域；
- 參數空間刪除了奇異集合，產生拓撲障礙。

但閉路測試接近零不證明存在全域損失函數。若要主張全域保守性，仍需解析結構與域條件。

### 2. 合成養殖感測校準

考慮完全合成的兩參數校準座標

$$
z=
\begin{pmatrix}
u\\v
\end{pmatrix},
$$

其中 $u=(T-T_0)/s_T$ 是無因次溫度偏差，$s_T$ 單位為攝氏度；$v=(S-S_0)/s_S$ 是無因次鹽度偏差，$s_S$ 單位為 PSU。令合成校準代價

$$
\Phi(u,v)=\frac12u^2+uv+2v^2.
$$

則

$$
\nabla_z\Phi=
\begin{pmatrix}
u+v\\
u+4v
\end{pmatrix}.
$$

在無因次座標中可直接使用 Euclidean 內積。返回物理變數時，

$$
\frac{\partial\Phi}{\partial T}
=
\frac{u+v}{s_T},
\qquad
\frac{\partial\Phi}{\partial S}
=
\frac{u+4v}{s_S}.
$$

兩個分量的單位分別是「代價／攝氏度」與「代價／PSU」，不可把它們未加權地視為相同物理方向。尺度選擇會改變梯度的數值分量，但不改變同一物理微分

$$
d\Phi
=
\frac{\partial\Phi}{\partial T}\,dT
+
\frac{\partial\Phi}{\partial S}\,dS.
$$

若校準演算法沿兩條參數路徑從同一初值移到同一終值，精確梯度線積分應得到相同代價差。非零閉路殘差可作資料管線或梯度實作的稽核訊號，但不能直接解釋成感測器故障，更不能據此自動控制投餌、加藥或設備。模型校準不等於現場驗證；本案例中的 agent 只整理證據。

---

## 習題

### 習題 1：手算

令

$$
F(x,y)=(e^x\cos y,\,-e^x\sin y).
$$

1. 求一個勢函數。
2. 計算從 $(0,0)$ 到 $(1,\pi)$ 的任意分段 $C^1$ 路徑積分。
3. 說明答案為何與路徑無關。

### 習題 2：程式

修改本章程式，對橢圓

$$
\gamma(t)=(2\cos t,\sin t)
$$

計算渦旋場的閉路積分。再反向參數化並比較結果。列出預期值及至少一個故障測試。

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
4. 說明為何即使公式在穿孔平面上也有意義，本題選用的域仍使勢函數論證特別直接。

### 習題 5：必要條件與資料診斷

某離散系統在三條閉路上得到積分近似值 $10^{-8}$、$-3\times10^{-7}$ 與 $0.42$。有人宣稱前兩條已證明向量場保守，第三條只是雜訊。分析此說法需要哪些額外資訊，並說明目前可得到的嚴格結論。

---

## 習題解答

### 解答 1

由第一分量積分：

$$
\phi(x,y)=e^x\cos y+g(y).
$$

對 $y$ 微分：

$$
\phi_y=-e^x\sin y+g'(y).
$$

與第二分量比較得 $g'(y)=0$，故可取

$$
\phi(x,y)=e^x\cos y.
$$

因此

$$
\int_\gamma F\cdot dr
=
\phi(1,\pi)-\phi(0,0)
=
-e-1.
$$

因 $F=\nabla\phi$ 且 $\phi\in C^\infty(\mathbb R^2)$，由線積分基本定理，答案只依賴端點。

### 解答 2

橢圓逆時針繞原點一次，故預期積分為 $2\pi$。可加入：

```python
def ellipse(t):
    return np.column_stack((2.0 * np.cos(t), np.sin(t)))

def dellipse(t):
    return np.column_stack((-2.0 * np.sin(t), np.cos(t)))

value = line_integral(
    vortex_field, ellipse, dellipse, 0.0, 2.0 * np.pi
)
print(value)
```

反向可令 $\bar\gamma(t)=\gamma(2\pi-t)$，預期值為 $-2\pi$。與圓不同，此時 integrand 不必為常數，因此結果只會數值接近解析值。

故障測試可將橢圓平移，使其穿過原點，例如 $\gamma(t)=(1+\cos t,\sin t)$；在 $t=\pi$ 接觸原點，應拒絕積分或回報奇異點。僅靠格點未必剛好取到 $t=\pi$，故穩健實作還應做幾何最小距離檢查或確保節點包含已知奇異參數。

### 解答 3

敘述為假。取

$$
U=\mathbb R^2\setminus\{0\},
\qquad
\phi(x,y)=x^2+y^2.
$$

雖然 $U$ 非單連通，但

$$
F=\nabla\phi=(2x,2y)
$$

在 $U$ 上是保守場。單連通是「每個適當正則的無旋場皆保守」的一種充分域條件，不是「某個特定場保守」的必要條件。

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
P_y=-\frac{2xy}{(x^2+y^2)^2},
$$

所以 $Q_x-P_y=0$。

注意

$$
\phi(x,y)=\frac12\log(x^2+y^2)
$$

滿足

$$
\nabla\phi=F.
$$

因此

$$
\int_\gamma F\cdot dr
=
\phi(1,1)-\phi(1,0)
=
\frac12\log2.
$$

此勢函數其實也可定義在整個穿孔平面，因 $x^2+y^2>0$。選取右半平面後，域還是凸集，因而是星形且單連通；無旋推出保守的定理可直接套用。不過本題既已明確構造勢函數，就不必把單連通當成必要論據。

### 解答 5

前兩個小值不能證明全域保守，因為只測了有限閉路，且未知積分方法、離散解析度、誤差界、曲線取向與場的正則性。第三個值 $0.42$ 是否只是雜訊，至少需要：

- 數值求積誤差估計與解析度加密結果；
- 向量場評估誤差或隨機變異；
- 閉路是否位於場的定義域；
- 曲線是否包圍奇異點或域中的孔洞；
- 重複試驗與尺度比較；
- 是否有可獨立驗證的勢函數。

若能證明第三條閉路的真實積分非零，即可嚴格推出場不保守。僅有目前三個近似數字，無法證明保守或非保守；但 $0.42$ 是必須調查的反證候選，不能在沒有誤差模型時任意歸為雜訊。

---

## 本章小結

向量場線積分把場值與曲線切向量配對，因而同時依賴場、路徑與取向。若 $F=\nabla\phi$，鏈式法則與微積分基本定理給出

$$
\int_\gamma F\cdot dr
=
\phi(\gamma(b))-\phi(\gamma(a)).
$$

在開且路徑連通的域上，對連續向量場而言，存在勢函數、路徑獨立及所有閉路積分為零彼此等價。對足夠光滑的梯度場，旋度零是必要條件；反方向則需要域的全域結構。星形或適當的單連通條件提供常用充分條件，但不是特定向量場保守的必要條件。

穿孔平面渦旋場展示了拓撲界線：旋度在每個域內點皆為零，繞孔閉路積分卻為 $2\pi$。另一方面，穿孔域上仍可存在普通梯度場。故「域有孔」與「場不保守」都不能單獨推出另一方。

數值閉路積分適合用來核對符號、取向、參數化與尋找反例，但有限採樣不能證明無限多條閉路的命題。可靠分析必須同時標明向量場的定義域、正則性、拓撲條件、積分取向與數值誤差。

---

## 參考來源

1. Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>  
2. MIT OpenCourseWare，*18.100A Real Analysis*：<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>  
3. MIT OpenCourseWare，*18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>

以上來源用於一般實分析、多變量微積分與向量場背景。線上課程頁面與教材版本可能更新；本章定理仍應以本文明列的域、連續性及可微性條件為準。