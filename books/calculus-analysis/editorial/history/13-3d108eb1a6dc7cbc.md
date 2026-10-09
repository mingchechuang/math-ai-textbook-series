# 第13章 Hessian、二階微分與Taylor展開

## 學習目標與先備知識

讀完本章，應能把 Hessian 理解為二階導數的矩陣表示，而不只是「把偏導數排成表格」；能區分二階方向導數、二階微分與有限步長差分；能在明確條件下寫出二階 Taylor 展開，並辨別 $o(\|h\|^2)$ 與 $O(\|h\|^3)$；還能檢查交叉項係數，以及用實驗觀察餘項，而不把有限次實驗當作極限定理的證明。

先備知識包括 Fréchet 可微、Jacobian、矩陣乘法、Euclidean 範數及單變量 Taylor 定理。本章只研究**局部二階近似**；Hessian 如何用於極值判別，留待下一章。除特別註明外，向量均為直立的列向量（column vector），$\|h\|$ 表示 Euclidean 範數。

## 問題與直覺

一階近似以切平面描述函數在 $x$ 附近的變化：
$$
f(x+h)\approx f(x)+\nabla f(x)^Th.
$$
它描述斜率，卻沒有描述斜率如何改變。二階近似再加入一個關於 $h$ 的二次型：
$$
f(x+h)\approx f(x)+\nabla f(x)^Th+\frac12h^TH_f(x)h.
$$
其中 $H_f(x)$ 是 Hessian。若 $h=(h_1,h_2)^T$，二次項展開為
$$
\frac12h^TH_fh
=\frac12H_{11}h_1^2
+\frac12(H_{12}+H_{21})h_1h_2
+\frac12H_{22}h_2^2.
$$
在 Hessian 對稱時，交叉項係數是 $H_{12}$，**不是** $\tfrac12H_{12}$。前面的 $\tfrac12$ 同時作用於矩陣乘積中出現的兩個交叉項。

「二次近似」也有精確的誤差意思。若誤差是 $o(\|h\|^2)$，則誤差除以 $\|h\|^2$ 會在 $h\to0$ 時趨於零；這並不自動給出誤差小於某個常數乘 $\|h\|^3$。兩者所需的正則性不同，不能從幾次漂亮的數值結果推斷較強結論。

## 定義、定理與推導

設 $U\subset\mathbb R^n$ 為開集，$f:U\to\mathbb R$。標量導數 $Df(x)$ 是 $1\times n$ 線性泛函；在 Euclidean 內積下，其表示為 $\nabla f(x)^T$，其中梯度是 $n\times1$。若 $Df$ 在 $x$ Fréchet 可微，則二階導數 $D^2f(x)$ 是把兩個輸入擾動映至實數的雙線性映射：
$$
D^2f(x)[u,v]=D(Df)(x)[u][v].
$$
方括號的次序指出：先以 $u$ 擾動導數，再將得到的線性泛函作用於 $v$。選定標準座標後，若相應偏導存在，
$$
(H_f(x))_{ij}=\frac{\partial^2f}{\partial x_i\partial x_j}(x),
\qquad
D^2f(x)[u,v]=u^TH_f(x)v.
$$
Hessian 因而是雙線性映射在所選座標下的矩陣，維度為 $n\times n$；二階方向導數 $D^2f(x)[h,h]$ 則是一個純量。對向量值映射，其二階導數有向量輸出，不能不加說明便稱為單一 Hessian 矩陣。

**定理一：混合偏導的對稱性（充分條件）。** 若 $f\in C^2(U)$，即所有二階偏導在開集 $U$ 連續，則對每個 $x\in U$，
$$
H_f(x)^T=H_f(x),\qquad D^2f(x)[u,v]=D^2f(x)[v,u].
$$
這是常用的充分條件，不是說「二階偏導只要在一點存在就必然相等」。此處引用混合偏導交換定理；其適用條件包含所述鄰域上的連續性。

**定理二：二階 Taylor 展開。** 若 $f\in C^2(U)$，且 $x\in U$，則當 $h\to0$ 且 $x+h\in U$ 時，
$$
f(x+h)=f(x)+\nabla f(x)^Th
+\frac12h^TH_f(x)h+r_2(h),
\qquad
\frac{r_2(h)}{\|h\|^2}\longrightarrow0.
$$
這個極限要求所有充分小的非零 $h$，不是只沿一條直線或一組取樣方向成立。

**小命題與完整證明：積分餘項及其二階界。** 假設 $f\in C^2(U)$，且線段 $\{x+th:0\leq t\leq1\}$ 包含於 $U$。記 $g(t)=f(x+th)$。則
$$
\begin{aligned}
f(x+h)={}&f(x)+\nabla f(x)^Th\\
&+\int_0^1(1-t)\,h^TH_f(x+th)h\,dt,
\end{aligned}
$$
並且，若線段上滿足 $\|H_f(x+th)-H_f(x)\|_2\leq\varepsilon$，則
$$
|r_2(h)|\leq\frac{\varepsilon}{2}\|h\|^2.
$$

*證明。* 鏈式法則給出 $g'(t)=\nabla f(x+th)^Th$ 及 $g''(t)=h^TH_f(x+th)h$。由微積分基本定理，
$$
g(1)-g(0)-g'(0)=\int_0^1\int_0^s g''(t)\,dt\,ds
=\int_0^1(1-t)g''(t)\,dt.
$$
這得到積分公式。扣除 $\tfrac12h^TH_f(x)h$，並利用 $\int_0^1(1-t)\,dt=\tfrac12$，可得
$$
r_2(h)=\int_0^1(1-t)\,
h^T\!\left[H_f(x+th)-H_f(x)\right]h\,dt.
$$
由算子範數不等式 $|h^TAh|\leq\|A\|_2\|h\|^2$，立刻得到所宣稱的界。證畢。

此證明也解釋定理二的餘項：$x$ 是開集內點，故充分短的線段仍在 $U$；$H_f$ 在 $x$ 連續，使上式中的 $\varepsilon$ 能隨 $\|h\|\to0$ 而趨零。因此所得恰是 $o(\|h\|^2)$。

**較強界的充分條件。** 若上述線段所在鄰域內的 Hessian 滿足
$$
\|H_f(y)-H_f(z)\|_2\leq L\|y-z\|,
$$
則積分餘項進一步給出
$$
|r_2(h)|\leq L\|h\|^3
\int_0^1t(1-t)\,dt
=\frac{L}{6}\|h\|^3.
$$
因此，局部 Lipschitz 的 Hessian 足以保證三階界；例如連續三階導數在適當的小閉球上有界時，可取得這類界。僅有 $C^2$ 不保證它。

## 逐步手算例題

**例一：完整 Hessian 與交叉項。** 令
$$
f(x,y)=x^2+3xy+2y^2+x^3,\qquad a=(1,-1)^T.
$$
逐項微分得
$$
\nabla f(x,y)=
\begin{pmatrix}2x+3y+3x^2\\3x+4y\end{pmatrix},
\qquad
H_f(x,y)=
\begin{pmatrix}2+6x&3\\3&4\end{pmatrix}.
$$
在 $a$，$f(a)=1-3+2+1=1$、$\nabla f(a)=(2,-1)^T$，且
$$
H_f(a)=\begin{pmatrix}8&3\\3&4\end{pmatrix}.
$$
令 $h=(s,t)^T$，二階項是
$$
\frac12h^TH_f(a)h
=\frac12(8s^2+6st+4t^2)
=4s^2+3st+2t^2.
$$
直接將 $(1+s,-1+t)$ 代入原函數並展開，可核對
$$
f(a+h)=1+2s-t+4s^2+3st+2t^2+s^3.
$$
因此此例的**精確**餘項是 $s^3$。方向不同會改變餘項：沿 $s=0$ 它恆為零，但這不表示所有方向都沒有三階誤差。

**例二：二階方向差分與有限步長誤差。** 令
$$
q(x,y)=x^2+2xy+3y^2,\qquad
v=(2,-1)^T.
$$
其 Hessian 恆為
$$
H_q=\begin{pmatrix}2&2\\2&6\end{pmatrix},
\quad
H_qv=(2,-2)^T,
\quad
v^TH_qv=6.
$$
在任意基點 $a$，中心二階方向差分恰為
$$
\frac{q(a+tv)-2q(a)+q(a-tv)}{t^2}=6
\quad(t\ne0),
$$
這是二次多項式的代數恆等式，與「一般光滑函數於有限 $t$ 必定精確」不同。譬如把第一例改沿 $v=(1,0)^T$、基點取 $(1,-1)^T$，中心差分中的三次項會留下依基點而定的二階貢獻；逐步縮小 $t$ 才是在檢查其趨勢。手算時先展開 $f(a\pm tv)$，再相加消去奇次的 $t$ 項，可避免誤把中心差分當成一階差分。

## 實作與程式

以下程式只使用 Python 標準庫，在 CPU 上計算兩個二變量多項式。它同時檢查完整 Hessian、交叉項、Taylor 餘項，以及中心二階方向差分。`hessian_fd` 用函數值重建矩陣：對角元使用中心二階差分，非對角元使用四角混合差分。這些公式在有限步長與有限精度下是**估計**，並非 Hessian 存在性的定義或證明。

```python
from math import hypot, isfinite

def f(x, y):
    return x*x + 3*x*y + 2*y*y + x*x*x

def grad(x, y):
    return (2*x + 3*y + 3*x*x, 3*x + 4*y)

def hessian(x, y):
    return ((2 + 6*x, 3.0), (3.0, 4.0))

def quadratic(H, v):
    s, t = v
    return s*(H[0][0]*s + H[0][1]*t) + \
           t*(H[1][0]*s + H[1][1]*t)

def taylor_error(a, h):
    x, y = a
    s, t = h
    g = grad(x, y)
    H = hessian(x, y)
    model = f(x, y) + g[0]*s + g[1]*t + 0.5*quadratic(H, h)
    return f(x+s, y+t) - model

def directional_fd(a, v, step):
    if not isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    x, y = a
    s, t = v
    return (f(x+step*s, y+step*t) - 2*f(x, y)
            + f(x-step*s, y-step*t)) / (step*step)

def hessian_fd(a, step):
    if not isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    x, y = a
    z = f(x, y)
    dxx = (f(x+step, y) - 2*z + f(x-step, y)) / step**2
    dyy = (f(x, y+step) - 2*z + f(x, y-step)) / step**2
    dxy = (f(x+step, y+step) - f(x+step, y-step)
           - f(x-step, y+step) + f(x-step, y-step)) / (4*step**2)
    return ((dxx, dxy), (dxy, dyy))

a = (1.0, -1.0)
v = (1.0, 2.0)
H = hessian(*a)
print("Hessian:", H)
print("v^T H v:", quadratic(H, v))
print("function-value Hessian estimate:", hessian_fd(a, 1e-3))
for step in (0.2, 0.1, 0.05):
    h = (step*v[0], step*v[1])
    error = taylor_error(a, h)
    scale = hypot(*h)
    print("step, error, error/norm^2:",
          step, error, error/(scale*scale))
print("directional estimate:", directional_fd(a, v, 1e-3))
```

此程式的輸入座標與函數值均視為無因次。若實際輸入帶不同單位，必須先指定各分量尺度，或逐一追蹤 Hessian 元素的「輸出單位／兩個輸入單位之積」；不能直接把混合單位的 Euclidean 長度當作物理誤差尺度。極小步長還可能使相近函數值相減而損失有效數字，故掃描步長應包含中等值，不以「越小越準」作無條件規則。

## 測試與預期結果

以下皆為**未執行程式的預期**，不是測試通過紀錄。

| 類別 | 操作 | 預期與判讀 |
|---|---|---|
| 正常 | 取 $a=(1,-1)^T$、$v=(1,2)^T$ | $H_f(a)=\left(\begin{smallmatrix}8&3\\3&4\end{smallmatrix}\right)$，$v^TH_f(a)v=36$；步長 $10^{-3}$ 的差分估計應接近這些解析值。 |
| 餘項 | 依程式用 $h=\text{step}\,v$ | 精確餘項為 $\text{step}^3$，而 $\|h\|^2=5\text{step}^2$；故比值依次預期為 $0.04,0.02,0.01$，容許浮點捨入。 |
| 邊界 | 取 $h=(0,0)^T$ | `taylor_error` 為零；但不可計算 `error/norm²`，因分母為零。程式的比值迴圈刻意只用非零步長。 |
| 故障 | 呼叫 `directional_fd(a,v,0)` 或 `hessian_fd(a,-1)` | 應拋出 `ValueError`，而不是做零除或默默接受無效步長。 |

有限差分的正常輸出可核對實作、揭露轉置或交叉係數錯誤；即使一系列比值下降，也不能證明任意函數在所有方向滿足二階 Taylor 定理。

## 反例與常見陷阱

首先，$C^2$ 所給的 $o(\|h\|^2)$ 不能擅自升級為三階界。在實數線上令 $F(z)=|z|^{5/2}$。它在零附近有連續的二階導數，且 $F(0)=F'(0)=F''(0)=0$，所以零點的二階 Taylor 多項式為零，餘項是 $|h|^{5/2}=o(h^2)$。然而
$$
\frac{|h|^{5/2}}{|h|^3}=|h|^{-1/2}\longrightarrow\infty;
$$
因此該餘項不是 $O(|h|^3)$。

其次，二階偏導的符號及順序必須與定義一致。若只在某一點取得幾個偏導數，卻沒有足以交換混合偏導的鄰域條件，不能直接宣稱 Hessian 對稱；若二階 Fréchet 導數存在，也應清楚說明使用的是何種二階可微性假設，而不是拿數值矩陣的近似對稱作證明。

再者，對稱 Hessian 的交叉項在 $h^THh$ 中出現兩次。把 $\tfrac12h^THh$ 誤寫成「每個非對角元各乘 $\tfrac12$ 後只留一項」，會少算一半。相反地，有限差分所得的 $H_{12}$ 若受浮點誤差影響，與另一種計算次序所得值稍異，也不等同於解析上混合偏導不相等。

最後，對固定方向 $v$ 觀察
$$
\frac{f(x+tv)-2f(x)+f(x-tv)}{t^2}
$$
只是在檢查該方向的二階行為。若已有本章定理的條件，其極限才可辨識為 $v^TH_f(x)v$；有限個方向的吻合不足以建立全空間的二階可微性。

## AI、幾何與養殖案例

考慮純合成的雙參數感測校準量：溫度偏移 $\Delta T$ 以 K 計，鹽度代理量偏移 $\Delta S$ 以指定的無因次單位計，輸出 $\phi$ 是無因次模型分數。先選尺度 $T_*=1\,\mathrm K$、$S_*=0.1$，定義 $z_1=\Delta T/T_*$、$z_2=\Delta S/S_*$，再令
$$
\phi(z_1,z_2)=z_1^2+3z_1z_2+2z_2^2+z_1^3.
$$
這只是展示局部計算的合成函數，不是感測器物理定律。在 $z=(1,-1)^T$ 附近，本章例一的 Hessian 可直接描述分數對**無因次**擾動的二階變化。若返回原輸入 $p=(\Delta T,\Delta S)^T$，寫 $z=Sp$，其中 $S=\operatorname{diag}(1/T_*,1/S_*)$，則
$$
H_p=S^TH_zS.
$$
因此不能把無因次 Hessian 的數字原樣貼到有因次參數上；混合元素的單位與兩個輸入尺度都相關。

幾何上，$h^THh$ 描述指定方向的局部二次變化，但若座標改變，矩陣表示也須相應改變。AI 輔助整理時，可以列出解析式、尺度、差分步長、預期輸出及失效條件；不得把合成校準說成現場驗證，更不得據此控制設備、投餌或加藥。這裏的 agent 角色僅是唯讀整理證據，數學近似不等於操作安全結論。

## 習題

1. **手算。** 對 $p(x,y)=2x^2-4xy+y^2$，求 Hessian，並計算 $h=(s,t)^T$ 時的 $\tfrac12h^TH_ph$。另求 $v=(1,2)^T$ 的二階方向導數。
2. **程式。** 修改本章程式，使其列印 $v=(1,2)^T$、步長依次為 $0.2,0.1,0.05$ 時的 `directional_fd`。先不用執行程式，解析求出每個預期值；並說明為何步長極小時不宜要求逐位精確相等。
3. **反例。** 對 $F(z)=|z|^{5/2}$，直接求零點附近的一、二階導數，證明其零點二階 Taylor 餘項是 $o(h^2)$，卻不是 $O(|h|^3)$。
4. **整合。** 設無因次校準模型 $\psi(z)=z_1^2+2z_1z_2+3z_2^2$，其中 $z_1=\Delta T/(2\,\mathrm K)$、$z_2=\Delta S/0.5$。求對 $z$ 的 Hessian、對有因次輸入 $p=(\Delta T,\Delta S)^T$ 的 Hessian，以及 $p$ 沿 $(0.2\,\mathrm K,0.05)^T$ 擾動時，從 $p=0$ 算起的精確函數增量。說明所用公式為何在此沒有餘項。

## 習題解答

1. 逐項微分得到
   $$
   H_p=\begin{pmatrix}4&-4\\-4&2\end{pmatrix}.
   $$
   因此
   $$
   \frac12h^TH_ph=2s^2-4st+t^2.
   $$
   對 $v=(1,2)^T$，$H_pv=(-4,0)^T$，故 $D^2p(x)[v,v]=v^TH_pv=-4$。二次多項式的 Hessian 不依賴基點。

2. 可在現有程式末尾加入：
   ```python
   for step in (0.2, 0.1, 0.05):
       print(step, directional_fd(a, v, step))
   ```
   對本章的 $f$，在 $a=(1,-1)^T$，$v^TH_f(a)v=36$。沿 $a+tv$ 的 Taylor 展開含三次項 $t^3$；中心相加時，$t^3$ 與 $(-t)^3$ 抵消，所以三個步長的解析預期值**全是** $36$。實際浮點計算會將相近函數值相減再除以 $\text{step}^2$，可能產生捨入誤差；步長過小時尤其不應要求逐位精確相等。

3. 當 $z>0$，$F'(z)=\tfrac52z^{3/2}$、$F''(z)=\tfrac{15}{4}z^{1/2}$；當 $z<0$，$F'(z)=-\tfrac52|z|^{3/2}$、$F''(z)=\tfrac{15}{4}|z|^{1/2}$。利用差商可得 $F'(0)=F''(0)=0$，且 $F''(z)\to0$，故 $F\in C^2$。零點二階多項式為零，餘項比 $h^2$ 是 $|h|^{1/2}\to0$；餘項比 $|h|^3$ 則是 $|h|^{-1/2}\to\infty$，所以不存在所需的局部固定上界常數。

4. 無因次 Hessian 為
   $$
   H_z=\begin{pmatrix}2&2\\2&6\end{pmatrix}.
   $$
   取 $S=\operatorname{diag}(1/(2\,\mathrm K),1/0.5)$，由 $H_p=S^TH_zS$ 得到以指定輸入單位記數的矩陣
   $$
   H_p=\begin{pmatrix}0.5&2\\2&24\end{pmatrix}.
   $$
   其元素的單位須按各輸入分量分別解讀，不能視為相同量綱。給定擾動對應的無因次向量是 $(0.1,0.1)^T$，所以
   $$
   \psi(0.1,0.1)-\psi(0,0)
   =0.01+0.02+0.03=0.06.
   $$
   $\psi$ 恰為二次多項式，沒有三次或更高次項；從原點作二階 Taylor 展開因而是精確等式。

## 本章小結

二階導數本質上是雙線性映射；Hessian 是它在座標中的表示，二階方向變化由 $h^THh$ 給出。在 $C^2$ 鄰域條件下，Hessian 對稱，二階 Taylor 餘項為 $o(\|h\|^2)$。若另有局部 Lipschitz Hessian，才可由本章的積分餘項取得明確的三階界。手算展開能核對交叉項，CPU 差分能檢查實作與觀察趨勢；兩者都不能取代涵蓋所有充分小擾動的證明。

## 參考來源

- [A1] Jiří Lebl，*Basic Analysis*，作者教材入口：https://www.jirka.org/ra/
- [A2] MIT OCW，*18.100A Real Analysis*：https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/
- [A3] MIT OCW，*18.02SC Multivariable Calculus*：https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/

以上列為分析與多變量微積分的延伸入口；本章證明及程式均已在文內寫出，並未聲稱逐條核對教材全文或執行程式。