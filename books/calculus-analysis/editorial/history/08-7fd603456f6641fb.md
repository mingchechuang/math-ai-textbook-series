# 第08章 偏導、方向導數、梯度與Jacobian

## 學習目標與先備知識

本章要回答四個不同問題：改動單一輸入時，輸出如何變化？沿指定方向改動時又如何？若輸出只有一個數，如何以梯度表示局部變化？若輸出有多個分量，如何以Jacobian表示同一件事？讀完後，應能寫出各物件的形狀、單位及定義，並判斷已算出的偏導是否足以支持線性近似。

先備知識包括向量與矩陣乘法、Euclidean範數、單變量極限，以及上一章的Fréchet可微性。以下除特別說明外，$U\subset\mathbb R^n$ 是開集，$f:U\to\mathbb R^m$；向量採直立的**列向量**（column vector）。本章使用的餘項範數是輸入與輸出空間各自的Euclidean範數。若變數帶不同物理單位，這個範數須先經適當尺度化，才有可比較的數值意義。

## 問題與直覺

設一部合成感測器接收兩個設定值，回報溫度與濃度。工程師可能問：「第一個設定值增加一點，溫度讀值改變多少？」這是某個**偏導**；「兩個設定值按既定比例同時改變，兩項讀值如何變化？」這涉及**方向導數**；「哪些微小改動令某個標量損失上升最快？」在Euclidean度量下可用**梯度**回答；「如何一次記錄兩個輸入對兩個輸出的所有一階影響？」則使用**Jacobian**。

這些說法中的「微小」不是指定某個足夠小的浮點數，而是極限敘述。只沿座標軸量到局部斜率，不表示任意輸入擾動都遵守同一線性模型。尤其在不連續或具有尖點的模型中，把偏導排成矩陣可能得到一張形式正確、卻不是導數的表。

## 定義、定理與推導

令 $e_i$ 為第 $i$ 個座標基底列向量。若以下極限存在，稱為 $f$ 在 $x$ 對第 $i$ 個輸入的**偏導數**：

$$
\partial_i f(x)=\lim_{t\to0}\frac{f(x+te_i)-f(x)}{t}\in\mathbb R^m.
$$

它是輸出向量；其第 $a$ 個分量為 $\partial_i f_a(x)$。給定固定方向 $v\in\mathbb R^n$，若極限存在，**沿 $v$ 的雙側方向導數**定為

$$
D_vf(x)=\lim_{t\to0}\frac{f(x+tv)-f(x)}{t}.
$$

此處不要求 $v$ 為單位向量。若改用單位方向，結果代表「每單位Euclidean距離」的變化；有因次變數若未尺度化，這個距離通常沒有直接的物理意義。方向導數的定義只考察一條固定直線，而可微性須同時控制所有趨近零的擾動。

**定義：Fréchet可微。** 若存在線性映射 $L:\mathbb R^n\to\mathbb R^m$，使

$$
f(x+h)=f(x)+Lh+r(h),\qquad
\lim_{\substack{h\to0\\h\ne0}}\frac{\|r(h)\|_2}{\|h\|_2}=0,
$$

則稱 $f$ 在 $x$ 可微，並寫 $Df(x)[h]=Lh$。在標準座標下，$L$ 的矩陣為 $J_f(x)\in\mathbb R^{m\times n}$，其第 $a$ 條橫列、第 $i$ 條縱行的元素是 $\partial_i f_a(x)$。矩陣的每一個縱行記錄一個輸入方向的輸出變化。因此

$$
Df(x)[h]=J_f(x)h,\qquad h\in\mathbb R^{n\times1}.
$$

矩陣的**行空間**（column space）是 $\mathbb R^m$ 的子空間，維度為矩陣的秩；**列空間**（row space）是 $\mathbb R^n$ 的子空間，維度也為該秩。它們不是Jacobian的輸出、輸入空間本身，除非另有滿秩條件。

**命題（可微時的方向導數）。** 設 $U$ 開、$x\in U$，且 $f:U\to\mathbb R^m$ 在 $x$ Fréchet可微。則每個 $v\in\mathbb R^n$ 的方向導數均存在，且
$D_vf(x)=J_f(x)v$。特別地，所有偏導存在，並構成 $J_f(x)$。

**證明。** 若 $v=0$，差商恆為零。若 $v\ne0$，把可微性等式中的 $h$ 換成 $tv$。由 $U$ 開知，充分小的正負 $t$ 都可使用。於是

$$
\frac{f(x+tv)-f(x)}t
=J_f(x)v+\frac{r(tv)}t.
$$

餘項滿足
$\|r(tv)/t\|_2
=\bigl(\|r(tv)\|_2/\|tv\|_2\bigr)\|v\|_2\to0$。
故所求極限為 $J_f(x)v$。令 $v=e_i$ 即得第 $i$ 個偏導等於Jacobian的第 $i$ 個縱行。證畢。

這是**可微的必要後果**，不能倒過來說「所有偏導存在便可微」。一個常用的**充分條件**是：各分量的所有一階偏導在 $x$ 的某個開鄰域內存在，並在 $x$ 連續；此時 $f$ 在 $x$ 可微。條件只需在所論點滿足相應連續性，並非要求整個定義域的偏導都連續。該充分條件可由沿座標逐段分解增量、再以單變量中值定理控制誤差證得；這裡引用此結果，不把「偏導存在」誤當作其全部假設。

對標量函數 $\phi:U\to\mathbb R$，$D\phi(x)$ 是作用於輸入擾動的線性泛函，矩陣形狀為 $1\times n$。在**標準Euclidean內積**下，梯度定為 $n\times1$ 列向量

$$
\nabla\phi(x)=
\begin{pmatrix}\partial_1\phi(x)\\ \vdots\\ \partial_n\phi(x)\end{pmatrix},
\qquad
D\phi(x)[h]=(\nabla\phi(x))^{\mathsf T}h.
$$

所以 $D\phi(x)$ 與 $\nabla\phi(x)$ 形狀不同；前者作用在 $h$ 上，後者是經內積識別得到的向量。若選用非Euclidean度量，這項識別須隨度量修改。由Cauchy–Schwarz不等式可知，當梯度非零時，單位Euclidean方向中 $\nabla\phi/\|\nabla\phi\|_2$ 使方向導數最大；這是附有度量條件的結論，不是對混合單位輸入的無條件指令。

## 逐步手算例題

**例一：多輸出映射。** 令

$$
f(x,y)=
\begin{pmatrix}x^2y\\ \sin x+y^2\end{pmatrix}.
$$

逐分量、逐輸入求偏導，得到

$$
J_f(x,y)=
\begin{pmatrix}
2xy&x^2\\
\cos x&2y
\end{pmatrix}.
$$

在 $(0,1)$，Jacobian是
$\begin{pmatrix}0&0\\1&2\end{pmatrix}$。
取 $v=(1,-1)^{\mathsf T}$，方向導數為
$J_f(0,1)v=(0,-1)^{\mathsf T}$。
也可直接代入：$f(t,1-t)$ 的第一分量為 $t^2(1-t)$，除以 $t$ 的增量趨零；第二分量為 $\sin t+(1-t)^2$，與 $f(0,1)$ 相減後除以 $t$ 趨 $1-2=-1$。兩種算法相合，原因是此函數可微，並非因為任何沿線計算都能證明可微。

**例二：標量函數與形狀。** 令 $\phi(x,y)=3x^2+xy$，在 $(1,2)$ 有 $\partial_x\phi=8$、$\partial_y\phi=1$。因此

$$
D\phi(1,2)=\begin{pmatrix}8&1\end{pmatrix},
\qquad
\nabla\phi(1,2)=\begin{pmatrix}8\\1\end{pmatrix}.
$$

若 $h=(0.01,-0.02)^{\mathsf T}$，線性預測為
$D\phi(1,2)[h]=0.06$。直接展開所得實際增量為
$8h_1+h_2+3h_1^2+h_1h_2=0.0601$，餘項為 $0.0001$。這個單點數值核對可檢查算術，不能單靠它證明餘項比的極限。

## 實作與程式

以下NumPy程式建立**合成**雙輸入、雙輸出感測映射。輸入 $T$ 的單位為攝氏度、$c$ 的單位為 $\mathrm{mg/L}$；輸出第一分量 $\theta$ 的單位為攝氏度，第二分量 $s$ 的單位為 $\mathrm{mg/L}$。先設定參考值 $T_0=20\,^\circ\mathrm C$、$c_0=5\,\mathrm{mg/L}$，以尺度 $2\,^\circ\mathrm C$ 及 $1\,\mathrm{mg/L}$ 定義無因次輸入 $u,z$。模型只供數學演示，不聲稱描述真實設備：

$$
u=\frac{T-T_0}{2\,^\circ\mathrm C},\quad
z=\frac{c-c_0}{1\,\mathrm{mg/L}},\quad
\theta=20+2u+uz,\quad
s=5+3z+u^2.
$$

輸出式中的常數與係數分別帶有使結果成為攝氏度及 $\mathrm{mg/L}$ 的單位；程式儲存的是按上述單位表示的數值。完整物理Jacobian為

$$
J_F(T,c)=
\begin{pmatrix}
(2+z)/2&u/1\\
2u/2&3/1
\end{pmatrix}.
$$

四個元素的單位依位置依次為
$^\circ\mathrm C/^\circ\mathrm C$、
$^\circ\mathrm C/(\mathrm{mg/L})$、
$(\mathrm{mg/L})/^\circ\mathrm C$、
$(\mathrm{mg/L})/(\mathrm{mg/L})$。
分母中的 $2,1$ 各代表其對應輸入尺度的數值，而非抹去單位。

```python
import numpy as np

def sensor(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (2, 1) or not np.isfinite(x).all():
        raise ValueError("input must be a finite (2, 1) column")
    u = (x[0, 0] - 20.0) / 2.0
    z = (x[1, 0] - 5.0) / 1.0
    return np.array([[20.0 + 2.0*u + u*z],
                     [5.0 + 3.0*z + u*u]])

def jacobian(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (2, 1) or not np.isfinite(x).all():
        raise ValueError("input must be a finite (2, 1) column")
    u = (x[0, 0] - 20.0) / 2.0
    z = (x[1, 0] - 5.0) / 1.0
    return np.array([[(2.0 + z)/2.0, u/1.0],
                     [2.0*u/2.0,       3.0/1.0]])

def scaled_remainder_ratio(x, h):
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if h.shape != (2, 1) or not np.isfinite(h).all():
        raise ValueError("h must be a finite (2, 1) column")
    # 輸入尺度：(2 °C, 1 mg/L)；輸出尺度：(2 °C, 1 mg/L)。
    input_scale = np.array([[2.0], [1.0]])
    output_scale = np.array([[2.0], [1.0]])
    hn = np.linalg.norm(h / input_scale)
    if hn == 0.0:
        raise ValueError("h must be nonzero")
    r = sensor(x + h) - sensor(x) - jacobian(x) @ h
    return np.linalg.norm(r / output_scale) / hn

x = np.array([[20.0], [5.0]])
h = np.array([[0.2], [-0.1]])
print(sensor(x), jacobian(x))
print(jacobian(x) @ h, scaled_remainder_ratio(x, h))
```

程式用明確的 $(2,1)$ 形狀：NumPy的一維陣列即使加上 `.T`，仍是一維陣列，不能藉此宣稱已把列向量變成橫向量。餘項測試同時尺度化輸入與輸出，避免把攝氏度與 $\mathrm{mg/L}$ 的原始數值直接合成一個物理長度。尺度化後的比值仍只是指定擾動下的檢查，不是全方向極限證明。

## 測試與預期結果

以下均為**未執行的預期**，不是測試紀錄。

- **正常測試：** 在 $x=(20,5)^{\mathsf T}$，預期 `sensor(x)` 為 $(20,5)^{\mathsf T}$，`jacobian(x)` 為 $\begin{pmatrix}1&0\\0&3\end{pmatrix}$。給定程式中的 $h$，線性預測為 $(0.2,-0.3)^{\mathsf T}$；實際餘項為 $(-0.01,0.01)^{\mathsf T}$，尺度化餘項比預期約為 $0.0707$。
- **邊界測試：** $h=0$ 時分母為零，函式預期明確拋出 `ValueError`。這不是「零方向不存在」：零方向的方向導數依定義為零；不能計算的是此一餘項比的零分母。
- **故障測試：** 傳入形狀為 `(2,)` 的輸入，或含 `nan` 的擾動，預期拋出 `ValueError`，不暗中猜測其形狀或接受非有限數。

可另以逐次縮小的非零 $h$ 檢查餘項比是否下降，但浮點捨入終會干擾結果。有限個測試點既不能涵蓋所有方向，也不能建立極限。

## 反例與常見陷阱

**所有方向導數存在，仍不一定有導數矩陣。** 定義 $q:\mathbb R^2\to\mathbb R$ 為

$$
q(x,y)=
\begin{cases}
\dfrac{x^3}{x^2+y^2},&(x,y)\ne(0,0),\\
0,&(x,y)=(0,0).
\end{cases}
$$

對任意固定 $v=(a,b)^{\mathsf T}\ne0$，

$$
D_vq(0,0)
=\lim_{t\to0}\frac{q(ta,tb)}t
=\frac{a^3}{a^2+b^2}.
$$

零方向的值為零。因此所有雙側方向導數都存在；偏導為
$\partial_xq(0,0)=1$、$\partial_yq(0,0)=0$。若 $q$ 在原點可微，已證命題迫使方向導數等於候選橫列矩陣 $(1,0)v=a$。然而 $v=(1,1)^{\mathsf T}$ 時，實際方向導數為 $1/2$，候選值卻為 $1$；矛盾。這是完整反證，不依賴有限路徑採樣。此例甚至在原點連續，因為 $|q(x,y)|\leq|x|\to0$；問題不只是連續性缺失，而是方向變化不能組成同一線性映射。

另一種更基本的錯誤是以偏導存在推論連續。例如令 $p(x,y)=xy/(x^2+y^2)$ 於非原點，並令 $p(0,0)=0$。兩個座標軸上的偏導皆為零，但沿 $y=x$ 趨近原點時函數值為 $1/2$，故它甚至不連續。偏導只查兩條座標軸；方向導數雖查每條固定直線，仍未提供對所有小擾動**一致**的餘項控制。

## AI、幾何與養殖案例

若合成模型以輸入擾動 $h$ 預測感測輸出，$Jh$ 是局部線性預測，而實際與預測之差是應報告的餘項。AI代理可唯讀整理模型公式、尺度、Jacobian、測試擾動與殘差，並指出某些方向資料不足；它不因此取得改動設備、投餌或加藥的依據。模型校準也不等於現場驗證。

幾何上，Jacobian把輸入空間的一個小位移送到輸出空間的切向變化。對本章的雙輸出映射，其第一個縱行是只改變 $T$ 時的輸出切向量，第二個縱行是只改變 $c$ 時的切向量。然而兩種輸出單位不同，未尺度化的箭頭長度不能直接比較。若使用右手 $X/Y/Z$ 幾何座標，方向也須先說明其所屬座標系；不能把感測設定的兩個分量默認成空間的 $X/Y$ 位移。這些區別使線性代數圖像能輔助理解，而不遮蔽模型的單位與適用範圍。

## 習題

1. **手算。** 對 $f(x,y)=(xy,y^2+x)^{\mathsf T}$，求 $J_f(2,1)$ 及沿 $v=(1,-2)^{\mathsf T}$ 的方向導數；再求 $\phi(x,y)=xy+y^2$ 在同一點的梯度與標量導數橫列。
2. **程式。** 不執行程式，依本章 `sensor` 公式求在 $x=(20,5)^{\mathsf T}$、$h=(0.02,0)^{\mathsf T}$ 時的 $Jh$、精確餘項，以及 `scaled_remainder_ratio` 的預期值。說明這個結果能否證明可微。
3. **反例。** 計算上文 $q$ 在原點沿 $(1,0)^{\mathsf T}$、$(0,1)^{\mathsf T}$、$(1,1)^{\mathsf T}$ 的方向導數。僅以這三個**解析計算結果**，證明不存在與全部方向導數相符的線性映射。
4. **整合。** 一個無因次標量品質指標定為 $\phi(T,c)=u^2+z^2$，其中 $u,z$ 採本章尺度。求在 $(T,c)=(22,6)$ 的物理輸入梯度、$D\phi$ 的矩陣形狀，以及 $h=(0.2,-0.1)^{\mathsf T}$ 的一階預測。解釋為何不能把梯度的兩個數值直接視為「兩個物理方向同樣長」的比較。

## 習題解答

1. 分別微分得
   $J_f(x,y)=\begin{pmatrix}y&x\\1&2y\end{pmatrix}$，
   故 $J_f(2,1)=\begin{pmatrix}1&2\\1&2\end{pmatrix}$，
   $D_vf(2,1)=(-3,-3)^{\mathsf T}$。
   對 $\phi$，$\partial_x\phi=y$、$\partial_y\phi=x+2y$，所以
   $\nabla\phi(2,1)=(1,4)^{\mathsf T}$，
   $D\phi(2,1)=\begin{pmatrix}1&4\end{pmatrix}$。
2. 此時 $J=\begin{pmatrix}1&0\\0&3\end{pmatrix}$，故
   $Jh=(0.02,0)^{\mathsf T}$。相應無因次增量為 $\Delta u=0.01$、$\Delta z=0$；第一輸出的交叉項與第二輸出的平方項給出精確餘項
   $r=(0,0.0001)^{\mathsf T}$。
   尺度化輸入長度為 $0.01$，尺度化餘項長度為 $0.0001$，預期比值為 $0.01$。這只是一個擾動的算術結果，不能證明全方向極限；本模型可微須由其多項式形式及相應微分定理確立。
3. 三個值依次為 $1,0,1/2$。若線性映射 $L$ 符合所有方向導數，線性性必給
   $L(1,1)^{\mathsf T}
   =L(1,0)^{\mathsf T}+L(0,1)^{\mathsf T}=1$，
   與解析所得 $1/2$ 矛盾。因此不存在這樣的線性映射。此處三個方向足以**推翻**線性主張，與用有限採樣**證明**全方向性質不同。
4. 在 $(22,6)$，$u=z=1$。由尺度及單變量鏈式計算，
   $\partial_T\phi=2u/2=1\;(^\circ\mathrm C)^{-1}$，
   $\partial_c\phi=2z/1=2\;(\mathrm{mg/L})^{-1}$。
   梯度是 $(1,2)^{\mathsf T}$，形狀 $2\times1$；
   $D\phi=\begin{pmatrix}1&2\end{pmatrix}$，形狀 $1\times2$。
   一階預測為 $1(0.2)+2(-0.1)=0$。兩個梯度分量的單位不同，原始數值大小依輸入單位與尺度而變；須先指定尺度或度量，才能比較「同樣長」的方向。

## 本章小結

偏導考察座標軸，方向導數考察固定直線；Fréchet導數則要求**同一線性映射**以趨零的相對餘項控制所有小擾動。可微時，這個映射的標準矩陣為 $m\times n$ 的Jacobian，並給出 $D_vf=J_fv$。標量函數的導數是 $1\times n$ 的線性泛函；其Euclidean梯度是 $n\times1$ 的列向量。反例表明，即使每個方向導數都存在，也不能省略可微性檢查。計算與測試有助於稽核公式，極限結論仍須依定義或具備條件的定理建立。

## 參考來源

- [A1] Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>
- [A2] MIT OpenCourseWare，*18.100A Real Analysis*，課程概要：<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>
- [A3] MIT OpenCourseWare，*18.02SC Multivariable Calculus*，課程概要：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>

上述入口供延伸閱讀；本章沒有宣稱逐頁查證教材全文。程式與預期測試均未執行。