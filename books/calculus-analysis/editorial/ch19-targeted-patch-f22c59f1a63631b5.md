<<<PATCH 19>>>
<<<OLD>>>
在每個子矩形 $R_{ij} \cap D$ 中任取一點 $(x_{ij}^*, y_{ij}^*)$。Riemann和定義為：
$$
S(P, f) = \sum_{i=1}^m \sum_{j=1}^n f(x_{ij}^*, y_{ij}^*) \Delta A_{ij}
$$
當 $\|P\| \to 0$ 時，若 $S(P, f)$ 的極限存在且與分割 $P$ 及取點 $(x_{ij}^*, y_{ij}^*)$ 的選擇無關，則稱 $f$ 在 $D$ 上Riemann可積，該極限記為：
$$
\iint_D f(x,y) \, dA
$$

**注意**：對於非矩形區域 $D$，通常定義 $f$ 在 $D$ 上可積是指 $f$ 擴展到包含 $D$ 的矩形 $\bar{D}$ 上的函數 $F$ 在 $\bar{D}$ 上可積，其中 $F(x,y) = f(x,y)$ 若 $(x,y) \in D$，否則 $F(x,y) = 0$。此時 $\iint_D f \, dA = \iint_{\bar{D}} F \, dA$。
<<<NEW>>>
先令 $R=[a,b]\times[c,d]$ 為非退化閉矩形。每個子矩形 $R_{ij}$ 任取取樣點 $(x_{ij}^*,y_{ij}^*)$，矩形上的Riemann和定義為
$$
S(P,f)=\sum_{i=1}^m\sum_{j=1}^n f(x_{ij}^*,y_{ij}^*)\Delta A_{ij}.
$$
若當 $\|P\|\to0$ 時，所有分割與取樣點所產生的和皆趨於同一極限，則稱 $f$ 在 $R$ 上Riemann可積，並將此極限記為 $\iint_R f(x,y)\,dA$。

若 $D\subset R$，定義補零函數 $F=f1_D$；僅當 $F$ 在 $R$ 上Riemann可積時，才定義 $\iint_D f\,dA:=\iint_R F\,dA$。一般情況下這不保證成立。常用充分條件是 $D$ 為有界Jordan可測區域，且 $f$ 在包含 $D$ 的矩形上連續；此時 $f1_D$ Riemann可積。邊界相交的格子須按其與 $D$ 的交集面積加權，不能乘上完整格子面積。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
**定理 19.1（可積性充要條件）**：設 $f$ 在有界矩形 $D$ 上有界。則 $f$ 在 $D$ 上Riemann可積，當且僅當對任意 $\epsilon > 0$，存在分割 $P$ 使得上和 $U(P, f)$ 與下和 $L(P, f)$ 之差異小於 $\epsilon$。

**證明**：
充分性：若存在分割序列使得 $U - L < \epsilon$，則由單變量Riemann可積性的推廣（Cantor-Lebesgue引理），$f$ 可積。
必要性：若 $f$ 可積，則對於任意 $\epsilon > 0$，存在分割 $P$ 使得 $U(P, f) - \iint_D f < \epsilon/2$ 且 $\iint_D f - L(P, f) < \epsilon/2$，從而 $U(P, f) - L(P, f) < \epsilon$。
$\blacksquare$

**定理 19.2（連續函數可積性）**：若 $f$ 在緊集 $D$ 上連續，則 $f$ 在 $D$ 上Riemann可積。

**證明**：
由於 $f$ 在緊集上連續，故一致連續。對任意 $\epsilon > 0$，存在 $\delta > 0$ 使得對所有 $(x_1, y_1), (x_2, y_2) \in D$，若 $\|(x_1, y_1) - (x_2, y_2)\| < \delta$，則 $|f(x_1, y_1) - f(x_2, y_2)| < \epsilon / A(D)$，其中 $A(D)$ 是 $D$ 的面積（或包含 $D$ 的矩形面積）。
取分割 $P$ 使得 $\|P\| < \delta / \sqrt{2}$（考慮對角線長度）。則在每個子矩形 $R_{ij}$ 中，函數振盪（Oscillation）$\omega_{ij} < \epsilon / A(D)$。
上和與下和之差為：
$$
U(P, f) - L(P, f) = \sum_{i,j} \omega_{ij} \Delta A_{ij} < \frac{\epsilon}{A(D)} \sum_{i,j} \Delta A_{ij} = \epsilon
$$
由定理 19.1，$f$ 可積。
$\blacksquare$

**直覺**：連續性確保了函數值在微小區域內變化微小，從而矩形面積之和的誤差可以被控制。如果不連續點太多（例如Cantor集上非零），振盪可能無法同時在所有子塊上壓小。
<<<NEW>>>
**定理 19.1（可積性充要條件）**：設 $f$ 在非退化矩形 $R$ 上有界。對分割 $P$，令每格的上、下確界為 $M_{ij}$、$m_{ij}$，並定義 $U(P,f)=\sum M_{ij}\Delta A_{ij}$、$L(P,f)=\sum m_{ij}\Delta A_{ij}$。則 $f$ Riemann可積，當且僅當對每個 $\epsilon>0$，存在分割 $P$ 使 $U(P,f)-L(P,f)<\epsilon$。

**證明**：定義下積分與上積分
$$
\underline I=\sup_P L(P,f),\qquad \overline I=\inf_P U(P,f).
$$
任意兩個分割有共同細分；細分不增上和、不減下和，故 $\underline I\le\overline I$。若存在分割使 $U-L<\epsilon$，則
$$
0\le\overline I-\underline I\le U(P,f)-L(P,f)<\epsilon.
$$
由任意 $\epsilon>0$，得兩者相等。細分時振盪和可任意小，故任意標記和都夾在趨於共同值的上下和之間，$f$ Riemann可積。反之，若Riemann和趨於積分值 $I$，取網格充分細，使任意標記和與 $I$ 的差小於 $\epsilon/4$。逐格選取使函數值任意接近上確界、下確界的取樣點，令兩個標記和分別近似上和、下和，總誤差各小於 $\epsilon/4$；因此 $U-L<\epsilon$。$\\blacksquare$

**定理 19.2（連續函數可積性）**：若 $f$ 在非退化緊矩形 $R$ 上連續，則 $f$ 在 $R$ 上Riemann可積。

**證明**：$R$ 緊緻，故 $f$ 一致連續。給定 $\epsilon>0$，存在 $\delta>0$，使距離小於 $\delta$ 的兩點函數值相差小於 $\epsilon/A(R)$，其中 $A(R)=(b-a)(d-c)>0$。取分割使 $\|P\|<\delta/\sqrt2$，每格直徑小於 $\delta$，故其振盪 $\omega_{ij}<\epsilon/A(R)$。於是
$$
U(P,f)-L(P,f)=\sum_{i,j}\omega_{ij}\Delta A_{ij}<\frac{\epsilon}{A(R)}\sum_{i,j}\Delta A_{ij}=\epsilon.
$$
由定理19.1，$f$ 可積。$\\blacksquare$

**直覺**：一致連續性使所有小格中的振盪同時受控，格子面積總和恰為 $A(R)$。不能把此結論直接套到任意緊集再補零，因為補零函數可能在邊界不連續。矩形上的 $1_{\mathbb Q^2}$ 在每格上確界為1、下確界為0，故不可積；Cantor集的指示函數則不是不可積反例。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
**定理 19.3（Fubini）**：設 $f$ 在矩形 $D = [a,b] \times [c,d]$ 上Riemann可積。則：
1. 對幾乎所有 $x \in [a,b]$，函數 $g(x) = \int_c^d f(x,y) \, dy$ 存在。
2. $g(x)$ 在 $[a,b]$ 上Riemann可積。
3. $\iint_D f(x,y) \, dA = \int_a^b \left( \int_c^d f(x,y) \, dy \right) dx = \int_c^d \left( \int_a^b f(x,y) \, dx \right) dy$。

**證明**：
考慮Riemann和：
$$
S = \sum_{i=1}^m \sum_{j=1}^n f(x_i^*, y_j^*) \Delta x_i \Delta y_j
$$
固定 $i$，內部和 $\sum_{j=1}^n f(x_i^*, y_j^*) \Delta y_j$ 是 $y$ 方向關於 $f(x_i^*, \cdot)$ 的Riemann和。
由於 $f$ 在 $D$ 上可積，則 $f(x, y)$ 對 $y$ 的可積性對於幾乎所有 $x$ 成立（由單變量可積性推廣）。
令 $I_i = \sum_{j=1}^n f(x_i^*, y_j^*) \Delta y_j$。則 $S = \sum_{i=1}^m I_i \Delta x_i$。
當 $\|P\| \to 0$ 時，$I_i$ 趨向於 $\int_c^d f(x_i^*, y) \, dy$（假設 $f(x_i^*, y)$ 對 $y$ 連續）。
因此 $S$ 趨向於 $\int_a^b \left( \int_c^d f(x,y) \, dy \right) dx$。
同理可證反向順序。
$\blacksquare$
<<<NEW>>>
**定理 19.3（連續函數的迭代積分）**：若 $f$ 在矩形 $R=[a,b]\times[c,d]$ 上連續，則每條水平與垂直截面均連續，兩個迭代Riemann積分存在，且
$$
\iint_R f(x,y)\,dA=\int_a^b\left(\int_c^d f(x,y)\,dy\right)dx
=\int_c^d\left(\int_a^b f(x,y)\,dx\right)dy.
$$

**證明**：令 $g(x)=\int_c^d f(x,y)\,dy$。由一致連續性，若 $x,x'$ 接近，所有 $y$ 上的函數值差同時很小，且
$$
|g(x)-g(x')|\le(d-c)\sup_{y\in[c,d]}|f(x,y)-f(x',y)|.
$$
故 $g$ 連續。取矩形分割，先對每個固定 $x$ 的截面形成內層Riemann和，再乘以外層區間長度相加；這正是矩形上的標記Riemann和。由 $f$ 一致連續，內層和對所有 $x$ 一致逼近 $g(x)$；令網格大小趨零，外層Riemann和遂趨於 $\int_a^b g(x)\,dx$，而矩形Riemann和趨於 $\iint_R f\,dA$。交換兩座標同理。$\\blacksquare$
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
對於無界區域或無界函數，定義瑕積分。
例如，$\iint_D f \, dA$ 其中 $D$ 無界或 $f$ 在某點發散。
**定理 19.4（Fubini的瑕積分版本）**：設 $f \ge 0$ 在無界區域 $D$ 上可測。若 $\iint_D f \, dA < \infty$，則 Fubini 交換順序成立。若 $f$ 可正可負，則需 $\iint_D |f| \, dA < \infty$（絕對可積）才能保證交換順序後結果相同且有限。

**反例**：設 $f(x,y) = \frac{x^2 - y^2}{(x^2 + y^2)^2}$ 在 $D = [0,1] \times [0,1]$ 去除 $(0,0)$。
計算 $\int_0^1 \int_0^1 f \, dx \, dy$ 與 $\int_0^1 \int_0^1 f \, dy \, dx$ 可能得出一正一負或收斂到不同值（若未絕對收斂）。此處不展開計算，但強調：**非絕對收斂的瑕積分交換順序可能導致錯誤**。
<<<NEW>>>
瑕積分的換序須另行證明，本章不引用以「可測」為假設的Lebesgue–Tonelli定理。對矩形截斷上的連續函數，若兩個方向的瑕積分按一致截斷定義且絕對收斂，則有限截斷上的換序與絕對尾項控制可推出相等；未證明絕對收斂時不可任意交換。

**反例**：令
$$
f(x,y)=\frac{x^2-y^2}{(x^2+y^2)^2},\qquad (x,y)\in[0,1]^2\setminus\{(0,0)\}.
$$
固定 $y>0$，對 $x$ 積分的原函數為 $-x/(x^2+y^2)$，因此內層積分為 $-1/(1+y^2)$；再對 $y\in[0,1]$ 積分得 $-\pi/4$。反向固定 $x>0$，對 $y$ 積分的原函數為 $y/(x^2+y^2)$，內層積分為 $1/(1+x^2)$；再對 $x\in[0,1]$ 積分得 $\pi/4$。原點附近絕對積分的徑向部分含 $\int_0^\varepsilon dr/r$，故發散，兩個瑕迭代積分不能交換。
<<<END>>>