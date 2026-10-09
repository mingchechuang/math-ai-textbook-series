# 第29章 變分法、第一變分與能量泛函

## 學習目標與先備知識

本章把有限維的「沿方向改變變數」推廣為「沿容許函數改變函數」。學完後應能：指定泛函的定義域與範數；計算第一、第二變分；從容許擾動推導 Euler–Lagrange 方程及自由端的自然邊界條件；區分駐點、局部極小與全域極小。後續離散能量的座標梯度，仍須由離散泛函本身求導，不能直接把連續方程當成梯度。

先備知識包括一維積分與分部積分、Fréchet 導數、對稱二次型，以及有限維梯度依賴內積的觀念。本節先在 $C^1([0,1])$ 上工作，使用
$$
\|u\|_{C^1}=\max_{[0,1]}|u|+\max_{[0,1]}|u'|.
$$
固定兩端值時，容許集合是具有指定端點值的仿射集合；其擾動空間為 $X_0=\{v\in C^1([0,1]):v(0)=v(1)=0\}$。若只固定左端，則改用 $X_L=\{v\in C^1([0,1]):v(0)=0\}$。這些空間的區別會決定右端能否產生自然邊界條件。

處理弱解時，可另選 $H^1(0,1)=\{u\in L^2(0,1):u'\text{ 的弱導數屬於 }L^2(0,1)\}$，範數為 $\|u\|_{H^1}=(\|u\|_{L^2}^2+\|u'\|_{L^2}^2)^{1/2}$；$H^1_0(0,1)$ 是光滑緊支撐函數在此範數下的閉包。一維的端點跡、連續代表與 Poincaré 不等式屬於此處引用的結果，不能把光滑函數的逐點推導不加條件地套到任意弱解。

## 問題與直覺

有限維函數在內點取得局部極小時，每個可行方向的一階導數為零。對能量 $J[u]$ 而言，可行方向是函數 $v$：先考察 $J[u+tv]$ 這個單變量函數，再問其在 $t=0$ 的導數。若端點值已固定，擾動必須在該端點為零；若端點自由，該處的擾動可以非零，分部積分留下的端點項便提供額外條件。

此類問題有三個容易混淆的層次。沿每個固定方向存在導數，只是方向導數的敘述；Fréchet 可微還要求同一個線性泛函能以所選範數一致控制所有足夠小的擾動。第一變分為零只表示駐點，不能判定它是最小值。最後，連續泛函的導數是作用於擾動的線性泛函；把函數取樣成網格向量後，歐氏座標梯度又是另一個對象，兩者的對應取決於離散能量與所選內積。

以拉伸細線的模型為例，$\int_0^1(u')^2/2\,dx$ 懲罰斜率，外力項 $-\int_0^1fu\,dx$ 則偏好某些位移。兩項競爭後形成的平衡方程，可以由第一變分導出；但能否斷言平衡是最小值，仍要另查能量差、凸性或有充分條件的第二變分。

## 定義、定理與推導

### 第一變分、範數與交換積分

令 $X$ 為上述擾動空間，$u$ 屬於相應的容許仿射集合。若極限存在，第一變分定義為
$$
\delta J[u;v]=\left.\frac{d}{dt}J[u+tv]\right|_{t=0}.
$$
若存在有界線性泛函 $DJ(u)\in X^*$，使
$$
J[u+h]=J[u]+DJ(u)[h]+r(h),\qquad
\frac{|r(h)|}{\|h\|_X}\longrightarrow0
$$
（$h\ne0$ 且 $u+h$ 容許），則稱 $J$ 在 $u$ 處相對於 $X$ 的範數 Fréchet 可微；此時 $\delta J[u;v]=DJ(u)[v]$。反過來，所有方向導數存在並不足以保證這種一致餘項。

考慮
$$
J[u]=\int_0^1 L(x,u(x),u'(x))\,dx.
$$
若 $L$ 及其對後兩個變數的一階偏導在包含所考察函數值與導數值的鄰域連續，則對固定的 $u,v\in C^1([0,1])$，$t$ 限於足夠小的閉區間時，被積函數及其 $t$ 導數在緊集合上連續且有界。因此可交換對 $t$ 的微分與積分，得到
$$
\delta J[u;v]
=\int_0^1\bigl(L_u(x,u,u')v+L_p(x,u,u')v'\bigr)\,dx.
$$
在非緊域或改用弱函數空間時，單憑偏導連續不足以作此交換，還須另給可積支配及確保泛函有限的增長條件。

### Euler–Lagrange 方程與邊界

**必要條件。** 假設 $u$ 是容許集合中的局部極小點，上述方向導數對每個容許擾動存在，且 $u+tv$ 對充分小的正、負 $t$ 均容許。單變量函數 $t\mapsto J[u+tv]$ 在零處有局部極小，故 $\delta J[u;v]=0$。特別取 $v\in C_c^\infty(0,1)$，便得到弱形式
$$
\int_0^1(L_uv+L_pv')\,dx=0
\qquad(v\in C_c^\infty(0,1)).
$$
若合成函數 $x\mapsto L_p(x,u(x),u'(x))$ 可連續微分，且 $L_u(x,u,u')$ 連續，分部積分及變分法基本引理給出經典方程
$$
L_u-\frac{d}{dx}L_p=0\qquad(0<x<1).
$$
僅說 $u\in C^2$ 並不足夠；還須有使合成函數可微的 $L$ 的正則性。若只有弱形式，方程先按分布意義理解，不預設弱解已有二階經典導數。

對任意 $v\in X_L$，分部積分後的端點項是 $[L_pv]_0^1=L_p(1,u(1),u'(1))v(1)$。內部方程成立且右端擾動值可任意選取時，必要的自然條件為
$$
L_p(1,u(1),u'(1))=0.
$$
左端已固定，不能再從同一變分推出左端的自然條件。若兩端均固定，兩個端點項都因擾動為零而消失；若邊界另含端點能量，其導數也必須計入端點條件。

### 第二變分與極小的充分條件

若二階導數存在，定義 $\delta^2J[u;v]=D^2J(u)[v,v]$。局部極小且可沿正、負方向擾動時，一階變分必為零，第二變分必為非負；這些是必要條件，單靠它們通常不能反推極小。

**小命題。** 設 $X$ 為賦範線性空間，$u+X$ 中的 $u$ 為內點，$DJ(u)=0$，並有相對於 $\|\cdot\|_X$ 的二階展開
$$
J[u+h]-J[u]=\tfrac12D^2J(u)[h,h]+o(\|h\|_X^2).
$$
若存在與 $h$ 無關的 $c>0$，使 $D^2J(u)[h,h]\ge c\|h\|_X^2$ 對所有 $h\in X$ 成立，則 $u$ 為嚴格局部極小。

**證明。** 依小 $o$ 的定義，存在 $\rho>0$，使 $0<\|h\|_X<\rho$ 時餘項的絕對值不超過 $c\|h\|_X^2/4$。因此
$$
J[u+h]-J[u]\ge
\tfrac c2\|h\|_X^2-\tfrac c4\|h\|_X^2
=\tfrac c4\|h\|_X^2>0.
$$
這證明了充分性，並未聲稱每個局部極小的第二變分都須一致強制。無限維中「對每個非零 $h$ 為正」與「存在共同的正下界 $c$」尤其不能混用；若 $DJ(u)\ne0$，一階項也不能從展開中刪去。

## 逐步手算例題

### 例一：零端點的受力細線

取 $f(x)=\pi^2\sin(\pi x)$，在零端點的容許集合上考慮
$$
J[u]=\frac12\int_0^1(u')^2\,dx-\int_0^1fu\,dx.
$$
對 $v\in X_0$，第一變分是 $\int_0^1(u'v'-fv)\,dx$。對光滑解分部積分，端點項因 $v(0)=v(1)=0$ 消失，故 $-u''=f$。積分兩次可得 $u=\sin(\pi x)+C_1x+C_2$；兩個零端點依次給出 $C_2=C_1=0$，所以候選解為 $u^*(x)=\sin(\pi x)$。

直接計算能量時，$\int_0^1\cos^2(\pi x)\,dx=\int_0^1\sin^2(\pi x)\,dx=1/2$，因此
$$
J[u^*]=\frac{\pi^2}{2}\cdot\frac12-\pi^2\cdot\frac12
=-\frac{\pi^2}{4}.
$$
此例還可不依賴局部二階判據，直接證明全域性：將任何同端點函數寫成 $u^*+v$，展開後的交叉項是 $\int_0^1(u^{*\prime}v'-fv)\,dx=0$，從而
$$
J[u^*+v]-J[u^*]=\frac12\int_0^1(v')^2\,dx\ge0.
$$
等號要求 $v'$ 為零；配合零端點即得 $v=0$。在 $H^1_0(0,1)$ 中同一結論亦成立，其中使用弱導數及端點跡的性質；若要相對完整 $H^1$ 範數表述強制性，則須引用此區間的 Poincaré 不等式。

### 例二：固定左端、自由右端

在 $u(0)=0$ 而 $u(1)$ 不指定的集合上，考慮
$$
J[u]=\int_0^1\left(\frac12(u')^2+\frac14u^4-gu\right)dx.
$$
對 $v\in X_L$，第一變分為 $\int_0^1\{u'v'+(u^3-g)v\}\,dx$。分部積分後，內部方程與自由端條件分別是
$$
-u''+u^3-g=0,\qquad u'(1)=0.
$$
選取明確的資料 $u^*(x)=x(2-x)$ 及 $g(x)=2+[x(2-x)]^3$。逐項代回可見 $u^{*\prime}=2-2x$、$u^{*\prime\prime}=-2$，故 $u^*(0)=0$、$u^{*\prime}(1)=0$，而
$$
-u^{*\prime\prime}+(u^*)^3-g
=2+[x(2-x)]^3-\bigl(2+[x(2-x)]^3\bigr)=0.
$$
這不只是形式上列出方程，還給出符合混合端點條件的解。由 $s\mapsto s^4/4$ 的凸性及平方項的展開，對任意同樣滿足左端條件的擾動 $v$，一階交叉項因駐點條件為零，其餘部分非負；平方導數項為零時 $v$ 為常數，再由 $v(0)=0$ 得 $v=0$。因此此特定資料下的 $u^*$ 也是嚴格全域最小值，而「滿足 Euler–Lagrange 方程」本身並不是一般問題的全域最小證明。

## 實作與程式

令 $N\ge2$ 為 $[0,1]$ 的**分段數**，$h=1/N$。零 Dirichlet 端點為 $U_0=U_N=0$，自由向量 $U=(U_1,\ldots,U_{N-1})$ 只有 $N-1$ 個分量；$f$ 亦只在這些內點取樣。離散能量定義為
$$
J_h(U)=\frac1{2h}\sum_{i=0}^{N-1}(U_{i+1}-U_i)^2
-h\sum_{j=1}^{N-1}f_jU_j.
$$
逐一收集含 $U_j$ 的兩段平方項，可得歐氏座標梯度
$$
(\nabla_{\mathrm E}J_h)_j
=\frac{2U_j-U_{j-1}-U_{j+1}}h-hf_j.
$$
因此對同形狀方向 $V$，$DJ_h(U)[V]=(\nabla_{\mathrm E}J_h)^TV$，**不再額外乘 $h$**。若選離散內積 $\langle V,W\rangle_M=V^TMW$、$M=hI$，其梯度 $g_M$ 則滿足 $Mg_M=\nabla_{\mathrm E}J_h$。連續導數、歐氏座標梯度及依質量矩陣定義的梯度，必須連同各自的內積一起標記。

為連接相場能量，再取 $n\ge3$ 個**週期節點**，下標按模 $n$ 計算，並定義
$$
F_h(\phi)=h\sum_{i=0}^{n-1}\frac{(\phi_i^2-1)^2}{4}
+\frac{\kappa}{2h}\sum_{i=0}^{n-1}(\phi_{i+1}-\phi_i)^2,
\qquad \kappa\ge0.
$$
設 $(L_h\phi)_i=(\phi_{i-1}-2\phi_i+\phi_{i+1})/h^2$，則
$$
\nabla_{\mathrm E}F_h
=h(\phi^3-\phi-\kappa L_h\phi).
$$
這裡的 $h$ 來自已定義的離散能量，而不是可任意套用的「連續梯度轉換係數」。下面只核對靜態能量及方向導數，沒有執行相場的時間演化。

```python
import numpy as np

def vector(name, values, size=None, minimum=1):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size < minimum:
        raise ValueError(f"{name} must be a 1-D array of length >= {minimum}")
    if size is not None and a.size != size:
        raise ValueError(f"{name} has the wrong length")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite values")
    return a

def positive_h(h):
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    return h

def dirichlet_data(U, f, h):
    h = positive_h(h)
    U = vector("U", U)
    f = vector("f", f, size=U.size)
    return U, f, h

def energy(U, f, h):
    U, f, h = dirichlet_data(U, f, h)
    full = np.r_[0.0, U, 0.0]
    return float(np.dot(np.diff(full), np.diff(full)) / (2*h)
                 - h*np.dot(f, U))

def gradient(U, f, h):
    U, f, h = dirichlet_data(U, f, h)
    full = np.r_[0.0, U, 0.0]
    return (2*U - full[:-2] - full[2:]) / h - h*f

def solve_dirichlet(f, h):
    h = positive_h(h)
    f = vector("f", f)
    m = f.size
    A = np.diag(np.full(m, 2.0/h))
    A += np.diag(np.full(m-1, -1.0/h), 1)
    A += np.diag(np.full(m-1, -1.0/h), -1)
    return np.linalg.solve(A, h*f)

def phase_data(phi, h, kappa):
    phi = vector("phi", phi, minimum=3)
    h = positive_h(h)
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    return phi, h, kappa

def phase_energy(phi, h, kappa):
    phi, h, kappa = phase_data(phi, h, kappa)
    differences = np.roll(phi, -1) - phi
    return float(h*np.sum((phi**2 - 1)**2)/4
                 + kappa*np.dot(differences, differences)/(2*h))

def phase_gradient(phi, h, kappa):
    phi, h, kappa = phase_data(phi, h, kappa)
    lap = (np.roll(phi, 1) - 2*phi + np.roll(phi, -1))/h**2
    return h*(phi**3 - phi - kappa*lap)

def central_direction(energy_fn, x, direction, epsilon):
    x = vector("x", x)
    direction = vector("direction", direction, size=x.size)
    epsilon = positive_h(epsilon)
    return (energy_fn(x + epsilon*direction)
            - energy_fn(x - epsilon*direction))/(2*epsilon)

# 以下為可自行執行的核對範例；本章沒有宣稱已執行。
N = 20
h = 1.0/N
x = np.arange(1, N)*h
f = np.pi**2*np.sin(np.pi*x)
solution = solve_dirichlet(f, h)
U = 0.2 + 0.1*np.sin(2*np.pi*x)  # 非駐點
V = np.cos(3*np.pi*x)           # 與 U 同形狀
fd = central_direction(lambda z: energy(z, f, h), U, V, 1e-5)
analytic = np.dot(gradient(U, f, h), V)
print("solution residual:", np.linalg.norm(gradient(solution, f, h)))
print("Dirichlet directional discrepancy:", abs(fd - analytic))

phi = np.array([-0.8, -0.1, 0.4, 0.9, 0.2])
W = np.array([1.0, -2.0, 0.5, 1.5, -1.0])
hp, kappa = 1.0/phi.size, 0.03
fd_phase = central_direction(
    lambda z: phase_energy(z, hp, kappa), phi, W, 1e-5)
print("phase directional discrepancy:",
      abs(fd_phase - np.dot(phase_gradient(phi, hp, kappa), W)))
```

陣列在程式中以 NumPy 一維形狀儲存，`np.dot(gradient(...), V)` 表示數學上的行向量作用於方向列向量；一維陣列的 `.T` 並不會改變形狀。程式刻意選非駐點核對方向導數，避免在兩邊都近零時得到缺乏辨識力的比較。

## 測試與預期結果

以下是**未執行的預期檢查**，不是測試紀錄。正常測試先檢查 `solution` 與 `f` 均長 $N-1$，再檢查 `gradient(solution, f, h)` 的範數在合理浮點容差內接近零。解析解的節點值為 $\sin(\pi x_j)$；可隨分段數增加比較最大節點誤差，但有限次比較不能證明收斂階。方向導數測試應比較兩個有符號數值，再觀察其差額；中央差分的截斷誤差與浮點消去誤差都會受步長影響，不能要求每個步長都愈小愈好。週期相場測試同樣須確認 `W.shape == phi.shape`，並把差分結果與 `np.dot(phase_gradient(phi, hp, kappa), W)` 比較。

邊界測試方面，Dirichlet 的最小支援分段數是 $N=2$：只有一個內點，`solve_dirichlet(np.array([2.0]), 0.5)` 對應一階矩陣，`gradient` 仍須正確計入兩個零端點。週期相場只支援至少三個節點，以免本例的左右鄰點配置退化；常數 $\phi_i=1$ 時，兩項的導數均為零，可用作另一個邊界核對，但不應只用這個駐點檢查差分。

故障測試可逐一向公開函數提供負 $h$、零 $h$、含 `NaN` 的資料、二維 `U`、與 `U` 長度不同的 `f`、長度少於三的 `phi`，以及負 $\kappa$；預期均拋出 `ValueError`。例如 `energy(np.array([1.0]), np.array([1.0, 2.0]), 0.5)` 必須拒絕長度不合，而非讓陣列廣播掩蓋錯誤。即使這些輸入檢查與方向差分都符合預期，也只能支持指定實作的一致性，不能取代連續 Euler–Lagrange 推導或離散收斂證明。

## 反例與常見陷阱

**駐點可能是最大值或鞍點。** 在零端點擾動空間上，$J[u]=-\frac12\int_0^1(u')^2\,dx$ 於 $u=0$ 的第一變分為零，但任何非零擾動都令能量下降，所以它是嚴格全域最大值，而非最小值。若二階形式同時有正、負方向，駐點甚至沒有單一的極值型態；習題將給出可直接計算的例子。

**逐方向正定不是一致下界。** 無限維空間中，即使對每個非零方向都有 $D^2J(u)[v,v]>0$，其與 $\|v\|_X^2$ 的比值仍可能沿某些方向序列趨近零。此時不能套用本章的小命題。另一方面，一致強制性只是該命題的充分假設，並非所有嚴格極小值的必要條件；例如四次能量在原點可以嚴格極小，第二變分卻為零。

**梯度不能脫離內積。** $DJ_h(U)[V]$ 是一個數；$\nabla_{\mathrm E}J_h$ 是在歐氏內積下代表它的座標向量；若改選 $M=hI$，代表向量成為 $g_M=M^{-1}\nabla_{\mathrm E}J_h$。因此錯誤的式子 $DJ_h(U)[V]=h(\nabla_{\mathrm E}J_h)^TV$ 會重複計入權重。連續 $L^2$ 或 $H^1$ 梯度也須先證明相應 Riesz 代表存在，不能宣稱任何泛函的 $L^2$ 梯度「就是函數本身」。

**邊界與格點不可混用。** 固定左端的例子只在右端推出自然條件。零 Dirichlet 節點配置有 $N-1$ 個未知量；零通量的 cell 中心配置則把未知量放在每個 cell 中心，端點通量由邊界面指定。兩種配置的矩陣、求積權重與可解性條件不同，不能只改矩陣首末對角元卻保留原本的節點資料解釋。

## AI、幾何與養殖案例

第三卷的相場應用可把本章的週期能量視為一個**靜態離散泛函**。局部勢能偏好 $\phi$ 靠近 $1$ 或 $-1$，相鄰差分項懲罰劇烈空間變化。其歐氏梯度由每一個網格自由度的偏導組成；若用離散 $L^2$ 內積 $h\sum_i a_ib_i$，梯度則是歐氏梯度除以 $h$。計算第一變分或驗證這兩種表示的對應，並未證明任何 Allen–Cahn 或 Cahn–Hilliard 時間步進的穩定性、質量守恆或物理適用性。

一個與養殖資料有關、但完全合成且無因次的例子，是以平滑場 $u(x)$ 表示某項已標準化的空間偏差，令
$$
J[u]=\frac12\int_0^1\bigl((u-d)^2+\alpha(u')^2\bigr)\,dx,
\qquad \alpha>0.
$$
這裡 $x$ 是除以池段長度後的無因次位置，$u$ 與目標 $d$ 都已除以同一個參考尺度；$\alpha$ 是模型設定，不是已量測的物理常數。若兩端固定，容許擾動在兩端為零；若端點自由且解足夠光滑，第一變分給出內部方程 $u-d-\alpha u''=0$ 及兩端的自然條件 $u'=0$。由平方項可見這是凸能量，但模型的數學性質不等於現場校準、設備控制能力或操作安全。

AI 工具在此最多唯讀整理所採用的無因次尺度、端點假設、能量式、方向導數核對與未驗證項目。若原始感測量帶有不同 SI 單位，須先分別記錄單位及參考尺度，才可形成上述無因次量；不能把不同單位的偏差直接相加，也不能讓模型輸出自行改變設備、投餌或加藥。

## 習題

1. **手算。** 在 $u(0)=u(1)=0$ 下，求 $J[u]=\int_0^1((u')^2-u)\,dx$ 的 Euler–Lagrange 方程與光滑駐點，再用能量差判定其極值型態。

2. **程式與邊界。** 改用 $N\ge2$ 個等寬 cell 的中心值 $U_i$，離散零通量問題 $-u''=f$。寫出左右邊界面零通量所對應的矩陣 $A$、相容條件，以及固定離散平均為零的增廣系統。提供自足程式，並說明對常數非零右端應如何處理；不得用加入 $\epsilon I$ 冒充原問題。

3. **反例。** 在 $H^1_0(0,1)$ 上令
   $J[u]=\frac12\int_0^1((u')^2-2\pi^2u^2)\,dx$。證明 $u=0$ 是駐點，並找出一個使第二變分為負、另一個使其為正的方向。

4. **整合。** 固定零端點，取 $J[u]=\frac12\int_0^1(u')^2\,dx$。對足夠光滑且零端點的 $u$，分別在 $L^2$ 內積與 $\langle a,b\rangle_{H^1}=\int_0^1(ab+a'b')\,dx$ 下定義梯度。寫出兩個 Riesz 表示所滿足的關係，並說明為何不能無條件把兩個梯度視為同一函數。

## 習題解答

1. 因 $L_u=-1$、$L_{u'}=2u'$，Euler–Lagrange 方程是 $-1-2u''=0$，即 $u''=-1/2$。積分並代入兩個零端點，得到 $u^*(x)=x(1-x)/4$。對任意零端點擾動 $v$，展開能量並用駐點方程消去交叉項：
   $J[u^*+v]-J[u^*]=\int_0^1(v')^2\,dx$。非零 $v$ 不可能在零端點條件下導數恆為零，因此這是嚴格全域最小值。

2. 設 $h=1/N$，未知數 $U_0,\ldots,U_{N-1}$ 位於 cell 中心。內部面的通量由相鄰中心差分表示，兩個外邊界面的通量指定為零。於是 $A=K/h^2$，其中 $K$ 的首末對角元為 $1$、其餘對角元為 $2$、相鄰非對角元為 $-1$；例如 $N=3$ 時
   $$
   K=\begin{pmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{pmatrix}.
   $$
   每一橫列的元素和為零，故 $A\mathbf1=0$；對 $AU=f$ 求和，必要條件是 $h\sum_i f_i=0$。由
   $U^TKU=\sum_{i=0}^{N-2}(U_{i+1}-U_i)^2$ 可知零空間恰為常數向量，因此相容條件亦充分。平均零條件 $h\mathbf1^TU=0$ 選出唯一代表。下列程式採用容差判定浮點資料的相容性；超過容差時明確拒絕，而不暗中改動右端：
   
   ```python
   import numpy as np

   def solve_neumann_cells(f, h, tolerance=1e-12):
       a = np.asarray(f, dtype=float)
       if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)):
           raise ValueError("f must be a finite 1-D cell array, length >= 2")
       h = float(h)
       tolerance = float(tolerance)
       if not np.isfinite(h) or h <= 0:
           raise ValueError("h must be finite and positive")
       if not np.isfinite(tolerance) or tolerance < 0:
           raise ValueError("tolerance must be finite and nonnegative")
       if not np.isclose(a.size*h, 1.0, rtol=0, atol=tolerance):
           raise ValueError("cells must cover [0, 1]")
       if abs(h*np.sum(a)) > tolerance:
           raise ValueError("zero-flux right-hand side is incompatible")
       n = a.size
       K = np.diag(np.r_[1.0, np.full(n-2, 2.0), 1.0])
       K += np.diag(np.full(n-1, -1.0), 1)
       K += np.diag(np.full(n-1, -1.0), -1)
       A = K/h**2
       c = np.full(n, h)
       augmented = np.block([[A, c[:, None]],
                             [c[None, :], np.zeros((1, 1))]])
       answer = np.linalg.solve(augmented, np.r_[a, 0.0])
       return answer[:-1]

   # 預期：零平均、相容的右端可求解。
   u = solve_neumann_cells(np.array([1.0, -1.0]), 0.5)
   # 預期：常數非零右端被拒絕，而不是產生原問題的解。
   try:
       solve_neumann_cells(np.array([1.0, 1.0]), 0.5)
   except ValueError as error:
       print(error)
   ```
   
   增廣系統的最後一個方程是 $c^TU=0$。對**精確相容**的資料，第一組方程中的乘數為零；使用浮點容差時仍應另外檢查 $\|AU-f\|$ 與 $|c^TU|$，不可只因線性求解器返回向量就宣稱滿足原方程。加入 $\epsilon I$ 會改變方程，並不是另一種施加平均零條件的方法。

3. 此泛函是二次型，所以 $DJ(0)[v]=0$ 對所有 $v\in H^1_0(0,1)$ 成立，且
   $D^2J(0)[v,v]=\int_0^1((v')^2-2\pi^2v^2)\,dx$。取 $v_1=\sin(\pi x)$，利用正弦、餘弦平方的積分均為 $1/2$，得到 $D^2J(0)[v_1,v_1]=-\pi^2/2$。取 $v_2=\sin(2\pi x)$，則結果為 $\pi^2$。由 $J[tv_i]-J[0]=t^2D^2J(0)[v_i,v_i]/2$，任意小的非零 $t$ 都能分別產生較低與較高的能量，故零函數是鞍點。

4. 在零端點擾動空間中，$DJ(u)[v]=\int_0^1u'v'\,dx$。若 $u$ 夠光滑且 $u''\in L^2$，分部積分給出 $\int_0^1(-u'')v\,dx$，所以此條件下的 $L^2$ 代表元為 $g_{L^2}=-u''$。另一方面，$H^1_0$ 配上題定的 $H^1$ 內積是 Hilbert 空間；Riesz 表示給出唯一 $g_{H^1}\in H^1_0$，滿足
   $$
   \int_0^1(g_{H^1}v+g_{H^1}'v')\,dx
   =\int_0^1u'v'\,dx\qquad(v\in H^1_0(0,1)).
   $$
   若再有足夠正則性，可把它寫成弱邊值方程 $(I-\partial_{xx})g_{H^1}=-u''$、$g_{H^1}(0)=g_{H^1}(1)=0$。兩個代表元作用於方向後給出相同導數，但定義它們的內積不同；對一般 $u\in H^1_0$，$-u''$ 甚至未必屬於 $L^2$，此時不能先假設 $L^2$ 梯度存在。

## 本章小結

變分推導必須先指定容許函數、擾動空間與範數。第一變分在所有容許方向為零，是內點極小的必要條件；分部積分給出內部 Euler–Lagrange 方程，自由端才另外給出自然邊界條件。第二變分的一致強制性連同駐點及二階小餘項，提供嚴格局部極小的充分條件；特定凸能量的全域性則可由能量差直接證明。

離散實作應從所寫下的能量重新求導：歐氏方向導數是座標梯度與同形狀方向的內積，改用離散 $L^2$ 內積則要改變梯度代表。正常、邊界與故障輸入及非駐點方向差分均有明確的預期檢查，但本章沒有執行程式；有限次數值核對也不構成連續定理或收斂性的證明。

## 參考來源

- Daniel Liberzon，*Calculus of Variations and Optimal Control Theory*，〈[First variation](https://liberzon.csl.illinois.edu/teaching/cvoc/node15.html)〉、〈[Euler–Lagrange equation](https://liberzon.csl.illinois.edu/teaching/cvoc/node28.html)〉與〈[Variable-endpoint problems](https://liberzon.csl.illinois.edu/teaching/cvoc/node32.html)〉。主編已取得所列網頁並局部核對自由端點的容許擾動與自然條件；此處不表示已逐條查核全書。
- Jiří Lebl，[*Basic Analysis* 教材入口](https://www.jirka.org/ra/)；用於分析先備概念的延伸閱讀，未宣稱已查核完整教材內容。
- [MIT OCW 18.100A Real Analysis](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/) 與 [MIT OCW 18.02SC Multivariable Calculus](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/)：先備分析與多變量微積分的課程入口；已取得的是課程概要，非逐條定理查證。
