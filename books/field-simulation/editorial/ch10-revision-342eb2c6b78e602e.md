# 第10章 有限體積與守恆通量

## 學習目標與先備知識

本章建立有限體積法的核心觀念：把守恆律積分於每個控制體，並以面通量更新 cell average。完成本章後，讀者應能：

1. 從局部守恆律推導一維與二維有限體積更新式。
2. 分辨 cell average、面通量、面積、體積與 ghost cell。
3. 用共享面通量離散變係數擴散，理解週期及零通量邊界下總量守恆的來源。
4. 撰寫小型 NumPy 程式，檢查質量平衡及輸入錯誤；知道守恆不等於穩定、能量下降或非負。

先備知識是積分、微分、向量通量及基本 Python。只討論經典連續介質與小格網科學計算；範例是合成模型，不是工業 CFD 或現場養殖預測。

## 問題與直覺

考慮濃度 $c(\mathbf{x},t)$。有限體積法不把它只看成格點上的點值，而是以控制體 $V_i$ 內的平均值代表數值未知量：

$$
\bar c_i(t)=\frac{1}{|V_i|}\int_{V_i}c(\mathbf{x},t)\,dV.
$$

平均值乘體積才是該格的量。若 $c$ 的單位為 $\mathrm{kg/m^3}$，則三維格內質量為 $\bar c_i|V_i|$，單位為 $\mathrm{kg}$；二維例子若代表每單位厚度，則格面積乘濃度給出每單位厚度的質量。

有限體積法的關鍵不是「每個格子各自估算導數」，而是相鄰格子共用同一個面通量。某面流出一格，就必須是鄰格等量流入；內部傳輸因此成對抵消。邊界面則沒有相鄰控制體，通過它的通量會改變計算域內總量。

## 數學與物理推導

令 $\mathbf{J}$ 為向外傳輸通量密度，$s$ 為體積來源項。局部守恆律寫成

$$
\frac{\partial c}{\partial t}+\nabla\cdot\mathbf{J}=s.
$$

對控制體 $V_i$ 積分，並用散度定理：

$$
\frac{d}{dt}\int_{V_i}c\,dV
=-\int_{\partial V_i}\mathbf{J}\cdot\mathbf{n}\,dA
+\int_{V_i}s\,dV.
$$

把第 $f$ 個面向外的積分通量記為 $F_{if}$，並以 $S_i$ 表示體積來源積分，就得到半離散式

$$
\frac{d}{dt}(\bar c_i|V_i|)
=-\sum_{f\in\partial V_i}F_{if}+S_i.
$$

若相鄰格 $i,j$ 共用面，定義 $F_{ij}$ 為由 $i$ 指向 $j$ 的整面通量，則 $F_{ji}=-F_{ij}$。因此全域求和時，內部面逐對抵消：

$$
\frac{d}{dt}\sum_i\bar c_i|V_i|
=-\sum_{f\ \mathrm{on\ boundary}}F_f+\sum_iS_i.
$$

零通量或週期邊界、且無來源時，總量應守恆。週期邊界把域的一側與另一側視為相鄰，跨接面的通量也必須成對抵消。這個結論來自離散更新的代數結構，不是因為圖看起來平滑。

以 Fick 定律 $\mathbf{J}=-D\nabla c$ 描述擴散，$D$ 單位為 $\mathrm{m^2/s}$。在一維均勻格上，中心面近似給出通量密度

$$
J_{i+1/2}=-D_{i+1/2}\frac{c_{i+1}-c_i}{\Delta x}.
$$

若面積為 $A$，整個面的通量是 $F_{i+1/2}=AJ_{i+1/2}$。其中 $J$ 的單位為 $\mathrm{kg/(m^2s)}$，而 $F$ 的單位為 $\mathrm{kg/s}$。

若 $D$ 在格間變化，直接取算術平均並不總是符合串聯擴散阻力。對兩半格串聯，面通量應滿足相同通量穿越兩段；其等效係數為調和平均：

$$
D_{i+1/2}=
\frac{\Delta x}{(\Delta x/2)/D_i+(\Delta x/2)/D_{i+1}}
=\frac{2D_iD_{i+1}}{D_i+D_{i+1}}.
$$

此式假設兩段長度相等、每段係數為正且格面兩側連續傳輸。係數跳躍、非正係數或非正交網格需另選適當離散，不能把這個公式無條件套用。

對長度 $\Delta x$、截面積 $A$ 的格子，體積為 $A\Delta x$，故

$$
\frac{d\bar c_i}{dt}
=-\frac{F_{i+1/2}-F_{i-1/2}}{A\Delta x}+\bar s_i.
$$

面積不可漏掉；在同截面的一維例子中面積可約掉，但有來源或報告實際總質量時不可任意省略。

時間離散也影響數值性質。顯式 Euler 更新為 $\bar c_i^{n+1}=\bar c_i^n+\Delta t\,R_i^n$，其中 $R_i$ 是通量差與來源除以體積所得的時間變化率。只要同一內部面通量一正一負地加入兩格，離散總量平衡成立；但大時間步仍可能振盪或產生負值。守恆不是穩定、正性、能量下降或物理可信的同義詞。

## 逐步手算例題

### 例一：週期一維擴散更新

取四個等寬格，$\Delta x=1\ \mathrm{m}$、單位截面積，$D=1\ \mathrm{m^2/s}$，無來源。初始 cell average 為

$$
(c_0,c_1,c_2,c_3)=(1,0,0,0)\ \mathrm{kg/m^3}.
$$

對面 $i+1/2$ 定義由左格流向右格為正的通量密度：

$$
J_{i+1/2}=-D\frac{c_{i+1}-c_i}{\Delta x}.
$$

週期條件令 $c_4=c_0$。因面積取 $1\ \mathrm{m^2}$，整面通量 $F_{i+1/2}=AJ_{i+1/2}$ 的數值與通量密度相同，但單位不同。逐面計算整面通量：

- $F_{1/2}=1\ \mathrm{kg/s}$；
- $F_{3/2}=0\ \mathrm{kg/s}$；
- $F_{5/2}=0\ \mathrm{kg/s}$；
- $F_{7/2}=-1\ \mathrm{kg/s}$。

顯式 Euler 取 $\Delta t=0.1\ \mathrm{s}$。因單位截面積與格寬皆為一，每格體積為 $1\ \mathrm{m^3}$。逐格更新：

$$
c_i^{n+1}=c_i^n-\Delta t\,(F_{i+1/2}-F_{i-1/2})/(A\Delta x),
$$

得到 $(0.8,0.1,0,0.1)\ \mathrm{kg/m^3}$。總量由 $1$ 變成 $1$（乘以每格 $1\ \mathrm{m^3}$）；出現在第一格的損失，正好等於兩側相鄰格的增加。此更新符合一維均勻擴散的非負充分條件 $\Delta t D/\Delta x^2\le 1/2$。滿足該條件只支持此特定顯式格式下的穩定性與凸組合性，不能推廣成所有有限體積格式的通用保證。

### 例二：變係數兩格交換

兩格各長 $1\ \mathrm{m}$、面積 $1\ \mathrm{m^2}$，係數分別為 $D_0=1$、$D_1=3\ \mathrm{m^2/s}$。兩格中心間距為 $1\ \mathrm{m}$，故面等效係數為 $D_{1/2}=2(1)(3)/(1+3)=1.5\ \mathrm{m^2/s}$。令 $c_0=2$、$c_1=0\ \mathrm{kg/m^3}$，則從格 0 到格 1 的通量密度為 $J_{1/2}=3\ \mathrm{kg/(m^2s)}$，整面通量為 $F_{1/2}=AJ_{1/2}=3\ \mathrm{kg/s}$。零通量外邊界、無來源下，取 $\Delta t=0.1\ \mathrm{s}$：

$$
c_0^{n+1}=2-0.1(3)/(1\times1)=1.7,\qquad
c_1^{n+1}=0+0.1(3)/(1\times1)=0.3.
$$

每格體積為 $1\ \mathrm{m^3}$，總量仍是 $2\ \mathrm{kg}$。若兩格面積、距離或體積不同，更新必須除以各自體積；兩格的濃度變化率通常不相等，但質量改變仍等量反向。

## 實作與程式

以下完整程式只依賴 Python 3.10+ 與 NumPy，實作一維週期擴散與二維週期擴散的單步顯式更新。二維陣列採 `q[j, i]`，形狀為 `(Ny, Nx)`；$i$ 沿 $+X$、$j$ 沿 $+Y$。此例是每單位厚度的二維模型。週期面透過索引回捲取得，沒有 ghost cell；總量只對每個 cell average 乘其 cell 面積計算一次。

```python
import numpy as np


def require_finite(name, value):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 必須全部為有限值")
    return a


def check_scalar_diffusion_inputs(q, D, dt, dx, dy=None):
    q = require_finite("q", q)
    D = require_finite("D", D)
    dt = float(require_finite("dt", dt))
    dx = float(require_finite("dx", dx))
    if q.ndim not in (1, 2) or min(q.shape) < 2:
        raise ValueError("q 必須是每個方向至少兩格的一維或二維陣列")
    if D.ndim != 0 and D.shape != q.shape:
        raise ValueError("D 必須是純量或與 q 同形狀")
    if np.any(D <= 0) or dt <= 0 or dx <= 0:
        raise ValueError("D、dt、dx 必須為正")
    if q.ndim == 2:
        dy = float(require_finite("dy", dy))
        if dy <= 0:
            raise ValueError("dy 必須為正")
    return q, D, dt, dx, dy if q.ndim == 2 else None


def face_harmonic(a, b):
    return 2.0 * a * b / (a + b)


def diffuse_periodic_step(q, D, dt, dx, dy=None):
    q, D, dt, dx, dy = check_scalar_diffusion_inputs(
        q, D, dt, dx, dy
    )
    if q.ndim == 1:
        dleft = np.roll(q, 1) - q
        dright = np.roll(q, -1) - q
        if np.ndim(D) == 0:
            dl = dr = D
        else:
            dl = face_harmonic(np.roll(D, 1), D)
            dr = face_harmonic(D, np.roll(D, -1))
        rate = (dr * dright + dl * dleft) / dx**2
        dt_limit = 0.5 * dx**2 / float(np.max(D))
    else:
        if np.ndim(D) == 0:
            Dleft = Dright = Dup = Ddown = D
        else:
            Dleft = face_harmonic(np.roll(D, 1, axis=1), D)
            Dright = face_harmonic(D, np.roll(D, -1, axis=1))
            Ddown = face_harmonic(np.roll(D, 1, axis=0), D)
            Dup = face_harmonic(D, np.roll(D, -1, axis=0))
        gx = (Dright * (np.roll(q, -1, axis=1) - q)
              - Dleft * (q - np.roll(q, 1, axis=1))) / dx**2
        gy = (Dup * (np.roll(q, -1, axis=0) - q)
              - Ddown * (q - np.roll(q, 1, axis=0))) / dy**2
        rate = gx + gy
        dt_limit = 0.5 / (float(np.max(D))
                          * (1.0 / dx**2 + 1.0 / dy**2))
    return q + dt * rate, dt_limit


def total_amount(q, dx, dy=None):
    q = require_finite("q", q)
    if q.ndim == 1:
        return float(np.sum(q) * dx)
    if q.ndim == 2 and dy is not None:
        return float(np.sum(q) * dx * dy)
    raise ValueError("二維總量需提供 dy")
```

一維的 `dleft` 定義為左鄰格減本格，`dright` 定義為右鄰格減本格。因此左右面貢獻都以正號加入更新率：

$$
\frac{D_{i+1/2}(c_{i+1}-c_i)+D_{i-1/2}(c_{i-1}-c_i)}{\Delta x^2}.
$$

程式中的 `rate = (dr * dright + dl * dleft) / dx**2` 正是此式；若把左側差值再以減號相減，會造成符號錯誤，並破壞預期的擴散更新。

二維程式再加上 $Y$ 方向的面通量差。若 $D$ 是格中心係數陣列，`face_harmonic` 形成相鄰格間的等距調和平均。常係數情況下顯式擴散非負充分條件為

$$
\Delta t\,D\left(\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}\right)\le\frac12.
$$

二維程式的檢查採最大係數給出保守時間步提示；即使符合提示，對具跳躍係數、額外來源項或其他空間離散的格式，也不應未推導便宣稱其所有性質成立。

二維週期格的總物理量按每單位厚度計算為 $\sum_{j,i}q_{j,i}\Delta x\Delta y$。週期接縫由回捲陣列索引連接；程式不建立兩份週期面，所以沒有重複計入同一物理面。若採 MAC 面速度陣列或明確儲存兩側接縫面，需遵守相應面配置並同步兩份資料，但總量仍只按 cell average 乘面積計算。

## 測試與預期結果

以下測試接在上方程式之後。未在此執行，結果是依離散公式推得的預期行為。

```python
# 正常測試：四格週期擴散，總量維持 1
q0 = np.array([1.0, 0.0, 0.0, 0.0])
q1, limit = diffuse_periodic_step(q0, 1.0, 0.1, 1.0)
assert np.allclose(q1, [0.8, 0.1, 0.0, 0.1])
assert np.isclose(total_amount(q1, 1.0), 1.0)
assert 0.1 <= limit

# 二維正常測試：任意週期面內交換不改變總量
q2 = np.zeros((3, 4))
q2[1, 1] = 1.0
q2n, limit2 = diffuse_periodic_step(q2, 1.0, 0.05, 1.0, 1.0)
assert np.isclose(total_amount(q2n, 1.0, 1.0), 1.0)
assert 0.05 <= limit2

# 邊界測試：均勻場在週期條件下不變
uniform = np.full((3, 4), 2.5)
uniform_n, _ = diffuse_periodic_step(uniform, 0.2, 0.1, 1.0, 1.0)
assert np.allclose(uniform_n, uniform)

# 故障測試：NaN、負係數與零網格間距都應拒絕
for args in [
    (np.array([1.0, np.nan]), 1.0, 0.1, 1.0),
    (np.array([1.0, 0.0]), -1.0, 0.1, 1.0),
    (np.array([1.0, 0.0]), 1.0, 0.1, 0.0),
]:
    try:
        diffuse_periodic_step(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("預期拒絕不一致或非有限輸入")
```

正常測試預期第一格得到 $0.8$、兩側鄰格各得到 $0.1$；總量不變。二維點源測試預期每個內部面交換一正一負，總量仍相同。均勻場所有梯度為零，因此不變。故障測試預期在更新前拒絕 NaN、非正擴散係數或非正網格間距，而非繼續產生不可解讀的結果。

## 除錯與常見陷阱

- **把中心值當 cell average。** 平滑場上兩者可能接近，但有限體積守恆量是平均值乘體積；非均勻場或粗格網下不能混為一談。
- **內部面重複但不相反地計算。** 若各格獨立估通量，鄰格可能各自得到不同的面值，內部傳輸就不再抵消。實作上應建立唯一共享面通量，或確認相鄰更新使用完全相同的數值。
- **混淆通量與通量密度。** $J$ 是每面積每時間的傳輸率，$F=AJ$ 才是整個面的傳輸率。缺少面積會造成量綱錯誤。
- **把 ghost cell 算進質量。** ghost cell 是邊界條件的延拓值，不是新的實體控制體。質量只對物理域 cell average 乘物理體積求和。
- **錯置外法向符號。** 控制體外法向在西、南面朝負座標方向，在東、北面朝正座標方向。最可靠的檢查是對某一格寫出「流入增加、流出減少」。
- **把守恆誤認為非負或穩定。** 不合適的大步長可以在總量正確時仍產生負濃度或振盪。不得用裁零或平滑遮掩這種錯誤；若要修正格式，須重新分析並報告修正前後的質量。
- **忽略來源和邊界收支。** 有來源時總量改變應等於來源積分與外邊界淨流入之和。比較質量時須把這些項列入帳目。
- **誤用變係數面值。** 調和平均適用於等距一維分層通量的簡化情形。若係數為零、格距不同、幾何複雜或方程含各向異性張量，必須重新推導面通量。

## 養殖與相場案例

在合成池域溶質模型中，可令 $c$ 為溶質濃度，單位 $\mathrm{kg/m^3}$；$D$ 以 $\mathrm{m^2/s}$ 表示，來源 $s$ 的單位為 $\mathrm{kg/(m^3s)}$。封閉池域可作零通量邊界的數值試驗。若池域有進出水，邊界通量必須由模型或資料明確給定，總量變化才可解讀為物理收支。合成係數只能驗證演算法流程，不代表現場物性或管理閾值。

相場模型亦能採有限體積空間離散。以本卷無因次慣例為例，$F=\int[W(\phi)+\kappa|\nabla\phi|^2/2]\,dV$，其中 $\phi$ 無因次、$\kappa>0$，化學勢為 $\mu=\phi^3-\phi-\kappa\Delta\phi$。Cahn–Hilliard 方程為 $\phi_t=\nabla\cdot(M\nabla\mu)$；在週期或適當無通量邊界下，通量形式帶來 $\phi$ 總量守恆。Allen–Cahn 方程 $\phi_t=-M\mu$ 通常不守恆 $\phi$ 總量。兩者的連續能量耗散不能直接推論任意有限體積時間步都能量下降，需檢查空間、時間離散及非線性求解誤差。養殖中的溶氧跨管理閾值是管理事件，不是物理相變；合成模擬與逼真渲染都不構成現場驗證。

## 習題

### 1. 手算：來源與收支

兩個等體積控制體各為 $2\ \mathrm{m^3}$，初始 cell average 為 $(1,0)\ \mathrm{kg/m^3}$。兩格間有向內部傳輸率 $0.4\ \mathrm{kg/s}$，方向由第一格到第二格。另有總量 $0.2\ \mathrm{kg/s}$ 的來源只作用於第一格。寫出一秒後兩格的質量及平均濃度，並核對全域收支。

### 2. 程式：二維週期擴散

使用本章程式建立 $5\times 7$ 的非均勻合成場與常係數 $D>0$。以符合程式回傳時間步提示的 $\Delta t$ 更新一次。應檢查什麼量，才能確認週期離散的全域守恆？請說明總量檢查和非負檢查分別回答什麼問題。

### 3. 反例：守恆不代表物理解

考慮兩格等體積、零通量外邊界，指定一個內部面更新，使兩格濃度由 $(1,0)$ 變成 $(-1,2)$。說明為何總量仍守恆、但結果不可信。這個反例能否單獨證明所有有限體積法都不穩定？

### 4. 整合：週期面與 ghost cell

某程式用 ghost cell 實作零通量邊界，卻把 ghost 值也加進總質量；另一版本以兩份數值儲存同一個週期接縫通量，且只在一側更新。分別說明錯誤如何發生，並提出收支檢查方式。

## 習題解答

### 1. 解答

初始質量為 $m_1=1\times2=2\ \mathrm{kg}$、$m_2=0$。一秒內內部通量使第一格減少 $0.4\ \mathrm{kg}$、第二格增加 $0.4\ \mathrm{kg}$；第一格來源再增加 $0.2\ \mathrm{kg}$。故

$$
m_1^{1}=2-0.4+0.2=1.8\ \mathrm{kg},\qquad
m_2^{1}=0+0.4=0.4\ \mathrm{kg}.
$$

平均濃度為 $c_1^1=1.8/2=0.9\ \mathrm{kg/m^3}$、$c_2^1=0.4/2=0.2\ \mathrm{kg/m^3}$。初始總質量為 $2\ \mathrm{kg}$，末態為 $2.2\ \mathrm{kg}$，恰等於一秒累積來源 $0.2\ \mathrm{kg}$。

### 2. 解答

令每格起初濃度為已指定的任意有限非均勻值，擴散係數為正純量；網格間距須與格數、單位一致，並傳入正的 $\Delta x,\Delta y,\Delta t$。更新前後比較

$$
M=\sum_{j,i}q_{j,i}\Delta x\Delta y.
$$

週期網格沒有外界淨通量，且無來源時，預期 $M^{n+1}=M^n$，只可能有浮點捨入量級差異。這是守恆檢查。檢查 $\min(q^{n+1})\ge0$ 則是非負性檢查，回答不同問題；總量正確仍可能有負值，也可能有局部振盪。應將兩者分開報告，不因一項通過便宣稱另一項通過。

### 3. 解答

兩格初始總和為 $1+0=1$，末態總和為 $-1+2=1$，所以等體積時總量守恆。但第一格變成負濃度，違反此濃度模型的非負物理解讀，並顯示該更新可能使用了不適當時間步或通量。此例只證明「守恆不蘊含非負」；它不構成所有有限體積法都不穩定的證明。格式、時間積分、網格及係數都會影響穩定性。

### 4. 解答

ghost cell 是用來封閉邊界差分或通量計算的虛構延拓，不是物理域的一部分。把它加入質量總和會多算不存在的體積；總量應只加物理格的 cell average 乘其體積。週期接縫兩份儲存若不一致，兩側離散可能採用不同通量；若只更新一側，接縫傳輸就沒有等量反向的相鄰更新，內部抵消失效。應將每一個物理面通量明確配對至其兩側控制體，並核對

$$
M^{n+1}-M^n
=\Delta t\left(\text{來源總量}-\text{向外邊界淨通量}\right).
$$

週期及零通量、無來源時右側為零；質量求和仍只遍歷物理格一次。

## 本章小結

有限體積法將局部守恆律積分於控制體，以 cell average 作為未知量，並用面通量改變各格總量。共享面通量在相鄰格間一進一出，故內部傳輸在全域總量中抵消；邊界通量與來源則決定總量如何改變。變係數擴散需要與幾何和傳輸阻力相符的面係數，本文的調和平均限於明確列出的簡化假設。

程式中的總量守恆檢查不能替代穩定性、非負性、能量耗散或物理驗證。使用者應明確指定網格、單位、邊界條件、來源、時間步與總量定義；不符合契約的輸入應拒絕，而不是靠裁零或平滑把錯誤藏起來。

## 參考來源

- [F1] FiPy，有限體積離散與邊界：<https://pages.nist.gov/fipy/en/latest/numerical/discret.html>
- [F2] FEniCSx，Poisson 與弱形式：<https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html>
- [F3] PETSc，線性系統求解器：<https://petsc.org/release/manual/ksp/>
- [F4] SciPy，稀疏線性代數 API：<https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>
- [F6] FiPy，Cahn–Hilliard 相分離示範：<https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html>

上述來源供查閱有限體積、求解器及相場相關方法；本章 NumPy 範例是依所列離散自行撰寫，未宣稱已執行或驗證其效能。