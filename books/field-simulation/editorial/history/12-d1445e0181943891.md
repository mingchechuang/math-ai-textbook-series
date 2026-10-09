# 第12章 平流擴散反應與分裂誤差

## 學習目標與先備知識

本章研究純量濃度場的平流—擴散—反應方程，重點在三種機制同時存在時的尺度競爭、邊界收支、剛性與算子分裂誤差。完成本章後，讀者應能：

1. 寫出守恆形式方程並核對SI單位。
2. 推導Péclet數與Damköhler數，說明所採參考尺度。
3. 區分平流流入／流出邊界與擴散通量邊界。
4. 以有限體積共享面通量維持離散守恆。
5. 解釋Lie與Strang分裂及交換子誤差。
6. 辨認反應剛性，理解穩定子步不等於小分裂誤差。
7. 分開檢查穩定性、守恆、非負性、能量與物理可信度。
8. 以NumPy CPU程式記錄邊界收支、反應收支與設定雜湊。
9. 以場誤差而非僅看總質量進行時間步長細化。

先備知識包括初邊值問題、有限體積法、上風通量、中心擴散及Taylor展開。本章採一維cell-centered網格；第$i$格中心為$x_i=(i+1/2)\Delta x$。二維推廣時使用`q[j,i]`，$i$沿$+X$、$j$沿$+Y$。

---

## 問題與直覺

設$c(x,t)$為濃度，單位$\mathrm{kg/m^3}$。一維守恆方程為

$$
\frac{\partial c}{\partial t}
+\frac{\partial(uc)}{\partial x}
=
\frac{\partial}{\partial x}
\left(D\frac{\partial c}{\partial x}\right)
+R(c,x,t).
$$

其中$u$單位為$\mathrm{m/s}$，$D$為$\mathrm{m^2/s}$，$R$為$\mathrm{kg/(m^3\,s)}$。若$u$為常數，可寫成

$$
c_t+uc_x=Dc_{xx}+R.
$$

變速度時，$\partial_x(uc)$與$uc_x$通常不相等，有限體積法應從守恆形式出發。

三種機制的作用不同：

- 平流搬運濃度，資訊傳播方向由速度決定。
- 擴散削弱梯度，使尖峰展寬。
- 反應在局部生成或消耗物質。

以一階衰減$R=-kc$為例，$k$的單位為$\mathrm{s^{-1}}$。若有體積來源$s$，

$$
R=s-kc,
$$

則$s$的單位為$\mathrm{kg/(m^3\,s)}$。邊界注入是面通量，不能與體積來源混用。

算子分裂把傳輸與反應分開計算，例如先傳輸再反應。然而先做甲再做乙通常不等於先做乙再做甲；次序差就是分裂誤差的來源。

---

## 數學與物理推導

### 控制體收支

令截面面積為$A$，第$i$格體積為$A\Delta x$，格內質量為

$$
m_i=c_iA\Delta x.
$$

定義沿$+X$為正的總通量

$$
J=uc-Dc_x,
$$

其單位為$\mathrm{kg/(m^2\,s)}$。控制體積分給出

$$
\frac{d(c_iA\Delta x)}{dt}
=
AJ_{i-1/2}-AJ_{i+1/2}+A\Delta xR_i,
$$

即

$$
\frac{dc_i}{dt}
=
-\frac{J_{i+1/2}-J_{i-1/2}}{\Delta x}+R_i.
$$

所有格相加時，內部共享面通量兩兩抵消：

$$
\frac{dM}{dt}
=
A(J_{1/2}-J_{N+1/2})
+\sum_iR_iA\Delta x,
$$

其中

$$
M=\sum_i c_iA\Delta x.
$$

ghost cell不屬於物理域，不能算入$M$。

### 無因次化、Péclet數與Damköhler數

取

$$
x=Lx^\ast,\qquad
t=\frac{L}{U}t^\ast,\qquad
u=Uu^\ast,\qquad
c=Cc^\ast.
$$

對$R=-kc$，可得

$$
\frac{\partial c^\ast}{\partial t^\ast}
+\frac{\partial(u^\ast c^\ast)}{\partial x^\ast}
=
\frac1{Pe}\frac{\partial^2c^\ast}{\partial x^{\ast2}}
-Da\,c^\ast,
$$

其中

$$
Pe=\frac{UL}{D},\qquad
Da=\frac{kL}{U}.
$$

$Pe$比較平流時間$L/U$與擴散時間$L^2/D$；$Da$比較平流時間與反應時間$1/k$。若以擴散時間為基準，則

$$
Da_D=\frac{kL^2}{D}=Pe\,Da.
$$

只報Damköhler數而不報參考時間是不完整的。另有網格Péclet數

$$
Pe_h=\frac{|u|\Delta x}{D},
$$

它衡量單格尺度的平流與擴散競爭，不等同域尺度$Pe$。

### 有限體積通量與邊界

內部面採上風平流與中心擴散：

$$
J_{i+1/2}
=
u c_{\mathrm{up}}
-D\frac{c_{i+1}-c_i}{\Delta x}.
$$

若$u>0$，$c_{\mathrm{up}}=c_i$；若$u<0$，$c_{\mathrm{up}}=c_{i+1}$。

本章程式限定$u>0$，左側給流入濃度$c_{\rm in}$，右側為平流流出，兩端擴散通量為零：

$$
J_{1/2}=uc_{\rm in},\qquad
J_{N+1/2}=uc_{N-1}.
$$

零擴散通量不表示總通量為零。若$u<0$，流入端應改到右側；沿用左側流入資料是問題定義錯誤。

### 穩定性與非負性

令

$$
C=\frac{|u|\Delta t}{\Delta x},\qquad
r=\frac{D\Delta t}{\Delta x^2}.
$$

一維常係數下，上風平流、中心擴散與顯式Euler的單調性充分條件為

$$
C+2r\le1.
$$

在非負初值及邊界資料下，此條件使Euler更新係數非負。後文使用二階SSPRK2；因它可寫成滿足相同步長限制之Euler步的凸組合，所以同樣可繼承非負性條件。這不是任意二階法的普遍性質。

數值穩定、非負與守恆仍是不同概念：守恆格式可能振盪，穩定格式也可能產生小負值，非負格式則未必高精度。

### Lie分裂、Strang分裂與交換子

把半離散系統寫成

$$
\frac{dc}{dt}=(A+B)c,
$$

其中$A$代表傳輸，$B$代表反應。精確演化為$e^{\Delta t(A+B)}$。Lie分裂之一為

$$
c^{n+1}=e^{\Delta tB}e^{\Delta tA}c^n.
$$

展開可得其局部主誤差含

$$
\frac{\Delta t^2}{2}(BA-AB)
=
\frac{\Delta t^2}{2}[B,A].
$$

因此一般具有二階局部誤差與一階全域誤差。

Strang分裂為

$$
c^{n+1}
=
e^{\Delta tB/2}
e^{\Delta tA}
e^{\Delta tB/2}c^n.
$$

若算子與解足夠光滑、邊界相容，而且每個子流以至少二階精度近似，其全域時間誤差一般為$O(\Delta t^2)$。

這個條件不能省略。若傳輸子問題使用一階Euler，外層雖排列成Strang形式，完整實作通常仍只有一階時間精度。本章程式因此改用二階SSPRK2傳輸，而不是以一階Euler冒充二階Strang實作。

即使如此，觀測二階仍可能受下列因素破壞：

- 非光滑初值或邊界資料；
- 剛性反應造成降階；
- 時間相依邊界未在正確子步時刻取值；
- 參考解不夠細；
- 空間誤差或浮點誤差主導；
- 外層步長尚未進入漸近區。

若$AB=BA$，分裂可精確交換。例如週期域上常係數傳輸與均勻衰減$B=-kI$交換。空間變化的$k(x)$、非線性反應與不相容邊界通常使交換子不為零。

### 剛性反應

對

$$
c_t=-kc,
$$

顯式Euler為

$$
c^{n+1}=(1-k\Delta t)c^n.
$$

線性穩定要求$k\Delta t\le2$，但非負性要求更嚴格的$k\Delta t\le1$。精確反應子步為

$$
c^{n+1}=e^{-k\Delta t}c^n,
$$

對任意非負$\Delta t$保持非負並衰減。然而精確反應子步只消除反應子問題的積分誤差，不能消除傳輸—反應交換子誤差。

---

## 逐步手算例題

### 例題一：尺度與步長

給定$L=10\,\mathrm{m}$、$u=0.020\,\mathrm{m/s}$、$D=10^{-3}\,\mathrm{m^2/s}$及$k=5\times10^{-3}\,\mathrm{s^{-1}}$。

時間尺度為

$$
\tau_a=\frac{L}{u}=500\,\mathrm{s},\qquad
\tau_d=\frac{L^2}{D}=10^5\,\mathrm{s},\qquad
\tau_r=\frac1k=200\,\mathrm{s}.
$$

故

$$
Pe=\frac{uL}{D}=200,\qquad
Da=\frac{kL}{u}=2.5.
$$

域尺度上平流強於擴散，而一次穿越期間反應不可忽略。

取$\Delta x=0.10\,\mathrm{m}$，則

$$
C=0.2\Delta t,\qquad 2r=0.2\Delta t.
$$

由$C+2r\le1$得

$$
\Delta t\le2.5\,\mathrm{s}.
$$

若反應也使用顯式Euler，非負性限制為$\Delta t\le1/k=200\,\mathrm{s}$，所以此例由傳輸限制步長。若$k$提高至$5\,\mathrm{s^{-1}}$，反應限制變成$0.2\,\mathrm{s}$，顯示剛性來源可以隨參數改變。

### 例題二：兩格分裂次序

設兩格、$A=\Delta x=1$、$u=1$、$D=0$、$c_{\rm in}=2$，

$$
c^n=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\Delta t=0.25,
\qquad
(k_0,k_1)=(0,2).
$$

傳輸面通量為

$$
J_{1/2}=2,\qquad J_{3/2}=1,\qquad J_{5/2}=0.
$$

一次Euler傳輸步給出

$$
c^\star=
\begin{bmatrix}
1.25\\
0.25
\end{bmatrix}.
$$

總量增加$0.5$，恰等於

$$
\Delta t(J_{1/2}-J_{5/2})=0.5.
$$

若先傳輸再反應，

$$
c^{n+1}
=
\begin{bmatrix}
1.25\\
0.25e^{-0.5}
\end{bmatrix}
\approx
\begin{bmatrix}
1.25\\
0.15163
\end{bmatrix}.
$$

若先反應再傳輸，初始第二格為零，反應不改變狀態，結果為

$$
c^{n+1}=
\begin{bmatrix}
1.25\\
0.25
\end{bmatrix}.
$$

兩者都能正確記錄收支，卻有不同空間分布；這是分裂誤差，不是守恆誤差。

Strang排列先做半步反應、完整傳輸、再半步反應，得到

$$
c^{n+1}
=
\begin{bmatrix}
1.25\\
0.25e^{-0.25}
\end{bmatrix}
\approx
\begin{bmatrix}
1.25\\
0.19470
\end{bmatrix}.
$$

此數值不能單憑位於兩個Lie結果之間就稱為真解，仍須步長細化或與參考解比較。

---

## 實作與程式

下列自足程式只依賴NumPy。傳輸使用SSPRK2；每個Euler階段都服從$C+2r\le0.9$。反應採精確指數更新。SSPRK2的邊界質量使用兩階段通量的梯形權重，因此離散質量收支與狀態更新一致。

```python
import json
import hashlib
import numpy as np


def cfg_hash(cfg):
    text = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def validate(c, k, L, u, D, c_in, area):
    c = np.asarray(c, float)
    k = np.asarray(k, float)
    scalars = np.array([L, u, D, c_in, area], float)
    if c.ndim != 1 or c.size < 2 or k.shape != c.shape:
        raise ValueError("c與k必須是相同形狀的一維陣列")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(k)):
        raise ValueError("c與k必須有限")
    if not np.all(np.isfinite(scalars)):
        raise ValueError("設定必須有限")
    if L <= 0 or area <= 0 or D < 0 or np.any(k < 0):
        raise ValueError("要求L>0、area>0、D>=0、k>=0")
    if u <= 0:
        raise ValueError("本實作限定u>0；負速度必須改由右側給流入")
    if c_in < 0 or np.any(c < 0):
        raise ValueError("初始與流入濃度必須非負")
    return c.copy(), k.copy()


def reaction_exact(c, k, dt, dx, area):
    m0 = area * dx * np.sum(c)
    out = c * np.exp(-k * dt)
    dm = area * dx * np.sum(out) - m0
    return out, dm


def flux(c, u, D, dx, c_in):
    J = np.empty(c.size + 1)
    J[0] = u * c_in                 # 左流入，零擴散通量
    J[-1] = u * c[-1]              # 右流出，零擴散通量
    J[1:-1] = u * c[:-1] - D * (c[1:] - c[:-1]) / dx
    return J


def rhs_transport(c, u, D, dx, c_in):
    J = flux(c, u, D, dx, c_in)
    rhs = -(J[1:] - J[:-1]) / dx
    boundary_rate = J[0] - J[-1]   # 每單位面積的質量率
    return rhs, boundary_rate


def transport_ssprk2(c, duration, dx, area, u, D, c_in,
                     safety=0.9):
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration必須是有限非負數")
    if duration == 0:
        return c.copy(), 0.0, 0

    rate = abs(u) / dx + 2.0 * D / dx**2
    nsub = max(1, int(np.ceil(duration * rate / safety)))
    h = duration / nsub

    out = c.copy()
    boundary_mass = 0.0

    for _ in range(nsub):
        f0, b0 = rhs_transport(out, u, D, dx, c_in)
        stage = out + h * f0

        f1, b1 = rhs_transport(stage, u, D, dx, c_in)
        new = 0.5 * out + 0.5 * (stage + h * f1)

        boundary_mass += area * h * 0.5 * (b0 + b1)

        if not np.all(np.isfinite(new)):
            raise FloatingPointError("傳輸產生非有限值")
        if np.min(new) < -1.0e-12:
            raise FloatingPointError(
                "產生顯著負值；不可裁零，請檢查步長與邊界"
            )
        out = new

    return out, boundary_mass, nsub


def split_step(c, k, dt, dx, area, u, D, c_in, method):
    db = dr = 0.0
    ns = 0

    if method == "lie_TR":
        c, q, m = transport_ssprk2(
            c, dt, dx, area, u, D, c_in
        )
        db += q
        ns += m
        c, q = reaction_exact(c, k, dt, dx, area)
        dr += q

    elif method == "lie_RT":
        c, q = reaction_exact(c, k, dt, dx, area)
        dr += q
        c, q, m = transport_ssprk2(
            c, dt, dx, area, u, D, c_in
        )
        db += q
        ns += m

    elif method == "strang":
        c, q = reaction_exact(c, k, 0.5 * dt, dx, area)
        dr += q
        c, q, m = transport_ssprk2(
            c, dt, dx, area, u, D, c_in
        )
        db += q
        ns += m
        c, q = reaction_exact(c, k, 0.5 * dt, dx, area)
        dr += q

    else:
        raise ValueError("未知分裂方法")

    return c, db, dr, ns


def simulate(cfg, method="strang"):
    nx = int(cfg["nx"])
    L = float(cfg["L"])
    u = float(cfg["u"])
    D = float(cfg["D"])
    c_in = float(cfg["c_in"])
    area = float(cfg["area"])
    dt = float(cfg["dt"])
    t_end = float(cfg["t_end"])

    if nx < 2 or dt <= 0 or t_end < 0:
        raise ValueError("要求nx>=2、dt>0、t_end>=0")

    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx
    c0 = 0.15 + 0.80 * np.exp(
        -((x - 0.30 * L) / (0.09 * L))**2
    )
    k = 0.002 + 0.010 * (x / L)**2
    c, k = validate(c0, k, L, u, D, c_in, area)

    m0 = area * dx * np.sum(c)
    db = dr = 0.0
    nsub = 0
    t = 0.0

    while t < t_end:
        h = min(dt, t_end - t)
        c, qb, qr, ns = split_step(
            c, k, h, dx, area, u, D, c_in, method
        )
        db += qb
        dr += qr
        nsub += ns
        t += h

    m1 = area * dx * np.sum(c)
    return {
        "x": x, "c": c, "k": k,
        "initial_mass": m0,
        "final_mass": m1,
        "boundary_mass": db,
        "reaction_mass": dr,
        "balance_residual": m1 - m0 - db - dr,
        "minimum_c": float(np.min(c)),
        "transport_substeps": nsub,
        "config_hash": cfg_hash(cfg),
    }


def weighted_l2(a, b, dx):
    return np.sqrt(dx * np.sum((a - b)**2))


if __name__ == "__main__":
    base = {
        "nx": 80, "L": 20.0, "u": 0.025,
        "D": 0.004, "c_in": 0.60, "area": 1.0,
        "dt": 4.0, "t_end": 200.0,
    }

    for method in ("lie_TR", "lie_RT", "strang"):
        r = simulate(base, method)
        print(method, r["minimum_c"], r["balance_residual"])

    # 同一空間網格上的時間細化；參考解也只是較細數值解
    ref_cfg = dict(base)
    ref_cfg["dt"] = 0.125
    ref = simulate(ref_cfg, "strang")

    errors = []
    for dt in (4.0, 2.0, 1.0, 0.5):
        cfg = dict(base)
        cfg["dt"] = dt
        r = simulate(cfg, "strang")
        err = weighted_l2(r["c"], ref["c"], base["L"] / base["nx"])
        errors.append(err)
        print("dt=", dt, "field_error=", err)

    for i in range(len(errors) - 1):
        if errors[i + 1] > 0:
            p = np.log(errors[i] / errors[i + 1]) / np.log(2.0)
            print("observed_order=", p)
```

程式未在此執行。即使列出`observed_order`，也只能把結果解釋為同一空間離散下的經驗時間階。參考步長仍有限，若誤差接近參考解誤差或浮點尺度，階數估計會失真。

---

## 測試與預期結果

以下均為依推導得到的預期結果，不是執行報告。

### 正常測試

預設資料下預期：

- 濃度保持有限且不出現顯著負值。
- `reaction_mass`非正。
- `balance_residual`接近浮點累積尺度。
- 兩種Lie次序一般不同，因$k(x)$與傳輸不交換。
- Strang場解在適當漸近區可能呈現接近二階的時間誤差縮減，但不可預先宣稱實測階數必為二。

只比較`final_mass`不足以驗證分裂精度；兩個錯誤空間分布可能有相同總質量，因此程式比較加權場範數。

### 邊界與常數場測試

令$k=0$、$D=0$，並令初值與$c_{\rm in}$皆為常數$c_0$。預期所有面通量相同，濃度保持常數，邊界淨質量與反應質量皆為零。

若初值為零、$c_{\rm in}>0$，前緣到達右端前，流入大於流出，故總質量增加。

### 純反應測試

直接呼叫`reaction_exact`，常數$k$下應滿足

$$
c_i(t)=c_i(0)e^{-kt}.
$$

總質量變化應完全等於回傳的反應質量。程式主模擬限定$u>0$；不可用極小正速度冒充純反應。

### 故障測試

- `u<0`：應拒絕，因未提供右側流入資料。
- `D=np.nan`或輸入含無限值：應拒絕。
- 負$D$、負$k$或負初值：依本章模型契約拒絕。
- 若繞過子步切分，使Euler階段違反$C+2r\le1$，可能產生負值或振盪；不得用`np.maximum(c, 0)`掩蓋。

### 五種性質分開判斷

1. **穩定性**：擾動是否受控制。
2. **守恆**：總量是否符合邊界及反應收支。
3. **非負性**：更新後濃度是否保持非負。
4. **能量下降**：有流入與反應時，一般沒有可直接宣稱單調下降的簡單平方能量。
5. **物理可信度**：係數與機制是否代表目標系統；數值測試通過不等於現場驗證。

---

## 除錯與常見陷阱

### 重複計入邊界來源

若已用$J_{1/2}=uc_{\rm in}$注入，就不能再把同一量加入第一格體積來源。面通量乘面積，體積來源乘體積，量綱與幾何權重不同。

### 錯置流入端

速度改號後，流入端也必須改變。只修改上風索引而不修改邊界資料，得到的是未完整定義的問題。

### 把Strang排列等同二階程式

Strang的二階結論要求子求解器足階。若傳輸用一階Euler，即使程式順序是「半反應—全傳輸—半反應」，整體一般仍由一階傳輸誤差主導。本章採SSPRK2正是為修正此問題。

### 只看質量判斷準確度

質量平衡可以精確，而濃度峰值位置仍錯誤。守恆測試與場誤差測試不能互相替代。

### 事後裁零

把負值改成零會增加質量。例如$c_i=-0.1$被改成零，質量增加$0.1A\Delta x$。正確作法是保留故障證據並修正步長、通量或模型。

---

## 養殖與相場案例

合成池域的溶氧模型可寫成

$$
c_t+\nabla\cdot(\mathbf{u}c)
=
\nabla\cdot(D\nabla c)
+k_a(c_{\rm sat}-c)-k_cc.
$$

$k_a$與$k_c$單位皆為$\mathrm{s^{-1}}$，$c$與$c_{\rm sat}$為$\mathrm{kg/m^3}$。換算關係為

$$
1\,\mathrm{mg/L}
=
\frac{10^{-6}\,\mathrm{kg}}{10^{-3}\,\mathrm{m^3}}
=
10^{-3}\,\mathrm{kg/m^3}.
$$

此處參數皆為合成值，不提供現場管理閾值，也不控制設備。模擬時須分別記錄邊界輸入、復氧生成、耗氧損失與總量殘差。

溶氧跨越管理閾值不是熱力學相變。相場模型中的序參量、雙井自由能與界面能具有不同意義；濃度圖呈現兩個區域，不能據此稱為Cahn–Hilliard相分離。

---

## 習題

### 習題一：手算

給定$L=5\,\mathrm{m}$、$u=0.01\,\mathrm{m/s}$、$D=2\times10^{-4}\,\mathrm{m^2/s}$、$k=0.002\,\mathrm{s^{-1}}$。求$Pe$、$Da$、$Da_D$。若$\Delta x=0.05\,\mathrm{m}$，求單調性允許的最大顯式步長。

### 習題二：程式

修改測試，使$k=0$且初值與流入值相同。列出應檢查的量。再說明為何只比較最終質量不能驗證時間階。

### 習題三：反例

反駁「只要使用Strang排列，程式一定是二階」這句話，至少提出兩種失敗情況。

### 習題四：整合

對反應$R=s-kc$：

1. 推導精確反應更新。
2. 寫出Strang步驟。
3. 說明如何記錄反應質量。
4. 判斷在$s\ge0$、$k>0$及$c(0)\ge0$時是否保持非負。

### 習題五：交換子

給定

$$
A=
\begin{bmatrix}
-1&0\\
1&0
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
0&0\\
0&-2
\end{bmatrix},
$$

計算$[B,A]=BA-AB$並解釋其傳輸—反應意義。

---

## 習題解答

### 解答一

$$
Pe=\frac{0.01\times5}{2\times10^{-4}}=250,
$$

$$
Da=\frac{0.002\times5}{0.01}=1,
$$

$$
Da_D=Pe\,Da=250.
$$

步長條件為

$$
\left(
\frac{|u|}{\Delta x}
+\frac{2D}{\Delta x^2}
\right)\Delta t\le1.
$$

係數為

$$
\frac{0.01}{0.05}
+\frac{2(2\times10^{-4})}{0.05^2}
=0.2+0.16=0.36\,\mathrm{s^{-1}}.
$$

故

$$
\Delta t_{\max}=\frac1{0.36}\approx2.78\,\mathrm{s}.
$$

### 解答二

應檢查濃度是否保持常數、左右通量是否相等、反應質量是否為零、邊界淨質量是否為零、總質量是否不變，以及平衡殘差是否接近浮點尺度。

只比較最終質量不能驗證時間階，因為不同濃度場可能具有相同積分。時間階應使用固定空間網格下的場誤差，例如

$$
E_h=\left[\Delta x\sum_i(c_i-c_i^{\rm ref})^2\right]^{1/2}.
$$

### 解答三

第一，若傳輸子步只用一階Euler，其誤差會主導，Strang排列不會自動提升成二階。第二，非光滑資料、剛性反應或不相容的時間邊界可能造成降階。第三，若參考解不夠細或空間誤差主導，觀測階也不會接近二。

### 解答四

解

$$
\frac{dc}{dt}=s-kc
$$

可得

$$
c(t+\tau)
=
\frac{s}{k}
+
\left(c(t)-\frac{s}{k}\right)e^{-k\tau}.
$$

Strang步驟是反應$\Delta t/2$、傳輸$\Delta t$、反應$\Delta t/2$。每個反應子步以

$$
\Delta M_R
=
A\Delta x\sum_i
(c_i^{\rm after}-c_i^{\rm before})
$$

記錄質量改變。因更新亦可寫成

$$
c(t+\tau)
=
e^{-k\tau}c(t)
+\frac{s}{k}(1-e^{-k\tau}),
$$

兩係數皆非負，所以$s\ge0$且$c(t)\ge0$時保持非負。

### 解答五

$$
AB=
\begin{bmatrix}
0&0\\
0&0
\end{bmatrix},
\qquad
BA=
\begin{bmatrix}
0&0\\
-2&0
\end{bmatrix}.
$$

因此

$$
[B,A]
=
\begin{bmatrix}
0&0\\
-2&0
\end{bmatrix}.
$$

$A$把第一格物質送入第二格，$B$只在第二格衰減。先傳輸再反應會立刻消耗新送入的物質；先反應再傳輸則不會，故兩者不交換。

---

## 本章小結

Péclet數與Damköhler數描述平流、擴散及反應的尺度競爭，但必須列明參考尺度。有限體積法藉共享面通量維持守恆，流入端則由速度方向決定。

Lie分裂一般為一階；Strang分裂在算子、邊界及子求解器都滿足條件時一般為二階。Strang排列本身不是二階保證。本章程式使用二階SSPRK2傳輸與精確反應子步，並以場誤差進行時間細化；所有數值結果仍須實際執行後才能報告。

穩定性、守恆、非負性、能量下降及物理可信度必須分開檢查。不得以裁零、平滑圖像或單一總量掩蓋離散錯誤。

---

## 參考來源

1. **F1：FiPy有限體積離散與邊界**  
   https://pages.nist.gov/fipy/en/latest/numerical/discret.html

2. **F3：PETSc線性系統求解器**  
   https://petsc.org/release/manual/ksp/

3. **F4：SciPy稀疏線性代數API**  
   https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

4. **F6：FiPy Cahn–Hilliard相分離示範**  
   https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html

F6僅供守恆相場語意比較，其序參量與本章溶質濃度不是同一物理量。上述來源為延伸閱讀；本章未執行來源程式，也未以合成案例宣稱現場物理驗證。