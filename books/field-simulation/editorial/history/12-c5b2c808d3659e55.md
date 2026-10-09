# 第12章 平流擴散反應與分裂誤差

## 學習目標與先備知識

本章研究純量濃度場的平流—擴散—反應方程，核心是尺度競爭、有限體積收支、反應剛性及算子分裂誤差。完成本章後，讀者應能：

1. 寫出守恆形式方程並核對SI單位。
2. 推導Péclet數與Damköhler數，明示參考尺度。
3. 區分平流流入／流出與擴散通量邊界。
4. 以共享面通量維持離散守恆。
5. 說明Lie與Strang分裂及交換子誤差。
6. 理解子問題穩定不代表分裂誤差很小。
7. 分開檢查穩定性、守恆、非負性、能量下降及物理可信度。
8. 用NumPy CPU程式記錄邊界收支、反應收支與設定雜湊。
9. 用場誤差及參考解細化診斷時間收斂，不只比較總質量。

先備知識包括初邊值問題、有限體積法、上風通量、中心擴散、顯式時間積分與Taylor展開。本章採一維cell-centered網格，第$i$格中心為$x_i=(i+1/2)\Delta x$。二維推廣時使用`q[j,i]`，$i$沿$+X$、$j$沿$+Y$。

---

## 問題與直覺

令$c(x,t)$為溶質濃度，單位為$\mathrm{kg/m^3}$。一維守恆方程為

$$
\frac{\partial c}{\partial t}
+\frac{\partial(uc)}{\partial x}
=
\frac{\partial}{\partial x}
\left(D\frac{\partial c}{\partial x}\right)
+R(c,x,t).
$$

其中$u$的單位為$\mathrm{m/s}$，$D$為$\mathrm{m^2/s}$，而$R$為$\mathrm{kg/(m^3\,s)}$。若$u$為常數，方程亦可寫成

$$
c_t+uc_x=Dc_{xx}+R.
$$

變速度時，$\partial_x(uc)$與$uc_x$一般不相等，因此有限體積離散應以守恆形式為起點。

三種機制扮演不同角色：

- **平流**隨流體搬運濃度，資訊方向由速度決定。
- **擴散**削弱空間梯度，使尖峰展寬。
- **反應**在局部生成或消耗物質。

若$R=-kc$，則$k$的單位是$\mathrm{s^{-1}}$。若另有體積來源$s$，

$$
R=s-kc,
$$

則$s$的單位是$\mathrm{kg/(m^3\,s)}$。邊界注入是面通量，不能未經幾何換算就當作體積來源。

當三種機制的時間尺度相差很大時，可把傳輸與反應分成子問題依次求解。然而「先傳輸再反應」通常不等於「先反應再傳輸」；這個次序差異便是分裂誤差的直覺來源。

---

## 數學與物理推導

### 控制體收支

令一維管道截面面積為$A$。第$i$格體積是$A\Delta x$，其質量為

$$
m_i=c_iA\Delta x.
$$

定義沿$+X$為正的總通量

$$
J=uc-Dc_x,
$$

單位為$\mathrm{kg/(m^2\,s)}$。對控制體積分得

$$
\frac{d(c_iA\Delta x)}{dt}
=
AJ_{i-1/2}-AJ_{i+1/2}+A\Delta xR_i,
$$

所以

$$
\frac{dc_i}{dt}
=
-\frac{J_{i+1/2}-J_{i-1/2}}{\Delta x}+R_i.
$$

把所有控制體相加，內部共享面通量兩兩抵消：

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

這是本章質量診斷的基礎。ghost cell不是物理控制體，不得算入$M$。

### 無因次化與尺度參數

取參考尺度

$$
x=Lx^\ast,\qquad
t=\frac{L}{U}t^\ast,\qquad
u=Uu^\ast,\qquad
c=Cc^\ast.
$$

對一階反應$R=-kc$，代入後可得

$$
\frac{\partial c^\ast}{\partial t^\ast}
+\frac{\partial(u^\ast c^\ast)}{\partial x^\ast}
=
\frac1{Pe}
\frac{\partial^2c^\ast}{\partial x^{\ast2}}
-Da\,c^\ast,
$$

其中

$$
Pe=\frac{UL}{D},\qquad
Da=\frac{kL}{U}.
$$

$Pe$比較平流與擴散；$Da$比較反應與平流。若以擴散時間$L^2/D$為參考，則

$$
Da_D=\frac{kL^2}{D}=Pe\,Da.
$$

因此報告Damköhler數時，必須說明它以哪個時間尺度定義。網格Péclet數

$$
Pe_h=\frac{|u|\Delta x}{D}
$$

則描述單格尺度上的平流—擴散競爭，不能與域尺度$Pe$混為一談。

### 有限體積通量與邊界

內部面通量採一階上風平流與中心擴散：

$$
J_{i+1/2}
=
u c_{\rm up}
-D\frac{c_{i+1}-c_i}{\Delta x}.
$$

若$u>0$，則$c_{\rm up}=c_i$；若$u<0$，則$c_{\rm up}=c_{i+1}$。

本章實作限定$u>0$。左側提供流入濃度$c_{\rm in}$，右側為平流流出，兩端擴散通量皆為零：

$$
J_{1/2}=uc_{\rm in},\qquad
J_{N+1/2}=uc_{N-1}.
$$

零擴散通量不表示總通量為零。若速度為負，流入端必須移到右側；只把速度改號而仍使用左流入資料，問題本身便未正確定義。

### 顯式條件與非負性

令

$$
C=\frac{u\Delta t}{\Delta x},\qquad
r=\frac{D\Delta t}{\Delta x^2},
$$

其中本節假設$u>0$。內部格的顯式Euler更新為

$$
c_i^{n+1}
=
(C+r)c_{i-1}^n
+(1-C-2r)c_i^n
+rc_{i+1}^n.
$$

因此

$$
C+2r\le1
$$

是內部更新保持非負係數的充分條件。

含非負左流入資料時，第一格更新為

$$
c_0^{n+1}
=
(1-C-r)c_0^n
+rc_1^n
+Cc_{\rm in}.
$$

由$C+2r\le1$可得$1-C-r\ge r\ge0$，所以第一格也保持非負。右端零擴散通量、平流流出時，

$$
c_{N-1}^{n+1}
=
(C+r)c_{N-2}^n
+(1-C-r)c_{N-1}^n,
$$

其係數也非負。故在本章明定的常係數、$u>0$、非負流入濃度及零擴散通量條件下，顯式Euler映射即使含仿射邊界項，仍保持非負。

後文使用二階SSPRK2：

$$
c^{(1)}=c^n+\Delta tF(c^n),
$$

$$
c^{n+1}
=
\frac12c^n
+\frac12\left[c^{(1)}+\Delta tF(c^{(1)})\right].
$$

若每個Euler映射都使用同一非負邊界資料並滿足上述條件，兩個Euler結果與最後凸組合皆非負。因此本章特定邊界設定下，SSPRK2可繼承此非負性。這不是對任意時間相依邊界、負來源、變係數離散或任意二階法的普遍保證；程式中的負值檢查仍保留作故障偵測。

### Lie、Strang與交換子

把半離散方程寫成

$$
\frac{dc}{dt}=(A+B)c,
$$

其中$A$代表傳輸，$B$代表反應。Lie分裂之一為

$$
c^{n+1}=e^{\Delta tB}e^{\Delta tA}c^n.
$$

展開後，其局部主誤差含

$$
\frac{\Delta t^2}{2}(BA-AB)
=
\frac{\Delta t^2}{2}[B,A].
$$

所以Lie分裂一般具有$O(\Delta t^2)$局部誤差及$O(\Delta t)$全域誤差。

Strang分裂為

$$
c^{n+1}
=
e^{\Delta tB/2}
e^{\Delta tA}
e^{\Delta tB/2}c^n.
$$

若解與算子足夠光滑、子問題邊界相容，而且子求解器至少具有所需的二階精度，則其全域時間誤差一般為$O(\Delta t^2)$。Strang排列本身不是二階程式的充分條件；若傳輸子問題只用一階Euler，完整實作通常仍由一階誤差主導。因此本章程式使用二階SSPRK2傳輸。

若$AB=BA$，兩算子可交換。例如週期域上的常係數傳輸與均勻衰減$B=-kI$交換。空間變化的$k(x)$、非線性反應、濃度相依傳輸係數及不相容邊界通常使交換子不為零。

### 剛性反應

對

$$
c_t=-kc,
$$

顯式Euler給出

$$
c^{n+1}=(1-k\Delta t)c^n.
$$

線性穩定要求$k\Delta t\le2$，非負性則要求更嚴格的$k\Delta t\le1$。精確子步為

$$
c^{n+1}=e^{-k\Delta t}c^n,
$$

對任意$\Delta t\ge0$保持非負並衰減。然而精確反應子步只消除反應子問題的時間積分誤差，不能消除傳輸與反應之間的分裂誤差。

---

## 逐步手算例題

### 例題一：Péclet數、Damköhler數與步長

給定

$$
L=10\,\mathrm{m},\quad
u=0.020\,\mathrm{m/s},\quad
D=10^{-3}\,\mathrm{m^2/s},\quad
k=5\times10^{-3}\,\mathrm{s^{-1}}.
$$

三個時間尺度為

$$
\tau_a=\frac{L}{u}=500\,\mathrm{s},
$$

$$
\tau_d=\frac{L^2}{D}=10^5\,\mathrm{s},
$$

$$
\tau_r=\frac1k=200\,\mathrm{s}.
$$

所以

$$
Pe=\frac{uL}{D}=200,\qquad
Da=\frac{kL}{u}=2.5.
$$

域尺度上平流強於擴散，且一次平流穿越期間反應不可忽略。

若$\Delta x=0.10\,\mathrm{m}$，則

$$
C=0.2\Delta t,\qquad 2r=0.2\Delta t.
$$

由$C+2r\le1$得

$$
\Delta t\le2.5\,\mathrm{s}.
$$

若反應也用顯式Euler，非負性要求$\Delta t\le1/k=200\,\mathrm{s}$，故此例的限制來自傳輸。若把$k$提高為$5\,\mathrm{s^{-1}}$，反應限制變為$0.2\,\mathrm{s}$，此時反應成為剛性來源。

### 例題二：兩格收支與分裂次序

設$A=\Delta x=1$、$u=1$、$D=0$、$c_{\rm in}=2$，

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

三個面通量為

$$
J_{1/2}=2,\qquad
J_{3/2}=1,\qquad
J_{5/2}=0.
$$

一次Euler傳輸步得到

$$
c^\star=
\begin{bmatrix}
1.25\\
0.25
\end{bmatrix}.
$$

總量增加$0.5$，恰等於邊界淨輸入

$$
\Delta t(J_{1/2}-J_{5/2})=0.5.
$$

若先傳輸再精確反應，

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

若先反應再傳輸，由於初始第二格為零，反應步不改變狀態，結果為

$$
c^{n+1}=
\begin{bmatrix}
1.25\\
0.25
\end{bmatrix}.
$$

兩種結果都可滿足各自的離散收支，卻具有不同分布。這是分裂誤差，不是守恆錯誤。

Strang排列為半步反應、完整傳輸、半步反應。本例按相同單步傳輸計算得

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

它不能僅因位於兩個Lie結果之間就被稱為真解。

---

## 實作與程式

以下自足程式只使用Python 3.10+與NumPy。傳輸採SSPRK2，反應採精確指數更新。邊界質量使用兩個Runge–Kutta階段的相同權重，因而與狀態更新的離散收支一致。程式不使用`clip`。

```python
import json
import hashlib
import numpy as np


def cfg_hash(cfg):
    text = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def validate(c, k, L, u, D, c_in, area):
    c = np.asarray(c, dtype=float)
    k = np.asarray(k, dtype=float)
    scalars = np.array([L, u, D, c_in, area], dtype=float)

    if c.ndim != 1 or c.size < 2 or k.shape != c.shape:
        raise ValueError("c與k必須是相同形狀的一維陣列")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(k)):
        raise ValueError("c與k必須有限")
    if not np.all(np.isfinite(scalars)):
        raise ValueError("所有設定必須有限")
    if L <= 0.0 or area <= 0.0 or D < 0.0:
        raise ValueError("要求L>0、area>0、D>=0")
    if np.any(k < 0.0):
        raise ValueError("本模型要求k>=0")
    if u <= 0.0:
        raise ValueError("本實作限定u>0；負速度需提供右側流入")
    if c_in < 0.0 or np.any(c < 0.0):
        raise ValueError("初始與流入濃度必須非負")
    return c.copy(), k.copy()


def reaction_exact(c, k, dt, dx, area):
    before = area * dx * np.sum(c)
    out = c * np.exp(-k * dt)
    after = area * dx * np.sum(out)
    return out, after - before


def face_flux(c, u, D, dx, c_in):
    J = np.empty(c.size + 1, dtype=float)
    J[0] = u * c_in
    J[-1] = u * c[-1]
    J[1:-1] = u * c[:-1] - D * (c[1:] - c[:-1]) / dx
    return J


def transport_rhs(c, u, D, dx, c_in):
    J = face_flux(c, u, D, dx, c_in)
    rhs = -(J[1:] - J[:-1]) / dx
    boundary_rate = J[0] - J[-1]
    return rhs, boundary_rate


def transport_ssprk2(c, duration, dx, area, u, D, c_in,
                     safety=0.9):
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration必須為有限非負數")
    if not (0.0 < safety <= 1.0):
        raise ValueError("safety必須位於(0,1]")
    if duration == 0.0:
        return c.copy(), 0.0, 0

    rate = u / dx + 2.0 * D / dx**2
    nsub = max(1, int(np.ceil(duration * rate / safety)))
    h = duration / nsub

    out = c.copy()
    boundary_mass = 0.0

    for _ in range(nsub):
        f0, b0 = transport_rhs(out, u, D, dx, c_in)
        stage = out + h * f0

        if np.min(stage) < -1.0e-12:
            raise FloatingPointError(
                "SSPRK2第一Euler階段出現負值"
            )

        f1, b1 = transport_rhs(stage, u, D, dx, c_in)
        new = 0.5 * out + 0.5 * (stage + h * f1)

        boundary_mass += area * h * 0.5 * (b0 + b1)

        if not np.all(np.isfinite(new)):
            raise FloatingPointError("傳輸步驟產生非有限值")
        if np.min(new) < -1.0e-12:
            raise FloatingPointError(
                "產生顯著負濃度；不可用裁零掩蓋"
            )
        out = new

    return out, boundary_mass, nsub


def split_step(c, k, dt, dx, area, u, D, c_in, method):
    boundary_mass = 0.0
    reaction_mass = 0.0
    nsub = 0

    if method == "lie_TR":
        c, q, m = transport_ssprk2(
            c, dt, dx, area, u, D, c_in
        )
        boundary_mass += q
        nsub += m
        c, q = reaction_exact(c, k, dt, dx, area)
        reaction_mass += q

    elif method == "lie_RT":
        c, q = reaction_exact(c, k, dt, dx, area)
        reaction_mass += q
        c, q, m = transport_ssprk2(
            c, dt, dx, area, u, D, c_in
        )
        boundary_mass += q
        nsub += m

    elif method == "strang":
        c, q = reaction_exact(c, k, 0.5 * dt, dx, area)
        reaction_mass += q
        c, q, m = transport_ssprk2(
            c, dt, dx, area, u, D, c_in
        )
        boundary_mass += q
        nsub += m
        c, q = reaction_exact(c, k, 0.5 * dt, dx, area)
        reaction_mass += q

    else:
        raise ValueError("未知分裂方法")

    return c, boundary_mass, reaction_mass, nsub


def simulate(cfg, method="strang"):
    nx_raw = cfg["nx"]
    nx = int(nx_raw)
    if nx != nx_raw:
        raise ValueError("nx必須是整數")

    L = float(cfg["L"])
    u = float(cfg["u"])
    D = float(cfg["D"])
    c_in = float(cfg["c_in"])
    area = float(cfg["area"])
    dt = float(cfg["dt"])
    t_end = float(cfg["t_end"])

    values = np.array([L, u, D, c_in, area, dt, t_end])
    if not np.all(np.isfinite(values)):
        raise ValueError("設定含非有限值")
    if nx < 2 or dt <= 0.0 or t_end < 0.0:
        raise ValueError("要求nx>=2、dt>0、t_end>=0")

    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx

    # 全為合成資料，不代表現場物性
    c0 = 0.15 + 0.80 * np.exp(
        -((x - 0.30 * L) / (0.09 * L))**2
    )
    k = 0.002 + 0.010 * (x / L)**2
    c, k = validate(c0, k, L, u, D, c_in, area)

    initial_mass = area * dx * np.sum(c)
    boundary_mass = 0.0
    reaction_mass = 0.0
    total_substeps = 0
    t = 0.0

    while t < t_end:
        h = min(dt, t_end - t)
        c, db, dr, ns = split_step(
            c, k, h, dx, area, u, D, c_in, method
        )
        boundary_mass += db
        reaction_mass += dr
        total_substeps += ns
        t += h

    final_mass = area * dx * np.sum(c)
    balance = (
        final_mass - initial_mass
        - boundary_mass - reaction_mass
    )

    return {
        "x": x,
        "c": c,
        "k": k,
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "boundary_mass": boundary_mass,
        "reaction_mass": reaction_mass,
        "balance_residual": balance,
        "minimum_c": float(np.min(c)),
        "transport_substeps": total_substeps,
        "config_hash": cfg_hash(cfg),
    }


def weighted_l2(a, b, dx):
    return np.sqrt(dx * np.sum((a - b)**2))


if __name__ == "__main__":
    base = {
        "nx": 80,
        "L": 20.0,       # m
        "u": 0.025,      # m/s
        "D": 0.004,      # m^2/s
        "c_in": 0.60,    # kg/m^3
        "area": 1.0,     # m^2
        "dt": 4.0,       # s
        "t_end": 200.0,  # s
    }

    for method in ("lie_TR", "lie_RT", "strang"):
        result = simulate(base, method)
        print(
            method,
            "min=", result["minimum_c"],
            "balance=", result["balance_residual"],
        )

    # 待執行診斷：先檢查參考解是否對進一步細化不敏感
    ref1_cfg = dict(base)
    ref1_cfg["dt"] = 0.125
    ref2_cfg = dict(base)
    ref2_cfg["dt"] = 0.0625

    ref1 = simulate(ref1_cfg, "strang")
    ref2 = simulate(ref2_cfg, "strang")
    dx = base["L"] / base["nx"]

    ref_gap = weighted_l2(ref1["c"], ref2["c"], dx)
    print("reference_gap=", ref_gap)

    errors = []
    for dt in (4.0, 2.0, 1.0, 0.5):
        cfg = dict(base)
        cfg["dt"] = dt
        result = simulate(cfg, "strang")
        err = weighted_l2(result["c"], ref2["c"], dx)
        errors.append(err)
        print(
            "dt=", dt,
            "field_error=", err,
            "substeps=", result["transport_substeps"],
        )

    # 只有誤差明顯大於reference_gap且連續比值穩定時才解讀
    for i in range(len(errors) - 1):
        if errors[i + 1] > 0.0:
            p = np.log(errors[i] / errors[i + 1]) / np.log(2.0)
            print("diagnostic_order=", p)
```

這段程式只是待執行診斷，不是二階收斂的證明，也未在本章中實際執行。報告觀測階之前至少應確認：

1. `reference_gap`明顯小於被測誤差。
2. 連續數個步長的誤差比值趨於穩定。
3. 外層步長與內部傳輸子步數的改變已被記錄。
4. 結果未進入浮點誤差主導區。
5. 時間相依邊界若存在，已在正確Runge–Kutta階段取值。

固定空間網格可用來估計同一空間離散系統的時間誤差；但它不是連續PDE整體空間—時間收斂的證明。即使某次執行得到接近二的數字，也只能稱為指定測試區間內的觀測階。

---

## 測試與預期結果

以下皆為依推導得到的預期結果，不是已執行報告。

### 正常測試

對預設合成資料，預期：

- 所有輸出保持有限。
- `reaction_mass`非正。
- `balance_residual`接近浮點累積尺度。
- 兩種Lie次序一般不同，因$k(x)$與傳輸不交換。
- 本章邊界及步長條件下，兩個Euler階段與SSPRK2結果應保持非負。
- 時間細化誤差應減少才支持收斂；不能預先宣稱其觀測階必接近二。

總質量相同不代表濃度場相同，因此時間診斷必須比較場範數。

### 邊界測試：常數場

令$k=0$、$D=0$，並令$c_i=c_{\rm in}=c_0$。所有面通量均為$uc_0$，故濃度應保持常數，邊界淨質量與反應質量皆為零。

### 非零流入的非負測試

令初值全零、$D=0$、$c_{\rm in}>0$。第一個Euler階段的第一格為

$$
c_0^{(1)}=Cc_{\rm in}\ge0,
$$

其餘格仍為零。第二個Euler映射仍由非負係數及非負流入組成，因此SSPRK2結果應非負。這個測試直接涵蓋非齊次仿射流入，而不是只測週期或零邊界。

### 純反應測試

直接呼叫`reaction_exact`。對常數$k$，應得到

$$
c_i(t)=c_i(0)e^{-kt}.
$$

總質量變化應完全等於回傳的反應質量。主模擬限定$u>0$，不可用極小正速度冒充純反應。

### 故障測試

- `u<0`：應拒絕，因程式沒有右側流入參數。
- `D=np.nan`、`c_in=np.inf`：應拒絕。
- 負$D$、負$k$或負初值：依本模型契約拒絕。
- 若繞過子步切分，使Euler階段違反$C+2r\le1$，非負性不再受上述證明保障。
- 若出現負值，不得使用`np.maximum(c,0)`後假裝測試通過。

### 五種性質分開

1. **數值穩定性**：擾動是否受控制。
2. **守恆**：總量是否符合邊界及反應收支。
3. **非負性**：濃度是否不小於零；本章只在明列條件下給出充分條件。
4. **能量下降**：有流入與反應時，一般不能宣稱簡單的平方能量單調下降。
5. **物理可信度**：係數、來源與邊界是否代表目標系統；數值正確不等於模型已驗證。

---

## 除錯與常見陷阱

### 重複計入邊界來源

若已用$J_{1/2}=uc_{\rm in}$注入，就不能再把相同量加入第一格體積來源。面通量乘面積，體積來源乘體積，兩者量綱與幾何權重不同。

### 錯置流入端

速度反向後，流入端也必須反向。只修改上風索引而不修改邊界資料，是模型定義錯誤。

### 把Strang排列等同二階程式

若傳輸子問題採一階Euler，Strang外觀不會自動產生二階精度。即使採SSPRK2，仍需參考解細化、誤差區間及邊界相容性診斷。

### 忽略仿射流入邊界

不能只引用「SSPRK是凸組合」便結束論證。必須先證明含流入項的Euler映射保持非負。本章已逐格列出左端、內部與右端係數；其他邊界條件必須重新分析。

### 只看質量

守恆殘差很小只能表示離散記帳一致。錯誤的峰值位置、過度數值擴散及錯誤參數仍可能有完美質量平衡。

### 事後裁零

若$c_i=-0.1$被改成零，質量增加$0.1A\Delta x$。裁零會同時破壞收支與收斂診斷，不能用來掩蓋不穩定。

---

## 養殖與相場案例

合成池域中的溶氧模型可寫成

$$
c_t+\nabla\cdot(\mathbf{u}c)
=
\nabla\cdot(D\nabla c)
+k_a(c_{\rm sat}-c)-k_cc.
$$

$k_a$與$k_c$的單位皆為$\mathrm{s^{-1}}$，$c$及$c_{\rm sat}$為$\mathrm{kg/m^3}$。常見單位換算為

$$
1\,\mathrm{mg/L}
=
\frac{10^{-6}\,\mathrm{kg}}{10^{-3}\,\mathrm{m^3}}
=
10^{-3}\,\mathrm{kg/m^3}.
$$

這些係數在本章均為合成值，不代表現場物性，也不提供管理閾值或設備控制。模擬必須分別記錄邊界傳輸、復氧生成、耗氧損失及總量殘差。

溶氧跨越管理閾值不是熱力學相變。相場模型中的序參量、雙井自由能及界面能具有不同物理意義；濃度分布形成兩個區域，不能據此稱為Cahn–Hilliard相分離。

---

## 習題

### 習題一：手算尺度

給定$L=5\,\mathrm{m}$、$u=0.01\,\mathrm{m/s}$、$D=2\times10^{-4}\,\mathrm{m^2/s}$及$k=0.002\,\mathrm{s^{-1}}$。求$Pe$、$Da$及$Da_D$。若$\Delta x=0.05\,\mathrm{m}$，求單調性允許的最大顯式步長。

### 習題二：程式診斷

令$k=0$且初值與流入值相同。列出應檢查的量。再說明為何只比較最終質量不能驗證時間階，以及如何檢查參考解是否足夠細。

### 習題三：反例

反駁「只要使用Strang排列，程式一定是二階」這句話，至少提出兩種失敗情況。

### 習題四：整合反應

對$R=s-kc$：

1. 推導精確反應更新。
2. 寫出Strang步驟。
3. 說明如何記錄反應質量。
4. 證明$s\ge0$、$k>0$與$c(0)\ge0$時反應子步保持非負。

### 習題五：含流入的非負性

對$u>0$、$D\ge0$、左側非負流入及兩端零擴散通量，寫出第一格與最後一格的Euler更新，並由$C+2r\le1$證明其係數非負。

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
\frac{u}{\Delta x}
+\frac{2D}{\Delta x^2}
\right)\Delta t\le1.
$$

係數為

$$
\frac{0.01}{0.05}
+\frac{2(2\times10^{-4})}{0.05^2}
=0.36\,\mathrm{s^{-1}},
$$

所以

$$
\Delta t_{\max}\approx2.78\,\mathrm{s}.
$$

### 解答二

應檢查濃度是否保持常數、左右通量是否相等、反應質量是否為零、邊界淨質量是否為零、總質量是否不變，以及平衡殘差是否接近浮點尺度。

總質量相同的兩個場可以有不同峰值位置及形狀，所以必須比較

$$
E=\left[\Delta x\sum_i(c_i-c_i^{\rm ref})^2\right]^{1/2}.
$$

再用兩個逐次細化的參考步長計算`reference_gap`。只有當它顯著小於被測誤差，且連續誤差比值進入穩定區間時，觀測階才有解釋價值。

### 解答三

第一，傳輸若只用一階Euler，其時間誤差可主導整體。第二，非光滑資料、剛性反應或不相容的時間邊界可能造成降階。第三，參考解不夠細、子步切分改變或浮點誤差主導時，計算出的觀測階也不代表理論階。

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
+\left(c(t)-\frac{s}{k}\right)e^{-k\tau}.
$$

等價地，

$$
c(t+\tau)
=
e^{-k\tau}c(t)
+\frac{s}{k}(1-e^{-k\tau}).
$$

Strang步驟依次為半步反應、完整傳輸、半步反應。每個反應子步的質量改變記為

$$
\Delta M_R
=
A\Delta x\sum_i
(c_i^{\rm after}-c_i^{\rm before}).
$$

因$e^{-k\tau}\ge0$且$1-e^{-k\tau}\ge0$，若$c(t)\ge0$與$s\ge0$，更新後仍非負。

### 解答五

第一格更新為

$$
c_0^{n+1}
=
(1-C-r)c_0^n+rc_1^n+Cc_{\rm in}.
$$

最後一格更新為

$$
c_{N-1}^{n+1}
=
(C+r)c_{N-2}^n+(1-C-r)c_{N-1}^n.
$$

由$C+2r\le1$可得

$$
1-C-r\ge r\ge0.
$$

再加上$C\ge0$、$r\ge0$及$c_{\rm in}\ge0$，故兩個邊界更新都保持非負。內部格亦由非負係數組成。因此每個Euler映射保持非負，而使用相同步長限制的SSPRK2凸組合亦保持非負。

---

## 本章小結

Péclet數與Damköhler數描述平流、擴散及反應的尺度競爭，但必須列明參考尺度。有限體積法藉共享面通量維持守恆，流入端則由速度方向決定。

Lie分裂一般為一階；Strang分裂只有在算子、邊界與子求解器滿足條件時才一般具有二階全域時間精度。本章採SSPRK2傳輸與精確反應子步，但觀測階仍須經參考解細化及漸近區間檢查，不能由程式排列預先宣稱。

本章特定非負流入邊界下，顯式Euler的仿射更新可逐格證明保持非負，SSPRK2因而繼承此性質；其他邊界及來源必須重新分析。最後，穩定性、守恆、非負性、能量下降與物理可信度始終是不同問題，不得用裁零、平滑圖或單一總量互相替代。

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

F6僅供守恆相場語意比較，其序參量與本章溶質濃度並非同一物理量。上述來源為延伸閱讀；本章未執行來源程式，也未以合成案例宣稱完成現場物理驗證。