# 第27章 Allen–Cahn與非守恆相場

## 學習目標與先備知識

本章研究 Allen–Cahn 方程，也就是自由能在 $L^2$ 度量下的梯度流。完成本章後，讀者應能：

1. 由自由能泛函推導化學勢與 Allen–Cahn 方程。
2. 說明序參量總量為何一般不守恆。
3. 分辨連續能量耗散、空間半離散耗散與時間離散能量下降。
4. 理解界面厚度、平面界面輪廓與曲率驅動的關係及限制。
5. 在二維週期單元中心網格上實作顯式 Euler 格式。
6. 分開檢查數值穩定、能量下降、總量守恆、值域限制與物理可信度。
7. 拒絕非有限輸入、非法步長與不一致陣列，而不以裁切掩蓋故障。

先備知識包括 Laplacian、週期有限差分、變分導數、梯度流與顯式時間積分。本章使用無因次模型；若改成有因次模型，必須重新列出自由能密度、梯度係數與遷移率的單位。

---

## 問題與直覺

相場方法以連續序參量 $\phi$ 描述兩種狀態。依本卷慣例，兩個均勻穩定狀態位於 $\phi=-1$ 與 $\phi=1$，但這不表示所有離散值都必須被硬性限制在 $[-1,1]$。界面是 $\phi$ 在兩個狀態間平滑過渡的區域。

本章採用無因次自由能

$$
F[\phi]
=
\int_\Omega
\left[
W(\phi)+\frac{\kappa}{2}|\nabla\phi|^2
\right]\,d\boldsymbol{x},
$$

其中

$$
W(\phi)=\frac{(\phi^2-1)^2}{4},
\qquad \kappa>0.
$$

雙井勢 $W$ 偏好 $\phi=\pm1$，梯度項則懲罰過於尖銳的空間變化。兩者競爭後形成有限厚度界面。

Allen–Cahn 動力學為

$$
\phi_t=-M\mu,
\qquad M>0,
$$

其中 $\mu$ 是自由能的變分導數。此方程允許序參量在每個位置直接鬆弛，因此一般不守恆

$$
\int_\Omega\phi\,d\boldsymbol{x}.
$$

這是它與 Cahn–Hilliard 方程的核心差別。若物理量必須嚴格守恆，例如封閉系統中的某種總濃度，不能只因 Allen–Cahn 較容易計算就忽略模型不相容性。

---

## 數學與物理推導

### 1. 第一變分與化學勢

令擾動為 $\phi+\epsilon\eta$。自由能的一階變化為

$$
\left.\frac{d}{d\epsilon}F[\phi+\epsilon\eta]\right|_{\epsilon=0}
=
\int_\Omega
\left[
W'(\phi)\eta+\kappa\nabla\phi\cdot\nabla\eta
\right]d\boldsymbol{x}.
$$

由

$$
W'(\phi)=\phi^3-\phi
$$

並對梯度項分部積分：

$$
\int_\Omega\kappa\nabla\phi\cdot\nabla\eta\,d\boldsymbol{x}
=
-\int_\Omega\kappa\Delta\phi\,\eta\,d\boldsymbol{x}
+
\int_{\partial\Omega}\kappa\partial_n\phi\,\eta\,dS.
$$

在週期邊界，或自然邊界 $\partial_n\phi=0$ 下，邊界項消失，因此

$$
\mu=\frac{\delta F}{\delta\phi}
=
\phi^3-\phi-\kappa\Delta\phi.
$$

Allen–Cahn 方程遂為

$$
\phi_t
=
-M(\phi^3-\phi-\kappa\Delta\phi)
=
M(\kappa\Delta\phi+\phi-\phi^3).
$$

### 2. 連續能量耗散

沿解軌跡，

$$
\frac{dF}{dt}
=
\int_\Omega
\frac{\delta F}{\delta\phi}\phi_t\,d\boldsymbol{x}
=
\int_\Omega\mu(-M\mu)\,d\boldsymbol{x}.
$$

若 $M$ 為正常數，

$$
\frac{dF}{dt}
=
-M\int_\Omega\mu^2\,d\boldsymbol{x}
\leq0.
$$

若 $M=M(\boldsymbol{x},\phi)>0$，則相應地有

$$
\frac{dF}{dt}
=
-\int_\Omega M\mu^2\,d\boldsymbol{x}
\leq0.
$$

這是連續方程的耗散律。它不等於「任何空間離散、任何時間步長、任何非線性求解容差下，計算出的離散能量都必然下降」。

### 3. 為何總量通常不守恆

定義序參量總量

$$
Q(t)=\int_\Omega\phi\,d\boldsymbol{x}.
$$

則

$$
\frac{dQ}{dt}
=
-M\int_\Omega\mu\,d\boldsymbol{x}.
$$

在週期或零法向梯度邊界下，

$$
\int_\Omega\Delta\phi\,d\boldsymbol{x}=0,
$$

所以

$$
\frac{dQ}{dt}
=
-M\int_\Omega(\phi^3-\phi)\,d\boldsymbol{x},
$$

一般不為零。Laplacian 部分不改變總量，但局部反應 $\phi-\phi^3$ 會改變它。

即使某個對稱初始條件恰好使積分為零，也只是特殊解或數值對稱造成的結果，不能據此宣稱 Allen–Cahn 是守恆模型。

### 4. 平面界面與界面厚度

一維靜態界面滿足 $\mu=0$：

$$
\phi^3-\phi-\kappa\phi_{xx}=0.
$$

連接 $-1$ 與 $1$ 的解為

$$
\phi(x)
=
\tanh\left(\frac{x-x_0}{\sqrt{2\kappa}}\right).
$$

驗證時令 $\xi=(x-x_0)/\sqrt{2\kappa}$。由

$$
\phi_x=\frac{1}{\sqrt{2\kappa}}(1-\phi^2)
$$

可得

$$
\kappa\phi_{xx}=\phi^3-\phi.
$$

因此 $\mu=0$。特徵界面厚度與 $\sqrt{\kappa}$ 成正比。數值上若界面只有一兩個格點，雖然程式仍能輸出陣列，曲率與能量通常無法可靠解析。所需格點數取決於誤差要求，不能把單一經驗數字當成普遍定理。

### 5. 曲率驅動的意義

在薄界面漸近極限、尺度與遷移率採適當縮放、遠離其他強迫且界面光滑時，Allen–Cahn 界面法向速度與平均曲率相關。直覺上，小圓形區域具有較大曲率，通常傾向縮小。

但「曲率驅動」不是任意參數下的精確幾何公式。有限界面厚度、邊界作用、外場、各向異性能量及數值誤差都會改變速度。小液滴消失也會改變 $\int\phi\,d\boldsymbol{x}$，再次反映非守恆性。

### 6. 空間離散與半離散能量

使用二維週期單元中心網格：

$$
\phi_{j,i}\approx
\phi\left(
\left(i+\frac12\right)\Delta x,
\left(j+\frac12\right)\Delta y
\right).
$$

陣列 `phi[j,i]` 的形狀為 `(Ny,Nx)`，$i$ 沿 $+X$、$j$ 沿 $+Y$。五點 Laplacian 為

$$
(L\phi)_{j,i}
=
\frac{\phi_{j,i-1}-2\phi_{j,i}+\phi_{j,i+1}}{\Delta x^2}
+
\frac{\phi_{j-1,i}-2\phi_{j,i}+\phi_{j+1,i}}{\Delta y^2}.
$$

離散能量可取

$$
F_h
=
\Delta x\Delta y
\sum_{j,i}
\left[
W(\phi_{j,i})
+
\frac{\kappa}{2}
\left(
(D_x^+\phi)^2+(D_y^+\phi)^2
\right)_{j,i}
\right].
$$

週期差分下，其離散梯度為

$$
\mu_h=\phi^{\circ3}-\phi-\kappa L\phi.
$$

半離散系統

$$
\dot{\boldsymbol{\phi}}=-M\boldsymbol{\mu}_h
$$

滿足

$$
\frac{dF_h}{dt}
=
-M\Delta x\Delta y
\sum_{j,i}\mu_{h,j,i}^2
\leq0.
$$

這仍是時間連續的半離散結果。

### 7. 顯式 Euler 與條件性

顯式 Euler 更新為

$$
\phi^{n+1}
=
\phi^n
+
\Delta t\,M
\left[
\kappa L\phi^n+\phi^n-(\phi^n)^3
\right].
$$

在均勻穩定態 $\phi=\pm1$ 附近令 $\phi=\pm1+\epsilon$，線性化後

$$
\epsilon_t=M(\kappa\Delta\epsilon-2\epsilon).
$$

二維週期五點 Laplacian 的最負特徵值絕對值不超過

$$
4\left(\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}\right).
$$

因此線性化顯式 Euler 的必要且在該線性模型下充分的穩定條件為

$$
\Delta t\,M
\left[
2+
4\kappa
\left(
\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}
\right)
\right]
\leq2.
$$

這只是穩定態附近的線性條件，不是非線性全域能量下降保證。實作仍須逐步計算 $F_h^{n+1}-F_h^n$；若能量上升，應拒絕該步或縮小步長，而不是修改結果後假稱耗散。

---

## 逐步手算例題

### 例題一：均勻場會改變總量

令空間均勻、$M=1$、$\phi^0=0.5$、$\Delta t=0.1$。因 Laplacian 為零，

$$
\mu^0=(0.5)^3-0.5=-0.375.
$$

因此

$$
\phi^1
=
0.5-0.1(-0.375)
=
0.5375.
$$

每個格點都增加，故序參量總量增加。初始勢能密度為

$$
W(0.5)=\frac{(0.25-1)^2}{4}=0.140625.
$$

更新後

$$
W(0.5375)\approx0.1264.
$$

能量下降，但總量不守恆。這直接反駁「耗散梯度流必然守恆」的錯誤推論。

### 例題二：過大步長造成能量上升

仍取均勻場，但令 $\phi^0=2$、$M=1$、$\Delta t=1$：

$$
\mu^0=2^3-2=6,
$$

$$
\phi^1=2-6=-4.
$$

更新前後勢能為

$$
W(2)=\frac{9}{4}=2.25,
$$

$$
W(-4)=\frac{225}{4}=56.25.
$$

連續 Allen–Cahn 方程耗散自由能，但此顯式離散步驟使能量大幅上升。不能把連續耗散律當成任意 $\Delta t$ 的保證。

### 例題三：平面界面

令 $\kappa=0.02$、$x_0=0$：

$$
\phi(x)=\tanh\left(\frac{x}{0.2}\right),
$$

因為 $\sqrt{2\kappa}=0.2$。在 $x=0$，$\phi=0$ 且斜率為 $5$。若 $\Delta x=0.1$，特徵尺度 $0.2$ 只有兩個網格間距；是否足夠必須靠網格細化與能量、速度誤差檢查，不能只看曲線平滑。

---

## 實作與程式

以下為自足的 NumPy CPU 實作。它使用二維週期單元中心網格、顯式 Euler、離散能量診斷與可選的能量上升拒絕機制。它不會裁切 $\phi$。

```python
import numpy as np


def positive_finite(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必須正且有限")
    return value


def validate_phi(phi):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2 or min(phi.shape) < 2:
        raise ValueError("phi 必須是每一方向至少兩格的二維陣列")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi 含 NaN 或無窮大")
    return phi.copy()


def laplacian_periodic(phi, dx, dy):
    return (
        (np.roll(phi, -1, axis=1) - 2.0 * phi
         + np.roll(phi, 1, axis=1)) / dx**2
        +
        (np.roll(phi, -1, axis=0) - 2.0 * phi
         + np.roll(phi, 1, axis=0)) / dy**2
    )


def chemical_potential(phi, kappa, dx, dy):
    return phi**3 - phi - kappa * laplacian_periodic(phi, dx, dy)


def discrete_energy(phi, kappa, dx, dy):
    gx = (np.roll(phi, -1, axis=1) - phi) / dx
    gy = (np.roll(phi, -1, axis=0) - phi) / dy
    potential = 0.25 * (phi**2 - 1.0)**2
    density = potential + 0.5 * kappa * (gx**2 + gy**2)
    return float(dx * dy * np.sum(density))


def explicit_step(phi, dt, mobility, kappa, dx, dy):
    mu = chemical_potential(phi, kappa, dx, dy)
    return phi - dt * mobility * mu, mu


def simulate(phi0, steps, dt, mobility, kappa, dx, dy,
             reject_energy_increase=True, energy_tol=1.0e-12):
    phi = validate_phi(phi0)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps 必須是非布林整數")
    if steps < 0:
        raise ValueError("steps 不可為負")

    dt = positive_finite("dt", dt)
    mobility = positive_finite("mobility", mobility)
    kappa = positive_finite("kappa", kappa)
    dx = positive_finite("dx", dx)
    dy = positive_finite("dy", dy)
    energy_tol = positive_finite("energy_tol", energy_tol)

    records = []
    energy = discrete_energy(phi, kappa, dx, dy)

    for n in range(steps):
        trial, mu = explicit_step(
            phi, dt, mobility, kappa, dx, dy
        )
        if not np.all(np.isfinite(trial)):
            raise FloatingPointError("更新產生非有限值")

        new_energy = discrete_energy(trial, kappa, dx, dy)
        scale = max(1.0, abs(energy))
        increased = new_energy > energy + energy_tol * scale

        records.append({
            "step": n + 1,
            "energy_before": energy,
            "energy_after": new_energy,
            "mean_before": float(np.mean(phi)),
            "mean_after": float(np.mean(trial)),
            "max_abs_mu": float(np.max(np.abs(mu))),
            "energy_increased": bool(increased),
        })

        if increased and reject_energy_increase:
            raise RuntimeError(
                "離散能量上升；請縮小 dt 或改用具能量性質的格式"
            )

        phi = trial
        energy = new_energy

    return phi, records


def diagnostics():
    # 穩定均勻態
    phi = np.ones((4, 5))
    out, rec = simulate(phi, 1, 0.01, 1.0, 0.02, 0.1, 0.1)
    assert np.allclose(out, phi)

    # 非守恆均勻場：0.5 應更新為 0.5375
    phi = 0.5 * np.ones((3, 4))
    out, _ = simulate(phi, 1, 0.1, 1.0, 0.02, 1.0, 1.0)
    assert np.allclose(out, 0.5375)
    assert not np.isclose(np.mean(out), np.mean(phi))

    # 故障步長：均勻 phi=2、dt=1 會使能量上升
    try:
        simulate(
            2.0 * np.ones((2, 2)),
            1, 1.0, 1.0, 0.02, 1.0, 1.0
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("能量上升步驟未被拒絕")

    # 非有限輸入
    bad = np.ones((2, 2))
    bad[0, 0] = np.nan
    try:
        simulate(bad, 1, 0.01, 1.0, 0.02, 1.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN 未被拒絕")


if __name__ == "__main__":
    diagnostics()
```

`np.roll` 在每一方向實作週期連接，不額外儲存重複邊界面。總量若要以每單位厚度計算，應使用

$$
Q_h=\Delta x\Delta y\sum_{j,i}\phi_{j,i}.
$$

`mean` 只因均勻網格與固定面積而與 $Q_h$ 成正比。

---

## 測試與預期結果

### 正常測試

1. $\phi\equiv1$ 與 $\phi\equiv-1$ 的化學勢均為零，應保持不變。
2. $\phi\equiv0.5$、$M=1$、$\Delta t=0.1$ 應更新為 $0.5375$。
3. 小步長、平滑初值通常應使離散能量下降；這是該次計算的診斷，不是未經檢查的普遍保證。
4. 週期 Laplacian 的全域和應在浮點容差內為零。

### 邊界與故障測試

- `steps=0` 應原樣回傳初值與空紀錄。
- 非正的 $M$、$\kappa$、$\Delta t$、$\Delta x$ 或 $\Delta y$ 必須拒絕。
- `NaN`、無窮大及非二維初值必須拒絕。
- $\phi=2$、$\Delta t=1$ 的均勻反例預期觸發能量上升拒絕。
- 關閉 `reject_energy_increase` 只表示保留故障資料，不表示結果可信。

### 六項性質分開判斷

- **數值穩定**：擾動是否有界，依時間格式、步長、網格與狀態而定。
- **守恆**：Allen–Cahn 一般不守恆 $Q_h$；總量變化不是自動故障。
- **能量下降**：連續與半離散成立；顯式時間離散需條件及逐步診斷。
- **非負性**：$\phi$ 不是濃度，且模型不要求 $\phi\geq0$。
- **值域**：方程不保證離散 $\phi$ 永遠位於 $[-1,1]$，不得事後裁切冒充穩定。
- **物理可信度**：還需確認序參量定義、尺度、邊界、參數與目標現象相容。

---

## 除錯與常見陷阱

1. **Laplacian 符號錯誤**：正確形式為 $\phi_t=M(\kappa\Delta\phi+\phi-\phi^3)$。
2. **把 Allen–Cahn 當守恆方程**：週期邊界只能消除 Laplacian 的總和，不能消除反應項。
3. **宣稱任意步長能量下降**：顯式 Euler 有步長限制。
4. **以 `clip` 限制 $[-1,1]$**：裁切會改變總量與能量，也改變原方程。
5. **界面解析不足**：看似平滑不代表曲率速度或界面能已收斂。
6. **把非線性溢位當物理成核**：巨大數值或 `NaN` 是數值故障。
7. **混淆陣列與物理方向**：`phi[j,i]` 中 $j$ 沿 $+Y$；繪圖須使用 `origin="lower"`。
8. **把能量下降當成物理驗證**：錯誤參數也可能形成耗散數值系統。

---

## 養殖與相場案例

Allen–Cahn 可作為合成材料界面或非守恆序參量鬆弛的教學模型，但不能把溶氧濃度直接視為 $\phi$。溶氧以 $\mathrm{kg/m^3}$ 表示，在封閉控制體中需遵守輸運、反應與邊界通量收支；Allen–Cahn 的局部反應則可無通量地改變 $\int\phi$。

若為合成池域建立「狀態偏好指標」$\phi$，必須明列它是無因次模型變數，而非感測器濃度。$\phi$ 跨越零只表示雙井模型中的狀態轉換，不是養殖管理閾值，也不是熱力學相變的現場證據。

相場圖形即使逼真，也只表示所選模型與數值方法產生某種形態。未經獨立資料、尺度辨識與模型驗證，不能據此控制投餌、加藥、曝氣或真實設備。

---

## 習題

### 習題一：手算

對空間均勻 Allen–Cahn 方程，取 $M=2$、$\phi^0=-0.5$、$\Delta t=0.05$。計算一步顯式 Euler，並判斷總量方向。

### 習題二：程式

修改程式，記錄

$$
D_h^n=M\Delta x\Delta y\sum_{j,i}(\mu_{j,i}^n)^2
$$

並比較

$$
\frac{F_h^{n+1}-F_h^n}{\Delta t}
$$

與 $-D_h^n$。說明為何有限步長下兩者不必完全相等。

### 習題三：反例

構造一個連續模型能量耗散、但顯式 Euler 離散能量上升的均勻場例子。

### 習題四：整合

在正方形週期域中放置一個近似圓形正相液滴，外部為負相。列出至少五項診斷，用來區分曲率縮小、數值不穩定與網格不足。

### 習題五：模型選擇

封閉容器中某濃度的空間總量必須守恆。是否可直接使用標準 Allen–Cahn？若不適合，應從哪個模型結構重新考慮？

---

## 習題解答

### 解答一

均勻場的 Laplacian 為零：

$$
\mu=(-0.5)^3-(-0.5)=0.375.
$$

因此

$$
\phi^1
=
-0.5-(0.05)(2)(0.375)
=
-0.5375.
$$

場值整體下降，故序參量總量也下降。這不是邊界通量造成，而是非守恆反應項造成。

### 解答二

在 `explicit_step` 後加入

```python
dissipation = mobility * dx * dy * np.sum(mu**2)
rate = (new_energy - energy) / dt
```

連續或半離散關係為 $dF_h/dt=-D_h$。顯式 Euler 使用有限步長，能量差還包含高階時間截斷項，因此一般只有在 $\Delta t\to0$ 時趨近該關係。若步長過大，`rate` 甚至可能為正。

### 解答三

取均勻場 $\phi^0=2$、$M=1$、$\Delta t=1$。顯式更新給出 $\phi^1=-4$，而

$$
W(2)=2.25,\qquad W(-4)=56.25.
$$

離散能量上升。連續耗散律沒有失效；失效的是無條件套用顯式時間離散的推論。

### 解答四

至少記錄：

1. 離散自由能 $F_h$；
2. 序參量總量 $Q_h$；
3. 液滴等值線包圍面積或等效半徑；
4. $\max|\mu_h|$；
5. $\max|\phi|$ 與是否產生非有限值；
6. 不同 $\Delta t$ 的結果；
7. 不同 $\Delta x,\Delta y$ 的結果；
8. 界面寬度含有多少格點。

若半徑隨曲率規律縮小、能量下降且步長與網格細化後結果一致，才較支持曲率驅動解釋。若高頻振盪、能量上升或細化後速度大變，應先判為數值問題。

### 解答五

標準 Allen–Cahn 一般不適合嚴格守恆濃度，因為其局部反應會改變總量。應考慮守恆通量形式，例如 Cahn–Hilliard 結構

$$
\phi_t=\nabla\cdot(M\nabla\mu),
$$

並同時設定適當的無通量或週期邊界。若物理量是普通稀溶質濃度，還應先比較平流—擴散—反應模型，而不是預設必須使用相場方程。

---

## 本章小結

Allen–Cahn 方程是自由能的 $L^2$ 梯度流：

$$
\phi_t=-M\mu,
\qquad
\mu=\phi^3-\phi-\kappa\Delta\phi.
$$

在週期或適當自然邊界下，連續自由能單調不增，但序參量總量一般不守恆。平面界面具有雙曲正切輪廓，界面厚度尺度與 $\sqrt{\kappa}$ 相關；曲率驅動則需要薄界面與適當尺度等條件。

週期有限差分可保留半離散耗散結構，但顯式 Euler 並非無條件能量穩定。實作必須同時記錄能量、總量、化學勢、值域、非有限值、時間步長與網格解析度，不得以裁切或平滑掩蓋故障。

---

## 參考來源

1. **F1：FiPy 有限體積離散與邊界**  
   <https://pages.nist.gov/fipy/en/latest/numerical/discret.html>

2. **F2：FEniCSx Poisson 與弱形式**  
   <https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html>

3. **F6：FiPy Cahn–Hilliard 相分離示範**  
   <https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html>

4. **F7：FiPy 簡單相場與固液相變示範**  
   <https://pages.nist.gov/fipy/en/latest/generated/examples.phase.simple.html>

FiPy 範例提供相場離散與邊界處理的參考，但其序參量可能採 $[0,1]$ 慣例；本章採 $[-1,1]$ 雙井勢，不能直接複製參數。上述來源也不代表本章所有跨主題主張均已逐條由來源驗證。