# 第27章 Allen–Cahn與非守恆相場

## 學習目標與先備知識

本章研究 Allen–Cahn 方程，即自由能在 $L^2$ 度量下的梯度流。完成本章後，讀者應能：

1. 由自由能泛函推導化學勢與 Allen–Cahn 方程。
2. 說明序參量總量為何一般不守恆。
3. 區分連續耗散、空間半離散耗散與時間離散能量下降。
4. 解釋界面厚度、平面界面與曲率驅動的適用條件。
5. 在二維週期單元中心網格上實作完整顯式 Euler 格式。
6. 分別檢查數值穩定、守恆、能量下降、非負性與物理可信度。
7. 拒絕非有限輸入與非法參數，而不以裁切掩蓋不穩定。

先備知識包括 Laplacian、週期有限差分、變分導數與顯式時間積分。本章採無因次模型；若改用有因次形式，必須重新列出自由能、梯度係數與遷移率的單位。

---

## 問題與直覺

相場方法以連續序參量 $\phi$ 描述兩種狀態。本卷採用

$$
W(\phi)=\frac{(\phi^2-1)^2}{4},
$$

故均勻穩定狀態位於 $\phi=\pm1$。這不表示數值解必須被人為限制在 $[-1,1]$；超出此範圍可能是方程本身允許的暫態，也可能是數值故障，必須靠步長與網格診斷判別，而不是直接裁切。

自由能為

$$
F[\phi]=\int_\Omega
\left[
W(\phi)+\frac{\kappa}{2}|\nabla\phi|^2
\right]d\boldsymbol{x},
\qquad \kappa>0.
$$

雙井勢偏好 $\phi=\pm1$，梯度項則懲罰過於尖銳的空間變化，兩者競爭後形成有限厚度界面。

Allen–Cahn 動力學為

$$
\phi_t=-M\mu,\qquad M>0,
$$

其中 $\mu$ 是化學勢。此方程允許每個位置直接改變序參量，因此一般不守恆 $\int_\Omega\phi\,d\boldsymbol{x}$。若待模擬量是封閉系統中的守恆濃度，不能只因 Allen–Cahn 容易計算就忽略模型不相容性。

---

## 數學與物理推導

### 第一變分與化學勢

令擾動為 $\phi+\epsilon\eta$，則

$$
\left.\frac{d}{d\epsilon}F[\phi+\epsilon\eta]\right|_{\epsilon=0}
=
\int_\Omega
\left[
W'(\phi)\eta+\kappa\nabla\phi\cdot\nabla\eta
\right]d\boldsymbol{x}.
$$

因為

$$
W'(\phi)=\phi^3-\phi,
$$

分部積分後得到

$$
\delta F
=
\int_\Omega
\left[
\phi^3-\phi-\kappa\Delta\phi
\right]\eta\,d\boldsymbol{x}
+
\int_{\partial\Omega}\kappa\partial_n\phi\,\eta\,dS.
$$

在週期邊界或自然邊界 $\partial_n\phi=0$ 下，邊界項消失，因此

$$
\mu=\frac{\delta F}{\delta\phi}
=\phi^3-\phi-\kappa\Delta\phi.
$$

Allen–Cahn 方程可寫成

$$
\phi_t
=
-M(\phi^3-\phi-\kappa\Delta\phi)
=
M(\kappa\Delta\phi+\phi-\phi^3).
$$

### 連續能量耗散

沿解軌跡，

$$
\frac{dF}{dt}
=
\int_\Omega\mu\phi_t\,d\boldsymbol{x}
=
-M\int_\Omega\mu^2\,d\boldsymbol{x}
\leq0.
$$

若 $M=M(\boldsymbol{x},\phi)>0$，則右端改為 $-\int_\Omega M\mu^2\,d\boldsymbol{x}$，仍然非正。這是連續方程的耗散律，不代表任意空間格式、時間步長或非線性求解容差都保證離散能量下降。

### 非守恆性

定義

$$
Q(t)=\int_\Omega\phi\,d\boldsymbol{x}.
$$

在週期或零法向梯度邊界下，

$$
\frac{dQ}{dt}
=
-M\int_\Omega\mu\,d\boldsymbol{x}
=
-M\int_\Omega(\phi^3-\phi)\,d\boldsymbol{x},
$$

因為 Laplacian 的積分為零。右端一般不為零。即使某個對稱初值恰使 $Q$ 不變，也只是特殊情形，不能據此宣稱模型守恆。

若希望保持指定平均值，可以引入拉格朗日乘子形成受限 Allen–Cahn；但那已不是本章的標準方程，必須重新推導乘子、離散更新與耗散律，不能暗中於每一步減去均值修正。

### 平面界面與厚度

一維靜態界面滿足

$$
\phi^3-\phi-\kappa\phi_{xx}=0.
$$

連接 $-1$ 與 $1$ 的解為

$$
\phi(x)=
\tanh\left(\frac{x-x_0}{\sqrt{2\kappa}}\right).
$$

由

$$
\phi_x=\frac{1-\phi^2}{\sqrt{2\kappa}}
$$

可驗證 $\kappa\phi_{xx}=\phi^3-\phi$。因此界面厚度尺度與 $\sqrt{\kappa}$ 成正比。若界面只跨越一兩格，曲率、界面能及移動速度通常未被充分解析；是否足夠必須靠網格細化判斷。

### 曲率驅動

在薄界面極限、尺度與遷移率適當縮放、界面光滑且沒有額外強迫時，Allen–Cahn 界面法向速度與平均曲率相關。因此小圓形正相區域通常傾向縮小。

這不是任意有限 $\kappa$ 下的精確幾何公式。邊界、外場、各向異性、網格偏向與時間誤差都會改變速度。液滴消失同時改變 $Q$，正是非守恆性的表現。

### 空間半離散

使用二維週期單元中心網格：

$$
\phi_{j,i}\approx
\phi\left(
(i+\tfrac12)\Delta x,
(j+\tfrac12)\Delta y
\right).
$$

`phi[j,i]` 形狀為 `(Ny,Nx)`，$i$ 沿 $+X$，$j$ 沿 $+Y$。離散 Laplacian 為

$$
(L\phi)_{j,i}
=
\frac{\phi_{j,i-1}-2\phi_{j,i}+\phi_{j,i+1}}{\Delta x^2}
+
\frac{\phi_{j-1,i}-2\phi_{j,i}+\phi_{j+1,i}}{\Delta y^2}.
$$

離散能量取為

$$
F_h
=
\Delta x\Delta y\sum_{j,i}
\left[
W(\phi_{j,i})
+\frac{\kappa}{2}
\left((D_x^+\phi)^2+(D_y^+\phi)^2\right)_{j,i}
\right].
$$

其離散梯度為

$$
\mu_h=\phi^{\circ3}-\phi-\kappa L\phi.
$$

半離散系統 $\dot{\boldsymbol{\phi}}=-M\boldsymbol{\mu}_h$ 滿足

$$
\frac{dF_h}{dt}
=
-M\Delta x\Delta y\sum_{j,i}\mu_{h,j,i}^2
\leq0.
$$

### 顯式 Euler 與步長條件

完整更新為

$$
\phi^{n+1}
=
\phi^n+\Delta t\,M
\left[
\kappa L\phi^n+\phi^n-(\phi^n)^3
\right].
$$

在 $\phi=\pm1$ 附近線性化，令 $\phi=\pm1+\epsilon$：

$$
\epsilon_t=M(\kappa\Delta\epsilon-2\epsilon).
$$

二維五點週期 Laplacian 的最負特徵值絕對值不超過

$$
4\left(\Delta x^{-2}+\Delta y^{-2}\right).
$$

因此線性化顯式 Euler 的條件為

$$
\Delta t M
\left[
2+4\kappa
\left(\Delta x^{-2}+\Delta y^{-2}\right)
\right]
\leq2.
$$

這只適用於穩定均勻態附近的線性問題，不是非線性全域能量保證。實作仍須逐步檢查 $F_h^{n+1}\leq F_h^n$。

---

## 逐步手算例題

### 例題一：能量下降但總量改變

令均勻場 $\phi^0=0.5$、$M=1$、$\Delta t=0.1$。Laplacian 為零，所以

$$
\mu^0=(0.5)^3-0.5=-0.375,
$$

$$
\phi^1=0.5-0.1(-0.375)=0.5375.
$$

每格均增加，故總量增加。勢能由

$$
W(0.5)=0.140625
$$

降至約

$$
W(0.5375)\approx0.1264.
$$

因此能量下降與總量守恆是不同性質。

### 例題二：過大步長破壞能量下降

令均勻場 $\phi^0=2$、$M=1$、$\Delta t=1$：

$$
\mu^0=2^3-2=6,
\qquad
\phi^1=2-6=-4.
$$

更新前後勢能為

$$
W(2)=2.25,\qquad W(-4)=56.25.
$$

連續方程耗散自由能，但此顯式離散步驟使能量大幅上升。問題在時間離散，不在連續耗散推導。

### 例題三：界面解析尺度

若 $\kappa=0.02$，則 $\sqrt{2\kappa}=0.2$，平面界面為

$$
\phi(x)=\tanh(x/0.2).
$$

在中心的斜率為 $5$。若 $\Delta x=0.1$，特徵尺度只有兩個網格間距；必須比較更細網格上的能量與界面位置，不能僅因圖形平滑就宣稱解析充分。

---

## 實作與程式

以下是自足的 NumPy CPU 實作。能量上升時，試算步驟**不會被接受**，函式直接拋出例外；它不會自動縮小步長或續算。呼叫端必須保留原始初值，改用較小 `dt` 重新呼叫。由於 `phi` 只在檢查通過後才指派為 `trial`，例外發生前最後一個已接受狀態沒有被就地修改。

```python
import numpy as np


def positive_finite(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必須正且有限")
    return value


def nonnegative_finite(name, value):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} 必須非負且有限")
    return value


def validate_phi(phi):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2 or min(phi.shape) < 2:
        raise ValueError("phi 必須是每方向至少兩格的二維陣列")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi 含 NaN 或無窮大")
    return phi.copy()


def laplacian_periodic(phi, dx, dy):
    return (
        (np.roll(phi, -1, axis=1) - 2.0 * phi
         + np.roll(phi, 1, axis=1)) / dx**2
        + (np.roll(phi, -1, axis=0) - 2.0 * phi
           + np.roll(phi, 1, axis=0)) / dy**2
    )


def chemical_potential(phi, kappa, dx, dy):
    return phi**3 - phi - kappa * laplacian_periodic(phi, dx, dy)


def discrete_energy(phi, kappa, dx, dy):
    gx = (np.roll(phi, -1, axis=1) - phi) / dx
    gy = (np.roll(phi, -1, axis=0) - phi) / dy
    w = 0.25 * (phi**2 - 1.0)**2
    return float(
        dx * dy * np.sum(w + 0.5 * kappa * (gx**2 + gy**2))
    )


def explicit_step(phi, dt, mobility, kappa, dx, dy):
    mu = chemical_potential(phi, kappa, dx, dy)
    return phi - dt * mobility * mu, mu


def simulate(phi0, steps, dt, mobility, kappa, dx, dy,
             reject_energy_increase=True, energy_tol=1.0e-12):
    phi = validate_phi(phi0)

    if isinstance(steps, (bool, np.bool_)):
        raise TypeError("steps 不可為布林值")
    if not isinstance(steps, (int, np.integer)):
        raise TypeError("steps 必須是整數")
    if steps < 0:
        raise ValueError("steps 不可為負")

    dt = positive_finite("dt", dt)
    mobility = positive_finite("mobility", mobility)
    kappa = positive_finite("kappa", kappa)
    dx = positive_finite("dx", dx)
    dy = positive_finite("dy", dy)
    energy_tol = nonnegative_finite("energy_tol", energy_tol)

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

        record = {
            "step": n + 1,
            "energy_before": energy,
            "energy_after": new_energy,
            "mean_before": float(np.mean(phi)),
            "mean_after": float(np.mean(trial)),
            "max_abs_mu": float(np.max(np.abs(mu))),
            "dissipation_rate": float(
                mobility * dx * dy * np.sum(mu**2)
            ),
            "energy_increased": bool(increased),
        }

        if increased and reject_energy_increase:
            raise RuntimeError(
                "試算步驟未接受；呼叫端須以較小 dt 重新執行"
            )

        records.append(record)
        phi = trial
        energy = new_energy

    return phi, records


def diagnostics():
    # 穩定均勻態
    one = np.ones((4, 5))
    out, _ = simulate(one, 1, 0.01, 1.0, 0.02, 0.1, 0.1)
    assert np.allclose(out, one)

    # steps=0 與零能量容差均合法
    out, records = simulate(
        one, 0, 0.01, 1.0, 0.02, 0.1, 0.1,
        energy_tol=0.0
    )
    assert np.allclose(out, one)
    assert records == []

    # 非守恆均勻場
    half = 0.5 * np.ones((3, 4))
    out, _ = simulate(half, 1, 0.1, 1.0, 0.02, 1.0, 1.0)
    assert np.allclose(out, 0.5375)
    assert not np.isclose(np.mean(out), np.mean(half))

    # 過大步長應拒絕，試算值不會回傳為已接受狀態
    try:
        simulate(
            2.0 * np.ones((2, 2)),
            1, 1.0, 1.0, 0.02, 1.0, 1.0
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("能量上升步驟未被拒絕")

    # 負容差與非有限初值應拒絕
    try:
        simulate(one, 1, 0.01, 1.0, 0.02, 1.0, 1.0,
                 energy_tol=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("負容差未被拒絕")

    bad = one.copy()
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

週期邊界由 `np.roll` 實作，不另存重複邊界面。每單位厚度的離散總量為

$$
Q_h=\Delta x\Delta y\sum_{j,i}\phi_{j,i}.
$$

`mean` 只在均勻網格與固定域面積下與 $Q_h$ 成正比。

若需要自動回溯，可由呼叫端捕捉 `RuntimeError`，令 $\Delta t$ 減半後從上一個已接受狀態重試；重試步驟不得先累加時間或寫入接受紀錄。為保持核心程式透明，本章不把自動控制藏入求解器。

---

## 測試與預期結果

### 正常測試

1. $\phi\equiv\pm1$ 的化學勢為零，應保持不變。
2. $\phi\equiv0.5$、$M=1$、$\Delta t=0.1$ 應更新為 $0.5375$。
3. `steps=0` 應原樣回傳初值及空紀錄。
4. `energy_tol=0` 合法，表示不額外容許正的能量差；但浮點捨入仍可能使極小差異觸發拒絕。
5. 週期 Laplacian 的全域和應在浮點容差內為零。

### 故障測試

- 非正的 $M$、$\kappa$、$\Delta t$、$\Delta x$ 或 $\Delta y$ 必須拒絕。
- 負的 `energy_tol`、非整數步數、`NaN` 與無窮大必須拒絕。
- $\phi=2$、$\Delta t=1$ 的反例應觸發 `RuntimeError`。
- 關閉拒絕功能只適合保存故障證據，不表示結果可信。

### 各項性質分開檢查

- **數值穩定**：擾動是否受控，依步長、網格及狀態而定。
- **守恆**：標準 Allen–Cahn 一般不守恆 $Q_h$。
- **能量下降**：連續與相容半離散系統耗散；顯式時間格式需條件與逐步診斷。
- **非負性**：$\phi$ 不是濃度，模型也不要求 $\phi\geq0$。
- **值域限制**：不保證離散解永遠位於 $[-1,1]$。
- **物理可信度**：還需確認序參量定義、參數、尺度與邊界符合目標現象。

即使能量逐步下降，時間誤差仍可能很大；即使解保持在 $[-1,1]$，界面速度也可能未收斂。反之，總量改變是標準 Allen–Cahn 的模型特徵，不應自動當成守恆程式錯誤。

---

## 除錯與常見陷阱

1. **Laplacian 符號錯誤**：正確形式為 $\phi_t=M(\kappa\Delta\phi+\phi-\phi^3)$。
2. **把 Allen–Cahn 當守恆方程**：週期條件只消除 Laplacian 的總和。
3. **宣稱任意步長能量下降**：顯式 Euler 不是無條件能量穩定。
4. **誤以為例外會自動縮步**：本章程式只拒絕試算步驟，呼叫端必須重跑。
5. **以 `clip` 限制值域**：裁切會改變總量與能量，也改變原方程。
6. **界面解析不足**：圖形平滑不等於曲率速度或界面能已收斂。
7. **把溢位當成成核**：巨大數值與非有限值是數值故障。
8. **混淆陣列與物理方向**：`phi[j,i]` 中 $j$ 沿 $+Y$；繪圖應使用 `origin="lower"`。
9. **把能量下降當物理驗證**：錯誤模型也可能具有完善耗散律。

---

## 養殖與相場案例

Allen–Cahn 可作為合成材料界面或非守恆狀態指標的教學模型，但不能把溶氧濃度直接視為 $\phi$。溶氧以 $\mathrm{kg/m^3}$ 表示，必須遵守輸運、反應與邊界通量收支；Allen–Cahn 則可在無邊界通量時由局部反應改變總量。

若為合成池域建立「狀態偏好指標」$\phi$，必須明列它是無因次模型變數，不是感測器濃度。$\phi$ 跨越零只代表雙井模型中的狀態轉換，不是溶氧管理閾值，也不是現場熱力學相變證據。

相場圖形逼真只表示模型產生某種形態。未經尺度辨識、獨立資料與模型驗證，不得據此投餌、加藥、曝氣或控制真實設備。

---

## 習題

### 習題一：手算

對均勻場取 $M=2$、$\phi^0=-0.5$、$\Delta t=0.05$。計算一步顯式 Euler 並判斷總量方向。

### 習題二：程式

比較

$$
\frac{F_h^{n+1}-F_h^n}{\Delta t}
$$

與

$$
-D_h^n,\qquad
D_h^n=M\Delta x\Delta y\sum_{j,i}(\mu_{j,i}^n)^2.
$$

說明有限步長下兩者為何不必相等。

### 習題三：反例

構造連續模型耗散、但顯式 Euler 離散能量上升的均勻場。

### 習題四：整合

在週期正方形域放置近似圓形正相液滴。列出至少五項診斷，以區分曲率縮小、數值不穩定與網格不足。

### 習題五：模型選擇

封閉容器中的濃度總量必須守恆。是否可直接使用標準 Allen–Cahn？應考慮何種替代結構？

---

## 習題解答

### 解答一

均勻場的 Laplacian 為零：

$$
\mu=(-0.5)^3-(-0.5)=0.375.
$$

因此

$$
\phi^1=-0.5-(0.05)(2)(0.375)=-0.5375.
$$

場值整體下降，所以總量下降。這是局部非守恆反應造成，不是邊界通量造成。

### 解答二

程式已在紀錄中提供 `dissipation_rate`。可計算

```python
rate = (
    record["energy_after"] - record["energy_before"]
) / dt
difference = rate + record["dissipation_rate"]
```

半離散關係為 $dF_h/dt=-D_h$。有限步長的能量差含有時間截斷誤差，所以不必精確等於 $-D_h^n$；步長縮小後差異預期減少，但正式階數仍需多步長研究。

### 解答三

取 $\phi^0=2$、$M=1$、$\Delta t=1$。更新後 $\phi^1=-4$，且

$$
W(2)=2.25,\qquad W(-4)=56.25.
$$

離散能量上升。連續耗散律並未失效，錯誤是把它無條件套用到顯式時間步。

### 解答四

至少應記錄：

1. 離散自由能；
2. 序參量總量；
3. 液滴面積或等效半徑；
4. 最大化學勢；
5. 最大序參量絕對值及非有限值；
6. 不同時間步長的比較；
7. 不同網格間距的比較；
8. 界面跨越的格點數。

能量下降、半徑平滑縮小且時間與空間細化後結果一致，才較支持曲率驅動解釋。高頻振盪、能量上升或細化後速度大幅改變，則顯示數值問題。

### 解答五

標準 Allen–Cahn 一般不適合嚴格守恆濃度。可考慮 Cahn–Hilliard 型守恆通量結構

$$
\phi_t=\nabla\cdot(M\nabla\mu),
$$

並設定週期或無通量邊界。若待模擬量只是普通稀溶質濃度，還應先考慮平流—擴散—反應模型，而不是預設相場模型。

---

## 本章小結

Allen–Cahn 方程是自由能的 $L^2$ 梯度流：

$$
\phi_t=-M\mu,\qquad
\mu=\phi^3-\phi-\kappa\Delta\phi.
$$

連續與相容半離散系統耗散自由能，但序參量總量一般不守恆。平面界面具有雙曲正切輪廓，厚度尺度與 $\sqrt{\kappa}$ 相關；曲率驅動則需要薄界面與適當尺度等條件。

顯式 Euler 並非無條件能量穩定。本章程式在能量上升時拒絕試算步驟，但不自動縮小步長；呼叫端必須從上一個已接受狀態重跑。可信計算應同時記錄能量、總量、化學勢、值域、非有限值、時間步長與網格解析度，且不得以裁切或平滑掩蓋故障。

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

FiPy 範例提供相場離散與邊界處理的參考，但其序參量可能採 $[0,1]$ 慣例；本章採 $[-1,1]$ 雙井勢，不能直接複製參數。