# 第22章 多物理場耦合與時間尺度

## 學習目標與先備知識

完成本章後，讀者應能：

1. 區分單向耦合、雙向耦合、弱耦合與強耦合。
2. 由交換通量推導兩子系統的總量守恆與能量收支。
3. 比較整體式求解與分區式求解，理解Lie分裂與固定點迭代的角色。
4. 以Jacobian或Lipschitz常數判斷固定點迭代的局部收斂。
5. 辨識傳輸、擴散、反應、交換及耦合迭代等不同時間尺度。
6. 分開診斷時間離散誤差、耦合殘差、代數殘差與物理模型誤差。
7. 以自足NumPy程式模擬兩個交換子系統，檢查總量守恆與步長細化。

先備知識包括常微分方程、守恆律、顯式與隱式時間積分、矩陣特徵值及基本NumPy。本章處理經典場與連續介質模型，不宣稱提供完整工業多物理軟體。

---

## 問題與直覺

多物理場模型不只是把數條方程放在同一份程式中。真正的耦合必須回答：

- 哪個場影響哪個場？
- 交換的是質量、動量、熱量，還是僅為經驗參數？
- 一個子系統失去的量，是否成為另一子系統得到的量？
- 耦合資料在同一時間層、舊時間層，或某次內迭代上取值？
- 不同子系統的時間尺度相差多少？
- 內迭代停止時，留下的是耦合殘差還是物理誤差？

例如兩個水體區塊交換溶質。若區塊一流出質量的速率為$J$，區塊二便應以相反符號接收$J$。若兩邊都把$J$寫成正損失，總量會憑空消失；若兩邊都寫成正來源，總量會憑空增加。符號錯誤可能產生平滑曲線，因此「看起來合理」不是守恆證據。

耦合方式可先按方向分類。

- **單向耦合**：場$u$影響$v$，但$v$不回饋$u$。例如先給定速度，再求溫度傳輸。
- **雙向耦合**：$u$影響$v$，而$v$又改變$u$。例如溫度改變密度，密度再影響浮力流動。
- **弱耦合**：每一時間步只交換一次或少數幾次資料，各子系統看到的另一場可能是滯後值。
- **強耦合**：在同一時間步內反覆交換資料，直到耦合條件滿足指定容差。

「單向／雙向」描述物理依賴方向；「弱／強」描述數值協調程度，兩組概念不能混用。

---

## 數學與物理推導

### 兩個守恆子系統

令$M_1$與$M_2$為兩區塊內某守恆量，單位為kg。定義$J_{12}$為由區塊一流向區塊二的交換率，單位為$\mathrm{kg/s}$：

$$
\frac{dM_1}{dt}=S_1-J_{12},
\qquad
\frac{dM_2}{dt}=S_2+J_{12}.
$$

相加得到

$$
\frac{d}{dt}(M_1+M_2)=S_1+S_2.
$$

內部交換項精確抵消。若$S_1=S_2=0$，總量應保持常數。這是連續模型的守恆性；離散程式還必須讓兩式使用完全相同且符號相反的數值交換量。

若兩區塊體積為$V_1,V_2$，濃度為$C_i=M_i/V_i$，可採線性交換律

$$
J_{12}=K(C_1-C_2),
$$

其中$K$的單位為$\mathrm{m^3/s}$。濃度方程為

$$
V_1\frac{dC_1}{dt}=-K(C_1-C_2),
\qquad
V_2\frac{dC_2}{dt}=K(C_1-C_2).
$$

若體積不同，守恆量是$V_1C_1+V_2C_2$，不是$C_1+C_2$。

### 差模態與交換時間尺度

令$\delta=C_1-C_2$，則

$$
\frac{d\delta}{dt}
=
-K\left(\frac1{V_1}+\frac1{V_2}\right)\delta.
$$

因此

$$
\delta(t)=\delta(0)e^{-t/\tau_{\mathrm{ex}}},
$$

其中交換時間尺度為

$$
\tau_{\mathrm{ex}}
=
\frac{1}
{K(1/V_1+1/V_2)}.
$$

平均守恆模態保持不變，而差模態指數衰減。這種「一個零特徵值加上一個負特徵值」的結構，是交換系統的重要診斷。

場方程還可能含有其他時間尺度：

$$
\tau_{\mathrm{adv}}\sim\frac{L}{U},
\qquad
\tau_{\mathrm{diff}}\sim\frac{L^2}{D},
\qquad
\tau_{\mathrm{react}}\sim\frac1{k},
\qquad
\tau_{\mathrm{ex}}\sim\frac{V}{K}.
$$

最短時間尺度常限制顯式步長，但「小於最短尺度」並不自動保證所有離散性質。穩定性、非負性、守恆與收斂仍須分別檢查。

### 整體式與分區式求解

把未知量合併成$\mathbf z=[\mathbf u^T,\mathbf v^T]^T$，隱式一步可寫為

$$
\mathbf R(\mathbf z^{n+1};\mathbf z^n)=\mathbf0.
$$

**整體式方法**同時求解全部未知量，能直接處理強回饋，但矩陣較大，前置處理與軟體整合較困難。

**分區式方法**保留各子系統求解器。例如Gauss–Seidel型耦合迭代：

$$
\mathbf u^{(k+1)}
=
\mathcal S_u(\mathbf v^{(k)}),
$$

$$
\mathbf v^{(k+1)}
=
\mathcal S_v(\mathbf u^{(k+1)}).
$$

可定義尺度化耦合殘差

$$
r_{\mathrm c}^{(k)}
=
\max\left(
\frac{\|\mathbf u^{(k+1)}-\mathbf u^{(k)}\|}
{a_u+\|\mathbf u^{(k+1)}\|},
\frac{\|\mathbf v^{(k+1)}-\mathbf v^{(k)}\|}
{a_v+\|\mathbf v^{(k+1)}\|}
\right).
$$

其中$a_u,a_v>0$避免接近零時除零。此殘差只表示內迭代變化量，不等於離散方程真殘差，更不等於與真實世界之間的誤差。可靠實作應另外計算完整方程殘差$\mathbf R$。

### 固定點收斂與鬆弛

若合成迭代寫成

$$
\mathbf z^{(k+1)}=\mathcal G(\mathbf z^{(k)}),
$$

且在解附近滿足

$$
\|\mathcal G(\mathbf a)-\mathcal G(\mathbf b)\|
\le q\|\mathbf a-\mathbf b\|,
\qquad 0\le q<1,
$$

則局部固定點迭代收斂。可微情況下，常以Jacobian譜半徑

$$
\rho(J_{\mathcal G})<1
$$

作局部判據。若直接迭代振盪，可用欠鬆弛：

$$
\mathbf z^{(k+1)}
=
(1-\omega)\mathbf z^{(k)}
+\omega\mathcal G(\mathbf z^{(k)}),
\qquad 0<\omega\le1.
$$

欠鬆弛可能改善收斂，但不是萬能保證；過小$\omega$會使迭代極慢，強非線性或不良縮放仍可能失敗。

### 能量交換

若兩個熱子系統的熱容量為$H_1,H_2$，溫度為$T_1,T_2$，交換熱率為

$$
Q_{12}=G(T_1-T_2),
$$

其中$G$單位為$\mathrm{W/K}$，則

$$
H_1\frac{dT_1}{dt}=-Q_{12},
\qquad
H_2\frac{dT_2}{dt}=Q_{12}.
$$

總顯熱$H_1T_1+H_2T_2$守恆。定義

$$
E_\Delta=\frac12(T_1-T_2)^2,
$$

可得

$$
\frac{dE_\Delta}{dt}
=
-G\left(\frac1{H_1}+\frac1{H_2}\right)(T_1-T_2)^2
\le0.
$$

這是連續差異能量下降。它不表示任意時間步長的顯式離散都能量下降，也不表示每個溫度必定非負。

---

## 逐步手算例題

### 例一：兩區塊質量交換

令$V_1=V_2=1\,\mathrm{m^3}$、$K=0.1\,\mathrm{m^3/s}$，初值為

$$
C_1(0)=2\,\mathrm{kg/m^3},
\qquad
C_2(0)=0.
$$

總質量為

$$
M_{\mathrm{tot}}=V_1C_1+V_2C_2=2\,\mathrm{kg}.
$$

守恆平均濃度為$1\,\mathrm{kg/m^3}$。差值滿足

$$
\delta'= -0.2\delta,
\qquad
\delta(0)=2,
$$

故

$$
\delta(t)=2e^{-0.2t}.
$$

因此

$$
C_1(t)=1+e^{-0.2t},
\qquad
C_2(t)=1-e^{-0.2t}.
$$

兩者趨向同一平衡值，但總質量始終為$2\,\mathrm{kg}$。平衡不是質量消失，而是差模態衰減。

### 例二：一步顯式交換與故障步長

對同一系統採共享通量的顯式Euler：

$$
J^n=K(C_1^n-C_2^n),
$$

$$
C_1^{n+1}=C_1^n-\frac{\Delta t}{V_1}J^n,
\qquad
C_2^{n+1}=C_2^n+\frac{\Delta t}{V_2}J^n.
$$

取$\Delta t=1\,\mathrm s$，則$J^0=0.2\,\mathrm{kg/s}$，所以

$$
C_1^1=1.8,
\qquad
C_2^1=0.2.
$$

總質量仍為$2\,\mathrm{kg}$，且兩值非負。

若取$\Delta t=20\,\mathrm s$，則

$$
C_1^1=-2,
\qquad
C_2^1=4.
$$

總質量仍精確守恆，但非負性失敗，差值也翻轉並放大。這證明守恆不代表穩定，也不代表物理可信。事後把負值裁成零會改變總質量，不能用來掩蓋超大步長。

---

## 實作與程式

以下程式只使用Python 3.10+與NumPy。它比較共享通量顯式法與Backward Euler，並執行輸入驗證、總量診斷及步長細化。結果須視為未執行前的預期。

```python
import numpy as np

def validate(c0, volumes, K, dt, steps):
    c0 = np.asarray(c0, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if c0.shape != (2,) or volumes.shape != (2,):
        raise ValueError("c0與volumes必須為形狀(2,)")
    if not np.all(np.isfinite(c0)) or not np.all(np.isfinite(volumes)):
        raise ValueError("輸入必須有限")
    if np.any(volumes <= 0.0):
        raise ValueError("體積必須為正")
    if not np.isfinite(K) or K < 0.0:
        raise ValueError("K必須為有限非負值")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt必須為有限正值")
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps必須為非負整數")
    return c0.copy(), volumes

def total_mass(c, volumes):
    return float(np.dot(c, volumes))

def explicit_step(c, volumes, K, dt):
    flux = K * (c[0] - c[1])
    mass = volumes * c
    mass[0] -= dt * flux
    mass[1] += dt * flux
    return mass / volumes

def backward_euler_step(c, volumes, K, dt):
    v1, v2 = volumes
    A = np.array([
        [1.0 + dt*K/v1, -dt*K/v1],
        [-dt*K/v2, 1.0 + dt*K/v2]
    ])
    return np.linalg.solve(A, c)

def run(method, c0, volumes, K, dt, steps):
    c, volumes = validate(c0, volumes, K, dt, steps)
    m0 = total_mass(c, volumes)
    for _ in range(steps):
        if method == "explicit":
            c = explicit_step(c, volumes, K, dt)
        elif method == "backward_euler":
            c = backward_euler_step(c, volumes, K, dt)
        else:
            raise ValueError("未知方法")
    return {
        "concentration": c,
        "mass": total_mass(c, volumes),
        "mass_error": total_mass(c, volumes) - m0,
        "minimum": float(np.min(c))
    }

def exact_solution(t, c0, volumes, K):
    c0, volumes = validate(c0, volumes, K, 1.0, 0)
    v1, v2 = volumes
    mass = np.dot(c0, volumes)
    equilibrium = mass / (v1 + v2)
    delta0 = c0[0] - c0[1]
    rate = K * (1.0/v1 + 1.0/v2)
    delta = delta0 * np.exp(-rate*t)
    c1 = equilibrium + v2*delta/(v1 + v2)
    c2 = equilibrium - v1*delta/(v1 + v2)
    return np.array([c1, c2])

if __name__ == "__main__":
    c0 = np.array([2.0, 0.0])       # kg/m^3
    volumes = np.array([1.0, 1.0])  # m^3
    K = 0.1                         # m^3/s
    final_time = 10.0               # s

    for dt in (1.0, 0.5, 0.25):
        steps = int(round(final_time / dt))
        out = run("explicit", c0, volumes, K, dt, steps)
        ref = exact_solution(final_time, c0, volumes, K)
        error = np.linalg.norm(out["concentration"] - ref, ord=np.inf)
        print(dt, out, "error=", error)

    print(run("backward_euler", c0, volumes, K, 20.0, 1))
```

程式以質量更新後再除以體積，使同一個交換量在兩側以相反符號出現。若分別計算兩個略有不同的通量，即使差異只來自迭代容差，也會產生總量漂移。

---

## 測試與預期結果

### 正常測試

對$V_1=V_2=1$、$K=0.1$、$\Delta t=1$，第一步顯式結果預期為$[1.8,0.2]^T$，總質量誤差只應處於浮點捨入尺度。

### 步長細化

固定終止時間$t=10\,\mathrm s$，依序使用$\Delta t=1,0.5,0.25$。顯式Euler對光滑線性交換問題預期呈一階時間收斂，但未實際執行前不得捏造誤差值或觀測階。應以

$$
p_{\mathrm{obs}}
=
\log_2\frac{E(\Delta t)}{E(\Delta t/2)}
$$

計算觀測階，並確認仍位於漸近區。

### 非負性故障

顯式交換對單步凸組合要求

$$
\frac{\Delta t K}{V_1}\le1,
\qquad
\frac{\Delta t K}{V_2}\le1.
$$

超限時可能出現負濃度。不得使用`clip`後再宣稱方法非負；若裁切用於應急輸出，必須另記錄造成的質量改變。

### 守恆故障注入

若錯把第二式也寫成`mass[1] -= dt*flux`，總質量會每步減少$2\Delta t J$。此測試應失敗，證明總量檢查能抓到符號錯誤。

### 輸入拒絕

零或負體積、負交換係數、非有限濃度、非正時間步及未知方法都應引發`ValueError`。這些屬資料契約錯誤，不應繼續計算並產生看似正常的圖。

### 性質分離

- **守恆**：檢查$V_1C_1+V_2C_2$。
- **穩定性**：檢查擾動是否受控。
- **非負性**：檢查每個濃度是否小於零。
- **能量下降**：檢查差異能量是否下降。
- **收斂**：以步長細化比較解析解或高精度參考解。

其中任何一項通過，都不能替代其餘項目。

---

## 除錯與常見陷阱

1. **把單向耦合稱為弱耦合。**前者描述物理依賴，後者描述數值交換策略。
2. **兩側各自重算通量。**共享介面量應由單一來源產生，或經明確一致化。
3. **用濃度和代替質量。**體積不同時必須檢查$\sum_iV_iC_i$。
4. **只看內迭代增量。**固定點增量小不保證完整方程殘差小。
5. **把耦合殘差當成物理誤差。**模型假設錯誤時，內迭代完全收斂仍可能不符合現實。
6. **內迭代容差固定得過鬆。**時間步縮小後，耦合誤差可能主導，破壞預期時間階。
7. **盲目欠鬆弛。**它可能掩蓋縮放、符號或Jacobian錯誤。
8. **只檢查總量。**守恆方法仍可能不穩定、產生負值或嚴重相位誤差。
9. **用裁零恢復非負。**裁零會改變質量及方程，必須記錄而不能偷偷使用。
10. **忽略單位。**交換係數$K$為$\mathrm{m^3/s}$；若誤填成$\mathrm{1/s}$，方程量綱不成立。

---

## 養殖與相場案例

### 合成池域中的熱與溶氧單向耦合

可令給定溫度場影響合成耗氧率：

$$
R_C(T,C)=-k(T)C,
$$

其中$C$為$\mathrm{kg/m^3}$，$k$為$\mathrm{1/s}$。若不讓$C$回饋溫度方程，這是單向耦合。若進一步把生化放熱加入熱方程，便成為雙向模型，但必須另列反應焓、單位與適用假設。任意加入回饋係數不等於提高物理可信度。

本章不提供現場管理閾值，也不允許agent依模擬自行控制投餌、加藥或曝氣。溶氧跨越管理閾值不是物理相變。

### 相場與溫度的雙向耦合

固液相變模型可能令自由能依賴溫度：

$$
F[\phi,T]
=
\int_\Omega
\left[
W(\phi,T)+\frac{\kappa}{2}|\nabla\phi|^2
\right]d\mathbf x,
$$

而相場變化又透過潛熱項回饋熱方程。此時能量交換必須在兩條方程中以相容符號出現。若只把$T$代入$W$而忽略潛熱回饋，則是單向近似，不應宣稱總熱能守恆。

連續總能量守恆或耗散，也不保證任意分區時間離散具有相同性質。時間層滯後、非線性容差與介面投影都可能產生額外能量誤差。

---

## 習題

1. **手算題**  
   對$V_1=2\,\mathrm{m^3}$、$V_2=1\,\mathrm{m^3}$、$K=0.3\,\mathrm{m^3/s}$，求交換時間尺度。若初值為$C_1=3$、$C_2=0\,\mathrm{kg/m^3}$，求平衡濃度。

2. **程式題**  
   修改程式，使兩區塊加入等量反向外部來源$S_1=s$、$S_2=-s$。檢查總量是否保持不變，並以至少三個時間步比較誤差。

3. **反例題**  
   構造一個總量守恆但會產生負濃度的單步顯式例子，並說明為何守恆不能推出非負性。

4. **整合題**  
   某雙向分區算法的耦合殘差已低於$10^{-8}$，但網格細化後結果顯著改變。判斷哪些誤差已受控、哪些尚未受控，並提出至少四項診斷。

---

## 習題解答

### 第一題

交換衰減率為

$$
\lambda
=
K\left(\frac1{V_1}+\frac1{V_2}\right)
=
0.3\left(\frac12+1\right)
=
0.45\,\mathrm{1/s}.
$$

故

$$
\tau_{\mathrm{ex}}=\frac1{0.45}\approx2.22\,\mathrm s.
$$

初始總質量為

$$
M=2(3)+1(0)=6\,\mathrm{kg}.
$$

平衡濃度為

$$
C_{\mathrm{eq}}=\frac{6}{2+1}=2\,\mathrm{kg/m^3}.
$$

### 第二題

每步應對質量使用

$$
M_1^{n+1}=M_1^n+\Delta t(s-J^n),
$$

$$
M_2^{n+1}=M_2^n+\Delta t(-s+J^n).
$$

兩式相加後外部來源與內部交換均抵消，所以總量應僅有浮點尺度誤差。步長細化核對的是時間離散誤差；總量守恆本身不能給出時間收斂階。

### 第三題

取例二的$V_1=V_2=1$、$K=0.1$、$C^0=[2,0]^T$及$\Delta t=20$，得到

$$
C^1=[-2,4]^T.
$$

其總質量仍為$2\,\mathrm{kg}$，但第一區濃度為負。守恆只約束加權總和，不能約束各分量的符號。

### 第四題

低耦合殘差表示同一時間步內的分區固定點大致停止變化，但不表示空間離散誤差小。網格細化後結果顯著改變，顯示尚未進入空間收斂區，或網格、邊界、介面投影存在問題。

應至少：

1. 計算完整離散方程真殘差，而非只看迭代增量。
2. 分別細化空間網格與時間步。
3. 收緊耦合及代數求解容差，確認結果是否改變。
4. 檢查介面通量在兩側是否大小相等、符號相反。
5. 檢查守恆量、非負性與能量收支。
6. 使用製造解或解析簡化問題核對實作。

即使以上verification全部通過，物理模型誤差與現場validation仍是另一層問題。

---

## 本章小結

多物理場耦合的核心不是把多個求解器串接，而是明確定義依賴方向、交換量、符號、單位、時間層及收斂準則。單向與雙向描述物理回饋；弱耦合與強耦合描述數值協調程度。整體式方法直接處理聯立系統，分區式方法則以資料交換及固定點迭代協調既有求解器。

對兩個交換子系統，共享通量以相反符號加入兩側，才能保證內部交換不改變總量。守恆、穩定、非負、能量下降與收斂是不同性質，必須分別證明或測試。步長細化診斷時間離散誤差；固定點殘差診斷耦合迭代；完整方程殘差診斷代數求解；它們都不能直接量測物理模型與現實之差。

---

## 參考來源

1. FiPy，有限體積離散與邊界：https://pages.nist.gov/fipy/en/latest/numerical/discret.html  
   供守恆通量與分區離散背景參考。
2. FEniCSx，Poisson與弱形式：https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html  
   供後續弱形式與整體式組裝概念參考。
3. PETSc，KSP線性求解器：https://petsc.org/release/manual/ksp/  
   供耦合線性系統、殘差與求解診斷參考。
4. SciPy，稀疏線性代數API：https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html  
   本章核心程式不依賴SciPy。
5. FiPy，簡單相場與固液相變示例：https://pages.nist.gov/fipy/en/latest/generated/examples.phase.simple.html  
   來源中的相場慣例與本卷未必相同，參數不可直接抄用。