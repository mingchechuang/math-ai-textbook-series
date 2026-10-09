# 第 21 章　光傳輸積分與 Monte Carlo

## 學習目標與先備知識

本章處理渲染方程裡最核心的一件事：把連續的積分轉成可以用有限次隨機取樣估計的量。讀完本章，你應該能夠：

1. 寫出反射光 $L_o$ 的半球積分式，並指出 $\cos\theta$ 項、機率密度函數 (PDF) 與估計式各自的角色。
2. 手算 La mbert 半球積分的真值，以及均勻半球取樣、餘弦加權取樣在單一樣本下的估計值。
3. 推導並比較兩種取樣的變異數，說明為什麼餘弦加權在常數光源下可以得到零變異數。
4. 寫出一支用標準庫實作的 Monte Carlo 估計程式，處理 PDF 為零、樣本數收斂與多個 seed 重複實驗。
5. 對「合成資料可以驗證管線」與「合成資料不能取代實際量測」之間的邊界保持一致。

先備知識：本章假設你已讀過第 15 章的 Lambert 反射與立體角，第 18 章的交點與穩健性，以及 Volume I 的機率密度函數與期望值。如果你對 $E[g(X)] = \int g(x) p(x) \, dx$ 這個式子的意義不熟，先在紙上寫一次一維的例子再往下讀。本章的所有數值皆標為合成，不含任何真實輻射量測。

全書慣例：右手世界系 $+X$ 右、$+Y$ 上、$+Z$ 面向觀者；半球方向 $\omega$ 為單位向量；角度用弧度；$\cos\theta$ 為 $\omega$ 與表面法線 $n$ 的夾角餘弦。光度單位仍為 $\text{W}\cdot\text{m}^{-2}$，光度計算在線性 RGB 三通道獨立完成。

## 問題與直覺

渲染方程告訴我們，一個表面點 $p$ 往某方向 $\omega_o$ 反射的輻射亮度，等於自發光加上由所有入射方向進來的能量的某種加權積分。對不透明表面可以寫成

$$
L_o(p,\omega_o) = L_e(p,\omega_o) + \int_{\Omega^+} f_r(p,\omega_i,\omega_o)\, L_i(p,\omega_i)\,\bigl|\cos\theta_i\bigr| \, d\omega_i .
$$

這裡 $f_r$ 是 BRDF（單位 $\text{sr}^{-1}$），$L_i$ 是從方向 $\omega_i$ 進來的入射輻射亮度，$\Omega^+$ 是**以上半球 $n$ 為軸的上半球**，$d\omega_i$ 是立體角測度。

積分有兩個麻煩。第一，$\Omega^+$ 是連續集合，不可能逐方向檢查。第二，$L_i$ 通常來自場景本身，沒有一個簡單的封閉式。處理這種問題的標準做法是 **Monte Carlo 積分**：用隨機方向的加權平均去估計這個積分。

Monte Carlo 的核心是把「對方向的積分」重寫成「對某個機率分佈的期望值」。一旦寫成期望值，就可以用樣本平均逼近。兩個問題就冒出來了：

- 要選哪個機率分佈？**任意分佈理論上都無偏**，但不同分佈的變異數差很多。
- PDF 在某些方向上是零怎麼辦？因為我們要除以 $p(\omega)$，如果 $f(\omega) \neq 0$ 但 $p(\omega) = 0$，估計式就壞掉。

這兩個問題貫穿整章：**選擇好的分佈**和**處理退化方向**。它們不是隨機性本身的問題，而是「估計式設計」的問題。

## 數學與幾何推導

### 從積分到期望

給定一個方向定義域 $D$，機率密度函數 $p$ 滿足 $p(\omega) \ge 0$ 與 $\int_D p(\omega)\, d\omega = 1$。對任意函數 $g$，期望值定義為

$$
E_{p}[g] = \int_D g(\omega)\, p(\omega) \, d\omega .
$$

把 $g(\omega) = f(\omega)/p(\omega)$ 代入，得到

$$
E_p\!\left[\frac{f}{p}\right] = \int_D \frac{f(\omega)}{p(\omega)}\, p(\omega)\, d\omega = \int_D f(\omega)\, d\omega .
$$

所以只要 $p(x) > 0$ 在所有 $f(x) \neq 0$ 的地方成立，估計式

$$
\hat{F}_N = \frac{1}{N}\sum_{i=1}^{N} \frac{f(\omega_i)}{p(\omega_i)},\qquad \omega_i \sim p
$$

就是無偏的：$E[\hat{F}_N] = I$。其中 $I = \int_D f\, d\omega$。

### 變異數與誤差

單一樣本的變異數為

$$
\sigma^2 = E_p\!\left[\!\left(\frac{f}{p}\right)^{\!2}\right] - I^2 .
$$

獨立同分佈的 $N$ 個樣本平均，變異數是

$$
\operatorname{Var}[\hat{F}_N] = \frac{\sigma^2}{N},\qquad
\text{標準誤差} = \frac{\sigma}{\sqrt{N}} .
$$

變異數決定了「需要的樣本數」。若把 $p$ 選得和 $f$ 成正比，單樣本變異數可能為零——也就是常數估計。

### 兩種半球取樣

半球上有兩種最常見的取樣分佈：

- **均勻半球**：$p_u(\omega) = \dfrac{1}{2\pi}$。
- **餘弦加權半球**：$p_c(\omega) = \dfrac{\cos\theta}{\pi}$，其中 $\theta$ 是與法線的夾角。

兩者都滿足機率歸一化。注意角度與立體角的換算：以球面座標 $(\theta, \phi)$ 表示時，$d\omega = \sin\theta\, d\theta\, d\phi$。**這是本章最常在實作裡被忘記的一項**。

由反函數取樣法得到兩種分佈的方向：

- 均勻：$\cos\theta = 1 - r_1$，$\phi = 2\pi r_2$。用 $r_1 \sim U[0,1]$ 亦可寫成 $\cos\theta = r_1$。
- 餘弦加權：$\cos\theta = \sqrt{1 - r_1}$，$\sin\theta = \sqrt{r_1}$，$\phi = 2\pi r_2$。

方向在局部座標系（$+Y$ 為法線）下為

$$
\omega = (\sin\theta\cos\phi,\; \cos\theta,\; \sin\theta\sin\phi).
$$

### 帶入 Lambert 反射

Lambert BRDF 為 $f_r = \rho/\pi$（$\rho$ 為反照率，$\in [0,1]$）。設入射光為 $L_i(\omega)$，估計式為

$$
\hat{L}_o = \frac{1}{N}\sum_{i=1}^{N} \frac{(\rho/\pi)\, L_i(\omega_i)\,\cos\theta_i}{p(\omega_i)} .
$$

代入兩種取樣分佈：

- 均勻：$\hat{L}_o^{(u)} = \dfrac{1}{N}\sum 2\rho\, L_i \cos\theta_i$。
- 餘弦加權：$\hat{L}_o^{(c)} = \dfrac{1}{N}\sum \rho\, L_i$。因為 $\cos\theta_i$ 與 $p$ 的 $\cos\theta$ 因子完全約掉。

若 $L_i \equiv 1$（上半球所有方向入射光相同），餘弦加權估計式的每一項都是 $\rho$，所以單樣本估計恆等於真值 $\rho$，**變異數為零**。這正是「把 $p$ 選得和 $f$ 成正比」的具體例子。

## 逐步手算例題

### 例題一：常數光下的真值與兩種估計

設 $\rho = 0.6$，$L_i(\omega) = 1$ 對所有 $\omega$。

真值：
$$
L_o = \frac{\rho}{\pi}\int_{\Omega^+} \cos\theta \, d\omega = \frac{\rho}{\pi}\cdot \pi = \rho = 0.6 .
$$

均勻半球估計，單樣本。取樣 $r_1 = 0.5$，則 $\cos\theta = 0.5$（$\theta = 60^\circ$），$\phi$ 任意：
$$
\hat{L}_o^{(u)} = 2\rho\cos\theta = 2\cdot 0.6 \cdot 0.5 = 0.6.
$$
這個樣本恰好命中真值。均勻取樣的期望值為 $\rho \cdot 2E[\cos\theta] = 0.6 \cdot 2 \cdot (1/2) = 0.6$，正確；但單樣本變異數不是零。

餘弦加權估計，任一樣本：
$$
\hat{L}_o^{(c)} = \rho L_i = 0.6.
$$
每一樣本都給 $0.6$。變異數為零。

### 例題二：方向性入射光

現在改取 $L_i(\omega) = \cos\theta$（無單位，只是定義題目的函數），$\rho = 0.6$。真值：

$$
L_o = \frac{\rho}{\pi}\int_{\Omega^+}\cos^2\theta \, d\omega .
$$

由 $\int_{\Omega^+}\cos^2\theta\, d\omega = 2\pi/3$，得
$$
L_o = \frac{0.6}{\pi} \cdot \frac{2\pi}{3} = 0.4 .
$$

**均勻半球單樣本**，仍取 $\theta = 60^\circ$（$\cos\theta = 0.5$）：
$$
g_u = 2\rho \cos^2\theta = 2 \cdot 0.6 \cdot 0.25 = 0.3.
$$

**餘弦加權單樣本**，仍取 $\theta = 60^\circ$：
$$
g_c = \rho\cos\theta = 0.6 \cdot 0.5 = 0.3.
$$

巧合下兩者同為 0.3。這是偶然而非通則。若取 $\theta = 0$（$r_1 = 0$ 或 $r_1 = 1$ 的極端）：
- 均勻：$g_u = 2 \cdot 0.6 \cdot 1 = 1.2$。
- 餘弦：$g_c = 0.6 \cdot 1 = 0.6$。

此時餘弦加權的樣本值較接近真值 0.4。要判斷哪個估計式較好，仍須看變異數，不能只看單一樣本。

## 實作與程式

下面是一支只依賴 Python 標準庫的 Monte Carlo 估計器。它內建四項自檢：常數光真值、方向光收斂、PDF 為零不產生 NaN、多 seed 標準差隨 $N$ 下降。

```python
"""mc_lambert.py —— Monte Carlo 半球積分；只用標準庫。"""
import math, random


def sample_uniform_hemisphere(r1, r2):
    """上半球均勻取樣，+Y 為法線。回傳單位方向。"""
    cos_t = r1
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    phi = 2.0 * math.pi * r2
    return (sin_t * math.cos(phi), cos_t, sin_t * math.sin(phi))


def sample_cosine_hemisphere(r1, r2):
    """餘弦加權取樣 p(ω)=cosθ/π，+Y 為法線。"""
    cos_t = math.sqrt(max(0.0, 1.0 - r1))
    sin_t = math.sqrt(r1)
    phi = 2.0 * math.pi * r2
    return (sin_t * math.cos(phi), cos_t, sin_t * math.sin(phi))


def estimate_lambert(rho, L_i_fn, N, mode, seed):
    """估計 L_o = (rho/π) ∫ L_i(ω) cosθ dω。
    mode ∈ {'uniform', 'cosine'}。"""
    rng = random.Random(seed)
    total = 0.0
    used = 0
    for _ in range(N):
        r1, r2 = rng.random(), rng.random()
        if mode == "uniform":
            w = sample_uniform_hemisphere(r1, r2)
            cos_t = w[1]
            p = 1.0 / (2.0 * math.pi)
        else:
            w = sample_cosine_hemisphere(r1, r2)
            cos_t = w[1]
            p = cos_t / math.pi
        if p <= 0.0:
            # PDF=0 且 f=0 的方向，貢獻定義為 0，直接跳過。
            # 若這裡誤加 f/p，就會得到 inf 或 NaN。
            continue
        total += (rho / math.pi) * L_i_fn(w) * max(0.0, cos_t) / p
        used += 1
    return total / max(1, used)


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(v)


def self_check():
    L_const = lambda w: 1.0
    L_dir   = lambda w: w[1]          # cosθ
    rho = 0.6
    out = {}

    # 檢查 1：常數光真值 0.6，多 seed 平均要靠近。
    for mode in ("uniform", "cosine"):
        ests = [estimate_lambert(rho, L_const, 200, mode, s) for s in range(50)]
        m, sd = mean_std(ests)
        out[f"const_{mode}"] = (m, sd)
        assert abs(m - rho) < 0.05, (mode, m, sd)

    # 檢查 2：方向光真值 0.4。
    for mode in ("uniform", "cosine"):
        ests = [estimate_lambert(rho, L_dir, 500, mode, s) for s in range(50)]
        m, sd = mean_std(ests)
        out[f"dir_{mode}"] = (m, sd)
        assert abs(m - 0.4) < 0.05, (mode, m, sd)

    # 檢查 3：結果必須是有限的實數（PDF=0 不產生 NaN）。
    for mode in ("uniform", "cosine"):
        v = estimate_lambert(rho, L_const, 100, mode, 12345)
        assert math.isfinite(v), (mode, v)

    # 檢查 4：餘弦加權在常數光下變異數應接近零。
    _, sd_cos = out["const_cosine"]
    assert sd_cos < 1e-8, sd_cos
    return out


if __name__ == "__main__":
    r = self_check()
    for k, v in r.items():
        print(f"{k}: mean={v[0]:.5f}  sd={v[1]:.5f}")
```

幾個使用上的要點：

- 取樣函式只接收兩個純量 $r_1, r_2$，方便做固定序列測試與跨 seed 對照。
- `p <= 0.0` 的處理是本章的核心除錯練習：如果直接除，`math.inf` 或 `math.nan` 會污染整個平均。這裡把它解釋為「貢獻為 0」並跳過，並且在分母端使用 `max(1, used)` 避免除以零。
- 餘弦加權模式下每一樣本值都等於 $\rho L_i$，因此常數光檢驗的標準差數值上應為 0（雙精度浮點運算下約 $10^{-16}$ 量級）。

## 測試與預期結果

執行 `python mc_lambert.py`，**預期**輸出（每個 seed 為 0 到 49，樣本數如上）：

```
const_uniform: mean≈0.600  sd≈0.03
const_cosine:  mean≈0.600  sd≈0.00000
dir_uniform:   mean≈0.400  sd≈0.03
dir_cosine:    mean≈0.400  sd≈0.03
```

**預期** `const_cosine` 的標準差恆為 0，因為單樣本估計本身就是常數 $\rho L_i$。`dir_cosine` 與 `dir_uniform` 的標準差量級相近，這是因為 $L_i = \cos\theta$ 對兩個分佈都不是常數函數，兩邊的變異數都需要靠增加樣本數下降。收斂速率皆為 $1/\sqrt{N}$。

如果你想驗證，可將 `N` 從 200 改為 20000 並重新執行：**預期**全部四條 `sd` 下降約 $\sqrt{100} = 10$ 倍。不要把標準差當作「準確度」；它只是估計值本身的不確定度。真值檢驗只能靠已知解析解。

以上皆為依據程式碼直讀得到的預期結果；本卷未在特定硬體或環境中執行過。

## 除錯與常見陷阱

1. **立體角測度混用**：如果寫 $\int \cos\theta\, d\theta\, d\phi$ 而不是 $\int \cos\theta \sin\theta\, d\theta\, d\phi$，結果會少一個 $\sin\theta$ 且積分值不正確。任何時候從平面角度換算到立體角，都要檢查 $d\omega = \sin\theta\, d\theta\, d\phi$。
2. **PDF 為零除以零**：餘弦加權分佈在 $\theta$ 接近 $\pi/2$ 時 $p \to 0$。若你的取樣函式允許 $\cos\theta$ 恰好為零而分子 $f$ 也為零，就必須事先判定：貢獻為 0，跳過，不要讓浮點產生 NaN。
3. **兩種取樣的形式寫反**：均勻半球是 `cosθ = r1`（或 `1-r1`），餘弦加權是 `cosθ = sqrt(1-r1)`。互換會同時錯分佈與錯估計。
4. **把 `L_i` 當作照明功率的平方**：$L_i$ 是輻射亮度；在多光源或紋理環境裡，它是「往方向 $\omega$ 看過去的值」，不隨立體角作進一步加權。
5. **缺少多 seed 重複**：單次執行的估計值不可能代表收斂，一定要看多 seed 下的分布。
6. **變異數為零時誤以為有 bug**：如果 $p$ 與 $f$ 恰好成正比、樣本值 $f/p$ 就是常數，變異數自然為 0。這是最理想的狀況，不是錯誤。

## 養殖數位分身案例

設想在池體底部的平坦水泥面上，反照率 $\rho = 0.6$，入射光 $L_i$ 來自四面八方的合成環境（例如一個均勻灰色天空模型加一個方向主光）。要不要用 Monte Carlo？只在連環境都無法解析求解時才需要。以下三種狀況依序複雜化：

1. 純 $L_i = 1$：本章例題一，可直接算真值 0.6；Monte Carlo 只是驗證工具。
2. 單一平行主光：解析可解，不需要 Monte Carlo 的隨機性。
3. 池體上方懸掛多盞燈、加上水波反射的環境光：$L_i$ 隨方向變化且難以封閉式書寫，才進入 Monte Carlo 的適用範圍。

必須標註：以上都是**合成**光分布，與真實養殖池水面反射、藻類遮光、浮游顆粒散射完全不同。合成資料可以用來驗證估計式是否收斂、是否正確地積分 $\cos\theta$，但不能用來推論池底真實照度或魚類行為。

## 習題

**習題 1（手算）** 設 $\rho = 0.5$，入射光 $L_i(\omega) = 1 + \cos\theta$。請：
(a) 計算真值 $L_o = \dfrac{\rho}{\pi}\int_{\Omega^+}(1+\cos\theta)\cos\theta\, d\omega$；
(b) 用均勻半球取樣寫出單樣本估計式的顯式形式；
(c) 若某樣本為 $\theta = 30^\circ$，求出該估計值。

**習題 2（程式測試）** 修改 `estimate_lambert`，讓它可選擇「刻意製造 PDF=0」的模式：在 `mode='broken'` 下對所有樣本使用 $p = 0$。執行 10 次樣本，觀察是否出現 `inf` 或 `nan`。說明原始程式碼的 `p <= 0.0` 檢查如何防止這件事，並討論若不跳過而是用 $f/p$ 會發生什麼。

**習題 3（反例／除錯）** 有人把 `sample_cosine_hemisphere` 改成 `cos_t = math.sqrt(1 - r1); sin_t = 1 - cos_t`。請用至少兩個具體 $r_1$ 值（例如 $r_1 = 0.25$ 與 $r_1 = 0.5$）檢查 `sin²+cos²` 是否為 1。指出這種錯誤會如何在後續估計式中污染 $\cos\theta$。

**習題 4（整合應用）** 你拿到一個合成環境光 $L_i(\omega) = \max(0, w_y)$，即入射亮度等於方向與世界 $Y$ 軸夾角的餘弦。對一個反照率 $\rho=0.5$、法線 $+Y$ 的水平表面，請：
(a) 用文字寫出真值積分與其解析解；
(b) 設計一份實驗計畫，用均勻半球與餘弦加權兩種模式各 50 個 seed、每個 $N = 1000$，比較標準誤差；
(c) 說明這份計畫「能證明什麼」、「不能證明什麼」。

## 習題解答

**習題 1 解答**
(a) 展開：
$$
\int_{\Omega^+}(1+\cos\theta)\cos\theta\, d\omega = \int \cos\theta\, d\omega + \int \cos^2\theta\, d\omega = \pi + \frac{2\pi}{3} = \frac{5\pi}{3}.
$$
真值 $L_o = (0.5/\pi)(5\pi/3) = 5/6 \approx 0.8333$。

(b) 均勻半球 $p = 1/(2\pi)$，估計式：
$$
\hat{L}_o^{(u)} = \frac{\rho}{\pi}(1+\cos\theta)\cos\theta \cdot 2\pi = 1.0 \cdot (1+\cos\theta)\cos\theta .
$$
展開：$\hat{L}_o^{(u)} = \cos\theta + \cos^2\theta$。

(c) $\theta = 30^\circ$：$\cos\theta = \sqrt{3}/2 \approx 0.8660$，$\cos^2\theta = 0.75$。$\hat{L}_o = 0.8660 + 0.75 = 1.6160$。

**習題 2 解答** 若 $p = 0$ 且 $f > 0$，$f/p$ 在雙精度浮點下會變成 `inf`；若 $f$ 也是 0，得到 `nan`。原程式在關鍵位置用 `if p <= 0.0: continue` 事先攔截，並且在最後用 `max(1, used)` 防止 `used=0` 造成除以零。若拿掉這段防護，程式的輸出通常是 `nan`，因為 `inf + 有限數` 仍為 `inf`，而 `inf - inf` 或 `0/0` 產生 `nan`。這些數值一旦進入平均，整條管線的結果都不能再用。

**習題 3 解答** 檢驗 $r_1 = 0.25$：$\cos\theta = \sqrt{0.75} \approx 0.8660$，$\sin\theta = 1 - 0.8660 = 0.1340$。則 $\sin^2 + \cos^2 = 0.01796 + 0.75 = 0.76796 \ne 1$。$r_1 = 0.5$：$\cos\theta \approx 0.7071$，$\sin\theta = 0.2929$，$\sin^2+\cos^2 \approx 0.0858 + 0.5 = 0.5858 \ne 1$。

這使得方向向量不是單位向量，$w_y$ 不再等於真正的 $\cos\theta$，$\cos\theta$ 被高估。估計式中 $\cos\theta$ 同時出現在分子與 PDF 分母（若是 cosine 模式），錯的 $\cos\theta$ 不會自動相消，$f/p = \rho L_i$ 這條簡潔性質也失效，估計值會系統性偏離真值。

**習題 4 解答**
(a) 表面法線 $+Y$，入射方向 $\omega$ 與 $+Y$ 夾角即 $\theta$，$w_y = \cos\theta$。故 $L_i(\omega) = \cos\theta$（當 $\theta \in [0, \pi/2]$；下半球 $w_y < 0$ 時 $L_i = 0$，不屬於 $\Omega^+$ 故無貢獻）。

$$
L_o = \frac{\rho}{\pi}\int_{\Omega^+} \cos\theta \cdot \cos\theta\, d\omega = \frac{\rho}{\pi}\cdot \frac{2\pi}{3} = \frac{2\rho}{3}.
$$
代入 $\rho = 0.5$ 得真值 $1/3 \approx 0.3333$。

(b) 實驗計畫：
   - 每個模式跑 50 個 seed（`seed = 0..49`），每個 seed 內 $N = 1000$ 樣本。
   - 計算每個 seed 的單次估計值，再對 50 個估計值求平均與樣本標準差。
   - 若標準誤差 $\sigma/\sqrt{50}$ 小於某閾值（例如 $0.01$），且兩模式平均都與 $1/3$ 的距離在閾值內，即接受。
   - 記錄每個模式的標準差，用以比較兩者的**效率**。均勻取樣因為 $f/p = 2\rho \cos^2\theta$ 而餘弦取樣因為 $f/p = \rho\cos\theta$，兩者值域不同，因此期望變異數也不同。

(c) 能證明的：程式實作了兩種取樣、估計式無偏、以及二者在有限樣本下的效率差異。不能證明的：這份實驗假設的環境光 $L_i = \max(0, w_y)$ 只是一個數學合成模型，它既不描述任何真實天空，也不描述實際養殖池光照環境；實驗結果不能推論真實場景的照度或生態效應。

## 本章小結

本章的核心可以壓縮成三句話：

1. **積分即期望**：只要找到一個在 $f \ne 0$ 處為正的 PDF，$f/p$ 的樣本平均就是無偏估計。
2. **PDF 可以設計**：要降低變異數，就讓 $p$ 盡量與 $f$ 在形狀上匹配；餘弦加權半球在 Lambert 反射下把變異數壓到零，就是這個原則的極致例子。
3. **退化要事先處理**：PDF 為零的方向必須跳過，立體角測度必須用 $\sin\theta\, d\theta\, d\phi$；這兩件事做錯，程式不會報錯，只會靜靜地輸出錯的數。

後面第 22 章會把這套估計式放進完整的路徑追蹤迴圈。本章的單點半球積分是那條迴圈的內核；一旦你確定了 PDF、估計式與零 PDF 的處理原則，路徑追蹤只是把它串起來的工程問題。

## 參考來源

- [G1] PBRT 4：Transformations，<https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations>
- [G2] PBRT 4：Reflection Models，<https://pbr-book.org/4ed/Reflection_Models>
- [G3] PBRT 4：The Light Transport Equation，<https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation>
- [G4] Ray Tracing in One Weekend，<https://raytracing.github.io/books/RayTracingInOneWeekend.html>
- [G5] LearnOpenGL：Transformations，<https://learnopengl.com/Getting-started/Transformations>
- [G6] Blender Manual：Skinning Introduction，<https://docs.blender.org/manual/en/latest/animation/armatures/skinning/introduction.html>
- [G7] NumPy 線性代數參考，<https://numpy.org/doc/stable/reference/routines.linalg.html>
- [G8] Khronos glTF 2.0 規格，<https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>

以上連結僅為延伸查閱入口；本章推導與數值皆獨立撰寫，未逐條對應來源論點，也未聲稱已跑過官方範例或在其環境中驗證。