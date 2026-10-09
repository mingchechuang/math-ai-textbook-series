# 第 22 章　路徑追蹤與重要性取樣

> 第四部｜光線追蹤與光傳輸

## 學習目標與先備知識

本章把上一章的 Monte Carlo 積分推進到完整影像生成。完成後，讀者應能：

1. 從渲染方程理解路徑追蹤的遞迴結構。
2. 定義並更新路徑權重 `throughput`。
3. 區分 BSDF 取樣與顯式光源取樣。
4. 用機率密度函數 PDF 正確加權樣本。
5. 以俄羅斯輪盤終止長路徑而不系統性壓低期望值。
6. 理解多重重要性取樣 MIS 如何結合兩種取樣策略。
7. 建立固定亂數種子、低解析度、固定樣本預算的 CPU 路徑追蹤器。

先備知識包括射線與球面求交、Lambert BRDF、半球立體角、隨機變數、PDF 與 Monte Carlo 估計。全章顏色皆為線性 RGB；$L$、$f_r$ 等 RGB 值是三個色彩通道的數值近似，不代表完整光譜模型。

---

## 問題與直覺

光線投射器只找出相機最先看見的表面；局部照明通常只計算表面直接接收的光。實際場景中，池壁會把光反射到魚體，魚腹也可能被池底的間接反射照亮。這類多次反射形成全域光傳輸。

路徑追蹤從相機發出射線。射線碰到表面後，依材質抽樣新方向，再繼續追蹤。每條路徑可能：

- 直接碰到發光表面；
- 經一次反射後碰到光源；
- 經多次反射後才抵達光源；
- 離開場景而取得環境光；
- 被俄羅斯輪盤提前終止。

若只靠隨機反射方向偶然撞到小光源，大多數樣本不會得到直接光，影像便有強烈雜訊。重要性取樣不改變原積分，而是把更多樣本放在貢獻可能較大的方向，再以 PDF 補償抽樣不均。

光源取樣擅長尋找小面積光源；BSDF 取樣擅長尋找材質偏好的方向。MIS 讓兩者共同工作，避免只選一種策略時出現極端變異數。

---

## 數學與幾何推導

### 1. 渲染方程

表面點 $\mathbf{x}$ 沿出射方向 $\omega_o$ 的輻射亮度為

$$
L_o(\mathbf{x},\omega_o)
=
L_e(\mathbf{x},\omega_o)
+
\int_{\mathcal{H}^2}
f_r(\mathbf{x},\omega_i,\omega_o)
L_i(\mathbf{x},\omega_i)
|\mathbf{n}\cdot\omega_i|
\,d\omega_i.
$$

其中：

- $L_o$：出射輻射亮度，單位可寫為 $\mathrm{W\,m^{-2}\,sr^{-1}}$；
- $L_e$：表面自身發光；
- $L_i$：從 $\omega_i$ 入射的輻射亮度；
- $f_r$：BRDF，單位為 $\mathrm{sr^{-1}}$；
- $\mathbf{n}$：表面單位法線；
- $\mathcal{H}^2$：法線上方半球；
- $d\omega_i$：微小立體角，單位為 sr。

若射線從 $\mathbf{x}$ 沿 $\omega_i$ 首次碰到另一表面 $\mathbf{x}'$，則

$$
L_i(\mathbf{x},\omega_i)
=
L_o(\mathbf{x}',-\omega_i).
$$

代回後，右側又出現另一個出射輻射亮度，形成遞迴結構。路徑追蹤以隨機樣本估計這個遞迴積分。

### 2. 單次 Monte Carlo 估計

若方向 $\omega_i$ 由 PDF $p(\omega_i)$ 抽樣，且在非零被積函數處滿足 $p(\omega_i)>0$，則

$$
\int_{\mathcal{H}^2}g(\omega)\,d\omega
=
\mathbb{E}\left[
\frac{g(\omega)}{p(\omega)}
\right].
$$

對渲染方程，

$$
g(\omega_i)
=
f_r L_i |\mathbf{n}\cdot\omega_i|.
$$

單樣本估計量為

$$
\widehat{L}_o
=
L_e+
\frac{
f_r(\omega_i,\omega_o)
L_i(\omega_i)
|\mathbf{n}\cdot\omega_i|
}{
p(\omega_i)
}.
$$

除以 PDF 不是額外增亮，而是補償抽樣分布不均。若忘記除以 PDF，演算法通常有偏；若 PDF 為零但貢獻非零，估計器則無法涵蓋該部分積分。

### 3. Lambert 材質與餘弦加權取樣

Lambert BRDF 為

$$
f_r=\frac{\boldsymbol{\rho}}{\pi},
$$

其中 $\boldsymbol{\rho}=(\rho_r,\rho_g,\rho_b)$ 是線性 RGB 反照率，各分量通常位於 $[0,1]$。

餘弦加權半球取樣的 PDF 是

$$
p_{\mathrm{bsdf}}(\omega_i)
=
\frac{\cos\theta}{\pi},
\qquad
\cos\theta=\max(0,\mathbf{n}\cdot\omega_i).
$$

代入路徑權重：

$$
\frac{f_r\cos\theta}{p_{\mathrm{bsdf}}}
=
\frac{(\boldsymbol{\rho}/\pi)\cos\theta}
{\cos\theta/\pi}
=
\boldsymbol{\rho}.
$$

因此純 Lambert 表面使用餘弦取樣時，每次反射只需把 throughput 乘上反照率。這個簡化不適用於任意 BRDF。

### 4. 路徑 throughput

令 $\boldsymbol{\beta}$ 表示從相機到目前頂點累積的路徑權重。起始時

$$
\boldsymbol{\beta}_0=(1,1,1).
$$

每次抽樣方向後更新

$$
\boldsymbol{\beta}_{k+1}
=
\boldsymbol{\beta}_k\odot
\frac{
f_r(\omega_{k+1},\omega_k)
|\mathbf{n}\cdot\omega_{k+1}|
}{
p(\omega_{k+1})
}.
$$

若目前頂點有發光 $\mathbf{L}_e$，對像素的貢獻為

$$
\mathbf{L}\leftarrow
\mathbf{L}+\boldsymbol{\beta}\odot\mathbf{L}_e,
$$

其中 $\odot$ 表示 RGB 逐分量相乘。

`throughput` 不是「尚未使用的光能」，而是目前路徑樣本在估計器中的乘法權重。其值可能大於 1，尤其在 PDF 很小或材質具有尖峰分布時；不能任意截斷，否則會引入偏差。

### 5. 顯式光源取樣

在表面點 $\mathbf{x}$ 直接抽樣光源上的點 $\mathbf{y}$，稱為 next-event estimation，簡稱 NEE。若面積光源以面積 PDF $p_A(\mathbf{y})$ 取樣，必須轉成立體角 PDF。

令

$$
\mathbf{d}=\mathbf{y}-\mathbf{x},
\qquad
r^2=\|\mathbf{d}\|^2,
\qquad
\omega_i=\frac{\mathbf{d}}{\|\mathbf{d}\|}.
$$

光源法線為 $\mathbf{n}_L$，則

$$
\cos\theta_L
=
\max(0,\mathbf{n}_L\cdot(-\omega_i)).
$$

面積 PDF 轉成立體角 PDF：

$$
p_\omega(\omega_i)
=
p_A(\mathbf{y})
\frac{r^2}{\cos\theta_L}.
$$

若均勻抽樣面積為 $A_L$ 的單一光源，

$$
p_A=\frac{1}{A_L},
\qquad
p_\omega=\frac{r^2}{A_L\cos\theta_L}.
$$

直接光單樣本估計為

$$
\widehat{\mathbf{L}}_{\mathrm{direct}}
=
\boldsymbol{\beta}\odot
\frac{
f_r\,\mathbf{L}_e\,
\cos\theta
}{
p_\omega
}
V(\mathbf{x},\mathbf{y}),
$$

其中 $V$ 是可見性：兩點間無遮擋時為 1，否則為 0。

### 6. 多光源的完整 PDF

若場景有 $M$ 個光源，通常先抽樣離散光源索引 $J$，再於該光源上抽樣位置或方向。假設選到第 $j$ 個光源的機率是 $P(J=j)$，在選定該光源後，方向的條件 PDF 是 $p(\omega\mid J=j)$，則完整 PDF 為

$$
p_L(\omega,j)
=
P(J=j)\,p(\omega\mid J=j).
$$

直接光估計器的分母與 MIS 權重都必須使用這個完整生成機率，而不能只使用條件 PDF。

最簡單的選法是均勻選光源：

$$
P(J=j)=\frac{1}{M}.
$$

但若光源亮度與面積差異很大，均勻選擇可能把太多樣本分給弱小光源。可依光源功率近似值 $q_j$ 設定

$$
P(J=j)=\frac{q_j}{\sum_{k=1}^{M}q_k}.
$$

只要在估計量中使用相同機率補償，改變光源選取分布不會改變理想期望值，卻會改變變異數。

若多個光源都能沿同一方向生成相同事件，方向的混合 PDF 應考慮所有可生成該方向的分量：

$$
p_L(\omega)
=
\sum_{j=1}^{M}
P(J=j)\,p(\omega\mid J=j).
$$

對互不重疊、由首次交點唯一辨識的有限面積光源，實作常可只評估命中光源所屬分量；但擴充到重疊分布或環境光取樣時，必須明確處理混合 PDF。

### 7. 俄羅斯輪盤

固定最大深度會截斷所有更長路徑，因此一般會引入偏差。俄羅斯輪盤則以機率延續路徑。

令存活機率為 $p_s\in(0,1]$。路徑：

- 以機率 $1-p_s$ 終止；
- 以機率 $p_s$ 繼續，並令

$$
\boldsymbol{\beta}\leftarrow
\frac{\boldsymbol{\beta}}{p_s}.
$$

因為

$$
p_s\frac{\boldsymbol{\beta}}{p_s}
+(1-p_s)\mathbf{0}
=\boldsymbol{\beta},
$$

所以期望值不變。實作上通常先追蹤數次反射，再依 $\boldsymbol{\beta}$ 的最大通道設定存活機率，例如

$$
p_s=\min(0.95,\max(\beta_r,\beta_g,\beta_b)).
$$

存活機率不能設為零後仍嘗試除法，也不宜讓極低貢獻路徑總是存活。另一方面，太早啟用輪盤可能增加變異數：許多仍有明顯貢獻的路徑會被丟棄，少數存活樣本則取得很大權重。

### 8. 最大深度與俄羅斯輪盤的差別

最大深度與俄羅斯輪盤都能限制計算量，但數學性質不同。

若最大深度是 $D$，演算法完全忽略第 $D$ 次交點以後的貢獻。只要那些長路徑的真實積分不為零，結果便存在截斷偏差。提高 $D$ 通常會減少此偏差，但增加工作量。

俄羅斯輪盤不是按固定長度刪除所有路徑，而是讓每條路徑以非零機率繼續，並補償存活樣本。只要：

1. 存活決策使用的機率已知；
2. 存活後除以相同機率；
3. 任何仍可能有非零貢獻的路徑都有繼續的可能；

則輪盤本身不改變期望值。

實作仍常同時使用兩者：輪盤是主要統計終止機制，最大深度則是固定預算與防止程式錯誤的上限。本章 `max_depth=8` 到達上限時直接回傳目前累積值，因此可能保留截斷偏差；它不能被描述成完全無偏的無限深度積分器。

### 9. MIS 入門

同一直接光路徑可由兩種策略找到：

1. 從表面抽樣光源；
2. 從 BSDF 抽樣方向並碰到光源。

若兩者都加入完整貢獻，會重複計算。MIS 為各策略分配權重。單一光源樣本與單一 BSDF 樣本下，常用 power heuristic：

$$
w_a=
\frac{p_a^2}{p_a^2+p_b^2},
\qquad
w_b=
\frac{p_b^2}{p_a^2+p_b^2}.
$$

因此

$$
w_a+w_b=1.
$$

當 $p_a\gg p_b$ 時，$w_a$ 接近 1；當兩者相等時，各得 $1/2$。PDF 必須使用相同測度，例如都以 $\mathrm{sr}^{-1}$ 表示，不能把面積 PDF 直接與立體角 PDF 比較。

### 10. MIS 估計器成立條件

令欲積分函數為 $g(x)$，兩種策略 PDF 為 $p_a(x)$、$p_b(x)$。若各取一個獨立樣本，MIS 估計器可寫成

$$
\widehat{I}
=
w_a(X_a)\frac{g(X_a)}{p_a(X_a)}
+
w_b(X_b)\frac{g(X_b)}{p_b(X_b)}.
$$

要維持正確期望，權重至少應滿足：

$$
w_a(x)+w_b(x)=1
$$

於有貢獻且可被策略生成的區域成立；若某策略在 $x$ 處的 PDF 為零，其權重應為零。所有有非零貢獻的區域也必須被至少一種策略覆蓋。

若策略 $a$ 取 $n_a$ 個樣本、策略 $b$ 取 $n_b$ 個樣本，power heuristic 應納入樣本數：

$$
w_a(x)
=
\frac{(n_a p_a(x))^2}
{(n_a p_a(x))^2+(n_b p_b(x))^2}.
$$

本章程式每個表面頂點各取一個光源樣本與一個 BSDF 樣本，所以 $n_a=n_b=1$，公式才化為只含 PDF 的形式。若日後把光源樣本增為四個，不能原封不動沿用目前權重。

此外，MIS 只降低不同取樣策略失配造成的變異數，不保證每個場景都低雜訊。可見性突變、極亮小光源與接近零 PDF 仍可能產生高變異樣本。

本章程式在 NEE 貢獻上使用 $w_{\mathrm{light}}$；當 BSDF 路徑真的碰到光源時使用 $w_{\mathrm{bsdf}}$。相機直接看見光源時沒有前一個競爭策略，故加入完整發光值。

---

## 逐步手算例題

### 例題一：Lambert throughput 更新

某表面的反照率為

$$
\boldsymbol{\rho}=(0.8,0.6,0.4).
$$

入射方向與法線餘弦為 $0.5$。Lambert BRDF 與餘弦 PDF 分別為

$$
f_r=\frac{\boldsymbol{\rho}}{\pi},
\qquad
p_{\mathrm{bsdf}}=\frac{0.5}{\pi}.
$$

若進入表面前

$$
\boldsymbol{\beta}=(0.5,0.5,0.5),
$$

則

$$
\boldsymbol{\beta}'
=
\boldsymbol{\beta}\odot
\frac{(\boldsymbol{\rho}/\pi)0.5}{0.5/\pi}
=
\boldsymbol{\beta}\odot\boldsymbol{\rho}
=
(0.4,0.3,0.2).
$$

$\cos\theta$ 與 $\pi$ 消掉，是 BRDF 與取樣分布配對的結果。

### 例題二：面積 PDF 轉成立體角 PDF

一個面積為 $A_L=2\ \mathrm{m}^2$ 的光源被均勻取樣。表面點到光源樣本的平方距離為

$$
r^2=9\ \mathrm{m}^2,
$$

且光源端餘弦為

$$
\cos\theta_L=0.5.
$$

面積 PDF 是

$$
p_A=\frac12=0.5\ \mathrm{m}^{-2}.
$$

轉成立體角 PDF：

$$
p_\omega
=
p_A\frac{r^2}{\cos\theta_L}
=
0.5\frac{9}{0.5}
=9\ \mathrm{sr}^{-1}.
$$

PDF 大於 1 並不違法；密度不是機率本身。對有限區域積分後，機率才必須位於 $[0,1]$。

### 例題三：MIS 權重

某方向的光源取樣 PDF 為

$$
p_L=0.8\ \mathrm{sr}^{-1},
$$

BSDF PDF 為

$$
p_B=0.2\ \mathrm{sr}^{-1}.
$$

power heuristic 給出

$$
w_L
=
\frac{0.8^2}{0.8^2+0.2^2}
=
\frac{0.64}{0.68}
\approx0.941176,
$$

$$
w_B
=
\frac{0.2^2}{0.8^2+0.2^2}
\approx0.058824.
$$

兩者相加為 1。這個方向較容易由光源策略找到，因此光源策略取得較大權重。

### 例題四：俄羅斯輪盤

目前 throughput 為

$$
\boldsymbol{\beta}=(0.4,0.2,0.1).
$$

令存活機率 $p_s=0.4$。若路徑存活，更新為

$$
\frac{\boldsymbol{\beta}}{p_s}
=(1,0.5,0.25).
$$

期望 throughput 是

$$
0.4(1,0.5,0.25)+0.6(0,0,0)
=(0.4,0.2,0.1),
$$

與輪盤前相同。

### 例題五：光源選取機率

場景有三個光源，選取機率分別為

$$
(0.2,0.3,0.5).
$$

選到第二個光源後，其條件立體角 PDF 為

$$
p(\omega\mid J=2)=4\ \mathrm{sr}^{-1}.
$$

完整 PDF 為

$$
p_L(\omega,J=2)
=
0.3\times4
=
1.2\ \mathrm{sr}^{-1}.
$$

直接光估計量與 MIS 必須使用 $1.2$，而不是只使用 4。若漏掉光源選取機率，亮度與 MIS 權重都會錯誤。

---

## 實作與程式

以下是可獨立閱讀的小型 CPU 路徑追蹤器。場景只包含 Lambert 球、單一球形面積光源與常數環境光；輸出為 ASCII PPM。預設解析度為 $32\times24$、每像素 16 樣本、最多 8 次交點，使用固定亂數種子。

它是教學參考實作，不包含 BVH、透明介質、紋理、微表面材質或完整水下光學。

```python
import math
from dataclasses import dataclass
import numpy as np


PI = math.pi
EPS = 1e-5


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("不能正規化零長或非有限向量")
    return v / n


def power_heuristic(pdf_a, pdf_b):
    if pdf_a < 0.0 or pdf_b < 0.0:
        raise ValueError("PDF 不可為負")
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    denominator = a2 + b2
    return 0.0 if denominator == 0.0 else a2 / denominator


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    albedo: np.ndarray
    emission: np.ndarray

    @property
    def area(self):
        return 4.0 * PI * self.radius * self.radius


@dataclass
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    sphere_index: int


def intersect_sphere(origin, direction, sphere, t_min=EPS, t_max=math.inf):
    oc = origin - sphere.center
    half_b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - sphere.radius * sphere.radius)
    discriminant = half_b * half_b - c
    if discriminant < 0.0:
        return None

    root = math.sqrt(discriminant)
    for t in (-half_b - root, -half_b + root):
        if t_min < t < t_max:
            point = origin + t * direction
            normal = normalize(point - sphere.center)
            return t, point, normal
    return None


def intersect_scene(origin, direction, spheres, t_max=math.inf):
    closest = t_max
    best = None
    for index, sphere in enumerate(spheres):
        result = intersect_sphere(
            origin, direction, sphere, EPS, closest
        )
        if result is not None:
            t, point, normal = result
            closest = t
            best = Hit(t, point, normal, index)
    return best


def make_basis(normal):
    normal = normalize(normal)
    if abs(normal[1]) < 0.999:
        tangent = normalize(np.cross([0.0, 1.0, 0.0], normal))
    else:
        tangent = normalize(np.cross([1.0, 0.0, 0.0], normal))
    bitangent = np.cross(normal, tangent)
    return tangent, bitangent, normal


def sample_cosine_hemisphere(normal, rng):
    u1 = rng.random()
    u2 = rng.random()
    r = math.sqrt(u1)
    phi = 2.0 * PI * u2

    local = np.array([
        r * math.cos(phi),
        r * math.sin(phi),
        math.sqrt(max(0.0, 1.0 - u1)),
    ])

    tangent, bitangent, n = make_basis(normal)
    direction = normalize(
        local[0] * tangent
        + local[1] * bitangent
        + local[2] * n
    )
    cosine = max(0.0, float(np.dot(n, direction)))
    pdf = cosine / PI
    return direction, pdf


def sample_sphere_surface(sphere, rng):
    z = 1.0 - 2.0 * rng.random()
    phi = 2.0 * PI * rng.random()
    r_xy = math.sqrt(max(0.0, 1.0 - z * z))
    normal = np.array([
        r_xy * math.cos(phi),
        z,
        r_xy * math.sin(phi),
    ])
    point = sphere.center + sphere.radius * normal
    pdf_area = 1.0 / sphere.area
    return point, normal, pdf_area


def light_pdf_solid_angle(shading_point, light_point, light_normal, light):
    offset = light_point - shading_point
    distance2 = float(np.dot(offset, offset))
    if distance2 <= 0.0:
        return 0.0
    wi = offset / math.sqrt(distance2)
    cosine_light = max(0.0, float(np.dot(light_normal, -wi)))
    if cosine_light <= 0.0:
        return 0.0
    return distance2 / (light.area * cosine_light)


def visible_to_light(point, light_point, spheres):
    offset = light_point - point
    distance = float(np.linalg.norm(offset))
    if distance <= 2.0 * EPS:
        return False
    direction = offset / distance
    blocker = intersect_scene(
        point + EPS * direction,
        direction,
        spheres,
        t_max=distance - EPS,
    )
    return blocker is None


def trace_path(origin, direction, spheres, light_index, rng, max_depth=8):
    radiance = np.zeros(3, dtype=np.float64)
    beta = np.ones(3, dtype=np.float64)
    environment = np.array([0.02, 0.03, 0.05])

    previous_point = None
    previous_bsdf_pdf = 0.0

    for depth in range(max_depth):
        hit = intersect_scene(origin, direction, spheres)

        if hit is None:
            radiance += beta * environment
            break

        obj = spheres[hit.sphere_index]
        emitted = obj.emission

        if np.any(emitted > 0.0):
            if depth == 0 or previous_point is None:
                weight = 1.0
            else:
                light_pdf = light_pdf_solid_angle(
                    previous_point, hit.point, hit.normal, obj
                )
                weight = power_heuristic(
                    previous_bsdf_pdf, light_pdf
                )
            radiance += beta * emitted * weight
            break

        normal = hit.normal
        outgoing = -direction
        if np.dot(normal, outgoing) < 0.0:
            normal = -normal

        # 顯式取樣唯一的球形面積光源。
        # 因為只有一盞燈，離散光源選取機率為 1。
        light = spheres[light_index]
        light_point, light_normal, _ = sample_sphere_surface(light, rng)
        to_light = light_point - hit.point
        distance2 = float(np.dot(to_light, to_light))

        if distance2 > 0.0:
            wi = to_light / math.sqrt(distance2)
            cosine_surface = max(0.0, float(np.dot(normal, wi)))
            cosine_light = max(
                0.0, float(np.dot(light_normal, -wi))
            )

            if (
                cosine_surface > 0.0
                and cosine_light > 0.0
                and visible_to_light(hit.point, light_point, spheres)
            ):
                light_pdf = distance2 / (
                    light.area * cosine_light
                )
                bsdf_pdf = cosine_surface / PI
                mis_weight = power_heuristic(
                    light_pdf, bsdf_pdf
                )
                brdf = obj.albedo / PI
                radiance += (
                    beta
                    * brdf
                    * light.emission
                    * cosine_surface
                    * mis_weight
                    / light_pdf
                )

        new_direction, bsdf_pdf = sample_cosine_hemisphere(
            normal, rng
        )
        if bsdf_pdf <= 0.0:
            break

        # Lambert BRDF 配合餘弦取樣：f*cos/pdf = albedo。
        beta *= obj.albedo

        previous_point = hit.point.copy()
        previous_bsdf_pdf = bsdf_pdf

        # 前三個表面交點不使用俄羅斯輪盤。
        if depth >= 2:
            survival = min(0.95, float(np.max(beta)))
            if survival <= 0.0 or rng.random() > survival:
                break
            beta /= survival

        origin = hit.point + EPS * new_direction
        direction = new_direction

    # 若 for 迴圈因 max_depth 耗盡，直接回傳目前累積值；
    # 尚未追蹤的更長路徑因此可能造成截斷偏差。
    return radiance


def make_camera_ray(x, y, width, height, rng):
    origin = np.array([0.0, 1.0, 4.5])
    target = np.array([0.0, 0.0, -2.0])
    world_up = np.array([0.0, 1.0, 0.0])

    forward = normalize(target - origin)
    right = normalize(np.cross(forward, world_up))
    up = np.cross(right, forward)

    aspect = width / height
    scale = math.tan(math.radians(45.0) * 0.5)

    pixel_x = (x + rng.random()) / width
    pixel_y = (y + rng.random()) / height

    screen_x = (2.0 * pixel_x - 1.0) * aspect * scale
    screen_y = (1.0 - 2.0 * pixel_y) * scale

    direction = normalize(
        forward + screen_x * right + screen_y * up
    )
    return origin, direction


def linear_to_srgb(value):
    value = max(0.0, value)
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * value ** (1.0 / 2.4) - 0.055


def write_ppm(filename, image):
    height, width, _ = image.shape
    with open(filename, "w", encoding="ascii") as file:
        file.write(f"P3\n{width} {height}\n255\n")
        for y in range(height):
            values = []
            for x in range(width):
                rgb = [
                    linear_to_srgb(float(c))
                    for c in image[y, x]
                ]
                rgb8 = [
                    max(0, min(255, int(round(c * 255.0))))
                    for c in rgb
                ]
                values.extend(str(v) for v in rgb8)
            file.write(" ".join(values) + "\n")


def self_test():
    sphere = Sphere(
        center=np.array([0.0, 0.0, -3.0]),
        radius=1.0,
        albedo=np.array([0.8, 0.8, 0.8]),
        emission=np.zeros(3),
    )
    result = intersect_sphere(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        sphere,
    )
    assert result is not None
    assert np.isclose(result[0], 2.0)

    assert np.isclose(power_heuristic(1.0, 1.0), 0.5)
    assert np.isclose(
        power_heuristic(0.8, 0.2),
        0.64 / 0.68,
    )

    rng = np.random.default_rng(7)
    direction, pdf = sample_cosine_hemisphere(
        np.array([0.0, 1.0, 0.0]), rng
    )
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert direction[1] >= 0.0
    assert pdf >= 0.0

    # 確認零長向量會被拒絕；若未拋例外，測試必須失敗。
    try:
        normalize([0.0, 0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("零長向量應拋出 ValueError")


def main():
    self_test()

    width = 32
    height = 24
    samples_per_pixel = 16
    max_depth = 8
    rng = np.random.default_rng(20250308)

    spheres = [
        Sphere(
            np.array([0.0, -1001.0, -2.0]),
            1000.0,
            np.array([0.65, 0.72, 0.75]),
            np.zeros(3),
        ),
        Sphere(
            np.array([-0.9, -0.2, -2.7]),
            0.8,
            np.array([0.15, 0.55, 0.75]),
            np.zeros(3),
        ),
        Sphere(
            np.array([0.9, -0.4, -2.0]),
            0.6,
            np.array([0.85, 0.35, 0.12]),
            np.zeros(3),
        ),
        Sphere(
            np.array([0.0, 3.0, -2.5]),
            0.6,
            np.zeros(3),
            np.array([12.0, 11.0, 9.0]),
        ),
    ]
    light_index = 3

    image = np.zeros((height, width, 3), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            total = np.zeros(3)
            for _ in range(samples_per_pixel):
                origin, direction = make_camera_ray(
                    x, y, width, height, rng
                )
                total += trace_path(
                    origin,
                    direction,
                    spheres,
                    light_index,
                    rng,
                    max_depth,
                )
            image[y, x] = total / samples_per_pixel

    write_ppm("path_trace.ppm", image)
    print("預期：寫出 path_trace.ppm")


if __name__ == "__main__":
    main()
```

程式只寫出 PPM，不讀取檔案。固定種子使同一 NumPy 隨機數實作與相同程式參數下可重現；不同函式庫版本或修改取樣呼叫順序後，不保證逐位元相同。

---

## 測試與預期結果

讀者執行前應檢查：

1. `self_test()` 的球面交點預期為 $t=2$。
2. 相同 PDF 的 MIS 權重預期為 $0.5$。
3. `power_heuristic(0.8, 0.2)` 預期約為 `0.94117647`。
4. 餘弦半球樣本預期為單位向量，且位於法線上方。
5. 零長向量傳入 `normalize` 時，測試預期捕捉 `ValueError`；若未拋出，`else` 分支會使測試失敗。
6. 成功完成後，終端文字預期為：

```text
預期：寫出 path_trace.ppm
```

影像的預期結構是灰藍地面、左側藍色球、右側橙色球及上方亮球；陰影邊緣與間接光仍會帶有 Monte Carlo 雜訊。這是場景結構的預期，不是宣稱已比對過的參考影像。

可進行以下可重現實驗：

- 保持種子、解析度及場景不變，只把每像素樣本數由 16 改為 1，預期雜訊增加。
- 保持其他設定不變，註解 NEE 區塊，預期小光源造成的直接光更難取樣。
- 把 `beta /= survival` 錯誤刪除，預期影像系統性偏暗。
- 將兩個 MIS 權重都強制設為 1，直接光路徑可能被重複計算而偏亮。
- 將 `max_depth` 從 8 改成 1，預期只保留相機第一交點的直接光與直接看見的環境或光源，間接反射顯著減少。
- 將 `max_depth` 提高但保持樣本數不變，結果不保證逐像素更平滑；路徑更長也可能提高變異數。

效能應由讀者在自己的環境計時；本章不提供未執行的每秒射線數或 FPS。

---

## 除錯與常見陷阱

### 1. 忘記除以 PDF

Monte Carlo 估計量必須包含 $1/p$。若採非均勻取樣卻不除以 PDF，亮度會依取樣策略改變。

### 2. PDF 測度不一致

面積 PDF 的單位是 $\mathrm{m}^{-2}$，立體角 PDF 的單位是 $\mathrm{sr}^{-1}$。MIS 比較前必須先轉到相同測度。

### 3. 漏掉光源選取機率

多光源場景的完整 PDF 是離散選燈機率乘上條件 PDF。只使用光源表面或方向的條件 PDF，會使直接光估計與 MIS 權重都不正確。

### 4. NEE 與碰光源貢獻重複

若直接光源取樣已計算某條路徑，BSDF 路徑又完整加入相同貢獻，結果可能偏亮。應使用 MIS 權重，或採用不重複計算的明確策略。

### 5. 多樣本 MIS 沿用單樣本公式

若每個策略取樣數不同，權重應納入 $n_i p_i$，不能只比較 $p_i$。本章公式假設每個策略各一個樣本。

### 6. 俄羅斯輪盤終止後未補償

存活路徑必須除以 $p_s$。只丟棄路徑而不補償會造成偏暗。

### 7. 把最大深度當成無偏終止

有限最大深度會排除更長路徑。它適合作為固定預算與防呆上限，但不等於俄羅斯輪盤的期望值補償。

### 8. 陰影射線自相交

陰影與反射射線若從精確表面點發出，可能立刻撞回原表面。本章沿射線方向偏移 `EPS`。`EPS` 應依場景尺度設定，不能視為物理厚度。

### 9. 在 sRGB 中累積

路徑貢獻必須在線性 RGB 累加並取平均，最後輸出時才轉成 sRGB。不能先編碼每個樣本再平均。

### 10. 對 radiance 任意截斷

為減少亮點而直接夾住單樣本輻射亮度會引入偏差。可將截斷視為明確的偏差—變異數折衷，但不能再宣稱估計器保持無偏。

### 11. 法線朝向與單面發光

本章球形光源只由外表面向外發光。若使用錯誤法線，$\cos\theta_L$ 會變成零。雙面光源必須另行定義，不能偷偷對所有餘弦取絕對值。

---

## 養殖數位分身案例

可把程式中的大球頂部視為簡化池底，兩個小球視為魚體占位幾何，球形光源視為合成照明設備。藍色球與橙色球使用不同反照率，可觀察直接光、陰影與間接反射如何改變外觀。

若要建立養殖池的可稽核合成資料，至少應記錄：

- 影像寬高與每像素樣本數；
- 最大路徑深度與俄羅斯輪盤起始深度；
- 固定亂數種子；
- 所有球體位置、半徑、反照率與發光值；
- 光源選取分布與每個光源的條件 PDF；
- `EPS` 與場景長度單位；
- 是否啟用 NEE 與 MIS；
- 輸出色彩轉換。

若把照明設備擴充為多盞燈，可依近似功率選燈。例如第 $j$ 盞面積光源的面積為 $A_j$，平均發光亮度近似為 $\bar L_j$，可令

$$
q_j=A_j\bar L_j
$$

作為選燈權重，再正規化為機率。這只是降低變異數的取樣設計，不會改變材質或光源的物理參數。

增加樣本數只會降低此模擬器的取樣雜訊，不會使簡化球體、Lambert 材質或常數環境光自動變成真實養殖環境。水面反射、折射、吸收與水中散射屬於下一章的介質問題；本例不能用來推論真實水下照度或生物行為。

---

## 習題

### 習題 1：手算

某 Lambert 表面的反照率為 $(0.6,0.3,0.2)$，目前 throughput 為 $(0.5,0.8,1.0)$。使用餘弦加權半球取樣後，求新的 throughput。

### 習題 2：程式測試

為 `power_heuristic` 加入測試，驗證：

1. $p_a=p_b=0.4$ 時權重為 $0.5$；
2. $p_a=0$、$p_b=1$ 時權重為 0；
3. 兩個 PDF 都為零時函式回傳 0；
4. 負 PDF 會拋出 `ValueError`。

### 習題 3：反例與除錯

某俄羅斯輪盤程式如下：

```python
if rng.random() > survival:
    break
# 繼續追蹤，但未修改 beta
```

說明錯誤及偏差方向。若原 throughput 為 $(0.3,0.2,0.1)$，存活機率為 $0.25$，求正確的存活後 throughput。

### 習題 4：整合應用

表面點使用 Lambert 反照率 $\rho=0.5$。某光源樣本具有

$$
L_e=8,\quad
\cos\theta=0.6,\quad
p_L=0.9,\quad
p_B=0.3,
$$

且可見性為 1、進入頂點前的 throughput 為 $\beta=0.75$。使用 power heuristic，求單色直接光估計值

$$
\beta\frac{(\rho/\pi)L_e\cos\theta}{p_L}w_L.
$$

### 習題 5：多光源 PDF

場景有四個面積光源，選取機率為 $(0.1,0.2,0.3,0.4)$。第三個光源內部的條件立體角 PDF 為 $5\ \mathrm{sr}^{-1}$。求完整光源 PDF，並說明 MIS 應使用完整 PDF 還是條件 PDF。

### 習題 6：多樣本 MIS

光源策略每個頂點取 $n_L=4$ 個樣本，BSDF 策略取 $n_B=1$ 個樣本。某方向上

$$
p_L=0.2,\qquad p_B=0.4.
$$

用含樣本數的 power heuristic 求 $w_L$ 與 $w_B$。

---

## 習題解答

### 習題 1 解答

Lambert BRDF 配合餘弦加權取樣時，

$$
\frac{f_r\cos\theta}{p_{\mathrm{bsdf}}}
=\boldsymbol{\rho}.
$$

所以

$$
\boldsymbol{\beta}'
=
(0.5,0.8,1.0)\odot(0.6,0.3,0.2)
=
(0.3,0.24,0.2).
$$

### 習題 2 解答

```python
def test_power_heuristic():
    assert np.isclose(power_heuristic(0.4, 0.4), 0.5)
    assert np.isclose(power_heuristic(0.0, 1.0), 0.0)
    assert np.isclose(power_heuristic(0.0, 0.0), 0.0)

    try:
        power_heuristic(-0.1, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("負 PDF 應被拒絕")
```

兩個 PDF 都為零代表兩種策略都不會生成該樣本；本章輔助函式回傳 0，避免 $0/0$。

### 習題 3 解答

終止機率為 $1-p_s$。若存活後不除以 $p_s$，期望值會變成原值的 $p_s$ 倍，造成偏暗。

正確補償為

$$
\boldsymbol{\beta}'
=
\frac{(0.3,0.2,0.1)}{0.25}
=
(1.2,0.8,0.4).
$$

雖然存活樣本權重變大，但只有四分之一的路徑存活，因此期望值保持不變。

### 習題 4 解答

光源策略的 MIS 權重為

$$
w_L
=
\frac{0.9^2}{0.9^2+0.3^2}
=
\frac{0.81}{0.90}
=0.9.
$$

直接光估計為

$$
0.75
\frac{(0.5/\pi)(8)(0.6)}{0.9}
(0.9).
$$

因此

$$
\widehat{L}_{\mathrm{direct}}
=
0.75\frac{2.4}{\pi}
=
\frac{1.8}{\pi}
\approx0.573.
$$

### 習題 5 解答

選到第三個光源的機率為 $0.3$，因此完整 PDF 為

$$
p_L
=
0.3\times5
=
1.5\ \mathrm{sr}^{-1}.
$$

MIS 必須使用完整生成機率，也就是同時包含選燈機率與條件方向 PDF。只使用 $5\ \mathrm{sr}^{-1}$ 會漏掉離散選擇機率。

### 習題 6 解答

含樣本數的 power heuristic 為

$$
w_L
=
\frac{(n_Lp_L)^2}
{(n_Lp_L)^2+(n_Bp_B)^2}.
$$

先算

$$
n_Lp_L=4(0.2)=0.8,
\qquad
n_Bp_B=1(0.4)=0.4.
$$

所以

$$
w_L
=
\frac{0.8^2}{0.8^2+0.4^2}
=
\frac{0.64}{0.80}
=0.8,
$$

$$
w_B
=
\frac{0.4^2}{0.8^2+0.4^2}
=
0.2.
$$

兩權重相加為 1。若忽略樣本數，會錯誤得到 $w_L=0.2$、$w_B=0.8$，無法反映光源策略實際取得了四倍樣本。

---

## 本章小結

路徑追蹤把渲染方程的遞迴積分轉成隨機光路徑。throughput 累積每次反射的

$$
\frac{f_r\cos\theta}{p}
$$

因子；對 Lambert BRDF 與餘弦加權取樣，此因子簡化為反照率。

顯式光源取樣能有效尋找小光源，但面積 PDF 必須轉成立體角 PDF。多光源場景還必須把離散選燈機率納入完整 PDF。BSDF 取樣與光源取樣可能生成相同路徑，MIS 以相同測度下的完整 PDF 分配權重；每種策略使用多個樣本時，權重還要納入樣本數。

俄羅斯輪盤透過存活後除以存活機率，在不改變期望值的前提下終止路徑；固定最大深度則可能直接丟棄長路徑而留下截斷偏差。固定種子、解析度、樣本數、深度與場景參數，是建立可重現實驗的最低要求，但不能取代真實材質、介質與量測驗證。

---

## 參考來源

- [G2] *Physically Based Rendering, 4th ed.*：Reflection Models  
  <https://pbr-book.org/4ed/Reflection_Models>
- [G3] *Physically Based Rendering, 4th ed.*：The Light Transport Equation  
  <https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation>
- [G4] *Ray Tracing in One Weekend*  
  <https://raytracing.github.io/books/RayTracingInOneWeekend.html>
- [G7] NumPy 線性代數參考  
  <https://numpy.org/doc/stable/reference/routines.linalg.html>

以上來源供讀者回查 BRDF、光傳輸、路徑追蹤與 NumPy API 背景；本章推導、場景與程式依本書符號、座標、單位及可重現性約定整理。