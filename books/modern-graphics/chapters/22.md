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
7. 分辨「表面點事件」與「方向事件」，避免混用 PDF。
8. 建立固定種子、低解析度、固定樣本預算的 CPU 路徑追蹤器。

先備知識包括射線求交、Lambert BRDF、半球立體角、隨機變數、PDF 與 Monte Carlo 估計。全章顏色皆為線性 RGB；$L$、$f_r$ 等 RGB 值是三個色彩通道的數值近似，不代表完整光譜模型。

---

## 問題與直覺

光線投射器只找出相機最先看見的表面；局部照明通常只計算表面直接接收的光。實際場景中，池壁會把光反射到魚體，魚腹也可能被池底的間接反射照亮。這類多次反射形成全域光傳輸。

路徑追蹤從相機發出射線。射線碰到表面後，依材質抽樣新方向，再繼續追蹤。每條路徑可能：

- 直接碰到發光表面；
- 經一次或多次反射後碰到光源；
- 離開場景而取得環境光；
- 被俄羅斯輪盤提前終止；
- 到達工程設定的最大深度。

若只靠隨機反射方向偶然撞到小光源，大多數樣本不會得到直接光，影像便有強烈雜訊。重要性取樣不改變原積分，而是把更多樣本放在貢獻可能較大的方向，再以 PDF 補償抽樣不均。

光源取樣擅長尋找小面積光源；BSDF 取樣擅長尋找材質偏好的方向。MIS 讓兩者共同工作。不過，兩種策略的 PDF 必須描述同一種隨機事件。若一邊描述「抽到光源表面點」，另一邊描述「抽到方向」，便不能直接比較。

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
- $L_i$：沿 $\omega_i$ 入射的輻射亮度；
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

若方向 $\omega$ 由 PDF $p(\omega)$ 抽樣，且在非零被積函數處滿足 $p(\omega)>0$，則

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

除以 PDF 是補償抽樣分布不均。若忘記除以 PDF，結果通常有偏；若 PDF 為零但貢獻非零，估計器則無法涵蓋該部分積分。

### 3. Lambert 材質與餘弦加權取樣

Lambert BRDF 為

$$
f_r=\frac{\boldsymbol{\rho}}{\pi},
$$

其中 $\boldsymbol{\rho}=(\rho_r,\rho_g,\rho_b)$ 是線性 RGB 反照率。

餘弦加權半球取樣的 PDF 是

$$
p_{\mathrm{bsdf}}(\omega_i)
=
\frac{\cos\theta}{\pi},
\qquad
\cos\theta=\max(0,\mathbf{n}\cdot\omega_i).
$$

因此

$$
\frac{f_r\cos\theta}{p_{\mathrm{bsdf}}}
=
\frac{(\boldsymbol{\rho}/\pi)\cos\theta}
{\cos\theta/\pi}
=
\boldsymbol{\rho}.
$$

純 Lambert 表面使用餘弦取樣時，每次反射只需把 throughput 乘上反照率。這個化簡不適用於任意 BRDF。

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

若目前頂點有發光 $\mathbf{L}_e$，像素貢獻為

$$
\mathbf{L}\leftarrow
\mathbf{L}+\boldsymbol{\beta}\odot\mathbf{L}_e.
$$

`throughput` 是路徑樣本在估計器中的乘法權重，不是「剩餘光能」。其值可能大於 1；任意夾住它會引入偏差。

### 5. 面積取樣與立體角 PDF

在表面點 $\mathbf{x}$ 抽樣光源點 $\mathbf{y}$，稱為 next-event estimation，簡稱 NEE。若以面積 PDF $p_A(\mathbf{y})$ 抽樣，令

$$
\mathbf{d}=\mathbf{y}-\mathbf{x},
\qquad
r^2=\|\mathbf{d}\|^2,
\qquad
\omega_i=\frac{\mathbf{d}}{\|\mathbf{d}\|}.
$$

光源法線為 $\mathbf{n}_L$，定義幾何 Jacobian 中的絕對餘弦

$$
|\cos\theta_L|
=
|\mathbf{n}_L\cdot(-\omega_i)|.
$$

單一表面分支轉成立體角 PDF：

$$
p_{\omega,j}(\omega_i)
=
p_A(\mathbf{y}_j)
\frac{r_j^2}{|\cos\theta_{L,j}|}.
$$

這裡的絕對值來自面積與立體角的幾何轉換，不等於發光模型。單面光源是否沿該方向發光，仍應另以

$$
\max(0,\mathbf{n}_L\cdot(-\omega_i))
$$

判定。某表面點背向著色點時，貢獻可為零，但只要取樣器可能抽到該點，其面積 PDF 就不能被說成零。

### 6. 多對一映射與方向事件

面積點映射到方向不一定一對一。若同一方向 $\omega$ 可由光源上的多個表面點 $\mathbf{y}_j$ 生成，方向總 PDF 應加總所有分支：

$$
p_\omega(\omega)
=
\sum_j
p_A(\mathbf{y}_j)
\frac{r_j^2}{|\cos\theta_{L,j}|}.
$$

例如從球外觀察球面，一條穿過球的射線通常與球面有前、後兩個交點。若取樣器均勻抽整個球面，兩個點都可能映射到同一方向，因此不能只把前交點的 Jacobian 當成完整方向 PDF。

此外，後表面對單面向外發光球的直接貢獻為零，而且通常被前表面遮擋；但「貢獻為零」與「取樣機率為零」是兩件不同的事。

MIS 最簡單的做法，是讓所有策略都在方向事件空間工作：

- BSDF 策略直接抽樣方向；
- 光源策略也直接抽樣方向；
- 兩者 PDF 都以 $\mathrm{sr}^{-1}$ 表示；
- 射線的首次交點決定實際光源事件。

本章程式採此方法：從著色點直接均勻抽樣球形光源所張成的可見方向圓錐，而不是均勻抽整個球面。圓錐內每個方向對應首次可見球面交點，避免前、後表面多分支歧義。

### 7. 球形光源的方向圓錐取樣

設著色點為 $\mathbf{x}$，球心為 $\mathbf{c}$，半徑為 $R$，且著色點位於球外。令

$$
d=\|\mathbf{c}-\mathbf{x}\|,
\qquad d>R.
$$

球在著色點所張成圓錐的半角 $\theta_{\max}$ 滿足

$$
\sin\theta_{\max}=\frac{R}{d},
$$

因此

$$
\cos\theta_{\max}
=
\sqrt{1-\frac{R^2}{d^2}}.
$$

圓錐立體角為

$$
\Omega
=
2\pi(1-\cos\theta_{\max}).
$$

若在此圓錐內均勻抽樣方向，PDF 為

$$
p_L(\omega)
=
\frac{1}{\Omega}
=
\frac{1}
{2\pi(1-\cos\theta_{\max})}.
$$

這個 PDF 直接定義於方向事件，不需再做面積 Jacobian 轉換。抽得方向後，射線與球的第一交點就是 NEE 使用的光源點。若光源只向外發光，該可見前表面的出射餘弦應為正；程式仍會明確檢查朝向。

### 8. 多光源的完整 PDF

若場景有 $M$ 個光源，通常先抽樣離散光源索引 $J$，再由該光源抽樣方向。完整 PDF 為

$$
p_L(\omega,j)
=
P(J=j)\,p(\omega\mid J=j).
$$

直接光估計器與 MIS 權重都必須使用完整生成機率。最簡單是均勻選燈：

$$
P(J=j)=\frac1M.
$$

也可依近似功率 $q_j$ 選燈：

$$
P(J=j)=
\frac{q_j}{\sum_{k=1}^{M}q_k}.
$$

若多個光源分布都能生成同一方向，邊際方向 PDF 是混合密度：

$$
p_L(\omega)
=
\sum_{j=1}^{M}
P(J=j)\,p(\omega\mid J=j).
$$

本章程式只有一個光源，所以選燈機率為 1。

### 9. 顯式光源估計

若光源策略直接抽樣方向 $\omega_i$，直接光單樣本估計為

$$
\widehat{\mathbf{L}}_{\mathrm{direct}}
=
\boldsymbol{\beta}\odot
\frac{
f_r\,\mathbf{L}_e\,
\cos\theta
}{
p_L(\omega_i)
}
V(\mathbf{x},\omega_i),
$$

其中

$$
\cos\theta=\max(0,\mathbf{n}\cdot\omega_i).
$$

$V$ 表示沿該方向的首次交點是否為所選光源。光源背面、被遮擋或材質半球外的樣本，其貢獻為零；這不表示原取樣 PDF 必須為零。

### 10. 俄羅斯輪盤與最大深度

令存活機率為 $p_s\in(0,1]$。路徑以機率 $1-p_s$ 終止；若存活，則

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

所以期望值不變。實作可用

$$
p_s=\min(0.95,\max(\beta_r,\beta_g,\beta_b)).
$$

最大深度與輪盤不同。若最大深度為 $D$，所有更長路徑都被捨棄；只要其真實貢獻不為零，就會留下截斷偏差。輪盤則讓長路徑仍有存活機率並補償權重。本章同時使用兩者：輪盤控制平均成本，最大深度提供固定預算與防呆上限。

### 11. MIS 與成立條件

光源取樣與 BSDF 取樣可生成相同直接光方向。若各取一個樣本，power heuristic 為

$$
w_L=
\frac{p_L^2}{p_L^2+p_B^2},
\qquad
w_B=
\frac{p_B^2}{p_L^2+p_B^2}.
$$

兩個 PDF 必須描述相同方向事件，且使用相同測度。若某策略在該方向不可能生成樣本，其 PDF 與權重才應為零。

一般而言，MIS 權重需在有貢獻區域滿足

$$
\sum_i w_i(x)=1,
$$

且所有非零貢獻都至少被一種策略覆蓋。

若策略 $i$ 取 $n_i$ 個樣本，power heuristic 應使用 $n_i p_i$：

$$
w_i(x)
=
\frac{(n_i p_i(x))^2}
{\sum_j(n_j p_j(x))^2}.
$$

本章每個頂點各取一個光源方向樣本與一個 BSDF 樣本，所以 $n_L=n_B=1$。

---

## 逐步手算例題

### 例題一：Lambert throughput 更新

某表面反照率為

$$
\boldsymbol{\rho}=(0.8,0.6,0.4),
$$

入射方向餘弦為 $0.5$，進入表面前

$$
\boldsymbol{\beta}=(0.5,0.5,0.5).
$$

由

$$
f_r=\frac{\boldsymbol{\rho}}{\pi},
\qquad
p_B=\frac{0.5}{\pi},
$$

可得

$$
\boldsymbol{\beta}'
=
\boldsymbol{\beta}\odot
\frac{(\boldsymbol{\rho}/\pi)0.5}{0.5/\pi}
=
(0.4,0.3,0.2).
$$

### 例題二：面積 PDF 轉成立體角 PDF

面積為 $A_L=2\ \mathrm{m}^2$ 的平面光源被均勻取樣。若

$$
r^2=9\ \mathrm{m}^2,
\qquad
|\cos\theta_L|=0.5,
$$

則

$$
p_A=\frac12=0.5\ \mathrm{m}^{-2},
$$

以及

$$
p_\omega
=
p_A\frac{r^2}{|\cos\theta_L|}
=
0.5\frac9{0.5}
=
9\ \mathrm{sr}^{-1}.
$$

PDF 大於 1 並不違法；密度不是機率本身。

### 例題三：球形光源的圓錐 PDF

著色點到球心距離為 $d=5$ 公尺，球半徑為 $R=1$ 公尺。則

$$
\cos\theta_{\max}
=
\sqrt{1-\frac{1^2}{5^2}}
=
\sqrt{\frac{24}{25}}
\approx0.979796.
$$

圓錐立體角為

$$
\Omega
=
2\pi(1-0.979796)
\approx0.126946\ \mathrm{sr}.
$$

均勻方向 PDF 為

$$
p_L=\frac1\Omega
\approx7.877\ \mathrm{sr}^{-1}.
$$

圓錐外方向的 PDF 為零；圓錐內方向皆為此常數。這與均勻抽整個球面不是同一個取樣分布。

### 例題四：球面前後分支

若改成均勻抽整個球面，一條穿過球心附近的方向通常對應前、後兩個球面點。假設兩分支轉換後密度分別為

$$
p_{\omega,1}=3\ \mathrm{sr}^{-1},
\qquad
p_{\omega,2}=5\ \mathrm{sr}^{-1},
$$

則方向總 PDF 為

$$
p_\omega=3+5=8\ \mathrm{sr}^{-1}.
$$

即使後表面因單面發光或遮擋而貢獻為零，只要取樣器可能抽到它，它仍屬於生成該方向的分支。不能因貢獻為零就把其取樣密度改成零。

### 例題五：MIS 與俄羅斯輪盤

若

$$
p_L=0.8,\qquad p_B=0.2,
$$

則

$$
w_L=\frac{0.8^2}{0.8^2+0.2^2}
\approx0.941176,
$$

$$
w_B\approx0.058824.
$$

另設 throughput 為

$$
\boldsymbol{\beta}=(0.4,0.2,0.1),
$$

存活機率 $p_s=0.4$。存活後

$$
\boldsymbol{\beta}'=(1,0.5,0.25).
$$

其期望為

$$
0.4(1,0.5,0.25)+0.6(0,0,0)
=(0.4,0.2,0.1).
$$

---

## 實作與程式

以下是可獨立閱讀的小型 CPU 路徑追蹤器。場景只包含 Lambert 球、單一球形面積光源與常數環境光；輸出 ASCII PPM。解析度為 $32\times24$、每像素 16 樣本、最多 8 次交點，使用固定亂數種子。

程式直接抽樣球形光源的可見方向圓錐，因此 NEE 與 BSDF 都使用方向事件及 $\mathrm{sr}^{-1}$ PDF。

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
    if not np.isfinite(pdf_a) or not np.isfinite(pdf_b):
        raise ValueError("PDF 必須是有限值")
    if pdf_a < 0.0 or pdf_b < 0.0:
        raise ValueError("PDF 不可為負")
    scale = max(pdf_a, pdf_b)
    if scale == 0.0:
        return 0.0
    a = pdf_a / scale
    b = pdf_b / scale
    return (a * a) / (a * a + b * b)


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    albedo: np.ndarray
    emission: np.ndarray


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
    return direction, cosine / PI


def sphere_light_cone(point, light):
    to_center = light.center - point
    distance2 = float(np.dot(to_center, to_center))
    radius2 = light.radius * light.radius
    if distance2 <= radius2:
        raise ValueError("本章球光源取樣要求著色點位於光源球外")

    axis = to_center / math.sqrt(distance2)
    cos_theta_max = math.sqrt(max(0.0, 1.0 - radius2 / distance2))
    solid_angle = 2.0 * PI * (1.0 - cos_theta_max)
    if solid_angle <= 0.0:
        raise ValueError("球形光源立體角無法可靠表示")
    return axis, cos_theta_max, solid_angle


def sample_sphere_light_direction(point, light, rng):
    axis, cos_theta_max, solid_angle = sphere_light_cone(
        point, light
    )

    u1 = rng.random()
    u2 = rng.random()
    cos_theta = 1.0 - u1 * (1.0 - cos_theta_max)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * PI * u2

    tangent, bitangent, axis = make_basis(axis)
    direction = normalize(
        sin_theta * math.cos(phi) * tangent
        + sin_theta * math.sin(phi) * bitangent
        + cos_theta * axis
    )
    pdf = 1.0 / solid_angle

    result = intersect_sphere(point, direction, light)
    if result is None:
        raise RuntimeError("圓錐方向理應與球形光源相交")
    _, light_point, light_normal = result
    return direction, light_point, light_normal, pdf


def sphere_light_direction_pdf(point, direction, light):
    """
    回傳球形光源的方向圓錐 PDF。
    PDF 描述方向能否由取樣器生成，不以發光朝向歸零。
    """
    _, _, solid_angle = sphere_light_cone(point, light)
    result = intersect_sphere(point, direction, light)
    if result is None:
        return 0.0
    return 1.0 / solid_angle


def visible_to_light(point, direction, distance, spheres):
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

        if np.any(obj.emission > 0.0):
            # 單面向外發光：-direction 是從光源指向前一頂點。
            front_emission = float(np.dot(hit.normal, -direction)) > 0.0
            if front_emission:
                if depth == 0 or previous_point is None:
                    weight = 1.0
                else:
                    light_pdf = sphere_light_direction_pdf(
                        previous_point, direction, obj
                    )
                    weight = power_heuristic(
                        previous_bsdf_pdf, light_pdf
                    )
                radiance += beta * obj.emission * weight
            break

        normal = hit.normal
        outgoing = -direction
        if np.dot(normal, outgoing) < 0.0:
            normal = -normal

        # 顯式抽樣唯一球形光源的可見方向圓錐。
        light = spheres[light_index]
        (
            wi,
            light_point,
            light_normal,
            light_pdf,
        ) = sample_sphere_light_direction(hit.point, light, rng)

        cosine_surface = max(0.0, float(np.dot(normal, wi)))
        cosine_emission = max(
            0.0, float(np.dot(light_normal, -wi))
        )
        distance = float(np.linalg.norm(light_point - hit.point))

        if (
            cosine_surface > 0.0
            and cosine_emission > 0.0
            and visible_to_light(
                hit.point, wi, distance, spheres
            )
        ):
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

        # Lambert 配合餘弦取樣：f*cos/pdf = albedo。
        beta *= obj.albedo

        previous_point = hit.point.copy()
        previous_bsdf_pdf = bsdf_pdf

        if depth >= 2:
            survival = min(0.95, float(np.max(beta)))
            if survival <= 0.0 or rng.random() > survival:
                break
            beta /= survival

        origin = hit.point + EPS * new_direction
        direction = new_direction

    # max_depth 耗盡時直接回傳，可能留下截斷偏差。
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

    return origin, normalize(
        forward + screen_x * right + screen_y * up
    )


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
        np.array([0.0, 0.0, -3.0]),
        1.0,
        np.array([0.8, 0.8, 0.8]),
        np.zeros(3),
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
        power_heuristic(0.8, 0.2), 0.64 / 0.68
    )

    rng = np.random.default_rng(7)
    direction, pdf = sample_cosine_hemisphere(
        np.array([0.0, 1.0, 0.0]), rng
    )
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert direction[1] >= 0.0
    assert pdf >= 0.0

    try:
        normalize([0.0, 0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("零長向量應拋出 ValueError")

    light = Sphere(
        np.array([0.0, 0.0, -5.0]),
        1.0,
        np.zeros(3),
        np.ones(3),
    )
    point = np.array([0.0, 0.0, 0.0])
    center_direction = np.array([0.0, 0.0, -1.0])
    cone_pdf = sphere_light_direction_pdf(
        point, center_direction, light
    )
    expected_cos = math.sqrt(24.0 / 25.0)
    expected_pdf = 1.0 / (
        2.0 * PI * (1.0 - expected_cos)
    )
    assert np.isclose(cone_pdf, expected_pdf)

    miss_pdf = sphere_light_direction_pdf(
        point, np.array([1.0, 0.0, 0.0]), light
    )
    assert miss_pdf == 0.0


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

程式只寫出 PPM，不讀取檔案。固定種子使相同程式、參數與隨機數實作可重現；修改取樣呼叫順序或函式庫版本後，不保證逐位元相同。

---

## 測試與預期結果

`self_test()` 檢查：

1. 球面交點預期為 $t=2$。
2. 相同 PDF 的 MIS 權重預期為 $0.5$。
3. `power_heuristic(0.8, 0.2)` 預期約為 `0.94117647`。
4. 餘弦半球樣本為單位向量且位於法線上方。
5. 零長向量預期拋出 `ValueError`。
6. 距離 5、公尺半徑 1 的球光源，其中心方向 PDF 預期約為 $7.877\ \mathrm{sr}^{-1}$。
7. 完全錯過球光源的方向 PDF 預期為 0。

成功完成後，終端文字預期為：

```text
預期：寫出 path_trace.ppm
```

影像預期包含灰藍地面、左側藍球、右側橙球與上方亮球。陰影與間接光仍帶有 Monte Carlo 雜訊；這是結構預期，不是已執行的參考圖比對。

可重現實驗包括：

- 將每像素樣本數由 16 改為 1，預期雜訊增加。
- 關閉 NEE，預期小光源更難由 BSDF 路徑命中。
- 刪除 `beta /= survival`，預期結果系統性偏暗。
- 把兩個 MIS 權重都設為 1，可能重複計算直接光。
- 將 `max_depth` 改為 1，間接反射預期顯著減少。

效能應由讀者在自己的環境計時；本章不提供未執行的 FPS。

---

## 除錯與常見陷阱

### 1. 忘記除以 PDF

非均勻取樣後必須以 $1/p$ 補償，否則亮度會隨取樣策略改變。

### 2. 混用事件與測度

面積 PDF 單位為 $\mathrm{m}^{-2}$，方向 PDF 單位為 $\mathrm{sr}^{-1}$。MIS 比較前必須統一事件空間及測度。

### 3. 把零貢獻當成零 PDF

背向光源點可能不發光，被遮擋點也可能沒有貢獻；只要取樣器可能生成該樣本，其 PDF 就不能因此改成零。

### 4. 忽略多對一映射

均勻抽整個球面時，同一方向可能對應前後兩個表面點。方向 PDF 必須加總分支。若不想處理，可像本章一樣直接取樣可見方向圓錐。

### 5. 漏掉選燈機率

多光源完整 PDF 是選燈機率乘上條件 PDF。漏掉離散機率會使估計量與 MIS 權重錯誤。

### 6. 多樣本 MIS 沿用單樣本公式

不同策略樣本數不同時，權重應使用 $n_i p_i$。

### 7. 俄羅斯輪盤未補償

存活後必須除以 $p_s$。只刪除路徑而不補償會偏暗。

### 8. 把最大深度當成無偏終止

固定深度會捨棄更長路徑；它是預算上限，不是輪盤補償。

### 9. 自相交與 epsilon

次級射線需避開原表面，但 `EPS` 應依場景尺度設定，不能視為物理厚度。

### 10. 在 sRGB 中累積

所有樣本必須在線性 RGB 累積與平均，輸出時才轉為 sRGB。

---

## 養殖數位分身案例

可把程式中的大球頂部視為簡化池底，兩個小球視為魚體占位幾何，球形光源視為合成照明設備。不同反照率可展示直接光、陰影與間接反射如何改變外觀。

可稽核合成資料至少應記錄：

- 影像寬高、每像素樣本數與亂數種子；
- 最大路徑深度與輪盤起始深度；
- 幾何位置、半徑、反照率與發光值；
- 光源取樣事件空間、選燈分布及方向 PDF；
- `EPS` 與長度單位；
- 是否啟用 NEE 與 MIS；
- 線性 RGB 至 sRGB 的輸出轉換。

若擴充多盞燈，可用光源面積 $A_j$ 與平均發光亮度 $\bar L_j$ 建立近似權重

$$
q_j=A_j\bar L_j,
$$

再正規化成選燈機率。這只改變取樣效率，不改變光源物理參數。

增加樣本數只會降低此簡化模擬器的取樣雜訊，不會使 Lambert 球、常數環境光或空氣中的光傳輸自動成為真實水下模型。水面折射、吸收與體散射需另行建模。

---

## 習題

### 習題 1：手算

某 Lambert 表面的反照率為 $(0.6,0.3,0.2)$，目前 throughput 為 $(0.5,0.8,1.0)$。使用餘弦加權半球取樣後，求新 throughput。

### 習題 2：程式測試

為 `power_heuristic` 加入測試：

1. $p_a=p_b=0.4$ 時權重為 $0.5$；
2. $p_a=0,p_b=1$ 時權重為 0；
3. 兩個 PDF 都為零時回傳 0；
4. 負 PDF 會拋出 `ValueError`。

### 習題 3：反例／除錯

某程式均勻抽樣整個球面，卻在抽到背向著色點的球面位置時把 PDF 設為零。說明錯誤。若同一方向的前、後分支密度分別為 2 與 $3\ \mathrm{sr}^{-1}$，求方向總 PDF。

### 習題 4：俄羅斯輪盤

若 throughput 為 $(0.3,0.2,0.1)$，存活機率為 $0.25$，求存活後的正確 throughput，並說明未補償時的偏差方向。

### 習題 5：整合應用

Lambert 反照率 $\rho=0.5$，且

$$
L_e=8,\quad
\cos\theta=0.6,\quad
p_L=0.9,\quad
p_B=0.3,\quad
\beta=0.75.
$$

可見性為 1。求使用 power heuristic 的直接光估計

$$
\beta\frac{(\rho/\pi)L_e\cos\theta}{p_L}w_L.
$$

### 習題 6：多光源與多樣本 MIS

四個光源的選取機率為 $(0.1,0.2,0.3,0.4)$。第三個光源的條件方向 PDF 為 $5\ \mathrm{sr}^{-1}$。另有 $n_L=4$、$n_B=1$，且某方向上 $p_L=0.2,p_B=0.4$。

1. 求第三個光源的完整 PDF。
2. 求含樣本數的 $w_L,w_B$。

---

## 習題解答

### 習題 1 解答

Lambert BRDF 配合餘弦取樣時，權重因子等於反照率：

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

### 習題 3 解答

背向表面點可能不發光，但均勻球面取樣器仍可能抽到它，所以其面積 PDF 不是零。若映射到同一方向的兩個分支密度為 2 與 3，方向總 PDF 是

$$
p_\omega=2+3=5\ \mathrm{sr}^{-1}.
$$

貢獻函數可因朝向或遮擋而為零；PDF 必須忠實描述取樣器實際生成樣本的機率。

### 習題 4 解答

存活後應除以 $0.25$：

$$
\boldsymbol{\beta}'
=
\frac{(0.3,0.2,0.1)}{0.25}
=
(1.2,0.8,0.4).
$$

若不補償，期望值只剩原值的四分之一，結果系統性偏暗。

### 習題 5 解答

光源策略權重為

$$
w_L
=
\frac{0.9^2}{0.9^2+0.3^2}
=0.9.
$$

因此

$$
\widehat{L}_{\mathrm{direct}}
=
0.75
\frac{(0.5/\pi)(8)(0.6)}{0.9}
(0.9)
=
\frac{1.8}{\pi}
\approx0.573.
$$

### 習題 6 解答

第三個光源的完整 PDF 為

$$
0.3\times5
=
1.5\ \mathrm{sr}^{-1}.
$$

多樣本 power heuristic 使用

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
=0.8,
$$

$$
w_B=0.2.
$$

---

## 本章小結

路徑追蹤把渲染方程的遞迴積分轉成隨機光路徑。throughput 累積每次反射的

$$
\frac{f_r\cos\theta}{p}
$$

因子；對 Lambert BRDF 與餘弦取樣，此因子簡化為反照率。

顯式光源取樣與 BSDF 取樣進行 MIS 時，必須描述同一事件並使用相同測度。面積取樣映射到方向若不是一對一，方向 PDF 必須加總所有分支。背向、遮擋或零發光只會令貢獻為零，不能任意改寫取樣 PDF。本章直接抽樣球形光源的可見方向圓錐，以首次交點定義光源事件。

俄羅斯輪盤透過存活後除以存活機率維持期望值；固定最大深度則可能留下截斷偏差。固定種子、解析度、樣本數、事件定義、PDF 與場景參數，是建立可重現路徑追蹤實驗的最低要求。

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

以上來源供讀者回查 BRDF、光傳輸、路徑追蹤與 NumPy API 背景；本章推導、案例與程式依本書符號、單位及可重現性約定整理。