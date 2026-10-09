# 第 2 章　向量、外積與幾何判定

> 第一部｜幾何、座標與成像

## 學習目標與先備知識

本章把 Volume I 的向量與內積知識，轉化為可直接用於圖學管線的幾何工具。完成本章後，讀者應能：

1. 用內積計算長度、夾角、投影及判斷方向關係。
2. 用外積計算三角形面積、表面法線與頂點繞序。
3. 從三個非共線點建立平面方程。
4. 計算點到平面的帶號距離與無號距離。
5. 辨認零長向量、退化三角形及近退化幾何。
6. 區分長度、面積與無因次容差。
7. 實作並測試 `dot`、`cross`、`normalize` 與點到平面距離。

本書採右手世界座標系：$+X$ 向右、$+Y$ 向上、$+Z$ 由畫面指向觀者。數學向量皆為 column vector。位置與位移的長度單位為公尺，面積單位為平方公尺。

本章程式使用 Python 3.10+ 與 NumPy 2.2.6 相容寫法。程式僅供讀者自行執行；列出的輸出均標為「預期」，不宣稱已在特定平台執行。

---

## 問題與直覺

養殖場數位分身不只是把池體與魚畫成彩色像素。幾何處理與渲染程式會反覆回答：

- 池壁朝向池內，還是池外？
- 光線從表面的正面還是背面射入？
- 某位置位於水面上方還是下方？
- 三個頂點是否真的形成三角形？
- 魚鰭上的兩條邊幾乎平行時，法線是否可靠？
- 相鄰網格面是否使用一致的頂點繞序？

這些問題的共同核心是向量。內積把兩個向量縮成純量，可回答「有多同向」；外積把兩個三維向量轉成垂直方向，可回答「它們張成的平面朝哪裡」以及「張成多大面積」。

理想數學與浮點運算之間仍有差距。理論上，平行向量的外積等於零；實際模型卻可能含有量化誤差、程序化建模誤差或浮點捨入。另一方面，任意寫死 `1e-6` 也不安全：若拿帶長度單位的門檻去比較面積，量綱便不一致；同一個門檻用在毫米級魚鰭與百公尺池體，也可能得到完全不同的判定。

因此，本章會區分：

- 精確幾何條件與浮點近似判定；
- 絕對容差與相對容差；
- 長度量、面積量與無因次量；
- 「數學上不退化」與「工程上品質足夠」兩種要求。

---

## 數學與幾何推導

### 1. 向量、點與位移

令三維向量為

$$
\mathbf{a}=
\begin{pmatrix}
a_x\\a_y\\a_z
\end{pmatrix},
\qquad
\mathbf{b}=
\begin{pmatrix}
b_x\\b_y\\b_z
\end{pmatrix}.
$$

若 $\mathbf{p}$、$\mathbf{q}$ 是位置，則

$$
\mathbf{v}=\mathbf{q}-\mathbf{p}
$$

是從 $\mathbf{p}$ 指向 $\mathbf{q}$ 的位移。位置可以相減得到位移，位置加位移則得到另一位置。

向量長度為

$$
\|\mathbf{a}\|
=\sqrt{a_x^2+a_y^2+a_z^2}.
$$

若座標以公尺表示，位移與其長度的單位都是公尺。長度具有非負性：

$$
\|\mathbf{a}\|\ge 0,
$$

而且只有零向量才滿足 $\|\mathbf{a}\|=0$。

### 2. 內積

內積定義為

$$
\mathbf{a}\cdot\mathbf{b}
=a_xb_x+a_yb_y+a_zb_z.
$$

其幾何形式是

$$
\mathbf{a}\cdot\mathbf{b}
=\|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta,
$$

其中 $\theta\in[0,\pi]$ 是兩向量夾角。兩向量皆非零時，

$$
\cos\theta=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|\,\|\mathbf{b}\|}.
$$

因此：

- $\mathbf{a}\cdot\mathbf{b}>0$：夾角小於 $90^\circ$；
- $\mathbf{a}\cdot\mathbf{b}=0$：理論上互相垂直；
- $\mathbf{a}\cdot\mathbf{b}<0$：夾角大於 $90^\circ$。

內積也給出長度平方：

$$
\|\mathbf{a}\|^2=\mathbf{a}\cdot\mathbf{a}.
$$

若 $\hat{\mathbf{b}}$ 是單位向量，$\mathbf{a}$ 沿其方向的純量投影為

$$
s=\mathbf{a}\cdot\hat{\mathbf{b}},
$$

向量投影為

$$
\operatorname{proj}_{\hat{\mathbf{b}}}(\mathbf{a})
=(\mathbf{a}\cdot\hat{\mathbf{b}})\hat{\mathbf{b}}.
$$

與 $\hat{\mathbf{b}}$ 垂直的剩餘分量是

$$
\mathbf{a}_\perp
=\mathbf{a}-
(\mathbf{a}\cdot\hat{\mathbf{b}})\hat{\mathbf{b}}.
$$

若 $\mathbf{a}$、$\mathbf{b}$ 都是以公尺表示的位移，內積單位為平方公尺。只有兩者都是無因次單位方向時，內積才可直接解讀為 $\cos\theta$。

### 3. 正規化與零長向量

非零向量的正規化為

$$
\hat{\mathbf{a}}
=\frac{\mathbf{a}}{\|\mathbf{a}\|}.
$$

結果滿足 $\|\hat{\mathbf{a}}\|=1$。若 $\mathbf{a}=\mathbf{0}$，方向不存在，不能正規化。把零向量直接替換成某個固定方向，可能掩蓋重複頂點、退化面或錯誤輸入。

核心幾何函式通常應採取以下策略之一：

1. 拒絕零長或過短向量；
2. 回傳明確的失敗狀態；
3. 由呼叫端依應用語意提供備援方向。

本章程式採第一種策略。預設門檻只是示範值；正式場景應依資料尺度傳入 `eps_length`，不可把 `1e-12` 視為通用幾何標準。

### 4. 三維外積

外積定義為

$$
\mathbf{a}\times\mathbf{b}
=
\begin{pmatrix}
a_yb_z-a_zb_y\\
a_zb_x-a_xb_z\\
a_xb_y-a_yb_x
\end{pmatrix}.
$$

它具有反交換性：

$$
\mathbf{a}\times\mathbf{b}
=-(\mathbf{b}\times\mathbf{a}),
$$

且同時垂直於兩個輸入：

$$
\mathbf{a}\cdot(\mathbf{a}\times\mathbf{b})=0,
\qquad
\mathbf{b}\cdot(\mathbf{a}\times\mathbf{b})=0.
$$

方向由右手定則決定：右手手指由 $\mathbf{a}$ 捲向 $\mathbf{b}$，拇指方向即為 $\mathbf{a}\times\mathbf{b}$。

外積長度為

$$
\|\mathbf{a}\times\mathbf{b}\|
=\|\mathbf{a}\|\,\|\mathbf{b}\|\,|\sin\theta|.
$$

這是兩向量張成之平行四邊形的面積。若兩向量單位為公尺，外積各分量及其長度的單位皆為平方公尺。

### 5. 三角形面積、法線與繞序

令三角形頂點為 $\mathbf{p}_0,\mathbf{p}_1,\mathbf{p}_2$，定義

$$
\mathbf{e}_1=\mathbf{p}_1-\mathbf{p}_0,
\qquad
\mathbf{e}_2=\mathbf{p}_2-\mathbf{p}_0.
$$

未正規化法線為

$$
\mathbf{n}_{\mathrm{raw}}
=\mathbf{e}_1\times\mathbf{e}_2.
$$

其長度是三角形雙倍面積，因此

$$
A_\triangle
=\frac12\|\mathbf{n}_{\mathrm{raw}}\|.
$$

若 $\mathbf{n}_{\mathrm{raw}}\ne\mathbf{0}$，單位法線為

$$
\hat{\mathbf{n}}
=\frac{\mathbf{n}_{\mathrm{raw}}}
{\|\mathbf{n}_{\mathrm{raw}}\|}.
$$

交換 $\mathbf{p}_1$、$\mathbf{p}_2$ 會反轉法線：

$$
(\mathbf{p}_2-\mathbf{p}_0)\times
(\mathbf{p}_1-\mathbf{p}_0)
=-\mathbf{n}_{\mathrm{raw}}.
$$

依本書約定，從外向法線方向觀察時，三角形前面為逆時針。外積只依頂點順序產生方向，不知道何處是池內、池外或物體表面；這些語意必須由模型約定提供。

### 6. 退化與近退化三角形

精確數學中，若

$$
\|\mathbf{e}_1\times\mathbf{e}_2\|=0,
$$

則三點共線，或至少有兩點重合，三角形面積為零。此時不存在唯一表面法線。

浮點運算中還要辨認近退化狀態。一種尺度相對判定是

$$
\|\mathbf{e}_1\times\mathbf{e}_2\|
\le
\tau_{\mathrm{rel}}
\|\mathbf{e}_1\|\,\|\mathbf{e}_2\|,
$$

其中 $\tau_{\mathrm{rel}}\ge 0$ 是無因次門檻。當兩邊皆非零時，比例

$$
\rho=
\frac{\|\mathbf{e}_1\times\mathbf{e}_2\|}
{\|\mathbf{e}_1\|\,\|\mathbf{e}_2\|}
=|\sin\theta|
$$

描述兩邊有多接近平行。$\rho$ 越小，三角形越狹長，法線對頂點擾動越敏感。

程式還應另設長度門檻 $\varepsilon_L$，拒絕

$$
\|\mathbf{e}_1\|\le\varepsilon_L
\quad\text{或}\quad
\|\mathbf{e}_2\|\le\varepsilon_L.
$$

$\varepsilon_L$ 與座標具有相同單位。相對門檻不能取代長度門檻：一個邊長僅 $10^{-15}$ 公尺、形狀卻接近直角的三角形，角度品質可能良好，但對公尺級場景仍可能小到沒有工程意義。

即使使用者令 $\tau_{\mathrm{rel}}=0$，程式仍必須拒絕精確零面積，否則後續會除以零。零相對容差表示「不額外拒絕非零的近共線三角形」，而不是允許零法線。

### 7. 絕對面積容差與尺度選擇

相對容差只描述形狀，不描述三角形的絕對大小。有些流程還需要絕對面積門檻 $\varepsilon_A$：

$$
A_\triangle\le\varepsilon_A.
$$

其中 $\varepsilon_A$ 的單位是平方公尺。它適用於：

- 匯入模型時排除小於製作解析度的碎片面；
- 避免極小面造成不穩定的法線加權；
- 依合成場景的最小可見特徵清理資料。

三種門檻解決不同問題：

| 門檻 | 單位 | 用途 |
|---|---:|---|
| $\varepsilon_L$ | m | 拒絕過短邊或重合頂點 |
| $\varepsilon_A$ | $\mathrm{m}^2$ | 拒絕絕對面積太小的面 |
| $\tau_{\mathrm{rel}}$ | 無因次 | 拒絕形狀過度狹長、近共線的面 |

例如同樣是直角等腰三角形，邊長 $1$ 公尺與 $10^{-5}$ 公尺的 $\rho$ 都是 1，但後者可能低於模型製作解析度。反之，一個面積不小但長寬比極端的三角形，可能通過 $\varepsilon_A$，卻因 $\rho$ 太小而不適合後續計算。

若模型尺度約為 $L_{\mathrm{scene}}$，可先根據資料來源決定最小有意義長度 $L_{\min}$，再令 $\varepsilon_L$ 接近該尺度，而非單純取機器精度。面積門檻可依最小面片需求設定；相對門檻則依演算法對狹長面的容忍程度設定。這些都是幾何品質規則，不是養殖設備的物理安全閾值。

### 8. 平面方程

通過點 $\mathbf{p}_0$、法線為非零向量 $\mathbf{n}$ 的平面滿足

$$
\mathbf{n}\cdot(\mathbf{x}-\mathbf{p}_0)=0.
$$

展開可得

$$
\mathbf{n}\cdot\mathbf{x}+d=0,
\qquad
d=-\mathbf{n}\cdot\mathbf{p}_0.
$$

若 $\mathbf{n}=(a,b,c)^T$，則

$$
ax+by+cz+d=0.
$$

同一平面的四個係數可同乘任何非零常數，因此表示並不唯一。若 $\mathbf{n}$ 是單位法線，係數具有較直接的距離意義。

三個點可建立平面的必要條件是它們不共線。計算順序為：

1. 建立 $\mathbf{e}_1$、$\mathbf{e}_2$；
2. 檢查短邊與退化；
3. 計算並正規化外積；
4. 令 $d=-\hat{\mathbf{n}}\cdot\mathbf{p}_0$。

若退化檢查失敗，不能用任意備援法線假裝三點定義了平面。

### 9. 點到平面的距離與半空間判定

令查詢點為 $\mathbf{q}$。若 $\hat{\mathbf{n}}$ 是單位法線，帶號距離為

$$
s=\hat{\mathbf{n}}\cdot(\mathbf{q}-\mathbf{p}_0).
$$

- $s>0$：位於法線所指一側；
- $s=0$：位於平面上；
- $s<0$：位於相反一側。

無號距離為

$$
D=|s|.
$$

若使用未正規化法線，則必須除以其長度：

$$
s=
\frac{\mathbf{n}\cdot(\mathbf{q}-\mathbf{p}_0)}
{\|\mathbf{n}\|}.
$$

點在平面上的正交投影為

$$
\mathbf{q}_{\mathrm{proj}}
=\mathbf{q}-s\hat{\mathbf{n}}.
$$

浮點數中不宜直接用 `s == 0.0` 判斷「位於平面」。若應用允許距離誤差 $\varepsilon_D$，可分類為

$$
\begin{cases}
s>\varepsilon_D & \text{正側},\\
s<-\varepsilon_D & \text{負側},\\
|s|\le\varepsilon_D & \text{邊界帶}.
\end{cases}
$$

$\varepsilon_D$ 是長度量。邊界帶能避免一個幾乎在平面上的點因微小捨入誤差在正負兩側反覆切換。

---

## 逐步手算例題

### 例題一：池底三角形的面積與法線

某池底局部三角形頂點為

$$
\mathbf{p}_0=(0,0,0)^T,\quad
\mathbf{p}_1=(2,0,0)^T,\quad
\mathbf{p}_2=(0,0,3)^T,
$$

單位為公尺。兩條邊是

$$
\mathbf{e}_1=(2,0,0)^T,
\qquad
\mathbf{e}_2=(0,0,3)^T.
$$

外積為

$$
\mathbf{e}_1\times\mathbf{e}_2
=
\begin{pmatrix}
0\\-6\\0
\end{pmatrix}
=(0,-6,0)^T\ \mathrm{m}^2.
$$

所以面積為

$$
A_\triangle
=\frac12\sqrt{(-6)^2}
=3\ \mathrm{m}^2.
$$

單位法線是

$$
\hat{\mathbf{n}}=(0,-1,0)^T.
$$

此法線朝下。若池底外向法線應朝上，須交換後兩點，改用順序 $(\mathbf{p}_0,\mathbf{p}_2,\mathbf{p}_1)$。面積不受繞序影響，但法線方向會反轉。

### 例題二：點到斜平面的距離

平面通過

$$
\mathbf{p}_0=(0,1,0)^T
$$

且法線為

$$
\mathbf{n}=(0,2,2)^T.
$$

查詢點是

$$
\mathbf{q}=(0,4,1)^T.
$$

法線長度為

$$
\|\mathbf{n}\|=2\sqrt2,
$$

故單位法線為

$$
\hat{\mathbf{n}}
=
\left(0,\frac1{\sqrt2},\frac1{\sqrt2}\right)^T.
$$

位移是

$$
\mathbf{q}-\mathbf{p}_0=(0,3,1)^T.
$$

帶號距離為

$$
s
=\hat{\mathbf{n}}\cdot(\mathbf{q}-\mathbf{p}_0)
=\frac3{\sqrt2}+\frac1{\sqrt2}
=2\sqrt2\ \mathrm{m}.
$$

因 $s>0$，查詢點位於法線所指一側。投影點為

$$
\mathbf{q}_{\mathrm{proj}}
=\mathbf{q}-s\hat{\mathbf{n}}
=(0,4,1)-(0,2,2)
=(0,2,-1)^T.
$$

檢查：

$$
\mathbf{n}\cdot
(\mathbf{q}_{\mathrm{proj}}-\mathbf{p}_0)
=(0,2,2)\cdot(0,1,-1)=0.
$$

所以投影點確實位於平面上。

### 例題三：形狀容差與絕對面積

令

$$
\mathbf{e}_1=(1000,0,0)^T,\qquad
\mathbf{e}_2=(1000,0.001,0)^T.
$$

外積為

$$
\mathbf{e}_1\times\mathbf{e}_2=(0,0,1)^T,
$$

雙倍面積為 $1\ \mathrm{m}^2$，三角形面積為 $0.5\ \mathrm{m}^2$。其形狀比例為

$$
\rho=
\frac{1}{1000\sqrt{1000^2+0.001^2}}.
$$

因

$$
\sqrt{1000^2+0.001^2}
=\sqrt{1000000.000001}
\approx1000.0000000005,
$$

所以

$$
\rho
\approx9.999999999995\times10^{-7}.
$$

也就是說，$\rho$ 的量級約為 $10^{-6}$，但並非精確等於 $10^{-6}$。面積並不極小，形狀卻非常狹長。若 $\tau_{\mathrm{rel}}=10^{-5}$，它會被判為近退化；若 $\tau_{\mathrm{rel}}=10^{-8}$，則通過形狀檢查。

相反地，邊向量

$$
\mathbf{u}=(10^{-5},0,0)^T,\qquad
\mathbf{v}=(0,10^{-5},0)^T
$$

形成完好的直角，但面積僅

$$
A=\frac12\times10^{-10}
=5\times10^{-11}\ \mathrm{m}^2.
$$

它的 $\rho=1$，不近共線；是否接受，應由長度或絕對面積規則決定。這兩例說明相對與絕對門檻不可互相取代。

---

## 實作與程式

以下程式完整定義本章函式。輸入統一轉成 `float64`、shape `(3,)` 的 NumPy 陣列。`triangle_geometry` 先明確拒絕零面積，再套用相對門檻，因此即使 `rel_sine_tol=0`，也不會接受零法線或發生除以零。

```python
import numpy as np


def vec3(value, name="vector"):
    """轉成 shape (3,) 的有限 float64 向量。"""
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(
            f"{name} 必須是 shape (3,)，目前為 {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 含有 NaN 或無限值")
    return result


def dot(a, b):
    a = vec3(a, "a")
    b = vec3(b, "b")
    return float(np.dot(a, b))


def cross(a, b):
    a = vec3(a, "a")
    b = vec3(b, "b")
    return np.cross(a, b)


def length(v):
    v = vec3(v, "v")
    return float(np.linalg.norm(v))


def normalize(v, eps_length=1e-12):
    """
    正規化三維向量。

    eps_length 與 v 的單位相同；若 v 是無因次方向，
    eps_length 也無因次。正式應用應依資料尺度設定。
    """
    v = vec3(v, "v")
    if not np.isfinite(eps_length) or eps_length < 0.0:
        raise ValueError("eps_length 必須是有限非負數")

    magnitude = length(v)
    if magnitude <= eps_length:
        raise ValueError("無法正規化零長或過短向量")
    return v / magnitude


def triangle_geometry(
    p0,
    p1,
    p2,
    eps_length=1e-12,
    rel_sine_tol=1e-10,
    eps_area=0.0,
):
    """
    回傳三角形面積、單位法線、未正規化法線與形狀比例。

    eps_length：長度門檻，單位與座標相同。
    rel_sine_tol：無因次近共線門檻，可為 0。
    eps_area：面積門檻，單位為座標單位的平方，可為 0。
    """
    p0 = vec3(p0, "p0")
    p1 = vec3(p1, "p1")
    p2 = vec3(p2, "p2")

    tolerances = (eps_length, rel_sine_tol, eps_area)
    if not all(np.isfinite(x) and x >= 0.0 for x in tolerances):
        raise ValueError("所有容差必須是有限非負數")

    e1 = p1 - p0
    e2 = p2 - p0
    l1 = length(e1)
    l2 = length(e2)

    if l1 <= eps_length or l2 <= eps_length:
        raise ValueError("三角形含有重合或過近的頂點")

    raw_normal = cross(e1, e2)
    double_area = length(raw_normal)

    # 即使 rel_sine_tol == 0，也必須拒絕精確零面積。
    if double_area == 0.0:
        raise ValueError("三角形精確退化，無法定義法線")

    sine_ratio = double_area / (l1 * l2)
    if sine_ratio <= rel_sine_tol:
        raise ValueError("三角形近共線，法線不可靠")

    area = 0.5 * double_area
    if area <= eps_area:
        raise ValueError("三角形面積低於絕對面積門檻")

    return {
        "area": area,
        "normal": raw_normal / double_area,
        "raw_normal": raw_normal,
        "sine_ratio": sine_ratio,
    }


def signed_point_plane_distance(
    point,
    plane_point,
    plane_normal,
    eps_normal=1e-12,
):
    """
    計算點到平面的帶號距離。
    point 與 plane_point 若以公尺表示，結果也是公尺。
    """
    point = vec3(point, "point")
    plane_point = vec3(plane_point, "plane_point")
    unit_normal = normalize(plane_normal, eps_normal)
    return dot(unit_normal, point - plane_point)


def classify_point_to_plane(
    point,
    plane_point,
    plane_normal,
    eps_distance,
):
    """回傳 'positive'、'negative' 或 'boundary'。"""
    if not np.isfinite(eps_distance) or eps_distance < 0.0:
        raise ValueError("eps_distance 必須是有限非負數")

    distance = signed_point_plane_distance(
        point, plane_point, plane_normal
    )
    if distance > eps_distance:
        return "positive"
    if distance < -eps_distance:
        return "negative"
    return "boundary"


def project_point_to_plane(
    point,
    plane_point,
    plane_normal,
    eps_normal=1e-12,
):
    point = vec3(point, "point")
    plane_point = vec3(plane_point, "plane_point")
    unit_normal = normalize(plane_normal, eps_normal)
    distance = dot(unit_normal, point - plane_point)
    return point - distance * unit_normal


def main():
    print("dot =", dot([1, 2, 3], [4, -1, 2]))
    print("cross =", cross([1, 0, 0], [0, 1, 0]))
    print("normalize =", normalize([0, 3, 4]))

    tri = triangle_geometry(
        [0, 0, 0],
        [2, 0, 0],
        [0, 0, 3],
    )
    print("area =", tri["area"])
    print("normal =", tri["normal"])
    print("sine ratio =", tri["sine_ratio"])

    distance = signed_point_plane_distance(
        point=[0, 4, 1],
        plane_point=[0, 1, 0],
        plane_normal=[0, 2, 2],
    )
    print("signed distance =", distance)

    projected = project_point_to_plane(
        point=[0, 4, 1],
        plane_point=[0, 1, 0],
        plane_normal=[0, 2, 2],
    )
    print("projected =", projected)

    for label, action in [
        ("zero vector", lambda: normalize([0, 0, 0])),
        (
            "degenerate triangle",
            lambda: triangle_geometry(
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                rel_sine_tol=0.0,
            ),
        ),
    ]:
        try:
            action()
        except ValueError as error:
            print(label + ":", error)


if __name__ == "__main__":
    main()
```

這裡使用 `double_area == 0.0` 並不是拿精確比較取代容差，而是建立不可跨越的除零防線。非零但品質差的三角形仍由 `rel_sine_tol`、`eps_length` 與 `eps_area` 分別處理。

這份入門實作假設座標尺度適中，使 `np.linalg.norm`、外積及 `l1 * l2` 不會溢位或下溢。若座標接近浮點格式的極端範圍，單靠 epsilon 並不足以保證穩健；應先將局部幾何縮放到合理範圍，或採用經尺度化的範數與謂詞演算法。公尺級池體與一般合成模型不應故意使用接近 `float64` 上下限的座標。

---

## 測試與預期結果

上列程式的**預期**重點如下；浮點顯示格式可能略有差異。

```text
dot = 8.0
cross = [0. 0. 1.]
normalize = [0.  0.6 0.8]
area = 3.0
normal = [ 0. -1.  0.]
sine ratio = 1.0
signed distance = 2.828427124746...
projected = [ 0.  2. -1.]
zero vector: 無法正規化零長或過短向量
degenerate triangle: 三角形精確退化，無法定義法線
```

可加入以下斷言。比較容差明確寫出，以免把 NumPy 預設值誤認為場景規格。預期會失敗的呼叫採 `try`、`except`、`else` 結構：只有收到指定的 `ValueError` 才算通過；沒有拋出例外便在 `else` 中明確使測試失敗。

```python
def run_assertions():
    rtol = 1e-12
    atol = 1e-12

    assert np.isclose(
        dot([1, 2, 3], [4, -1, 2]),
        8.0,
        rtol=rtol,
        atol=atol,
    )

    assert np.allclose(
        cross([1, 0, 0], [0, 1, 0]),
        [0, 0, 1],
        rtol=rtol,
        atol=atol,
    )

    assert np.allclose(
        normalize([0, 3, 4]),
        [0, 0.6, 0.8],
        rtol=rtol,
        atol=atol,
    )

    tri = triangle_geometry(
        [0, 0, 0], [2, 0, 0], [0, 0, 3]
    )
    assert np.isclose(
        tri["area"], 3.0, rtol=rtol, atol=atol
    )
    assert np.allclose(
        tri["normal"], [0, -1, 0],
        rtol=rtol, atol=atol
    )

    d = signed_point_plane_distance(
        [0, 4, 1], [0, 1, 0], [0, 2, 2]
    )
    assert np.isclose(
        d, 2.0 * np.sqrt(2.0),
        rtol=rtol, atol=atol
    )

    try:
        normalize([0, 0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("零向量測試應該失敗")

    # 零相對容差仍不得接受精確共線三角形。
    try:
        triangle_geometry(
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            rel_sine_tol=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("精確共線三角形應該失敗")

    # 錯誤維度必須被拒絕。
    try:
        dot([[1, 2, 3]], [1, 2, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("shape (1, 3) 應該失敗")

    # 交換繞序只反轉法線，不改變面積。
    reversed_tri = triangle_geometry(
        [0, 0, 0], [0, 0, 3], [2, 0, 0]
    )
    assert np.isclose(
        reversed_tri["area"], tri["area"],
        rtol=rtol, atol=atol
    )
    assert np.allclose(
        reversed_tri["normal"], -tri["normal"],
        rtol=rtol, atol=atol
    )

    # 法線縮放不應改變幾何距離。
    d1 = signed_point_plane_distance(
        [0, 3, 0], [0, 1, 0], [0, 1, 0]
    )
    d2 = signed_point_plane_distance(
        [0, 3, 0], [0, 1, 0], [0, 10, 0]
    )
    assert np.isclose(d1, d2, rtol=rtol, atol=atol)
```

測試至少應涵蓋正常案例、符號反轉、零長向量、精確退化、近退化及容差邊界。只測一個成功案例，無法證明錯誤輸入會被安全拒絕。

---

## 除錯與常見陷阱

### 1. 外積順序反了

`cross(e1, e2)` 與 `cross(e2, e1)` 大小相同、方向相反。若整個池壁朝內，應優先檢查頂點繞序與外積順序。

### 2. 把未正規化法線直接當距離

下式通常不是幾何距離：

$$
\mathbf{n}\cdot(\mathbf{q}-\mathbf{p}_0).
$$

除非 $\|\mathbf{n}\|=1$，否則必須除以 $\|\mathbf{n}\|$。法線放大十倍時，真實距離不應跟著變成十倍。

### 3. 對零向量正規化

重複頂點、共線三角形及相同位置相減都可能產生零向量。不要讓 `NaN` 流入後續著色、裁切或求交；應在幾何資料進入流程時拒絕。

### 4. 把零容差理解成允許退化

`rel_sine_tol=0` 只能表示不拒絕非零的近共線三角形。精確零面積仍然沒有法線，必須單獨拒絕。否則 `raw_normal / double_area` 會除以零。

### 5. 容差單位不一致

`length(e1)` 是長度，`length(cross(e1, e2))` 是面積，不能直接與同一個帶單位常數比較。應分別使用長度、面積與無因次形狀門檻。

### 6. 只檢查從同一頂點出發的兩條邊

若只檢查 $\|\mathbf{p}_1-\mathbf{p}_0\|$ 與 $\|\mathbf{p}_2-\mathbf{p}_0\|$，第三條邊 $\|\mathbf{p}_2-\mathbf{p}_1\|$ 仍可能非常短。外積通常會揭露面積問題，但若應用要完整診斷重複頂點，應檢查三條邊並回報具體頂點對。

### 7. 直接用 `acos` 判角度

浮點誤差可能使餘弦略超出 $[-1,1]$。若確實需要角度，先做：

```python
cos_theta = np.clip(cos_theta, -1.0, 1.0)
theta = np.arccos(cos_theta)
```

若只需判斷正面、背面或近垂直，通常直接比較內積即可。

### 8. 混淆位置與方向

平面法線與邊向量是方向，不應受平移影響。後續齊次座標會把位置寫成 $w=1$，方向寫成 $w=0$；本章的幾何語意應先保持清楚。

### 9. 修正法線卻不修正網格繞序

若只將法線乘以 $-1$，但不反轉三角形索引，後續背面剔除、陰影與相鄰面拓撲仍可能互相矛盾。資料問題應優先在網格層修正，不能只改顯示結果。

### 10. 將容差當成浮點穩健性的完整解法

合理容差能表達應用接受範圍，但不能防止所有溢位、下溢或消去誤差。例如極大向量的外積可能先溢位，程式尚未比較容差便已得到無限值。實務上應採合理單位與局部座標，避免把模型放在遠超場景需求的數值範圍。

---

## 養殖數位分身案例

考慮用兩個三角形表示矩形水面，高度為 $y=1.5$ 公尺，希望法線朝上：

$$
\hat{\mathbf{n}}=(0,1,0)^T.
$$

四個角點為

$$
\mathbf{p}_{00}=(-2,1.5,-3)^T,\quad
\mathbf{p}_{10}=(2,1.5,-3)^T,
$$

$$
\mathbf{p}_{01}=(-2,1.5,3)^T,\quad
\mathbf{p}_{11}=(2,1.5,3)^T.
$$

第一個三角形使用

$$
(\mathbf{p}_{00},\mathbf{p}_{01},\mathbf{p}_{10}).
$$

兩條邊為

$$
\mathbf{e}_1=(0,0,6)^T,\qquad
\mathbf{e}_2=(4,0,0)^T,
$$

所以

$$
\mathbf{e}_1\times\mathbf{e}_2=(0,24,0)^T.
$$

法線朝 $+Y$。第二個三角形使用

$$
(\mathbf{p}_{10},\mathbf{p}_{01},\mathbf{p}_{11}),
$$

其法線也朝 $+Y$。每個三角形面積為 $12\ \mathrm{m}^2$，合計水面面積為 $24\ \mathrm{m}^2$。

假設一條魚的合成參考位置為

$$
\mathbf{f}=(0,1.1,0)^T.
$$

以水面點與上向法線計算：

$$
s=(0,1,0)\cdot(\mathbf{f}-\mathbf{p}_{00})
=1.1-1.5=-0.4\ \mathrm{m}.
$$

該位置位於水面下方 $0.4$ 公尺。若分類門檻為 $\varepsilon_D=0.01$ 公尺，則：

- $s>0.01$：水面上方；
- $s<-0.01$：水面下方；
- $|s|\le0.01$：水面邊界帶。

網格匯入流程還可逐面執行：

1. 檢查三個索引合法且頂點有限；
2. 計算三條邊並找出重複頂點；
3. 檢查面積與形狀比例；
4. 計算法線；
5. 與預期外向方向做內積；
6. 若內積為負，反轉該面的頂點繞序；
7. 重新計算法線並留下修正紀錄。

對水平水面，可使用預期方向 $(0,1,0)^T$。若

$$
\hat{\mathbf{n}}_{\mathrm{face}}\cdot(0,1,0)<0,
$$

表示該面朝下。但對曲面魚體，不能假設所有法線都朝 $+Y$；應使用模型中心到面中心的方向、封閉網格方向規則或拓撲一致性判定。

這些結果只描述合成幾何位置與網格品質，不代表魚隻健康、水質安全或真實量測。波浪水面也不能由單一水平平面完整表示。

---

## 習題

### 習題 1：手算

給定

$$
\mathbf{a}=(1,2,-2)^T,\qquad
\mathbf{b}=(2,0,1)^T.
$$

1. 計算 $\mathbf{a}\cdot\mathbf{b}$。
2. 計算 $\mathbf{a}\times\mathbf{b}$。
3. 驗證外積垂直於兩輸入向量。
4. 求兩向量張成之三角形面積。

### 習題 2：程式測試

補上測試，驗證：

1. `cross(a, b) == -cross(b, a)`；
2. `normalize([3, 0, 4])` 長度接近 1；
3. 法線由 `[0, 1, 0]` 改成 `[0, 10, 0]` 時距離不變；
4. shape `(1, 3)` 會被拒絕；
5. `rel_sine_tol=0` 時，精確共線三角形仍會被拒絕。

### 習題 3：反例與除錯

某程式以

```python
raw = np.cross(p1 - p0, p2 - p0)
if np.linalg.norm(raw) < 1e-6:
    degenerate = True
```

判斷退化。指出至少兩個問題，並給出一個「形狀相同但整體縮放後判定改變」的反例。再說明何時應加入絕對面積門檻。

### 習題 4：整合應用

某垂直池壁由三點構成：

$$
\mathbf{p}_0=(2,0,-1)^T,\quad
\mathbf{p}_1=(2,2,-1)^T,\quad
\mathbf{p}_2=(2,0,3)^T.
$$

1. 求面積與單位法線。
2. 求 $\mathbf{q}=(1.25,1,0)^T$ 到平面的帶號距離。
3. 若希望法線朝池內的 $-X$，目前繞序是否正確？
4. 求 $\mathbf{q}$ 的平面投影。

### 習題 5：容差設計

某合成場景以公尺為單位，最短有意義邊長為 $0.5$ 毫米，最小保留面積為 $0.2$ 平方毫米。請把這兩個值換成公尺與平方公尺，並寫出適合傳給 `triangle_geometry` 的 `eps_length`、`eps_area`。說明為何仍需另選 `rel_sine_tol`。

---

## 習題解答

### 習題 1 解答

內積：

$$
\mathbf{a}\cdot\mathbf{b}
=1(2)+2(0)+(-2)(1)=0.
$$

外積：

$$
\mathbf{a}\times\mathbf{b}
=
\begin{pmatrix}
2\\-5\\-4
\end{pmatrix}.
$$

驗證：

$$
\mathbf{a}\cdot(\mathbf{a}\times\mathbf{b})
=2-10+8=0,
$$

$$
\mathbf{b}\cdot(\mathbf{a}\times\mathbf{b})
=4-4=0.
$$

外積長度為

$$
\sqrt{2^2+(-5)^2+(-4)^2}
=3\sqrt5.
$$

三角形面積為

$$
A_\triangle=\frac{3\sqrt5}{2}.
$$

### 習題 2 解答

以下寫法不依賴額外測試框架。預期失敗的函式呼叫放在 `try` 區塊中；若沒有收到 `ValueError`，便由 `else` 明確使測試失敗。

```python
def exercise_tests():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-2.0, 1.0, 4.0])

    assert np.allclose(cross(a, b), -cross(b, a))

    unit = normalize([3, 0, 4])
    assert np.isclose(length(unit), 1.0)

    d1 = signed_point_plane_distance(
        [0, 3, 0], [0, 1, 0], [0, 1, 0]
    )
    d2 = signed_point_plane_distance(
        [0, 3, 0], [0, 1, 0], [0, 10, 0]
    )
    assert np.isclose(d1, 2.0)
    assert np.isclose(d1, d2)

    try:
        dot([[1, 2, 3]], [1, 2, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("shape (1, 3) 應被拒絕")

    try:
        triangle_geometry(
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            rel_sine_tol=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("共線三角形應被拒絕")
```

最後一項是容差邊界測試：零相對門檻仍不能使零面積法線合法化。

### 習題 3 解答

第一，`1e-6` 的單位不明。外積長度是雙倍面積；若座標為公尺，其單位是平方公尺。

第二，絕對面積門檻隨縮放改變判定。令

$$
\mathbf{p}_0=(0,0,0),\quad
\mathbf{p}_1=(1,0,0),\quad
\mathbf{p}_2=(0,1,0).
$$

雙倍面積為 1。將全部座標乘上 $10^{-4}$ 後，形狀仍為直角等腰三角形，但雙倍面積變成 $10^{-8}$，會被原程式判為退化。

改進方式是：

1. 用 `eps_length` 檢查短邊；
2. 用無因次比例

$$
\frac{\|\mathbf{e}_1\times\mathbf{e}_2\|}
{\|\mathbf{e}_1\|\,\|\mathbf{e}_2\|}
$$

檢查近共線；
3. 先明確拒絕零面積，避免除以零；
4. 若應用確實不保留極小碎片面，再另設帶面積單位的 `eps_area`。

絕對面積門檻適合表示製作解析度或資料清理需求，但不能單獨代表形狀品質。

### 習題 4 解答

兩條邊為

$$
\mathbf{e}_1=(0,2,0)^T,\qquad
\mathbf{e}_2=(0,0,4)^T.
$$

外積：

$$
\mathbf{e}_1\times\mathbf{e}_2=(8,0,0)^T.
$$

所以

$$
A_\triangle=4\ \mathrm{m}^2,
\qquad
\hat{\mathbf{n}}=(1,0,0)^T.
$$

查詢點位移為

$$
\mathbf{q}-\mathbf{p}_0=(-0.75,1,1)^T.
$$

帶號距離：

$$
s=(1,0,0)\cdot(-0.75,1,1)
=-0.75\ \mathrm{m}.
$$

目前法線朝 $+X$，不符合所需的 $-X$，應交換 $\mathbf{p}_1$、$\mathbf{p}_2$。

投影點為

$$
\mathbf{q}_{\mathrm{proj}}
=\mathbf{q}-s\hat{\mathbf{n}}
=(1.25,1,0)-(-0.75)(1,0,0)
=(2,1,0)^T.
$$

### 習題 5 解答

因

$$
1\ \mathrm{mm}=10^{-3}\ \mathrm{m},
$$

所以

$$
0.5\ \mathrm{mm}=5\times10^{-4}\ \mathrm{m}.
$$

又因

$$
1\ \mathrm{mm}^2=10^{-6}\ \mathrm{m}^2,
$$

所以

$$
0.2\ \mathrm{mm}^2=2\times10^{-7}\ \mathrm{m}^2.
$$

可設定：

```python
eps_length = 5e-4
eps_area = 2e-7
```

邊長與面積門檻仍不能判斷三角形是否過度狹長。例如一個面積高於 `eps_area` 的細長三角形，法線仍可能對頂點誤差十分敏感，因此還要依演算法需求選擇無因次的 `rel_sine_tol`。

---

## 本章小結

內積描述向量的方向關係，可用於長度、投影、垂直與正反側判定。外積產生垂直於兩輸入向量的方向，其長度等於平行四邊形面積，因此能建立三角形面積與法線。

三角形法線依賴頂點繞序；交換兩個頂點會反轉法線。零面積三角形沒有合法法線，即使相對容差設為零也必須拒絕。近退化幾何則應分別使用長度門檻、面積門檻與無因次形狀門檻，不能用單一 epsilon 混合處理。

平面可寫成

$$
\mathbf{n}\cdot(\mathbf{x}-\mathbf{p}_0)=0.
$$

使用單位法線後，點代入所得值就是帶號距離，也可用來建立含容差的半空間分類。這些工具將成為後續變換、網格、光柵化、法線處理與射線求交的共同基礎。

---

## 參考來源

- [G1] *Physically Based Rendering, 4th ed.*：Transformations  
  <https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations>
- [G4] *Ray Tracing in One Weekend*  
  <https://raytracing.github.io/books/RayTracingInOneWeekend.html>
- [G7] NumPy 線性代數參考  
  <https://numpy.org/doc/stable/reference/routines.linalg.html>

以上來源供讀者回查向量、幾何與數值 API 背景；本章公式、案例與程式依本書座標、單位及錯誤處理約定整理。