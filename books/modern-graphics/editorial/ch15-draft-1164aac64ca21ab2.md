# 第15章　局部照明與BRDF基礎

## 學習目標與先備知識

讀完本章，你將能：

- 區分光源、表面與相機方向，計算 Lambert 漫反射。
- 解釋餘弦因子、立體角，以及輻射度量中常用量的單位。
- 以局部照明近似計算一個球面，並測試背面光源及輸出尺度。
- 說明 Phong 是經驗性著色模型，並區分它與具能量約束的物理BRDF。

先備知識為向量、內積、矩陣與基本Python。角度以弧度表示；長度用公尺，時間用秒。顏色運算採線性RGB；顯示前才轉為sRGB。本章的場景與光源皆為合成資料，不代表養殖池的實測光照。

## 問題與直覺

光柵化決定哪些表面覆蓋哪些像素，但它不會自行判斷表面看起來有多亮。局部照明模型為每個可見表面點，根據表面材質、點的法線、光源方向及觀察方向估算顏色。

以魚體上一個點為例：若光由表面正前方照來，單位面積接收的光較多；若光幾乎沿表面掠過，接收量較少；若光來自表面背面，理想的不透明漫反射面不會直接受光。這個幾何效應由法線與光線方向的內積表達。

然而，「像素亮度」與「光的物理量」不是同一概念。若只為了做出像塑膠或金屬的高光而調參，模型可以產生好看的圖像，卻不一定符合能量守恆。理解Lambert模型與BRDF，能讓我們知道近似的範圍，以及什麼時候需要更完整的光傳輸模型。

## 數學與幾何推導

### 方向與Lambert餘弦

在表面點 \(p\) 定義：

- \(\mathbf n\)：朝向物體外部的單位表面法線。
- \(\mathbf l\)：由表面點指向光源的單位方向。
- \(\mathbf v\)：由表面點指向相機的單位方向。
- \(\boldsymbol\omega_i\)：入射光的傳播方向，指向表面；因此 \(\boldsymbol\omega_i=-\mathbf l\)。
- \(\boldsymbol\omega_o=\mathbf v\)：離開表面、朝觀察者的方向。

漫反射表面的直接受光量與

$$
\max(0,\mathbf n\cdot\mathbf l)
$$

成正比。當光線與法線夾角為 \(\theta\)，且 \(0\leq\theta\leq\pi/2\)，內積就是 \(\cos\theta\)。這個餘弦因子不只是著色慣例：斜射到平面上的平行光，會分散到較大的表面面積，因此單位面積接收的能量下降。

令 \(E\) 為表面照度，單位是 \(\mathrm{W/m^2}\)；令 \(\rho\) 為無因次漫反射反照率，對線性RGB可寫成三分量 \(\boldsymbol\rho\)。理想Lambert表面的出射輻亮度為

$$
\mathbf L_o=\frac{\boldsymbol\rho}{\pi}E.
$$

若照度來自一個方向的平行光，該方向上的入射輻亮度為 \(\mathbf L_i\)，則

$$
\mathbf L_o=
\frac{\boldsymbol\rho}{\pi}
\mathbf L_i
\max(0,\mathbf n\cdot\mathbf l).
$$

此處輻亮度的單位為 \(\mathrm{W/(m^2\,sr)}\)，\(sr\) 是立體角單位。係數 \(1/\pi\) 讓理想漫反射BRDF滿足能量尺度：若所有半球方向都均勻射入，反射出去的總能量不會超過入射能量乘以 \(\rho\)。

圖學程式也常把光源顏色寫成 \(\mathbf C_L\)，直接使用

$$
\mathbf C_{\mathrm{diffuse}}
=\boldsymbol\rho\odot\mathbf C_L
\max(0,\mathbf n\cdot\mathbf l).
$$

這是方便的顏色模型；\(\mathbf C_L\) 是否代表照度、輻亮度或已調整過的藝術控制值，必須由應用自行約定。除非校準了光源、材質與顯示轉換，不應把這種RGB值宣稱為實際的輻射功率或相機量測值。

### 立體角與BRDF

平面角描述二維方向差；立體角描述從一點看出去的三維方向範圍，單位為球面度 \(sr\)。整個球面的立體角是 \(4\pi\,sr\)，朝外的半球是 \(2\pi\,sr\)。對小面積 \(dA\)，若它與視線夾角為 \(\theta\)，距離為 \(r\)，其張角近似

$$
d\omega=\frac{\cos\theta\,dA}{r^2}.
$$

立體角使我們能把來自不同方向的光加總。以表面反射為例，出射輻亮度 \(\mathbf L_o(p,\boldsymbol\omega_o)\) 由半球內各入射方向貢獻：

$$
\mathbf L_o(p,\boldsymbol\omega_o)=
\int_{\Omega^+}
f_r(p,\boldsymbol\omega_i,\boldsymbol\omega_o)
\mathbf L_i(p,\boldsymbol\omega_i)
(\mathbf n\cdot\boldsymbol\omega_i)
\,d\boldsymbol\omega_i.
$$

\(\Omega^+\) 是表面外側半球；\(f_r\) 是雙向反射分布函數（BRDF），描述入射輻亮度如何轉成指定觀察方向的出射輻亮度。BRDF的單位是 \(sr^{-1}\)。由於 \(\boldsymbol\omega_i\) 指向表面，表面外側入射方向滿足 \(\mathbf n\cdot\boldsymbol\omega_i>0\)。

Lambert BRDF不依賴入射或觀察方向：

$$
f_r=\frac{\rho}{\pi}.
$$

因此它在半球積分中留下餘弦權重，但不會形成方向集中的高光。真實魚皮可能同時呈現散射、鏡面反射、濕潤表面反光及尺度各異的細節；單一Lambert項通常只適合簡化的漫反射近似。

### Phong與物理BRDF的差別

經典Phong模型以反射方向 \(\mathbf r\) 和觀察方向 \(\mathbf v\) 的對齊程度塑造高光：

$$
I_{\mathrm{Phong}}
=k_d\max(0,\mathbf n\cdot\mathbf l)
+k_s\max(0,\mathbf r\cdot\mathbf v)^s.
$$

\(k_d,k_s\) 是調色用係數，\(s\) 控制高光集中程度。常見的反射方向可由入射傳播方向 \(-\mathbf l\) 得到

$$
\mathbf r=-\mathbf l-2\mathbf n\bigl(\mathbf n\cdot(-\mathbf l)\bigr).
$$

Phong容易實作、便於藝術調整，但常見形式未必符合能量守恆，也不保證在不同光源強度、材質參數與觀察角度之間一致。規範化的Phong變體可以改善尺度，但仍是經驗性模型。物理微表面模型則試圖描述由許多微小鏡面構成的表面，並用法線分布、遮蔽遮蔽項與Fresnel效應建模；其參數較有物理意義，但仍依賴模型假設與參數映射。本章只建立BRDF的基本概念，不把任一著色公式等同於完整真實光學。

## 逐步手算例題

### 例一：傾斜表面的Lambert明暗

設表面法線與光源方向皆為單位向量：

$$
\mathbf n=(0,1,0), \qquad
\mathbf l=(0,0.6,0.8).
$$

若 \(\rho=0.7\)，並採用照度 \(E=10\,\mathrm{W/m^2}\)，依Lambert式計算出射輻亮度：

1. 先算內積：

   $$
   \mathbf n\cdot\mathbf l=0.6.
   $$

2. 點積為正，因此不截成零。照度為

   $$
   E_{\mathrm{surface}}=10\times0.6
   =6\,\mathrm{W/m^2}.
   $$

3. 出射輻亮度為

   $$
   L_o=\frac{0.7}{\pi}\times6
   =\frac{4.2}{\pi}
   \approx1.337\,\mathrm{W/(m^2\,sr)}.
   $$

若同一光源改成 \(\mathbf l=(0,-0.6,0.8)\)，則 \(\mathbf n\cdot\mathbf l=-0.6\)，Lambert直接受光項截成零。若要顯示環境反射或背面補光，必須另加一項；那不是背面光源突然變成正面光源。

### 例二：依球面法線計算三個點

考慮單位球面上三個法線，皆面向外：

$$
\mathbf n_1=(0,1,0),\quad
\mathbf n_2=\left(\frac{\sqrt2}{2},\frac{\sqrt2}{2},0\right),\quad
\mathbf n_3=(0,-1,0).
$$

令平行光方向為 \(\mathbf l=(0,1,0)\)，線性光色為 \(\mathbf C_L=(2,1,0.5)\)，漫反射反照率為 \(\boldsymbol\rho=(0.5,0.25,0.8)\)。使用簡化RGB模型：

$$
\mathbf C=\boldsymbol\rho\odot\mathbf C_L
\max(0,\mathbf n\cdot\mathbf l).
$$

先計算 \(\boldsymbol\rho\odot\mathbf C_L=(1,0.25,0.4)\)。

- 對 \(\mathbf n_1\)，點積為 \(1\)，故 \(\mathbf C_1=(1,0.25,0.4)\)。
- 對 \(\mathbf n_2\)，點積為 \(\sqrt2/2\approx0.7071\)，故 \(\mathbf C_2\approx(0.7071,0.1768,0.2828)\)。
- 對 \(\mathbf n_3\)，點積為 \(-1\)，截成 \(0\)，故直接受光 \(\mathbf C_3=(0,0,0)\)。

這些是線性RGB，不是顯示器上的sRGB碼值。若輸出時直接把大於1的分量截斷為1，會損失高光或強光資訊；本章程式只為了產生可查看的示意圖而限制輸出範圍，並在測試中保留原始線性計算。

## 實作與程式

以下程式以純Python建立 \(128\times128\) PPM影像。每個像素沿相機射線與單位球求交；球面點即為位置向量，故其外法線為該點的單位向量。可見部分以正面最近交點為準。光源是遠處平行光，使用簡化Lambert線性RGB，再限制至PPM可表示的範圍。

PPM的P3格式使用文字列出整數RGB，每個通道範圍為0至255。本程式最後直接以線性值量化到整數，不作sRGB轉換，因此它適合檢查幾何和Lambert計算，不是色彩管理正確的顯示輸出。若要取得通常顯示器使用的sRGB編碼，需先逐通道做線性RGB至sRGB的分段轉換。

```python
import math

W, H = 128, 128
LIGHT = (0.35, 0.80, 0.48)  # 程式稍後正規化
ALBEDO = (0.70, 0.32, 0.12)
AMBIENT = 0.08               # 無因次的簡化補光係數


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def normalize(v):
    length = math.sqrt(dot(v, v))
    if length == 0.0:
        raise ValueError("不可正規化零向量")
    return tuple(x / length for x in v)


def ray_sphere(origin, direction):
    # 單位球：|origin + t * direction|^2 = 1
    a = dot(direction, direction)
    b = 2.0 * dot(origin, direction)
    c = dot(origin, origin) - 1.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    positive = [t for t in (t0, t1) if t > 0.0]
    return min(positive) if positive else None


def lambert(albedo, normal, light, ambient=0.0):
    ndotl = max(0.0, dot(normal, light))
    return tuple(
        c * (ambient + ndotl) for c in albedo
    )


def to_byte(x):
    # 示意輸出：限制高於1的線性值，直接量化，不做sRGB轉換
    return max(0, min(255, round(255 * x)))


light = normalize(LIGHT)
pixels = []

for y in range(H):
    for x in range(W):
        # 像素原點左上，中心取樣；相機看向 -Z。
        sx = 2.0 * (x + 0.5) / W - 1.0
        sy = 1.0 - 2.0 * (y + 0.5) / H
        origin = (0.0, 0.0, 3.0)
        direction = normalize((sx, sy, -2.0))

        t = ray_sphere(origin, direction)
        if t is None:
            color = (0.025, 0.045, 0.07)
        else:
            point = tuple(origin[i] + t * direction[i] for i in range(3))
            normal = normalize(point)
            color = lambert(ALBEDO, normal, light, AMBIENT)

        pixels.append(tuple(to_byte(c) for c in color))

with open("lambert_sphere.ppm", "w", encoding="ascii") as f:
    f.write(f"P3\n{W} {H}\n255\n")
    for i in range(0, len(pixels), W):
        row = pixels[i:i + W]
        f.write(" ".join(f"{r} {g} {b}" for r, g, b in row) + "\n")

# 不依賴輸出影像的數值測試
n = (0.0, 1.0, 0.0)
assert lambert((1.0, 1.0, 1.0), n, (0.0, 1.0, 0.0)) == (1.0, 1.0, 1.0)
assert lambert((1.0, 1.0, 1.0), n, (0.0, -1.0, 0.0)) == (0.0, 0.0, 0.0)
assert ray_sphere((0.0, 0.0, 3.0), (0.0, 0.0, -1.0)) == 2.0
assert ray_sphere((0.0, 0.0, 3.0), (0.0, 1.0, 0.0)) is None
```

## 測試與預期結果

執行程式後，預期在目前目錄建立 `lambert_sphere.ppm`。這是依程式公式推得的預期行為，並非作者已執行檢查的結果。

程式中的斷言提供四項可重現測試：

1. **正面光源：**法線與光源方向相同時，白色反照率輸出為 \((1,1,1)\)。
2. **背面光源：**法線與光源方向相反時，直接漫反射項為零。
3. **球面正中射線：**從 \((0,0,3)\) 沿 \(-Z\) 前進，第一次碰到單位球的參數 \(t=2\)。
4. **不相交射線：**從同一起點沿 \(+Y\) 前進，不與單位球相交。

可再加測斜射例：正規化 \((0,1,1)\) 後，將其與 \((0,1,0)\) 內積，結果應為 \(1/\sqrt2\)。光源輸入若非單位向量，內積大小也會隨其長度縮放，所以幾何方向在進入著色計算前須正規化。

## 除錯與常見陷阱

- **法線或光源方向未正規化。** 若向量長度不是1，內積同時混入長度因子。正規化方向，並檢查零向量。
- **混淆光線傳播方向與指向光源的方向。** 本章 \(\mathbf l\) 由表面指向光源；若手上的向量是光線傳播方向 \(\boldsymbol\omega_i\)，符號相反。採用不同慣例時必須一致。
- **沒有截除背面光。** 直接Lambert項應使用 \(\max(0,\mathbf n\cdot\mathbf l)\)。不截除會讓負值變成不合理的負亮度。
- **把環境補光當成Lambert直接照明。** 程式的 `AMBIENT` 是為了避免背光側全黑的藝術性簡化項，不含方向積分，也不是量測到的環境輻亮度。
- **將線性RGB直接誤認為顯示輸出。** 線性RGB適合做加乘運算；sRGB為非線性編碼。計算光照時不應先把材質顏色的sRGB碼值當線性值相乘。
- **把經驗模型當成能量守恆保證。** Phong高光可以超過合理能量尺度。若使用物理BRDF，需同時考慮BRDF單位、餘弦因子、光源定義與顏色空間。

## 養殖數位分身案例

為合成養殖池中的魚體指定線性反照率 \(\boldsymbol\rho\)，並讓每個可見表面點帶有外向法線 \(\mathbf n\)。對遠處面積光源，可以先以一個或多個方向樣本近似；對單一方向光，按 \(\mathbf n\cdot\mathbf l\) 計算漫反射。池底、水面與魚身應分別使用材質參數，避免一個反照率同時代表不同表面。

例如將魚身簡化為偏紅褐的漫反射材質，把入射光設定為合成白光，先檢查朝光側是否隨法線角度平滑變暗，再單獨加入水面反光或環境光項。若以Ph​​ong項模擬魚身光澤，必須把它標為外觀近似；不能由漂亮高光推論魚皮的真實粗糙度、濕度或水下輻射度。

這個模型也不包含水體吸收與散射、遮蔽、折射、體積光傳輸或相機響應。若用來產生訓練影像，資料紀錄應保存合成光源、材質與模型設定，並將合成結果與實測影像分開標示。

## 習題

### 習題1：手算Lambert

表面法線為 \(\mathbf n=(0,1,0)\)，光源方向為單位向量 \(\mathbf l=(0,0.8,0.6)\)，\(\rho=0.5\)，照度 \(E=4\,\mathrm{W/m^2}\)。求出射輻亮度。若光源方向改成 \((0,-0.8,0.6)\)，直接Lambert出射輻亮度為何？

### 習題2：程式測試

沿用程式中的 `normalize` 和 `lambert`，寫出一個最小測試，驗證白色反照率、垂直入射時的輸出；再測試傾斜 \(\pi/3\) 的單位方向，使輸出係數為 \(1/2\)。說明為什麼不能將斜向量 \((0,0.5,0.5)\) 直接當作單位方向。

### 習題3：反例與除錯

某段著色程式直接計算 `ndotl = dot(n, l)`，不檢查正負號，且把 `l` 設為從光源指向表面的傳播方向。指出兩個問題，並用 \(\mathbf n=(0,1,0)\)、傳播方向 \((0,-1,0)\) 算出錯誤與正確的受光係數。

### 習題4：整合應用

合成魚體表面點的單位法線為
\(\mathbf n=(0,1,0)\)，光源方向為
\(\mathbf l=(0,1,0)\)。線性反照率為 \((0.4,0.2,0.1)\)，光源RGB為 \((1.5,1.0,0.5)\)。使用本章簡化RGB模型計算該點顏色，再計算光源方向改成 \((0,-1,0)\) 時的直接漫反射。此結果可否直接當成sRGB影像值或真實魚皮反射量？說明理由。

## 習題解答

### 解答1

正面情況的餘弦為

$$
\mathbf n\cdot\mathbf l=0.8.
$$

表面接收照度為 \(4\times0.8=3.2\,\mathrm{W/m^2}\)，因此

$$
L_o=\frac{0.5}{\pi}\times3.2
=\frac{1.6}{\pi}
\approx0.5093\,\mathrm{W/(m^2\,sr)}.
$$

反向情況的內積是 \(-0.8\)，截為零，直接出射輻亮度為 \(0\)。

### 解答2

可在程式末尾加入：

```python
import math

white = (1.0, 1.0, 1.0)
n = (0.0, 1.0, 0.0)

assert lambert(white, n, (0.0, 1.0, 0.0)) == white
l = normalize((0.0, 0.5, math.sqrt(3.0) / 2.0))
result = lambert(white, n, l)
assert abs(result[0] - 0.5) < 1e-12
```

這個斜向單位向量的 \(Y\) 分量為 \(1/2\)，因此與 \(\mathbf n\) 的點積為 \(1/2\)。\((0,0.5,0.5)\) 的長度是 \(\sqrt{0.5}\)，不是1；未正規化時，點積會受方向向量長度影響，不能單獨解讀為餘弦。

### 解答3

第一個問題是沒有把負的餘弦截成零；第二個問題是方向慣例相反。已知傳播方向 \(\boldsymbol\omega_i=(0,-1,0)\)，由表面指向光源的方向是 \(\mathbf l=-\boldsymbol\omega_i=(0,1,0)\)。

若錯把傳播方向當作 \(\mathbf l\)，點積為 \(-1\)。正確計算則為 \(\mathbf n\cdot\mathbf l=1\)，受光係數為1。即使修正方向慣例，仍應對點積取 \(\max(0,\cdot)\)，使背面直接受光不變成負能量。

### 解答4

兩個方向相同時，點積為1，因此

$$
\mathbf C=(0.4,0.2,0.1)\odot(1.5,1.0,0.5)
=(0.6,0.2,0.05).
$$

光源反向時，點積為 \(-1\)，截為零，直接漫反射為 \((0,0,0)\)。

計算結果是此簡化模型下的線性RGB，不是sRGB碼值；輸出前需要做適當的色彩轉換。它也不是實測魚皮反射量，因為光源量的物理定義、相機響應、表面其他反射成分及水體傳輸都未建模。

## 本章小結

Lambert模型以 \(\max(0,\mathbf n\cdot\mathbf l)\) 表示表面朝向對直接照明的影響；在輻射度量形式中，理想漫反射BRDF為 \(\rho/\pi\)。立體角讓光可以按方向積分，BRDF則描述入射光如何反射到特定觀察方向。Phong提供容易調整的經驗性高光，不等同於能量守恆的物理模型。實作時要一致地定義方向、正規化向量、截除背面照明，並區分線性RGB、sRGB與實際物理量。

## 參考來源

下列為延伸閱讀；章內推導與程式以本章定義的慣例為準。引用來源不表示本章每項敘述已逐條外部查核。

- G2，*Physically Based Rendering: From Theory to Implementation, 4th Edition*，Reflection Models：<https://pbr-book.org/4ed/Reflection_Models>
- G3，*Physically Based Rendering: From Theory to Implementation, 4th Edition*，The Light Transport Equation：<https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation>
- G4，*Ray Tracing in One Weekend*：<https://raytracing.github.io/books/RayTracingInOneWeekend.html>