# 第05章 透視投影、裁切與深度

## 學習目標與先備知識

讀完本章後，你應能：

- 說明透視投影與正交投影如何把相機座標映射到剪裁座標。
- 推導右手相機、視線沿 $-Z$、OpenGL 式 NDC 深度範圍 $[-1,1]$ 的透視投影矩陣。
- 區分剪裁座標、NDC、視窗座標與深度緩衝值，並解釋透視除法。
- 以近平面條件裁切線段或三角形，並說明深度精度如何受到近平面、遠平面及深度緩衝格式影響。

先備知識是向量、矩陣乘法與齊次座標。本文沿用全書約定：向量為直向量，世界座標系為右手系；相機位於原點時看向 $-Z$，相機上方為 $+Y$。長度以公尺表示，角度運算使用弧度。

本章主線採 OpenGL 式剪裁與 NDC 慣例。其他圖形 API 可能使用不同的深度範圍或矩陣慣例；不能把它們的投影矩陣與本章的裁切及深度映射規則混用。

## 問題與直覺

養殖池的三維場景必須顯示在有限大小的影像上。相機投影先將場景位置轉成剪裁座標，再依剪裁體積移除不可見部分，最後將可見點映射到 NDC 與像素範圍。

透視投影符合近大遠小：距離相機兩倍遠的物體，投影尺寸約縮小為一半。正交投影則不隨深度縮放，常用於工程或量測視圖。兩者都需要裁切及視窗映射；主要差異在投影如何處理深度。

尤其要分清相機空間的 $z$、剪裁座標的 $z_c$、透視除法後的 $z_{\mathrm{ndc}}$，以及深度緩衝值。它們不是同一個量，也不能不經轉換就互相替代。

## 數學與幾何推導

### 從視錐推導透視投影

令相機座標中的點為

$$
p_{\mathrm{cam}}=(x,y,z,1)^T,
$$

相機前方的可見點滿足 $z<0$。近平面與遠平面距離分別為 $n$ 與 $f$，條件為 $0<n<f$，兩平面位於 $z=-n$ 與 $z=-f$。

令垂直視野角為 $\theta$，影像長寬比為 $a=W/H$。近平面上、下界為

$$
t=n\tan\frac{\theta}{2},\qquad b=-t,
$$

左右界為

$$
r=at,\qquad l=-r.
$$

由相似三角形，點投影到近平面後的座標為

$$
x_{\mathrm{near}}=n\frac{x}{-z},\qquad
y_{\mathrm{near}}=n\frac{y}{-z}.
$$

因此，離相機越遠，投影座標的絕對值越小。為了讓剪裁座標的齊次分量等於 $w_c=-z$，並將近平面邊界映射到 NDC 的 $[-1,1]$，得到

$$
P=
\begin{bmatrix}
\frac{2n}{r-l}&0&\frac{r+l}{r-l}&0\\
0&\frac{2n}{t-b}&\frac{t+b}{t-b}&0\\
0&0&-\frac{f+n}{f-n}&-\frac{2fn}{f-n}\\
0&0&-1&0
\end{bmatrix}.
$$

對稱視錐滿足 $l=-r$、$b=-t$，矩陣簡化為

$$
P=
\begin{bmatrix}
\frac{1}{a\tan(\theta/2)}&0&0&0\\
0&\frac{1}{\tan(\theta/2)}&0&0\\
0&0&-\frac{f+n}{f-n}&-\frac{2fn}{f-n}\\
0&0&-1&0
\end{bmatrix}.
$$

剪裁座標為

$$
p_{\mathrm{clip}}=Pp_{\mathrm{cam}}=(x_c,y_c,z_c,w_c)^T.
$$

對相機前方的點，$w_c=-z>0$。透視除法後為

$$
x_{\mathrm{ndc}}=\frac{x_c}{w_c},\qquad
y_{\mathrm{ndc}}=\frac{y_c}{w_c},\qquad
z_{\mathrm{ndc}}=\frac{z_c}{w_c}.
$$

視錐內點需滿足

$$
-w_c\le x_c\le w_c,\qquad
-w_c\le y_c\le w_c,\qquad
-w_c\le z_c\le w_c,\qquad w_c>0.
$$

因 $w_c>0$，除以 $w_c$ 後，這些條件等價於 NDC 三分量皆在 $[-1,1]$。實作時通常先在剪裁座標裁切幾何，再做透視除法，避免近平面相交的三角形在除法後出現不穩定的巨大座標。

將 $z=-n$ 代入可得 $z_{\mathrm{ndc}}=-1$，將 $z=-f$ 代入則得 $z_{\mathrm{ndc}}=1$。深度緩衝值定義為

$$
d=\frac{z_{\mathrm{ndc}}+1}{2},
$$

故近平面映到 $d=0$，遠平面映到 $d=1$。深度與相機座標的關係為

$$
z_{\mathrm{ndc}}
=\frac{f+n}{f-n}+\frac{2fn}{(f-n)z}.
$$

含有 $1/z$ 的形式表示深度分布不是距離的線性分布。近平面附近通常分配到較多深度精度；把 $n$ 設得遠小於場景所需距離，可能加劇遠處遮擋次序不穩定。

精度也取決於深度緩衝的表示格式與位元數，以及實際投影範圍。較大的遠平面與極小的近平面通常會令深度值分布更不利；選值應覆蓋實際可見場景，又避免無謂地拉大範圍。

### 正交投影

正交投影不依深度縮小物體。若視錐左右、上下界為 $l,r,b,t$，深度仍由 $z=-n$ 到 $z=-f$，則可用

$$
P_{\mathrm{ortho}}=
\begin{bmatrix}
\frac{2}{r-l}&0&0&-\frac{r+l}{r-l}\\
0&\frac{2}{t-b}&0&-\frac{t+b}{t-b}\\
0&0&-\frac{2}{f-n}&-\frac{f+n}{f-n}\\
0&0&0&1
\end{bmatrix}.
$$

其 $w_c=1$，沒有依深度縮放的透視除法。正交投影仍有裁切範圍，但不產生近大遠小。

### 視窗與像素座標

對寬 $W$、高 $H$ 的影像，若像素原點在左上角，NDC 到連續影像座標的映射為

$$
u=\frac{x_{\mathrm{ndc}}+1}{2}W,\qquad
v=\frac{1-y_{\mathrm{ndc}}}{2}H.
$$

$v$ 軸反向，是因 NDC 的 $Y$ 向上而影像座標的 $v$ 向下。像素索引 $(i,j)$ 的中心在 $(i+0.5,j+0.5)$。映射得到的連續座標不必是整數；如何取樣或判定像素涵蓋範圍，須由光柵化規則決定。

## 逐步手算例題

### 例一：近平面、遠平面與中間深度

取 $n=1$ 公尺、$f=10$ 公尺。深度矩陣係數為

$$
A=-\frac{f+n}{f-n}=-\frac{11}{9},\qquad
B=-\frac{2fn}{f-n}=-\frac{20}{9}.
$$

因此 $z_c=Az+B$，$w_c=-z$。

近平面 $z=-1$：

$$
z_c=-\frac{11}{9}(-1)-\frac{20}{9}=-1,\qquad w_c=1.
$$

所以 $z_{\mathrm{ndc}}=-1$、深度 $d=0$。

遠平面 $z=-10$：

$$
z_c=-\frac{11}{9}(-10)-\frac{20}{9}=10,\qquad w_c=10.
$$

所以 $z_{\mathrm{ndc}}=1$、深度 $d=1$。

在 $z=-2$：

$$
z_c=-\frac{11}{9}(-2)-\frac{20}{9}=\frac{2}{9},\qquad w_c=2.
$$

因此 $z_{\mathrm{ndc}}=1/9$，深度緩衝值為 $5/9$，不是距離比例 $1/9$。

### 例二：由相機座標算到連續影像位置

令 $n=1$、$f=10$、垂直視野角 $\theta=90^\circ$、長寬比 $a=2$。由於 $\tan(\theta/2)=1$，投影矩陣的 $x,y$ 縮放係數分別為 $1/2$ 與 $1$。

取點 $p_{\mathrm{cam}}=(1,0,-2,1)^T$，剪裁座標為

$$
(x_c,y_c,z_c,w_c)=\left(\frac12,0,\frac29,2\right).
$$

透視除法後，

$$
(x_{\mathrm{ndc}},y_{\mathrm{ndc}},z_{\mathrm{ndc}})
=\left(\frac14,0,\frac19\right).
$$

深度緩衝值為 $5/9$。若影像為 $800\times400$，則連續影像位置為

$$
u=\frac{1+1/4}{2}(800)=500,\qquad
v=\frac{1-0}{2}(400)=200.
$$

因此結果是連續座標 $(500,200)$，不是指定的像素索引。光柵化時還要依像素中心與取樣規則，判定哪些像素涵蓋此點。

### 近平面線段裁切

令近平面 $n=1$，線段端點深度為 $z_A=-0.5$、$z_B=-2$。以相機前方距離 $s=-z$ 表示，交點參數為

$$
\lambda=\frac{n-s_A}{s_B-s_A}
=\frac{1-0.5}{2-0.5}=\frac13.
$$

對端點向量 $A,B$，交點為

$$
C=A+\lambda(B-A).
$$

三角形裁切時，逐邊測試頂點是否位於近平面可見側，並在跨越平面的邊上計算交點。輸出可能是三角形、四邊形或空集合；四邊形可切成兩個三角形。若頂點帶有 UV、顏色等屬性，也要用同一個邊參數內插。實務上通常在剪裁空間處理所有裁切平面，不先對平面外頂點做透視除法。

## 實作與程式

以下完整範例只使用 Python 標準庫。它建立右手 OpenGL 式透視矩陣，檢查近平面與遠平面的深度端點、剪裁範圍，以及一條線段的近平面裁切。程式使用巢狀串列表示矩陣，數學上仍採矩陣乘直向量。

```python
import math


def perspective_rh_opengl(fovy, aspect, near, far):
    """右手相機看向 -Z；NDC z 範圍為 [-1, 1]。"""
    if not (0.0 < fovy < math.pi):
        raise ValueError("fovy 必須介於 0 與 pi 之間")
    if aspect <= 0.0 or near <= 0.0 or far <= near:
        raise ValueError("aspect、near、far 不符合條件")

    q = 1.0 / math.tan(fovy / 2.0)
    return [
        [q / aspect, 0.0, 0.0, 0.0],
        [0.0, q, 0.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near),
         -(2.0 * far * near) / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def mat_vec(matrix, vector):
    return [
        sum(matrix[r][c] * vector[c] for c in range(4))
        for r in range(4)
    ]


def ndc_and_depth(matrix, point):
    clip = mat_vec(matrix, [point[0], point[1], point[2], 1.0])
    w = clip[3]
    if w <= 0.0:
        raise ValueError("點不在相機前方，不能做本例透視除法")
    ndc = tuple(clip[i] / w for i in range(3))
    depth = (ndc[2] + 1.0) / 2.0
    return clip, ndc, depth


def inside_clip(clip, eps=1e-12):
    x, y, z, w = clip
    return (
        w > 0.0
        and -w - eps <= x <= w + eps
        and -w - eps <= y <= w + eps
        and -w - eps <= z <= w + eps
    )


def clip_segment_near(a, b, near):
    """輸入端點 (x, y, z)，保留 z <= -near 的線段部分。"""
    a_in = a[2] <= -near
    b_in = b[2] <= -near

    if not a_in and not b_in:
        return None
    if a_in and b_in:
        return a, b

    t = (-near - a[2]) / (b[2] - a[2])
    c = tuple(a[i] + t * (b[i] - a[i]) for i in range(3))
    return (a, c) if a_in else (c, b)


def main():
    near, far = 1.0, 10.0
    matrix = perspective_rh_opengl(math.pi / 2.0, 2.0, near, far)

    for z, expected_depth in [(-near, 0.0), (-far, 1.0)]:
        clip, ndc, depth = ndc_and_depth(matrix, (0.0, 0.0, z))
        assert inside_clip(clip)
        assert abs(depth - expected_depth) < 1e-12
        assert abs(ndc[2] - (2.0 * expected_depth - 1.0)) < 1e-12

    outside = mat_vec(matrix, [20.0, 0.0, -2.0, 1.0])
    assert not inside_clip(outside)

    clipped = clip_segment_near(
        (0.0, 0.0, -0.5), (1.5, 0.0, -2.0), near
    )
    assert clipped is not None
    assert abs(clipped[0][2] + near) < 1e-12
    assert clipped[1] == (1.5, 0.0, -2.0)

    assert clip_segment_near(
        (0.0, 0.0, -0.2), (1.0, 0.0, -0.7), near
    ) is None

    print("near/far, side clipping, and near-plane tests passed")


if __name__ == "__main__":
    main()
```

這段線段函式只處理近平面，不是完整三角形裁切器。實際渲染器還須處理其餘剪裁平面，並對交點上的所有頂點屬性同步插值。程式使用的 $10^{-12}$ 是此簡單數值案例的容差，不是通用常數；場景尺度或浮點格式改變時，應重新選擇。

## 測試與預期結果

程式斷言檢查：

- 近平面深度映到 $0$，遠平面映到 $1$。
- 超出視錐側邊的點不通過剪裁條件。
- 跨越近平面的線段交點位於 $z=-n$。
- 兩端都在近平面外的線段裁切結果為空。

若讀者執行程式，預期最後一行會印出：

```text
near/far, side clipping, and near-plane tests passed
```

這是依程式與公式可推得的預期結果，不代表作者已執行程式或驗證特定環境。

## 除錯與常見陷阱

- **相機前方符號錯誤：** 本章相機看向 $-Z$，可見點的 $z<0$，因此 $w_c=-z>0$。
- **忘記透視除法：** 剪裁座標還不是 NDC。直接以 $x_c,y_c$ 映射像素會造成投影比例錯誤。
- **混用 NDC 深度慣例：** 本章 NDC 深度範圍是 $[-1,1]$。若目標 API 使用其他範圍，投影矩陣、剪裁條件與深度映射都須配套調整。
- **用未裁切的近平面外頂點做除法：** 三角形跨越近平面時應先裁切，再透視除法，避免產生不穩定的投影結果。
- **把 NDC 深度當距離：** 深度是非線性投影量。若要還原相機距離，必須使用同一套投影參數反算。
- **忽略緩衝格式：** 深度精度不只受 $n,f$ 影響，也受深度緩衝的格式與位元數影響。
- **影像 $Y$ 軸顛倒：** NDC 的 $Y$ 向上，影像 $v$ 向下，映射時要反轉。
- **把連續座標當像素索引：** 連續影像位置不是整數像素編號；要按像素中心與光柵化取樣規則判定。

## 養殖數位分身案例

假設合成池景的池體長 $8$ 公尺、寬 $4$ 公尺，魚群活動深度約為 $0.5$ 至 $3$ 公尺。相機放在池邊後，先將池體與魚群頂點轉到相機座標，再根據要呈現的範圍選擇視野角。近平面須小於最近要成像物體的距離，但不應無故接近零；遠平面需涵蓋最遠池體構件或背景物件，也不必無限制延伸。

三角形若跨過近平面，先計算交點、裁切並內插 UV 等屬性，再進行透視除法與光柵化。深度緩衝值可用來比較遮擋，卻不是以公尺表示的魚與相機距離。

輸出合成深度標註時，應記錄 NDC 慣例、近平面、遠平面、深度緩衝格式，以及深度值是否已反算為距離。單獨保存灰階深度圖而不保存這些設定，無法確定其尺度與意義。結果描述的是合成場景與相機設定，不是對真實魚體位置或水下光學現象的驗證。

## 習題

### 1. 手算：深度映射

取 $n=0.5$ 公尺、$f=8$ 公尺。求相機座標 $z=-0.5$、$z=-8$、$z=-2$ 時的 NDC 深度與深度緩衝值。

### 2. 程式測試：擴充近平面檢查

在程式中測試 $z=-n$ 與 $z=-f$ 的點都通過剪裁範圍檢查，且深度分別為 $0$、$1$。另測兩端都滿足 $z>-n$ 的線段，確認近平面裁切結果為空。

### 3. 反例／除錯：除法後測 NDC

相機前方的點位於 $z=-0.5$，近平面為 $n=1$。對本章的投影矩陣，計算其 $z_{\mathrm{ndc}}$，並說明為何它不可能同時落在三個 NDC 範圍 $[-1,1]$ 內。再說明當 $w_c<0$ 時，只看除法後的 NDC 有何危險。

### 4. 整合應用：選擇近平面

合成池景中最近魚距相機約 $0.8$ 公尺，最遠池壁約 $12$ 公尺。提出一組合理的 $n,f$ 並說明設定原則；再說明把 $n$ 改成 $0.001$ 公尺可能造成什麼影響。

## 習題解答

### 1. 手算：深度映射

使用

$$
z_{\mathrm{ndc}}
=\frac{f+n}{f-n}+\frac{2fn}{(f-n)z},
\qquad
d=\frac{z_{\mathrm{ndc}}+1}{2}.
$$

本題 $(f+n)/(f-n)=17/15$，$2fn/(f-n)=16/15$。

- $z=-0.5$：$z_{\mathrm{ndc}}=-1$，$d=0$。
- $z=-8$：$z_{\mathrm{ndc}}=1$，$d=1$。
- $z=-2$：$z_{\mathrm{ndc}}=17/15-8/15=3/5$，$d=4/5$。

### 2. 程式測試：擴充近平面檢查

使用程式中的 `ndc_and_depth` 與 `inside_clip`，分別帶入 $z=-n$ 與 $z=-f$；預期深度為 $0$ 與 $1$。對 $n=1$，線段端點可取 $(0,0,-0.2)$ 與 $(1,0,-0.7)$；兩端都滿足 $z>-1$，因此 `clip_segment_near` 預期回傳 `None`。這些測試只涵蓋近平面和程式明列的條件，不能代替完整多邊形剪裁驗證。

### 3. 反例／除錯：除法後測 NDC

投影矩陣給出

$$
z_{\mathrm{ndc}}
=\frac{f+n}{f-n}+\frac{2fn}{(f-n)z}.
$$

令 $z=-0.5$、$n=1$，則

$$
z_{\mathrm{ndc}}
=\frac{f+1}{f-1}-\frac{4f}{f-1}
=\frac{-3f+1}{f-1}.
$$

對 $f>1$，此值小於 $-1$，所以該點不可能同時讓三個 NDC 分量都在 $[-1,1]$ 內。對 $w_c>0$ 且使用同一投影矩陣，三個 NDC 範圍測試與齊次剪裁不等式等價，不能聲稱 NDC 測試會漏掉本例近平面條件。

若 $w_c<0$，直接除以負數會反轉不等式方向；只按一般的 NDC 範圍判斷，可能把相機後方點誤認為可見。因此必須在剪裁判斷中要求 $w_c>0$，並按齊次剪裁條件處理。

### 4. 整合應用：選擇近平面

例如取 $n=0.5$ 公尺、$f=12$ 公尺。最近魚仍位於近平面之後，最遠池壁落在遠平面內；若相機會移動，則可依場景留出合理餘量。近平面與遠平面不宜設得比必要範圍更寬。

把 $n$ 降到 $0.001$ 公尺會使深度分布更不均勻；在相同深度緩衝格式下，遠處較容易出現深度精度不足及遮擋次序錯誤。具體程度也取決於深度緩衝格式與場景深度範圍。

## 本章小結

透視投影以 $w_c=-z$ 產生近大遠小，並以透視除法將剪裁座標轉為 NDC。本章 OpenGL 式深度範圍為 $[-1,1]$，深度緩衝值為 $(z_{\mathrm{ndc}}+1)/2$。三角形跨越近平面時，應在透視除法前裁切並同步內插屬性。正交投影的 $w_c=1$，物體大小不隨深度改變。透視深度不是距離的線性編碼；深度精度取決於投影範圍，也受緩衝格式與位元數影響。

## 參考來源

以下列出延伸閱讀；列出來源不表示本章已逐條外部驗證其所有論點。

- [G1] *Physically Based Rendering, Fourth Edition*，Transformations：<https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations>
- [G5] LearnOpenGL，Transformations：<https://learnopengl.com/Getting-started/Transformations>