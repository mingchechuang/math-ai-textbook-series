# 第05章 透視投影、裁切與深度

## 學習目標與先備知識

讀完本章後，你應能：

- 說明透視投影與正交投影如何把相機座標映射到剪裁座標。
- 推導右手相機、視線沿 $-Z$、OpenGL 式 NDC 深度範圍 $[-1,1]$ 的透視投影矩陣。
- 區分剪裁座標、NDC、視窗座標與深度緩衝值，並解釋透視除法。
- 以近平面條件裁切線段或三角形，並說明深度精度受到近平面、遠平面及緩衝格式的影響。

先備知識是向量、矩陣乘法與齊次座標。本文採用直向量與右手座標系；相機位於原點時看向 $-Z$，相機上方為 $+Y$。長度以公尺表示，角度運算使用弧度。

本章主線採 OpenGL 式剪裁與 NDC 慣例。其他圖形 API 可能使用不同的深度範圍或矩陣慣例；不能把它們的投影矩陣與本章的裁切及深度映射規則混用。

## 問題與直覺

三維養殖池場景必須顯示在有限大小的影像上。相機投影將場景位置轉成剪裁座標，裁切程序移除不可見部分，最後才映射到 NDC 與像素範圍。

透視投影符合近大遠小：同一物體距離相機越遠，投影尺寸越小。正交投影則不隨深度縮放，適合工程或量測視圖。兩種投影都需要裁切；差異主要是投影如何處理深度與大小。

要正確處理可見性，必須分清相機空間的 $z$、剪裁座標的 $z_c$、透視除法後的 $z_{\mathrm{ndc}}$，以及深度緩衝值。它們不是同一個量，也不能不經轉換就互相替代。

## 數學與幾何推導

### 從相似三角形到剪裁座標

令相機座標中的點為

$$
p_{\mathrm{cam}}=(x,y,z,1)^T,
$$

相機前方的點滿足 $z<0$。近平面與遠平面距離分別為 $n$、$f$，且 $0<n<f$；兩平面位於 $z=-n$ 與 $z=-f$。

令垂直視野角為 $\theta$，影像長寬比為 $a=W/H$。近平面邊界為

$$
t=n\tan\frac{\theta}{2},\qquad b=-t,\qquad r=at,\qquad l=-r.
$$

由相似三角形，三維點投影到近平面的座標是

$$
x_{\mathrm{near}}=n\frac{x}{-z},\qquad
y_{\mathrm{near}}=n\frac{y}{-z}.
$$

由於相機前方 $z<0$，分母 $-z$ 為正。遠處點的 $|z|$ 較大，所以投影位置的絕對值較小，這就是透視縮小。

把近平面橫向座標映射到 NDC 的 $[-1,1]$，其線性映射為

$$
x_{\mathrm{ndc}}=\frac{2x_{\mathrm{near}}-(r+l)}{r-l}.
$$

代入 $x_{\mathrm{near}}=nx/(-z)$，並令剪裁座標的齊次分量 $w_c=-z$，可寫成

$$
x_{\mathrm{ndc}}
=\frac{\frac{2n}{r-l}x+\frac{r+l}{r-l}z}{-z}.
$$

因此剪裁座標第一分量應為

$$
x_c=\frac{2n}{r-l}x+\frac{r+l}{r-l}z.
$$

同理，

$$
y_c=\frac{2n}{t-b}y+\frac{t+b}{t-b}z.
$$

第三分量需要把 $z=-n$ 映至 $z_{\mathrm{ndc}}=-1$，並把 $z=-f$ 映至 $z_{\mathrm{ndc}}=1$。設

$$
z_c=Az+B,\qquad w_c=-z,
$$

則兩個端點條件分別給出

$$
\frac{-An+B}{n}=-1,\qquad
\frac{-Af+B}{f}=1.
$$

第一式乘以 $n$ 得 $-An+B=-n$，第二式乘以 $f$ 得 $-Af+B=f$。兩式相減：

$$
-A(f-n)=f+n,
$$

所以

$$
A=-\frac{f+n}{f-n}.
$$

代回任一端點式，例如 $-An+B=-n$，得到

$$
B=-n+An=-n-\frac{n(f+n)}{f-n}
=-\frac{2fn}{f-n}.
$$

因此

$$
z_c=-\frac{f+n}{f-n}z-\frac{2fn}{f-n},
\qquad w_c=-z.
$$

合併三個分量，透視投影矩陣為

$$
P=
\begin{bmatrix}
\frac{2n}{r-l}&0&\frac{r+l}{r-l}&0\\
0&\frac{2n}{t-b}&\frac{t+b}{t-b}&0\\
0&0&-\frac{f+n}{f-n}&-\frac{2fn}{f-n}\\
0&0&-1&0
\end{bmatrix}.
$$

對稱視錐滿足 $l=-r$、$b=-t$，因此

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

### 裁切、NDC 與深度緩衝

相機前方的點有 $w_c=-z>0$。透視除法得到

$$
x_{\mathrm{ndc}}=\frac{x_c}{w_c},\qquad
y_{\mathrm{ndc}}=\frac{y_c}{w_c},\qquad
z_{\mathrm{ndc}}=\frac{z_c}{w_c}.
$$

點在剪裁體積內的條件為

$$
-w_c\le x_c\le w_c,\qquad
-w_c\le y_c\le w_c,\qquad
-w_c\le z_c\le w_c,\qquad w_c>0.
$$

在 $w_c>0$ 的前提下，除以 $w_c$ 後，條件等價於三個 NDC 分量都在 $[-1,1]$。若 $w_c<0$，除法會反轉不等式方向；因此不能忽略正 $w_c$ 條件，也不能把相機後方的點照一般 NDC 範圍判斷為可見。

第三列係數的推導保證 $z=-n$ 映到 $z_{\mathrm{ndc}}=-1$，$z=-f$ 映到 $z_{\mathrm{ndc}}=1$。本章深度緩衝值為

$$
d=\frac{z_{\mathrm{ndc}}+1}{2},
$$

因此近平面深度為 $0$、遠平面深度為 $1$。由投影矩陣可得

$$
z_{\mathrm{ndc}}
=\frac{f+n}{f-n}+\frac{2fn}{(f-n)z}.
$$

深度與相機距離並非線性關係。近平面附近通常分配到較多深度精度；近平面設得過小、遠平面設得過大，會令遠處深度更容易出現精度不足。實際精度也取決於深度緩衝的格式與位元數。

### 正交投影

正交投影不依深度縮小物體。若範圍為 $[l,r]\times[b,t]$，深度區間為 $z=-n$ 至 $z=-f$，可用

$$
P_{\mathrm{ortho}}=
\begin{bmatrix}
\frac{2}{r-l}&0&0&-\frac{r+l}{r-l}\\
0&\frac{2}{t-b}&0&-\frac{t+b}{t-b}\\
0&0&-\frac{2}{f-n}&-\frac{f+n}{f-n}\\
0&0&0&1
\end{bmatrix}.
$$

其 $w_c=1$，物體投影大小不隨深度改變。它仍有裁切範圍，但不產生透視的近大遠小。

### NDC 到影像座標

對寬 $W$、高 $H$ 的影像，若像素原點位於左上角，連續影像座標為

$$
u=\frac{x_{\mathrm{ndc}}+1}{2}W,\qquad
v=\frac{1-y_{\mathrm{ndc}}}{2}H.
$$

$v$ 軸反向，是因 NDC 的 $Y$ 向上而影像座標的 $v$ 向下。像素索引 $(i,j)$ 的中心在 $(i+0.5,j+0.5)$；連續座標不一定是像素索引，實際涵蓋與取樣方式由光柵化規則決定。

## 逐步手算例題

### 例一：近平面、遠平面與中間深度

取 $n=1$ 公尺、$f=10$ 公尺。深度係數為

$$
A=-\frac{f+n}{f-n}=-\frac{11}{9},\qquad
B=-\frac{2fn}{f-n}=-\frac{20}{9}.
$$

由 $z_c=Az+B$、$w_c=-z$：

在近平面 $z=-1$，

$$
z_c=-\frac{11}{9}(-1)-\frac{20}{9}=-1,\qquad w_c=1.
$$

所以 $z_{\mathrm{ndc}}=-1$、深度 $d=0$。

在遠平面 $z=-10$，

$$
z_c=-\frac{11}{9}(-10)-\frac{20}{9}=10,\qquad w_c=10.
$$

所以 $z_{\mathrm{ndc}}=1$、深度 $d=1$。

在 $z=-2$，

$$
z_c=-\frac{11}{9}(-2)-\frac{20}{9}=\frac{2}{9},\qquad w_c=2.
$$

所以 $z_{\mathrm{ndc}}=1/9$、深度 $d=5/9$。這不是線性的距離比例 $1/9$。

### 例二：由相機座標算到影像位置

令 $n=1$、$f=10$、垂直視野角 $\theta=90^\circ$、長寬比 $a=2$。此時 $\tan(\theta/2)=1$，投影矩陣的 $x,y$ 縮放係數分別是 $1/2$ 與 $1$。

取 $p_{\mathrm{cam}}=(1,0,-2,1)^T$，得到剪裁座標

$$
(x_c,y_c,z_c,w_c)=\left(\frac12,0,\frac29,2\right).
$$

透視除法後

$$
(x_{\mathrm{ndc}},y_{\mathrm{ndc}},z_{\mathrm{ndc}})
=\left(\frac14,0,\frac19\right).
$$

深度緩衝值為 $5/9$。若影像尺寸為 $800\times400$，連續影像位置為

$$
u=\frac{1+1/4}{2}(800)=500,\qquad
v=\frac{1-0}{2}(400)=200.
$$

結果是連續座標 $(500,200)$，不是指定的像素索引。要確定哪個像素取樣此點，還須套用像素中心與光柵化規則。

### 近平面線段裁切

令 $n=1$，線段端點深度為 $z_A=-0.5$、$z_B=-2$。以相機前方距離 $s=-z$ 表示，交點內插參數為

$$
\lambda=\frac{n-s_A}{s_B-s_A}
=\frac{1-0.5}{2-0.5}=\frac13.
$$

對三維端點向量 $A,B$，交點為

$$
C=A+\lambda(B-A).
$$

三角形逐邊測試近平面時，在跨越平面的邊上建立交點，再保留可見側的多邊形。輸出通常是三角形、四邊形或空集合；頂點恰落在平面上時，實作也可能產生重複頂點或退化面，需在後續處理中辨識。若頂點帶有 UV、顏色等屬性，也要用同一個邊參數內插。實際渲染器通常在剪裁空間處理所有剪裁平面，不先對近平面外的頂點做透視除法。

## 實作與程式

以下完整範例只使用 Python 標準庫。它建立右手 OpenGL 式透視矩陣，檢查近平面與遠平面的深度端點、剪裁範圍，以及線段的近平面裁切。矩陣以巢狀串列儲存，數學慣例仍是矩陣乘直向量。

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

線段函式只處理近平面，不是完整三角形裁切器。完整渲染流程還須處理其餘剪裁平面，並同步內插交點上的所有頂點屬性。程式使用的 $10^{-12}$ 是此簡單案例的容差，不是通用常數；場景尺度與浮點格式改變時，應重新選擇容差。

## 測試與預期結果

程式斷言檢查：

- 近平面深度映到 $0$，遠平面深度映到 $1$。
- 超出視錐側邊的點不通過剪裁條件。
- 跨越近平面的線段交點位於 $z=-n$。
- 兩端都在近平面外的線段裁切結果為空。

若讀者執行程式，預期最後一行會印出：

```text
near/far, side clipping, and near-plane tests passed
```

這是依公式與程式可推得的預期結果，不表示作者已執行程式或驗證特定環境。

## 除錯與常見陷阱

- **相機前方符號錯誤：** 本章相機看向 $-Z$，可見點的 $z<0$，因此 $w_c=-z>0$。
- **忘記透視除法：** 剪裁座標還不是 NDC。直接以 $x_c,y_c$ 映射像素會造成比例錯誤。
- **混用深度慣例：** 本章 NDC 深度範圍是 $[-1,1]$。改用另一 API 時，投影矩陣、裁切條件與深度映射都須配套調整。
- **未裁切就對近平面外點做除法：** 三角形跨越近平面時，先裁切，再透視除法，避免不穩定的投影座標。
- **忽略 $w_c$ 的正負：** 當 $w_c<0$，除法會反轉不等式方向；不能直接用相機前方點的判斷方式處理。
- **把深度當距離：** 深度是非線性投影量。要還原相機距離，必須依同一組投影參數反算。
- **忽略緩衝格式：** 深度精度也受深度緩衝格式與位元數影響。
- **影像上下顛倒：** NDC 的 $Y$ 向上，影像 $v$ 向下，映射時要反轉。
- **連續影像座標誤當像素索引：** 需依像素中心與光柵化規則決定取樣位置。

## 養殖數位分身案例

假設合成池景長 $8$ 公尺、寬 $4$ 公尺，魚群活動深度約 $0.5$ 至 $3$ 公尺。將池體與魚群轉到相機座標後，依可見範圍選擇視野角。近平面要小於最近需要成像物體的距離，但不應無故接近零；遠平面需涵蓋最遠的池體構件或背景物件，不必無限制延伸。

三角形若跨越近平面，先算交點、裁切並內插 UV 等頂點屬性，再做透視除法與光柵化。深度緩衝值可用於遮擋比較，但不是以公尺為單位的距離。

若輸出合成深度標註，資料應記錄 NDC 慣例、近平面、遠平面、深度緩衝格式，以及深度值是否已反算成距離。單獨保存灰階深度圖而不保存這些設定，無法確定其尺度與意義。結果只描述合成場景及相機設定，不是對真實魚體位置或水下光學現象的驗證。

## 習題

### 1. 手算：深度映射

取 $n=0.5$ 公尺、$f=8$ 公尺。求 $z=-0.5$、$z=-8$、$z=-2$ 時的 $z_{\mathrm{ndc}}$ 與深度緩衝值。

### 2. 程式測試：端點與近平面線段

新增測試，確認 $z=-n$ 與 $z=-f$ 的點通過剪裁範圍檢查，深度分別為 $0$、$1$。再測兩端都滿足 $z>-n$ 的線段，確認裁切結果為空。

### 3. 反例與除錯：除法後檢查 NDC

相機前方的點位於 $z=-0.5$，近平面為 $n=1$。計算其 $z_{\mathrm{ndc}}$，說明為何不可能同時讓三個 NDC 分量都在 $[-1,1]$。再說明當 $w_c<0$ 時，只看除法後 NDC 的風險。

### 4. 整合應用：選擇近平面

合成池景中最近魚距相機約 $0.8$ 公尺，最遠池壁約 $12$ 公尺。提出一組合理的 $n,f$，說明設定原則；再說明把 $n$ 改成 $0.001$ 公尺可能造成什麼影響。

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

### 2. 程式測試：端點與近平面線段

用程式中的 `ndc_and_depth` 與 `inside_clip` 測試 $z=-n,-f$；預期深度分別為 $0,1$。當 $n=1$，線段端點可用 $(0,0,-0.2)$ 與 $(1,0,-0.7)$；兩端都滿足 $z>-1$，所以 `clip_segment_near` 預期回傳 `None`。這些測試涵蓋近平面與程式明列的剪裁條件，不代表已測完整多邊形裁切器。

### 3. 反例與除錯：除法後檢查 NDC

投影矩陣給出

$$
z_{\mathrm{ndc}}
=\frac{f+n}{f-n}+\frac{2fn}{(f-n)z}.
$$

令 $z=-0.5,n=1$，則

$$
z_{\mathrm{ndc}}
=\frac{f+1}{f-1}-\frac{4f}{f-1}
=\frac{-3f+1}{f-1}.
$$

對 $f>1$，此值小於 $-1$，所以三個 NDC 分量不可能同時落在 $[-1,1]$。對 $w_c>0$ 且使用同一投影矩陣，NDC 範圍檢查與齊次剪裁不等式等價；不能說它會漏掉本例的近平面條件。

若 $w_c<0$，除以負值會反轉不等式方向。若忽略 $w_c>0$、直接套用可見點的判斷方式，可能把相機後方的點誤判為可見。

### 4. 整合應用：選擇近平面

例如取 $n=0.5$ 公尺、$f=12$ 公尺。最近魚在近平面之後，最遠池壁在遠平面內；若相機會移動，可依場景保留合理餘量，但不必把近平面設得接近零。

將 $n$ 降到 $0.001$ 公尺會讓深度分布更不均勻；在相同緩衝格式下，遠處較容易遇到深度精度不足與遮擋次序錯誤。實際影響仍取決於深度緩衝格式與整個場景的深度範圍。

## 本章小結

透視投影以 $w_c=-z$ 產生近大遠小，透視除法將剪裁座標轉為 NDC。本章 OpenGL 式 NDC 深度為 $[-1,1]$，深度緩衝值為 $(z_{\mathrm{ndc}}+1)/2$。近平面與遠平面的兩個端點條件可解出投影矩陣深度列的係數。幾何跨越近平面時，應在透視除法前裁切並同步內插屬性。正交投影則令 $w_c=1$，不依深度縮放物體。深度不是距離的線性編碼；其精度受投影範圍與緩衝格式影響。

## 參考來源

以下列出延伸閱讀；列出來源不表示本章已逐條外部驗證所有論點。

- [G1] *Physically Based Rendering, Fourth Edition*，Transformations：<https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations>
- [G5] LearnOpenGL，Transformations：<https://learnopengl.com/Getting-started/Transformations>