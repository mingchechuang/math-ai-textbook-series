# 第 10 章　參數曲面與曲面離散化

## 學習目標與先備知識

本章從連續參數曲面推導切向量、切平面與法線，再將曲面取樣成可供渲染使用的三角網格。完成後，你應能：

- 用參數方程描述曲面，並由偏導數取得局部切向量。
- 用切向量外積計算依參數順序定向的法線，辨認退化情形。
- 選擇取樣數與索引連接方式，建立頂點、三角形及 UV 座標。
- 處理旋轉曲面的極點、週期接縫及退化三角形。
- 用 Python 標準庫產生簡化魚體 OBJ，並測試索引、面積與繞序。

先備為向量、內積、外積、三角函數及矩陣的基本概念。長度以公尺表示，角度以弧度表示；座標遵守右手系，三角形外向前面依外向法線觀看時為逆時針。網格取樣只近似幾何，不保證模型符合真實魚類形態或生理特性。

## 問題與直覺

魚身表面可視為連續曲面，但圖形管線通常以有限個三角形表示它。參數方程回答「如何由參數得到位置」；離散化則回答「取哪些參數位置，以及如何把它們連成面」。

令魚身長軸沿X軸。每個X位置都有橫截面，截面的半長軸、半短軸隨X改變；再以角度繞X軸掃過一圈，就能生成簡化魚身。

離散化時要處理兩個問題：

1. **取樣密度**：取樣過疏，曲面顯出折角；取樣加密則增加頂點與面數。
2. **邊界與退化**：角度繞一圈後首尾相接；若端點截面半徑縮為零，整圈頂點重合，可能形成零面積三角形。

因此，網格不能只看起來像魚；還要檢查索引、三角形面積、繞序與接縫。

## 數學與幾何推導

### 參數曲面與偏導數

令參數為 \(u,v\)，定義曲面

$$
\mathbf{S}(u,v)=
\begin{pmatrix}
x(u,v)\\
y(u,v)\\
z(u,v)
\end{pmatrix}.
$$

對兩個參數分別微分，得到

$$
\mathbf{S}_u=\frac{\partial \mathbf{S}}{\partial u},
\qquad
\mathbf{S}_v=\frac{\partial \mathbf{S}}{\partial v}.
$$

偏導向量描述位置隨單一參數微小變化的方向。若曲面可微且兩向量不平行，便張成局部切平面。以參數點 \((u_0,v_0)\) 為基準，切平面為

$$
\mathbf{X}(a,b)=\mathbf{S}(u_0,v_0)+a\mathbf{S}_u(u_0,v_0)+b\mathbf{S}_v(u_0,v_0),
$$

其中 \(a,b\) 為任意實數。依參數順序定向的單位法線為

$$
\mathbf{n}_{uv}=
\frac{\mathbf{S}_u\times\mathbf{S}_v}
{\|\mathbf{S}_u\times\mathbf{S}_v\|}.
$$

交換外積順序會反轉方向。這個公式不保證所得法線朝向曲面的外側；外向方向須由參數化方向或參考方向判定。若外積長度為零，兩切向量平行或至少一者為零，這個參數化在該處不能提供唯一法線，不能直接正規化。

### 旋轉曲面模型

令 \(x\) 為沿魚身的參數，\(\theta\) 為繞魚身的角度。以 \(r_y(x)\)、\(r_z(x)\) 表示橫截面在Y、Z方向的半徑，則

$$
\mathbf{S}(x,\theta)=
\begin{pmatrix}
x\\
r_y(x)\cos\theta\\
r_z(x)\sin\theta
\end{pmatrix},
\qquad 0\leq\theta\leq 2\pi.
$$

這是橢圓截面；若 \(r_y=r_z\)，截面為圓。其偏導數為

$$
\mathbf{S}_x=
\begin{pmatrix}
1\\
r_y'(x)\cos\theta\\
r_z'(x)\sin\theta
\end{pmatrix},
\qquad
\mathbf{S}_\theta=
\begin{pmatrix}
0\\
-r_y(x)\sin\theta\\
r_z(x)\cos\theta
\end{pmatrix}.
$$

依 \(\mathbf{S}_x\times\mathbf{S}_\theta\) 次序，外積為

$$
\mathbf{S}_x\times\mathbf{S}_\theta=
\begin{pmatrix}
r_y'r_z\cos^2\theta+r_z'r_y\sin^2\theta\\
-r_z\cos\theta\\
-r_y\sin\theta
\end{pmatrix}.
$$

若半徑沿長軸變化不過陡，魚身外向法線採用相反次序：

$$
\mathbf{n}_{\mathrm{out}}\ \propto\ \mathbf{S}_\theta\times\mathbf{S}_x.
$$

實際網格的法線由三角形索引繞序決定，仍須以面積向量和外向參考方向檢查。

### 取樣、面片與接縫

將 \(x\) 分成 \(N_x\) 段、角度分成 \(N_\theta\) 段：

$$
x_i=x_{\min}+\frac{i}{N_x}(x_{\max}-x_{\min}),
\quad i=0,\ldots,N_x,
$$

$$
\theta_j=\frac{2\pi j}{N_\theta},
\quad j=0,\ldots,N_\theta.
$$

保留首尾重複的角度欄時，頂點索引為

$$
k(i,j)=i(N_\theta+1)+j.
$$

雖然 \(\theta=0\) 與 \(2\pi\) 的位置相同，仍保留兩份頂點，使 UV 的 \(v\) 可以由0走到1，而不在貼圖座標上跳回0。此處約定 \(u=i/N_x\) 沿魚身，\(v=j/N_\theta\) 沿角度。若不需 UV 接縫，也可只存 \(N_\theta\) 個角度頂點，並以模數回接。

令四邊形角點為

$$
a=k(i,j),\quad b=k(i+1,j),\quad
c=k(i+1,j+1),\quad d=k(i,j+1).
$$

可用 \((a,b,c)\)、\((a,c,d)\) 填滿四邊形。以 \(\theta=0\) 附近的中段為例，令 \(\mathbf{S}_x\approx(1,r_y',0)\)、\(\mathbf{S}_\theta\approx(0,0,r_z)\)，則

$$
\mathbf{S}_x\times\mathbf{S}_\theta
=(r_y'r_z,-r_z,0),
$$

其Y分量朝向 \(-Y\)，而此處魚身外側朝 \(+Y\)。因此上述面片順序朝內，需反轉為 \((a,c,b)\)、\((a,d,c)\)。後面的程式採用此繞序，並測試中段面的面積向量是否朝外。

### 極點退化

若端點半徑同時為零，該端每個角度樣本都落在同一位置，形成極點。常見處理方式有：

- **共用單一極點頂點**：相鄰環帶的面都連到同一點。拓撲較精簡，但UV與法線需特別安排。
- **保留重複極點並略去退化面**：實作直接，但可能留下非標準拓撲；容差及極點UV、法線也須另外處理。

本章程式採第二種方式，並在產生三角形時略過面積不大於容差的面。若需要封閉流形網格，應另建極點拓撲並檢查邊鄰接。

均勻取樣不代表固定的表面誤差。曲率較大的區域通常需要更密取樣；只增加整體取樣數可能使平坦區域過度細分。可進一步依弦高誤差或曲率自適應取樣。

## 逐步手算例題

### 例一：計算切向量與法線

考慮半徑為1公尺的圓柱：

$$
\mathbf{S}(x,\theta)=
\begin{pmatrix}
x\\
\cos\theta\\
\sin\theta
\end{pmatrix}.
$$

在 \(x=0,\theta=0\)：

1. 對 \(x\) 微分，得 \(\mathbf{S}_x=(1,0,0)^T\)。
2. 對 \(\theta\) 微分，得 \(\mathbf{S}_\theta=(0,-\sin\theta,\cos\theta)^T\)，因此此處 \(\mathbf{S}_\theta=(0,0,1)^T\)。
3. 依 \(\mathbf{S}_x\times\mathbf{S}_\theta\) 次序：

   $$
   \mathbf{S}_x\times\mathbf{S}_\theta
   =
   \begin{pmatrix}1\\0\\0\end{pmatrix}
   \times
   \begin{pmatrix}0\\0\\1\end{pmatrix}
   =
   \begin{pmatrix}0\\-1\\0\end{pmatrix}.
   $$

此法線長度為1，方向朝 \(-Y\)。圓柱在該點的外側朝 \(+Y\)，所以外向法線應使用 \(\mathbf{S}_\theta\times\mathbf{S}_x\)，或反轉網格索引繞序。

### 例二：接縫上的離散取樣

令 \(N_\theta=4\)，半徑為1，固定 \(x=0\)。角度樣本為

$$
0,\quad \frac{\pi}{2},\quad \pi,\quad
\frac{3\pi}{2},\quad 2\pi.
$$

其位置依序為

$$
(0,1,0),\ (0,0,1),\ (0,-1,0),\
(0,0,-1),\ (0,1,0).
$$

首尾位置相同，但參數相差 \(2\pi\)。依本章 UV 約定，兩者的 \(v\) 分別為0與1。最後一個角度區段連接第3點與第4點，建立索引時不可漏掉。此取樣以四個區段近似圓周；增加 \(N_\theta\) 會改善輪廓近似，不改變接縫處理原則。

## 實作與程式

以下程式只使用 Python 3.10+ 標準庫。它建立簡化魚體，寫出含頂點、UV與三角面的 OBJ，並測試接縫、索引、退化面與外向繞序。魚身長軸為X軸，兩端縮尖；這是教學用合成幾何，不是生物量測模型。

網格長度單位為公尺；OBJ 面索引從1開始。`twice_area` 是兩倍三角形面積，量綱為平方公尺。面法線可由三角形邊向量外積得到。

```python
import math


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def length(v):
    return math.sqrt(dot(v, v))


def fish_radii(x):
    # 魚身長度為 2 公尺；端點明確設為精確極點。
    t = (x + 1.0) / 2.0
    if t == 0.0 or t == 1.0:
        return 0.0, 0.0
    profile = math.sin(math.pi * t)
    return 0.34 * profile, 0.24 * profile


def make_fish(nx=20, ntheta=24):
    if nx < 2 or ntheta < 3:
        raise ValueError("nx 至少為 2，ntheta 至少為 3")

    vertices = []
    uvs = []

    # 角度接縫兩側各存一欄，角度欄數為 ntheta + 1。
    for i in range(nx + 1):
        x = -1.0 + 2.0 * i / nx
        ry, rz = fish_radii(x)
        for j in range(ntheta + 1):
            theta = 2.0 * math.pi * j / ntheta
            vertices.append((
                x,
                ry * math.cos(theta),
                rz * math.sin(theta),
            ))
            uvs.append((i / nx, j / ntheta))

    def index(i, j):
        return i * (ntheta + 1) + j

    faces = []
    area_epsilon = 1e-12  # 本例尺度的平方公尺容差。

    for i in range(nx):
        for j in range(ntheta):
            a = index(i, j)
            b = index(i + 1, j)
            c = index(i + 1, j + 1)
            d = index(i, j + 1)

            # 反轉繞序，使中段外側面法線朝向魚身外部。
            for tri in ((a, c, b), (a, d, c)):
                p0, p1, p2 = (vertices[k] for k in tri)
                twice_area = length(cross(sub(p1, p0), sub(p2, p0)))
                if twice_area > area_epsilon:
                    faces.append(tri)

    return vertices, uvs, faces


def validate(vertices, uvs, faces):
    assert len(vertices) == len(uvs)
    assert all(len(p) == 3 for p in vertices)
    assert all(len(uv) == 2 for uv in uvs)

    for tri in faces:
        assert len(set(tri)) == 3
        assert all(0 <= k < len(vertices) for k in tri)
        p0, p1, p2 = (vertices[k] for k in tri)
        twice_area = length(cross(sub(p1, p0), sub(p2, p0)))
        assert twice_area > 1e-12


def write_obj(path, vertices, uvs, faces):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 合成簡化魚體；長度單位為公尺\n")
        for x, y, z in vertices:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uvs:
            f.write(f"vt {u:.9g} {v:.9g}\n")
        for a, b, c in faces:
            a += 1
            b += 1
            c += 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def test_fish():
    nx, ntheta = 20, 24
    vertices, uvs, faces = make_fish(nx, ntheta)
    validate(vertices, uvs, faces)

    columns = ntheta + 1

    # 每個長度環的接縫位置重合，UV 的角度分量分別為 0 與 1。
    for i in range(nx + 1):
        first = vertices[i * columns]
        seam = vertices[i * columns + ntheta]
        assert length(sub(first, seam)) < 1e-12
        assert uvs[i * columns][1] == 0.0
        assert uvs[i * columns + ntheta][1] == 1.0

    # 兩端各欄均重合；退化三角形不得出現在輸出面中。
    for i in (0, nx):
        base = i * columns
        for j in range(1, columns):
            assert length(sub(vertices[base], vertices[base + j])) == 0.0
    assert all(
        not all(i == 0 for i in tri) and
        not all(i == nx * columns for i in tri)
        for tri in (
            tuple(k // columns for k in face)
            for face in faces
        )
    )

    # 檢查一個非退化中段面：面積向量須朝向該處魚身外側。
    i, j = nx // 2, 0
    a = i * columns + j
    b = (i + 1) * columns + j
    c = (i + 1) * columns + j + 1
    expected = (0.0, 1.0, 0.0)
    p0, p1, p2 = (vertices[k] for k in (a, c, b))
    area_vector = cross(sub(p1, p0), sub(p2, p0))
    assert dot(area_vector, expected) > 0.0


if __name__ == "__main__":
    test_fish()
    vertices, uvs, faces = make_fish()
    write_obj("fish.obj", vertices, uvs, faces)
    print(len(vertices), len(faces))
```

測試中的極點面檢查以頂點所屬長度環編號辨認兩端；被保留的面不能整個落在同一個端點環上。接縫測試使用浮點容差比較位置，而不要求一般三角函數運算所得座標位元完全相同。

OBJ只描述位置、UV與面索引，不含頂點法線。平滑著色通常需要額外計算並寫入 `vn`；一種常見方法是累加相鄰面的面積加權法線再正規化，但接縫及極點處仍須依模型需求決定是否共用法線。

## 測試與預期結果

將程式存為 `surface_fish.py`，在有 Python 3.10+ 的環境執行：

```text
python surface_fish.py
```

程式會呼叫內建測試並寫出 `fish.obj`。依預設參數 \(N_x=20,N_\theta=24\)，頂點數為

$$
(20+1)(24+1)=525.
$$

未略去退化面時，原始四邊形網格可拆成 \(2\cdot20\cdot24=960\) 個三角形。兩端各有24個角度區段；依此參數化，兩端各有每區段一個退化三角形。若只略去這些面，預期面數為

$$
960-2\cdot24=912.
$$

以上是依程式網格構造推算的預期，不代表已執行或檢視OBJ。因程式另有面積容差，修改尺度或取樣數後，應重新檢查被略去面的數量。

檢查OBJ時至少確認：

- `v` 行為有限數值，範圍符合公尺尺度預期。
- `f` 行索引不超出頂點數，且三角形沒有重複索引。
- 接縫環首尾位置相同，UV角度分量分別為0與1。
- 極點未留下退化面；中段面的繞序朝向外側。

## 除錯與常見陷阱

- **外積方向與面片繞序混淆**：交換外積順序會反轉法線；改變索引繞序則改變網格面的方向。除錯時先固定三角形索引，再計算面積向量並與外向參考方向作內積。
- **接縫重複頂點被誤認為裂縫**：接縫兩側位置相同但UV不同，是常見設計。若焊接位置相同的頂點，需另外保留UV接縫。
- **用精確相等判斷一般浮點幾何**：三角函數結果可能有浮點誤差；比較位置通常應採與模型尺度相稱的容差。程式只對明確回傳的零半徑端點使用精確位置測試。
- **極點退化被容差掩蓋**：容差過大會刪除有效小面，過小則留下近零面積面。容差量綱須與面積一致，並配合模型尺度選擇。
- **UV方向與影像方向不同**：本章沿魚身的參數是 \(u\)，沿角度的是 \(v\)。讀入影像時應明確說明是否翻轉影像列；世界Y向上與影像列向下是不同約定。
- **均勻參數取樣不等於均勻幾何品質**：輪廓變化較快處可能需要加密。可先從低面數檢查輪廓，再依誤差需求調整。

## 養殖數位分身案例

在合成養殖場場景中，可把魚體長度設定為2公尺，令X軸表示魚身長軸，並以 \(r_y(x)\)、\(r_z(x)\) 控制寬高。生成後可將模型放入池體場景圖，套用世界變換、材質與姿勢動畫。此魚體是用於測試渲染、遮擋、相機視角及幾何流程的合成資產，不是某一物種的量測或生物學驗證。

資產檢查可分三層：

1. **參數層**：半徑非負，長度與半徑使用相同單位。
2. **網格層**：索引有效、面積大於容差、繞序一致、接縫座標可對應。
3. **場景層**：網格局部到世界的變換與池體、相機座標一致；若使用非均勻縮放，法線須以線性變換的逆轉置處理。

渲染外形平滑，不能證明解剖形狀、運動或水中行為真實。若任務要求對照實際個體，需另有量測資料與明確驗證方法。

## 習題

### 習題一：手算法線

對曲面

$$
\mathbf{S}(u,v)=(u,\ 2\cos v,\ \sin v)^T
$$

計算 \(\mathbf{S}_u\)、\(\mathbf{S}_v\)，並求 \(v=0\) 時依照 \(\mathbf{S}_u\times\mathbf{S}_v\) 得到的未正規化法線與單位法線。

### 習題二：程式測試

將程式的 `ntheta` 改為12。說明如何逐一檢查各長度環的角度接縫位置；指出應比較哪些索引，以及位置和UV應有何性質。

### 習題三：反例與除錯

某人以頂點 \((a,b,c,d)\) 建立四邊形，產生三角形 \((a,c,b)\)、\((a,d,c)\)，之後發現法線朝內。說明為何不能只靠交換 `cross` 運算順序修正網格，並給出可靠的診斷步驟。

### 習題四：整合應用

建立一個長度1.6公尺的局部魚身，取樣角度24段、長度16段。計算保留角度接縫頂點時的頂點數、四邊形數及未處理極點時的理論三角形數。再說明尖端極點的兩種處理方式及各自代價。

## 習題解答

### 解答一

$$
\mathbf{S}_u=(1,0,0)^T,
\qquad
\mathbf{S}_v=(0,-2\sin v,\cos v)^T.
$$

在 \(v=0\)，\(\mathbf{S}_v=(0,0,1)^T\)，因此

$$
\mathbf{S}_u\times\mathbf{S}_v=(0,-1,0)^T.
$$

其長度為1，故單位法線亦為 \((0,-1,0)^T\)。

### 解答二

保留接縫頂點時，每個長度環有 \(12+1=13\) 個角度欄。對每一個環 \(i\)，比較索引 `i * 13` 與 `i * 13 + 12`：位置應在浮點容差內相同，UV角度分量 \(v\) 則分別為0與1。這是逐環檢查接縫；若還要檢查面索引，可確認每個角度區段都包含由欄11連到欄12的最後一段。

### 解答三

交換外積順序會反轉計算所得法線，卻不會改變三角形索引，也不會修正相鄰面間不一致的繞序。可先選一個非退化三角形，依其索引計算

$$
(\mathbf{p}_1-\mathbf{p}_0)\times(\mathbf{p}_2-\mathbf{p}_0),
$$

再將面中心至預期外部的參考方向與面積向量作內積。內積為負表示該面繞序與外向方向相反；交換後兩個頂點索引，再檢查相鄰面、接縫與封口是否一致。

### 解答四

\(N_x=16\)、\(N_\theta=24\)。保留角度接縫重複頂點時：

$$
(16+1)(24+1)=425
$$

個頂點。四邊形數為

$$
16\cdot24=384,
$$

拆成三角形後，未略去退化面的理論數為 \(768\)。

尖端可共用單一極點，讓相鄰環帶三角形扇形連接；代價是極點UV與法線要仔細處理。也可保留重複極點並略去零面積或低於容差的三角形；代價是容差須配合尺度選定，拓撲也未必是理想流形。

## 本章小結

參數曲面以 \(\mathbf{S}(u,v)\) 將參數映射至三維位置；偏導向量張成切平面，外積得到依參數順序定向的法線。將有限參數樣本連成三角形，就得到離散網格。旋轉曲面的角度接縫須兼顧幾何閉合與UV表示；半徑縮至零的極點可能產生退化面，需用拓撲設計或面積檢查處理。取樣與容差應配合幾何尺度及用途；能渲染不等於經過生物或物理驗證。

## 參考來源

以下資料可供延伸閱讀；本章以自身符號與例子推導，未照錄來源段落。

- [G1] PBRT 4：Transformations，https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations
- [G5] LearnOpenGL：Transformations，https://learnopengl.com/Getting-started/Transformations
- [G7] NumPy 線性代數參考，https://numpy.org/doc/stable/reference/routines.linalg.html