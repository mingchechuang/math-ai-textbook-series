# 第 17 章　法線貼圖與表面細節

## 學習目標與先備知識

完成本章後，讀者應能：

1. 區分幾何法線、著色法線與法線貼圖所儲存的局部法線。
2. 由三角形的位置與 UV 建立切線空間基底 TBN。
3. 解碼 RGB 法線貼圖，並將切線空間法線轉到世界空間。
4. 以 Gram–Schmidt 方法正交化切線，正確處理鏡射 UV 的手性。
5. 在非均勻縮放下分別變換切線與法線。
6. 說明法線貼圖、凹凸貼圖與位移貼圖對幾何與輪廓的不同影響。
7. 測試中性法線、退化 UV、鏡射 UV 與奇異縮放。

先備知識為向量、內積、外積、矩陣逆與第 13～16 章的線性色彩、UV、局部照明及材質概念。本章長度單位仍為公尺；方向向量無長度單位。

---

## 問題與直覺

若要表現魚鱗、混凝土池壁或水槽刮痕，直接把每一條紋路建成三角形通常代價很高。法線貼圖不改變網格位置，而是在每個片段提供一個看似不同的表面方向，使照明產生細小明暗變化。

關鍵困難是：貼圖中的 RGB 並不是世界空間方向。大多數法線貼圖儲存的是**切線空間法線**。對表面每一點建立三個局部軸：

- $\mathbf T$：UV 的 $u$ 增加方向，稱切線；
- $\mathbf B$：UV 的 $v$ 增加方向，稱副切線；
- $\mathbf N$：表面法線。

三者組成 TBN 基底。中性法線貼圖代表方向 $(0,0,1)^T$，經 TBN 轉換後恰好成為原本的 $\mathbf N$，因此不應改變照明。

法線貼圖只改變著色所見的方向，不會：

- 改變物體輪廓；
- 真的遮擋其他幾何；
- 改變射線與表面的交點；
- 自動產生正確的自陰影。

若表面起伏需要改變輪廓或交點，就要使用位移幾何、細分後位移，或其他實際改變表面位置的方法。

---

## 數學與幾何推導

### 1. 幾何法線、著色法線與貼圖法線

三角形頂點為 $\mathbf p_0,\mathbf p_1,\mathbf p_2\in\mathbb R^3$，令

$$
\mathbf e_1=\mathbf p_1-\mathbf p_0,\qquad
\mathbf e_2=\mathbf p_2-\mathbf p_0.
$$

其單位幾何法線為

$$
\mathbf N_g=
\frac{\mathbf e_1\times\mathbf e_2}
{\|\mathbf e_1\times\mathbf e_2\|}.
$$

若外積長度接近零，三角形退化，法線沒有穩定定義。

**著色法線** $\mathbf N_s$ 可以由頂點法線插值得到，通常比逐面幾何法線平滑。法線貼圖解碼出的 $\mathbf n_t$ 則位於切線空間。最終著色法線是

$$
\mathbf n_w=
\operatorname{normalize}
\left(
\mathbf T_w n_x+\mathbf B_w n_y+\mathbf N_w n_z
\right).
$$

這裡下標 $t$ 表示 tangent space，下標 $w$ 表示 world space。

### 2. 由 UV 導出切線與副切線

三個頂點的 UV 為

$$
\mathbf q_i=(u_i,v_i)^T.
$$

定義

$$
\Delta u_1=u_1-u_0,\quad \Delta v_1=v_1-v_0,
$$

$$
\Delta u_2=u_2-u_0,\quad \Delta v_2=v_2-v_0.
$$

假設三角形內位置對 UV 局部近似線性：

$$
\mathbf e_1=\mathbf T_{\rm raw}\Delta u_1+
             \mathbf B_{\rm raw}\Delta v_1,
$$

$$
\mathbf e_2=\mathbf T_{\rm raw}\Delta u_2+
             \mathbf B_{\rm raw}\Delta v_2.
$$

令

$$
D=\Delta u_1\Delta v_2-\Delta u_2\Delta v_1.
$$

若 $|D|$ 太小，表示 UV 三角形面積接近零，無法可靠反推出切線。否則解二乘二線性系統可得

$$
\mathbf T_{\rm raw}
=
\frac{\Delta v_2\mathbf e_1-\Delta v_1\mathbf e_2}{D},
$$

$$
\mathbf B_{\rm raw}
=
\frac{-\Delta u_2\mathbf e_1+\Delta u_1\mathbf e_2}{D}.
$$

這兩個向量包含 UV 尺度資訊，而且不一定與著色法線正交。

### 3. TBN 正交化與手性

先將切線中沿法線的分量移除：

$$
\mathbf T=
\operatorname{normalize}
\left(
\mathbf T_{\rm raw}
-\mathbf N(\mathbf N\cdot\mathbf T_{\rm raw})
\right).
$$

再以原始副切線判斷手性：

$$
h=\operatorname{sign}
\left[
(\mathbf N\times\mathbf T)\cdot\mathbf B_{\rm raw}
\right],
\qquad h\in\{-1,+1\}.
$$

最後重建

$$
\mathbf B=h(\mathbf N\times\mathbf T).
$$

因此 TBN 矩陣的三個**縱行**為

$$
\mathbf M_{\rm TBN}
=
\begin{bmatrix}
|&|&|\\
\mathbf T&\mathbf B&\mathbf N\\
|&|&|
\end{bmatrix},
$$

而切線空間到世界空間的主動轉換為

$$
\mathbf n_w=\operatorname{normalize}
(\mathbf M_{\rm TBN}\mathbf n_t).
$$

鏡射 UV 通常會改變 $D$ 的符號，進而使 $h$ 改變。若一律使用 $\mathbf B=\mathbf N\times\mathbf T$，鏡射區域的凹凸方向便可能翻轉。

網格格式常只儲存四分量切線 $(T_x,T_y,T_z,h)$，副切線在著色時重建。不同資產格式可能另有 UV 軸與手性約定，匯入時必須確認，不能只靠外觀猜測。

### 4. 法線貼圖解碼

貼圖通道通常儲存在 $[0,1]$。完整 RGB 解碼為

$$
\tilde{\mathbf n}_t=2\mathbf c-\mathbf 1,
\qquad
\mathbf n_t=
\frac{\tilde{\mathbf n}_t}{\|\tilde{\mathbf n}_t\|}.
$$

中性值為

$$
\mathbf c=(0.5,0.5,1),
\qquad
\mathbf n_t=(0,0,1).
$$

法線貼圖是方向資料，不是色彩，不可套用 sRGB 到線性 RGB 的轉換。若錯把 $0.5$ 當作 sRGB 解碼，線性值約成 $0.214$，映射到 $[-1,1]$ 後約為 $-0.572$，中性法線會嚴重偏斜。

有些格式只存 $x,y$，並假設法線位於正 $z$ 半球：

$$
z=\sqrt{\max(0,1-x^2-y^2)}.
$$

若 $x^2+y^2>1$，可能是量化、壓縮或資料錯誤；截斷到零能避免平方根失效，但不能恢復已遺失的資訊。

### 5. 非均勻縮放

物件到世界空間的仿射變換，其線性部分為可逆矩陣 $\mathbf A$。切向量屬於方向，因此

$$
\mathbf T'_{\rm raw}=\mathbf A\mathbf T.
$$

法線必須保持與變換後切平面垂直，所以使用逆轉置：

$$
\mathbf N'=
\operatorname{normalize}
(\mathbf A^{-T}\mathbf N).
$$

接著再相對 $\mathbf N'$ 正交化 $\mathbf T'_{\rm raw}$。若直接以 $\mathbf A\mathbf N$ 變換法線，非均勻縮放下通常不再垂直切面。

若 $\det(\mathbf A)<0$，物件變換包含鏡射，空間手性翻轉。使用儲存的切線符號時，可令

$$
h'=h\,\operatorname{sign}(\det\mathbf A),
$$

再以 $\mathbf B'=h'(\mathbf N'\times\mathbf T')$ 重建。渲染器同時還要一致處理三角形繞序與背面剔除。

若 $\det(\mathbf A)=0$，例如某軸縮放為零，逆矩陣不存在；本章實作直接拒絕此變換。

### 6. 凹凸、法線與位移

高度函數 $H(u,v)$ 可描述沿基準法線的小位移。對局部平面近似

$$
\mathbf P(u,v)=u\mathbf T+v\mathbf B+H(u,v)\mathbf N.
$$

偏導為

$$
\mathbf P_u=\mathbf T+H_u\mathbf N,\qquad
\mathbf P_v=\mathbf B+H_v\mathbf N.
$$

在正交右手基底中，其法線方向近似

$$
\mathbf n_t\propto(-H_u,-H_v,1).
$$

這就是凹凸貼圖：由高度差分估計斜率，再修改法線。法線貼圖則直接儲存方向，能表示較自由的細節，但不一定能積分回單一一致高度場。位移貼圖直接改變 $\mathbf P$，因此可能改變輪廓、遮擋與交點，成本也較高。

---

## 逐步手算例題

### 例 1：標準 UV 三角形與中性法線

令

$$
\mathbf p_0=(0,0,0),\quad
\mathbf p_1=(2,0,0),\quad
\mathbf p_2=(0,1,0),
$$

$$
\mathbf q_0=(0,0),\quad
\mathbf q_1=(1,0),\quad
\mathbf q_2=(0,1).
$$

因此

$$
\mathbf e_1=(2,0,0),\quad \mathbf e_2=(0,1,0),
$$

且 $D=1$。代入公式：

$$
\mathbf T_{\rm raw}=(2,0,0),\qquad
\mathbf B_{\rm raw}=(0,1,0).
$$

幾何法線為

$$
\mathbf N=(0,0,1).
$$

正規化後

$$
\mathbf T=(1,0,0),\quad
h=\operatorname{sign}[(0,1,0)\cdot(0,1,0)]=+1,
$$

$$
\mathbf B=(0,1,0).
$$

中性法線貼圖 $\mathbf c=(0.5,0.5,1)$ 解碼為 $\mathbf n_t=(0,0,1)$，所以

$$
\mathbf n_w=\mathbf T(0)+\mathbf B(0)+\mathbf N(1)
=(0,0,1).
$$

### 例 2：鏡射 UV

位置不變，但交換第二、第三個 UV 方向：

$$
\mathbf q_1=(0,1),\qquad \mathbf q_2=(1,0).
$$

此時

$$
D=0\cdot0-1\cdot1=-1.
$$

計算得到

$$
\mathbf T_{\rm raw}=(0,1,0),\qquad
\mathbf B_{\rm raw}=(2,0,0).
$$

正規化切線為 $\mathbf T=(0,1,0)$，而

$$
(\mathbf N\times\mathbf T)\cdot\mathbf B_{\rm raw}
=(-1,0,0)\cdot(2,0,0)=-2,
$$

故 $h=-1$。因此

$$
\mathbf B=-[\mathbf N\times\mathbf T]=(1,0,0).
$$

鏡射 UV 若遺失 $h=-1$，副切線會錯成 $(-1,0,0)$。

### 例 3：非均勻縮放

取單位切線與法線

$$
\mathbf T=\frac{(1,1,0)}{\sqrt2},\qquad
\mathbf N=\frac{(-1,1,0)}{\sqrt2},
$$

並令

$$
\mathbf A=
\begin{bmatrix}
2&0&0\\
0&1&0\\
0&0&1
\end{bmatrix}.
$$

切線直接變換：

$$
\mathbf A\mathbf T=\frac{(2,1,0)}{\sqrt2}.
$$

若錯誤地直接變換法線，得到 $(-2,1,0)/\sqrt2$，其內積為

$$
(2,1,0)\cdot(-2,1,0)=-3\ne0.
$$

正確法線為

$$
\mathbf A^{-T}\mathbf N
=
\frac{(-1/2,1,0)}{\sqrt2}.
$$

忽略共同尺度後，

$$
(2,1,0)\cdot(-1/2,1,0)=0,
$$

仍與切線垂直。

---

## 實作與程式

以下程式只需 Python 3.10+ 與 NumPy。它不讀檔、不連網，也不執行 GPU 程式。

```python
import numpy as np

EPS = 1e-10

def normalize(v, eps=EPS):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if length <= eps:
        raise ValueError("無法正規化零長或近零向量")
    return v / length

def triangle_tangent_frame(p0, p1, p2, uv0, uv1, uv2, eps=EPS):
    p0, p1, p2 = map(lambda x: np.asarray(x, dtype=float),
                     (p0, p1, p2))
    uv0, uv1, uv2 = map(lambda x: np.asarray(x, dtype=float),
                        (uv0, uv1, uv2))

    e1 = p1 - p0
    e2 = p2 - p0
    ng_raw = np.cross(e1, e2)
    if np.linalg.norm(ng_raw) <= eps:
        raise ValueError("退化幾何三角形")
    n = normalize(ng_raw)

    d1 = uv1 - uv0
    d2 = uv2 - uv0
    det_uv = d1[0] * d2[1] - d2[0] * d1[1]
    if abs(det_uv) <= eps:
        raise ValueError("退化 UV 三角形")

    t_raw = (d2[1] * e1 - d1[1] * e2) / det_uv
    b_raw = (-d2[0] * e1 + d1[0] * e2) / det_uv

    t_ortho = t_raw - n * np.dot(n, t_raw)
    t = normalize(t_ortho)

    triple = np.dot(np.cross(n, t), b_raw)
    if abs(triple) <= eps:
        raise ValueError("無法穩定判定切線空間手性")
    handedness = 1.0 if triple > 0.0 else -1.0
    b = handedness * np.cross(n, t)

    return t, b, n, handedness

def decode_normal(rgb):
    rgb = np.asarray(rgb, dtype=float)
    if rgb.shape != (3,):
        raise ValueError("RGB 必須是三分量")
    n = 2.0 * rgb - 1.0
    return normalize(n)

def tangent_to_world(n_tangent, t, n, handedness):
    t = normalize(t - n * np.dot(n, t))
    n = normalize(n)
    b = handedness * np.cross(n, t)
    return normalize(t * n_tangent[0] +
                     b * n_tangent[1] +
                     n * n_tangent[2])

def transform_frame(t, n, handedness, A, eps=EPS):
    A = np.asarray(A, dtype=float)
    if A.shape != (3, 3):
        raise ValueError("A 必須是 3x3 矩陣")

    det_a = np.linalg.det(A)
    if abs(det_a) <= eps:
        raise ValueError("奇異線性變換沒有可用的法線逆轉置")

    n_world = normalize(np.linalg.inv(A).T @ n)
    t_raw = A @ t
    t_world = normalize(t_raw - n_world * np.dot(n_world, t_raw))

    mirror_sign = 1.0 if det_a > 0.0 else -1.0
    h_world = handedness * mirror_sign
    b_world = h_world * np.cross(n_world, t_world)
    return t_world, b_world, n_world, h_world

if __name__ == "__main__":
    p0 = [0, 0, 0]
    p1 = [2, 0, 0]
    p2 = [0, 1, 0]

    t, b, n, h = triangle_tangent_frame(
        p0, p1, p2, [0, 0], [1, 0], [0, 1]
    )
    neutral = decode_normal([0.5, 0.5, 1.0])
    nw = tangent_to_world(neutral, t, n, h)

    assert np.allclose(nw, n, atol=1e-9)
    assert np.isclose(np.dot(t, n), 0.0, atol=1e-9)
    assert np.isclose(np.dot(b, n), 0.0, atol=1e-9)

    tm, bm, nm, hm = triangle_tangent_frame(
        p0, p1, p2, [0, 0], [0, 1], [1, 0]
    )
    assert hm == -1.0

    A = np.diag([2.0, 1.0, 0.5])
    tw, bw, nw_geom, hw = transform_frame(t, n, h, A)
    assert np.isclose(np.dot(tw, nw_geom), 0.0, atol=1e-9)
    assert np.isclose(np.dot(bw, nw_geom), 0.0, atol=1e-9)

    try:
        transform_frame(t, n, h, np.diag([1.0, 0.0, 1.0]))
        raise AssertionError("奇異縮放應被拒絕")
    except ValueError:
        pass

    print("所有測試完成")
```

實際資產通常會對共享頂點所累積的切線加權平均，再正交化。若同一位置跨越 UV 接縫、硬法線邊界或手性相反的鏡射區域，必須拆成不同頂點屬性，否則平均後可能互相抵消。

---

## 測試與預期結果

上述程式未在此執行；依公式，其預期結果如下：

1. 中性 RGB $(0.5,0.5,1)$ 解碼為 $(0,0,1)$。
2. 標準 UV 三角形得到 $h=+1$。
3. 鏡射 UV 三角形得到 $h=-1$。
4. 非均勻縮放後，$\mathbf T_w\cdot\mathbf N_w$ 與 $\mathbf B_w\cdot\mathbf N_w$ 應在容差內為零。
5. 縮放矩陣 $\operatorname{diag}(1,0,1)$ 為奇異矩陣，必須拋出 `ValueError`。
6. 最後一行預期印出：

```text
所有測試完成
```

容差 `1e-10` 是本例在公尺尺度與雙精度運算下的數值判定，不是任何物理安全閾值。若場景座標非常大或非常小，應依資產尺度調整。

---

## 除錯與常見陷阱

### 法線貼圖被當成 sRGB

症狀是平坦區域仍有明顯斜向光照。應把法線貼圖作為線性資料取樣，不進行 sRGB 解碼。顯示法線貼圖預覽時可做色彩轉換，但著色計算讀取的數值不可因此改變。

### 綠色通道方向相反

不同工具可能把貼圖的 $y$ 軸定義為朝上或朝下。若所有凹槽看起來都像凸起，可測試 $n_y\leftarrow-n_y$。這是格式轉換問題，不應用任意翻轉掩蓋錯誤的 TBN。

### 忽略鏡射 UV 手性

若接縫一側正常、另一側光影翻轉，先檢查 $h$，不要只重新計算法線。鏡射區共享同一幾何位置時，通常仍需拆開切線屬性。

### 直接以模型矩陣變換法線

均勻縮放或純旋轉時可能看不出錯誤，但非均勻縮放會暴露問題。法線使用 $\mathbf A^{-T}$；切線使用 $\mathbf A$，之後再正交化。

### 逐頂點正交但不在片段重新正規化

線性或透視正確插值不保持單位長度，也不保證 TBN 完全正交。片段階段至少應重新正規化最終法線；高品質流程也會重新正交化插值後的基底。

### 退化 UV 仍硬算除法

當 $D\approx0$，切線公式會放大誤差。應修正 UV、拆分三角形，或使用明確的後備切線；不可只把分母加上 epsilon，因為那會製造任意方向。

### 法線翻過幾何背面

過強法線可能使 $\mathbf n_w\cdot\mathbf N_g<0$，造成漏光或不穩定反射。可限制擾動、使用較合理的法線資產，或採用著色法線修正策略；單純取絕對值通常會破壞方向一致性。

---

## 養殖數位分身案例

考慮合成養殖場景中的魚體與混凝土池壁：

- 魚體網格描述主要輪廓與魚鰭。
- 魚鱗使用切線空間法線貼圖，只增加高頻照明細節。
- 池壁的細刮痕可由高度圖差分形成凹凸法線。
- 破損邊角或突出管線會改變輪廓與遮擋，必須建成幾何或位移後重新離散化。

若魚模型沿身體方向縮放以產生不同尺寸，應把切線以 $\mathbf A$ 變換、法線以 $\mathbf A^{-T}$ 變換，再重建 TBN。若左右魚身共用鏡射 UV，兩側的切線手性通常不同。

在合成標註中也要區分：

- 深度與實例 ID 由實際幾何和可見性決定；
- 法線貼圖只影響著色法線，不應假造幾何深度；
- 若輸出「表面法線」標註，需明確註明是幾何法線、插值法線或貼圖擾動後法線。

這些影像是圖學模型的結果，不足以證明真實魚鱗、池壁磨損或水下光學已被準確重現。

---

## 習題

### 1. 手算題

已知

$$
\mathbf T=(1,0,0),\quad
\mathbf B=(0,1,0),\quad
\mathbf N=(0,0,1),
$$

法線貼圖值為 $\mathbf c=(0.75,0.5,1)$。求解碼並正規化後的世界空間法線。

### 2. 程式測試題

為程式加入測試，證明鏡射模型矩陣

$$
\mathbf A=\operatorname{diag}(-1,1,1)
$$

會使 $h=+1$ 的切線框架變為 $h'=-1$，同時仍維持 TBN 互相垂直。

### 3. 反例與除錯題

某程式使用

```python
n_world = normalize(A @ n)
```

處理法線。給出一組非均勻縮放、切線與法線，使變換後兩者不垂直，並寫出修正式。

### 4. 整合應用題

魚身左右兩側使用鏡射 UV，共享相同位置與法線。工程師把兩側切線直接平均，結果接縫附近切線長度接近零。說明原因，並提出資料與著色流程上的修正方案。

### 5. 概念比較題

分別判斷下列需求應優先使用法線貼圖、凹凸貼圖或位移／幾何：

1. 大量細小魚鱗，只需影響高光。
2. 由單通道程序高度函數生成池壁細紋。
3. 池壁破口必須在剪影與深度圖中可見。
4. 表面方向由雕刻軟體直接輸出，不要求可還原為高度。

---

## 習題解答

### 1. 手算題解答

先映射到 $[-1,1]$：

$$
\tilde{\mathbf n}_t
=2(0.75,0.5,1)-(1,1,1)
=(0.5,0,1).
$$

長度為

$$
\sqrt{0.5^2+1^2}=\frac{\sqrt5}{2}.
$$

故

$$
\mathbf n_t=
\left(\frac1{\sqrt5},0,\frac2{\sqrt5}\right).
$$

TBN 為單位矩陣，所以世界空間法線相同，約為

$$
\mathbf n_w=(0.4472,0,0.8944).
$$

### 2. 程式測試題解答

可加入：

```python
A_mirror = np.diag([-1.0, 1.0, 1.0])
tw, bw, nw, hw = transform_frame(t, n, 1.0, A_mirror)

assert hw == -1.0
assert np.isclose(np.dot(tw, nw), 0.0, atol=1e-9)
assert np.isclose(np.dot(bw, nw), 0.0, atol=1e-9)
assert np.isclose(np.dot(tw, bw), 0.0, atol=1e-9)
```

因 $\det(\mathbf A)=-1$，故

$$
h'=h\operatorname{sign}(\det\mathbf A)=1(-1)=-1.
$$

### 3. 反例與除錯題解答

取

$$
\mathbf T=(1,1,0),\qquad
\mathbf N=(-1,1,0),
$$

兩者原先內積為零。令

$$
\mathbf A=\operatorname{diag}(2,1,1).
$$

錯誤變換得到

$$
\mathbf T'=(2,1,0),\qquad
\mathbf N'_{\rm wrong}=(-2,1,0),
$$

其內積為 $-3$，不再垂直。正確方法是

$$
\mathbf N'=\operatorname{normalize}(\mathbf A^{-T}\mathbf N),
$$

並在正規化法線後，將 $\mathbf A\mathbf T$ 相對於 $\mathbf N'$ 再正交化。

### 4. 整合應用題解答

鏡射 UV 使兩側的 UV 參數方向或手性相反。若把方向相反的切線直接平均，可能得到近零向量，無法正規化。

修正方式為：

1. 在 UV 接縫與手性改變處拆分頂點屬性。
2. 每一側分別累積切線。
3. 相對各自的著色法線正交化。
4. 儲存切線 $\mathbf T$ 與手性 $h$。
5. 片段階段以 $\mathbf B=h(\mathbf N\times\mathbf T)$ 重建副切線。
6. 確認法線貼圖的綠色通道約定與 UV 的 $v$ 方向一致。

### 5. 概念比較題解答

1. 魚鱗高光：法線貼圖。
2. 單通道程序高度：凹凸貼圖，由高度梯度求法線。
3. 可見破口：位移或直接幾何，因為必須改變輪廓與深度。
4. 雕刻輸出方向：法線貼圖，因其不必對應可積分的高度場。

---

## 本章小結

法線貼圖的核心不是把 RGB 當成世界方向，而是建立可靠的切線空間。由位置與 UV 可解出原始切線及副切線，再以著色法線正交化，並保存鏡射 UV 的手性。

中性法線 $(0.5,0.5,1)$ 應還原原本表面法線；法線貼圖屬於線性資料，不可套用 sRGB 解碼。物件具有非均勻縮放時，切線使用線性矩陣 $\mathbf A$，法線使用逆轉置 $\mathbf A^{-T}$。奇異縮放則沒有有效的法線逆轉置。

法線與凹凸技術只修改著色方向，位移才會改變實際表面。選擇技術時，應先判斷細節是否需要影響輪廓、可見性、深度與交點。

---

## 參考來源

- [PBRT 4：Transformations](https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations)——法線與一般向量在變換下的差異。
- [PBRT 4：Reflection Models](https://pbr-book.org/4ed/Reflection_Models)——表面方向與反射模型的關係。
- [LearnOpenGL：Transformations](https://learnopengl.com/Getting-started/Transformations)——圖學變換與矩陣慣例的入門說明。
- [NumPy 線性代數參考](https://numpy.org/doc/stable/reference/routines.linalg.html)——矩陣行列式、反矩陣與向量範數介面。
- [Khronos glTF 2.0 規格](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)——資產交換中的頂點切線與材質資料規範；使用前應依實際版本回查相關條文。