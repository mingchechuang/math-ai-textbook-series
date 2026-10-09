# 跨章審稿結果

前輪指出的章號、Phong 背面光、取樣映射差異、TBN 縱列、微表面背面方向與粗糙度零值等問題，多數已修正。仍有以下實質問題。

## 一、必須修正

### 1. 第 16 章仍把輻射度量來源指到錯誤章節

**原句：**

> 輻射度量沿用第14章橋接……

第 14 章是 UV 與貼圖座標；輻亮度、照度、立體角與反射方程是在第 15 章建立。

**修法：**

> 輻射度量沿用第15章的定義……

---

### 2. 第 16 章例題 16.2 對「入射輻照度」重複乘了餘弦

**原句：**

> 若入射輻照度 $E=2\ \mathrm{W/m^2}$，出射輻射亮度約為  
> $f_r\cdot E\cdot(N\cdot L)\approx0.226$。

如果 $E$ 已是落在該表面上的**輻照度**，反射方程中的入射方向積分與餘弦已包含在 $E$ 裡，因此應為

$$
L_o=f_rE.
$$

依稿中 $f_r\approx0.1597\ \mathrm{sr}^{-1}$：

$$
L_o\approx0.1597\times2
=0.3194\ \mathrm{W/(m^2\,sr)}.
$$

只有當 $2\ \mathrm{W/m^2}$ 表示第 15 章所定義的「垂直於光束平面的法向照度」$E_\perp$，表面照度才是

$$
E_{\mathrm{surface}}
=
E_\perp(N\cdot L)
=
2\times0.7071,
$$

此時

$$
L_o=f_rE_{\mathrm{surface}}\approx0.226.
$$

**修法二選一：**

- 保留 $0.226$，將文字改成「法向照度 $E_\perp=2$」；
- 保留「表面入射輻照度 $E=2$」，將答案改為 $0.3194$。

這必須與第 15 章的 $E_\perp$／$E_{\mathrm{surface}}$ 接口一致。

---

### 3. 第 16 章仍把 Cook–Torrance 整個分母稱為單一 Jacobian

**原句：**

> 因此 BRDF 不可能只是 $FDG$，必須除以這個 Jacobian。

以及小結：

> Jacobian 分母 $4(N\cdot V)(N\cdot L)$ 由微面立體角到巨觀方向立體角的轉換產生……

目前前段已正確承認：

> 完整結果取決於 $D$ 與 $G$ 的定義，不能由單一立體角等式直接推出。

但後文又把整個

$$
4(N\cdot V)(N\cdot L)
$$

稱為「這個 Jacobian」，前後不一致。半向量映射本身的 Jacobian 是

$$
d\omega_h
=
\frac{d\omega_o}{4|\omega_o\cdot h|}.
$$

Cook–Torrance 分母則來自半向量映射、微面投影面積與巨觀入射／出射投影等因素的合併，不能等同於上式的單一 Jacobian。

**修法：**

將相關句子統一改成：

> 結合半向量映射的 Jacobian、微面投影面積及巨觀方向的投影餘弦後，得到標準分母 $4(N\cdot\omega_i)(N\cdot\omega_o)$。

並刪除「必須除以這個 Jacobian」的單因果說法。

---

### 4. 第 16 章粗糙度「峰值角寬」表數值錯誤

**原表：**

| roughness | 峰值掉到 $1/10$ 的角度 |
|---|---:|
| $0.2$ | $<2^\circ$ |
| $0.5$ | $\approx18^\circ$ |
| $0.8$ | $\approx45^\circ$ |

依本章 GGX：

$$
D(c)=
\frac{a^2}
{\pi\left(c^2(a^2-1)+1\right)^2},
$$

其中為避免符號混亂，令 $a=\alpha=r^2$。正向峰值為

$$
D(1)=\frac1{\pi a^2}.
$$

要求

$$
D(c)=\frac1{10}D(1)
$$

可得

$$
c^2
=
\frac{1-\sqrt{10}\,a^2}{1-a^2}.
$$

若表中的角度是半向量 $h$ 與 $N$ 的夾角：

- $r=0.2$：$a=0.04$，角度約 $3.37^\circ$；
- $r=0.5$：$a=0.25$，角度約 $22.3^\circ$；
- $r=0.8$：$a=0.64$ 時，$D$ 即使到 $N\cdot h=0$ 也尚未降到峰值的 $1/10$，因此沒有該半球內解。

若固定 $V=N$，光源角度 $\theta_L$ 是半向量角度的兩倍，因此前兩者約為：

- $6.74^\circ$；
- $44.6^\circ$。

$r=0.8$ 仍沒有「降到峰值 $1/10$」的光源角度。

**修法：**

先明定表格使用的是半向量角還是光源角，再以正確數值重算。不能保留目前的 `<2°、18°、45°`。

---

### 5. 第 14 章紋素索引的 row／column 敘述顛倒且有歧義

**原句：**

> 影像陣列第 $(i,j)$ 個像素（$i$ 為列，$j$ 為行，$j=0$ 為頂部）……

隨後卻寫：

$$
u_i=\frac{i+0.5}{W},\qquad
v_j=1-\frac{j+0.5}{H}.
$$

由公式可知 $i$ 是水平方向、範圍對應寬度 $W$；$j$ 是垂直方向、範圍對應高度 $H$。正文的「列／行」容易被理解成相反方向，也違反全書要求避免中文行列歧義的約定。

**修法：**

> 影像陣列 `tex[j, i]` 中，$i$ 是水平 column index，向右增加；$j$ 是垂直 row index，向下增加，且 $j=0$ 位於頂部。

如此與第 13 章、NumPy 的 `[row, column]` 及程式 `tex[y, x]` 一致。

---

## 二、程式接口與邊界

### 1. 第 16 章 `albedo` shape 檢查仍過寬

**原程式：**

```python
if albedo.ndim > 1 or not np.all(np.isfinite(albedo)):
```

它會接受 shape 為 `(2,)`、`(4,)` 的陣列，但後續語意只支援純量或 RGB。

**修法：**

```python
if albedo.ndim == 0:
    pass
elif albedo.shape == (3,):
    pass
else:
    raise ValueError("albedo 須為有限純量或 shape (3,) 的 RGB")
if not np.all(np.isfinite(albedo)):
    raise ValueError(...)
if np.any((albedo < 0.0) | (albedo > 1.0)):
    raise ValueError("albedo must be in [0, 1]")
```

若有意支援 HDR 反射色，則不能稱為反照率，且必須另訂能量政策。

### 2. 第 16 章應新增背面與零 roughness 測試

程式已修正背面方向和 $V+L=0$，但測試沒有覆蓋。建議新增：

```python
total, diff, spec = cook_torrance(
    [0, 0, 1], [0, 0, 1], [0, 0, -1],
    0.5, 0.5, 0.0
)
assert np.allclose(total, 0.0)
```

以及：

```python
assert np.isfinite(ggx_d(1.0, 0.0))
```

另測 `metallic=-0.1`、`roughness=np.nan`、錯誤 albedo shape 必須拋錯。

---

## 三、跨章已修正且核對通過

- 第 13、14 章兩種取樣映射的差異已明示。
- 第 14、15 章線性 PPM 已以 `_linear.ppm` 命名。
- 第 15 章 Phong 已加入 $N\cdot L>0$ 的單面半球條件。
- 第 16 章章號依賴已改為第 14 章 UV、第 15 章 Lambert／BRDF。
- 第 16 章 `cook_torrance` 已拒絕背面方向並處理零半角。
- 第 16 章已實作 $\alpha_{\min}$，避免 roughness 為零時形成 $0/0$。
- 第 17 章 TBN 已改稱三個縱列。
- 第 18 章透視屬性插值、NDC 深度插值及 sRGB 出口仍保持一致。

## 四、引用與驗證待辦

第 16 章仍列出 G1～G8 全部來源，其中 Transformations、Skinning 等並不直接支持微表面 Jacobian、GGX 或白色爐論述。建議只保留實際相關來源，並將自行推導與候選延伸閱讀分開。

本審查未執行任何程式，也未獨立查核來源。修訂後仍需人工驗證 GGX 角寬、背面 BRDF、零 roughness、albedo shape、TBN 極端尺度及各 PPM 的編碼標記。

VERDICT: REVISE