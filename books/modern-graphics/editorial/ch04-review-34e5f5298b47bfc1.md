# 審稿結果

本稿的右手系、外積次序、矩陣作用順序、手算案例、近平面界線及程式預期結果均已大致正確；仍有少量須修正的術語與輸入邊界問題。

## 一、必須修正

### 1. 「縱行」應改為「縱列」

**原句：**

> 令 $B=[x_c\;y_c\;z_c]$ 為以這三個三維向量作為縱行的 $3\times3$ 矩陣

以及：

> 其旋轉部分 $B$ 的三個縱行……

依公式

$$
B=[x_c\;y_c\;z_c],
$$

三個向量是矩陣的 **column（縱列）**，不是「縱行」。全書約定也要求使用 row／column 或橫列／縱行時避免歧義，而「縱行」本身容易與 row 混淆。

**修法：**

統一改為：

> $B$ 的三個縱列（column）依序為 $x_c,y_c,z_c$。

相對地，$B^T$ 的三個橫列（row）才是 $x_c^T,y_c^T,z_c^T$。

---

### 2. `angular_eps` 必須限制在叉積長度的有效範圍

正規化方向的叉積長度滿足

$$
0\le \|f\times u_n\|\le1.
$$

但目前只檢查：

```python
value <= 0
```

若呼叫者傳入 `angular_eps=2`，所有方向及備援方向都會被判為退化，最後拋出與真正原因不符的錯誤。

**修法：**

分開驗證：

```python
for name, value in (
    ("position_eps", position_eps),
    ("direction_eps", direction_eps),
):
    if (not np.isscalar(value)
            or not np.isreal(value)
            or not np.isfinite(value)
            or value <= 0):
        raise ValueError(f"{name} must be positive and finite")

if (not np.isscalar(angular_eps)
        or not np.isreal(angular_eps)
        or not np.isfinite(angular_eps)
        or not (0 < angular_eps <= 1)):
    raise ValueError("angular_eps must be in (0, 1]")
```

實務上通常還應遠小於 1，但上式至少保證量綱與數學範圍合法。

---

### 3. `vertices_before_near` 對錯誤形狀的空陣列會靜默接受

**原程式：**

```python
pts = np.asarray(vertices, dtype=np.float64)
if pts.size == 0:
    pts = pts.reshape((0, 3))
```

例如輸入 `np.empty((0, 2))` 原本明確是錯誤的 `(0,2)`，卻會被重塑成 `(0,3)` 而通過檢查。

**修法：**

只對一般空串列的 `(0,)` 情況特殊處理：

```python
pts = np.asarray(vertices, dtype=np.float64)
if pts.shape == (0,):
    pts = np.empty((0, 3), dtype=np.float64)
if pts.ndim != 2 or pts.shape[1] != 3:
    raise ValueError("vertices must have shape (N, 3)")
```

如此 `(0,2)` 仍會被正確拒絕。

---

## 二、核對正確的重點

### 座標系與法線／手性

下列定義一致且正確：

$$
f=\frac{c-e}{\|c-e\|},\qquad
x_c=\frac{f\times u}{\|f\times u\|},
$$

$$
y_c=x_c\times f,\qquad z_c=-f.
$$

它們滿足

$$
x_c\times y_c=z_c,\qquad \det B=+1.
$$

沒有引入鏡射。

### 視圖矩陣

推導正確：

$$
V=
\begin{bmatrix}
B^T&-B^Te\\
0\;0\;0&1
\end{bmatrix}
=
R\,T(-e).
$$

在 column vector 約定下確實是先平移、再旋轉。位置與方向的 $w=1$、$w=0$ 差異也交代完整。

### 手算案例

簡易相機、平行 `up`、養殖場頂點與習題 1 的數值均重算正確。養殖案例得到

$$
P_{pool}^{cam}
=
\left(
10,-\frac5{\sqrt2},-\frac{15}{\sqrt2},1
\right)^T
\approx(10,-3.536,-10.607,1)^T.
$$

### 近平面與索引

題目採嚴格條件

$$
z_{cam}>-near
$$

來找出過近或位於相機後方的頂點，因此位於近平面上的點不列入，程式與測試一致。稿件也正確區分頂點分類與三角形裁切。

---

## 三、可選文風與引用建議

- `Norm is`、`Translation`、`Fallback` 可分別改為「長度為」、「平移欄」、「備援策略」，使正文語言一致；這屬文風，不影響數學正確性。
- 「`up`向量平行」宜寫成「`up` 與視線方向平行」。
- G1、G4、G5 是候選背景來源；目前來源清單仍不等於獨立查證。尤其 G5 的 Transformations 頁面是否直接支持 look-at 敘述，仍待人工回查。
- 本審查未執行 NumPy 程式。修正後仍需人工執行 shape、非有限值、極端容差、平行方向及矩陣逆轉換測試。

VERDICT: REVISE