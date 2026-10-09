# 審稿結果

本次已修正最近球根、重心座標順序、零方向檢查、平面實作與 PPM 輸出等主要問題，但仍有數值維度、測試完整性及程式邊界錯誤，尚不宜批准。

## 一、必須修正的實質問題

### 1. 判別式容差混合不同維度

**原句／程式：**

```python
tol = 1e-8 * max(1.0, abs(b), abs(c))
if delta < -tol:
```

**錯誤原因：**

方向正規化且座標以公尺計時：

- $b$ 單位為 m；
- $c$ 單位為 m²；
- $\Delta=b^2-4ac$ 單位為 m²。

`max(1.0, abs(b), abs(c))` 混合無因次、m、m²，不能作為 $\Delta$ 的容差尺度。

**修法：**

以判別式兩項的量級建立 m² 容差，例如：

```python
disc_scale = max(b * b, abs(4.0 * a * c), 1.0)
disc_tol = 1e-12 * disc_scale
if delta < -disc_tol:
    return None
delta = max(delta, 0.0)
```

其中常數仍是教學選值，需說明與 `float64`、座標尺度有關。若要維度更嚴謹，可先將場景尺度正規化。

---

### 2. 正文仍保留錯誤手算再回頭更正

**原句：**

> $\mathbf q=(0,0,0.2)$  
> $v=-0.2$。  
> 等等，這裡 $v$ 為負？讓我們重新檢查外積。

**錯誤原因：**

最終更正雖正確，但正式教材不應把未標成「常見錯誤示範」的錯誤計算留在主推導中，會妨礙讀者核對。

**修法：**

直接保留正確結果：

$$
\mathbf q
=(0.2,0.2,-1)\times(1,0,0)
=(0,-1,-0.2).
$$

若要教學展示錯誤，應另設「符號錯誤示例」，清楚指出外積第二分量的負號來源。

---

### 3. 平面法線只寫「假設正規化」，程式未驗證

**原句：**

```python
class Plane:
    ...
    # 假設 normal 已正規化
    self.normal = normal
    self.d = d
```

**錯誤原因：**

正文宣告正規化是容差判斷的前提，但 API 可接受任意法線，因而無法保證 `1e-8` 的平行容差具有一致意義。零法線也未拒絕。

**修法二選一：**

1. 驗證 $\|\mathbf n\|\approx1$，否則拋出例外；或
2. 在建構時同時縮放 $\mathbf n$ 與平面常數：

```python
length = np.linalg.norm(normal)
if not np.isfinite(length) or length <= 0.0:
    raise ValueError("plane normal must be finite and non-zero")
self.normal = normal / length
self.d = d / length
```

只正規化法線而不縮放 $d$ 會改變平面，必須避免。

---

### 4. 法線偏移在內積為零時完全失效

**原句／程式：**

```python
o_new = p + np.sign(np.dot(d, n)) * 1e-4 * n
```

**錯誤原因：**

當 $\mathbf d\cdot\mathbf n=0$ 時，`np.sign(0)` 為 0，起點完全不偏移。這正是掠射方向可能需要特別處理的邊界。

**修法：**

明確制定零值規則：

```python
side = 1.0 if np.dot(d, n) >= 0.0 else -1.0
o_new = p + side * origin_eps * n
```

並先驗證或正規化 `n`。

---

### 5. 陰影射線偏移後沒有重新指向光源

**原句／程式：**

```python
d = (light_pos - p) / dist
o_new = p + ...
ray = Ray(o_new, d)
ray.t_max = dist - 1e-4
```

**錯誤原因：**

起點改成 `o_new` 後，舊方向 `d` 不一定精確指向 `light_pos`，原來的 `dist` 也不再是新起點到光源的距離。雖然偏差通常很小，但此章主題正是數值邊界，應保持幾何一致。

**修法：**

```python
to_light = light_pos - o_new
light_dist = np.linalg.norm(to_light)
ray = Ray(o_new, to_light, t_min=ray_eps)
ray.t_max = light_dist - endpoint_eps
```

若 `ray.t_max < ray.t_min`，應直接判定無遮擋。

---

### 6. `t_min` 與起點偏移仍被混稱

**原句：**

> 使用本章的求交函數，並考慮 $t_{min}$ 偏移。

以及：

> $t_{min}$ 偏移是避免自相交的關鍵。

**錯誤原因：**

兩者不同：

- 起點偏移：改變 $\mathbf o$；
- `t_min`：限制有效參數區間。

**修法：**

改為「起點法線偏移與射線下界 `t_min`」。兩種方法可以一起使用，但不能視為同一操作。

---

### 7. 相機基底缺少必要邊界檢查

**原句／程式：**

```python
right = np.cross(cam_dir, up)
right = right / np.linalg.norm(right)
```

**錯誤原因：**

- `cam_dir` 未先正規化；
- 若 `cam_dir` 與 `up` 平行，會除以零；
- 未拒絕零方向；
- `fov` 未限制在 $(0,180^\circ)$；
- 一般非正方形影像缺少 aspect ratio。

**修法：**

驗證並正規化 `cam_dir`，檢查外積長度；若 `fov` 是垂直視角，應使用：

```python
aspect = width / height
x = (2.0 * (i + 0.5) / width - 1.0) * aspect / focal_length
```

目前 16×16 因 aspect 為 1，不影響此例，但函式宣稱可接受任意寬高。

---

### 8. 色彩空間敘述不正確／不完整

**原句：**

```python
# 將線性顏色轉為 8-bit
pixel = np.clip(color, 0.0, 1.0) * 255.0
```

**錯誤原因：**

乘以 255 只是量化，不是由線性 RGB 轉成 sRGB。示例使用純紅、純綠、純黃，端點值碰巧在線性與 sRGB 下相同，但註解對一般顏色不成立。

**修法：**

二選一：

- 將 `color` 明確定義為已編碼的顯示 RGB，刪除「線性」；
- 或提供分段的 linear RGB → sRGB 轉換後再量化。

---

### 9. 球體與物件輸入缺少基本幾何驗證

**可定位處：**

```python
self.radius = radius
```

**問題：**

半徑可為零或負數；座標、顏色、法線可能含 `NaN`／無限值。這會破壞求交判定。

**修法：**

至少拒絕非有限座標與 `radius <= 0`，檢查向量形狀為 `(3,)`。顏色則裁切前也應拒絕非有限值。

---

## 二、測試與預期結果仍未達章綱

### 1. 沒有實際可重現的測試程式

「測試與預期結果」目前只有文字條列，沒有 `assert` 或逐筆數值輸入。章綱要求測：

- 相切；
- 平行；
- 球內部出射；
- epsilon／`t_min`；
- 最近交點；
- 三角形內外；
- PPM 尺寸與 RGB 範圍。

應在完整程式中加入例如：

```python
assert np.isclose(intersect_sphere(...), 4.0)
assert np.isclose(intersect_sphere(inside_ray, sphere), 1.0)
assert intersect_plane(parallel_ray, plane) is None
assert img.shape == (16, 16, 3)
assert img.dtype == np.uint8
```

並把全部結果標為「預期」，不可暗示已執行。

### 2. 相切案例仍未落實

目前程式有近零判別式處理，但沒有明確相切輸入及預期值。可使用：

$$
\mathbf o=(1,0,-5),\quad
\mathbf d=(0,0,1),\quad
\mathbf c=(0,0,0),\quad R=1,
$$

其唯一根為 $t=5$。

### 3. 自相交主張沒有對應測試

**原句：**

> 若移除 $t_{min}$，自相交測試會錯誤地報告命中。

稿中沒有該測試，且精確位於表面時，若仍用嚴格正下界，不一定會命中。應使用刻意擾動的起點並列出數值，例如距平面 $10^{-8}$ m：

- `t_min=0` 或更小時，預期命中 $10^{-8}$；
- `t_min=10^{-6}` 時，預期忽略。

### 4. 影像視覺預期需更謹慎

**原句：**

> 預期：畫面下方有綠色地板，中間有紅色球體，後方有黃色三角形。

平面、球與三角形互相遮擋，黃色三角形未必完整可見；無光照時也不能稱為「陰影」。建議寫成：

> 預期背景為白色；最近命中的物件以指定純色顯示。紅球會遮住三角形的一部分，平面可能遮住三角形下部。

---

## 三、程式品質與索引核對

- `img[j, i]` 與「像素原點左上、影像 Y 向下」一致。
- `x` 向右、`y` 向上，且 `cam_dir=(0,0,-1)`，符合全書座標約定。
- PPM 的 `height, width` 與 `img.tobytes()` row-major 順序一致。
- RGB 維度為 3，且 `uint8` 可保證輸出位元組範圍為 0–255。
- `hit_normal` 從未用到，可刪除。
- `struct` 未使用，可刪除。
- `dir` 會遮蔽 Python 內建函式 `dir`，建議改成 `ray_dir`。
- 建構函式的預設 `np.array(...)` 是共享可變物件；即使目前未修改，也建議改成 `color=None` 後在函式內建立。
- 型別標註應用 `np.ndarray`，不是 `np.array`：

```python
color: np.ndarray | None = None
```

---

## 四、引用與篇幅

1. G1「Transformations」仍不是球、平面、三角形求交的直接來源，建議移除或說明只支持座標／變換背景。
2. 「Kahan 或類似技術」目前沒有對應的精確來源。若無法在候選來源中定位，改稱「穩定二次求根形式」，不要作具名歸因。
3. G4 可支持入門 ray tracing，但來源清單不等於逐條查證；稿件目前沒有虛構已執行官方範例，這點可保留。
4. 自動檢查仍顯示正文不足最低 3000 字。應以完整測試、相切推導、共面平面語意、`t_min/t_max` 區間與相對容差分析補足，不宜用案例宣傳填充。

## 五、可選文風建議

- 「Kahan 或类似技術」中的「类似」應改為繁體「類似」。
- `Generated output_19.ppm` 可改為「已寫入」但這只是程式執行時的訊息；正文仍須標記整段輸出為預期。
- 將閉區間 `[t_min,t_max]` 與程式使用 `>=`、`<=` 的一致性保留；若陰影端點需排除，應由 `t_max` 扣除端點容差，而非悄悄改成開區間。
- 三角形函式目前回傳交點而非重心座標，雖然內部重心順序已正確，但若教學目標包含重心驗證，建議也回傳 `(w0,w1,w2)`。

本版方向已大幅改善，但尚缺指定的明確測試，且判別式容差與陰影射線幾何仍有實質問題。

VERDICT: REVISE