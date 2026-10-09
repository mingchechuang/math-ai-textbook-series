# 審稿結果

## 一、必須修正的數學／統計錯誤

### 1. 程式中的有限 seed「偏誤平方＋變異數」分解不成立

**原程式：**

```python
mean_img = np.mean(stack, axis=0)
bias2 = np.mean((mean_img - ref_true) ** 2)
variance = np.mean(np.var(stack, axis=0, ddof=1))
total_mse = bias2 + variance
```

`bias2` 並不只是估計器的真實偏誤平方。有限 $K$ 下，樣本平均本身仍有方差：

$$
\mathbb E[(\bar I-I)^2]
=
\operatorname{Bias}(\hat I)^2
+
\frac{\operatorname{Var}(\hat I)}{K}.
$$

因此目前的 `bias2 + variance` 在期望上約為

$$
\operatorname{Bias}^2+
\left(1+\frac1K\right)\operatorname{Var},
$$

會重複計入一部分方差。

對程式中的無偏、`sigma=1`、`K=10`、`spp=1` 案例，`bias2` 的期望約為 $1/10=0.1$，並不會如預期結果所稱「接近 0」。

**修法一：直接做精確的有限樣本經驗分解。**

令

```python
empirical_mse = np.mean((stack - ref_true[None, ...]) ** 2)
mean_error2 = np.mean((mean_img - ref_true) ** 2)
sample_variance = np.mean(np.var(stack, axis=0, ddof=1))
decomposed_mse = mean_error2 + ((K - 1) / K) * sample_variance
```

則有限資料上有

$$
\frac1K\sum_{k=1}^K\|\hat I_k-R\|^2
=
\|\bar I-R\|^2
+
\frac{K-1}{K}S^2,
$$

其中 $S^2$ 使用 `ddof=1`。

**修法二：估計真實偏誤平方。**

可使用

```python
estimated_bias2 = mean_error2 - sample_variance / K
```

但有限樣本下它可能為負，因為它只是帶抽樣誤差的估計值。正文須明說不能強制把負值解讀為物理負偏誤平方。

---

### 2. 預期結果中的 `Bias² 接近 0` 與程式定義矛盾

**原句：**

> 無偏時（Bias=0.0）……Bias² 接近 0。

目前列印的 `Bias²` 是跨 seed 平均影像對真值的平方誤差，不是真實偏誤平方。其期望為

$$
\frac{\sigma^2}{spp\cdot K}.
$$

例如 `sigma=1`、`spp=1`、`K=10` 時，期望約為 $0.1$；`spp=64` 時才約為

$$
\frac1{640}\approx0.0015625.
$$

**修法：**

- 將變數與輸出改名為 `Mean-image error²`；或
- 使用上一項的有限 $K$ 修正估計；
- 預期結果需列出有限 seed 殘差，不能宣稱接近零而不給尺度。

---

### 3. 「含噪參考圖底限」用詞需更精確

**原句：**

> 實測 MSE 期望值會包含參考圖噪聲底限。

若測試估計器本身的方差隨 spp 趨近零，獨立且無偏的參考圖方差才形成極限底限。有限 spp 時總期望是兩者之和，不應把整個值都稱為底限。

案例 2 可具體補算：若參考方差為 $0.001$，則

$$
\mathbb E[MSE_{16}]
=
0.0025+0.001
=
0.0035,
$$

$$
\mathbb E[MSE_{64}]
=
0.000625+0.001
=
0.001625.
$$

當 $n\to\infty$ 時才趨近 $0.001$。

---

## 二、效能驗證仍未真正完成

### 1. 輸出的 `Samples/s` 沒有實際意義

**原程式：**

```python
test_img = render_simulated(spp=spp, ...)
...
total_samples = 16 * 16 * spp
rate = total_samples / avg_time
```

`render_simulated` 不會執行 $WH\cdot spp$ 個樣本，只生成固定數量的高斯亂數，並把標準差除以 $\sqrt{spp}$。因此分子宣稱處理了 `WH*spp` 個樣本，但計時區塊沒有做這些工作。

雖然正文已警告不代表路徑追蹤效能，仍不應輸出名為 `Rate` 的虛假吞吐量。

**修法：**

二選一：

1. 刪除此模擬程式的 `Samples/s`，只報實際函式呼叫時間；或
2. 讓模擬器真的建立 `(spp,H,W,C)` 個獨立樣本再取平均，但這仍只是 NumPy 亂數吞吐量，不是路徑追蹤效能。

真正的實驗協議應另列：

- 固定場景、解析度、最大深度與 seed 集合；
- 計時第 22 章渲染核心；
- 排除或另報檔案 I/O；
- 記錄暖機政策；
- 報告每 seed 時間、平均值及樣本標準差；
- 記錄 Python、NumPy、CPU 與執行緒設定；
- 未實跑時全部標為待辦，不提供虛構 FPS。

---

### 2. 單次極短 `perf_counter` 計時不穩定

16×16 高斯陣列生成時間很短，單次計時容易受排程、配置與計時解析度影響。若保留微型基準測試，應重複多次並以批次總時間除以次數，且將第一批視需要列為暖機，不可把十次單次呼叫平均當成可靠效能結論。

---

## 三、降噪實作與邊界問題

### 1. `box_blur` 缺少輸入維度、型別及 `radius` 整數檢查

**原程式：**

```python
if radius < 0:
    raise ValueError(...)
...
H, W, C = img.shape
blurred = np.zeros_like(img)
```

問題包括：

- `radius=1.5` 通過第一項檢查，之後 `np.pad` 或 `range` 才以不清楚的方式失敗；
- 整數影像會令 `zeros_like` 產生整數輸出，`blurred /= ...` 可能發生轉型錯誤或截斷；
- 未檢查 shape 是否為 `(H,W,C)`；
- 未檢查有限值；
- 空影像未處理。

**修法：**

```python
img = np.asarray(img, dtype=np.float64)
if img.ndim != 3 or img.shape[2] != 3:
    raise ValueError("img must have shape (H, W, 3)")
if img.shape[0] == 0 or img.shape[1] == 0:
    raise ValueError("image dimensions must be non-zero")
if not np.all(np.isfinite(img)):
    raise ValueError("img must contain finite values")
if not isinstance(radius, (int, np.integer)) or radius < 0:
    raise ValueError("radius must be a non-negative integer")
```

另外以下兩行完全未使用，應刪除：

```python
y_start = y_stop = None
```

---

### 2. 仍沒有實際測試邊緣滲色、過度平滑或假細節

正文已正確區分三種現象，但程式只測試常數真值。常數圖沒有幾何或色彩邊緣，因此不能驗證章節核心中的「降噪假細節」。

至少加入：

1. 左半為 0、右半為 1 的階梯影像；
2. 單一亮像素；
3. 無訊號純噪聲；
4. 細棋盤格。

並量測：

- 邊緣兩側的偏誤；
- 邊緣寬度；
- 單點能量是否擴散；
- 高頻結構是否被移除；
- 輸出是否出現參考圖不存在的局部極值。

Box filter 主要能展示模糊與滲色，不能充分代表會生成假細節的非線性／資料驅動降噪器；此界線需保留。

---

## 四、習題與程式完整性

### 1. `compare_renders` 尚未拒絕錯誤影像維度

雖然檢查 shape 相同，但若三張圖都是 `(H,W)`，之後

```python
H, W, C = img1.shape
```

會出現一般拆包錯誤。應先明確檢查：

```python
if ref.ndim != 3 or ref.shape[2] != 3:
    raise ValueError("images must have shape (H, W, 3)")
```

### 2. 習題 2 說「修改」函式，但函式已有 `bias` 參數

**原題：**

> 修改 `render_simulated`，引入常數偏誤 $+0.05$。

現有函式已能直接呼叫：

```python
render_simulated(spp, seed, bias=0.05)
```

應將題目改為「使用 `bias=0.05` 設計跨 spp、跨 seed 實驗」，避免題目與程式現況矛盾。

### 3. 習題 3 的判準仍需考慮有限 spp

「平均誤差不隨 spp 下降」只能說偏誤證據增強。應指定 spp 序列及 seed 數，並避免以固定的 1000 spp 當作普遍充分條件。現有「傾向」與「不能單獨證明」措辭是正確改進。

---

## 五、論述、篇幅與引用界線

### 1. 仍未達最低篇幅

`measured_characters` 為 2259，低於至少 3000 中文字的硬性規格，自動檢查亦判定正文不足。可補充：

- 有限 seed 經驗分解的完整推導；
- 配對 seed／共同隨機數對差值方差的作用；
- 真實計時實驗表格欄位；
- 邊緣降噪案例；
- MSE 對極端 firefly 的敏感性；
- 參考圖自身誤差與獨立性限制。

### 2. 小結過度宣稱

**原句：**

> 這些方法確保了合成資料的物理可信度……

MSE、MAE 與計時只能提供驗證證據，不能單獨確保物理可信度。應改為：

> 這些方法提供可稽核的數值證據，但不能取代模型假設、單位、材質、介質及實作正確性的獨立驗證。

### 3. 引用界線仍不充分

- G2 可支援 BRDF 背景，不等於支援降噪、MSE 或計時方法。
- G3 的指定頁面是否涵蓋蒙特卡洛估計，不能只依標題推定。
- G4 可作入門路徑追蹤背景，不能支援本章的有限樣本統計公式。
- 目前沒有直接支持偏誤—變異數、統計區間或影像降噪的來源；若來源集不可擴充，應把相關內容清楚標為本章自行推導，而非暗示由現列來源全面支持。

本審查未執行程式，也未獨立查核來源。修正後仍需人工執行跨 seed、邊界影像與真實路徑追蹤計時實驗。

VERDICT: REVISE